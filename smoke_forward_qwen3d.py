import os
import torch

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

from qwen3d import (
    add_maskformer2_config,
    add_maskformer2_video_config,
)

from qwen3d.data_video.data_utils import (
    qwen_preprocess_frames,
    get_multiview_xyz,
)


class SmokeStop(Exception):
    """Intentional stop after the real Qwen-3D mask decoder finishes."""
    pass


def print_tree(x, prefix=""):
    if torch.is_tensor(x):
        print(
            f"{prefix}Tensor "
            f"shape={tuple(x.shape)}, "
            f"dtype={x.dtype}, "
            f"device={x.device}"
        )
    elif isinstance(x, dict):
        for k, v in x.items():
            print(f"{prefix}{k}:")
            print_tree(v, prefix + "  ")
    elif isinstance(x, (list, tuple)):
        print(f"{prefix}{type(x).__name__}[{len(x)}]")
        for i, v in enumerate(x):
            print(f"{prefix}[{i}]")
            print_tree(v, prefix + "  ")
    else:
        print(f"{prefix}{type(x).__name__}: {x}")


def build_qwen3d():
    cfg = get_cfg()

    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)

    cfg.merge_from_file("qwen3d/configs/qwen_3d.yaml")

    cfg.defrost()

    cfg.QWEN_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
    cfg.MODEL.WEIGHTS = "ckpts/qwen3d_3b.pth"
    cfg.MODEL.DEVICE = "cuda"

    cfg.OUTPUT_DIR = "./output/smoke_forward"
    cfg.USE_WANDB = False

    # -------------------------------------------------
    # Important:
    # We do NOT want ScanNet full-scene ghost points.
    # We only want Qwen-3D's core RGB-D -> 3D -> LLM path.
    # -------------------------------------------------
    cfg.USE_GHOST_POINTS = False
    cfg.USE_SEGMENTS = False
    cfg.CACHE_QWEN_FEATURES = False

    # Keep the real Qwen-3D geometry settings.
    cfg.ROPE_TYPE = "custom_txyz"
    cfg.INPUT.VOXELIZE = True

    cfg.freeze()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print("[1] Building Qwen-3D...")
    model = build_model(cfg)

    print("[2] Loading Qwen-3D checkpoint...")
    DetectionCheckpointer(
        model,
        save_dir=cfg.OUTPUT_DIR,
    ).load(cfg.MODEL.WEIGHTS)

    model.eval()

    return model, cfg


def make_synthetic_scene(model, cfg):
    # -------------------------------------------------
    # One 448x448 synthetic RGB frame.
    # Same spatial size used by the official config.
    # -------------------------------------------------
    H = 448
    W = 448
    V = 1

    torch.manual_seed(0)

    image = torch.randint(
        low=0,
        high=256,
        size=(3, H, W),
        dtype=torch.uint8,
    )

    images = [image]

    # -------------------------------------------------
    # Synthetic depth:
    # every pixel is exactly 2 metres away.
    # -------------------------------------------------
    depth = torch.full(
        (H, W),
        2.0,
        dtype=torch.float32,
    )

    depths = [depth]

    # -------------------------------------------------
    # Synthetic camera pose:
    # camera == world coordinate system.
    # -------------------------------------------------
    pose = torch.eye(4, dtype=torch.float32)

    poses = [pose]

    # -------------------------------------------------
    # Simple pinhole camera intrinsics.
    # Exact calibration is irrelevant for smoke testing;
    # it only needs to produce valid 3D coordinates.
    # -------------------------------------------------
    fx = 400.0
    fy = 400.0
    cx = W / 2.0
    cy = H / 2.0

    intrinsic = torch.tensor(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    intrinsics = [intrinsic]

    # -------------------------------------------------
    # Qwen's official image preprocessing.
    #
    # This generates:
    #   qwen_pixel_values
    #   qwen_grid_thw
    #
    # exactly like their dataset mapper.
    # -------------------------------------------------
    image_processor = model.qwen_processor.image_processor

    pv_list, grid_list = qwen_preprocess_frames(
        image_processor,
        images,
    )

    print("\nSynthetic Qwen inputs:")
    print("  image:", tuple(image.shape))
    print("  pixel_values:", [tuple(x.shape) for x in pv_list])
    print("  grid_thw:", [x.tolist() for x in grid_list])

    # -------------------------------------------------
    # Convert depth + camera geometry into world XYZ.
    #
    # This is the SAME utility used by the official
    # ScanNet mapper.
    # -------------------------------------------------
    (
        multi_scale_xyz,
        _,
        original_xyz,
        new_h,
        new_w,
    ) = get_multiview_xyz(
        shape=(V, H, W),
        size_divisibility=cfg.INPUT.SIZE_DIVISIBILITY,
        depths=depths,
        poses=poses,
        intrinsics=intrinsics,
        is_train=False,
        augment_3d=False,
        interpolation_method=cfg.MODEL.INTERPOLATION_METHOD,
        mask_valid=cfg.MASK_VALID,
        mean_center=cfg.MEAN_CENTER,
        do_rot_scale=False,
        scannet_pc=None,
        align_matrix=None,
        vil3d=False,
        scales=cfg.MULTIVIEW_XYZ_SCALES,
        min_pixel=cfg.INPUT.MIN_PIXEL,
        max_pixel=cfg.INPUT.MAX_PIXEL,
    )

    print("\nSynthetic 3D geometry:")
    print("  new_h/new_w:", new_h, new_w)

    for i, xyz in enumerate(multi_scale_xyz):
        print(
            f"  multi_scale_xyz[{i}]:",
            tuple(xyz.shape),
        )

    print("  original_xyz:", tuple(original_xyz.shape))

    # -------------------------------------------------
    # Critical sanity check:
    #
    # Qwen ViT merges spatial tokens by spatial_merge_size.
    # The XYZ map used at index [1] must have exactly
    # the same number of elements.
    # -------------------------------------------------
    merge = model.visual.spatial_merge_size

    expected_visual_tokens = 0

    for g in grid_list:
        t, gh, gw = [int(x) for x in g]
        expected_visual_tokens += (
            t * gh * gw // (merge * merge)
        )

    xyz_tokens = multi_scale_xyz[1].reshape(-1, 3).shape[0]

    print("\nToken alignment check:")
    print("  Qwen visual tokens:", expected_visual_tokens)
    print("  XYZ tokens:        ", xyz_tokens)
    print("  feature map:       ", new_h * new_w * V)

    assert expected_visual_tokens == xyz_tokens, (
        f"Qwen tokens ({expected_visual_tokens}) "
        f"!= XYZ tokens ({xyz_tokens})"
    )

    assert xyz_tokens == new_h * new_w * V, (
        f"XYZ tokens ({xyz_tokens}) "
        f"!= feature map ({new_h * new_w * V})"
    )

    print("  TOKEN ALIGNMENT OK")

    # -------------------------------------------------
    # Minimal batched_inputs expected by Qwen3D.forward.
    #
    # No ScanNet.
    # No Sr3D.
    # No GT masks.
    # -------------------------------------------------
    sample = {
        "decoder_3d": True,
        "actual_decoder_3d": True,

        "images": images,

        "new_h": new_h,
        "new_w": new_w,

        "multi_scale_xyz": multi_scale_xyz,

        "qwen_pixel_values": pv_list,
        "qwen_grid_thw": grid_list,

        # prepare_targets() can take this minimal path
        # when instances_all is absent.
        "text_caption": "the object",

        "dataset_name": "synthetic_qwen3d_smoke",

        # Not required before mask decoder, but useful
        # if code paths inspect it.
        "num_classes": 1,

        "do_generate": False,
        "generate_only": False,
    }

    return sample


def main():
    print("=" * 72)
    print("QWEN-3D SYNTHETIC FORWARD SMOKE TEST")
    print("=" * 72)

    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))

    model, cfg = build_qwen3d()

    sample = make_synthetic_scene(model, cfg)

    # -------------------------------------------------
    # We want the REAL mask decoder to execute.
    #
    # But after that the official forward enters
    # ScanNet / dataset-specific evaluation code.
    #
    # So wrap the decoder:
    #   real decoder runs
    #   -> print outputs
    #   -> deliberately stop
    # -------------------------------------------------
    real_mask_forward = model.mask_decoder.forward

    def mask_forward_and_stop(*args, **kwargs):
        print("\n[Mask Decoder] entering real SimpleMaskDecoder...")

        args = list(args)

        mask_features = args[0]

        print(
            "  original mask_features:",
            tuple(mask_features.shape)
        )

        # -----------------------------------------------------
        # Official Qwen-3D normally uses USE_GHOST_POINTS=True
        # for the real voxelized 3D decoder.
        #
        # We deliberately disabled ScanNet ghost points for this
        # synthetic smoke test, so model.py leaves the Qwen point
        # features in a 2D [B,C,H,W] layout.
        #
        # Convert them back to the 3D decoder layout:
        #
        # [B,C,H,W] -> [B,C,N,1]
        # -----------------------------------------------------

        if (
            kwargs.get("decoder_3d", False)
            and mask_features.ndim == 4
            and mask_features.shape[-1] != 1
        ):
            mask_features = mask_features.flatten(2).unsqueeze(-1)

            print(
                "  converted 3D mask_features:",
                tuple(mask_features.shape)
            )

            # -------------------------------------------------
            # Reconstruct the exact voxel-space XYZ coordinates
            # using Qwen-3D's own load_3d_data().
            # -------------------------------------------------

            _, multiview_data = model.load_3d_data(
                [sample],
                images_shape=[1, 1, 448, 448],
            )

            pointcloud = (
                multiview_data["multi_scale_xyz"][1]
                .reshape(-1, 3)
                .to(model.device)
            )

            point2voxel = (
                multiview_data["multi_scale_p2v"][1]
                .squeeze(0)
                .to(model.device)
            )

            from torch_scatter import scatter_mean

            pointcloud = scatter_mean(
                pointcloud,
                point2voxel,
                dim=0,
            ).unsqueeze(0)

            print(
                "  voxel XYZ:",
                tuple(pointcloud.shape)
            )

            assert (
                mask_features.shape[-2]
                == pointcloud.shape[1]
            ), (
                mask_features.shape,
                pointcloud.shape,
            )

            args[0] = mask_features

            kwargs["mask_features_xyz"] = pointcloud

        outputs = real_mask_forward(
            *args,
            **kwargs,
        )

        print("\n[Mask Decoder] finished successfully.")
        print("\nMask decoder outputs:")

        print_tree(outputs, prefix="  ")

        raise SmokeStop()

    model.mask_decoder.forward = mask_forward_and_stop

    torch.cuda.reset_peak_memory_stats()

    try:
        print("\n[3] Running real Qwen-3D forward...")

        with torch.inference_mode():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                model([sample])

    except SmokeStop:
        print("\n" + "=" * 72)
        print("SUCCESS")
        print("=" * 72)
        print(
            "Synthetic RGB -> Qwen ViT -> 3D world tokens -> "
            "Qwen LLM -> connector -> SimpleMaskDecoder"
        )
        print("completed successfully.")

    peak = torch.cuda.max_memory_allocated() / 1024**3

    print(f"\nPeak allocated GPU memory: {peak:.2f} GB")


if __name__ == "__main__":
    main()