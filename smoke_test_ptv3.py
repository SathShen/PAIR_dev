import time
import torch

from ptv3.model import PointTransformerV3


NUM_SIDE = 16
NUM_POINTS = NUM_SIDE ** 3
IN_CHANNELS = 6
PATCH_SIZE = 128


def make_synthetic_point_cloud(device):
    axis = torch.arange(NUM_SIDE, dtype=torch.int32)
    gx, gy, gz = torch.meshgrid(axis, axis, axis, indexing="ij")

    grid_coord = torch.stack(
        [gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)],
        dim=1,
    ).contiguous()

    coord = grid_coord.float() * 0.05
    xyz_norm = coord / max(float(coord.max().item()), 1e-6)

    x, y, z = xyz_norm.unbind(dim=1)
    extra = torch.stack(
        [
            torch.sin(x * 3.14159265),
            torch.cos(y * 3.14159265),
            z,
        ],
        dim=1,
    )
    feat = torch.cat([xyz_norm, extra], dim=1).float()

    batch = torch.zeros(NUM_POINTS, dtype=torch.long)

    return {
        "coord": coord.to(device),
        "grid_coord": grid_coord.to(device),
        "feat": feat.to(device),
        "batch": batch.to(device),
    }


def tensor_info(name, x):
    print(
        f"  {name}: shape={tuple(x.shape)}, "
        f"dtype={x.dtype}, device={x.device}"
    )


def main():
    print("=" * 72)
    print("POINT TRANSFORMER V3 STANDALONE FORWARD SMOKE TEST")
    print("=" * 72)

    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    assert torch.cuda.is_available(), "CUDA is required for this smoke test."

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(device))
    print("Synthetic points:", NUM_POINTS)
    print("Input channels:", IN_CHANNELS)
    print("FlashAttention: disabled")
    print("Patch size:", PATCH_SIZE)
    print()

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.reset_peak_memory_stats()

    print("[1] Building PointTransformerV3...")

    model = PointTransformerV3(
        in_channels=IN_CHANNELS,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(PATCH_SIZE,) * 5,
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(PATCH_SIZE,) * 4,
        drop_path=0.0,
        shuffle_orders=False,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=True,
        upcast_softmax=True,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
    ).to(device)

    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print("Model built.")
    print(f"  total parameters: {total_params / 1e6:.3f} M")
    print(f"  trainable parameters: {trainable_params / 1e6:.3f} M")
    print()

    print("[2] Building synthetic point cloud...")

    data_dict = make_synthetic_point_cloud(device)

    tensor_info("coord", data_dict["coord"])
    tensor_info("grid_coord", data_dict["grid_coord"])
    tensor_info("feat", data_dict["feat"])
    tensor_info("batch", data_dict["batch"])

    assert data_dict["coord"].shape == (NUM_POINTS, 3)
    assert data_dict["grid_coord"].shape == (NUM_POINTS, 3)
    assert data_dict["feat"].shape == (NUM_POINTS, IN_CHANNELS)
    assert data_dict["batch"].shape == (NUM_POINTS,)
    assert torch.isfinite(data_dict["coord"]).all()
    assert torch.isfinite(data_dict["feat"]).all()

    unique_voxels = torch.unique(data_dict["grid_coord"], dim=0).shape[0]
    print("  unique grid coordinates:", unique_voxels)
    assert unique_voxels == NUM_POINTS
    print("  SYNTHETIC INPUT OK")
    print()

    print("[3] Running PTv3 forward...")

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.inference_mode():
        output = model(data_dict)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    print("Forward OK.")
    print("  output type:", type(output).__name__)
    print(f"  forward time: {elapsed:.3f} s")
    print()

    print("[4] Inspecting PTv3 output...")

    print("  output keys:")
    for key in output.keys():
        value = output[key]
        if torch.is_tensor(value):
            print(
                f"    {key}: shape={tuple(value.shape)}, "
                f"dtype={value.dtype}"
            )
        else:
            print(f"    {key}: {type(value).__name__}")

    assert "feat" in output
    assert "coord" in output
    assert "batch" in output

    out_feat = output.feat
    out_coord = output.coord
    out_batch = output.batch

    print()
    tensor_info("output.feat", out_feat)
    tensor_info("output.coord", out_coord)
    tensor_info("output.batch", out_batch)

    assert out_feat.ndim == 2
    assert out_feat.shape[0] == NUM_POINTS, (
        f"Expected decoder to return {NUM_POINTS} point features, "
        f"got {out_feat.shape[0]}"
    )
    assert out_coord.shape == (NUM_POINTS, 3)
    assert out_batch.shape == (NUM_POINTS,)
    assert torch.isfinite(out_feat).all()
    assert torch.isfinite(out_coord).all()

    print("  output feature dimension:", out_feat.shape[1])
    print("  OUTPUT FEATURE CHECK OK")
    print()

    print("[5] Sanity checks...")

    feat_mean = out_feat.float().mean().item()
    feat_std = out_feat.float().std().item()
    feat_abs_max = out_feat.float().abs().max().item()

    print(f"  feature mean: {feat_mean:.6f}")
    print(f"  feature std: {feat_std:.6f}")
    print(f"  feature |max|: {feat_abs_max:.6f}")

    assert feat_std > 0.0, "PTv3 output features are constant."
    assert feat_abs_max > 0.0, "PTv3 output features are all zero."

    coord_delta = (
        out_coord.float() - data_dict["coord"].float()
    ).abs().max().item()

    print(f"  max |output.coord - input.coord|: {coord_delta:.8f}")
    print("  SANITY CHECKS OK")
    print()

    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Peak allocated GPU memory: {peak_mem:.2f} GB")
    print()

    print("=" * 72)
    print("SUCCESS")
    print("=" * 72)
    print(
        "Synthetic point cloud -> PTv3 serialization/spconv -> "
        "encoder -> decoder -> point features completed successfully."
    )
    print()
    print(f"Final PTv3 feature tensor: {tuple(out_feat.shape)}")


if __name__ == "__main__":
    main()
