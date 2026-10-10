#!/usr/bin/env python3
"""PAIR standalone train/checkpoint/grad smoke test (single file).

Copy this ONE file to the PAIR project root and run:
    CUDA_VISIBLE_DEVICES=2 python test4.py --config configs/pair_train_qwen4b.json
    CUDA_VISIBLE_DEVICES=2 python test4.py --dataset NYC-SCD --config configs/pair_train_qwen4b.json
    python test4.py --mode mock

No imports of other test files or packaged fixtures.

Mock mode uses REAL PAIRModel / adapters / decoder / loss / train.py checkpoint
routines, but a tiny fake Qwen and Utonia. It cannot verify pretrained model
loading, actual GPU memory or Qwen's native processor/hook compatibility.

Real mode loads the actual pretrained Qwen (and Utonia for NYC-SCD) and one
actual dataset sample. It is deliberately a SINGLE-GPU smoke test, not DDP.

A clean checkpoint load is NOT proof of bitwise mid-epoch reproducibility:
train.py v2 stores optimizer/scheduler/position but not Python/NumPy/Torch RNG
states, workers' RNG, or the data-loader/sampler iterator state.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import patch

# Works both when copied to project root as test4.py and from project/tests/.
HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "models" / "pair.py").exists() else HERE.parent
if not (ROOT / "models" / "pair.py").is_file():
    raise RuntimeError("Place test4.py in the PAIR project root (or tests/ subdirectory).")
sys.path.insert(0, str(ROOT))


# Embedded CPU foundation-model fixtures: no test_train_v2/test_pair_integration imports.
class Tokenizer:
    eos_token = '!'
    def __call__(self, prompts, **kwargs):
        seqs = [[min(127, ord(c)%110+1) for c in p] for p in prompts]
        max_len = max(len(x) for x in seqs)
        ids = torch.tensor([x + [0]*(max_len-len(x)) for x in seqs],dtype=torch.long)
        return {'input_ids':ids, 'attention_mask':ids.ne(0).long()}

class FakeModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.visual = nn.Identity()
        self.embedding = nn.Embedding(128, dim)
        self.text_proj = nn.Linear(dim, dim)
    def forward(self, **kwargs):
        return SimpleNamespace(hidden_states=(self.text_proj(self.embedding(kwargs['input_ids'])),))

class FakeQwen(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.hidden_size=16
        self.vision_hidden_size=8
        self.vision_intermediate_layers=(5,11,17)
        self.model=FakeModel(16)
        self.tokenizer=Tokenizer()
        self.task_seed=nn.Parameter(torch.randn(16)*.15)
        self.image_proj=nn.Linear(3,8)
        self.visual_proj=nn.Linear(8,16)
        self.point_proj=nn.Linear(16,16)
        self.generated=0
    @property
    def model_device(self):
        return next(self.parameters()).device
    def freeze(self):
        self.model.requires_grad_(False)
    def unfreeze(self):
        self.model.requires_grad_(True)
    def _language_module(self):
        return self.model
    def forward(self, *, prompt, images_t1=None, images_t2=None, point_tokens_t1=None, point_tokens_t2=None, **kwargs):
        b=len(prompt)
        base=self.task_seed.unsqueeze(0).expand(b,-1)
        out={'task_hidden':base}
        if images_t1 is not None:
            for key, images in [('t1',images_t1), ('t2',images_t2)]:
                im=[]
                for image in images:
                    # These are actual PIL inputs, as the real Qwen processor expects.
                    assert hasattr(image,'convert')
                    import numpy as np
                    rgb=torch.from_numpy(np.asarray(image).copy()).float().permute(2,0,1)[None]/255
                    small=F.adaptive_avg_pool2d(rgb,(4,4)).squeeze(0).permute(1,2,0)
                    levels=self.image_proj(small)
                    im.append(levels.permute(2,0,1))
                merged=torch.stack(im)
                out['premerge_'+key]={idx: merged*(1.0+.01*idx) for idx in (5,11,17)}
                vis=F.adaptive_avg_pool2d(merged,(2,2)).permute(0,2,3,1)
                out['llm_visual_'+key]=[self.visual_proj(vis[j]).unsqueeze(0) for j in range(b)]
        if point_tokens_t1 is not None:
            for key, tokens in [('t1',point_tokens_t1),('t2',point_tokens_t2)]:
                token_list = [tokens] if torch.is_tensor(tokens) else tokens
                out['llm_point_'+key]=[self.point_proj(token) for token in token_list]
            out['task_hidden']=base+torch.stack([self.point_proj(x).mean(0) for x in (
                point_tokens_t1 if isinstance(point_tokens_t1,list) else [point_tokens_t1]
            )])
        return out
    def generate_box_proposals(self, *, prompt, kind, max_boxes, **kwargs):
        self.generated += 1
        b=len(prompt)
        dim=4 if kind=='2d' else 6
        boxes=torch.zeros(b,max_boxes,dim)
        if kind=='2d':
            boxes[:,0] = torch.tensor([.1,.1,.9,.9])
        else:
            bounds=kwargs['scene_bounds']
            xyz_min,xyz_max=bounds[:,0],bounds[:,1]
            boxes[:,0,:3]=xyz_min + (xyz_max-xyz_min)*.10
            boxes[:,0,3:]=xyz_min + (xyz_max-xyz_min)*.90
        valid=torch.zeros(b,max_boxes,dtype=torch.bool)
        valid[:,0]=True
        scores=torch.zeros(b,max_boxes)
        scores[:,0]=.91
        return {('boxes_2d' if kind=='2d' else 'boxes_3d'):boxes,
                'box_valid':valid, 'box_scores':scores}

class FakeUtoniaCfg:
    def __init__(self, **kwargs): self.args=kwargs
class FakeUtonia(nn.Module):
    output_dim=20
    def __init__(self,cfg):
        super().__init__();self.proj=nn.Linear(3,20)
    def forward(self, x):
        return SimpleNamespace(features=self.proj(x['coord']),
             coord=x['coord'],batch=x['batch'],offset=x['offset'],
             intensity=x.get('intensity'), intensity_mask=x.get('intensity_mask'))

model_cfg={
  'qwen_model':'fake', 'qwen_tuning':'frozen', 'decoder_dim':16,
  'point_encoder':{'checkpoint':'fake','voxel_size':0.5},
  'image_adapter':{'up_channels':(12,8),'rgb_channels':8},
  'point_adapter':{'llm_tokens_per_cloud':6,'llm_sampling':'uniform',
                   'llm_max_fps_candidates':20,'memory_tokens_per_cloud':7,
                   'detail_chunk_size':11,'utonia_chunk_size':13,
                   'llm_align_chunk_size':12,'utonia_checkpoint':False,
                   'detail_checkpoint':False},
  'unified_decoder':{'num_layers':1,'num_heads':4,'mlp_ratio':2},
  'change_decoder':{'temporal_channels':8,'event_hidden_dim':12,
                    'event_chunk_size':11,'event_checkpoint':False},
  'box_guidance':{'max_proposals':3}
}


def new_pair(enable_2d=True, enable_3d=True):
    from models import pair
    with patch.object(pair,'Qwen3VLBackbone',FakeQwen), \
         patch.object(pair,'UtoniaPointEncoder',FakeUtonia), \
         patch.object(pair,'UtoniaPointEncoderConfig',FakeUtoniaCfg):
        return pair.PAIRModel.from_config(model_cfg,'cpu',enable_2d=enable_2d,enable_3d=enable_3d)


def pointcloud(n,shift=0.,with_attrs=True):
    xyz=torch.randn(n,3)*.15+shift
    result={'coord':xyz}
    if with_attrs:
        result['rgb']=torch.rand(n,3)
        result['normal']=torch.randn(n,3)
        result['intensity']=torch.rand(n,1)
    return result



def make_2d_sample(name='a', w=16):
    return {
        'prompt': f'compare {name}',
        'images_t1': torch.rand(3,w,w),
        'images_t2': torch.rand(3,w,w),
        'target': {
            'semantic_t1': torch.randint(0,3,(w,w)),
            'semantic_t2': torch.randint(0,3,(w,w)),
            'change': torch.randint(0,2,(w,w)),
        },
    }

def make_3d_sample(n1,n2):
    return {
        'prompt':'Detect urban change',
        'point_dict_t1':pointcloud(n1),
        'point_dict_t2':pointcloud(n2,.05),
        'target':{
            'semantic_t1':torch.randint(0,4,(n1,)),
            'semantic_t2':torch.randint(0,4,(n2,)),
            'event_t1':torch.randint(0,2,(n1,)),
            'event_t2':torch.randint(0,2,(n2,))*2,
            'event_valid_t1':torch.ones(n1,dtype=torch.bool),
            'event_valid_t2':torch.ones(n2,dtype=torch.bool),
        },
    }


def setup_mock_imports():
    # This stub is installed ONLY in --mode mock, before importing models.pair.
    # Real mode never imports a fake backbone or fake foundation checkpoint.
    import types
    if 'models.pair' in sys.modules:
        raise RuntimeError('Mock stubs must be installed before importing models.pair')
    stub = types.ModuleType('models.qwen3vl_backbone')
    stub.Qwen3VLBackbone = object
    sys.modules['models.qwen3vl_backbone'] = stub
    try:
        import transformers  # noqa: F401
    except ImportError:
        trans = types.ModuleType('transformers')
        trans.get_constant_schedule_with_warmup = lambda optimizer, **kwargs: (
            torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        )
        trans.get_cosine_schedule_with_warmup = lambda optimizer, **kwargs: (
            torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        )
        sys.modules['transformers'] = trans


def require(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def report(message: str):
    print(f"[PASS] {message}", flush=True)


def train_settings(lr: float = 5e-4):
    return SimpleNamespace(
        lr=lr, main_weight_decay=0.0,
        llm_lora_lr=lr, llm_lora_weight_decay=0.0,
        vision_lora_lr=lr, vision_lora_weight_decay=0.0,
        point_lora_lr=lr, point_lora_weight_decay=0.0,
    )


def select_spec(dataset: str):
    if dataset == "SECOND":
        return SimpleNamespace(route="2d", label_mode="semantic_pair", class_names={
            0: "unchanged", 1: "building", 2: "vegetation"})
    if dataset == "LEVIR-CD":
        return SimpleNamespace(route="2d", label_mode="binary", class_names={
            0: "unchanged", 255: "changed"})
    if dataset == "NYC-SCD":
        return SimpleNamespace(route="3d", label_mode="semantic_pair", class_names={
            0: "ground", 1: "building", 2: "vegetation", 3: "clutter"})
    raise ValueError(dataset)


def make_mock_data():
    a = [make_2d_sample("first"), make_2d_sample("second")]
    a[0]["boxes_2d"] = torch.tensor([[.1, .1, .8, .85], [.2, .1, .65, .6]])
    a[0]["box_scores"] = torch.tensor([.9, .4])
    a[1]["boxes_2d"] = torch.empty(0, 4)
    b = [make_2d_sample("bcd")]
    b[0]["target"]["semantic_valid_t1"] = torch.zeros(16, 16, dtype=torch.bool)
    b[0]["target"]["semantic_valid_t2"] = torch.zeros(16, 16, dtype=torch.bool)
    c = [make_3d_sample(23, 26), make_3d_sample(19, 22)]
    c[0]["boxes_3d"] = torch.tensor([[-1., -1., -1., 1., 1., 1.]])
    c[1]["boxes_3d"] = torch.empty(0, 6)
    return [
        ("SECOND", a, "provided"),
        ("LEVIR-CD", b, "none"),
        ("NYC-SCD", c, "provided"),
        ("SECOND", a, "provided"),
    ]


def validate_logits(dataset: str, prediction: Any, target: dict, *, semantic_classes=None):
    n1, n2 = target["semantic_t1"].numel(), target["semantic_t2"].numel()
    require(prediction.semantic_logits_t1 is None if dataset == "LEVIR-CD"
            else prediction.semantic_logits_t1.shape[0] == n1,
            f"{dataset}: semantic T1 shape/availability")
    if dataset == "LEVIR-CD":
        require(prediction.change_logits.shape == target["change"].shape,
                "BCD must output SINGLE-channel flattened Change logits")
    elif dataset in ("SECOND", "LandsatSCD"):
        c = int(semantic_classes) if semantic_classes is not None else 3
        require(prediction.semantic_logits_t1.shape == (n1, c), "SCD T1 semantic shape")
        require(prediction.semantic_logits_t2.shape == (n2, c), "SCD T2 semantic shape")
        require(prediction.change_logits.shape == target["change"].shape, "SCD Change shape")
    else:
        require(prediction.semantic_logits_t1.shape == (n1, 4), "NYC semantic T1")
        require(prediction.semantic_logits_t2.shape == (n2, 4), "NYC semantic T2")
        require(prediction.event_logits_t1.shape == (n1, 3), "NYC event T1")
        require(prediction.event_logits_t2.shape == (n2, 3), "NYC event T2")
        require(bool((prediction.event_logits_t1[:, 2] <= -1e3).all()),
                "T1 Added class was not masked")
        require(bool((prediction.event_logits_t2[:, 1] <= -1e3).all()),
                "T2 Removed class was not masked")
    for key in ("semantic_logits_t1", "semantic_logits_t2", "change_logits",
                "event_logits_t1", "event_logits_t2"):
        tensor = getattr(prediction, key, None)
        if tensor is not None:
            require(bool(torch.isfinite(tensor).all()), f"{dataset}: {key} contains NaN/Inf")


def grad_check(model, dataset: str, box_mode: str):
    names = dict(model.named_parameters())
    if dataset == "SECOND":
        keys = ["image_adapter.llm_visual_project.weight",
                "decoder.temporal_reduce.weight",
                "decoder.query_decoder.layers.0.cross_attention.in_proj_weight"]
        if box_mode == "provided":
            keys.append("decoder.box_guidance.strength_2d")
    elif dataset == "LEVIR-CD":
        keys = ["image_adapter.llm_visual_project.weight", "decoder.temporal_reduce.weight"]
    else:
        keys = ["backbone.point_adapter.llm_point_project.weight",
                "backbone.point_adapter.llm_token_project.weight",
                "backbone.point_adapter.level_fusion.0.0.weight",
                "decoder.temporal_event_mlp.0.weight",
                "decoder.query_decoder.layers.0.cross_attention.in_proj_weight"]
        if box_mode == "provided":
            keys.append("decoder.box_guidance.strength_3d")
    summaries = []
    for key in keys:
        require(key in names, f"Expected trainable parameter missing: {key}")
        p = names[key]
        require(p.requires_grad, f"Parameter unexpectedly frozen: {key}")
        require(p.grad is not None, f"No gradient on {key} ({dataset})")
        require(bool(torch.isfinite(p.grad).all()), f"Nonfinite gradient on {key}")
        grad_l1 = float(p.grad.detach().float().abs().sum())
        require(grad_l1 > 0, f"Zero gradient on {key} ({dataset}); potential detached path")
        summaries.append(f"{key.split('.')[-2] if '.' in key else key}={grad_l1:.2e}")
    return ", ".join(summaries)


def update_step(model, optimizer, scheduler, criterion, train_mod, item, *, check_grad=True):
    dataset, samples, box_mode = item
    optimizer.zero_grad(set_to_none=True)
    pred, loss, target = train_mod.forward_loss(model, criterion, samples,
                                               select_spec(dataset), box_mode=box_mode)
    require(bool(torch.isfinite(loss.total.detach())), f"{dataset}: nonfinite loss")
    validate_logits(dataset, pred, target)
    if box_mode == "provided":
        require(bool(pred.box_guidance_applied), f"{dataset}: Box guidance ignored")
    loss.total.backward()
    detail = grad_check(model, dataset, box_mode) if check_grad else ""
    before = {name: p.detach().clone() for name, p in model.named_parameters()
              if p.requires_grad and p.grad is not None and bool((p.grad != 0).any())}
    if before:
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 3.0)
    optimizer.step()
    scheduler.step()
    after = dict(model.named_parameters())
    changed = [key for key, tensor in before.items() if not torch.equal(tensor, after[key].detach())]
    require(changed, f"{dataset}: optimizer.step() did not change trainable weights")
    print(f"    {dataset:8s} loss={float(loss.total.detach()):.6f} "
          f"updated={len(changed)} {detail}", flush=True)
    return float(loss.total.detach())


def compare_nested(a, b, path="root", *, atol=0.0, rtol=0.0):
    if isinstance(a, torch.Tensor):
        require(isinstance(b, torch.Tensor), f"Type mismatch at {path}")
        torch.testing.assert_close(a.detach().cpu(), b.detach().cpu(),
                                   atol=atol, rtol=rtol, msg=lambda msg: f"{path}: {msg}")
    elif isinstance(a, dict):
        require(isinstance(b, dict) and a.keys() == b.keys(), f"Keys differ at {path}")
        for key in a:
            compare_nested(a[key], b[key], f"{path}.{key}", atol=atol, rtol=rtol)
    elif isinstance(a, (list, tuple)):
        require(type(a) is type(b) and len(a) == len(b), f"Sequence differs at {path}")
        for i, (x, y) in enumerate(zip(a, b)):
            compare_nested(x, y, f"{path}[{i}]", atol=atol, rtol=rtol)
    else:
        require(a == b, f"Mismatch {path}: {a!r} != {b!r}")


def checkpoint_experiment(selected_names):
    return SimpleNamespace(hash_resolved=lambda cfg: "resume-smoke-hash",
                           selected_names=tuple(selected_names))


def state_snapshot(train_mod, model, optimizer, scheduler):
    return {
        "model": {k: v.detach().clone() for k, v in train_mod.pair_checkpoint_state(model).items()},
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
    }


def assert_checkpoint_structure(path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    require(saved.get("checkpoint_format_version") == 3, "Wrong checkpoint version")
    require(saved.get("architecture"), "Missing architecture signature")
    require("pair_state" in saved and "optimizer" in saved and "scheduler" in saved,
            "Missing model/optimizer/scheduler state")
    for name in saved["pair_state"]:
        require(not name.startswith("backbone.qwen_backbone.model."),
                f"Frozen Qwen accidentally embedded in checkpoint: {name}")
        require(not name.startswith("backbone.point_encoder.model."),
                f"Frozen Utonia accidentally embedded in checkpoint: {name}")
    return saved


def checkpoint_roundtrip(train_mod, model, optimizer, scheduler, item,
                         criterion, *, model_config, dataset_names, factory=None,
                         device=torch.device("cpu"), file_path: Path):
    """Compare next step uninterrupted vs resumed, with the SAME RNG + input.

    No RNG is saved by train.py itself; we restore it here only for comparison.
    """
    ex = checkpoint_experiment(dataset_names)
    cfg = {"model": copy.deepcopy(model_config), "optimizer": {"lr": 5e-4}}
    before_save = state_snapshot(train_mod, model, optimizer, scheduler)
    train_mod.save_checkpoint(
        file_path, model, optimizer, scheduler,
        epoch=0, update_in_epoch=3, optimizer_step=3,
        experiment=ex, resolved_config=cfg,
        dataset_best_values={dataset_names[0]: 0.28},
        dataset_best_epochs={dataset_names[0]: 1})
    saved = assert_checkpoint_structure(file_path)
    require(saved["epoch"] == 0 and saved["update_in_epoch"] == 3 and saved["optimizer_step"] == 3,
            "Checkpoint position is wrong")
    report("Checkpoint format v3 + training position + pretrained-weight exclusion")

    # Restore into a fresh model if factory supplied; otherwise use same heavy model
    # to avoid keeping two multi-billion-parameter Qwens in GPU memory.
    if factory is not None:
        restored_model = factory()
        restored_opt, _, _ = train_mod.build_optimizer(restored_model, train_settings())
        restored_sched = torch.optim.lr_scheduler.LambdaLR(restored_opt, lr_lambda=lambda s: 0.95 ** s)
    else:
        restored_model, restored_opt, restored_sched = model, optimizer, scheduler

    position = train_mod.load_checkpoint(
        file_path, restored_model, restored_opt, restored_sched,
        selected_datasets=dataset_names, model_config=model_config)
    require(position[:3] == (0, 3, 3), f"Wrong restored position: {position[:3]}")
    require(position[3] == {dataset_names[0]: .28}, "Best validation metric not restored")
    loaded = state_snapshot(train_mod, restored_model, restored_opt, restored_sched)
    compare_nested(before_save, loaded, "saved -> loaded", atol=0, rtol=0)
    report("Fresh/load: model weights + optimizer moments + scheduler + epoch/update/step")

    # Check post-resume optimizer step agrees with uninterrupted optimizer step.
    rng_cpu = torch.get_rng_state().clone()
    rng_cuda = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    loss_res = update_step(restored_model, restored_opt, restored_sched,
                           criterion, train_mod, item, check_grad=True)
    res_next = state_snapshot(train_mod, restored_model, restored_opt, restored_sched)

    # Reload saved pre-step state and replay exactly the SAME batch/RNG.
    train_mod.load_checkpoint(file_path, model, optimizer, scheduler,
                              selected_datasets=dataset_names, model_config=model_config)
    torch.set_rng_state(rng_cpu)
    if rng_cuda is not None:
        torch.cuda.set_rng_state_all(rng_cuda)
    loss_ref = update_step(model, optimizer, scheduler, criterion, train_mod, item, check_grad=True)
    ref_next = state_snapshot(train_mod, model, optimizer, scheduler)
    require(math.isclose(loss_res, loss_ref, rel_tol=1e-6, abs_tol=1e-7),
            f"Next loss diverged: resumed={loss_res} ref={loss_ref}")
    compare_nested(res_next, ref_next, "resumed-next vs uninterrupted-next",
                   atol=2e-6 if device.type == "cuda" else 0,
                   rtol=2e-6 if device.type == "cuda" else 0)
    report("Next Batch: loss, updated parameters, optimizer moments, scheduler agree")

    corrupt = dict(saved)
    corrupt["architecture"] = "legacy-pyramid"
    try:
        train_mod.load_model_state_from_checkpoint(corrupt, model)
    except RuntimeError:
        report("Legacy checkpoint is rejected")
    else:
        raise AssertionError("Legacy architecture silently accepted")
    try:
        train_mod.load_checkpoint(file_path, model, optimizer, scheduler,
                                  selected_datasets=("wrong-dataset",))
    except RuntimeError:
        report("Mismatched dataset selection is rejected")
    else:
        raise AssertionError("Dataset mismatch silently accepted")
    return saved


def run_mock(args):
    # Test fixtures only stub external FOUNDATION models, not PAIR adapters/decoder/loss.
    setup_mock_imports()
    import train
    from loss import PAIRSemanticChangeLoss

    # Tiny FakeUtonia exposes itself as pretrained model for checkpoint exclusions.
    if not hasattr(FakeUtonia, "model"):
        FakeUtonia.model = property(lambda self: self)
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    items = make_mock_data()

    def factory():
        torch.manual_seed(777)
        return new_pair()

    model = factory()
    criterion = PAIRSemanticChangeLoss()
    optimizer, _, _ = train.build_optimizer(model, train_settings())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: .95 ** s)
    model.train()
    print("[INFO] MOCK: real PAIR modules/train.py + fake Qwen/Utonia", flush=True)
    for item in items[:3]:
        update_step(model, optimizer, scheduler, criterion, train, item)
    report("2D SCD + 2D BCD + 3D Event/Box: forward/loss/backward/grad/update")
    # Verify recovery of saved epoch/update/step metadata.
    # These values simulate a checkpoint taken inside an epoch.
    with tempfile.TemporaryDirectory(prefix="pair_resume_") as temp:
        checkpoint_roundtrip(train, model, optimizer, scheduler, items[3], criterion,
                             model_config=model_cfg,
                             dataset_names=("SECOND", "LEVIR-CD", "NYC-SCD"),
                             factory=factory, file_path=Path(temp) / "resume.pt")
    report("Mock checkpoint resume + gradients + main flow complete")


def make_actual_samples(config, dataset):
    from datasets.config_loader import load_experiment_config
    from datasets.pair_dataset import UnifiedPAIRDataset
    exp = load_experiment_config(Path(config), [dataset])
    ds_spec = exp.datasets[dataset]
    ds = UnifiedPAIRDataset(ds_spec.train_manifest, ds_spec.spec)
    require(len(ds) > 0, f"Empty training dataset: {dataset}")
    sample = ds[0]
    require(isinstance(sample, dict) and "target" in sample,
            f"Dataset {dataset} did not yield a PAIR sample")
    return exp, [sample]


def run_real(args):
    require(torch.cuda.is_available(), "Real mode requires CUDA. For CPU simulation run: python test4.py --mode mock")
    # No test stubs! Both Qwen3-VL and Utonia run from actual checkpoints.
    from models.pair import PAIRModel
    from loss import PAIRSemanticChangeLoss
    import train
    exp, samples = make_actual_samples(args.config, args.dataset)
    spec = exp.datasets[args.dataset].spec
    flags = dict(enable_2d=spec.route == "2d", enable_3d=spec.route == "3d",
                 enable_semantic=spec.label_mode == "semantic_pair")
    require(spec.route in ("2d", "3d"), "2d3d route requires calibrated data")
    print(f"[INFO] REAL: dataset={args.dataset}, route={spec.route}, "
          f"Qwen={exp.model['qwen_model']}, device=cuda:0", flush=True)
    # Same activation dtype as train.py; no fake Qwen / pretrained modules.
    torch.cuda.set_device(0)
    model = PAIRModel.from_config(exp.model, "cuda:0", **flags)
    criterion = PAIRSemanticChangeLoss().to("cuda:0")
    settings = train_settings(lr=args.lr)
    optimizer, _, _ = train.build_optimizer(model, settings)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: .95 ** s)
    model.train()
    # Actual dataset examples do NOT provide boxes by default. Do NOT use GT to
    # generate proposals, including during validation.
    item = (args.dataset, samples, "none")

    def real_step(check_grad=False):
        dataset, data, box_mode = item
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction, losses, target = train.forward_loss(model, criterion, data, spec,
                                                             box_mode=box_mode)
        validate_logits(dataset, prediction, target, semantic_classes=len(spec.class_names))
        require(bool(torch.isfinite(losses.total)), "Nonfinite real loss")
        losses.total.backward()
        nonzero = 0
        buckets = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                grp = train._architecture_module_bucket(name)
                entry = buckets.setdefault(grp, [0, 0])
                entry[0] += 1
                if param.grad is not None:
                    require(bool(torch.isfinite(param.grad).all()), f"Nonfinite gradient {name}")
                    if bool((param.grad != 0).any()):
                        entry[1] += 1
                        nonzero += 1
        require(nonzero > 0, "No nonzero gradient in actual model")
        before = [(p, p.detach().clone()) for name,p in model.named_parameters()
                  if p.requires_grad and p.grad is not None and bool((p.grad!=0).any())][:5]
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 3.0)
        optimizer.step(); scheduler.step()
        require(any(not torch.equal(p.detach(), old) for p, old in before),
                "No parameter update after optimizer.step")
        print(f"[INFO] loss={float(losses.total):.6f}, nonzero_grad_tensors={nonzero}", flush=True)
        for key, (total, active) in sorted(buckets.items()):
            print(f"       {key}: active gradients {active}/{total}", flush=True)
        return float(losses.total)

    real_step()
    report("Real Qwen/Utonia forward, original loss, backward, finite gradients, optimizer step")
    ck_dir = Path(args.save_dir).expanduser().resolve()
    ck_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ck_dir / f"pair_smoke_{args.dataset.replace('-', '_')}.pt"
    config = {"model": dict(exp.model), "optimizer": dict(exp.optimizer)}
    train.save_checkpoint(ckpt, model, optimizer, scheduler,
                          epoch=0, update_in_epoch=1, optimizer_step=1,
                          experiment=checkpoint_experiment((args.dataset,)),
                          resolved_config=config)
    original = state_snapshot(train, model, optimizer, scheduler)
    # Corrupt a trainable parameter to ensure reload truly restores it.
    name, first = next((n,p) for n,p in model.named_parameters() if p.requires_grad
                       and n in original["model"])
    with torch.no_grad(): first.add_(.25)
    require(not torch.equal(first.cpu(), original["model"][name]), "Mutation failed")
    position = train.load_checkpoint(ckpt, model, optimizer, scheduler,
                                      selected_datasets=(args.dataset,), model_config=exp.model)
    require(position[:3] == (0,1,1), "Real resume position mismatch")
    compare_nested(original, state_snapshot(train, model, optimizer, scheduler),
                   "real model/optimizer/scheduler", atol=0, rtol=0)
    report(f"Real checkpoint save/load + restored optimizer and scheduler: {ckpt}")
    # One post-resume update; strong compatibility test, not data iterator resume.
    real_step()
    report("Real resumed optimization step works")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["mock", "real"], default="real",
                        help="Default real: run CUDA/Qwen on your data. Use --mode mock for CPU only.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pair_train_qwen4b.json")
    parser.add_argument("--dataset", choices=["SECOND", "LEVIR-CD", "LandsatSCD", "NYC-SCD"],
                        default="SECOND", help="Real mode: a dataset with local training samples")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--save-dir", type=Path, default=Path("outputs/pair_smoke"))
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(2026); np.random.seed(2026); torch.manual_seed(2026)
    if args.mode == "mock":
        run_mock(args)
    else:
        run_real(args)
    print("\n[WARN] train.py v2 checkpoint lacks RNG and sampler/worker state: "
          "this test cannot certify bitwise-identical mid-epoch full-run continuation.", flush=True)
    print("[PASS] ALL REQUESTED SMOKE CHECKS COMPLETED", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[FAIL] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)
