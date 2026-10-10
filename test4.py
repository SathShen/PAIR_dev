#!/usr/bin/env python3
"""Standalone REAL GPU joint Qwen grounding+mask checkpoint smoke.

Run from project root:
  CUDA_VISIBLE_DEVICES=2 python test_joint_box_gpu.py --config configs/pair_train_qwen4b_joint.json --dataset SECOND
  CUDA_VISIBLE_DEVICES=2 python test_joint_box_gpu.py --config configs/pair_train_qwen4b_joint.json --dataset NYC-SCD

No dependencies on tests/test_train_v2.py, test4.py or other fixture files.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace
import torch

from datasets.config_loader import load_experiment_config
from datasets.pair_dataset import UnifiedPAIRDataset
from models.change_decoder import build_grounding_supervision
from loss import PAIRSemanticChangeLoss
from models.pair import PAIRModel
import train


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--config',type=Path,required=True)
    a.add_argument('--dataset',default='SECOND')
    a.add_argument('--output-dir',type=Path,default=Path('outputs/pair_joint_box_smoke'))
    a.add_argument('--max-attempts',type=int,default=8)
    args=a.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for REAL smoke')
    experiment=load_experiment_config(args.config,[args.dataset])
    s=train.build_settings(experiment,SimpleNamespace(output_dir=args.output_dir,resume=None))
    train.validate_box_modes(s)
    if s.train_box_mode!='joint':
        raise RuntimeError('Supply *_joint.json config with training.box_mode=joint')
    handle=experiment.datasets[args.dataset]
    spec=handle.spec
    dataset=UnifiedPAIRDataset(handle.train_manifest,spec)
    sample=None
    for i in range(min(len(dataset),args.max_attempts)):
        candidate=dataset[i]
        boxes=build_grounding_supervision([candidate],spec.route,
               max_boxes=s.box_gt_max_boxes,device='cpu',box_dropout=0)
        if boxes['num_boxes']:
            sample=candidate
            break
    if sample is None:
        raise RuntimeError('No changed samples in first attempts; increase --max-attempts')
    print('[INFO] route:',spec.route,'sample:',sample.get('sample_id'),
          'weak_gt_boxes:',boxes['num_boxes'],flush=True)
    model=PAIRModel.from_config(experiment.model,torch.device('cuda:0'),
             **train.active_model_flags(experiment))
    model.train()
    criterion=PAIRSemanticChangeLoss().cuda()
    optimizer,_,_=train.build_optimizer(model,s)
    scheduler=train.build_scheduler(optimizer,20,s.warmup_ratio,s.scheduler)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        _, out, _=train.forward_loss(model,criterion,[sample],spec,
          box_mode='joint',box_ce_weight=s.box_ce_weight,
          box_gt_max_boxes=s.box_gt_max_boxes,box_dropout=0)
    assert torch.isfinite(out.total) and torch.isfinite(out.grounding_ce)
    print(f'[INFO] Mask Loss={out.mask_total.detach().item():.5f}; Box CE={out.grounding_ce.detach().item():.5f}',flush=True)
    out.total.backward()
    grad_count={}
    for name, param in model.named_parameters():
        if param.grad is not None and torch.count_nonzero(param.grad).item()>0:
            bucket=('qwen_lora' if 'lora' in name.lower() else
                    'box_guidance' if 'box_guidance' in name else
                    'image_adapter' if 'image_adapter' in name else
                    'point_adapter' if 'point_adapter' in name else
                    'decoder' if 'decoder' in name else 'other')
            grad_count[bucket]=grad_count.get(bucket,0)+1
        if param.grad is not None and not torch.isfinite(param.grad).all():
            raise RuntimeError('Nonfinite gradient: '+name)
    print('[INFO] nonzero gradients:',grad_count,flush=True)
    assert grad_count.get('decoder',0)>0
    # No PyTorch tensor detach trick can make this pass accidentally.
    assert out.grounding_ce.requires_grad
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],s.max_grad_norm)
    optimizer.step()
    scheduler.step()
    path=args.output_dir / f'joint_{args.dataset.replace("-","_")}.pt'
    resolved=experiment.resolved_dict(runtime={'smoke':True})
    train.save_checkpoint(path,model,optimizer,scheduler,0,1,1,
                          experiment,resolved)
    train.load_checkpoint(path,model,optimizer,scheduler,
                          selected_datasets=experiment.selected_names,
                          model_config=experiment.model)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        _, next_loss,_=train.forward_loss(model,criterion,[sample],spec,
           box_mode='joint',box_ce_weight=s.box_ce_weight,
           box_gt_max_boxes=s.box_gt_max_boxes,box_dropout=0)
    next_loss.total.backward()
    optimizer.step()
    print('[PASS] REAL joint forward/backward/optimizer/Checkpoint resume',flush=True)
    print('[INFO] Checkpoint:',path,flush=True)

if __name__=='__main__':main()
