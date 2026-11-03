import os
import sys

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch  # noqa: E402
    from timesfm.torch import util as tfm_util  # noqa: E402
except ImportError:
    from timesfm_test.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch  # noqa: E402
    from timesfm_test.torch import util as tfm_util  # noqa: E402


class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16, dropout=0.1):
        super().__init__()
        self.original_layer = original_layer

        for p in self.original_layer.parameters():
            p.requires_grad_(False)
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        device = original_layer.weight.device
        dtype = original_layer.weight.dtype
        
        self.A = nn.Parameter(torch.empty(rank, in_features, device=device, dtype=dtype), requires_grad=True)
        self.B = nn.Parameter(torch.empty(out_features, rank, device=device, dtype=dtype), requires_grad=True)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(p=dropout)

        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x):
        result   = self.original_layer(x)
        lora_out = (self.dropout(x) @ self.A.t() @ self.B.t()) * self.scaling
        
        return result + lora_out


def apply_lora_to_tsfm(model, target_modules=['qkv_proj'], rank=4, alpha=16, dropout=0.1):
    for param in model.parameters():
        param.requires_grad = False

    adapters = {}
    for i, block in enumerate(model.stacked_xf):
        if 'qkv_proj' in target_modules:
            block.attn.qkv_proj = LoRALayer(block.attn.qkv_proj, rank, alpha, dropout)
            adapters[f"{i}.qkv"] = block.attn.qkv_proj
        if 'out' in target_modules:
            block.attn.out = LoRALayer(block.attn.out, rank, alpha, dropout)
            adapters[f"{i}.out"] = block.attn.out
        if 'ff0' in target_modules:
            block.ff0 = LoRALayer(block.ff0, rank, alpha, dropout)
            adapters[f"{i}.ff0"] = block.ff0
        if 'ff1' in target_modules:
            block.ff1 = LoRALayer(block.ff1, rank, alpha, dropout)
            adapters[f"{i}.ff1"] = block.ff1

    total_params = sum(p.numel() for p in model.parameters())
    tr_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Apply LoRA: Target Modules -> {target_modules}")
    print(f"Total Params: {total_params:,} | Trainable params: {tr_params:,} | Trainable ratio: ({tr_params/total_params:.2%})")

    return adapters


def get_trainable_params_ratio(model):
    total_params = sum(p.numel() for p in model.parameters())
    tr_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return tr_params / total_params if total_params > 0 else 0.0


def remove_lora_from_tsfm(model):
    for block in model.stacked_xf:
        if isinstance(block.attn.qkv_proj, LoRALayer):
            block.attn.qkv_proj = block.attn.qkv_proj.original_layer
        if isinstance(block.attn.out, LoRALayer):
            block.attn.out = block.attn.out.original_layer
        if hasattr(block, 'ff0') and isinstance(block.ff0, LoRALayer):
            block.ff0 = block.ff0.original_layer
        if hasattr(block, 'ff1') and isinstance(block.ff1, LoRALayer):
            block.ff1 = block.ff1.original_layer
    print("✅ LoRA layers reverted.")
    return model


def save_lora(adapters, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({k: (a.A.detach().cpu(), a.B.detach().cpu())
                for k, a in adapters.items()}, path)


def load_lora(adapters, path):
    state = torch.load(path, map_location="cpu")
    for k, a in adapters.items():
        A, B = state[k]
        a.A.data.copy_(A.to(a.A.device))
        a.B.data.copy_(B.to(a.B.device))


def reset_lora_parameters(adapters):
    """각 Sector별 LoRA 어댑터를 독립적으로 학습시키기 전에 가중치(A, B)를 초기 상태로 리셋합니다."""
    for a in adapters.values():
        nn.init.kaiming_uniform_(a.A, a=math.sqrt(5))
        nn.init.zeros_(a.B)


def get_lora_state(adapters):
    """현재 장착된 LoRA 어댑터의 가중치(A, B) 사본을 CPU 딕셔너리로 추출합니다."""
    return {k: (a.A.detach().cpu().clone(), a.B.detach().cpu().clone()) for k, a in adapters.items()}


def set_lora_state(adapters, state):
    """사전 학습된 특정 Sector 전문가의 LoRA 가중치(A, B)를 백본에 온디맨드로 주입(Stitching)합니다."""
    for k, a in adapters.items():
        A, B = state[k]
        a.A.data.copy_(A.to(a.A.device))
        a.B.data.copy_(B.to(a.B.device))
