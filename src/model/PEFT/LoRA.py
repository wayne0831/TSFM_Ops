import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16, dropout=0.1):
        super().__init__()
        self.original_layer = original_layer

        for p in self.original_layer.parameters():
            p.requires_grad_(False)

        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        device = original_layer.weight.device
        dtype = original_layer.weight.dtype

        self.A = nn.Parameter(torch.empty(rank, self.in_features, device=device, dtype=dtype), requires_grad=True)
        self.B = nn.Parameter(torch.empty(self.out_features, rank, device=device, dtype=dtype), requires_grad=True)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(p=dropout)
        self.enabled = True

        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    @property
    def weight(self):
        return self.original_layer.weight

    @property
    def bias(self):
        return self.original_layer.bias

    def forward(self, x):
        result = self.original_layer(x)
        if not getattr(self, "enabled", True):
            return result
        lora_out = (self.dropout(x) @ self.A.t() @ self.B.t()) * self.scaling
        return result + lora_out


def _unwrap_lora_layer(layer):
    return layer.original_layer if isinstance(layer, LoRALayer) else layer


def _normalize_chronos_targets(target_modules):
    """
    Chronos-2 고유 모듈명(q, k, v, o, wi, wo, output_layer)과
    TimesFM 스타일 공통 별칭(qkv_proj, out, ff0, ff1)을 모두 지원합니다.
    """
    alias_map = {
        "qkv_proj": ["q", "k", "v"],
        "out": ["o"],
        "out_proj": ["o"],
        "ff0": ["wi"],
        "ff1": ["wo"],
    }
    resolved = []
    for m in target_modules:
        m_clean = str(m).strip()
        for mapped in alias_map.get(m_clean, [m_clean]):
            if mapped not in resolved:
                resolved.append(mapped)
    return resolved


def _normalize_moirai_targets(target_modules):
    """
    Moirai-2 고유 모듈명(q_proj, k_proj, v_proj, out_proj, fc1, fc_gate, fc2, output_layer)과
    TimesFM 스타일 공통 별칭(qkv_proj, out, ff0, ff1)을 모두 지원합니다.
    """
    alias_map = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "q": ["q_proj"],
        "k": ["k_proj"],
        "v": ["v_proj"],
        "out": ["out_proj"],
        "o": ["out_proj"],
        "ff0": ["fc1", "fc_gate"],
        "ff1": ["fc2"],
    }
    resolved = []
    for m in target_modules:
        m_clean = str(m).strip()
        for mapped in alias_map.get(m_clean, [m_clean]):
            if mapped not in resolved:
                resolved.append(mapped)
    return resolved


def apply_lora_to_chronos(model, target_modules=['q', 'k', 'v', 'o'], rank=8, alpha=16, dropout=0.1):
    """
    Amazon Chronos-2 (Chronos2Model 또는 Chronos2Pipeline)의 인코더 블록 내
    TimeSelfAttention, GroupSelfAttention 및 FeedForward(MLP) 선형 레이어에 LoRA 어댑터를 장착합니다.
    """
    core_model = model.model if hasattr(model, "model") and hasattr(model.model, "encoder") else model
    for param in core_model.parameters():
        param.requires_grad = False

    resolved_targets = _normalize_chronos_targets(target_modules)
    adapters = {}

    for i, block in enumerate(core_model.encoder.block):
        # block.layer[0]: TimeSelfAttention, block.layer[1]: GroupSelfAttention
        for attn_tag, layer_idx in (("time_attn", 0), ("group_attn", 1)):
            attn_mod = block.layer[layer_idx].self_attention
            for proj_name in ("q", "k", "v", "o"):
                if proj_name in resolved_targets or f"self_attention.{proj_name}" in resolved_targets:
                    orig = _unwrap_lora_layer(getattr(attn_mod, proj_name))
                    wrapped = LoRALayer(orig, rank, alpha, dropout)
                    setattr(attn_mod, proj_name, wrapped)
                    adapters[f"{i}.{attn_tag}.{proj_name}"] = wrapped

        # block.layer[2]: FeedForward (MLP: wi, wo)
        mlp_mod = block.layer[2].mlp
        for ff_name in ("wi", "wo"):
            if ff_name in resolved_targets or f"mlp.{ff_name}" in resolved_targets:
                orig = _unwrap_lora_layer(getattr(mlp_mod, ff_name))
                wrapped = LoRALayer(orig, rank, alpha, dropout)
                setattr(mlp_mod, ff_name, wrapped)
                adapters[f"{i}.mlp.{ff_name}"] = wrapped

    # 선택적 출력 패치 임베딩 헤드
    if (
        "output_layer" in resolved_targets
        or "output_patch_embedding.output_layer" in resolved_targets
    ) and hasattr(core_model, "output_patch_embedding"):
        orig = _unwrap_lora_layer(core_model.output_patch_embedding.output_layer)
        wrapped = LoRALayer(orig, rank, alpha, dropout)
        core_model.output_patch_embedding.output_layer = wrapped
        adapters["head.output_layer"] = wrapped

    if not adapters:
        raise ValueError(f"Chronos-2에서 일치하는 LoRA 대상 모듈을 찾지 못했습니다: {target_modules}")

    total_params = sum(p.numel() for p in core_model.parameters())
    tr_params = sum(p.numel() for p in core_model.parameters() if p.requires_grad)
    print(f"Apply LoRA (Chronos-2): Target Modules -> {target_modules} (resolved={resolved_targets})")
    print(f"Total Params: {total_params:,} | Trainable params: {tr_params:,} | Trainable ratio: ({tr_params/total_params:.2%})")

    return adapters


def remove_lora_from_chronos(model):
    core_model = model.model if hasattr(model, "model") and hasattr(model.model, "encoder") else model
    for block in core_model.encoder.block:
        for layer_idx in (0, 1):
            attn_mod = block.layer[layer_idx].self_attention
            for proj_name in ("q", "k", "v", "o"):
                if isinstance(getattr(attn_mod, proj_name, None), LoRALayer):
                    setattr(attn_mod, proj_name, getattr(attn_mod, proj_name).original_layer)
        mlp_mod = block.layer[2].mlp
        for ff_name in ("wi", "wo"):
            if isinstance(getattr(mlp_mod, ff_name, None), LoRALayer):
                setattr(mlp_mod, ff_name, getattr(mlp_mod, ff_name).original_layer)

    if hasattr(core_model, "output_patch_embedding") and isinstance(
        getattr(core_model.output_patch_embedding, "output_layer", None), LoRALayer
    ):
        core_model.output_patch_embedding.output_layer = core_model.output_patch_embedding.output_layer.original_layer

    print("✅ LoRA layers reverted (Chronos-2).")
    return model


def apply_lora_to_moirai(model, target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj'], rank=8, alpha=16, dropout=0.1):
    """
    Salesforce Moirai-2 (Moirai2Forecast 또는 Moirai2Module)의 트랜스포머 레이어 내
    GroupedQueryAttention(q_proj, k_proj, v_proj, out_proj) 및 GLU FeedForward(fc1, fc_gate, fc2)에 LoRA를 장착합니다.
    """
    core_module = model.module if hasattr(model, "module") and hasattr(model.module, "encoder") else model
    for param in core_module.parameters():
        param.requires_grad = False

    resolved_targets = _normalize_moirai_targets(target_modules)
    adapters = {}

    for i, layer in enumerate(core_module.encoder.layers):
        attn_mod = layer.self_attn
        for proj_name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            if proj_name in resolved_targets:
                orig = _unwrap_lora_layer(getattr(attn_mod, proj_name))
                wrapped = LoRALayer(orig, rank, alpha, dropout)
                setattr(attn_mod, proj_name, wrapped)
                adapters[f"{i}.attn.{proj_name}"] = wrapped

        ffn_mod = layer.ffn
        for ff_name in ("fc1", "fc_gate", "fc2"):
            if ff_name in resolved_targets and hasattr(ffn_mod, ff_name):
                orig = _unwrap_lora_layer(getattr(ffn_mod, ff_name))
                wrapped = LoRALayer(orig, rank, alpha, dropout)
                setattr(ffn_mod, ff_name, wrapped)
                adapters[f"{i}.ffn.{ff_name}"] = wrapped

    if "output_layer" in resolved_targets and hasattr(core_module, "out_proj") and hasattr(core_module.out_proj, "output_layer"):
        orig = _unwrap_lora_layer(core_module.out_proj.output_layer)
        wrapped = LoRALayer(orig, rank, alpha, dropout)
        core_module.out_proj.output_layer = wrapped
        adapters["head.output_layer"] = wrapped

    if not adapters:
        raise ValueError(f"Moirai-2에서 일치하는 LoRA 대상 모듈을 찾지 못했습니다: {target_modules}")

    total_params = sum(p.numel() for p in core_module.parameters())
    tr_params = sum(p.numel() for p in core_module.parameters() if p.requires_grad)
    print(f"Apply LoRA (Moirai-2): Target Modules -> {target_modules} (resolved={resolved_targets})")
    print(f"Total Params: {total_params:,} | Trainable params: {tr_params:,} | Trainable ratio: ({tr_params/total_params:.2%})")

    return adapters


def remove_lora_from_moirai(model):
    core_module = model.module if hasattr(model, "module") and hasattr(model.module, "encoder") else model
    for layer in core_module.encoder.layers:
        attn_mod = layer.self_attn
        for proj_name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            if isinstance(getattr(attn_mod, proj_name, None), LoRALayer):
                setattr(attn_mod, proj_name, getattr(attn_mod, proj_name).original_layer)

        ffn_mod = layer.ffn
        for ff_name in ("fc1", "fc_gate", "fc2"):
            if hasattr(ffn_mod, ff_name) and isinstance(getattr(ffn_mod, ff_name, None), LoRALayer):
                setattr(ffn_mod, ff_name, getattr(ffn_mod, ff_name).original_layer)

    if hasattr(core_module, "out_proj") and isinstance(getattr(core_module.out_proj, "output_layer", None), LoRALayer):
        core_module.out_proj.output_layer = core_module.out_proj.output_layer.original_layer

    print("✅ LoRA layers reverted (Moirai-2).")
    return model


def apply_lora_to_tsfm(model, target_modules=['qkv_proj'], rank=4, alpha=16, dropout=0.1):
    """
    TSFM 백본 종류(TimesFM 2.5 / Chronos 2 / Moirai 2)를 자동 감지하여 LoRA 어댑터를 장착합니다.
    """
    # 1) TimesFM 2.5
    if hasattr(model, "stacked_xf"):
        for param in model.parameters():
            param.requires_grad = False

        adapters = {}
        for i, block in enumerate(model.stacked_xf):
            if 'qkv_proj' in target_modules:
                block.attn.qkv_proj = LoRALayer(_unwrap_lora_layer(block.attn.qkv_proj), rank, alpha, dropout)
                adapters[f"{i}.qkv"] = block.attn.qkv_proj
            if 'out' in target_modules:
                block.attn.out = LoRALayer(_unwrap_lora_layer(block.attn.out), rank, alpha, dropout)
                adapters[f"{i}.out"] = block.attn.out
            if 'ff0' in target_modules:
                block.ff0 = LoRALayer(_unwrap_lora_layer(block.ff0), rank, alpha, dropout)
                adapters[f"{i}.ff0"] = block.ff0
            if 'ff1' in target_modules:
                block.ff1 = LoRALayer(_unwrap_lora_layer(block.ff1), rank, alpha, dropout)
                adapters[f"{i}.ff1"] = block.ff1

        total_params = sum(p.numel() for p in model.parameters())
        tr_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Apply LoRA: Target Modules -> {target_modules}")
        print(f"Total Params: {total_params:,} | Trainable params: {tr_params:,} | Trainable ratio: ({tr_params/total_params:.2%})")
        return adapters

    # 2) Amazon Chronos 2 (Chronos2Model or Chronos2Pipeline)
    core_chronos = model.model if hasattr(model, "model") and hasattr(model.model, "encoder") else model
    if hasattr(core_chronos, "chronos_config") and hasattr(core_chronos, "encoder") and hasattr(core_chronos.encoder, "block"):
        return apply_lora_to_chronos(model, target_modules=target_modules, rank=rank, alpha=alpha, dropout=dropout)

    # 3) Salesforce Moirai 2 (Moirai2Forecast or Moirai2Module)
    core_moirai = model.module if hasattr(model, "module") and hasattr(model.module, "encoder") else model
    if hasattr(core_moirai, "encoder") and hasattr(core_moirai.encoder, "layers"):
        return apply_lora_to_moirai(model, target_modules=target_modules, rank=rank, alpha=alpha, dropout=dropout)

    raise TypeError(f"지원하지 않는 TSFM 모델 구조입니다: {type(model)}")


def get_trainable_params_ratio(model):
    if hasattr(model, "model") and isinstance(model.model, nn.Module):
        target_mod = model.model
    elif hasattr(model, "module") and isinstance(model.module, nn.Module):
        target_mod = model.module
    else:
        target_mod = model
    total_params = sum(p.numel() for p in target_mod.parameters())
    tr_params = sum(p.numel() for p in target_mod.parameters() if p.requires_grad)
    return tr_params / total_params if total_params > 0 else 0.0


def remove_lora_from_tsfm(model):
    if hasattr(model, "stacked_xf"):
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

    core_chronos = model.model if hasattr(model, "model") and hasattr(model.model, "encoder") else model
    if hasattr(core_chronos, "chronos_config") and hasattr(core_chronos, "encoder") and hasattr(core_chronos.encoder, "block"):
        return remove_lora_from_chronos(model)

    core_moirai = model.module if hasattr(model, "module") and hasattr(model.module, "encoder") else model
    if hasattr(core_moirai, "encoder") and hasattr(core_moirai.encoder, "layers"):
        return remove_lora_from_moirai(model)

    raise TypeError(f"지원하지 않는 TSFM 모델 구조입니다: {type(model)}")


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
        a.enabled = True


def enable_lora(adapters, enabled: bool = True):
    """LoRA 어댑터 활성/비활성 상태를 전환합니다. (enabled=False이면 순수 TSFM 백본만 동작)"""
    for a in adapters.values():
        a.enabled = enabled


def get_lora_state(adapters):
    """현재 장착된 LoRA 어댑터의 가중치(A, B) 사본을 CPU 딕셔너리로 추출합니다."""
    return {k: (a.A.detach().cpu().clone(), a.B.detach().cpu().clone()) for k, a in adapters.items()}


def set_lora_state(adapters, state):
    """사전 학습된 특정 Sector 전문가의 LoRA 가중치(A, B)를 백본에 온디맨드로 주입(Stitching)합니다."""
    for k, a in adapters.items():
        A, B = state[k]
        a.A.data.copy_(A.to(a.A.device))
        a.B.data.copy_(B.to(a.B.device))
        a.enabled = True
