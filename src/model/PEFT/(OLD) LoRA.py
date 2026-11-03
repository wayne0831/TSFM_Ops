###########################################################################################################
# import libraries
###########################################################################################################

import torch
import torch.nn as nn
import math
import time
from config import *

###########################################################################################################
# define LoRA layer
###########################################################################################################

class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16, dropout=0.1):
        super().__init__()
        self.original_layer = original_layer
        self.original_layer.weight.requires_grad = False
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        self.A = nn.Parameter(torch.empty(rank, in_features), requires_grad=True)
        self.B = nn.Parameter(torch.empty(out_features, rank), requires_grad=True)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(p=dropout)

        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x):
        result = self.original_layer(x)
        lora_out = (self.dropout(x) @ self.A.t() @ self.B.t()) * self.scaling
        return result + lora_out

def apply_lora_to_tsfm(model, target_modules=['qkv_proj'], rank=4, alpha=16, dropout=0.1):
    for param in model.parameters():
        param.requires_grad = False

    for i, block in enumerate(model.stacked_xf):
        if 'qkv_proj' in target_modules:
            block.attn.qkv_proj = LoRALayer(block.attn.qkv_proj, rank, alpha, dropout)
        if 'out' in target_modules:
            block.attn.out = LoRALayer(block.attn.out, rank, alpha, dropout)
        if 'ff0' in target_modules:
            block.ff0 = LoRALayer(block.ff0, rank, alpha, dropout)
        if 'ff1' in target_modules:
            block.ff1 = LoRALayer(block.ff1, rank, alpha, dropout)
    
    for name, param in model.named_parameters():
        if ".A" in name or ".B" in name:
            param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    tr_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Apply LoRA: Target Modules -> {target_modules}")
    print(f"Total Params: {total_params:,} | Trainable: {tr_params:,} ({tr_params/total_params:.2%})")

    return model, tr_params/total_params

def remove_lora_from_tsfm(model):
    for i, block in enumerate(model.stacked_xf):
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

def train(model, train_loader, max_horizon, patch_size, lr, epochs):
    p_size = int(patch_size) # 32
    tr_params = [p for p in model.model.parameters() if p.requires_grad]

    if len(tr_params) == 0:
        raise RuntimeError("No trainable parameters found!")

    optimizer = torch.optim.AdamW(tr_params, lr=lr)
    criterion = nn.MSELoss()
    
    model.model.train()
    history = []

    print(f"\n🚀 Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        
        for batch_x, batch_y, batch_mask in train_loader:
            batch_x, batch_y, batch_mask = batch_x.to(DEVICE), batch_y.to(DEVICE), batch_mask.to(DEVICE)
            optimizer.zero_grad()

            # [추가] 배치 단위 내부 정규화 (Instance Normalization)
            batch_mean = batch_x.mean(dim=1, keepdim=True)
            batch_std = batch_x.std(dim=1, keepdim=True) + 1e-8
            
            # 입력값과 정답 모두 로컬 정규화 스케일로 변환
            batch_x_norm = (batch_x - batch_mean) / batch_std
            batch_y_norm = (batch_y - batch_mean) / batch_std
            
            num_patches = batch_x_norm.shape[1] // p_size
            inputs_ts = batch_x_norm.view(batch_x_norm.shape[0], num_patches, p_size)
            masks_ts = batch_mask.view(batch_mask.shape[0], num_patches, p_size)

            with torch.enable_grad():
                tokenizer_input = torch.cat([inputs_ts, masks_ts], dim=-1)
                
                token_out = model.model.tokenizer(tokenizer_input) 
                x = token_out[0] if isinstance(token_out, (tuple, list)) else token_out
                
                patch_mask_b = masks_ts.mean(dim=-1) 

                # 1. Transformer Blocks 통과
                for block in model.model.stacked_xf:
                    block_out = block(x, patch_mask=patch_mask_b)
                    x = block_out[0] if isinstance(block_out, (tuple, list)) else block_out
                
                # 2. Post Attention Layer Normalization 유연하게 찾기 (없으면 자동 패스)
                for attr_name in ['ln_f', 'norm', 'post_attn_ln', 'layernorm']:
                    if hasattr(model.model, attr_name):
                        norm_layer = getattr(model.model, attr_name)
                        x = norm_layer(x)
                        break 
                
                # 3. Final Output Projection(Head) 유연하게 찾기 (이름 추가됨)
                head_layer = None
                for attr_name in ['output_projection_point', 'head', 'output_layer', 'head_layer', 'output_proj']:
                    if hasattr(model.model, attr_name):
                        head_layer = getattr(model.model, attr_name)
                        break
                
                if head_layer is None:
                    module_names = [n for n, _ in model.model.named_children()]
                    raise AttributeError(f"출력 레이어(Head)를 찾을 수 없습니다. 현재 모듈: {module_names}")
                
                outputs = head_layer(x)
                
                while isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                
                # 4. Loss 계산을 위한 형태 맞추기
                if outputs.dim() == 4:
                    pred_all = outputs[:, :, :, 0].reshape(batch_x_norm.shape[0], -1)
                else:
                    pred_all = outputs.reshape(batch_x_norm.shape[0], -1)

                target_len = min(pred_all.shape[1], batch_y_norm.shape[1])
                # [수정] Loss 계산 시 스케일링 된 타겟 사용
                loss = criterion(pred_all[:, -target_len:], batch_y_norm[:, :target_len])

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.6f} | Time: {time.time() - epoch_start_time:.2f}s")
    
    return model, history