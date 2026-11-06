"""
merge_lora.py

从训练checkpoint中读取LoRA增量，合并进base权重，
输出一个不含任何peft结构的纯净safetensors。

用法:
    python merge_lora.py \
        --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
        --checkpoint_path /path/to/checkpoint-20000/model.safetensors \
        --output_path ./merged/model.safetensors \
        --lora_r 128 \
        --lora_alpha 32
"""

import argparse
import os
from collections import defaultdict

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str,
                        default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_path",     type=str, required=True)
    parser.add_argument("--lora_r",     type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=32)
    return parser.parse_args()


def collect_lora_pairs(ckpt: dict, scaling: float) -> dict:
    """
    从ckpt中收集所有lora_A/B对，返回:
      {(component, base_key): merged_delta_tensor}
    其中 component ∈ {"encoder","decoder"}
        base_key 与 base_sd 的key完全一致，例如 "model.layers.0.self_attn.q_proj.weight"

    ckpt key格式:
      {component}.base_model.model.{base_key_without_weight}.lora_{A|B}.default.weight
    例如:
      decoder.base_model.model.model.layers.0.mlp.down_proj.lora_A.default.weight
      -> component = "decoder"
      -> base_key  = "model.layers.0.mlp.down_proj.weight"
    """
    A_dict = {}  # (component, base_key) -> tensor
    B_dict = {}

    for k, v in ckpt.items():
        if ".lora_A.default.weight" in k:
            # decoder.base_model.model.model.layers.0.mlp.down_proj.lora_A.default.weight
            component = k.split(".")[0]                          # "decoder"
            # 去掉 "{component}.base_model.model." 前缀，再去掉 ".lora_A.default.weight" 后缀
            prefix = f"{component}.base_model.model."
            suffix = ".lora_A.default.weight"
            layer_path = k[len(prefix): -len(suffix)]           # "model.layers.0.mlp.down_proj"
            base_key = layer_path + ".weight"                   # "model.layers.0.mlp.down_proj.weight"
            A_dict[(component, base_key)] = v

        elif ".lora_B.default.weight" in k:
            component = k.split(".")[0]
            prefix = f"{component}.base_model.model."
            suffix = ".lora_B.default.weight"
            layer_path = k[len(prefix): -len(suffix)]
            base_key = layer_path + ".weight"
            B_dict[(component, base_key)] = v

    # 合并 A、B -> delta
    deltas = {}
    for key, A in A_dict.items():
        if key in B_dict:
            B = B_dict[key]
            deltas[key] = (B.float() @ A.float()) * scaling
        else:
            print(f"[警告] 找到lora_A但缺少lora_B: {key}")

    missing_b = set(A_dict) - set(B_dict)
    missing_a = set(B_dict) - set(A_dict)
    if missing_b:
        print(f"[警告] 缺少lora_B的key数: {len(missing_b)}")
    if missing_a:
        print(f"[警告] 缺少lora_A的key数: {len(missing_a)}")

    return deltas


def merge(args):
    scaling = args.lora_alpha / args.lora_r
    print(f"LoRA scaling = {args.lora_alpha} / {args.lora_r} = {scaling}\n")

    # ── 1. 加载ckpt ──────────────────────────────────────────────────────────
    print(f"加载ckpt: {args.checkpoint_path}")
    ckpt = load_file(args.checkpoint_path, device="cpu")
    print(f"ckpt共 {len(ckpt)} 个key，示例:")
    for k in list(ckpt.keys())[:3]:
        print(f"  {k}")

    # ── 2. 加载base模型权重 ───────────────────────────────────────────────────
    print(f"\n加载base模型: {args.model_name_or_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    base_sd = base_model.state_dict()
    print(f"base模型共 {len(base_sd)} 个key，示例:")
    for k in list(base_sd.keys())[:3]:
        print(f"  {k}")
    del base_model

    # ── 3. 收集lora对并计算delta ──────────────────────────────────────────────
    print("\n解析LoRA权重...")
    deltas = collect_lora_pairs(ckpt, scaling)
    print(f"找到 {len(deltas)} 个有效LoRA层对")

    # 验证：LoRA对应的base_key是否都在base_sd里
    unmatched = [(c, bk) for (c, bk) in deltas if bk not in base_sd]
    if unmatched:
        print(f"[错误] 以下LoRA层在base_sd中找不到对应key（前5条）:")
        for c, bk in unmatched[:5]:
            print(f"  component={c}  base_key={bk}")
        raise RuntimeError("LoRA key与base模型不匹配，请检查model_name_or_path是否正确")

    # ── 4. 构建merged_sd ──────────────────────────────────────────────────────
    # 对每个component，所有base权重都需要独立clone（断开shared memory）
    merged_sd = {}
    lora_count = 0
    base_count = 0

    for component in ("encoder", "decoder"):
        for base_key, base_weight in base_sd.items():
            target_key = f"{component}.{base_key}"
            if (component, base_key) in deltas:
                delta = deltas[(component, base_key)].to(base_weight.dtype)
                merged_sd[target_key] = (base_weight.float() + delta.float()).to(base_weight.dtype)
                lora_count += 1
            else:
                # 必须clone，否则encoder和decoder会共享同一内存块
                merged_sd[target_key] = base_weight.clone()
                base_count += 1

    print(f"\n合并统计:")
    print(f"  LoRA合并层数:   {lora_count}")
    print(f"  直接复制base层: {base_count}")
    print(f"  输出总key数:    {len(merged_sd)}")

    # ── 5. 保存 ───────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    print(f"\n保存至: {args.output_path}")
    save_file(merged_sd, args.output_path)
    print("=== 合并完成！===")

    print("\n输出key样本（前6条）:")
    for k in list(merged_sd.keys())[:6]:
        print(f"  {k}  shape={merged_sd[k].shape}")


if __name__ == "__main__":
    args = parse_args()
    merge(args)
