import argparse
from typing import Any, Dict, Optional, Tuple, Callable

import torch
import triton
from transformers import AutoConfig
from flashinfer import SegmentGEMMWrapper
from vllm import _custom_ops as ops
import sgl_kernel

from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
    fused_moe as fused_moe_sglang,
)


segment_gemm = None

def init_flashinfer_segment_gemm():
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")
    global segment_gemm
    segment_gemm = SegmentGEMMWrapper(workspace_buffer)

def trt_fused_moe(hidden_states: torch.Tensor, # [num_tokens, hidden_size]
    w1: torch.Tensor,            # [num_experts, 2 * inter_size, hidden_size]
    w2: torch.Tensor,            # [num_experts, hidden_size, inter_size]
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    inplace: bool=False,
    use_grouped_topk: bool = False,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    custom_routing_function: Optional[Callable] = None,
    use_fp8_w8a8: bool=False,
    use_int8_w8a16: bool=False,
    override_config: Optional[Dict[str, Any]] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None
) -> torch.Tensor:
    num_tokens = gating_output.shape[:-1].numel()
    num_experts = gating_output.shape[-1]
    hidden_size = hidden_states.shape[-1]
    inter_size = w2.shape[-1]
    dtype = hidden_states.dtype
    device = hidden_states.device

    assert gating_output.shape[1] == w1.shape[0], "Number of experts mismatch"
    assert use_fp8_w8a8 == False, "FlashInfer Grouped GEMM not support fp8 dtype now"

    intermediate_cache1 = torch.empty(num_tokens * topk, inter_size * 2, dtype=dtype, device=device)
    intermediate_cache2 = torch.empty(num_tokens * topk, inter_size, dtype=dtype, device=device)
    intermediate_cache3 = torch.empty(num_tokens * topk, hidden_size, dtype=dtype, device=device)
    output_tokens = torch.empty_like(hidden_states)

    topk_weights = torch.empty(num_tokens, topk, dtype=torch.float32, device=device)
    topk_ids = torch.empty(num_tokens, topk, dtype=torch.int32, device=device)
    token_expert_indicies = torch.empty_like(topk_ids)

    ops.topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indicies,
        gating_output.float(),
    )
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    permuted_tokens = torch.empty(num_tokens * topk, hidden_size, dtype=dtype, device=device)
    cum_num_tokens_per_expert = torch.empty(num_experts, dtype=torch.int64, device=device)
    reverse_permutation_map = torch.empty(num_tokens * topk, dtype=torch.int32, device=device)
    sgl_kernel.trt_moe_expand_and_permute(
        permuted_tokens,
        cum_num_tokens_per_expert,
        reverse_permutation_map,
        hidden_states,
        topk_ids,
        token_expert_indicies,
    )

    # grouped gemm 1
    # A: [num_tokens * topk, hidden_size] = [M, K]
    # B: [num_experts, 2 * inter_size, hidden_size] = [E, N, K]
    # gmm(permuted_tokens, w1, intermediate_cache1, num_tokens_per_expert, weight_column_major=True)
    num_tokens_per_expert = torch.diff(cum_num_tokens_per_expert, prepend=torch.zeros(1, dtype=torch.int64, device=device))
    intermediate_cache1 = segment_gemm.run(permuted_tokens, w1, num_experts, True, seg_lens=num_tokens_per_expert)

    if use_fp8_w8a8:
        # intermediate_cache1.shape: [num_tokens * topk, inter_size * 2]
        # a1_scale.shape: [1]
        # w1_scale.shape: [num_experts]
        ops.silu_and_mul(intermediate_cache2, (intermediate_cache1 * a1_scale * w1_scale[topk_ids][:,:,None].reshape(-1, 1)).to(hidden_states.dtype).view(-1, inter_size * 2))
    else:
        ops.silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, inter_size * 2))

    # grouped gemm 2
    # A: [num_tokens * topk, inter_size] = [M, K]
    # B: [num_experts, hidden_size, inter_size] = [E, N, K]
    # gmm(intermediate_cache2, w2, intermediate_cache3, num_tokens_per_expert, weight_column_major=True)
    intermediate_cache3 = segment_gemm.run(intermediate_cache2, w2, num_experts, True, seg_lens=num_tokens_per_expert)

    if use_fp8_w8a8:
        sgl_kernel.trt_moe_unpermute_and_reduce(
            output_tokens,
            (intermediate_cache3 * a2_scale * w2_scale[topk_ids][:,:,None].reshape(-1, 1)).to(hidden_states.dtype).view(*intermediate_cache3.shape),
            topk_weights,
            topk_ids,
            reverse_permutation_map,
            renormalize,
        )
    else:
        sgl_kernel.trt_moe_unpermute_and_reduce(
            output_tokens,
            intermediate_cache3.view(*intermediate_cache3.shape),
            topk_weights,
            topk_ids,
            reverse_permutation_map,
            renormalize,
        )
    
    return output_tokens

def get_model_config(model_name: str, tp_size: int):
    """Get model configuration parameters"""
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    if config.architectures[0] == "DbrxForCausalLM":
        E = config.ffn_config.moe_num_experts
        topk = config.ffn_config.moe_top_k
        intermediate_size = config.ffn_config.ffn_hidden_size
        shard_intermediate_size = 2 * intermediate_size // tp_size
    elif config.architectures[0] == "JambaForCausalLM":
        E = config.num_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size
        shard_intermediate_size = 2 * intermediate_size // tp_size
    elif config.architectures[0] == "Qwen2MoeForCausalLM":
        E = config.num_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        shard_intermediate_size = 2 * intermediate_size // tp_size
    elif config.architectures[0] == "DeepseekV2ForCausalLM":
        E = config.n_routed_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    else:
        # Default: Mixtral
        E = config.num_local_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size
        shard_intermediate_size = 2 * intermediate_size // tp_size

    shape_configs = {
        "num_experts": E,
        "topk": topk,
        "hidden_size": config.hidden_size,
        "shard_intermediate_size": shard_intermediate_size,
        "dtype": config.torch_dtype,
    }
    print(f"{shape_configs=}")
    return shape_configs


def fused_moe_trt_api(
    x,
    w1,
    w2,
    input_gating,
    topk,
    use_fp8_w8a8=False,
    w1_scale=None,
    w2_scale=None,
    a1_scale=None,
    a2_scale=None,
):
    return trt_fused_moe(
        x,
        w1,
        w2,
        input_gating,
        topk,
        renormalize=True,
        inplace=True,
        use_fp8_w8a8=use_fp8_w8a8,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
    )


def fused_moe_sglang_api(
    x,
    w1,
    w2,
    input_gating,
    topk,
    use_fp8_w8a8=False,
    w1_scale=None,
    w2_scale=None,
    a1_scale=None,
    a2_scale=None,
):
    return fused_moe_sglang(
        x,
        w1,
        w2,
        input_gating,
        topk,
        renormalize=True,
        inplace=True,
        use_fp8_w8a8=use_fp8_w8a8,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
    )


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=list(range(1, 128)),
        line_arg="provider",
        line_vals=[
            "trt_fused_moe",
            "sglang_fused_moe_triton",
        ],
        line_names=[
            "trt_fused_moe",
            "sglang_fused_moe_triton",
        ],
        styles=[
            ("blue", "-"),
            ("green", "-"),
        ],
        ylabel="Time (ms)",
        plot_name="fused-moe-performance",
        args={},
    )
)
def benchmark(batch_size, provider, model_config, use_fp8=False):
    print(f"benchmark {provider} with batch_size={batch_size}")
    assert use_fp8 == False, "FlashInfer Grouped GEMM not support fp8 dtype now"
    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(0)

    num_tokens = batch_size
    num_experts = model_config["num_experts"]
    hidden_size = model_config["hidden_size"]
    shard_intermediate_size = model_config["shard_intermediate_size"]
    topk = model_config["topk"]
    dtype = model_config["dtype"]

    x = torch.randn(num_tokens, hidden_size, dtype=dtype)

    if use_fp8:
        init_dtype = dtype
        w1 = torch.randn(
            num_experts, shard_intermediate_size, hidden_size, dtype=init_dtype
        )
        w2 = torch.randn(
            num_experts, hidden_size, shard_intermediate_size // 2, dtype=init_dtype
        )
        w1 = w1.to(torch.float8_e4m3fn)
        w2 = w2.to(torch.float8_e4m3fn)
        w1_scale = torch.randn(num_experts, dtype=torch.float32)
        w2_scale = torch.randn(num_experts, dtype=torch.float32)
        a1_scale = torch.randn(1, dtype=torch.float32)
        a2_scale = torch.randn(1, dtype=torch.float32)
    else:
        w1 = torch.randn(num_experts, shard_intermediate_size, hidden_size, dtype=dtype)
        w2 = torch.randn(
            num_experts, hidden_size, shard_intermediate_size // 2, dtype=dtype
        )
        w1_scale = w2_scale = a1_scale = a2_scale = None

    input_gating = torch.randn(num_tokens, num_experts, dtype=torch.float32)

    # Warmup
    api_func = (
        fused_moe_trt_api
        if provider == "trt_fused_moe"
        else fused_moe_sglang_api
    )
    
    # Initialize FlashInfer segment_gemm if using trt_fused_moe
    if provider == "trt_fused_moe":
        init_flashinfer_segment_gemm()
        
    for _ in range(10):
        y = api_func(
            x,
            w1,
            w2,
            input_gating,
            topk,
            use_fp8_w8a8=use_fp8,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
        )
    torch.cuda.synchronize()

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: api_func(
            x,
            w1,
            w2,
            input_gating,
            topk,
            use_fp8_w8a8=use_fp8,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
        )[0],
        quantiles=quantiles,
    )
    return ms, min_ms, max_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1"
    )
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--use-fp8", action="store_true")
    parser.add_argument(
        "--save-path",
        type=str,
        default="./configs/benchmark_ops/vllm_sglang_fused_moe/",
    )
    args = parser.parse_args()

    model_config = get_model_config(args.model, args.tp_size)
    benchmark.run(
        show_plots=True,
        print_data=True,
        # save_path=args.save_path,
        model_config=model_config,
        use_fp8=args.use_fp8,
    )


if __name__ == "__main__":
    main()
