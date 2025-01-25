from typing import Any, Dict, Optional, Tuple, Callable
import unittest
import torch

from flashinfer import SegmentGEMMWrapper
from vllm import _custom_ops as ops

import sgl_kernel
from sglang.srt.layers.moe.topk import select_experts
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
    fused_moe as fused_moe_sglang,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe

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


class TestFusedMOE(unittest.TestCase):
    NUM_EXPERTS = [8, 64]
    TOP_KS = [2, 6]

    def torch_naive_moe(self, a, w1, w2, score, topk):
        B, D = a.shape
        a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
        out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
        score = torch.softmax(score, dim=-1, dtype=torch.float32)
        topk_weight, topk_ids = torch.topk(score, topk)
        topk_weight = topk_weight.view(-1)
        topk_ids = topk_ids.view(-1)
        for i in range(w1.shape[0]):
            mask = topk_ids == i
            if mask.sum():
                out[mask] = SiluAndMul()(a[mask] @ w1[i].transpose(0, 1)) @ w2[
                    i
                ].transpose(0, 1)
        return (
            out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
        ).sum(dim=1)

    def _test_case(self, m, n, k, e, topk, dtype, use_fp8_w8a8=False):
        if use_fp8_w8a8:
            # AssertionError: fp8e4nv data type is not supported on CUDA arch < 89
            capability = torch.cuda.get_device_capability()
            if not (capability[0] >= 9 or capability == (8, 9)):
                return

            a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
            w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
            w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
            w1 = w1.to(torch.float8_e4m3fn)
            w2 = w2.to(torch.float8_e4m3fn)
            score = torch.randn((m, e), device="cuda", dtype=dtype)

            w1_scale = torch.randn(e, dtype=torch.float32, device="cuda")
            w2_scale = torch.randn(e, dtype=torch.float32, device="cuda")
            a1_scale = torch.randn(1, dtype=torch.float32, device="cuda")
            a2_scale = torch.randn(1, dtype=torch.float32, device="cuda")

            sglang_output = fused_moe(
                a,
                w1,
                w2,
                score,
                topk,
                renormalize=False,
                use_fp8_w8a8=True,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
            )

            trt_output = trt_fused_moe(
                a,
                w1,
                w2,
                score,
                topk,
                renormalize=False,
                use_fp8_w8a8=True,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
            )

            torch.testing.assert_close(sglang_output, trt_output, atol=2e-2, rtol=0)

        else:
            a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
            w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
            w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
            score = torch.randn((m, e), device="cuda", dtype=dtype)

            triton_output = fused_moe(a, w1, w2, score, topk, renormalize=False)
            torch_output = self.torch_naive_moe(a, w1, w2, score, topk)
            torch.testing.assert_close(triton_output, torch_output, atol=2e-2, rtol=0)

    def test_various_configurations(self):
        m_values = [1, 33, 64, 222, 1024 * 128]
        n_values = [128, 1024, 2048]
        k_values = [128, 511, 1024]
        dtypes = [torch.float16, torch.bfloat16]
        fp8_modes = [False, True]

        for m in m_values:
            for n in n_values:
                for k in k_values:
                    for e in self.NUM_EXPERTS:
                        for topk in self.TOP_KS:
                            for dtype in dtypes:
                                for use_fp8_w8a8 in fp8_modes:
                                    with self.subTest(
                                        m=m,
                                        n=n,
                                        k=k,
                                        e=e,
                                        topk=topk,
                                        dtype=dtype,
                                        fp8=use_fp8_w8a8,
                                    ):
                                        self._test_case(
                                            m,
                                            n,
                                            k,
                                            e,
                                            topk,
                                            dtype,
                                            use_fp8_w8a8=use_fp8_w8a8,
                                        )


if __name__ == "__main__":
    unittest.main()
