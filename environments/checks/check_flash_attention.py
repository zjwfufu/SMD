"""Direct FlashAttention CUDA checks; no pretrained weights or SDPA fallback."""
from importlib import metadata
from pathlib import Path
import sys


def check_flash_attention(cuda=False):
    import torch
    import flash_attn
    import flash_attn_2_cuda

    assert metadata.version('flash-attn') == '2.7.4.post1'
    extension = Path(flash_attn_2_cuda.__file__).resolve()
    assert extension.is_relative_to(Path(sys.prefix).resolve()), extension
    result = dict(version=metadata.version('flash-attn'), extension=str(extension),
                  torch=torch.__version__, cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI)
    if not cuda:
        return result

    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from unittest.mock import patch
    from smd.networks.Wan import network_causal

    assert torch.cuda.is_available()
    assert network_causal._OFFICIAL_FLASH_ATTN_AVAILABLE
    cases = []
    wrappers = []
    for device in range(torch.cuda.device_count()):
        with torch.cuda.device(device):
            gen = torch.Generator(device=device).manual_seed(42)
            for dtype in (torch.float16, torch.bfloat16):
                # Tolerances fixed before running; reference computes attention in FP32.
                atol, rtol = (3e-4, 3e-3) if dtype == torch.float16 else (2e-3, 2e-2)
                for causal in (False, True):
                    for varlen in (False, True):
                        shape = (36, 2, 64) if varlen else (2, 23, 2, 64)
                        live = [(torch.randn(shape, device=device, dtype=dtype, generator=gen) * .3)
                                .requires_grad_() for _ in range(3)]
                        reference = [x.detach().float().requires_grad_() for x in live]
                        if varlen:
                            cu = torch.tensor([0, 13, 36], device=device, dtype=torch.int32)
                            output = flash_attn.flash_attn_varlen_func(
                                *live, cu, cu, 23, 23, dropout_p=0., causal=causal, deterministic=True)
                            pieces = []
                            with sdpa_kernel(SDPBackend.MATH):
                                for start, end in [(0, 13), (13, 36)]:
                                    q, k, v = [x[start:end].transpose(0, 1).unsqueeze(0) for x in reference]
                                    pieces.append(F.scaled_dot_product_attention(q, k, v, is_causal=causal)
                                                  .squeeze(0).transpose(0, 1))
                            expected = torch.cat(pieces)
                        else:
                            output = flash_attn.flash_attn_func(
                                *live, dropout_p=0., causal=causal, deterministic=True)
                            with sdpa_kernel(SDPBackend.MATH):
                                expected = F.scaled_dot_product_attention(
                                    *(x.transpose(1, 2) for x in reference), is_causal=causal).transpose(1, 2)
                        upstream = torch.randn(output.shape, device=device, dtype=dtype, generator=gen) * .25
                        output.backward(upstream)
                        expected.backward(upstream.float())
                        torch.testing.assert_close(output.float(), expected, atol=atol, rtol=rtol)
                        gradient_error = 0.
                        for actual, ref in zip(live, reference):
                            assert torch.isfinite(actual.grad).all()
                            torch.testing.assert_close(actual.grad.float(), ref.grad, atol=atol, rtol=rtol)
                            gradient_error = max(gradient_error, float((actual.grad.float() - ref.grad).abs().max()))
                        cases.append(dict(device=device, dtype=str(dtype), causal=causal, varlen=varlen,
                                          output_max_abs=float((output.float() - expected).abs().max()),
                                          gradient_max_abs=gradient_error, atol=atol, rtol=rtol))
                # Exercise SMD's real Wan helper, including differing query/key lengths.
                q = torch.randn(2, 17, 2, 64, device=device, dtype=dtype, generator=gen).requires_grad_()
                k = torch.randn(2, 23, 2, 64, device=device, dtype=dtype, generator=gen).requires_grad_()
                v = torch.randn(2, 23, 2, 64, device=device, dtype=dtype, generator=gen).requires_grad_()
                with patch.object(flash_attn, 'flash_attn_varlen_func', wraps=flash_attn.flash_attn_varlen_func) as call:
                    output = network_causal._official_flash_attention(q, k, v)
                    output.float().square().mean().backward()
                    assert call.call_count == 1, 'SMD did not invoke FlashAttention'
                assert all(torch.isfinite(x.grad).all() for x in (q, k, v))
                wrappers.append(dict(device=device, dtype=str(dtype), varlen_calls=call.call_count))
            torch.cuda.synchronize()
    result.update(kernel_cases=cases, smd_wan_wrappers=wrappers,
                  scope='Direct fixed/variable-length FlashAttention forward/backward vs FP32 math; SMD Wan helper dispatch')
    return result
