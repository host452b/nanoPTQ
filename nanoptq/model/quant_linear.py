# nanoptq/model/quant_linear.py
"""
QuantLinear: the single abstraction that unifies all quantization methods.
量化线性层：统一所有量化方法的唯一抽象。

Design principles:
  1. Stores weights as int8 (int4 values fit in int8 containers) + float16 scales
  2. Dequantizes on the fly during forward() — no custom CUDA needed
  3. All algorithms (RTN, AWQ, GPTQ) produce a QuantLinear by filling weight_q + scales
  4. Fully compatible with model.generate() — just swap nn.Linear → QuantLinear

Trade-off: dequant on the fly costs runtime FLOP vs memory saved.
This is the "educational" trade-off — production systems fuse dequant into the kernel.
"""
import torch
import torch.nn as nn
from nanoptq.core.group_quant import group_dequantize


class QuantLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with quantized weights.
    Weights stored quantized; dequantized to fp16/fp32 at forward time.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int,
        group_size: int,
        bias: bool = True,
        symmetric: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.symmetric = symmetric

        num_groups = in_features // group_size

        self.register_buffer("weight_q", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scales", torch.ones(out_features, num_groups, dtype=torch.float16))
        if not symmetric:
            self.register_buffer("zero_points", torch.zeros(out_features, num_groups, dtype=torch.uint8))
        else:
            self.zero_points = None

        if bias:
            self.register_buffer("bias", torch.zeros(out_features))
        else:
            self.bias = None

    def dequantize(self) -> torch.Tensor:
        """Reconstruct fp32 weight matrix from quantized storage."""
        zp = self.zero_points if not self.symmetric else None
        return group_dequantize(
            self.weight_q, self.scales.float(), self.group_size, self.symmetric, zp
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # AWQ-lite: if input_channel_scales is set, the stored weights are W*s,
        # so we must pre-scale x by 1/s to recover y = (W*s)*(x/s) = W*x.
        if hasattr(self, "input_channel_scales") and self.input_channel_scales is not None:
            x = x / self.input_channel_scales.to(x.dtype)
        W = self.dequantize().to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return nn.functional.linear(x, W, bias)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        bits: int,
        group_size: int,
        symmetric: bool = True,
    ) -> "QuantLinear":
        """
        Create a QuantLinear shell from an existing nn.Linear.
        The weights are NOT quantized yet — algorithms fill weight_q and scales.
        """
        has_bias = linear.bias is not None
        ql = cls(linear.in_features, linear.out_features, bits, group_size, has_bias, symmetric)
        if has_bias:
            ql.bias = linear.bias.detach().clone()
        return ql

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"bits={self.bits}, group_size={self.group_size}, symmetric={self.symmetric}"
        )
