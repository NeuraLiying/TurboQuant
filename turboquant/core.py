"""Torch implementation of TurboQuant core algorithms."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .codebook import lloyd_max_codebook


def _generator(device: torch.device, seed: int | None) -> torch.Generator:
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    return gen


def random_orthogonal(dimension: int, *, device: torch.device, dtype: torch.dtype, seed: int | None) -> torch.Tensor:
    gen = _generator(device, seed)
    mat = torch.randn((dimension, dimension), generator=gen, device=device, dtype=torch.float64)
    q, r = torch.linalg.qr(mat)
    signs = torch.sign(torch.diagonal(r))
    signs[signs == 0] = 1
    q = q * signs
    return q.to(dtype=dtype)


@dataclass
class MSEQuantized:
    indices: torch.Tensor
    norms: torch.Tensor


@dataclass
class ProdQuantized:
    mse: MSEQuantized
    qjl: torch.Tensor
    residual_norms: torch.Tensor


class TurboQuantMSE:
    """MSE-optimized TurboQuant.

    Input tensors are shaped `(..., d)`. Non-unit vectors are normalized before
    quantization and their norms are restored during dequantization.
    """

    def __init__(
        self,
        dimension: int,
        bits: int,
        *,
        seed: int = 0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        codebook_grid_size: int = 200_001,
    ) -> None:
        if dimension < 2:
            raise ValueError("dimension must be >= 2")
        if bits < 0:
            raise ValueError("bits must be non-negative")
        self.dimension = dimension
        self.bits = bits
        self.seed = seed
        self.device = torch.device(device)
        self.dtype = dtype
        self.rotation = random_orthogonal(dimension, device=self.device, dtype=dtype, seed=seed)
        scalar = lloyd_max_codebook(dimension, bits, grid_size=codebook_grid_size)
        self.scalar_codebook = scalar
        self.centroids = torch.tensor(scalar.centroids, device=self.device, dtype=dtype)

    def quantize(self, x: torch.Tensor) -> MSEQuantized:
        x = x.to(device=self.device, dtype=self.dtype)
        if x.shape[-1] != self.dimension:
            raise ValueError(f"expected last dimension {self.dimension}, got {x.shape[-1]}")
        norms = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(torch.finfo(self.dtype).eps)
        unit = x / norms
        rotated = unit @ self.rotation.T
        distances = torch.abs(rotated.unsqueeze(-1) - self.centroids)
        indices = torch.argmin(distances, dim=-1).to(torch.int16)
        return MSEQuantized(indices=indices, norms=norms.squeeze(-1))

    def dequantize(self, q: MSEQuantized) -> torch.Tensor:
        indices = q.indices.to(device=self.device, dtype=torch.long)
        y_hat = self.centroids[indices]
        x_hat_unit = y_hat @ self.rotation
        norms = q.norms.to(device=self.device, dtype=self.dtype).unsqueeze(-1)
        return x_hat_unit * norms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.dequantize(self.quantize(x))


class TurboQuantProd:
    """Inner-product optimized TurboQuant using QJL on the MSE residual."""

    def __init__(
        self,
        dimension: int,
        bits: int,
        *,
        seed: int = 0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        codebook_grid_size: int = 200_001,
    ) -> None:
        if bits < 1:
            raise ValueError("TurboQuantProd requires bits >= 1")
        self.dimension = dimension
        self.bits = bits
        self.seed = seed
        self.device = torch.device(device)
        self.dtype = dtype
        self.mse_quantizer = TurboQuantMSE(
            dimension,
            bits - 1,
            seed=seed,
            device=self.device,
            dtype=dtype,
            codebook_grid_size=codebook_grid_size,
        )
        gen = _generator(self.device, seed + 10_000)
        self.qjl_matrix = torch.randn(
            (dimension, dimension),
            generator=gen,
            device=self.device,
            dtype=dtype,
        )

    def quantize(self, x: torch.Tensor) -> ProdQuantized:
        x = x.to(device=self.device, dtype=self.dtype)
        mse_q = self.mse_quantizer.quantize(x)
        mse_hat = self.mse_quantizer.dequantize(mse_q)
        residual = x - mse_hat
        residual_norms = torch.linalg.vector_norm(residual, dim=-1)
        safe = residual_norms.clamp_min(torch.finfo(self.dtype).eps).unsqueeze(-1)
        residual_unit = residual / safe
        signs = torch.sign(residual_unit @ self.qjl_matrix.T)
        signs[signs == 0] = 1
        return ProdQuantized(mse=mse_q, qjl=signs.to(torch.int8), residual_norms=residual_norms)

    def dequantize(self, q: ProdQuantized) -> torch.Tensor:
        mse_hat = self.mse_quantizer.dequantize(q.mse)
        signs = q.qjl.to(device=self.device, dtype=self.dtype)
        gamma = q.residual_norms.to(device=self.device, dtype=self.dtype).unsqueeze(-1)
        scale = (torch.pi / 2.0) ** 0.5 / self.dimension
        residual_hat = scale * gamma * (signs @ self.qjl_matrix)
        return mse_hat + residual_hat

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.dequantize(self.quantize(x))
