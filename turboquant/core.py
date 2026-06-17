"""Torch implementation of TurboQuant core algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

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


def hadamard_orthogonal(dimension: int, *, device: torch.device, dtype: torch.dtype, seed: int | None) -> torch.Tensor:
    if dimension < 1 or dimension & (dimension - 1):
        raise ValueError("Hadamard rotation requires a power-of-two dimension")
    mat = torch.ones((1, 1), device=device, dtype=torch.float32)
    while mat.shape[0] < dimension:
        mat = torch.cat(
            [
                torch.cat([mat, mat], dim=1),
                torch.cat([mat, -mat], dim=1),
            ],
            dim=0,
        )
    mat = mat / (dimension**0.5)
    if seed is not None:
        gen = _generator(device, seed)
        signs = torch.randint(0, 2, (dimension,), generator=gen, device=device, dtype=torch.int8)
        signs = signs.to(dtype=torch.float32).mul_(2).sub_(1)
        mat = mat * signs
    return mat.to(dtype=dtype)


def _prepare_transform_pair(
    dimension: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int | None,
    transform_matrix: torch.Tensor | None,
    inverse_transform_matrix: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if transform_matrix is None and inverse_transform_matrix is None:
        rotation = random_orthogonal(dimension, device=device, dtype=dtype, seed=seed)
        return rotation, rotation.T
    if transform_matrix is None or inverse_transform_matrix is None:
        raise ValueError("transform_matrix and inverse_transform_matrix must be provided together")
    if tuple(transform_matrix.shape) != (dimension, dimension):
        raise ValueError(f"transform_matrix must have shape {(dimension, dimension)}, got {tuple(transform_matrix.shape)}")
    if tuple(inverse_transform_matrix.shape) != (dimension, dimension):
        raise ValueError(
            f"inverse_transform_matrix must have shape {(dimension, dimension)}, got {tuple(inverse_transform_matrix.shape)}"
        )
    return (
        transform_matrix.to(device=device, dtype=dtype).detach(),
        inverse_transform_matrix.to(device=device, dtype=dtype).detach(),
    )


@dataclass
class MSEQuantized:
    indices: torch.Tensor
    norms: torch.Tensor


@dataclass
class BlockMSEQuantized:
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
        transform_matrix: torch.Tensor | None = None,
        inverse_transform_matrix: torch.Tensor | None = None,
        project_unit_norm: bool = False,
        reconstruction_scale: str = "norm",
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
        if project_unit_norm:
            reconstruction_scale = "unit_norm"
        if reconstruction_scale not in {
            "norm",
            "unit_norm",
            "lsq",
            "norm_gain",
            "half_gain",
            "clipped_gain",
            "selected_gain",
            "dot_gain",
        }:
            raise ValueError(f"unsupported reconstruction_scale: {reconstruction_scale}")
        self.project_unit_norm = reconstruction_scale == "unit_norm"
        self.reconstruction_scale = reconstruction_scale
        self.transform, self.inverse_transform = _prepare_transform_pair(
            dimension,
            device=self.device,
            dtype=dtype,
            seed=seed,
            transform_matrix=transform_matrix,
            inverse_transform_matrix=inverse_transform_matrix,
        )
        self.rotation = self.transform
        scalar = lloyd_max_codebook(dimension, bits, grid_size=codebook_grid_size)
        self.scalar_codebook = scalar
        self.centroids = torch.tensor(scalar.centroids, device=self.device, dtype=dtype)
        quantized_norm_sq = max(1.0 - dimension * float(scalar.cost), torch.finfo(torch.float32).eps)
        full_norm_gain = quantized_norm_sq ** -0.5
        if reconstruction_scale == "norm_gain":
            self.reconstruction_gain = full_norm_gain
        elif reconstruction_scale == "half_gain":
            self.reconstruction_gain = 1.0 + 0.5 * (full_norm_gain - 1.0)
        elif reconstruction_scale == "dot_gain":
            self.reconstruction_gain = quantized_norm_sq**-1.0
        else:
            self.reconstruction_gain = 1.0
        self.clipped_gain_max = quantized_norm_sq ** -0.5 if reconstruction_scale == "clipped_gain" else 1.0
        self.selected_gain = full_norm_gain if reconstruction_scale == "selected_gain" else 1.0

    def quantize(self, x: torch.Tensor) -> MSEQuantized:
        x = x.to(device=self.device, dtype=self.dtype)
        if x.shape[-1] != self.dimension:
            raise ValueError(f"expected last dimension {self.dimension}, got {x.shape[-1]}")
        norms = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(torch.finfo(self.dtype).eps)
        unit = x / norms
        rotated = unit @ self.transform.T
        distances = torch.abs(rotated.unsqueeze(-1) - self.centroids)
        indices = torch.argmin(distances, dim=-1).to(torch.int16)
        scales = norms.squeeze(-1)
        if self.reconstruction_scale in {"lsq", "clipped_gain", "selected_gain"}:
            y_hat = self.centroids[indices.to(dtype=torch.long)]
            x_hat_unit = y_hat @ self.inverse_transform.T
            x_f = x.to(dtype=torch.float32)
            x_hat_f = x_hat_unit.to(dtype=torch.float32)
            numerator = (x_f * x_hat_f).sum(dim=-1)
            denominator = x_hat_f.square().sum(dim=-1).clamp_min(torch.finfo(torch.float32).eps)
            lsq_scales = numerator / denominator
            if self.reconstruction_scale == "clipped_gain":
                gain = (lsq_scales / norms.squeeze(-1).to(dtype=torch.float32)).clamp(
                    min=1.0,
                    max=float(self.clipped_gain_max),
                )
                scales = (norms.squeeze(-1).to(dtype=torch.float32) * gain).to(dtype=self.dtype)
            elif self.reconstruction_scale == "selected_gain":
                norm_scales = norms.squeeze(-1).to(dtype=torch.float32)
                base_hat = x_hat_f * norm_scales.unsqueeze(-1)
                gain_hat = x_hat_f * (norm_scales * float(self.selected_gain)).unsqueeze(-1)
                base_error = (x_f - base_hat).square().sum(dim=-1)
                gain_error = (x_f - gain_hat).square().sum(dim=-1)
                selected = torch.where(
                    gain_error <= base_error,
                    norm_scales * float(self.selected_gain),
                    norm_scales,
                )
                scales = selected.to(dtype=self.dtype)
            else:
                scales = lsq_scales.to(dtype=self.dtype)
        return MSEQuantized(indices=indices, norms=scales)

    def dequantize(self, q: MSEQuantized) -> torch.Tensor:
        indices = q.indices.to(device=self.device, dtype=torch.long)
        y_hat = self.centroids[indices]
        x_hat_unit = y_hat @ self.inverse_transform.T
        if self.project_unit_norm:
            x_hat_unit = x_hat_unit / torch.linalg.vector_norm(x_hat_unit, dim=-1, keepdim=True).clamp_min(
                torch.finfo(self.dtype).eps
            )
        elif self.reconstruction_gain != 1.0:
            x_hat_unit = x_hat_unit * self.reconstruction_gain
        norms = q.norms.to(device=self.device, dtype=self.dtype).unsqueeze(-1)
        return x_hat_unit * norms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.dequantize(self.quantize(x))


class TurboQuantBlockMSE:
    """Block vector quantizer in TurboQuant's rotated domain.

    The block codebook is a Cartesian product of the scalar Lloyd-Max centroids
    followed by an L2 nearest-neighbor search inside each block. For block size
    2, this is a 2D lattice-like product code with the same bits-per-dimension
    budget as scalar TurboQuant but joint assignment over coordinate pairs.
    """

    def __init__(
        self,
        dimension: int,
        bits: int,
        *,
        block_size: int = 2,
        seed: int = 0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        codebook_grid_size: int = 200_001,
        transform_matrix: torch.Tensor | None = None,
        inverse_transform_matrix: torch.Tensor | None = None,
    ) -> None:
        if dimension < 2:
            raise ValueError("dimension must be >= 2")
        if dimension % block_size != 0:
            raise ValueError("dimension must be divisible by block_size")
        if bits <= 0 or bits * block_size > 16:
            raise ValueError("block quantization supports 1..16 total bits per block")
        self.dimension = dimension
        self.bits = bits
        self.block_size = block_size
        self.seed = seed
        self.device = torch.device(device)
        self.dtype = dtype
        self.transform, self.inverse_transform = _prepare_transform_pair(
            dimension,
            device=self.device,
            dtype=dtype,
            seed=seed,
            transform_matrix=transform_matrix,
            inverse_transform_matrix=inverse_transform_matrix,
        )
        scalar = lloyd_max_codebook(dimension, bits, grid_size=codebook_grid_size)
        self.scalar_codebook = scalar
        centroids = torch.tensor(scalar.centroids, device=self.device, dtype=dtype)
        mesh = torch.cartesian_prod(*([centroids] * block_size))
        self.block_codebook = mesh.to(device=self.device, dtype=dtype)

    @property
    def block_bits(self) -> int:
        return self.bits * self.block_size

    def quantize(self, x: torch.Tensor) -> BlockMSEQuantized:
        x = x.to(device=self.device, dtype=self.dtype)
        if x.shape[-1] != self.dimension:
            raise ValueError(f"expected last dimension {self.dimension}, got {x.shape[-1]}")
        norms = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(torch.finfo(self.dtype).eps)
        unit = x / norms
        rotated = unit @ self.transform.T
        blocks = rotated.reshape(*rotated.shape[:-1], self.dimension // self.block_size, self.block_size)
        distances = torch.sum((blocks.unsqueeze(-2) - self.block_codebook) ** 2, dim=-1)
        indices = torch.argmin(distances, dim=-1).to(torch.int16)
        return BlockMSEQuantized(indices=indices, norms=norms.squeeze(-1))

    def dequantize(self, q: BlockMSEQuantized) -> torch.Tensor:
        indices = q.indices.to(device=self.device, dtype=torch.long)
        y_hat = self.block_codebook[indices].reshape(*indices.shape[:-1], self.dimension)
        x_hat_unit = y_hat @ self.inverse_transform.T
        norms = q.norms.to(device=self.device, dtype=self.dtype).unsqueeze(-1)
        return x_hat_unit * norms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.dequantize(self.quantize(x))


@lru_cache(maxsize=32)
def _learned_spherical_block_codebook_cpu(
    dimension: int,
    bits: int,
    block_size: int,
    num_samples: int,
    num_iters: int,
    seed: int,
) -> torch.Tensor:
    """Train a low-dimensional Lloyd codebook for sphere-coordinate blocks.

    The scalar TurboQuant codebook is optimal for one coordinate of a random
    point on the unit sphere. This extends that idea to a small vector block by
    fitting k-means centroids to the corresponding block marginal.
    """

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    samples = torch.randn((num_samples, dimension), generator=gen, dtype=torch.float32)
    samples = samples / samples.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).eps)
    blocks = samples[:, :block_size].contiguous()
    levels = 1 << (bits * block_size)
    if blocks.shape[0] < levels:
        raise ValueError("num_samples must be at least the number of block centroids")
    centroids = blocks[torch.randperm(blocks.shape[0], generator=gen)[:levels]].clone()
    for _ in range(num_iters):
        distances = torch.cdist(blocks, centroids).square()
        assignments = torch.argmin(distances, dim=-1)
        new_centroids = torch.zeros_like(centroids)
        counts = torch.bincount(assignments, minlength=levels).to(dtype=torch.float32)
        new_centroids.index_add_(0, assignments, blocks)
        nonempty = counts > 0
        new_centroids[nonempty] = new_centroids[nonempty] / counts[nonempty].unsqueeze(-1)
        if bool((~nonempty).any()):
            replacement = torch.randperm(blocks.shape[0], generator=gen)[: int((~nonempty).sum().item())]
            new_centroids[~nonempty] = blocks[replacement]
        if float((new_centroids - centroids).abs().max().item()) < 1e-7:
            centroids = new_centroids
            break
        centroids = new_centroids
    return centroids.contiguous()


class TurboQuantLearnedBlockMSE:
    """Learned block-vector TurboQuant in the rotated domain.

    Unlike `TurboQuantBlockMSE`, this is not the Cartesian product of scalar
    centroids. The block codebook is trained directly on the block marginal of
    a uniform sphere source, giving a genuine vector quantizer at the same
    bits-per-dimension budget.
    """

    def __init__(
        self,
        dimension: int,
        bits: int,
        *,
        block_size: int = 2,
        seed: int = 0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        codebook_samples: int = 100_000,
        codebook_iters: int = 60,
        codebook_seed: int = 91_337,
        transform_matrix: torch.Tensor | None = None,
        inverse_transform_matrix: torch.Tensor | None = None,
        project_unit_norm: bool = False,
    ) -> None:
        if dimension < 2:
            raise ValueError("dimension must be >= 2")
        if dimension % block_size != 0:
            raise ValueError("dimension must be divisible by block_size")
        if bits <= 0 or bits * block_size > 8:
            raise ValueError("learned block quantization supports 1..8 total bits per block")
        self.dimension = dimension
        self.bits = bits
        self.block_size = block_size
        self.seed = seed
        self.device = torch.device(device)
        self.dtype = dtype
        self.project_unit_norm = project_unit_norm
        self.transform, self.inverse_transform = _prepare_transform_pair(
            dimension,
            device=self.device,
            dtype=dtype,
            seed=seed,
            transform_matrix=transform_matrix,
            inverse_transform_matrix=inverse_transform_matrix,
        )
        self.rotation = self.transform
        codebook = _learned_spherical_block_codebook_cpu(
            dimension,
            bits,
            block_size,
            codebook_samples,
            codebook_iters,
            codebook_seed + 10_000 * dimension + 100 * bits + block_size,
        )
        self.block_codebook = codebook.to(device=self.device, dtype=dtype)

    @property
    def block_bits(self) -> int:
        return self.bits * self.block_size

    def _nearest_indices(self, blocks: torch.Tensor, *, chunk_blocks: int = 65_536) -> torch.Tensor:
        flat = blocks.reshape(-1, self.block_size)
        output = torch.empty(flat.shape[0], device=flat.device, dtype=torch.int16)
        codebook = self.block_codebook
        for start in range(0, flat.shape[0], chunk_blocks):
            end = min(start + chunk_blocks, flat.shape[0])
            chunk = flat[start:end].to(dtype=torch.float32)
            distances = (chunk.unsqueeze(1) - codebook.to(dtype=torch.float32).unsqueeze(0)).square().sum(dim=-1)
            output[start:end] = torch.argmin(distances, dim=-1).to(dtype=torch.int16)
        return output.reshape(blocks.shape[:-1])

    def quantize(self, x: torch.Tensor) -> BlockMSEQuantized:
        x = x.to(device=self.device, dtype=self.dtype)
        if x.shape[-1] != self.dimension:
            raise ValueError(f"expected last dimension {self.dimension}, got {x.shape[-1]}")
        norms = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(torch.finfo(self.dtype).eps)
        unit = x / norms
        rotated = unit @ self.transform.T
        blocks = rotated.reshape(*rotated.shape[:-1], self.dimension // self.block_size, self.block_size)
        indices = self._nearest_indices(blocks)
        return BlockMSEQuantized(indices=indices, norms=norms.squeeze(-1))

    def dequantize(self, q: BlockMSEQuantized) -> torch.Tensor:
        indices = q.indices.to(device=self.device, dtype=torch.long)
        y_hat = self.block_codebook[indices].reshape(*indices.shape[:-1], self.dimension)
        x_hat_unit = y_hat @ self.inverse_transform.T
        if self.project_unit_norm:
            x_hat_unit = x_hat_unit / torch.linalg.vector_norm(x_hat_unit, dim=-1, keepdim=True).clamp_min(
                torch.finfo(self.dtype).eps
            )
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
        transform_matrix: torch.Tensor | None = None,
        inverse_transform_matrix: torch.Tensor | None = None,
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
            transform_matrix=transform_matrix,
            inverse_transform_matrix=inverse_transform_matrix,
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
