import torch

from turboquant import TurboQuantDynamicCache, TurboQuantMSE, TurboQuantProd
from turboquant.core import TurboQuantLearnedBlockMSE, hadamard_orthogonal
from turboquant.kv_cache import (
    CenteredSegment,
    HeadAdaptiveOutlierMSESegment,
    HadamardResidualSegment,
    HeadHadamardSegment,
    HeadRotationMSESegment,
    OutlierHadamardSegment,
    OutlierMSESegment,
    PackedMSESegment,
    PackedProdSegment,
    RotatedOutlierMSESegment,
    RmsScaledSegment,
    UniformAffineSegment,
    TokenProtectedSegment,
    make_kv_config_from_effective_bits,
    VectorAdaptiveRotationSegment,
    pack_indices,
    unpack_indices,
)


def test_mse_roundtrip_shapes_and_monotonic_error():
    torch.manual_seed(0)
    x = torch.randn(64, 32)
    x = x / x.norm(dim=-1, keepdim=True)

    errors = []
    for bits in [1, 2, 3]:
        quantizer = TurboQuantMSE(32, bits, seed=123, codebook_grid_size=20_001)
        q = quantizer.quantize(x)
        x_hat = quantizer.dequantize(q)
        assert q.indices.shape == x.shape
        assert x_hat.shape == x.shape
        errors.append(torch.mean(torch.sum((x - x_hat) ** 2, dim=-1)).item())

    assert errors[2] < errors[1] < errors[0]


def test_mse_accepts_custom_transform_pair():
    torch.manual_seed(10)
    x = torch.randn(16, 8)
    transform = torch.eye(8)
    inverse = torch.eye(8)
    quantizer = TurboQuantMSE(
        8,
        3,
        seed=123,
        codebook_grid_size=10_001,
        transform_matrix=transform,
        inverse_transform_matrix=inverse,
    )
    q = quantizer.quantize(x)
    x_hat = quantizer.dequantize(q)

    assert q.indices.shape == x.shape
    assert x_hat.shape == x.shape
    assert torch.allclose(quantizer.transform.cpu(), transform)
    assert torch.allclose(quantizer.inverse_transform.cpu(), inverse)


def test_unit_mse_restores_original_norms():
    torch.manual_seed(13)
    x = torch.randn(16, 8) * 3.0
    quantizer = TurboQuantMSE(
        8,
        2,
        seed=123,
        codebook_grid_size=10_001,
        project_unit_norm=True,
    )
    x_hat = quantizer(x)

    assert torch.allclose(x_hat.norm(dim=-1), x.norm(dim=-1), rtol=1e-4, atol=1e-4)


def test_lsq_mse_reconstruction_scale_reduces_error():
    torch.manual_seed(15)
    x = torch.randn(128, 16) * 2.0
    base = TurboQuantMSE(16, 2, seed=123, codebook_grid_size=10_001)
    lsq = TurboQuantMSE(
        16,
        2,
        seed=123,
        codebook_grid_size=10_001,
        reconstruction_scale="lsq",
    )

    base_err = torch.mean(torch.sum((x - base(x)) ** 2, dim=-1))
    lsq_err = torch.mean(torch.sum((x - lsq(x)) ** 2, dim=-1))
    assert lsq_err <= base_err


def test_gain_mse_applies_global_reconstruction_gain():
    torch.manual_seed(16)
    x = torch.randn(32, 16)
    base = TurboQuantMSE(16, 2, seed=123, codebook_grid_size=10_001)
    gain = TurboQuantMSE(
        16,
        2,
        seed=123,
        codebook_grid_size=10_001,
        reconstruction_scale="norm_gain",
    )
    q_base = base.quantize(x)
    q_gain = gain.quantize(x)
    x_base = base.dequantize(q_base)
    x_gain = gain.dequantize(q_gain)

    assert torch.equal(q_base.indices, q_gain.indices)
    assert torch.allclose(q_base.norms, q_gain.norms)
    assert gain.reconstruction_gain > 1.0
    assert torch.mean(x_gain.norm(dim=-1)) > torch.mean(x_base.norm(dim=-1))


def test_clipped_gain_mse_bounds_per_vector_gain():
    torch.manual_seed(18)
    x = torch.randn(32, 16)
    base = TurboQuantMSE(16, 2, seed=123, codebook_grid_size=10_001)
    clipped = TurboQuantMSE(
        16,
        2,
        seed=123,
        codebook_grid_size=10_001,
        reconstruction_scale="clipped_gain",
    )
    q_base = base.quantize(x)
    q_clipped = clipped.quantize(x)
    gains = q_clipped.norms / q_base.norms.clamp_min(torch.finfo(q_base.norms.dtype).eps)

    assert torch.equal(q_base.indices, q_clipped.indices)
    assert torch.all(gains >= 1.0)
    assert torch.all(gains <= clipped.clipped_gain_max + 1e-3)
    assert clipped.reconstruction_gain == 1.0
    assert clipped.clipped_gain_max > 1.0


def test_half_gain_mse_is_between_base_and_full_gain():
    torch.manual_seed(20)
    x = torch.randn(32, 16)
    base = TurboQuantMSE(16, 2, seed=123, codebook_grid_size=10_001)
    half = TurboQuantMSE(
        16,
        2,
        seed=123,
        codebook_grid_size=10_001,
        reconstruction_scale="half_gain",
    )
    full = TurboQuantMSE(
        16,
        2,
        seed=123,
        codebook_grid_size=10_001,
        reconstruction_scale="norm_gain",
    )

    assert torch.equal(base.quantize(x).indices, half.quantize(x).indices)
    assert 1.0 < half.reconstruction_gain < full.reconstruction_gain


def test_selected_gain_mse_selects_per_vector_gain_without_changing_indices():
    torch.manual_seed(21)
    x = torch.randn(64, 16)
    base = TurboQuantMSE(16, 2, seed=123, codebook_grid_size=10_001)
    selected = TurboQuantMSE(
        16,
        2,
        seed=123,
        codebook_grid_size=10_001,
        reconstruction_scale="selected_gain",
    )
    q_base = base.quantize(x)
    q_selected = selected.quantize(x)
    gains = q_selected.norms / q_base.norms.clamp_min(torch.finfo(q_base.norms.dtype).eps)

    assert torch.equal(q_base.indices, q_selected.indices)
    assert torch.all((torch.isclose(gains, torch.ones_like(gains))) | (torch.isclose(gains, gains.new_full(gains.shape, selected.selected_gain))))


def test_hadamard_orthogonal_is_orthonormal():
    transform = hadamard_orthogonal(16, device=torch.device("cpu"), dtype=torch.float32, seed=11)

    assert torch.allclose(transform @ transform.T, torch.eye(16), atol=1e-6)


def test_block_mse_roundtrip_shapes():
    torch.manual_seed(11)
    x = torch.randn(8, 16)
    from turboquant.core import TurboQuantBlockMSE

    quantizer = TurboQuantBlockMSE(16, 2, block_size=2, seed=123, codebook_grid_size=10_001)
    q = quantizer.quantize(x)
    x_hat = quantizer.dequantize(q)

    assert q.indices.shape == (8, 8)
    assert x_hat.shape == x.shape


def test_learned_block_mse_roundtrip_shapes_and_improves_sphere_mse():
    torch.manual_seed(12)
    x = torch.randn(512, 32)
    x = x / x.norm(dim=-1, keepdim=True)

    scalar = TurboQuantMSE(32, 2, seed=123, codebook_grid_size=10_001)
    learned = TurboQuantLearnedBlockMSE(
        32,
        2,
        block_size=2,
        seed=123,
        codebook_samples=8_192,
        codebook_iters=25,
    )
    q = learned.quantize(x)
    x_hat = learned.dequantize(q)

    scalar_err = torch.mean(torch.sum((x - scalar(x)) ** 2, dim=-1))
    learned_err = torch.mean(torch.sum((x - x_hat) ** 2, dim=-1))
    assert q.indices.shape == (512, 16)
    assert x_hat.shape == x.shape
    assert learned_err < scalar_err


def test_learned_unit_block_mse_restores_original_norms():
    torch.manual_seed(14)
    x = torch.randn(16, 16) * 2.0
    quantizer = TurboQuantLearnedBlockMSE(
        16,
        2,
        block_size=2,
        seed=123,
        codebook_samples=8_192,
        codebook_iters=25,
        project_unit_norm=True,
    )
    x_hat = quantizer(x)

    assert torch.allclose(x_hat.norm(dim=-1), x.norm(dim=-1), rtol=1e-4, atol=1e-4)


def test_prod_shapes():
    torch.manual_seed(1)
    x = torch.randn(8, 32)
    y = torch.randn(8, 32)
    quantizer = TurboQuantProd(32, 2, seed=123, codebook_grid_size=20_001)
    q = quantizer.quantize(x)
    x_hat = quantizer.dequantize(q)
    assert q.mse.indices.shape == x.shape
    assert q.qjl.shape == x.shape
    assert q.residual_norms.shape == (8,)
    assert x_hat.shape == x.shape
    assert torch.isfinite((x_hat * y).sum(dim=-1)).all()


def test_turboquant_dynamic_cache_shapes():
    torch.manual_seed(2)
    cache = TurboQuantDynamicCache()
    key = torch.randn(1, 2, 4, 16)
    value = torch.randn(1, 2, 4, 16)
    out_k, out_v = cache.update(key, value, layer_idx=0)
    assert out_k.shape == key.shape
    assert out_v.shape == value.shape
    next_k = torch.randn(1, 2, 1, 16)
    next_v = torch.randn(1, 2, 1, 16)
    out_k, out_v = cache.update(next_k, next_v, layer_idx=0)
    assert out_k.shape == (1, 2, 5, 16)
    assert out_v.shape == (1, 2, 5, 16)


def test_pack_indices_roundtrip_for_non_byte_aligned_bits():
    indices = torch.arange(35, dtype=torch.int16) % 8
    packed = pack_indices(indices, bits=3)
    restored = unpack_indices(packed, bits=3, shape=tuple(indices.shape))
    assert packed.dtype == torch.uint8
    assert packed.numel() == (indices.numel() * 3 + 7) // 8
    assert torch.equal(restored, indices)


def test_turboquant_dynamic_cache_stores_packed_segments():
    torch.manual_seed(3)
    cache = TurboQuantDynamicCache()
    key = torch.randn(1, 2, 4, 16)
    value = torch.randn(1, 2, 4, 16)

    dense_nbytes = 2 * key.numel() * key.element_size()
    out_k, out_v = cache.update(key, value, layer_idx=0)

    assert out_k.shape == key.shape
    assert out_v.shape == value.shape
    assert cache.get_seq_length() == 4
    assert isinstance(cache.key_cache[0][0], PackedMSESegment)
    assert isinstance(cache.value_cache[0][0], PackedMSESegment)
    assert cache.key_cache[0][0].packed_indices.dtype == torch.uint8
    assert cache.storage_nbytes() < dense_nbytes

    next_k = torch.randn(1, 2, 1, 16)
    next_v = torch.randn(1, 2, 1, 16)
    out_k, out_v = cache.update(next_k, next_v, layer_idx=0)

    assert out_k.shape == (1, 2, 5, 16)
    assert out_v.shape == (1, 2, 5, 16)
    assert cache.get_seq_length() == 5
    assert len(cache.key_cache[0]) == 2
    assert torch.isfinite(out_k).all()
    assert torch.isfinite(out_v).all()


def test_turboquant_dynamic_cache_supports_learned_block_mse():
    cfg = make_kv_config_from_effective_bits(
        2,
        codebook_grid_size=10_001,
        key_quantizer="learned_mse_block2",
        value_quantizer="learned_mse_block2",
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 4, 16)
    value = torch.randn(1, 2, 4, 16)

    out_k, out_v = cache.update(key, value, layer_idx=0)
    key_segment = cache.key_cache[0][0]
    value_segment = cache.value_cache[0][0]

    assert out_k.shape == key.shape
    assert out_v.shape == value.shape
    assert isinstance(key_segment, PackedMSESegment)
    assert isinstance(value_segment, PackedMSESegment)
    assert key_segment.block_size == 2
    assert value_segment.block_size == 2
    assert key_segment.index_shape == (1, 2, 4, 8)


def test_turboquant_dynamic_cache_fractional_bits_use_outlier_segments():
    torch.manual_seed(4)
    cache = TurboQuantDynamicCache(make_kv_config_from_effective_bits(2.5, codebook_grid_size=10_001))
    key = torch.randn(1, 2, 4, 16)
    value = torch.randn(1, 2, 4, 16)

    out_k, out_v = cache.update(key, value, layer_idx=0)
    key_segment = cache.key_cache[0][0]
    value_segment = cache.value_cache[0][0]

    assert out_k.shape == key.shape
    assert out_v.shape == value.shape
    assert isinstance(key_segment, OutlierMSESegment)
    assert isinstance(value_segment, OutlierMSESegment)
    assert key_segment.regular_bits == 2
    assert key_segment.outlier_bits == 3
    assert key_segment.outlier_indices.numel() == 8
    assert key_segment.regular_indices.numel() == 8
    assert key_segment.effective_index_bits == 2.5
    assert cache.storage_nbytes() < cache.materialized_nbytes()
    summary = cache.compression_summary()
    assert summary["segment_types"] == {"OutlierMSESegment": 2}
    assert summary["avg_effective_index_bits"] == 2.5
    assert summary["outlier_counts"] == [8]
    assert summary["regular_bits"] == [2]
    assert summary["outlier_bits"] == [3]


def test_srht_mse_uses_structured_hadamard_rotation():
    torch.manual_seed(23)
    cfg = make_kv_config_from_effective_bits(
        3,
        codebook_grid_size=10_001,
        key_quantizer="srht_mse",
        value_quantizer="srht_mse",
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 5, 16)
    value = torch.randn(1, 2, 5, 16)

    out_key, out_value = cache.update(key, value, layer_idx=0)
    quantizer = cache._get_quantizer(
        quantizer_kind="srht_mse",
        name="key",
        dimension=16,
        bits=3,
        layer_idx=0,
        device=key.device,
        dtype=key.dtype,
    )
    transform = quantizer.transform.to(dtype=torch.float32)

    assert out_key.shape == key.shape
    assert out_value.shape == value.shape
    assert isinstance(cache.key_cache[0][0], PackedMSESegment)
    assert torch.allclose(transform @ transform.T, torch.eye(16), atol=1e-5)
    assert torch.allclose(transform.abs(), torch.full((16, 16), 16**-0.5), atol=1e-6)
    assert cache.compression_summary()["avg_effective_index_bits"] == 3.0


def test_rotated_outlier_mse_uses_rotated_coordinate_allocation():
    torch.manual_seed(24)
    cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        key_quantizer="rotated_outlier_mse",
        value_quantizer="rotated_outlier_mse",
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 6, 16)
    value = torch.randn(1, 2, 6, 16)

    out_key, out_value = cache.update(key, value, layer_idx=0)
    key_segment = cache.key_cache[0][0]
    value_segment = cache.value_cache[0][0]

    assert out_key.shape == key.shape
    assert out_value.shape == value.shape
    assert isinstance(key_segment, RotatedOutlierMSESegment)
    assert isinstance(value_segment, RotatedOutlierMSESegment)
    assert key_segment.regular_bits == 2
    assert key_segment.outlier_bits == 3
    assert key_segment.outlier_indices.numel() == 8
    assert key_segment.regular_indices.numel() == 8
    assert key_segment.effective_index_bits == 2.5
    summary = cache.compression_summary()
    assert summary["segment_types"] == {"RotatedOutlierMSESegment": 2}
    assert summary["avg_effective_index_bits"] == 2.5
    assert summary["outlier_counts"] == [8]
    assert summary["regular_bits"] == [2]
    assert summary["outlier_bits"] == [3]
    assert cache.storage_nbytes() < cache.materialized_nbytes()


def test_shared_rotated_outlier_mse_uses_shared_kv_coordinate_allocation():
    torch.manual_seed(240)
    cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        key_quantizer="shared_rotated_outlier_mse",
        value_quantizer="shared_rotated_outlier_mse",
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 6, 16)
    value = torch.randn(1, 2, 6, 16)

    out_key, out_value = cache.update(key, value, layer_idx=0)
    key_segment = cache.key_cache[0][0]
    value_segment = cache.value_cache[0][0]

    assert out_key.shape == key.shape
    assert out_value.shape == value.shape
    assert isinstance(key_segment, RotatedOutlierMSESegment)
    assert isinstance(value_segment, RotatedOutlierMSESegment)
    assert key_segment.quantizer_kind == "shared_rotated_outlier_mse"
    assert value_segment.quantizer_kind == "shared_rotated_outlier_mse"
    assert torch.equal(key_segment.outlier_indices, value_segment.outlier_indices)
    assert torch.equal(key_segment.regular_indices, value_segment.regular_indices)
    assert key_segment.regular_bits == 2
    assert key_segment.outlier_bits == 3
    assert key_segment.outlier_indices.numel() == 8
    assert key_segment.effective_index_bits == 2.5
    assert cache.compression_summary()["avg_effective_index_bits"] == 2.5


def test_paired_rotated_outlier_mse_shares_rotation_but_keeps_independent_indices():
    torch.manual_seed(243)
    cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        key_quantizer="paired_rotated_outlier_mse",
        value_quantizer="paired_rotated_outlier_mse",
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 6, 16)
    value = torch.randn(1, 2, 6, 16)

    out_key, out_value = cache.update(key, value, layer_idx=0)
    key_segment = cache.key_cache[0][0]
    value_segment = cache.value_cache[0][0]

    assert out_key.shape == key.shape
    assert out_value.shape == value.shape
    assert isinstance(key_segment, RotatedOutlierMSESegment)
    assert isinstance(value_segment, RotatedOutlierMSESegment)
    assert key_segment.quantizer_kind == "paired_rotated_outlier_mse"
    assert value_segment.quantizer_kind == "paired_rotated_outlier_mse"
    assert key_segment.outlier_indices.numel() == 8
    assert value_segment.outlier_indices.numel() == 8
    assert key_segment.effective_index_bits == 2.5
    assert value_segment.effective_index_bits == 2.5
    assert cache.compression_summary()["avg_effective_index_bits"] == 2.5


def test_attention_rotated_outlier_mse_accepts_query_states():
    torch.manual_seed(241)
    cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        key_quantizer="attention_rotated_outlier_mse",
        value_quantizer="attention_rotated_outlier_mse",
        attention_error_query_tokens=2,
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 6, 16)
    value = torch.randn(1, 2, 6, 16)
    query = torch.randn(1, 4, 6, 16)

    out_key, out_value = cache.update(
        key,
        value,
        layer_idx=0,
        cache_kwargs={"query_states": query, "scaling": 16**-0.5},
    )
    key_segment = cache.key_cache[0][0]
    value_segment = cache.value_cache[0][0]

    assert out_key.shape == key.shape
    assert out_value.shape == value.shape
    assert isinstance(key_segment, RotatedOutlierMSESegment)
    assert isinstance(value_segment, RotatedOutlierMSESegment)
    assert key_segment.quantizer_kind == "attention_rotated_outlier_mse"
    assert value_segment.quantizer_kind == "attention_rotated_outlier_mse"
    assert key_segment.regular_bits == 2
    assert key_segment.outlier_bits == 3
    assert key_segment.outlier_indices.numel() == 8
    assert key_segment.effective_index_bits == 2.5
    assert cache.compression_summary()["avg_effective_index_bits"] == 2.5


def test_attention_adaptive_rotated_outlier_mse_keeps_fractional_budget():
    torch.manual_seed(242)
    cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        key_quantizer="attention_adaptive_rotated_outlier_mse",
        value_quantizer="mse",
        attention_error_query_tokens=2,
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 6, 16)
    value = torch.randn(1, 2, 6, 16)
    query = torch.randn(1, 4, 6, 16)

    out_key, out_value = cache.update(
        key,
        value,
        layer_idx=0,
        cache_kwargs={"query_states": query, "scaling": 16**-0.5},
    )

    assert out_key.shape == key.shape
    assert out_value.shape == value.shape
    assert isinstance(cache.key_cache[0][0], (OutlierMSESegment, RotatedOutlierMSESegment))
    assert cache.compression_summary()["avg_effective_index_bits"] == 2.5


def test_attention_adaptive_paired_rotated_outlier_mse_accepts_query_states():
    torch.manual_seed(244)
    cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        key_quantizer="attention_adaptive_paired_rotated_outlier_mse",
        value_quantizer="attention_adaptive_paired_rotated_outlier_mse",
        attention_error_query_tokens=2,
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 6, 16)
    value = torch.randn(1, 2, 6, 16)
    query = torch.randn(1, 4, 6, 16)

    out_key, out_value = cache.update(
        key,
        value,
        layer_idx=0,
        cache_kwargs={"query_states": query, "scaling": 16**-0.5},
    )

    assert out_key.shape == key.shape
    assert out_value.shape == value.shape
    assert isinstance(cache.key_cache[0][0], (OutlierMSESegment, RotatedOutlierMSESegment))
    assert isinstance(cache.value_cache[0][0], (OutlierMSESegment, RotatedOutlierMSESegment))
    assert cache.compression_summary()["avg_effective_index_bits"] == 2.5


def test_entropy_guarded_paired_rotated_outlier_mse_switches_by_attention_entropy():
    torch.manual_seed(245)
    key = torch.randn(1, 2, 6, 16)
    value = torch.randn(1, 2, 6, 16)
    query = torch.randn(1, 4, 6, 16)
    high_cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        key_quantizer="entropy_guarded_paired_rotated_outlier_mse",
        value_quantizer="entropy_guarded_paired_rotated_outlier_mse",
        attention_error_query_tokens=2,
        attention_entropy_threshold=1.0,
    )
    low_cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        key_quantizer="entropy_guarded_paired_rotated_outlier_mse",
        value_quantizer="entropy_guarded_paired_rotated_outlier_mse",
        attention_error_query_tokens=2,
        attention_entropy_threshold=0.0,
    )

    high_cache = TurboQuantDynamicCache(high_cfg)
    low_cache = TurboQuantDynamicCache(low_cfg)
    high_cache.update(key, value, layer_idx=0, cache_kwargs={"query_states": query, "scaling": 16**-0.5})
    low_cache.update(key, value, layer_idx=0, cache_kwargs={"query_states": query, "scaling": 16**-0.5})

    assert isinstance(high_cache.key_cache[0][0], RotatedOutlierMSESegment)
    assert isinstance(high_cache.value_cache[0][0], RotatedOutlierMSESegment)
    assert high_cache.key_cache[0][0].quantizer_kind == "entropy_guarded_paired_rotated_outlier_mse"
    assert isinstance(low_cache.key_cache[0][0], OutlierMSESegment)
    assert isinstance(low_cache.value_cache[0][0], OutlierMSESegment)
    assert high_cache.compression_summary()["avg_effective_index_bits"] == 2.5
    assert low_cache.compression_summary()["avg_effective_index_bits"] == 2.5


def test_rotated_outlier_mse_integer_bits_match_mse_path():
    torch.manual_seed(25)
    key = torch.randn(1, 2, 6, 16)
    value = torch.randn(1, 2, 6, 16)
    baseline = TurboQuantDynamicCache(
        make_kv_config_from_effective_bits(3, codebook_grid_size=10_001, key_quantizer="mse", value_quantizer="mse")
    )
    candidate = TurboQuantDynamicCache(
        make_kv_config_from_effective_bits(
            3,
            codebook_grid_size=10_001,
            key_quantizer="rotated_outlier_mse",
            value_quantizer="rotated_outlier_mse",
        )
    )

    baseline_key, baseline_value = baseline.update(key, value, layer_idx=0)
    candidate_key, candidate_value = candidate.update(key, value, layer_idx=0)

    assert isinstance(candidate.key_cache[0][0], PackedMSESegment)
    assert torch.allclose(candidate_key, baseline_key)
    assert torch.allclose(candidate_value, baseline_value)


def test_fractional_bits_can_use_quarter_high2_allocation():
    cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        effective_bit_allocation="quarter_high2",
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 4, 16)
    value = torch.randn(1, 2, 4, 16)

    cache.update(key, value, layer_idx=0)
    key_segment = cache.key_cache[0][0]

    assert isinstance(key_segment, OutlierMSESegment)
    assert key_segment.regular_bits == 2
    assert key_segment.outlier_bits == 4
    assert key_segment.outlier_indices.numel() == 4
    assert key_segment.regular_indices.numel() == 12
    assert key_segment.effective_index_bits == 2.5


def test_fractional_bits_support_error_gain_outlier_policy():
    torch.manual_seed(9)
    cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        outlier_policy="error_gain",
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 4, 16)
    value = torch.randn(1, 2, 4, 16)

    cache.update(key, value, layer_idx=0)
    key_segment = cache.key_cache[0][0]
    value_segment = cache.value_cache[0][0]

    assert isinstance(key_segment, OutlierMSESegment)
    assert isinstance(value_segment, OutlierMSESegment)
    assert key_segment.regular_bits == 2
    assert key_segment.outlier_bits == 3
    assert key_segment.effective_index_bits == 2.5
    assert key_segment.policy == "error_gain"
    assert cache.compression_summary()["outlier_policy"] == "error_gain"


def test_fractional_bits_support_attention_error_gain_outlier_policy():
    torch.manual_seed(123)
    cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        outlier_policy="attention_error_gain",
        attention_error_query_tokens=2,
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 6, 16)
    value = torch.randn(1, 2, 6, 16)
    query = torch.randn(1, 4, 6, 16)

    key_out, value_out = cache.update(
        key,
        value,
        layer_idx=0,
        cache_kwargs={"query_states": query, "scaling": 16**-0.5},
    )
    key_segment = cache.key_cache[0][0]
    value_segment = cache.value_cache[0][0]

    assert key_out.shape == key.shape
    assert value_out.shape == value.shape
    assert isinstance(key_segment, OutlierMSESegment)
    assert isinstance(value_segment, OutlierMSESegment)
    assert key_segment.regular_bits == 2
    assert key_segment.outlier_bits == 3
    assert key_segment.effective_index_bits == 2.5
    assert key_segment.policy == "attention_error_gain"
    assert cache.compression_summary()["outlier_policy"] == "attention_error_gain"


def test_joint_error_gain_reallocates_shared_kv_outlier_budget():
    torch.manual_seed(13)
    cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        outlier_policy="joint_error_gain",
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 4, 16)
    value = torch.randn(1, 2, 4, 16)

    cache.update(key, value, layer_idx=0)
    key_segment = cache.key_cache[0][0]
    value_segment = cache.value_cache[0][0]

    assert isinstance(key_segment, OutlierMSESegment)
    assert isinstance(value_segment, OutlierMSESegment)
    assert key_segment.policy == "joint_error_gain"
    assert value_segment.policy == "joint_error_gain"
    assert key_segment.outlier_indices.numel() + value_segment.outlier_indices.numel() == 16
    assert key_segment.regular_indices.numel() > 0
    assert value_segment.regular_indices.numel() > 0
    summary = cache.compression_summary()
    assert summary["avg_effective_index_bits"] == 2.5
    assert summary["outlier_policy"] == "joint_error_gain"


def test_joint_attention_error_gain_uses_same_budget_at_3p5_bits():
    torch.manual_seed(17)
    cfg = make_kv_config_from_effective_bits(
        3.5,
        codebook_grid_size=10_001,
        outlier_policy="joint_attention_error_gain",
        attention_error_query_tokens=2,
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 6, 16)
    value = torch.randn(1, 2, 6, 16)
    query = torch.randn(1, 4, 6, 16)

    key_out, value_out = cache.update(
        key,
        value,
        layer_idx=0,
        cache_kwargs={"query_states": query, "scaling": 16**-0.5},
    )
    key_segment = cache.key_cache[0][0]
    value_segment = cache.value_cache[0][0]

    assert key_out.shape == key.shape
    assert value_out.shape == value.shape
    assert isinstance(key_segment, OutlierMSESegment)
    assert isinstance(value_segment, OutlierMSESegment)
    assert key_segment.policy == "joint_attention_error_gain"
    assert value_segment.policy == "joint_attention_error_gain"
    assert key_segment.regular_bits == 3
    assert key_segment.outlier_bits == 4
    assert value_segment.regular_bits == 3
    assert value_segment.outlier_bits == 4
    assert key_segment.outlier_indices.numel() + value_segment.outlier_indices.numel() == 16
    assert cache.compression_summary()["avg_effective_index_bits"] == 3.5


def test_lowbit_gain_only_applies_to_two_bit_subsegments():
    cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        key_quantizer="lowbit_gain_mse",
        value_quantizer="lowbit_gain_mse",
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 4, 16)
    value = torch.randn(1, 2, 4, 16)

    cache.update(key, value, layer_idx=0)
    quantizers = cache._quantizers
    regular = [
        quantizer
        for (kind, name, dimension, dtype, bits, layer_idx, device_type, device_index), quantizer in quantizers.items()
        if kind == "lowbit_gain_mse" and bits == 2
    ]
    outlier = [
        quantizer
        for (kind, name, dimension, dtype, bits, layer_idx, device_type, device_index), quantizer in quantizers.items()
        if kind == "lowbit_gain_mse" and bits == 3
    ]

    assert regular
    assert outlier
    assert all(getattr(quantizer, "reconstruction_gain") > 1.0 for quantizer in regular)
    assert all(getattr(quantizer, "reconstruction_gain") == 1.0 for quantizer in outlier)


def test_lowbit_clipped_gain_only_applies_to_two_bit_subsegments():
    cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        key_quantizer="lowbit_clipped_gain_mse",
        value_quantizer="lowbit_clipped_gain_mse",
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 4, 16)
    value = torch.randn(1, 2, 4, 16)

    cache.update(key, value, layer_idx=0)
    quantizers = cache._quantizers
    regular = [
        quantizer
        for (kind, name, dimension, dtype, bits, layer_idx, device_type, device_index), quantizer in quantizers.items()
        if kind == "lowbit_clipped_gain_mse" and bits == 2
    ]
    outlier = [
        quantizer
        for (kind, name, dimension, dtype, bits, layer_idx, device_type, device_index), quantizer in quantizers.items()
        if kind == "lowbit_clipped_gain_mse" and bits == 3
    ]

    assert regular
    assert outlier
    assert all(getattr(quantizer, "clipped_gain_max") > 1.0 for quantizer in regular)
    assert all(getattr(quantizer, "clipped_gain_max") == 1.0 for quantizer in outlier)


def test_lowbit_gain_variants_only_apply_to_two_bit_subsegments():
    for quantizer_kind, attribute in [
        ("lowbit_half_gain_mse", "reconstruction_gain"),
        ("lowbit_selected_gain_mse", "selected_gain"),
    ]:
        cfg = make_kv_config_from_effective_bits(
            2.5,
            codebook_grid_size=10_001,
            key_quantizer=quantizer_kind,
            value_quantizer=quantizer_kind,
        )
        cache = TurboQuantDynamicCache(cfg)
        key = torch.randn(1, 2, 4, 16)
        value = torch.randn(1, 2, 4, 16)

        cache.update(key, value, layer_idx=0)
        regular = [
            quantizer
            for (kind, name, dimension, dtype, bits, layer_idx, device_type, device_index), quantizer in cache._quantizers.items()
            if kind == quantizer_kind and bits == 2
        ]
        outlier = [
            quantizer
            for (kind, name, dimension, dtype, bits, layer_idx, device_type, device_index), quantizer in cache._quantizers.items()
            if kind == quantizer_kind and bits == 3
        ]

        assert regular
        assert outlier
        assert all(getattr(quantizer, attribute) > 1.0 for quantizer in regular)
        assert all(getattr(quantizer, attribute) == 1.0 for quantizer in outlier)


def test_selected_gain_mse_applies_to_all_effective_bit_subsegments():
    cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        key_quantizer="selected_gain_mse",
        value_quantizer="selected_gain_mse",
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 4, 16)
    value = torch.randn(1, 2, 4, 16)

    cache.update(key, value, layer_idx=0)
    quantizers = cache._quantizers
    regular = [
        quantizer
        for (kind, name, dimension, dtype, bits, layer_idx, device_type, device_index), quantizer in quantizers.items()
        if kind == "selected_gain_mse" and bits == 2
    ]
    outlier = [
        quantizer
        for (kind, name, dimension, dtype, bits, layer_idx, device_type, device_index), quantizer in quantizers.items()
        if kind == "selected_gain_mse" and bits == 3
    ]

    assert regular
    assert outlier
    assert all(getattr(quantizer, "selected_gain") > 1.0 for quantizer in regular)
    assert all(getattr(quantizer, "selected_gain") > 1.0 for quantizer in outlier)


def test_regular_gain_mse_applies_only_to_fractional_regular_subsegments():
    for quantizer_kind, expected_regular_kind in [
        ("regular_gain_mse", "gain_mse"),
        ("regular_half_gain_mse", "regular_half_gain_mse"),
        ("regular_selected_gain_mse", "selected_gain_mse"),
        ("regular_clipped_gain_mse", "clipped_gain_mse"),
    ]:
        for bits, regular_bits, outlier_bits in [(2.5, 2, 3), (3.5, 3, 4)]:
            cfg = make_kv_config_from_effective_bits(
                bits,
                codebook_grid_size=10_001,
                key_quantizer=quantizer_kind,
                value_quantizer=quantizer_kind,
            )
            cache = TurboQuantDynamicCache(cfg)
            key = torch.randn(1, 2, 4, 16)
            value = torch.randn(1, 2, 4, 16)

            cache.update(key, value, layer_idx=0)
            quantizers = cache._quantizers
            regular = [
                quantizer
                for (kind, name, dimension, dtype, qbits, layer_idx, device_type, device_index), quantizer in quantizers.items()
                if kind == expected_regular_kind and qbits == regular_bits
            ]
            outlier = [
                quantizer
                for (kind, name, dimension, dtype, qbits, layer_idx, device_type, device_index), quantizer in quantizers.items()
                if kind == "mse" and qbits == outlier_bits
            ]

            assert regular
            assert outlier
            if quantizer_kind == "regular_selected_gain_mse":
                assert all(getattr(quantizer, "selected_gain") > 1.0 for quantizer in regular)
            elif quantizer_kind == "regular_clipped_gain_mse":
                assert all(getattr(quantizer, "clipped_gain_max") > 1.0 for quantizer in regular)
            else:
                assert all(getattr(quantizer, "reconstruction_gain") > 1.0 for quantizer in regular)
            assert all(getattr(quantizer, "reconstruction_gain") == 1.0 for quantizer in outlier)
            assert cache.compression_summary()["avg_effective_index_bits"] == bits


def test_adaptive_regular_gain_mse_selects_lower_reconstruction_error_candidate():
    torch.manual_seed(309)
    key = torch.randn(1, 2, 8, 16)
    value = torch.randn(1, 2, 8, 16)
    baseline = TurboQuantDynamicCache(
        make_kv_config_from_effective_bits(
            2.5,
            seed=123,
            codebook_grid_size=10_001,
            key_quantizer="mse",
            value_quantizer="mse",
        )
    )
    gain = TurboQuantDynamicCache(
        make_kv_config_from_effective_bits(
            2.5,
            seed=123,
            codebook_grid_size=10_001,
            key_quantizer="regular_gain_mse",
            value_quantizer="regular_gain_mse",
        )
    )
    adaptive = TurboQuantDynamicCache(
        make_kv_config_from_effective_bits(
            2.5,
            seed=123,
            codebook_grid_size=10_001,
            key_quantizer="adaptive_regular_gain_mse",
            value_quantizer="adaptive_regular_gain_mse",
        )
    )

    baseline_key, baseline_value = baseline.update(key, value, layer_idx=0)
    gain_key, gain_value = gain.update(key, value, layer_idx=0)
    adaptive_key, adaptive_value = adaptive.update(key, value, layer_idx=0)

    def mse(original: torch.Tensor, decoded: torch.Tensor) -> torch.Tensor:
        return torch.mean((original.to(torch.float32) - decoded.to(torch.float32)) ** 2)

    assert mse(key, adaptive_key) <= torch.minimum(mse(key, baseline_key), mse(key, gain_key)) + 1e-6
    assert mse(value, adaptive_value) <= torch.minimum(mse(value, baseline_value), mse(value, gain_value)) + 1e-6
    assert adaptive.compression_summary()["avg_effective_index_bits"] == 2.5


def test_attention_adaptive_regular_gain_mse_selects_lower_attention_proxy_candidate():
    torch.manual_seed(310)
    query = torch.randn(1, 4, 8, 16)
    key = torch.randn(1, 2, 8, 16)
    value = torch.randn(1, 2, 8, 16)
    common = {
        "seed": 123,
        "codebook_grid_size": 10_001,
        "attention_error_query_tokens": 2,
    }
    baseline = TurboQuantDynamicCache(
        make_kv_config_from_effective_bits(2.5, key_quantizer="mse", value_quantizer="mse", **common)
    )
    gain = TurboQuantDynamicCache(
        make_kv_config_from_effective_bits(
            2.5,
            key_quantizer="regular_gain_mse",
            value_quantizer="regular_gain_mse",
            **common,
        )
    )
    adaptive = TurboQuantDynamicCache(
        make_kv_config_from_effective_bits(
            2.5,
            key_quantizer="attention_adaptive_regular_gain_mse",
            value_quantizer="attention_adaptive_regular_gain_mse",
            **common,
        )
    )
    kwargs = {"query_states": query, "scaling": 16**-0.5}

    baseline_key, baseline_value = baseline.update(key, value, layer_idx=0, cache_kwargs=kwargs)
    gain_key, gain_value = gain.update(key, value, layer_idx=0, cache_kwargs=kwargs)
    adaptive_key, adaptive_value = adaptive.update(key, value, layer_idx=0, cache_kwargs=kwargs)

    def proxy(cache: TurboQuantDynamicCache, original: torch.Tensor, decoded: torch.Tensor, name: str) -> float:
        return cache._attention_auto_mse_error(
            original,
            decoded,
            name=name,
            query_states=query,
            key_states_for_attention=key,
            scaling=16**-0.5,
        )

    assert proxy(adaptive, key, adaptive_key, "key") <= min(
        proxy(baseline, key, baseline_key, "key"),
        proxy(gain, key, gain_key, "key"),
    ) + 1e-7
    assert proxy(adaptive, value, adaptive_value, "value") <= min(
        proxy(baseline, value, baseline_value, "value"),
        proxy(gain, value, gain_value, "value"),
    ) + 1e-7
    assert adaptive.compression_summary()["avg_effective_index_bits"] == 2.5


def test_distortion_regime_mse_switches_reconstruction_by_subsegment_bits():
    torch.manual_seed(244)
    cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        key_quantizer="distortion_regime_mse",
        value_quantizer="distortion_regime_mse",
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 6, 16)
    value = torch.randn(1, 2, 6, 16)

    out_key, out_value = cache.update(key, value, layer_idx=0)
    key_segment = cache.key_cache[0][0]

    assert out_key.shape == key.shape
    assert out_value.shape == value.shape
    assert isinstance(key_segment, OutlierMSESegment)
    assert isinstance(key_segment.regular, PackedMSESegment)
    assert key_segment.regular.quantizer_kind == "gain_mse"
    assert key_segment.regular.bits == 2
    assert isinstance(key_segment.outlier, PackedMSESegment)
    assert key_segment.outlier.quantizer_kind == "learned_unit_mse_block2"
    assert key_segment.outlier.bits == 3
    assert key_segment.outlier.block_size == 2
    assert cache.compression_summary()["avg_effective_index_bits"] == 2.5


def test_gain_unit_regime_mse_switches_by_subsegment_bits():
    torch.manual_seed(245)
    key = torch.randn(1, 2, 6, 16)
    value = torch.randn(1, 2, 6, 16)

    low_cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        key_quantizer="gain_unit_regime_mse",
        value_quantizer="gain_unit_regime_mse",
    )
    low_cache = TurboQuantDynamicCache(low_cfg)
    low_key, low_value = low_cache.update(key, value, layer_idx=0)
    low_segment = low_cache.key_cache[0][0]

    assert low_key.shape == key.shape
    assert low_value.shape == value.shape
    assert isinstance(low_segment, OutlierMSESegment)
    assert isinstance(low_segment.regular, PackedMSESegment)
    assert isinstance(low_segment.outlier, PackedMSESegment)
    assert low_segment.regular.quantizer_kind == "gain_mse"
    assert low_segment.regular.bits == 2
    assert low_segment.outlier.quantizer_kind == "unit_mse"
    assert low_segment.outlier.bits == 3
    assert low_cache.compression_summary()["avg_effective_index_bits"] == 2.5

    high_cfg = make_kv_config_from_effective_bits(
        3.5,
        codebook_grid_size=10_001,
        key_quantizer="gain_unit_regime_mse",
        value_quantizer="gain_unit_regime_mse",
    )
    high_cache = TurboQuantDynamicCache(high_cfg)
    high_key, high_value = high_cache.update(key, value, layer_idx=0)
    high_segment = high_cache.key_cache[0][0]

    assert high_key.shape == key.shape
    assert high_value.shape == value.shape
    assert isinstance(high_segment, OutlierMSESegment)
    assert isinstance(high_segment.regular, PackedMSESegment)
    assert isinstance(high_segment.outlier, PackedMSESegment)
    assert high_segment.regular.quantizer_kind == "unit_mse"
    assert high_segment.regular.bits == 3
    assert high_segment.outlier.quantizer_kind == "unit_mse"
    assert high_segment.outlier.bits == 4
    assert high_cache.compression_summary()["avg_effective_index_bits"] == 3.5


def test_rate_regime_mse_switches_by_requested_effective_bits():
    torch.manual_seed(246)
    key = torch.randn(1, 2, 6, 16)
    value = torch.randn(1, 2, 6, 16)

    low_cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        key_quantizer="rate_regime_mse",
        value_quantizer="rate_regime_mse",
    )
    low_cache = TurboQuantDynamicCache(low_cfg)
    low_key, low_value = low_cache.update(key, value, layer_idx=0)
    low_segment = low_cache.key_cache[0][0]

    assert low_key.shape == key.shape
    assert low_value.shape == value.shape
    assert isinstance(low_segment, OutlierMSESegment)
    assert isinstance(low_segment.regular, PackedMSESegment)
    assert isinstance(low_segment.outlier, PackedMSESegment)
    assert low_segment.regular.quantizer_kind == "gain_mse"
    assert low_segment.regular.bits == 2
    assert low_segment.outlier.quantizer_kind == "gain_mse"
    assert low_segment.outlier.bits == 3
    assert low_cache.compression_summary()["avg_effective_index_bits"] == 2.5

    high_cfg = make_kv_config_from_effective_bits(
        3.5,
        codebook_grid_size=10_001,
        key_quantizer="rate_regime_mse",
        value_quantizer="rate_regime_mse",
    )
    high_cache = TurboQuantDynamicCache(high_cfg)
    high_key, high_value = high_cache.update(key, value, layer_idx=0)
    high_segment = high_cache.key_cache[0][0]

    assert high_key.shape == key.shape
    assert high_value.shape == value.shape
    assert isinstance(high_segment, OutlierMSESegment)
    assert isinstance(high_segment.regular, PackedMSESegment)
    assert isinstance(high_segment.outlier, PackedMSESegment)
    assert high_segment.regular.quantizer_kind == "learned_unit_mse_block2"
    assert high_segment.regular.bits == 3
    assert high_segment.regular.block_size == 2
    assert high_segment.outlier.quantizer_kind == "learned_unit_mse_block2"
    assert high_segment.outlier.bits == 4
    assert high_segment.outlier.block_size == 2
    assert high_cache.compression_summary()["avg_effective_index_bits"] == 3.5


def test_hadamard_rate_regime_mse_uses_lowrate_value_hadamard_and_highrate_unit_blocks():
    torch.manual_seed(247)
    key = torch.randn(1, 2, 8, 16)
    value = torch.randn(1, 2, 8, 16)

    low_cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        key_quantizer="hadamard_rate_regime_mse",
        value_quantizer="hadamard_rate_regime_mse",
        outlier_hadamard_block_size=8,
    )
    low_cache = TurboQuantDynamicCache(low_cfg)
    low_key, low_value = low_cache.update(key, value, layer_idx=0)
    low_key_segment = low_cache.key_cache[0][0]
    low_value_segment = low_cache.value_cache[0][0]

    assert low_key.shape == key.shape
    assert low_value.shape == value.shape
    assert isinstance(low_key_segment, OutlierMSESegment)
    assert isinstance(low_key_segment.regular, PackedMSESegment)
    assert low_key_segment.regular.quantizer_kind == "mse"
    assert isinstance(low_value_segment, OutlierMSESegment)
    assert isinstance(
        low_value_segment.regular,
        (PackedMSESegment, OutlierHadamardSegment, VectorAdaptiveRotationSegment),
    )
    assert isinstance(
        low_value_segment.outlier,
        (PackedMSESegment, OutlierHadamardSegment, VectorAdaptiveRotationSegment),
    )
    assert low_cache.compression_summary()["avg_effective_index_bits"] == 2.5

    high_cfg = make_kv_config_from_effective_bits(
        3.5,
        codebook_grid_size=10_001,
        key_quantizer="hadamard_rate_regime_mse",
        value_quantizer="hadamard_rate_regime_mse",
        outlier_hadamard_block_size=8,
    )
    high_cache = TurboQuantDynamicCache(high_cfg)
    high_key, high_value = high_cache.update(key, value, layer_idx=0)
    high_key_segment = high_cache.key_cache[0][0]
    high_value_segment = high_cache.value_cache[0][0]

    assert high_key.shape == key.shape
    assert high_value.shape == value.shape
    for segment in (high_key_segment, high_value_segment):
        assert isinstance(segment, OutlierMSESegment)
        assert isinstance(segment.regular, PackedMSESegment)
        assert isinstance(segment.outlier, PackedMSESegment)
        assert segment.regular.quantizer_kind == "learned_unit_mse_block2"
        assert segment.regular.bits == 3
        assert segment.regular.block_size == 2
        assert segment.outlier.quantizer_kind == "learned_unit_mse_block2"
        assert segment.outlier.bits == 4
        assert segment.outlier.block_size == 2
    assert high_cache.compression_summary()["avg_effective_index_bits"] == 3.5


def test_rate_hadamard_value_mse_uses_lowrate_margin_and_late_highrate_value_hadamard():
    torch.manual_seed(248)
    key = torch.randn(1, 2, 8, 16)
    value = torch.randn(1, 2, 8, 16)

    low_cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        key_quantizer="rate_hadamard_value_mse",
        value_quantizer="rate_hadamard_value_mse",
        outlier_hadamard_block_size=8,
    )
    low_cache = TurboQuantDynamicCache(low_cfg)
    low_cache.update(key, value, layer_idx=0)
    low_key_segment = low_cache.key_cache[0][0]
    low_value_segment = low_cache.value_cache[0][0]

    assert isinstance(low_key_segment, OutlierMSESegment)
    assert isinstance(low_key_segment.regular, PackedMSESegment)
    assert low_key_segment.regular.quantizer_kind == "mse"
    assert isinstance(low_value_segment, OutlierMSESegment)
    assert isinstance(
        low_value_segment.regular,
        (PackedMSESegment, OutlierHadamardSegment, VectorAdaptiveRotationSegment),
    )
    assert low_cache.compression_summary()["avg_effective_index_bits"] == 2.5

    high_cfg = make_kv_config_from_effective_bits(
        3.5,
        codebook_grid_size=10_001,
        key_quantizer="rate_hadamard_value_mse",
        value_quantizer="rate_hadamard_value_mse",
        outlier_hadamard_block_size=8,
    )
    high_cache = TurboQuantDynamicCache(high_cfg)
    high_cache.update(key, value, layer_idx=0)
    high_cache.update(key, value, layer_idx=16)

    early_key_segment = high_cache.key_cache[0][0]
    early_value_segment = high_cache.value_cache[0][0]
    late_key_segment = high_cache.key_cache[16][0]
    late_value_segment = high_cache.value_cache[16][0]

    assert isinstance(early_key_segment, OutlierMSESegment)
    assert isinstance(early_key_segment.regular, PackedMSESegment)
    assert early_key_segment.regular.quantizer_kind == "mse"
    assert isinstance(early_value_segment, OutlierMSESegment)
    assert isinstance(early_value_segment.regular, PackedMSESegment)
    assert early_value_segment.regular.quantizer_kind == "mse"
    assert isinstance(late_key_segment, OutlierMSESegment)
    assert isinstance(late_key_segment.regular, PackedMSESegment)
    assert late_key_segment.regular.quantizer_kind == "mse"
    assert isinstance(late_value_segment, OutlierMSESegment)
    assert isinstance(late_value_segment.regular, OutlierHadamardSegment)
    assert isinstance(late_value_segment.outlier, OutlierHadamardSegment)
    assert high_cache.compression_summary()["avg_effective_index_bits"] == 3.5


def test_auto_mse_keeps_fractional_bit_budget():
    cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        key_quantizer="auto_mse",
        value_quantizer="auto_mse",
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 4, 16)
    value = torch.randn(1, 2, 4, 16)

    cache.update(key, value, layer_idx=0)
    summary = cache.compression_summary()

    assert summary["regular_bits"] == [2]
    assert summary["outlier_bits"] == [3]
    assert summary["avg_effective_index_bits"] == 2.5


def test_attention_auto_mse_keeps_fractional_bit_budget_with_queries():
    cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        key_quantizer="attention_auto_mse",
        value_quantizer="attention_auto_mse",
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 4, 16)
    value = torch.randn(1, 2, 4, 16)
    query = torch.randn(1, 2, 4, 16)

    cache.update(key, value, layer_idx=0, cache_kwargs={"query_states": query, "scaling": 0.25})
    summary = cache.compression_summary()

    assert summary["regular_bits"] == [2]
    assert summary["outlier_bits"] == [3]
    assert summary["avg_effective_index_bits"] == 2.5


def test_layer_quantizer_schedule_selects_per_layer_quantizers():
    cfg = make_kv_config_from_effective_bits(
        2,
        codebook_grid_size=10_001,
        key_quantizer="mse",
        value_quantizer="mse",
        layer_key_quantizers=("gain_mse", "mse"),
        layer_value_quantizers=("mse", "gain_mse"),
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 4, 16)
    value = torch.randn(1, 2, 4, 16)

    cache.update(key, value, layer_idx=0)
    cache.update(key, value, layer_idx=1)

    keys = set(cache._quantizers)
    assert any(kind == "gain_mse" and name == "key" and layer_idx == 0 for kind, name, _, _, _, layer_idx, _, _ in keys)
    assert any(kind == "mse" and name == "key" and layer_idx == 1 for kind, name, _, _, _, layer_idx, _, _ in keys)
    assert any(kind == "mse" and name == "value" and layer_idx == 0 for kind, name, _, _, _, layer_idx, _, _ in keys)
    assert any(kind == "gain_mse" and name == "value" and layer_idx == 1 for kind, name, _, _, _, layer_idx, _, _ in keys)


def test_fractional_bits_support_static_score_outlier_policy():
    cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        outlier_policy="static_score",
        layer_key_channel_scores=(tuple(range(16)),),
        layer_value_channel_scores=(tuple(reversed(range(16))),),
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 4, 16)
    value = torch.randn(1, 2, 4, 16)

    cache.update(key, value, layer_idx=0)
    key_segment = cache.key_cache[0][0]
    value_segment = cache.value_cache[0][0]

    assert isinstance(key_segment, OutlierMSESegment)
    assert isinstance(value_segment, OutlierMSESegment)
    assert key_segment.effective_index_bits == 2.5
    assert value_segment.effective_index_bits == 2.5
    assert key_segment.outlier_indices.tolist() == list(range(8, 16))
    assert value_segment.outlier_indices.tolist() == list(range(8))
    summary = cache.compression_summary()
    assert summary["outlier_policy"] == "static_score"
    assert summary["layer_key_channel_scores"] is True
    assert summary["layer_value_channel_scores"] is True


def test_key_value_outlier_policies_can_differ():
    cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        key_outlier_policy="static_score",
        value_outlier_policy="dynamic_absmean",
        layer_key_channel_scores=(tuple(range(16)),),
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 4, 16)
    value = torch.randn(1, 2, 4, 16)

    cache.update(key, value, layer_idx=0)
    key_segment = cache.key_cache[0][0]
    value_segment = cache.value_cache[0][0]

    assert isinstance(key_segment, OutlierMSESegment)
    assert isinstance(value_segment, OutlierMSESegment)
    assert key_segment.policy == "static_score"
    assert value_segment.policy == "dynamic_absmean"
    assert key_segment.outlier_indices.tolist() == list(range(8, 16))


def test_turboquant_fast_materialized_eval_matches_regular_materialization():
    torch.manual_seed(5)
    cfg = make_kv_config_from_effective_bits(2.5, codebook_grid_size=10_001)
    fast_cfg = make_kv_config_from_effective_bits(2.5, codebook_grid_size=10_001, fast_materialized_eval=True)
    cache = TurboQuantDynamicCache(cfg)
    fast_cache = TurboQuantDynamicCache(fast_cfg)

    for _ in range(3):
        key = torch.randn(1, 2, 2, 16)
        value = torch.randn(1, 2, 2, 16)
        out_k, out_v = cache.update(key, value, layer_idx=0)
        fast_k, fast_v = fast_cache.update(key, value, layer_idx=0)
        assert torch.equal(out_k, fast_k)
        assert torch.equal(out_v, fast_v)

    assert fast_cache.fast_materialized_nbytes() == fast_cache.materialized_nbytes()
    assert fast_cache.storage_nbytes() < fast_cache.materialized_nbytes()
    assert fast_cache.compression_summary()["fast_materialized_eval"] is True


def test_turboquant_dynamic_cache_supports_prod_key_segments():
    torch.manual_seed(6)
    cfg = make_kv_config_from_effective_bits(
        3,
        codebook_grid_size=10_001,
        key_quantizer="prod",
        value_quantizer="mse",
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 4, 16)
    value = torch.randn(1, 2, 4, 16)

    out_k, out_v = cache.update(key, value, layer_idx=0)

    assert out_k.shape == key.shape
    assert out_v.shape == value.shape
    assert isinstance(cache.key_cache[0][0], PackedProdSegment)
    assert isinstance(cache.value_cache[0][0], PackedMSESegment)
    assert cache.storage_nbytes() < cache.materialized_nbytes()


def test_turboquant_dynamic_cache_supports_block_mse_segments():
    torch.manual_seed(12)
    cfg = make_kv_config_from_effective_bits(
        2,
        codebook_grid_size=10_001,
        key_quantizer="mse_block2",
        value_quantizer="mse_block2",
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 4, 16)
    value = torch.randn(1, 2, 4, 16)

    out_k, out_v = cache.update(key, value, layer_idx=0)

    assert out_k.shape == key.shape
    assert out_v.shape == value.shape
    assert isinstance(cache.key_cache[0][0], PackedMSESegment)
    assert isinstance(cache.value_cache[0][0], PackedMSESegment)
    assert cache.key_cache[0][0].block_size == 2
    assert cache.value_cache[0][0].block_size == 2
    assert cache.storage_nbytes() < cache.materialized_nbytes()


def test_turboquant_dynamic_cache_supports_centered_mse_segments():
    torch.manual_seed(16)
    cfg = make_kv_config_from_effective_bits(
        2,
        codebook_grid_size=10_001,
        key_quantizer="centered_mse",
        value_quantizer="centered_mse",
    )
    cache = TurboQuantDynamicCache(cfg)
    offset = torch.randn(1, 2, 1, 16) * 4
    key = offset + torch.randn(1, 2, 32, 16) * 0.1
    value = offset + torch.randn(1, 2, 32, 16) * 0.1

    out_k, out_v = cache.update(key, value, layer_idx=0)

    assert out_k.shape == key.shape
    assert out_v.shape == value.shape
    assert isinstance(cache.key_cache[0][0], CenteredSegment)
    assert isinstance(cache.value_cache[0][0], CenteredSegment)
    assert cache.storage_nbytes() < cache.materialized_nbytes()


def test_turboquant_dynamic_cache_supports_hadamard_mse_segments():
    torch.manual_seed(17)
    cfg = make_kv_config_from_effective_bits(
        2,
        codebook_grid_size=10_001,
        key_quantizer="hadamard_mse",
        value_quantizer="hadamard_mse",
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 32, 16)
    value = torch.randn(1, 2, 32, 16)

    out_k, out_v = cache.update(key, value, layer_idx=0)

    assert out_k.shape == key.shape
    assert out_v.shape == value.shape
    assert isinstance(cache.key_cache[0][0], PackedMSESegment)
    assert cache.key_cache[0][0].quantizer_kind == "hadamard_mse"
    assert cache.storage_nbytes() < cache.materialized_nbytes()


def test_turboquant_dynamic_cache_supports_outlier_hadamard_mse_segments():
    torch.manual_seed(28)
    cfg = make_kv_config_from_effective_bits(
        2,
        codebook_grid_size=10_001,
        key_quantizer="outlier_hadamard_mse",
        value_quantizer="outlier_hadamard_mse",
        outlier_hadamard_block_size=8,
    )
    cache = TurboQuantDynamicCache(cfg)
    channel_scale = torch.linspace(0.5, 4.0, 16).view(1, 1, 1, 16)
    key = torch.randn(1, 2, 32, 16) * channel_scale
    value = torch.randn(1, 2, 32, 16) * channel_scale

    out_k, out_v = cache.update(key, value, layer_idx=0)

    assert out_k.shape == key.shape
    assert out_v.shape == value.shape
    key_segment = cache.key_cache[0][0]
    value_segment = cache.value_cache[0][0]
    assert isinstance(key_segment, OutlierHadamardSegment)
    assert isinstance(value_segment, OutlierHadamardSegment)
    assert key_segment.block_size == 8
    assert key_segment.permutation.numel() == 16
    assert key_segment.signs.numel() == 16
    assert sorted(key_segment.permutation.tolist()) == list(range(16))
    assert cache.compression_summary()["segment_types"] == {"OutlierHadamardSegment": 2}
    assert cache.compression_summary()["avg_effective_index_bits"] == 2.0
    assert cache.storage_nbytes() < cache.materialized_nbytes()


def test_outlier_hadamard_mse_supports_fractional_effective_bits():
    torch.manual_seed(29)
    cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        key_quantizer="outlier_hadamard_mse",
        value_quantizer="outlier_hadamard_mse",
        outlier_hadamard_block_size=8,
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 12, 16)
    value = torch.randn(1, 2, 12, 16)

    out_k, out_v = cache.update(key, value, layer_idx=0)

    assert out_k.shape == key.shape
    assert out_v.shape == value.shape
    key_segment = cache.key_cache[0][0]
    assert isinstance(key_segment, OutlierMSESegment)
    assert isinstance(key_segment.regular, OutlierHadamardSegment)
    assert isinstance(key_segment.outlier, OutlierHadamardSegment)
    assert cache.compression_summary()["avg_effective_index_bits"] == 2.5
    assert cache.storage_nbytes() < cache.materialized_nbytes()


def test_adaptive_outlier_hadamard_mse_never_exceeds_baseline_segment_mse():
    torch.manual_seed(301)
    baseline_cfg = make_kv_config_from_effective_bits(
        2,
        codebook_grid_size=10_001,
        key_quantizer="mse",
        value_quantizer="mse",
        outlier_hadamard_block_size=8,
    )
    adaptive_cfg = make_kv_config_from_effective_bits(
        2,
        codebook_grid_size=10_001,
        key_quantizer="adaptive_outlier_hadamard_mse",
        value_quantizer="adaptive_outlier_hadamard_mse",
        outlier_hadamard_block_size=8,
    )
    baseline_cache = TurboQuantDynamicCache(baseline_cfg)
    adaptive_cache = TurboQuantDynamicCache(adaptive_cfg)
    channel_scale = torch.linspace(0.25, 5.0, 16).view(1, 1, 1, 16)
    key = torch.randn(1, 2, 24, 16) * channel_scale
    value = torch.randn(1, 2, 24, 16) * channel_scale

    base_k, base_v = baseline_cache.update(key, value, layer_idx=0)
    adaptive_k, adaptive_v = adaptive_cache.update(key, value, layer_idx=0)

    assert adaptive_k.shape == key.shape
    assert adaptive_v.shape == value.shape
    assert torch.mean((key - adaptive_k) ** 2) <= torch.mean((key - base_k) ** 2) + 1e-6
    assert torch.mean((value - adaptive_v) ** 2) <= torch.mean((value - base_v) ** 2) + 1e-6


def test_vector_adaptive_outlier_hadamard_selects_per_vector_candidates():
    torch.manual_seed(302)
    baseline_cfg = make_kv_config_from_effective_bits(
        2,
        codebook_grid_size=10_001,
        key_quantizer="mse",
        value_quantizer="mse",
        outlier_hadamard_block_size=8,
    )
    adaptive_cfg = make_kv_config_from_effective_bits(
        2,
        codebook_grid_size=10_001,
        key_quantizer="vector_adaptive_outlier_hadamard_mse",
        value_quantizer="vector_adaptive_outlier_hadamard_mse",
        outlier_hadamard_block_size=8,
    )
    baseline_cache = TurboQuantDynamicCache(baseline_cfg)
    adaptive_cache = TurboQuantDynamicCache(adaptive_cfg)
    channel_scale = torch.linspace(0.25, 5.0, 16).view(1, 1, 1, 16)
    key = torch.randn(1, 2, 24, 16) * channel_scale
    value = torch.randn(1, 2, 24, 16) * channel_scale

    base_k, base_v = baseline_cache.update(key, value, layer_idx=0)
    adaptive_k, adaptive_v = adaptive_cache.update(key, value, layer_idx=0)

    assert adaptive_k.shape == key.shape
    assert adaptive_v.shape == value.shape
    assert torch.mean((key - adaptive_k) ** 2) <= torch.mean((key - base_k) ** 2) + 1e-6
    assert torch.mean((value - adaptive_v) ** 2) <= torch.mean((value - base_v) ** 2) + 1e-6
    segment_types = adaptive_cache.compression_summary()["segment_types"]
    assert sum(segment_types.values()) >= 2
    assert adaptive_cache.compression_summary()["avg_effective_index_bits"] == 2.0
    assert adaptive_cache.storage_nbytes() < adaptive_cache.materialized_nbytes()


def test_vector_adaptive_outlier_hadamard_supports_fractional_bits():
    torch.manual_seed(303)
    cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        key_quantizer="vector_adaptive_outlier_hadamard_mse",
        value_quantizer="vector_adaptive_outlier_hadamard_mse",
        outlier_hadamard_block_size=8,
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 12, 16)
    value = torch.randn(1, 2, 12, 16)

    out_k, out_v = cache.update(key, value, layer_idx=0)

    assert out_k.shape == key.shape
    assert out_v.shape == value.shape
    key_segment = cache.key_cache[0][0]
    assert isinstance(key_segment, OutlierMSESegment)
    assert isinstance(
        key_segment.regular,
        (VectorAdaptiveRotationSegment, PackedMSESegment, OutlierHadamardSegment),
    )
    assert isinstance(
        key_segment.outlier,
        (VectorAdaptiveRotationSegment, PackedMSESegment, OutlierHadamardSegment),
    )
    assert cache.compression_summary()["avg_effective_index_bits"] == 2.5
    assert cache.storage_nbytes() < cache.materialized_nbytes()


def test_margin_vector_outlier_hadamard_supports_fractional_bits():
    torch.manual_seed(304)
    cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        key_quantizer="margin_vector_outlier_hadamard_mse",
        value_quantizer="margin_vector_outlier_hadamard_mse",
        outlier_hadamard_block_size=8,
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 12, 16)
    value = torch.randn(1, 2, 12, 16)

    out_k, out_v = cache.update(key, value, layer_idx=0)

    assert out_k.shape == key.shape
    assert out_v.shape == value.shape
    key_segment = cache.key_cache[0][0]
    assert isinstance(key_segment, OutlierMSESegment)
    assert isinstance(
        key_segment.regular,
        (VectorAdaptiveRotationSegment, PackedMSESegment, OutlierHadamardSegment),
    )
    assert isinstance(
        key_segment.outlier,
        (VectorAdaptiveRotationSegment, PackedMSESegment, OutlierHadamardSegment),
    )
    assert cache.compression_summary()["avg_effective_index_bits"] == 2.5
    assert cache.storage_nbytes() < cache.materialized_nbytes()


def test_attention_adaptive_outlier_hadamard_supports_fractional_bits():
    torch.manual_seed(305)
    cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        key_quantizer="attention_adaptive_outlier_hadamard_mse",
        value_quantizer="attention_adaptive_outlier_hadamard_mse",
        outlier_hadamard_block_size=8,
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 12, 16)
    value = torch.randn(1, 2, 12, 16)
    query = torch.randn(1, 4, 12, 16)

    out_k, out_v = cache.update(key, value, layer_idx=0, cache_kwargs={"query_states": query, "scaling": 16**-0.5})

    assert out_k.shape == key.shape
    assert out_v.shape == value.shape
    key_segment = cache.key_cache[0][0]
    value_segment = cache.value_cache[0][0]
    assert isinstance(key_segment, (OutlierMSESegment, PackedMSESegment, OutlierHadamardSegment))
    assert isinstance(value_segment, (OutlierMSESegment, PackedMSESegment, OutlierHadamardSegment))
    assert cache.compression_summary()["avg_effective_index_bits"] == 2.5
    assert cache.storage_nbytes() < cache.materialized_nbytes()


def test_regular_outlier_hadamard_keeps_outlier_subsegment_on_mse():
    torch.manual_seed(30)
    cfg = make_kv_config_from_effective_bits(
        3.5,
        codebook_grid_size=10_001,
        key_quantizer="regular_outlier_hadamard_mse",
        value_quantizer="regular_outlier_hadamard_mse",
        outlier_hadamard_block_size=8,
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 12, 16)
    value = torch.randn(1, 2, 12, 16)

    out_k, out_v = cache.update(key, value, layer_idx=0)

    assert out_k.shape == key.shape
    assert out_v.shape == value.shape
    key_segment = cache.key_cache[0][0]
    assert isinstance(key_segment, OutlierMSESegment)
    assert isinstance(key_segment.regular, OutlierHadamardSegment)
    assert isinstance(key_segment.outlier, PackedMSESegment)
    assert key_segment.outlier.quantizer_kind == "mse"
    assert cache.compression_summary()["avg_effective_index_bits"] == 3.5


def test_outlier_only_hadamard_keeps_regular_subsegment_on_mse():
    torch.manual_seed(301)
    cfg = make_kv_config_from_effective_bits(
        3.5,
        codebook_grid_size=10_001,
        key_quantizer="outlier_only_hadamard_mse",
        value_quantizer="outlier_only_hadamard_mse",
        outlier_hadamard_block_size=8,
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 12, 16)
    value = torch.randn(1, 2, 12, 16)

    out_k, out_v = cache.update(key, value, layer_idx=0)

    assert out_k.shape == key.shape
    assert out_v.shape == value.shape
    key_segment = cache.key_cache[0][0]
    assert isinstance(key_segment, OutlierMSESegment)
    assert isinstance(key_segment.regular, PackedMSESegment)
    assert key_segment.regular.quantizer_kind == "mse"
    assert isinstance(key_segment.outlier, OutlierHadamardSegment)
    assert cache.compression_summary()["avg_effective_index_bits"] == 3.5


def test_attention_scale_mse_uses_query_projection_for_key_scales():
    torch.manual_seed(31)
    cfg = make_kv_config_from_effective_bits(
        2,
        codebook_grid_size=10_001,
        key_quantizer="attention_scale_mse",
        value_quantizer="attention_scale_mse",
        attention_error_query_tokens=2,
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 8, 16)
    value = torch.randn(1, 2, 8, 16)
    query = torch.randn(1, 4, 8, 16)

    out_key, out_value = cache.update(
        key,
        value,
        layer_idx=0,
        cache_kwargs={"query_states": query, "scaling": 16**-0.5},
    )

    assert out_key.shape == key.shape
    assert out_value.shape == value.shape
    key_segment = cache.key_cache[0][0]
    value_segment = cache.value_cache[0][0]
    assert isinstance(key_segment, PackedMSESegment)
    assert isinstance(value_segment, PackedMSESegment)
    assert key_segment.quantizer_kind == "mse"
    assert value_segment.quantizer_kind == "mse"
    base = TurboQuantDynamicCache(
        make_kv_config_from_effective_bits(2, codebook_grid_size=10_001, key_quantizer="mse", value_quantizer="mse")
    )
    base.update(key, value, layer_idx=0)
    assert not torch.allclose(key_segment.norms, base.key_cache[0][0].norms)
    assert torch.allclose(value_segment.norms, base.value_cache[0][0].norms)


def test_layer_rotation_indices_change_seed_without_metadata():
    torch.manual_seed(33)
    key = torch.randn(1, 2, 8, 16)
    value = torch.randn(1, 2, 8, 16)
    baseline = TurboQuantDynamicCache(
        make_kv_config_from_effective_bits(2, codebook_grid_size=10_001, key_quantizer="mse", value_quantizer="mse")
    )
    scheduled = TurboQuantDynamicCache(
        make_kv_config_from_effective_bits(
            2,
            codebook_grid_size=10_001,
            key_quantizer="mse",
            value_quantizer="mse",
            layer_key_rotation_indices=(1,),
            layer_value_rotation_indices=(2,),
        )
    )

    base_key, base_value = baseline.update(key, value, layer_idx=0)
    scheduled_key, scheduled_value = scheduled.update(key, value, layer_idx=0)
    key_segment = scheduled.key_cache[0][0]

    assert isinstance(key_segment, PackedMSESegment)
    assert key_segment.packed_rotation_ids is None
    assert not torch.allclose(base_key, scheduled_key)
    assert not torch.allclose(base_value, scheduled_value)
    summary = scheduled.compression_summary()
    assert summary["layer_key_rotation_indices"] == [1]
    assert summary["layer_value_rotation_indices"] == [2]
    assert scheduled.storage_nbytes() == baseline.storage_nbytes()


def test_layer_rotation_indices_apply_to_fractional_subsegments():
    torch.manual_seed(34)
    key = torch.randn(1, 2, 8, 16)
    value = torch.randn(1, 2, 8, 16)
    baseline = TurboQuantDynamicCache(
        make_kv_config_from_effective_bits(2.5, codebook_grid_size=10_001, key_quantizer="mse", value_quantizer="mse")
    )
    scheduled = TurboQuantDynamicCache(
        make_kv_config_from_effective_bits(
            2.5,
            codebook_grid_size=10_001,
            key_quantizer="mse",
            value_quantizer="mse",
            layer_key_rotation_indices=(1,),
            layer_value_rotation_indices=(2,),
        )
    )

    base_key, base_value = baseline.update(key, value, layer_idx=0)
    scheduled_key, scheduled_value = scheduled.update(key, value, layer_idx=0)
    key_segment = scheduled.key_cache[0][0]

    assert isinstance(key_segment, OutlierMSESegment)
    assert isinstance(key_segment.regular, PackedMSESegment)
    assert isinstance(key_segment.outlier, PackedMSESegment)
    assert key_segment.regular.packed_rotation_ids is None
    assert key_segment.outlier.packed_rotation_ids is None
    assert not torch.allclose(base_key, scheduled_key)
    assert not torch.allclose(base_value, scheduled_value)
    assert scheduled.storage_nbytes() == baseline.storage_nbytes()


def test_regular_attention_scale_mse_applies_to_regular_key_subsegment():
    torch.manual_seed(32)
    cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        key_quantizer="regular_attention_scale_mse",
        value_quantizer="regular_attention_scale_mse",
        attention_error_query_tokens=2,
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 8, 16)
    value = torch.randn(1, 2, 8, 16)
    query = torch.randn(1, 4, 8, 16)

    out_key, out_value = cache.update(
        key,
        value,
        layer_idx=0,
        cache_kwargs={"query_states": query, "scaling": 16**-0.5},
    )

    assert out_key.shape == key.shape
    assert out_value.shape == value.shape
    key_segment = cache.key_cache[0][0]
    value_segment = cache.value_cache[0][0]
    assert isinstance(key_segment, OutlierMSESegment)
    assert isinstance(value_segment, OutlierMSESegment)
    assert isinstance(key_segment.regular, PackedMSESegment)
    assert isinstance(key_segment.outlier, PackedMSESegment)
    assert key_segment.regular.quantizer_kind == "mse"
    assert key_segment.outlier.quantizer_kind == "mse"
    assert cache.compression_summary()["avg_effective_index_bits"] == 2.5


def test_rotation_bank_mse_bank1_matches_mse():
    torch.manual_seed(18)
    key = torch.randn(1, 2, 8, 16)
    value = torch.randn(1, 2, 8, 16)
    baseline = TurboQuantDynamicCache(
        make_kv_config_from_effective_bits(2, codebook_grid_size=10_001, key_quantizer="mse", value_quantizer="mse")
    )
    candidate = TurboQuantDynamicCache(
        make_kv_config_from_effective_bits(
            2,
            codebook_grid_size=10_001,
            key_quantizer="rotation_bank_mse",
            value_quantizer="rotation_bank_mse",
            rotation_bank_size=1,
        )
    )

    base_key, base_value = baseline.update(key, value, layer_idx=0)
    cand_key, cand_value = candidate.update(key, value, layer_idx=0)
    key_segment = candidate.key_cache[0][0]

    assert isinstance(key_segment, PackedMSESegment)
    assert key_segment.quantizer_kind == "rotation_bank_mse"
    assert key_segment.packed_rotation_ids is not None
    assert key_segment.rotation_id_bits == 1
    rotation_ids = unpack_indices(key_segment.packed_rotation_ids, bits=1, shape=key_segment.shape[:-1])
    assert torch.equal(rotation_ids, torch.zeros_like(rotation_ids))
    assert torch.allclose(cand_key, base_key)
    assert torch.allclose(cand_value, base_value)


def test_rotation_bank_mse_selects_lower_reconstruction_error_than_base_rotation():
    torch.manual_seed(19)
    key = torch.randn(1, 2, 16, 16)
    value = torch.randn(1, 2, 16, 16)
    baseline = TurboQuantDynamicCache(
        make_kv_config_from_effective_bits(2, codebook_grid_size=10_001, key_quantizer="mse", value_quantizer="mse")
    )
    candidate = TurboQuantDynamicCache(
        make_kv_config_from_effective_bits(
            2,
            codebook_grid_size=10_001,
            key_quantizer="rotation_bank_mse",
            value_quantizer="rotation_bank_mse",
            rotation_bank_size=4,
        )
    )

    base_key, base_value = baseline.update(key, value, layer_idx=0)
    cand_key, cand_value = candidate.update(key, value, layer_idx=0)

    assert torch.mean((key - cand_key) ** 2) <= torch.mean((key - base_key) ** 2) + 1e-6
    assert torch.mean((value - cand_value) ** 2) <= torch.mean((value - base_value) ** 2) + 1e-6
    assert candidate.key_cache[0][0].rotation_id_bits == 2
    assert candidate.storage_nbytes() > baseline.storage_nbytes()
    assert candidate.storage_nbytes() < candidate.materialized_nbytes()


def test_attention_rotation_bank_mse_supports_fractional_bits_with_queries():
    torch.manual_seed(21)
    cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        key_quantizer="attention_rotation_bank_mse",
        value_quantizer="attention_rotation_bank_mse",
        rotation_bank_size=3,
        attention_error_query_tokens=2,
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 6, 16)
    value = torch.randn(1, 2, 6, 16)
    query = torch.randn(1, 4, 6, 16)

    out_key, out_value = cache.update(
        key,
        value,
        layer_idx=0,
        cache_kwargs={"query_states": query, "scaling": 16**-0.5},
    )

    assert out_key.shape == key.shape
    assert out_value.shape == value.shape
    key_segment = cache.key_cache[0][0]
    value_segment = cache.value_cache[0][0]
    assert isinstance(key_segment, OutlierMSESegment)
    assert isinstance(value_segment, OutlierMSESegment)
    assert isinstance(key_segment.regular, PackedMSESegment)
    assert key_segment.regular.quantizer_kind == "attention_rotation_bank_mse"
    assert key_segment.regular.packed_rotation_ids is not None
    assert key_segment.regular.rotation_id_bits == 2
    assert cache.compression_summary()["avg_effective_index_bits"] == 2.5
    assert cache.storage_nbytes() < cache.materialized_nbytes()


def test_segment_rotation_bank_mse_bank1_matches_mse():
    torch.manual_seed(26)
    key = torch.randn(1, 2, 8, 16)
    value = torch.randn(1, 2, 8, 16)
    baseline = TurboQuantDynamicCache(
        make_kv_config_from_effective_bits(2.5, codebook_grid_size=10_001, key_quantizer="mse", value_quantizer="mse")
    )
    candidate = TurboQuantDynamicCache(
        make_kv_config_from_effective_bits(
            2.5,
            codebook_grid_size=10_001,
            key_quantizer="segment_rotation_bank_mse",
            value_quantizer="segment_rotation_bank_mse",
            rotation_bank_size=1,
        )
    )

    base_key, base_value = baseline.update(key, value, layer_idx=0)
    cand_key, cand_value = candidate.update(key, value, layer_idx=0)

    assert torch.allclose(cand_key, base_key)
    assert torch.allclose(cand_value, base_value)
    assert candidate.compression_summary()["avg_effective_index_bits"] == 2.5


def test_segment_rotation_bank_mse_selects_lower_segment_error_than_base_rotation():
    torch.manual_seed(27)
    key = torch.randn(1, 2, 16, 16)
    value = torch.randn(1, 2, 16, 16)
    baseline = TurboQuantDynamicCache(
        make_kv_config_from_effective_bits(2.5, codebook_grid_size=10_001, key_quantizer="mse", value_quantizer="mse")
    )
    candidate = TurboQuantDynamicCache(
        make_kv_config_from_effective_bits(
            2.5,
            codebook_grid_size=10_001,
            key_quantizer="segment_rotation_bank_mse",
            value_quantizer="segment_rotation_bank_mse",
            rotation_bank_size=4,
        )
    )

    base_key, base_value = baseline.update(key, value, layer_idx=0)
    cand_key, cand_value = candidate.update(key, value, layer_idx=0)

    assert torch.mean((key - cand_key) ** 2) <= torch.mean((key - base_key) ** 2) + 1e-6
    assert torch.mean((value - cand_value) ** 2) <= torch.mean((value - base_value) ** 2) + 1e-6
    assert candidate.storage_nbytes() > baseline.storage_nbytes()
    assert candidate.storage_nbytes() < candidate.materialized_nbytes()


def test_head_hadamard_mse_mixes_and_restores_kv_heads():
    torch.manual_seed(302)
    cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        key_quantizer="head_hadamard_mse",
        value_quantizer="head_hadamard_mse",
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 4, 12, 16)
    value = torch.randn(1, 4, 12, 16)

    transformed = cache._apply_head_hadamard_transform(key, name="key", layer_idx=0)
    restored = cache._inverse_head_hadamard_transform(transformed, name="key", layer_idx=0)
    out_key, out_value = cache.update(key, value, layer_idx=0)

    assert torch.allclose(restored, key, rtol=1e-4, atol=1e-4)
    assert out_key.shape == key.shape
    assert out_value.shape == value.shape
    key_segment = cache.key_cache[0][0]
    assert isinstance(key_segment, HeadHadamardSegment)
    assert isinstance(key_segment.residual, OutlierMSESegment)
    assert cache.compression_summary()["avg_effective_index_bits"] == 2.5
    assert cache.storage_nbytes() < cache.materialized_nbytes()


def test_hadamard_residual_mse_uses_fractional_bits_for_residual_signs():
    torch.manual_seed(303)
    cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        key_quantizer="hadamard_residual_mse",
        value_quantizer="hadamard_residual_mse",
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 12, 16)
    value = torch.randn(1, 2, 12, 16)

    out_key, out_value = cache.update(key, value, layer_idx=0)

    assert out_key.shape == key.shape
    assert out_value.shape == value.shape
    key_segment = cache.key_cache[0][0]
    assert isinstance(key_segment, HadamardResidualSegment)
    assert isinstance(key_segment.base, PackedMSESegment)
    assert key_segment.base_bits == 2
    assert key_segment.residual_indices.numel() == 8
    assert cache.compression_summary()["avg_effective_index_bits"] == 2.5
    assert cache.storage_nbytes() < cache.materialized_nbytes()


def test_attention_hadamard_residual_mse_accepts_query_states():
    torch.manual_seed(304)
    cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        key_quantizer="attention_hadamard_residual_mse",
        value_quantizer="mse",
        attention_error_query_tokens=2,
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 12, 16)
    value = torch.randn(1, 2, 12, 16)
    query = torch.randn(1, 4, 12, 16)

    out_key, out_value = cache.update(
        key,
        value,
        layer_idx=0,
        cache_kwargs={"query_states": query, "scaling": 16**-0.5},
    )

    assert out_key.shape == key.shape
    assert out_value.shape == value.shape
    assert isinstance(cache.key_cache[0][0], (OutlierMSESegment, HadamardResidualSegment))
    assert cache.compression_summary()["avg_effective_index_bits"] == 2.5


def test_attention_weighted_hadamard_residual_selects_query_sensitive_coefficients():
    torch.manual_seed(3041)
    baseline_cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        key_quantizer="hadamard_residual_mse",
        value_quantizer="mse",
        attention_error_query_tokens=2,
    )
    attention_cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        key_quantizer="attention_weighted_hadamard_residual_mse",
        value_quantizer="mse",
        attention_error_query_tokens=2,
    )
    baseline = TurboQuantDynamicCache(baseline_cfg)
    attention = TurboQuantDynamicCache(attention_cfg)
    key = torch.randn(1, 2, 12, 16)
    value = torch.randn(1, 2, 12, 16)
    query = torch.randn(1, 4, 12, 16)

    base_key, _ = baseline.update(
        key,
        value,
        layer_idx=0,
        cache_kwargs={"query_states": query, "scaling": 16**-0.5},
    )
    out_key, out_value = attention.update(
        key,
        value,
        layer_idx=0,
        cache_kwargs={"query_states": query, "scaling": 16**-0.5},
    )

    assert out_key.shape == key.shape
    assert out_value.shape == value.shape
    base_segment = baseline.key_cache[0][0]
    attention_segment = attention.key_cache[0][0]
    assert isinstance(base_segment, HadamardResidualSegment)
    assert isinstance(attention_segment, HadamardResidualSegment)
    assert attention_segment.base_bits == 2
    assert attention_segment.residual_indices.numel() == 8
    assert not torch.equal(attention_segment.residual_indices, base_segment.residual_indices)
    assert not torch.allclose(out_key, base_key)
    assert attention.compression_summary()["avg_effective_index_bits"] == 2.5


def test_attention_outlier_hadamard_mse_accepts_query_states_for_values():
    torch.manual_seed(305)
    cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        key_quantizer="mse",
        value_quantizer="attention_outlier_hadamard_mse",
        outlier_hadamard_block_size=8,
        attention_error_query_tokens=2,
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 12, 16)
    value = torch.randn(1, 2, 12, 16)
    query = torch.randn(1, 4, 12, 16)

    out_key, out_value = cache.update(
        key,
        value,
        layer_idx=0,
        cache_kwargs={"query_states": query, "scaling": 16**-0.5},
    )

    assert out_key.shape == key.shape
    assert out_value.shape == value.shape
    assert isinstance(cache.value_cache[0][0], (OutlierMSESegment, PackedMSESegment))
    assert cache.compression_summary()["avg_effective_index_bits"] == 2.5


def test_bitwidth_attention_outlier_hadamard_switches_by_base_bits():
    torch.manual_seed(306)
    query = torch.randn(1, 4, 12, 16)
    key = torch.randn(1, 2, 12, 16)
    value = torch.randn(1, 2, 12, 16)
    low_cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        key_quantizer="mse",
        value_quantizer="bitwidth_attention_outlier_hadamard_mse",
        outlier_hadamard_block_size=8,
        attention_error_query_tokens=2,
    )
    high_cfg = make_kv_config_from_effective_bits(
        3.5,
        codebook_grid_size=10_001,
        key_quantizer="mse",
        value_quantizer="bitwidth_attention_outlier_hadamard_mse",
        outlier_hadamard_block_size=8,
        attention_error_query_tokens=2,
    )
    low_cache = TurboQuantDynamicCache(low_cfg)
    high_cache = TurboQuantDynamicCache(high_cfg)

    low_cache.update(key, value, layer_idx=0, cache_kwargs={"query_states": query, "scaling": 16**-0.5})
    high_cache.update(key, value, layer_idx=0, cache_kwargs={"query_states": query, "scaling": 16**-0.5})

    assert isinstance(low_cache.value_cache[0][0], OutlierMSESegment)
    assert isinstance(high_cache.value_cache[0][0], (OutlierMSESegment, PackedMSESegment))
    assert low_cache.compression_summary()["avg_effective_index_bits"] == 2.5
    assert high_cache.compression_summary()["avg_effective_index_bits"] == 3.5


def test_entropy_guarded_outlier_hadamard_uses_attention_entropy_gate():
    torch.manual_seed(307)
    query = torch.randn(1, 4, 12, 16)
    key = torch.randn(1, 2, 12, 16)
    value = torch.randn(1, 2, 12, 16)
    active_cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        key_quantizer="mse",
        value_quantizer="entropy_guarded_outlier_hadamard_mse",
        outlier_hadamard_block_size=8,
        attention_error_query_tokens=2,
        attention_entropy_threshold=1.0,
    )
    inactive_cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        key_quantizer="mse",
        value_quantizer="entropy_guarded_outlier_hadamard_mse",
        outlier_hadamard_block_size=8,
        attention_error_query_tokens=2,
        attention_entropy_threshold=0.0,
    )
    active_cache = TurboQuantDynamicCache(active_cfg)
    inactive_cache = TurboQuantDynamicCache(inactive_cfg)

    active_cache.update(key, value, layer_idx=0, cache_kwargs={"query_states": query, "scaling": 16**-0.5})
    inactive_cache.update(key, value, layer_idx=0, cache_kwargs={"query_states": query, "scaling": 16**-0.5})

    active_segment = active_cache.value_cache[0][0]
    inactive_segment = inactive_cache.value_cache[0][0]
    assert isinstance(active_segment, OutlierMSESegment)
    assert isinstance(active_segment.regular, OutlierHadamardSegment)
    assert isinstance(inactive_segment, OutlierMSESegment)
    assert isinstance(inactive_segment.regular, PackedMSESegment)
    assert active_cache.compression_summary()["avg_effective_index_bits"] == 2.5
    assert inactive_cache.compression_summary()["avg_effective_index_bits"] == 2.5


def test_rms_rotation_mse_applies_segment_preconditioning():
    torch.manual_seed(22)
    cfg = make_kv_config_from_effective_bits(
        2,
        codebook_grid_size=10_001,
        key_quantizer="rms_rotation_mse",
        value_quantizer="rms_rotation_mse",
    )
    cache = TurboQuantDynamicCache(cfg)
    channel_scale = torch.linspace(0.5, 3.0, 16).view(1, 1, 1, 16)
    key = torch.randn(1, 2, 12, 16) * channel_scale
    value = torch.randn(1, 2, 12, 16) * channel_scale

    out_key, out_value = cache.update(key, value, layer_idx=0)

    assert out_key.shape == key.shape
    assert out_value.shape == value.shape
    key_segment = cache.key_cache[0][0]
    assert isinstance(key_segment, RmsScaledSegment)
    assert isinstance(key_segment.residual, PackedMSESegment)
    assert key_segment.scales.shape == (16,)
    assert torch.std(key_segment.scales.to(torch.float32)) > 0
    assert cache.storage_nbytes() < cache.materialized_nbytes()


def test_rms_regular_gain_mse_combines_preconditioning_and_regular_gain():
    torch.manual_seed(23)
    cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        key_quantizer="rms_regular_gain_mse",
        value_quantizer="rms_regular_gain_mse",
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 12, 16)
    value = torch.randn(1, 2, 12, 16)

    out_key, out_value = cache.update(key, value, layer_idx=0)

    assert out_key.shape == key.shape
    assert out_value.shape == value.shape
    key_segment = cache.key_cache[0][0]
    assert isinstance(key_segment, OutlierMSESegment)
    assert isinstance(key_segment.regular, RmsScaledSegment)
    assert isinstance(key_segment.outlier, RmsScaledSegment)
    assert isinstance(key_segment.regular.residual, PackedMSESegment)
    assert key_segment.regular.residual.quantizer_kind == "gain_mse"
    assert cache.compression_summary()["avg_effective_index_bits"] == 2.5


def test_head_rotation_mse_uses_independent_head_segments_for_fractional_bits():
    torch.manual_seed(34)
    cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        key_quantizer="head_rotation_mse",
        value_quantizer="head_rotation_mse",
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 8, 16)
    value = torch.randn(1, 2, 8, 16)

    out_key, out_value = cache.update(key, value, layer_idx=0)

    assert out_key.shape == key.shape
    assert out_value.shape == value.shape
    key_segment = cache.key_cache[0][0]
    value_segment = cache.value_cache[0][0]
    assert isinstance(key_segment, HeadRotationMSESegment)
    assert isinstance(value_segment, HeadRotationMSESegment)
    assert len(key_segment.head_segments) == 2
    assert len(value_segment.head_segments) == 2
    assert all(isinstance(segment, OutlierMSESegment) for segment in key_segment.head_segments)
    assert all(isinstance(segment, OutlierMSESegment) for segment in value_segment.head_segments)
    assert cache.compression_summary()["avg_effective_index_bits"] == 2.5
    assert cache.storage_nbytes() < cache.materialized_nbytes()


def test_calibrated_rotated_outlier_mse_uses_fractional_full_rotation():
    torch.manual_seed(35)
    transform = torch.eye(16).tolist()
    cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        key_quantizer="calibrated_rotated_outlier_mse",
        value_quantizer="calibrated_rotated_outlier_mse",
        layer_key_rotation_matrices=(tuple(tuple(row) for row in transform),),
        layer_value_rotation_matrices=(tuple(tuple(row) for row in transform),),
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 8, 16)
    value = torch.randn(1, 2, 8, 16)

    out_key, out_value = cache.update(key, value, layer_idx=0)

    assert out_key.shape == key.shape
    assert out_value.shape == value.shape
    key_segment = cache.key_cache[0][0]
    assert isinstance(key_segment, RotatedOutlierMSESegment)
    assert key_segment.quantizer_kind == "calibrated_rotated_outlier_mse"
    summary = cache.compression_summary()
    assert summary["avg_effective_index_bits"] == 2.5
    assert summary["layer_key_rotation_matrices"] is True


def test_token_protection_keeps_budgeted_raw_tokens():
    torch.manual_seed(13)
    cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        token_protection_policy="sink_recent_budget",
        protected_start_tokens=2,
        token_quant_bits=2,
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 32, 16)
    value = torch.randn(1, 2, 32, 16)
    baseline_cache = TurboQuantDynamicCache(make_kv_config_from_effective_bits(2.5, codebook_grid_size=10_001))
    baseline_cache.update(key, value, layer_idx=0)

    out_k, out_v = cache.update(key, value, layer_idx=0)
    key_segment = cache.key_cache[0][0]

    assert out_k.shape == key.shape
    assert out_v.shape == value.shape
    assert isinstance(key_segment, TokenProtectedSegment)
    assert key_segment.quant_bits == 2
    assert key_segment.raw_indices.numel() > 0
    assert key_segment.raw_indices[:2].tolist() == [0, 1]
    assert key_segment.nbytes <= baseline_cache.key_cache[0][0].nbytes
    assert cache.storage_nbytes() <= baseline_cache.storage_nbytes()
    assert cache.storage_nbytes() < cache.materialized_nbytes()


def test_token_protection_can_match_storage_ratio_budget():
    cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        token_protection_policy="sink_recent_budget",
        protected_start_tokens=2,
        token_quant_bits=2,
        token_protection_target_ratio=0.25,
    )
    cache = TurboQuantDynamicCache(cfg)
    assert cache._token_protected_raw_count(100, target_bits=2.5, quant_bits=2) == 14


def test_token_protection_can_target_keys_only():
    torch.manual_seed(14)
    cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        token_protection_policy="sink_recent_budget",
        token_quant_bits=2,
        token_protection_targets="key",
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 32, 16)
    value = torch.randn(1, 2, 32, 16)

    cache.update(key, value, layer_idx=0)

    assert isinstance(cache.key_cache[0][0], TokenProtectedSegment)
    assert isinstance(cache.value_cache[0][0], OutlierMSESegment)


def test_attention_error_token_protection_uses_query_scores():
    torch.manual_seed(15)
    cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        token_protection_policy="attention_error_budget",
        token_quant_bits=2,
        token_protection_targets="key",
    )
    cache = TurboQuantDynamicCache(cfg)
    baseline_cache = TurboQuantDynamicCache(make_kv_config_from_effective_bits(2.5, codebook_grid_size=10_001))
    key = torch.randn(1, 2, 32, 16)
    value = torch.randn(1, 2, 32, 16)
    query = torch.zeros(1, 4, 1, 16)
    query[..., :] = key[:, :1, 20:21, :]

    baseline_cache.update(key, value, layer_idx=0)
    cache.update(key, value, layer_idx=0, cache_kwargs={"query_states": query, "scaling": 1.0})
    key_segment = cache.key_cache[0][0]

    assert isinstance(key_segment, TokenProtectedSegment)
    assert 20 in key_segment.raw_indices.tolist()
    assert key_segment.nbytes <= baseline_cache.key_cache[0][0].nbytes


def test_uniform_affine_cache_baselines_roundtrip_shapes():
    torch.manual_seed(7)
    cfg = make_kv_config_from_effective_bits(
        4,
        key_quantizer="uniform_channel",
        value_quantizer="uniform_token",
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 4, 16)
    value = torch.randn(1, 2, 4, 16)

    out_k, out_v = cache.update(key, value, layer_idx=0)

    assert out_k.shape == key.shape
    assert out_v.shape == value.shape
    assert isinstance(cache.key_cache[0][0], UniformAffineSegment)
    assert isinstance(cache.value_cache[0][0], UniformAffineSegment)
    assert cache.key_cache[0][0].granularity == "channel"
    assert cache.value_cache[0][0].granularity == "token"
    assert cache.storage_nbytes() < cache.materialized_nbytes()


def test_layer_bit_schedule_changes_segment_bits_by_layer():
    torch.manual_seed(8)
    cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        layer_key_bits=(2.0, 3.0),
        layer_value_bits=(2.0, 3.0),
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 4, 16)
    value = torch.randn(1, 2, 4, 16)

    cache.update(key, value, layer_idx=0)
    cache.update(key, value, layer_idx=1)

    assert isinstance(cache.key_cache[0][0], PackedMSESegment)
    assert isinstance(cache.value_cache[0][0], PackedMSESegment)
    assert cache.key_cache[0][0].bits == 2
    assert cache.value_cache[0][0].bits == 2
    assert isinstance(cache.key_cache[1][0], PackedMSESegment)
    assert isinstance(cache.value_cache[1][0], PackedMSESegment)
    assert cache.key_cache[1][0].bits == 3
    assert cache.value_cache[1][0].bits == 3
    summary = cache.compression_summary()
    assert summary["layer_key_bits"] == [2.0, 3.0]
    assert summary["layer_value_bits"] == [2.0, 3.0]


def test_sensitivity_error_gain_uses_calibrated_channel_scores():
    torch.manual_seed(16)
    key = torch.randn(1, 2, 8, 16)
    value = torch.randn(1, 2, 8, 16)
    scores = tuple([0.0] * 15 + [1_000.0])
    cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        outlier_policy="sensitivity_error_gain",
        layer_key_channel_scores=(scores,),
        layer_value_channel_scores=(scores,),
    )
    cache = TurboQuantDynamicCache(cfg)

    cache.update(key, value, layer_idx=0)

    key_segment = cache.key_cache[0][0]
    value_segment = cache.value_cache[0][0]
    assert isinstance(key_segment, OutlierMSESegment)
    assert isinstance(value_segment, OutlierMSESegment)
    assert 15 in key_segment.outlier_indices.tolist()
    assert 15 in value_segment.outlier_indices.tolist()
    assert cache.compression_summary()["sensitivity_score_power"] == 1.0


def test_head_adaptive_outlier_policy_quantizes_each_head_independently():
    torch.manual_seed(17)
    cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        outlier_policy="head_dynamic_absmean",
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 8, 16)
    value = torch.randn(1, 2, 8, 16)
    key[:, 0, :, 0] += 20
    key[:, 1, :, 15] += 20

    out_k, out_v = cache.update(key, value, layer_idx=0)
    key_segment = cache.key_cache[0][0]

    assert out_k.shape == key.shape
    assert out_v.shape == value.shape
    assert isinstance(key_segment, HeadAdaptiveOutlierMSESegment)
    assert len(key_segment.head_segments) == 2
    assert 0 in key_segment.head_segments[0].outlier_indices.tolist()
    assert 15 in key_segment.head_segments[1].outlier_indices.tolist()
    assert cache.storage_nbytes() < cache.materialized_nbytes()
