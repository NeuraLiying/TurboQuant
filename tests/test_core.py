import torch

from turboquant import TurboQuantDynamicCache, TurboQuantMSE, TurboQuantProd
from turboquant.core import TurboQuantLearnedBlockMSE, hadamard_orthogonal
from turboquant.kv_cache import (
    CenteredSegment,
    HeadAdaptiveOutlierMSESegment,
    OutlierMSESegment,
    PackedMSESegment,
    PackedProdSegment,
    UniformAffineSegment,
    TokenProtectedSegment,
    make_kv_config_from_effective_bits,
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
