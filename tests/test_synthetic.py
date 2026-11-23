#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Testes do gerador de dados sintéticos (src/synthetic.py)."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.geo import haversine
from src.gauss_newton import gauss_newton
from src.dop import dop_from_geometry
from src.synthetic import (
    generate_synthetic_network,
    simulate_rtt,
    servers_to_array,
    velocidade_gauss_newton,
)

LAT_SP, LON_SP = -23.5505, -46.6333
B_MS = 5.0
V = 100.0


@pytest.mark.parametrize("solver", ["qr", "normal"])
def test_convergencia_sem_ruido(solver):
    """σ = 0: recupera a posição com erro < 1e-5 km partindo de ~500 km."""
    rede = generate_synthetic_network(LAT_SP, LON_SP, n_servers=8, radius_km=2000.0,
                                      geometry="good", seed=42)
    rtt = simulate_rtt(rede, LAT_SP, LON_SP, target_b_ms=B_MS, v_km_per_ms=V,
                       noise_std_ms=0.0)

    chute = np.array([-20.0, -44.0, 0.0])
    dist_chute = haversine(LAT_SP, LON_SP, chute[0], chute[1])
    assert 400.0 < dist_chute < 600.0

    res = gauss_newton(servers_to_array(rede), rtt, chute,
                       v=velocidade_gauss_newton(V), solver=solver)

    assert res.converged is True
    erro_km = haversine(res.lat, res.lon, LAT_SP, LON_SP)
    assert erro_km < 1e-5, f"Erro de posição: {erro_km:.2e} km"

    b_estimado = res.beta_km / velocidade_gauss_newton(V)
    assert abs(b_estimado - B_MS) < 1e-6


def test_rtt_sem_ruido_segue_modelo_fisico():
    rede = generate_synthetic_network(LAT_SP, LON_SP, seed=7)
    rtt = simulate_rtt(rede, LAT_SP, LON_SP, target_b_ms=B_MS, v_km_per_ms=V)
    coords = servers_to_array(rede)
    d = haversine(LAT_SP, LON_SP, coords[:, 0], coords[:, 1])
    np.testing.assert_allclose(rtt, 2.0 * d / V + B_MS, rtol=0, atol=1e-12)


@pytest.mark.parametrize("geometry", ["good", "bad"])
@pytest.mark.parametrize("noise_type", ["gaussian", "lognormal"])
def test_reprodutibilidade_mesma_seed(geometry, noise_type):
    rede_a = generate_synthetic_network(LAT_SP, LON_SP, geometry=geometry, seed=123)
    rede_b = generate_synthetic_network(LAT_SP, LON_SP, geometry=geometry, seed=123)
    assert rede_a == rede_b

    rtt_a = simulate_rtt(rede_a, LAT_SP, LON_SP, noise_std_ms=2.0,
                         noise_type=noise_type, seed=123)
    rtt_b = simulate_rtt(rede_b, LAT_SP, LON_SP, noise_std_ms=2.0,
                         noise_type=noise_type, seed=123)
    np.testing.assert_array_equal(rtt_a, rtt_b)


def test_seeds_diferentes_geram_dados_diferentes():
    rede_a = generate_synthetic_network(LAT_SP, LON_SP, seed=1)
    rede_b = generate_synthetic_network(LAT_SP, LON_SP, seed=2)
    assert rede_a != rede_b

    rtt_a = simulate_rtt(rede_a, LAT_SP, LON_SP, noise_std_ms=1.0, seed=1)
    rtt_b = simulate_rtt(rede_a, LAT_SP, LON_SP, noise_std_ms=1.0, seed=2)
    assert not np.array_equal(rtt_a, rtt_b)


def test_formato_da_rede():
    rede = generate_synthetic_network(LAT_SP, LON_SP, n_servers=10, radius_km=1500.0)
    assert len(rede) == 10
    for lat, lon, nome in rede:
        assert isinstance(lat, float) and isinstance(lon, float)
        assert isinstance(nome, str)
        assert -90.0 <= lat <= 90.0 and -180.0 <= lon < 180.0
        assert haversine(LAT_SP, LON_SP, lat, lon) <= 1500.0 + 1e-6
    assert len({n for _, _, n in rede}) == 10


def test_ruido_lognormal_e_positivo():
    """Atraso de fila nunca reduz o RTT."""
    rede = generate_synthetic_network(LAT_SP, LON_SP, n_servers=50, seed=3)
    limpo = simulate_rtt(rede, LAT_SP, LON_SP)
    ruidoso = simulate_rtt(rede, LAT_SP, LON_SP, noise_std_ms=1.0,
                           noise_type="lognormal", seed=3)
    assert np.all(ruidoso > limpo)


@pytest.mark.parametrize("kwargs", [
    {"geometry": "triangular"},
    {"n_servers": 2},
    {"radius_km": 0.0},
])
def test_parametros_invalidos_rede(kwargs):
    with pytest.raises(ValueError):
        generate_synthetic_network(LAT_SP, LON_SP, **kwargs)


def test_tipo_de_ruido_invalido():
    rede = generate_synthetic_network(LAT_SP, LON_SP)
    with pytest.raises(ValueError):
        simulate_rtt(rede, LAT_SP, LON_SP, noise_std_ms=1.0, noise_type="uniforme")


@pytest.mark.parametrize("seed", [0, 1, 2, 42])
def test_geometria_ruim_tem_kappa_e_pdop_maiores(seed):
    """'bad' deve ter κ_2(G) e PDOP pelo menos 10x maiores que 'good'."""
    boa = generate_synthetic_network(LAT_SP, LON_SP, radius_km=2500.0,
                                     geometry="good", seed=seed)
    ruim = generate_synthetic_network(LAT_SP, LON_SP, radius_km=2500.0,
                                      geometry="bad", seed=seed)

    m_boa = dop_from_geometry(LAT_SP, LON_SP, servers_to_array(boa))
    m_ruim = dop_from_geometry(LAT_SP, LON_SP, servers_to_array(ruim))

    assert m_ruim["cond2"] > 10.0 * m_boa["cond2"], (m_boa, m_ruim)
    assert m_ruim["pdop"] > 10.0 * m_boa["pdop"], (m_boa, m_ruim)


if __name__ == "__main__":
    pytest.main([__file__])
