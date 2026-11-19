#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testes unitários do módulo de DOP (test_dop.py)
Critério do SDD (Seção 5):
- test_dop: caso simples analítico (ex.: 4 servidores simétricos) -> valor esperado.
- Validação da geometria boa vs. ruim (PDOP, GDOP e κ_2(G)).
- Validação da inversa via Cholesky próprio vs. NumPy.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dop import dop_metrics, jacobian_tangent_km, dop_from_geometry, _inverter_spd_cholesky


def test_dop_matriz_analitica_4_simetricos():
    """
    Caso analítico exato (SDD Seção 5):
    4 servidores em N, L, S, O -> G = [[-1,0,1], [0,-1,1], [1,0,1], [0,1,1]].
    G^T G = diag(2, 2, 4) -> Q = diag(1/2, 1/2, 1/4).
    Valores exatos:
        PDOP = sqrt(1/2 + 1/2) = 1.0
        TDOP = sqrt(1/4)       = 0.5
        GDOP = sqrt(5/4)       = sqrt(1.25) ≈ 1.1180339887
        κ_2  = sqrt(4 / 2)     = sqrt(2)    ≈ 1.4142135624
    """
    G = np.array([
        [-1.0,  0.0, 1.0],
        [ 0.0, -1.0, 1.0],
        [ 1.0,  0.0, 1.0],
        [ 0.0,  1.0, 1.0],
    ])
    res = dop_metrics(G)

    assert abs(res["pdop"] - 1.0) < 1e-12
    assert abs(res["tdop"] - 0.5) < 1e-12
    assert abs(res["gdop"] - np.sqrt(1.25)) < 1e-12
    assert abs(res["cond2"] - np.sqrt(2.0)) < 1e-12


def test_dop_4_servidores_simetricos_na_esfera():
    """
    4 servidores simétricos na esfera (Norte, Sul, Leste, Oeste a 5° do alvo no Equador)
    devem reproduzir os valores analíticos exatos do plano tangente.
    """
    servidores = np.array([
        [ 5.0,  0.0],  # Norte
        [-5.0,  0.0],  # Sul
        [ 0.0,  5.0],  # Leste
        [ 0.0, -5.0],  # Oeste
    ])
    res = dop_from_geometry(0.0, 0.0, servidores)

    assert abs(res["pdop"] - 1.0) < 1e-6
    assert abs(res["tdop"] - 0.5) < 1e-6
    assert abs(res["gdop"] - np.sqrt(1.25)) < 1e-6
    assert abs(res["cond2"] - np.sqrt(2.0)) < 1e-6


def test_jacobiana_tangente_tem_vetores_unitarios():
    """
    Cada linha de G_km deve ter [ -u_norte, -u_leste, 1.0 ], com
    u_norte^2 + u_leste^2 == 1.0 (equivalência com GPS — SDD Seção 2.4).
    """
    servidores = np.array([
        [-22.8631, -43.2307],  # UFRJ
        [-19.8691, -43.9664],  # UFMG
        [-30.0700, -51.1200],  # UFRGS
        [-20.5044, -54.6144],  # UFMS
    ])
    G_km = jacobian_tangent_km(-23.5505, -46.6333, servidores)

    normas_direcao = np.sqrt(G_km[:, 0] ** 2 + G_km[:, 1] ** 2)
    assert np.allclose(normas_direcao, 1.0, atol=1e-10)
    assert np.allclose(G_km[:, 2], 1.0, atol=1e-12)


def test_geometria_boa_vs_ruim():
    """
    Compara geometria boa (6 servidores em 360° ao redor de SP) vs.
    geometria ruim (6 servidores todos na direção da Europa).
    A geometria ruim deve ter PDOP, GDOP e κ_2 drasticamente maiores.
    """
    alvo_lat, alvo_lon = -23.5505, -46.6333

    # Geometria boa: 6 servidores igualmente espaçados em azimute ao redor de SP
    angulos = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    raio_graus = 8.0
    srv_boa = np.column_stack([
        alvo_lat + raio_graus * np.cos(angulos),
        alvo_lon + (raio_graus / np.cos(np.radians(alvo_lat))) * np.sin(angulos),
    ])

    # Geometria ruim: todos os servidores agrupados na Europa (mesmo quadrante NE)
    srv_ruim = np.array([
        [38.72, -9.14],   # Lisboa
        [40.42, -3.70],   # Madri
        [48.86,  2.35],   # Paris
        [51.51, -0.13],   # Londres
        [52.52, 13.40],   # Berlim
        [41.90, 12.50],   # Roma
    ])

    m_boa = dop_from_geometry(alvo_lat, alvo_lon, srv_boa)
    m_ruim = dop_from_geometry(alvo_lat, alvo_lon, srv_ruim)

    assert m_boa["pdop"] < 1.2
    assert m_ruim["pdop"] > 10.0 * m_boa["pdop"]
    assert m_ruim["gdop"] > 10.0 * m_boa["gdop"]
    assert m_ruim["cond2"] > 10.0 * m_boa["cond2"]


def test_inversa_cholesky_vs_numpy():
    """Valida a inversão de (G^T G) pelo nosso Cholesky próprio contra np.linalg.inv."""
    rng = np.random.default_rng(42)
    G = rng.standard_normal((8, 3))
    A = G.T @ G

    Q_nosso = _inverter_spd_cholesky(A)
    Q_numpy = np.linalg.inv(A)

    assert np.allclose(Q_nosso, Q_numpy, atol=1e-12)


def test_dop_matriz_singular_retorna_inf():
    """Quando todos os servidores são colineares/idênticos, DOP deve retornar inf sem quebrar."""
    G_singular = np.array([
        [1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
    ])
    res = dop_metrics(G_singular)
    assert np.isinf(res["pdop"])
    assert np.isinf(res["gdop"])


if __name__ == "__main__":
    pytest.main([__file__])
