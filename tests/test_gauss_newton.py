#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testes unitários do laço de Gauss-Newton (test_gauss_newton.py)
Critérios do SDD (Seção 5 e Fase 3):
- Dados sem ruído: recupera θ com erro < 1e-6 km.
- Convergência a partir de múltiplos chutes iniciais.
- Verificação dos dois solvers (QR e Equações Normais).
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.geo import haversine
from src.gauss_newton import gauss_newton


@pytest.fixture
def cenario_sintetico_sp():
    """Gera um cenário de teste com alvo em SP e 6 servidores conhecidos."""
    lat_real, lon_real, beta_real = -23.5505, -46.6333, 15.0
    theta_real = np.array([lat_real, lon_real, beta_real])

    servidores = np.array([
        [-22.8144, -47.0647],  # Unicamp
        [-22.8631, -43.2307],  # UFRJ
        [-19.8691, -43.9664],  # UFMG
        [-30.0700, -51.1200],  # UFRGS
        [-15.7634, -47.8703],  # UnB
        [-3.7319, -38.5267]    # Fortaleza
    ])

    v_rede = 100.0
    distancias = haversine(lat_real, lon_real, servidores[:, 0], servidores[:, 1])
    rtt_sintetico = (distancias + beta_real) / v_rede

    return servidores, rtt_sintetico, theta_real


def test_gauss_newton_recuperacao_exata_qr(cenario_sintetico_sp):
    """Critério SDD: dados sem ruído recuperam θ com erro < 1e-6 (via QR)."""
    servidores, rtt, theta_real = cenario_sintetico_sp
    chute_inicial = np.array([-22.0, -45.0, 0.0])  # ~200 km de SP

    res = gauss_newton(servidores, rtt, chute_inicial, solver="qr")

    assert res.converged is True
    erro_posicao_km = haversine(res.lat, res.lon, theta_real[0], theta_real[1])
    erro_beta_km = abs(res.beta_km - theta_real[2])

    assert erro_posicao_km < 1e-6, f"Erro posição: {erro_posicao_km:.2e} km"
    assert erro_beta_km < 1e-6, f"Erro viés beta: {erro_beta_km:.2e} km"
    assert res.n_iter <= 10


def test_gauss_newton_recuperacao_exata_normais(cenario_sintetico_sp):
    """Critério SDD: dados sem ruído recuperam θ com erro < 1e-6 (via Equações Normais)."""
    servidores, rtt, theta_real = cenario_sintetico_sp
    chute_inicial = np.array([-22.0, -45.0, 0.0])

    res = gauss_newton(servidores, rtt, chute_inicial, solver="normal")

    assert res.converged is True
    erro_posicao_km = haversine(res.lat, res.lon, theta_real[0], theta_real[1])
    assert erro_posicao_km < 1e-6
    assert abs(res.beta_km - theta_real[2]) < 1e-6


def test_convergencia_multiplos_chutes(cenario_sintetico_sp):
    """Critério SDD: Gauss-Newton converge a partir de ≥ 3 chutes diferentes."""
    servidores, rtt, theta_real = cenario_sintetico_sp

    chutes = [
        np.array([-23.0, -46.0, 10.0]),  # Chute 1: Perto (< 100 km)
        np.array([-19.8, -44.0, 0.0]),   # Chute 2: Médio (BH, ~500 km)
        np.array([-15.7, -47.8, 0.0])    # Chute 3: Distante (Brasília, ~870 km)
    ]

    for chute in chutes:
        res = gauss_newton(servidores, rtt, chute, solver="qr")
        assert res.converged is True
        erro_km = haversine(res.lat, res.lon, theta_real[0], theta_real[1])
        assert erro_km < 1e-5


def test_historico_iteracoes(cenario_sintetico_sp):
    """Verifica se o histórico de convergência é registrado corretamente a cada passo."""
    servidores, rtt, _ = cenario_sintetico_sp
    chute = np.array([-20.0, -45.0, 0.0])

    res = gauss_newton(servidores, rtt, chute, solver="qr")
    assert len(res.history) == res.n_iter

    # Verifica se os resíduos foram decrescendo no histórico
    normas_res = [h["res_norm"] for h in res.history]
    assert normas_res[-1] < normas_res[0]


if __name__ == "__main__":
    pytest.main([__file__])
