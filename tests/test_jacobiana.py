#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste unitário da Jacobiana (test_jacobiana.py)
Critério do SDD (Seção 5): derivada analítica vs. diferenças finitas com erro relativo < 1e-6.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import residuals, jacobian_analytical, jacobian_finite_diff, jacobian


def test_dimensoes_residuos_e_jacobiana():
    """Verifica se os formatos de r(θ) e G(θ) estão corretos para m servidores."""
    theta = np.array([-23.55, -46.63, 10.0])
    servidores = np.array([
        [-22.90, -43.17],
        [-19.86, -43.96],
        [-25.42, -49.27],
        [-15.76, -47.87],
        [-3.73, -38.52]
    ])
    rtt = np.array([12.0, 16.0, 18.0, 20.0, 52.0])

    r = residuals(theta, servidores, rtt)
    G = jacobian(theta, servidores)

    assert r.shape == (5,)
    assert G.shape == (5, 3)
    # A terceira coluna de G (derivada em relação a beta) deve ser estritamente 1.0
    assert np.allclose(G[:, 2], 1.0)


def test_jacobiana_analitica_vs_diferencas_finitas():
    """
    Critério obrigatório do SDD (Fase 2 e Seção 5):
    O erro relativo entre a Jacobiana analítica e as diferenças finitas deve ser < 1e-6.
    """
    # Alvo em São Paulo
    theta = np.array([-23.5505, -46.6333, 20.0])

    # Servidores espalhados pelo Brasil
    servidores = np.array([
        [-22.8144, -47.0647],  # Campinas
        [-22.8631, -43.2307],  # Rio
        [-19.8691, -43.9664],  # BH
        [-30.0700, -51.1200],  # Porto Alegre
        [-15.7634, -47.8703],  # Brasília
        [-3.7319, -38.5267],   # Fortaleza
        [40.4433, -79.9436]    # EUA
    ])

    G_analitica = jacobian_analytical(theta, servidores)
    G_diferencas = jacobian_finite_diff(theta, servidores, h_deg=1e-6)

    # Erro relativo na norma de Frobenius
    norma_analitica = np.linalg.norm(G_analitica)
    erro_absoluto = np.linalg.norm(G_analitica - G_diferencas)
    erro_relativo = erro_absoluto / norma_analitica

    print(f"\nErro relativo G_ana vs G_num: {erro_relativo:.2e}")
    assert erro_relativo < 1e-6, f"Erro relativo {erro_relativo:.2e} excedeu o limite de 1e-6"


def test_consistencia_residuos():
    """Sem atraso e com RTT exato, o resíduo deve ser zero."""
    theta = np.array([-23.55, -46.63, 0.0])  # beta = 0
    servidores = np.array([[-22.90, -43.17]])
    v = 100.0

    # Distância exata
    from src.geo import haversine
    dist_real = haversine(theta[0], theta[1], servidores[0, 0], servidores[0, 1])
    rtt_exato = np.array([dist_real / v])

    r = residuals(theta, servidores, rtt_exato, v=v)
    assert abs(r[0]) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__])
