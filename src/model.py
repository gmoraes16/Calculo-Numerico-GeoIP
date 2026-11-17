#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Modelagem (model.py) — Cálculo Numérico
Implementa o modelo direto, a função de resíduos r(θ) e a matriz Jacobiana G(θ).

Parâmetros do modelo:
    θ = (φ, λ, β)^T
    - φ: Latitude estimada do alvo (graus decimais)
    - λ: Longitude estimada do alvo (graus decimais)
    - β = v * b: Viés do relógio / atraso fixo convertido em distância (km)

Modelo direto:
    r_i(θ) = d_i(φ, λ) + β - v * t_i

Jacobiana (m × 3):
    G[i, :] = [ ∂d_i/∂φ , ∂d_i/∂λ , 1.0 ]
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.geo import haversine, RAIO_TERRA_KM


def converter_servidores(servers):
    """
    Garante que a entrada de servidores seja um array numpy (m, 2) [lat, lon].
    Aceita lista de tuplas, lista de dicts (do CSV) ou ndarray.
    """
    if isinstance(servers, np.ndarray):
        return servers[:, :2].astype(float)
    elif isinstance(servers, list):
        if len(servers) == 0:
            return np.empty((0, 2), dtype=float)
        if isinstance(servers[0], dict):
            return np.array([[float(s['lat']), float(s['lon'])] for s in servers], dtype=float)
        else:
            return np.array(servers, dtype=float)[:, :2]
    raise ValueError("Formato de servidores não reconhecido. Use np.ndarray ou lista de coordenadas.")


def residuals(theta, servers, rtt_ms, v=100.0):
    """
    Calcula o vetor de resíduos r(θ) de tamanho (m,):
        r_i(θ) = d_i(φ, λ) + β - v * t_i

    Parâmetros:
        theta: array-like (3,) contendo [lat_alvo, lon_alvo, beta_km]
        servers: array-like (m, 2) contendo [lat_servidor, lon_servidor]
        rtt_ms: array-like (m,) contendo os tempos medidos em milissegundos
        v: velocidade de propagação efetiva na fibra (padrão: 100.0 km/ms)

    Retorna:
        r: np.ndarray de tamanho (m,) com os resíduos em km
    """
    lat_alvo = float(theta[0])
    lon_alvo = float(theta[1])
    beta_km = float(theta[2])

    srv = converter_servidores(servers)
    rtt = np.asarray(rtt_ms, dtype=float)

    # Distância ortodrômica de círculo máximo em km
    d = haversine(lat_alvo, lon_alvo, srv[:, 0], srv[:, 1])

    # Distância prevista pelo modelo de rede: v * t_i
    dist_medida = v * rtt

    # Resíduo: d_i + beta - (v * t_i)
    r = d + beta_km - dist_medida
    return r


def jacobian_analytical(theta, servers, v=100.0):
    """
    Calcula a matriz Jacobiana analítica G (m × 3) de r(θ):
        G[i, :] = [ ∂d_i/∂φ_graus, ∂d_i/∂λ_graus, 1.0 ]

    Dedução analítica na esfera unitária:
        Seja u o vetor unitário do alvo e u_i o vetor do servidor i.
        cos(ψ_i) = u · u_i, onde ψ_i = d_i / R.
        Pela regra da cadeia:
            ∂d_i/∂u = -R * u_i / sin(ψ_i)
        Projetando no plano tangente local (Norte e Leste) e convertendo para graus:
            ∂d_i/∂φ = -R / sin(ψ_i) * (u_i · e_Norte) * (π / 180)
            ∂d_i/∂λ = -R / sin(ψ_i) * (u_i · e_Leste * cos(φ)) * (π / 180)
            ∂r_i/∂β = 1.0
    """
    lat_alvo = float(theta[0])
    lon_alvo = float(theta[1])

    srv = converter_servidores(servers)
    m = srv.shape[0]

    phi = np.radians(lat_alvo)
    lam = np.radians(lon_alvo)
    phi_s = np.radians(srv[:, 0])
    lam_s = np.radians(srv[:, 1])

    # Vetores unitários dos servidores no espaço 3D (m, 3)
    u_s = np.column_stack([
        np.cos(phi_s) * np.cos(lam_s),
        np.cos(phi_s) * np.sin(lam_s),
        np.sin(phi_s)
    ])

    # Vetores base do plano tangente local no alvo
    e_norte = np.array([-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi)])
    e_leste = np.array([-np.sin(lam), np.cos(lam), 0.0])

    # Distância angular psi = d / R
    d = haversine(lat_alvo, lon_alvo, srv[:, 0], srv[:, 1])
    psi = d / RAIO_TERRA_KM

    # Proteção para alvo exatamente em cima do servidor (d -> 0)
    sin_psi = np.sin(psi)
    sin_psi = np.where(sin_psi < 1e-12, 1e-12, sin_psi)

    # Projeções do vetor servidor na base tangente
    proj_norte = np.dot(u_s, e_norte)
    proj_leste = np.dot(u_s, e_leste)

    fator_rad_graus = np.pi / 180.0

    # Derivadas parciais
    d_lat = -RAIO_TERRA_KM * (proj_norte / sin_psi) * fator_rad_graus
    d_lon = -RAIO_TERRA_KM * (proj_leste * np.cos(phi) / sin_psi) * fator_rad_graus

    G = np.zeros((m, 3), dtype=float)
    G[:, 0] = d_lat
    G[:, 1] = d_lon
    G[:, 2] = 1.0  # derivada em relação a beta é constante e igual a 1

    return G


def jacobian_finite_diff(theta, servers, v=100.0, h_deg=1e-6):
    """
    Calcula a Jacobiana por Diferenças Finitas Centrais:
        ∂r/∂φ ≈ [r(φ+h, λ, β) - r(φ-h, λ, β)] / (2h)
        ∂r/∂λ ≈ [r(φ, λ+h, β) - r(φ, λ-h, β)] / (2h)
        ∂r/∂β = 1.0 (exata)
    """
    srv = converter_servidores(servers)
    m = srv.shape[0]

    dummy_rtt = np.zeros(m)

    th_plus_lat = np.array([theta[0] + h_deg, theta[1], theta[2]])
    th_minus_lat = np.array([theta[0] - h_deg, theta[1], theta[2]])
    d_lat = (residuals(th_plus_lat, srv, dummy_rtt, v) - residuals(th_minus_lat, srv, dummy_rtt, v)) / (2.0 * h_deg)

    th_plus_lon = np.array([theta[0], theta[1] + h_deg, theta[2]])
    th_minus_lon = np.array([theta[0], theta[1] - h_deg, theta[2]])
    d_lon = (residuals(th_plus_lon, srv, dummy_rtt, v) - residuals(th_minus_lon, srv, dummy_rtt, v)) / (2.0 * h_deg)

    G = np.zeros((m, 3), dtype=float)
    G[:, 0] = d_lat
    G[:, 1] = d_lon
    G[:, 2] = 1.0
    return G


def jacobian(theta, servers, v=100.0, method="analytical"):
    """
    Interface unificada para cálculo da Jacobiana G(θ).
    Métodos disponíveis: 'analytical' (padrão) ou 'finite_diff'.
    """
    if method == "analytical":
        return jacobian_analytical(theta, servers, v)
    elif method == "finite_diff":
        return jacobian_finite_diff(theta, servers, v)
    else:
        raise ValueError(f"Método desconhecido: {method}. Escolha 'analytical' ou 'finite_diff'.")


if __name__ == "__main__":
    print("--- Teste rápido de model.py ---")
    alvo = np.array([-23.5505, -46.6333, 15.0])  # SP, beta = 15 km
    servidores = np.array([
        [-22.9068, -43.1729],  # Rio
        [-19.8691, -43.9664],  # BH
        [-25.4284, -49.2733],  # Curitiba
        [-15.7634, -47.8703]   # Brasília
    ])

    G_ana = jacobian(alvo, servidores, method="analytical")
    G_num = jacobian(alvo, servidores, method="finite_diff")

    erro_max = np.max(np.abs(G_ana - G_num))
    print(f"Jacobiana Analítica:\n{G_ana}")
    print(f"\nDiferença máxima Analítica vs Diferenças Finitas: {erro_max:.2e}")
