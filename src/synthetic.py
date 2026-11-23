#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geração de dados sintéticos para testes e validação numérica.

Este módulo usa RTT = 2d/v + b, enquanto gauss_newton() usa v_gn * t = d + β.
Para os dois baterem, chame gauss_newton() com v_gn = v/2
(velocidade_gauss_newton) e recupere o viés como b = β / v_gn.
"""

from typing import List, Optional, Tuple
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.geo import haversine, RAIO_TERRA_KM

SETOR_GEOMETRIA_RUIM_DEG = 20.0
FORMA_LOGNORMAL = 0.5


def velocidade_gauss_newton(v_km_per_ms: float = 100.0) -> float:
    """Velocidade a passar para gauss_newton(): v/2, por causa da ida e volta do RTT."""
    return v_km_per_ms / 2.0


def servers_to_array(servers: List[Tuple[float, float, str]]) -> np.ndarray:
    """
    Converte [(lat, lon, nome), ...] em array (m, 2).
    Necessário porque model.converter_servidores() não aceita tuplas com nome.
    """
    return np.array([[float(s[0]), float(s[1])] for s in servers], dtype=float)


def ponto_destino(
    lat: float, lon: float, azimute_deg: np.ndarray, dist_km: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Ponto alcançado ao andar dist_km pelo círculo máximo com o azimute dado (0° = Norte)."""
    phi1 = np.radians(lat)
    lam1 = np.radians(lon)
    theta = np.radians(azimute_deg)
    delta = np.asarray(dist_km, dtype=float) / RAIO_TERRA_KM

    phi2 = np.arcsin(
        np.sin(phi1) * np.cos(delta) + np.cos(phi1) * np.sin(delta) * np.cos(theta)
    )
    lam2 = lam1 + np.arctan2(
        np.sin(theta) * np.sin(delta) * np.cos(phi1),
        np.cos(delta) - np.sin(phi1) * np.sin(phi2),
    )

    lat2 = np.degrees(phi2)
    lon2 = (np.degrees(lam2) + 180.0) % 360.0 - 180.0
    return lat2, lon2


def generate_synthetic_network(
    target_lat: float,
    target_lon: float,
    n_servers: int = 8,
    radius_km: float = 2000.0,
    geometry: str = "good",
    seed: Optional[int] = 42,
) -> List[Tuple[float, float, str]]:
    """
    Gera coordenadas sintéticas de servidores distribuídos em torno do alvo.

    geometry:
        'good': azimutes uniformes em 360°, distâncias entre 60% e 100% do raio.
        'bad' : azimutes concentrados num setor de 20° (quase colineares vistos
                do alvo), distâncias entre 30% e 100% do raio.

    Retorna [(lat_i, lon_i, 'Servidor_i'), ...].
    """
    if n_servers < 3:
        raise ValueError("n_servers deve ser >= 3 (θ = [lat, lon, β] tem 3 incógnitas).")
    if radius_km <= 0:
        raise ValueError("radius_km deve ser positivo.")

    rng = np.random.default_rng(seed)

    if geometry == "good":
        # Pequena perturbação (±15% do espaçamento) para a rede não ficar perfeitamente regular
        espacamento = 360.0 / n_servers
        rotacao = rng.uniform(0.0, espacamento)
        jitter = rng.uniform(-0.15, 0.15, n_servers) * espacamento
        azimutes = rotacao + np.arange(n_servers) * espacamento + jitter
        distancias = rng.uniform(0.6, 1.0, n_servers) * radius_km
    elif geometry == "bad":
        # Linhas de visada quase paralelas deixam a Jacobiana mal condicionada
        azimute_central = rng.uniform(0.0, 360.0)
        azimutes = azimute_central + rng.uniform(
            -SETOR_GEOMETRIA_RUIM_DEG / 2.0, SETOR_GEOMETRIA_RUIM_DEG / 2.0, n_servers
        )
        distancias = rng.uniform(0.3, 1.0, n_servers) * radius_km
    else:
        raise ValueError(f"Geometria desconhecida: '{geometry}'. Use 'good' ou 'bad'.")

    lats, lons = ponto_destino(target_lat, target_lon, azimutes % 360.0, distancias)

    return [
        (float(lat), float(lon), f"Servidor_{i + 1:02d}")
        for i, (lat, lon) in enumerate(zip(lats, lons))
    ]


def simulate_rtt(
    servers: List[Tuple[float, float, str]],
    target_lat: float,
    target_lon: float,
    target_b_ms: float = 5.0,
    v_km_per_ms: float = 100.0,
    noise_std_ms: float = 0.0,
    noise_type: str = "gaussian",
    seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Simula o RTT (ms) de cada servidor: RTT_i = 2 d_i / v + b + ε_i.

    noise_type:
        'gaussian' : ε ~ N(0, σ²).
        'lognormal': ε > 0 e assimétrico (atraso de fila só aumenta o RTT).
                     Escalado para ter desvio padrão σ; a média é positiva.
    """
    if v_km_per_ms <= 0:
        raise ValueError("v_km_per_ms deve ser positivo.")
    if noise_std_ms < 0:
        raise ValueError("noise_std_ms não pode ser negativo.")
    if noise_type not in ("gaussian", "lognormal"):
        raise ValueError(
            f"Tipo de ruído desconhecido: '{noise_type}'. Use 'gaussian' ou 'lognormal'."
        )

    coords = servers_to_array(servers)
    d = haversine(target_lat, target_lon, coords[:, 0], coords[:, 1])
    rtt = 2.0 * d / v_km_per_ms + target_b_ms

    if noise_std_ms > 0:
        rng = np.random.default_rng(seed)
        m = len(rtt)
        if noise_type == "gaussian":
            ruido = rng.normal(0.0, noise_std_ms, m)
        else:
            s = FORMA_LOGNORMAL
            # Desvio padrão da lognormal(0, s), usado para reescalar para σ
            std_base = np.sqrt((np.exp(s ** 2) - 1.0) * np.exp(s ** 2))
            ruido = rng.lognormal(0.0, s, m) * (noise_std_ms / std_base)
        rtt = rtt + ruido

    return rtt


if __name__ == "__main__":
    alvo = (-23.5505, -46.6333)
    for geo in ("good", "bad"):
        rede = generate_synthetic_network(*alvo, n_servers=8, radius_km=2500.0, geometry=geo)
        rtt = simulate_rtt(rede, *alvo, noise_std_ms=1.0)
        print(f"\nGeometria '{geo}':")
        for (lat, lon, nome), t in zip(rede, rtt):
            d = haversine(alvo[0], alvo[1], lat, lon)
            print(f"  {nome}: ({lat:8.3f}, {lon:8.3f})  d = {d:7.1f} km  RTT = {t:6.2f} ms")
