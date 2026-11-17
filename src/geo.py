#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo Geográfico (geo.py) — Cálculo Numérico
Contém as funções de trigonometria esférica: fórmula de Haversine, 
conversões de coordenadas e vetores unitários.

Convenções do projeto:
- Raio médio da Terra: R = 6371.0 km (modelo esférico)
- Coordenadas de entrada da interface: graus decimais (lat, lon)
- Ângulos internos para cálculo: radianos
- Distâncias retornadas: quilômetros (km)
"""

import numpy as np

# Raio médio da Terra em quilômetros (conforme seção 2.1 e 3.4 do SDD)
RAIO_TERRA_KM = 6371.0


def deg2rad(graus):
    """Converte graus decimais para radianos."""
    return np.radians(graus)


def rad2deg(rad):
    """Converte radianos para graus decimais."""
    return np.degrees(rad)


def haversine(lat1, lon1, lat2, lon2, r=RAIO_TERRA_KM):
    """
    Calcula a distância ortodrômica (círculo máximo) entre dois pontos na esfera.
    Utiliza a fórmula numéricamente estável do Haversine (via arctan2).

    Parâmetros:
        lat1, lon1: Latitude e longitude do ponto 1 (em graus decimais).
                    Podem ser escalares (float) ou arrays numpy.
        lat2, lon2: Latitude e longitude do ponto 2 (em graus decimais).
                    Podem ser escalares (float) ou arrays numpy (ex: lista de servidores).
        r: Raio da esfera em km (padrão: 6371.0 km).

    Retorna:
        Distância em quilômetros (float ou np.ndarray).
    """
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    # Termo do Haversine: a = sin²(Δφ/2) + cos(φ1)*cos(φ2)*sin²(Δλ/2)
    a = (np.sin(delta_phi / 2.0) ** 2 +
         np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2)

    # Evita inconsistências de ponto flutuante fora do domínio [0, 1]
    a = np.clip(a, 0.0, 1.0)

    # Distância angular central em radianos: c = 2 * atan2(√a, √(1-a))
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    return r * c


def latlon_to_unit_vector(lat, lon):
    """
    Converte latitude e longitude (em graus) para vetor unitário cartesiano 3D na esfera:
    u = [cos(φ)*cos(λ), cos(φ)*sin(λ), sin(φ)]

    Útil para geometria de visada e cálculo de vetores no plano tangente (DOP).
    """
    phi = np.radians(lat)
    lam = np.radians(lon)

    x = np.cos(phi) * np.cos(lam)
    y = np.cos(phi) * np.sin(lam)
    z = np.sin(phi)

    return np.array([x, y, z])


def azimute_inicial(lat1, lon1, lat2, lon2):
    """
    Calcula o azimute inicial (bearing) de navegação do ponto 1 para o ponto 2 em graus [0°, 360°].
    0° = Norte, 90° = Leste, 180° = Sul, 270° = Oeste.
    """
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_lambda = np.radians(lon2 - lon1)

    y = np.sin(delta_lambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_lambda)

    theta = np.arctan2(y, x)
    graus = (np.degrees(theta) + 360.0) % 360.0
    return graus


if __name__ == "__main__":
    print("--- Teste rápido de geo.py ---")
    sp_lat, sp_lon = -23.5505, -46.6333
    rj_lat, rj_lon = -22.9068, -43.1729
    lisboa_lat, lisboa_lon = 38.7223, -9.1393

    d_sp_rj = haversine(sp_lat, sp_lon, rj_lat, rj_lon)
    d_sp_lisboa = haversine(sp_lat, sp_lon, lisboa_lat, lisboa_lon)

    print(f"Distância SP -> Rio de Janeiro : {d_sp_rj:.2f} km (esperado ~360 km)")
    print(f"Distância SP -> Lisboa         : {d_sp_lisboa:.2f} km (esperado ~7915 km)")
    print(f"Distância SP -> SP             : {haversine(sp_lat, sp_lon, sp_lat, sp_lon):.6f} km (esperado 0 km)")
