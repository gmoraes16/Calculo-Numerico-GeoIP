#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Diluição de Precisão — DOP (dop.py) — Cálculo Numérico
Implementa o cálculo das métricas geométricas de condicionamento e propagação
de erro herdadas do GPS:
    Q = (G^T G)^(-1)
    GDOP = sqrt(trace(Q))           # posição (Norte, Leste) + viés β
    PDOP = sqrt(Q_11 + Q_22)        # apenas posição 2D (Norte, Leste)
    TDOP = sqrt(Q_33)               # apenas atraso fixo β
    κ_2(G) = σ_max(G) / σ_min(G)    # número de condição da Jacobiana

Relação teórica esperada (σ em km):
    erro_RMS_posição ≈ σ * PDOP

Observação de escala (SDD Seções 2.4 e 2.6):
    Para que PDOP seja adimensional (km/km) e satisfaça erro_RMS ≈ σ_km * PDOP,
    a Jacobiana G deve estar expressa no plano tangente local em km:
        G_km[i, :] = [ -u_{i,norte}, -u_{i,leste}, 1.0 ]
    onde u_i é o vetor unitário do alvo em direção ao servidor i.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.geo import RAIO_TERRA_KM
from src.model import jacobian_analytical, converter_servidores
from src.solvers import cholesky, forward_substitution, back_substitution, cond2


def _inverter_spd_cholesky(A):
    """
    Calcula a inversa de uma matriz simétrica positiva definida A (n × n)
    usando exclusivamente a nossa fatoração de Cholesky própria:
        Resolve A * q_j = e_j para cada coluna j da matriz identidade.
    """
    n = A.shape[0]
    L = cholesky(A)
    Q = np.zeros((n, n), dtype=np.float64)

    for j in range(n):
        e_j = np.zeros(n, dtype=np.float64)
        e_j[j] = 1.0
        y = forward_substitution(L, e_j)
        Q[:, j] = back_substitution(L.T, y)

    return Q


def jacobian_tangent_km(lat_alvo, lon_alvo, servers):
    """
    Constrói a Jacobiana G_km (m × 3) no plano tangente local em quilômetros.

    Converte as derivadas de model.py (em km/grau) para km/km:
        ∂d/∂y_norte = (∂d/∂φ_graus) / (R * π / 180)
        ∂d/∂x_leste = (∂d/∂λ_graus) / (R * cos(φ) * π / 180)
        ∂r/∂β       = 1.0

    Assim, cada linha tem norma 1 nas duas primeiras colunas:
        G_km[i, :] = [ -u_{i,norte}, -u_{i,leste}, 1.0 ]
    deixando todas as colunas na mesma escala física (adimensional).
    """
    srv = converter_servidores(servers)
    theta = np.array([float(lat_alvo), float(lon_alvo), 0.0])
    G_deg = jacobian_analytical(theta, srv)

    km_por_grau_lat = RAIO_TERRA_KM * (np.pi / 180.0)
    cos_phi = np.cos(np.radians(lat_alvo))
    km_por_grau_lon = km_por_grau_lat * max(abs(cos_phi), 1e-12)

    G_km = np.empty_like(G_deg, dtype=np.float64)
    G_km[:, 0] = G_deg[:, 0] / km_por_grau_lat   # direção Norte (-u_norte)
    G_km[:, 1] = G_deg[:, 1] / km_por_grau_lon   # direção Leste (-u_leste)
    G_km[:, 2] = 1.0                              # coluna do atraso fixo β

    return G_km


def dop_metrics(G):
    """
    Calcula as métricas de Diluição de Precisão (GDOP, PDOP, TDOP) e o número
    de condição κ_2(G) a partir de uma matriz Jacobiana G (m × 3).

    Parâmetros:
        G: Matriz Jacobiana (m × 3) em coordenadas consistentes (ex.: G_km).

    Retorna:
        dict com chaves: 'gdop', 'pdop', 'tdop', 'cond2'.
        Caso G^T G seja singular ou numericamente não-positiva definida,
        retorna np.inf para as métricas de DOP.
    """
    G_arr = np.asarray(G, dtype=np.float64)
    if G_arr.ndim != 2 or G_arr.shape[1] != 3:
        raise ValueError(f"G deve ter shape (m, 3); recebido {G_arr.shape}.")

    k2 = cond2(G_arr)
    A = np.dot(G_arr.T, G_arr)

    try:
        Q = _inverter_spd_cholesky(A)
        diag_pos = Q[0, 0] + Q[1, 1]
        diag_tempo = Q[2, 2]
        traco = diag_pos + diag_tempo

        if diag_pos < 0 or diag_tempo < 0 or traco < 0:
            raise np.linalg.LinAlgError("Variância negativa por mal condicionamento extremo.")

        pdop = float(np.sqrt(diag_pos))
        tdop = float(np.sqrt(diag_tempo))
        gdop = float(np.sqrt(traco))
    except np.linalg.LinAlgError:
        pdop = float(np.inf)
        tdop = float(np.inf)
        gdop = float(np.inf)

    return {
        "gdop": gdop,
        "pdop": pdop,
        "tdop": tdop,
        "cond2": k2,
    }


def dop_from_geometry(lat_alvo, lon_alvo, servers):
    """
    Atalho prático: calcula G_km no plano tangente local do alvo e retorna
    o dicionário de métricas DOP (gdop, pdop, tdop, cond2).
    """
    G_km = jacobian_tangent_km(lat_alvo, lon_alvo, servers)
    return dop_metrics(G_km)


if __name__ == "__main__":
    print("--- Teste rápido de dop.py ---")
    # Caso analítico: 4 servidores simétricos ao redor do Equador (N, S, L, O)
    servidores_sim = np.array([
        [5.0, 0.0],   # Norte
        [-5.0, 0.0],  # Sul
        [0.0, 5.0],   # Leste
        [0.0, -5.0],  # Oeste
    ])
    m = dop_from_geometry(0.0, 0.0, servidores_sim)
    print(f"4 servidores simétricos -> PDOP={m['pdop']:.4f} (esp. 1.0000), "
          f"GDOP={m['gdop']:.4f} (esp. 1.1180), κ_2={m['cond2']:.4f} (esp. 1.4142)")

    # Caso real: servidores de data/servers.csv ao redor de São Paulo
    import csv
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "servers.csv")
    if os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8") as f:
            srvs = [[float(r["lat"]), float(r["lon"])] for r in csv.DictReader(f)]
        m_real = dop_from_geometry(-23.5505, -46.6333, srvs)
        print(f"13 servidores reais (SP) -> PDOP={m_real['pdop']:.4f}, "
              f"GDOP={m_real['gdop']:.4f}, TDOP={m_real['tdop']:.4f}, κ_2={m_real['cond2']:.4f}")
