#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experimento 1: convergência e raio de atração do Gauss-Newton.

Alvo em São Paulo, 8 servidores sintéticos ('good', raio 2500 km, σ = 1 ms)
e quatro chutes iniciais (Campinas, Brasília, Manaus, Londres).

Observações:
    - RTT = 2d/v + b, então o Gauss-Newton é chamado com v_gn = v/2 e b = β / v_gn.
    - ||δ_k|| mistura graus (Δlat, Δlon) com km (Δβ); nas primeiras
      iterações o termo Δβ domina a norma.

Uso: python experiments/exp01_convergencia.py
"""

import csv
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, RAIZ)

from src.geo import haversine
from src.gauss_newton import gauss_newton, GaussNewtonResult
from src.synthetic import (
    generate_synthetic_network,
    simulate_rtt,
    servers_to_array,
    velocidade_gauss_newton,
)

LAT_REAL: float = -23.5505
LON_REAL: float = -46.6333
B_REAL_MS: float = 5.0

N_SERVIDORES: int = 8
RAIO_KM: float = 2500.0
GEOMETRIA: str = "good"
V_KM_POR_MS: float = 100.0
SIGMA_MS: float = 1.0
SEED: int = 42

TOL_X: float = 1e-3
SOLVER: str = "qr"

CHUTES: List[Tuple[str, float, float]] = [
    ("Campinas (proximo)", -22.9099, -47.0626),
    ("Brasilia (medio)", -15.7975, -47.8919),
    ("Manaus (distante)", -3.1190, -60.0217),
    ("Londres (extremo)", 51.5074, -0.1278),
]
CORES: List[str] = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]

DIR_RESULTADOS: str = os.path.join(RAIZ, "results")


def executar_chute(
    servidores: np.ndarray, rtt: np.ndarray, nome: str, lat0: float, lon0: float
) -> Dict[str, Any]:
    v_gn = velocidade_gauss_newton(V_KM_POR_MS)
    theta0 = np.array([lat0, lon0, 0.0])

    res: GaussNewtonResult = gauss_newton(
        servidores, rtt, theta0, v=v_gn, solver=SOLVER, tol_x=TOL_X
    )

    # history guarda θ_k antes de cada passo; o θ final é acrescentado no fim
    trajetoria = np.array([h["theta"] for h in res.history] + [res.theta])

    return {
        "nome": nome,
        "lat0": lat0,
        "lon0": lon0,
        "dist_inicial_km": float(haversine(LAT_REAL, LON_REAL, lat0, lon0)),
        "resultado": res,
        "trajetoria": trajetoria,
        "erro_final_km": float(haversine(LAT_REAL, LON_REAL, res.lat, res.lon)),
        "residuo_final": float(np.linalg.norm(res.final_residuals)),
        "b_estimado_ms": float(res.beta_km / v_gn),
    }


def salvar_resumo_csv(execucoes: List[Dict[str, Any]], caminho: str) -> None:
    campos = ["chute_nome", "lat0", "lon0", "dist_inicial_km", "convergiu",
              "iteracoes", "erro_final_km", "residuo_final"]
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(campos)
        for e in execucoes:
            r = e["resultado"]
            w.writerow([
                e["nome"], f"{e['lat0']:.4f}", f"{e['lon0']:.4f}",
                f"{e['dist_inicial_km']:.1f}", r.converged, r.n_iter,
                f"{e['erro_final_km']:.3f}", f"{e['residuo_final']:.4f}",
            ])


def salvar_historico_csv(execucoes: List[Dict[str, Any]], caminho: str) -> None:
    campos = ["chute_nome", "k", "lat_k", "lon_k", "beta_k_km",
              "residuo_norma_km", "passo_norma", "cond_G"]
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(campos)
        for e in execucoes:
            for h in e["resultado"].history:
                th = h["theta"]
                w.writerow([
                    e["nome"], h["iter"], f"{th[0]:.6f}", f"{th[1]:.6f}",
                    f"{th[2]:.4f}", f"{h['res_norm']:.6e}",
                    f"{h['step_norm']:.6e}", f"{h['cond_G']:.4f}",
                ])


def grafico_norma_passo(execucoes: List[Dict[str, Any]], caminho: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for e, cor in zip(execucoes, CORES):
        r = e["resultado"]
        ks = [h["iter"] for h in r.history]
        passos = [h["step_norm"] for h in r.history]
        status = "convergiu" if r.converged else "divergiu"
        rotulo = f"{e['nome']} — {e['dist_inicial_km']:.0f} km ({status}, {r.n_iter} it.)"
        estilo = "-" if r.converged else ":"
        ax.semilogy(ks, passos, estilo, marker="o", color=cor, lw=2, label=rotulo)
        if not r.converged:
            ax.plot(ks[-1], passos[-1], "x", color=cor, ms=12, mew=3)

    ax.axhline(TOL_X, color="0.3", ls="--", lw=1.2, label=f"tol_x = {TOL_X:g}")
    ax.set_xlabel("Iteração k")
    ax.set_ylabel(r"$\|\delta_k\|$  ([Δlat°, Δlon°, Δβ km])")
    ax.set_title(
        "Exp01 — Norma do passo de Gauss-Newton por chute inicial\n"
        f"Alvo: São Paulo | {N_SERVIDORES} servidores ('{GEOMETRIA}', ~{RAIO_KM:.0f} km) | "
        f"σ = {SIGMA_MS:g} ms",
        fontsize=11,
    )
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def _desenhar_trajetorias(ax, servidores: np.ndarray, execucoes: List[Dict[str, Any]],
                          com_legenda: bool) -> None:
    ax.scatter(servidores[:, 1], servidores[:, 0], marker="^", s=70, c="0.35",
               edgecolors="k", zorder=3, label="Servidores")
    for e, cor in zip(execucoes, CORES):
        tr = e["trajetoria"]
        estilo = "-" if e["resultado"].converged else ":"
        ax.plot(tr[:, 1], tr[:, 0], estilo, marker="o", ms=4, color=cor, lw=1.8,
                zorder=4, label=e["nome"])
        ax.scatter(tr[0, 1], tr[0, 0], marker="s", s=60, color=cor,
                   edgecolors="k", zorder=5)
    ax.scatter(LON_REAL, LAT_REAL, marker="*", s=320, c="gold", edgecolors="k",
               zorder=6, label="Alvo real (São Paulo)")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.grid(True, alpha=0.3)
    if com_legenda:
        ax.legend(fontsize=8, loc="best")


def grafico_trajetoria(servidores: np.ndarray, execucoes: List[Dict[str, Any]],
                       caminho: str) -> None:
    fig, (ax_g, ax_z) = plt.subplots(1, 2, figsize=(15, 6.5))

    _desenhar_trajetorias(ax_g, servidores, execucoes, com_legenda=True)
    ax_g.set_title("Visão global (quadrado = chute inicial)")

    _desenhar_trajetorias(ax_z, servidores, execucoes, com_legenda=False)
    margem = 4.0
    ax_z.set_xlim(servidores[:, 1].min() - margem, servidores[:, 1].max() + margem)
    ax_z.set_ylim(servidores[:, 0].min() - margem, servidores[:, 0].max() + margem)
    ax_z.set_aspect("equal", adjustable="box")
    ax_z.set_title("Zoom na região da rede de servidores")

    fig.suptitle("Exp01 — Trajetórias do Gauss-Newton a partir de cada chute inicial",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def main() -> None:
    os.makedirs(DIR_RESULTADOS, exist_ok=True)

    rede = generate_synthetic_network(LAT_REAL, LON_REAL, n_servers=N_SERVIDORES,
                                      radius_km=RAIO_KM, geometry=GEOMETRIA, seed=SEED)
    rtt = simulate_rtt(rede, LAT_REAL, LON_REAL, target_b_ms=B_REAL_MS,
                       v_km_per_ms=V_KM_POR_MS, noise_std_ms=SIGMA_MS,
                       noise_type="gaussian", seed=SEED)
    servidores = servers_to_array(rede)

    print("=== Exp01 — Convergência do Gauss-Newton ===")
    print(f"Alvo: ({LAT_REAL}, {LON_REAL}), b* = {B_REAL_MS} ms | "
          f"{N_SERVIDORES} servidores '{GEOMETRIA}', raio {RAIO_KM:.0f} km, "
          f"σ = {SIGMA_MS} ms, seed = {SEED}\n")

    execucoes = [executar_chute(servidores, rtt, nome, la, lo) for nome, la, lo in CHUTES]

    for e in execucoes:
        r = e["resultado"]
        print(f"- {e['nome']:<20} d0 = {e['dist_inicial_km']:7.1f} km | "
              f"{'CONVERGIU' if r.converged else 'NAO CONVERGIU':<13} | "
              f"{r.n_iter:2d} it. | erro = {e['erro_final_km']:9.3f} km | "
              f"||r|| = {e['residuo_final']:10.3f} km | b = {e['b_estimado_ms']:8.3f} ms")
        print(f"    motivo: {r.reason}")

    arquivos = {
        "norma": os.path.join(DIR_RESULTADOS, "exp01_convergencia_norma_passo.png"),
        "resumo": os.path.join(DIR_RESULTADOS, "exp01_resumo.csv"),
        "historico": os.path.join(DIR_RESULTADOS, "exp01_historico.csv"),
        "mapa": os.path.join(DIR_RESULTADOS, "exp01_trajetoria_mapa.png"),
    }
    grafico_norma_passo(execucoes, arquivos["norma"])
    salvar_resumo_csv(execucoes, arquivos["resumo"])
    salvar_historico_csv(execucoes, arquivos["historico"])
    grafico_trajetoria(servidores, execucoes, arquivos["mapa"])

    print("\nArquivos gerados:")
    for caminho in arquivos.values():
        print(f"  {os.path.relpath(caminho, RAIZ)}")


if __name__ == "__main__":
    main()
