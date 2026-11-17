#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo Gauss-Newton (gauss_newton.py) — Cálculo Numérico
Implementa o método iterativo de Gauss-Newton para resolução do problema
de mínimos quadrados não-linear:
    min_θ  (1/2) * ||r(θ)||_2^2

A cada iteração k:
    1. Calcula r(θ_k) e a Jacobiana G = ∂r/∂θ
    2. Resolve o subproblema linear min_δ ||G δ + r||_2 (via QR ou Normais)
    3. Atualiza o parâmetro: θ_{k+1} = θ_k + δ
    4. Avalia os critérios rigorosos de parada e divergência.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import os
import sys
import numpy as np

# Garante importação tanto como pacote quanto script direto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from src.model import residuals, jacobian, converter_servidores
from src.solvers import solve_normal, solve_qr, cond2


@dataclass
class GaussNewtonResult:
    """Estrutura com os resultados completos da execução de Gauss-Newton."""
    theta: np.ndarray                 # Vetor final [lat, lon, beta]
    converged: bool                   # Se convergiu dentro das tolerâncias
    n_iter: int                       # Número de iterações executadas
    reason: str                       # Motivo de parada (ex: "tol_x atingida")
    history: List[Dict[str, Any]]     # Histórico de cada passo (θ_k, ||r||, ||δ||, κ)
    final_residuals: np.ndarray       # Resíduos finais r(θ_final) em km
    cond_G: float                     # Número de condição κ_2(G) final
    solver_used: str                  # Solver utilizado ('qr' ou 'normal')

    @property
    def lat(self) -> float:
        return float(self.theta[0])

    @property
    def lon(self) -> float:
        return float(self.theta[1])

    @property
    def beta_km(self) -> float:
        return float(self.theta[2])

    @property
    def b_ms(self) -> float:
        """Converte atraso fixo de km de volta para milissegundos (v = 100 km/ms)."""
        return float(self.theta[2] / 100.0)

    def __str__(self) -> str:
        status = "CONVERGIU" if self.converged else "NÃO CONVERGIU"
        return (
            f"=== Resultado Gauss-Newton ({status}) ===\n"
            f"  Posicao estimada : Lat {self.lat:.5f} deg, Lon {self.lon:.5f} deg\n"
            f"  Vies beta        : {self.beta_km:.2f} km ({self.b_ms:.2f} ms)\n"
            f"  Iteracoes        : {self.n_iter}\n"
            f"  Motivo de parada : {self.reason}\n"
            f"  Norma do residuo : {np.linalg.norm(self.final_residuals):.4f} km\n"
            f"  Condicionamento  : kappa_2(G) = {self.cond_G:.2f}\n"
            f"  Solver           : {self.solver_used.upper()}"
        )


def gauss_newton(
    servers,
    rtt_ms,
    theta0,
    v=100.0,
    solver="qr",
    dtype=np.float64,
    tol_x=1e-3,       # Passo pequeno: 1e-3 km (1 metro)
    tol_f=1e-8,       # Estagnação da norma do resíduo
    max_iter=50,      # Limite de iterações
    max_step_deg=15.0, # Limite máximo de passo angular por iteração (evita ultrapassagem na esfera)
    verbose=False
) -> GaussNewtonResult:
    """
    Executa o laço iterativo de Gauss-Newton.

    Parâmetros:
        servers: array-like (m, 2) com [lat, lon] dos servidores
        rtt_ms: array-like (m,) com as latências RTT medidas em milissegundos
        theta0: chute inicial [lat_0, lon_0, beta_0]
        v: velocidade de rede em km/ms (padrão: 100.0 km/ms)
        solver: método linear ('qr' para Householder ou 'normal' para Cholesky)
        dtype: np.float64 ou np.float32 (para experimentos de precisão)
        tol_x: tolerância para o tamanho do passo ||δ_k|| (km)
        tol_f: tolerância para estagnação relativa do resíduo | ||r_k|| - ||r_{k-1}|| | / ||r_k||
        max_iter: número máximo de iterações
        verbose: se True, imprime detalhes a cada iteração

    Retorna:
        GaussNewtonResult contendo a solução, flags de convergência e histórico.
    """
    srv = converter_servidores(servers)
    rtt = np.asarray(rtt_ms, dtype=dtype)
    theta = np.asarray(theta0, dtype=dtype).copy()

    history = []
    converged = False
    reason = "Limite max_iter atingido"
    crescimento_resíduo = 0

    # Resíduo inicial
    r_k = residuals(theta, srv, rtt, v=v).astype(dtype)
    norm_r_k = float(np.linalg.norm(r_k))

    if verbose:
        print(f"\n[GN] Inicio com theta_0 = {theta}, ||r_0|| = {norm_r_k:.4f} km (Solver: {solver.upper()})")

    k = 0
    while k < max_iter:
        # 1. Calcula a Jacobiana no ponto atual
        G_k = jacobian(theta, srv, v=v).astype(dtype)
        cond_G = cond2(G_k)

        # 2. Resolve o problema de mínimos quadrados linear para o passo delta
        if solver == "qr":
            delta_k = solve_qr(G_k, r_k, dtype=dtype)
        elif solver == "normal":
            delta_k = solve_normal(G_k, r_k, dtype=dtype)
        else:
            raise ValueError(f"Solver desconhecido: '{solver}'. Escolha 'qr' ou 'normal'.")

        norm_delta_k = float(np.linalg.norm(delta_k))

        # Registra histórico
        history.append({
            "iter": k + 1,
            "theta": theta.copy(),
            "res_norm": norm_r_k,
            "step_norm": norm_delta_k,
            "cond_G": cond_G
        })

        if verbose:
            print(
                f"[GN] Iter {k+1:02d}: ||r||={norm_r_k:10.4f} km, "
                f"||delta||={norm_delta_k:10.6f}, kappa_2(G)={cond_G:7.2f}, "
                f"theta=[{theta[0]:.4f}, {theta[1]:.4f}, {theta[2]:.2f}]"
            )

        # 3. Atualiza parâmetro: theta_{k+1} = theta_k + delta
        # Limita passo angular para evitar instabilidade trigonométrica na esfera
        if max_step_deg is not None and max_step_deg > 0:
            delta_k[0] = np.clip(delta_k[0], -max_step_deg, max_step_deg)
            delta_k[1] = np.clip(delta_k[1], -max_step_deg, max_step_deg)

        theta = theta + delta_k

        # Normaliza coordenadas na esfera terrestre
        theta[0] = np.clip(theta[0], -89.9, 89.9)
        theta[1] = (theta[1] + 180.0) % 360.0 - 180.0
        k += 1

        # 4. Avalia novo resíduo
        r_novo = residuals(theta, srv, rtt, v=v).astype(dtype)
        norm_r_novo = float(np.linalg.norm(r_novo))

        # 5. Verificação de Critérios de Parada (SDD Seção 2.5)

        # Critério 1: Passo pequeno ||delta_k|| < tol_x
        if norm_delta_k < tol_x:
            converged = True
            reason = f"Convergencia: ||delta|| = {norm_delta_k:.2e} < tol_x ({tol_x})"
            norm_r_k = norm_r_novo
            r_k = r_novo
            break

        # Critério 2: Estagnação do resíduo | ||r_k|| - ||r_{k-1}|| | / ||r_k|| < tol_f
        mudanca_relativa = abs(norm_r_novo - norm_r_k) / max(norm_r_k, 1e-12)
        if mudanca_relativa < tol_f:
            converged = True
            reason = f"Estagnacao do residuo: Delta||r||/||r|| = {mudanca_relativa:.2e} < tol_f"
            norm_r_k = norm_r_novo
            r_k = r_novo
            break

        # Critério 3: Detecção de divergência (resíduo crescendo por 3 passos seguidos)
        if norm_r_novo > norm_r_k:
            crescimento_resíduo += 1
            if crescimento_resíduo >= 3:
                converged = False
                reason = "Divergência detectada: ||r|| cresceu por 3 iterações consecutivas"
                norm_r_k = norm_r_novo
                r_k = r_novo
                break
        else:
            crescimento_resíduo = 0

        norm_r_k = norm_r_novo
        r_k = r_novo

    # Jacobiana final para condicionamento
    G_final = jacobian(theta, srv, v=v).astype(dtype)
    cond_final = cond2(G_final)

    return GaussNewtonResult(
        theta=theta,
        converged=converged,
        n_iter=k,
        reason=reason,
        history=history,
        final_residuals=r_k,
        cond_G=cond_final,
        solver_used=solver
    )


if __name__ == "__main__":
    print("--- Teste rápido de gauss_newton.py com dados sintéticos ---")
    # Alvo real: São Paulo Marco Zero
    lat_real, lon_real, beta_real = -23.5505, -46.6333, 15.0
    theta_real = np.array([lat_real, lon_real, beta_real])

    # 6 servidores reais
    servidores = np.array([
        [-22.8144, -47.0647],  # Unicamp
        [-22.8631, -43.2307],  # UFRJ
        [-19.8691, -43.9664],  # UFMG
        [-30.0700, -51.1200],  # UFRGS
        [-15.7634, -47.8703],  # UnB
        [-3.7319, -38.5267]    # Fortaleza
    ])

    v_rede = 100.0
    from src.geo import haversine
    distancias_reais = haversine(lat_real, lon_real, servidores[:, 0], servidores[:, 1])
    rtt_sintetico = (distancias_reais + beta_real) / v_rede

    # Chute inicial distante (ex: Brasília, ~870 km de SP)
    chute_inicial = np.array([-15.76, -47.87, 0.0])

    res = gauss_newton(servidores, rtt_sintetico, chute_inicial, solver="qr", verbose=True)
    print("\n", res)
    erro_km = haversine(res.lat, res.lon, lat_real, lon_real)
    print(f"Erro em relação ao alvo real: {erro_km:.6f} km")
