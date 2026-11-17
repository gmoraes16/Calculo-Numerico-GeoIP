#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Solvers Numéricos (solvers.py) — Cálculo Numérico
Implementação do zero dos métodos de álgebra linear para mínimos quadrados:
1. Fatoração de Cholesky (A = L L^T)
2. Resolução de sistemas triangulares (substituição progressiva e regressiva)
3. Resolução via Equações Normais: (G^T G) δ = -G^T r
4. Fatoração QR via reflexões de Householder (G = Q R)
5. Resolução via QR: R δ = -Q^T r

Ambos os métodos suportam precisão simples (np.float32) e dupla (np.float64)
para viabilizar os experimentos de condicionamento e perda de precisão (Exp02).
"""

import numpy as np


# ==============================================================================
# 1. SUBSTITUIÇÃO PROGRESSIVA E REGRESSIVA
# ==============================================================================

def forward_substitution(L, b):
    """
    Resolve o sistema triangular inferior L y = b por substituição progressiva (para frente).
    L: matriz triangular inferior (n × n) com diagonal não nula
    b: vetor de termos independentes (n,)
    Retorna: vetor solução y (n,)
    """
    n = L.shape[0]
    y = np.zeros(n, dtype=L.dtype)

    for i in range(n):
        soma = np.dot(L[i, :i], y[:i])
        if abs(L[i, i]) < 1e-15 and L.dtype == np.float64:
            raise np.linalg.LinAlgError(f"Pivô nulo ou quase nulo na posição ({i},{i})")
        y[i] = (b[i] - soma) / L[i, i]

    return y


def back_substitution(U, b):
    """
    Resolve o sistema triangular superior U x = b por substituição regressiva (para trás).
    U: matriz triangular superior (n × n) com diagonal não nula
    b: vetor de termos independentes (n,)
    Retorna: vetor solução x (n,)
    """
    n = U.shape[0]
    x = np.zeros(n, dtype=U.dtype)

    for i in range(n - 1, -1, -1):
        soma = np.dot(U[i, i + 1:], x[i + 1:])
        if abs(U[i, i]) < 1e-15 and U.dtype == np.float64:
            raise np.linalg.LinAlgError(f"Pivô nulo ou quase nulo na posição ({i},{i})")
        x[i] = (b[i] - soma) / U[i, i]

    return x


# ==============================================================================
# 2. FATORAÇÃO DE CHOLESKY (A = L L^T)
# ==============================================================================

def cholesky(A):
    """
    Fatoração de Cholesky de matriz simétrica e positiva definida:
        A = L L^T
    onde L é uma matriz triangular inferior com diagonal estritamente positiva.

    Algoritmo de Banachiewicz (padrão em Cálculo Numérico):
        L_jj = sqrt(A_jj - sum_{k=0}^{j-1} L_jk^2)
        L_ij = (A_ij - sum_{k=0}^{j-1} L_ik * L_jk) / L_jj,  para i > j
    """
    n = A.shape[0]
    L = np.zeros((n, n), dtype=A.dtype)

    for j in range(n):
        # Elemento da diagonal principal
        soma_diag = np.dot(L[j, :j], L[j, :j])
        termo = A[j, j] - soma_diag

        if termo <= 0.0:
            raise np.linalg.LinAlgError(
                f"Matriz não é positiva definida na posição ({j},{j}): termo={termo}"
            )

        L[j, j] = np.sqrt(termo)

        # Elementos abaixo da diagonal (coluna j)
        for i in range(j + 1, n):
            soma_col = np.dot(L[i, :j], L[j, :j])
            L[i, j] = (A[i, j] - soma_col) / L[j, j]

    return L


def solve_cholesky(A, b):
    """
    Resolve o sistema linear A x = b com A simétrica e positiva definida
    utilizando a fatoração de Cholesky:
        1. A = L L^T
        2. L y = b  (substituição progressiva)
        3. L^T x = y (substituição regressiva)
    """
    L = cholesky(A)
    y = forward_substitution(L, b)
    x = back_substitution(L.T, y)
    return x


# ==============================================================================
# 3. EQUAÇÕES NORMAIS
# ==============================================================================

def solve_normal(G, r, dtype=np.float64):
    """
    Resolve o problema de mínimos quadrados linear:
        min_δ ||G δ + r||_2
    via Equações Normais:
        (G^T G) δ = -G^T r
    utilizando a fatoração de Cholesky.

    Parâmetros:
        G: Matriz Jacobiana (m × n), m >= n
        r: Vetor de resíduos (m,)
        dtype: np.float64 (dupla) ou np.float32 (simples)

    Retorna:
        δ: Vetor passo de atualização (n,)
    """
    G_conv = np.asarray(G, dtype=dtype)
    r_conv = np.asarray(r, dtype=dtype)

    # Montagem do sistema normal: A δ = b
    A = np.dot(G_conv.T, G_conv)
    b = -np.dot(G_conv.T, r_conv)

    delta = solve_cholesky(A, b)
    return delta


# ==============================================================================
# 4. FATORAÇÃO QR VIA REFLEXÕES DE HOUSEHOLDER
# ==============================================================================

def householder_qr(A):
    """
    Fatoração QR de matriz retangular A (m × n, m >= n) via reflexões de Householder:
        A = Q R
    onde Q é ortogonal (m × m) e R é triangular superior (m × n ou thin n × n).

    A reflexão ortogonal H = I - 2 * v v^T zera as entradas abaixo do pivô
    sem elevar o número de condição ao quadrado (estabilidade numérica superior).
    """
    m, n = A.shape
    R = A.copy()
    Q = np.eye(m, dtype=A.dtype)

    for k in range(min(n, m - 1)):
        # Vetor x a ser refletido na coluna k da linha k até m
        x = R[k:, k]
        norma_x = np.linalg.norm(x)

        if norma_x == 0.0:
            continue

        # Escolhe sinal de alpha para evitar cancelamento catastrófico
        alpha = -np.sign(x[0]) * norma_x if x[0] != 0 else -norma_x

        # Vetor de Householder v normalizado
        v = x.copy()
        v[0] -= alpha
        norma_v = np.linalg.norm(v)

        if norma_v < 1e-15:
            continue

        v = v / norma_v

        # Aplica a reflexão H_k à submatriz R[k:, k:] : R_sub = R_sub - 2 * v * (v^T R_sub)
        R[k:, k:] -= 2.0 * np.outer(v, np.dot(v, R[k:, k:]))

        # Acumula Q: Q = Q * H_k  =>  Q[:, k:] = Q[:, k:] - 2 * (Q[:, k:] v) * v^T
        Q[:, k:] -= 2.0 * np.outer(np.dot(Q[:, k:], v), v)

    return Q, R


def solve_qr(G, r, dtype=np.float64):
    """
    Resolve o problema de mínimos quadrados linear:
        min_δ ||G δ + r||_2
    via Fatoração QR (Householder):
        G = Q R
        R δ = -Q^T r  (sistema triangular superior)

    Previsão teórica de precisão:
        Erro relativo ≈ eps * κ(G), ao passo que nas normais é eps * κ(G)^2.
    """
    G_conv = np.asarray(G, dtype=dtype)
    r_conv = np.asarray(r, dtype=dtype)

    m, n = G_conv.shape
    Q, R = householder_qr(G_conv)

    # Projeta o vetor -r na base ortogonal de Q: d = -Q^T r
    d = -np.dot(Q.T, r_conv)

    # Resolve o sistema triangular superior R[:n, :n] δ = d[:n]
    R_quadrada = R[:n, :n]
    d_n = d[:n]

    delta = back_substitution(R_quadrada, d_n)
    return delta


# ==============================================================================
# 5. NÚMERO DE CONDICIONAMENTO
# ==============================================================================

def cond2(A):
    """
    Calcula o número de condição na norma 2:
        κ_2(A) = σ_max(A) / σ_min(A)
    """
    s = np.linalg.svd(A, compute_uv=False)
    sigma_max = s[0]
    sigma_min = s[-1]
    if sigma_min == 0.0:
        return np.inf
    return float(sigma_max / sigma_min)


if __name__ == "__main__":
    print("--- Teste rápido de solvers.py ---")
    # Matriz sintética sobredeterminada (5 equações, 3 incógnitas)
    np.random.seed(42)
    G_teste = np.random.randn(6, 3)
    r_teste = np.random.randn(6)

    delta_norm = solve_normal(G_teste, r_teste)
    delta_qr = solve_qr(G_teste, r_teste)
    delta_numpy = np.linalg.lstsq(G_teste, -r_teste, rcond=None)[0]

    print("Passo via Equações Normais (Cholesky):", delta_norm)
    print("Passo via Fatoração QR (Householder):", delta_qr)
    print("Gabarito via NumPy lstsq           :", delta_numpy)

    diff_norm = np.linalg.norm(delta_norm - delta_numpy)
    diff_qr = np.linalg.norm(delta_qr - delta_numpy)
    print(f"\nDiferença Normais vs NumPy: {diff_norm:.2e}")
    print(f"Diferença QR vs NumPy     : {diff_qr:.2e}")
    print(f"Número de condição κ_2(G) : {cond2(G_teste):.2f}")
