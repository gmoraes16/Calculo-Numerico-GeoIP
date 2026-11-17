#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testes unitários dos Solvers de Cálculo Numérico (test_solvers.py)
Critérios do SDD (Seção 5):
- Cholesky e QR vs numpy.linalg.lstsq: erro relativo < 1e-10 em float64 (κ moderado).
- Ortogonalidade de Q: ||Q^T Q - I|| < 1e-12.
- Reconstrução: ||Q R - G|| < 1e-12.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.solvers import (
    cholesky,
    forward_substitution,
    back_substitution,
    householder_qr,
    solve_normal,
    solve_qr,
    cond2
)


def test_cholesky_decomposicao():
    """Valida fatoração A = L L^T com matriz simétrica positiva definida."""
    # Gera matriz simétrica positiva definida A = B B^T + n*I
    np.random.seed(123)
    B = np.random.randn(4, 4)
    A = np.dot(B, B.T) + 4.0 * np.eye(4)

    L = cholesky(A)

    # Verifica se L é triangular inferior
    assert np.allclose(L, np.tril(L))
    # Verifica reconstrução L L^T ≈ A
    A_reconstruida = np.dot(L, L.T)
    erro = np.linalg.norm(A - A_reconstruida)
    assert erro < 1e-12

    # Compara diretamente com o gabarito do NumPy
    L_numpy = np.linalg.cholesky(A)
    assert np.allclose(L, L_numpy, atol=1e-12)


def test_substituicao_progressiva_e_regressiva():
    """Valida resolução de sistemas triangulares."""
    L = np.array([
        [2.0, 0.0, 0.0],
        [1.0, 3.0, 0.0],
        [4.0, 2.0, 5.0]
    ])
    # 4*2 + 2*2 + 5*3 = 8 + 4 + 15 = 27
    b = np.array([4.0, 8.0, 27.0])

    # L y = b => y = [2, 2, 3]
    y = forward_substitution(L, b)
    assert np.allclose(y, [2.0, 2.0, 3.0], atol=1e-12)

    # U x = y
    U = L.T
    x = back_substitution(U, y)
    # Testa se U x ≈ y
    assert np.allclose(np.dot(U, x), y, atol=1e-12)


def test_ortogonalidade_e_reconstrucao_qr():
    """
    Critério SDD: test_ortogonalidade:
    Q^T Q ≈ I, Q R ≈ G com erro < 1e-12.
    """
    np.random.seed(456)
    G = np.random.randn(8, 3)

    Q, R = householder_qr(G)

    # 1. Ortogonalidade: Q^T Q ≈ I
    m = G.shape[0]
    erro_ortogonalidade = np.linalg.norm(np.dot(Q.T, Q) - np.eye(m))
    assert erro_ortogonalidade < 1e-12, f"Erro ortogonalidade: {erro_ortogonalidade:.2e}"

    # 2. Reconstrução: Q R ≈ G
    erro_reconstrucao = np.linalg.norm(np.dot(Q, R) - G)
    assert erro_reconstrucao < 1e-12, f"Erro reconstrução: {erro_reconstrucao:.2e}"

    # 3. R é triangular superior
    assert np.allclose(R, np.triu(R))


def test_solve_normal_vs_numpy_lstsq():
    """
    Critério SDD: Cholesky/Normais vs numpy.linalg.lstsq
    com erro relativo < 1e-10 em float64.
    """
    np.random.seed(789)
    G = np.random.randn(10, 3)
    r = np.random.randn(10)

    delta_nosso = solve_normal(G, r, dtype=np.float64)
    delta_esperado = np.linalg.lstsq(G, -r, rcond=None)[0]

    erro_relativo = np.linalg.norm(delta_nosso - delta_esperado) / np.linalg.norm(delta_esperado)
    assert erro_relativo < 1e-10, f"Erro relativo: {erro_relativo:.2e}"


def test_solve_qr_vs_numpy_lstsq():
    """
    Critério SDD: QR/Householder vs numpy.linalg.lstsq
    com erro relativo < 1e-10 em float64.
    """
    np.random.seed(101)
    G = np.random.randn(10, 3)
    r = np.random.randn(10)

    delta_nosso = solve_qr(G, r, dtype=np.float64)
    delta_esperado = np.linalg.lstsq(G, -r, rcond=None)[0]

    erro_relativo = np.linalg.norm(delta_nosso - delta_esperado) / np.linalg.norm(delta_esperado)
    assert erro_relativo < 1e-10, f"Erro relativo: {erro_relativo:.2e}"


def test_concordancia_normais_vs_qr_em_float64():
    """Em float64 e com κ moderado, Normais e QR devem concordar perfeitamente."""
    np.random.seed(202)
    G = np.random.randn(7, 3)
    r = np.random.randn(7)

    delta_norm = solve_normal(G, r, dtype=np.float64)
    delta_qr = solve_qr(G, r, dtype=np.float64)

    assert np.allclose(delta_norm, delta_qr, rtol=1e-9, atol=1e-10)


def test_suporte_float32():
    """Valida se os solvers funcionam com tipo np.float32."""
    G32 = np.random.randn(6, 3).astype(np.float32)
    r32 = np.random.randn(6).astype(np.float32)

    delta_norm32 = solve_normal(G32, r32, dtype=np.float32)
    delta_qr32 = solve_qr(G32, r32, dtype=np.float32)

    assert delta_norm32.dtype == np.float32
    assert delta_qr32.dtype == np.float32
    assert np.allclose(delta_norm32, delta_qr32, rtol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__])
