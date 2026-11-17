#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testes unitários para o módulo geo.py
Critério do SDD (Seção 5): erro < 1 km frente a distâncias conhecidas na esfera (R = 6371 km).
"""

import os
import sys
import numpy as np
import pytest

# Adiciona a raiz do projeto e a pasta src ao sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.geo import haversine, latlon_to_unit_vector, azimute_inicial, RAIO_TERRA_KM


def test_distancia_identidade():
    """Distância de um ponto a ele mesmo deve ser estritamente zero."""
    lat, lon = -23.5505, -46.6333
    d = haversine(lat, lon, lat, lon)
    assert abs(d) < 1e-12


def test_distancia_simetria():
    """A distância de A até B deve ser igual à distância de B até A."""
    sp = (-23.5505, -46.6333)
    cmu = (40.4433, -79.9436)
    d_ab = haversine(sp[0], sp[1], cmu[0], cmu[1])
    d_ba = haversine(cmu[0], cmu[1], sp[0], sp[1])
    assert abs(d_ab - d_ba) < 1e-10


def test_distancia_um_grau_equador():
    """
    1 grau no equador deve medir exatamente: 2 * pi * R / 360 km.
    Para R = 6371.0 km, isso dá exatamente ~111.1949266 km.
    """
    esperado = (2.0 * np.pi * RAIO_TERRA_KM) / 360.0
    calculado = haversine(0.0, 0.0, 0.0, 1.0)
    assert abs(calculado - esperado) < 1e-6


def test_distancia_conhecida_sp_rio():
    """
    SP (Marco Zero: -23.5505, -46.6333) -> Rio (Centro: -22.9068, -43.1729).
    Distância esférica de círculo máximo (R = 6371 km): ~360.75 km.
    Conforme SDD: SP-Rio ≈ 360 km.
    """
    sp_lat, sp_lon = -23.5505, -46.6333
    rj_lat, rj_lon = -22.9068, -43.1729
    d = haversine(sp_lat, sp_lon, rj_lat, rj_lon)
    referencia = 360.75  # km na esfera
    assert abs(d - referencia) < 0.1, f"Obtido {d:.2f} km, esperado ~{referencia} km"


def test_distancia_conhecida_sp_lisboa():
    """
    SP (-23.5505, -46.6333) -> Lisboa (38.7223, -9.1393).
    Distância ortodrômica na esfera (R = 6371 km): ~7949.0 km.
    Conforme SDD: SP-Lisboa ≈ 7.900 km a 7.950 km.
    """
    sp_lat, sp_lon = -23.5505, -46.6333
    lisboa_lat, lisboa_lon = 38.7223, -9.1393
    d = haversine(sp_lat, sp_lon, lisboa_lat, lisboa_lon)
    referencia = 7949.03  # km na esfera
    assert abs(d - referencia) < 0.1, f"Obtido {d:.2f} km, esperado ~{referencia} km"


def test_haversine_vetorizado():
    """
    Valida se haversine aceita arrays do NumPy (ex: calcular a distância
    de 1 alvo para múltiplos servidores simultaneamente).
    """
    alvo_lat, alvo_lon = -23.55, -46.63
    servidores_lat = np.array([-22.90, -19.86, -30.03, -15.76])
    servidores_lon = np.array([-43.17, -43.96, -51.21, -47.87])

    distancias = haversine(alvo_lat, alvo_lon, servidores_lat, servidores_lon)
    assert isinstance(distancias, np.ndarray)
    assert len(distancias) == 4
    assert np.all(distancias > 0)


def test_vetor_unitario():
    """Vetor unitário gerado a partir de lat/lon deve ter norma 1."""
    u = latlon_to_unit_vector(-23.55, -46.63)
    norma = np.linalg.norm(u)
    assert abs(norma - 1.0) < 1e-12


if __name__ == "__main__":
    pytest.main([__file__])
