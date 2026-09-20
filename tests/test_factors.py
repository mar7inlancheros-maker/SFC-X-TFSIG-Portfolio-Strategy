"""Estandarizacion y combinacion de factores.

El error que estos tests existen para atrapar: comprar lo contrario de lo que se
pretende por un signo mal puesto, o confundir "pertenece a un sector caro" con
"es una mala empresa".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sfc_tfsig.factors.composite import build_scores
from sfc_tfsig.factors.scoring import combine_factors, factor_score, winsorize, zscore


def test_zscore_sin_dispersion_no_rompe():
    serie = pd.Series([5.0, 5.0, 5.0])
    assert zscore(serie).tolist() == [0.0, 0.0, 0.0]


def test_winsorize_acota_el_extremo_pero_conserva_el_orden():
    serie = pd.Series([1.0, 2.0, 3.0, 4.0, 400.0])
    acotada = winsorize(serie, 0.1)
    assert acotada.max() < 400.0
    assert acotada.idxmax() == 4  # sigue siendo la mayor


def test_direccion_negativa_invierte_el_score():
    """Mas deuda/patrimonio debe puntuar PEOR, no mejor."""
    panel = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-31"] * 4),
        "ticker": ["A", "B", "C", "D"],
        "sector": ["X"] * 4,
        "debt_to_equity": [0.1, 0.5, 1.5, 3.0],
    })
    scores = factor_score(
        panel, {"debt_to_equity": -1}, winsorize_pct=0.0, min_coverage=0.5,
        sector_neutral=False, min_sector_names=5,
    )
    assert scores.iloc[0] > scores.iloc[3]


def test_neutralidad_sectorial_compara_contra_pares():
    """Un ROIC mediocre en un sector malo puede ser mejor que uno bueno en uno excelente."""
    panel = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-31"] * 12),
        "ticker": [f"T{i}" for i in range(12)],
        "sector": ["Banca"] * 6 + ["Software"] * 6,
        "roic": [0.02, 0.03, 0.04, 0.05, 0.06, 0.20] + [0.25, 0.26, 0.27, 0.28, 0.29, 0.30],
    })
    metrics = {"roic": +1}

    neutral = factor_score(panel, metrics, winsorize_pct=0.0, min_coverage=0.5,
                           sector_neutral=True, min_sector_names=5)
    global_ = factor_score(panel, metrics, winsorize_pct=0.0, min_coverage=0.5,
                           sector_neutral=False, min_sector_names=5)

    # El mejor banco (0.20) es excepcional ENTRE BANCOS, pero mediocre en global.
    mejor_banco = neutral.iloc[5]
    peor_software = neutral.iloc[6]
    assert mejor_banco > peor_software
    # Sin neutralizar, el orden se invierte: el score mide sector, no calidad.
    assert global_.iloc[5] < global_.iloc[6]


def test_metrica_con_cobertura_insuficiente_se_descarta_entera():
    panel = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-31"] * 10),
        "ticker": [f"T{i}" for i in range(10)],
        "sector": ["X"] * 10,
        "roic": [0.1, 0.2] + [np.nan] * 8,          # 20% de cobertura
        "earnings_yield": list(np.linspace(0.01, 0.10, 10)),
    })
    scores = factor_score(
        panel, {"roic": +1, "earnings_yield": +1},
        winsorize_pct=0.0, min_coverage=0.6, sector_neutral=False, min_sector_names=5,
    )
    # Si roic hubiera entrado, T0 y T1 tendrian ventaja y T9 no seria el mejor.
    assert scores.idxmax() == 9


def test_combinar_renormaliza_sobre_los_factores_disponibles():
    scores = pd.DataFrame({
        "value": [1.0, 1.0],
        "quality": [1.0, 1.0],
        "momentum": [1.0, np.nan],
        "lowvol": [1.0, np.nan],
    })
    combined = combine_factors(scores, {"value": 0.25, "quality": 0.25,
                                        "momentum": 0.25, "lowvol": 0.25})
    # La segunda fila solo tiene dos factores, ambos a 1.0: su compuesto tambien
    # es 1.0, no 0.5. Rellenar con cero seria afirmar "momentum promedio".
    assert combined.iloc[0] == pytest.approx(1.0)
    assert combined.iloc[1] == pytest.approx(1.0)


def test_combinar_exige_un_minimo_de_factores():
    scores = pd.DataFrame({"value": [1.0], "quality": [np.nan],
                           "momentum": [np.nan], "lowvol": [np.nan]})
    combined = combine_factors(scores, {"value": 0.25, "quality": 0.25,
                                        "momentum": 0.25, "lowvol": 0.25}, min_factors=2)
    assert pd.isna(combined.iloc[0])


def test_build_scores_produce_las_columnas_esperadas(cfg):
    n = 20
    panel = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-31"] * n),
        "ticker": [f"T{i}" for i in range(n)],
        "sector": ["X"] * 10 + ["Y"] * 10,
        "earnings_yield": np.linspace(0.01, 0.20, n),
        "fcf_yield": np.linspace(0.01, 0.15, n),
        "roic": np.linspace(0.05, 0.30, n),
        "debt_to_equity": np.linspace(3.0, 0.1, n),
        "momentum": np.linspace(-0.2, 0.5, n),
        "volatility": np.linspace(0.6, 0.15, n),
    })
    out = build_scores(panel, cfg)
    for column in ("score_value", "score_quality", "score_momentum",
                   "score_lowvol", "score_composite"):
        assert column in out.columns

    # El panel esta construido de forma que la metrica mejora con el indice. Con
    # neutralidad sectorial, el mejor de CADA sector queda arriba y ambos
    # empatan: T9 es el mejor de X y T19 el mejor de Y. Que empaten no es un
    # empate cualquiera -- es la prueba de que el score mide posicion relativa
    # frente a los pares y no nivel absoluto.
    mejor_x = out.loc[out["sector"] == "X", "score_composite"].idxmax()
    mejor_y = out.loc[out["sector"] == "Y", "score_composite"].idxmax()
    assert mejor_x == 9
    assert mejor_y == 19
    assert out.loc[mejor_x, "score_composite"] == pytest.approx(
        out.loc[mejor_y, "score_composite"]
    )
