"""La capa de la SEC, probada sin red.

Lo que se prueba aqui no es "el codigo corre": es que el dato que llega al panel
sea el que estaba publicado ese dia. Un fallo en estos tests es un backtest
invalido, no un bug cosmetico.
"""

from __future__ import annotations

import pandas as pd
import pytest

from sfc_tfsig.data import sec


def _entry(start, end, val, filed, form="10-Q"):
    return {"start": start, "end": end, "val": val, "filed": filed, "form": form, "fy": 2023, "fp": "Q1"}


def _facts(**tags):
    """Construye un companyfacts minimo: {namespace: {tag: {units: {...}}}}."""
    facts: dict = {"us-gaap": {}, "dei": {}}
    for tag, (unit, entries) in tags.items():
        namespace = "dei" if tag.startswith("Entity") else "us-gaap"
        facts[namespace][tag] = {"units": {unit: entries}}
    return {"cik": 1, "entityName": "Test Co", "facts": facts}


# ---------------------------------------------------------------------------
#  Extraccion basica
# ---------------------------------------------------------------------------


def test_extrae_conceptos_y_descarta_formularios_no_periodicos():
    payload = _facts(
        Assets=("USD", [
            _entry(None, "2023-03-31", 1000.0, "2023-04-20", form="10-Q"),
            _entry(None, "2023-06-30", 1100.0, "2023-07-20", form="8-K"),  # se ignora
        ])
    )
    raw = sec.extract_raw_facts(payload)
    assert list(raw["concept"].unique()) == ["assets"]
    assert len(raw) == 1
    assert raw["value"].iloc[0] == 1000.0


def test_prioridad_de_etiquetas_no_mezcla_series():
    """Si hay etiqueta moderna y antigua, se usa solo la primera de la lista."""
    payload = _facts(
        RevenueFromContractWithCustomerExcludingAssessedTax=("USD", [
            _entry("2023-01-01", "2023-03-31", 500.0, "2023-04-20"),
        ]),
        Revenues=("USD", [
            _entry("2023-01-01", "2023-03-31", 9999.0, "2023-04-20"),
        ]),
    )
    raw = sec.extract_raw_facts(payload)
    revenue = raw[raw["concept"] == "revenue"]
    assert len(revenue) == 1
    assert revenue["value"].iloc[0] == 500.0


def test_facts_vacio_devuelve_marco_vacio_con_tipos():
    out = sec.facts_to_observations({})
    assert out.empty
    assert list(out.columns) == ["concept", "period_end", "filed", "value"]


# ---------------------------------------------------------------------------
#  Primera publicacion frente a reexpresion
# ---------------------------------------------------------------------------


def test_se_queda_con_la_primera_publicacion_no_con_la_reexpresion():
    payload = _facts(
        Assets=("USD", [
            _entry(None, "2023-03-31", 1000.0, "2023-04-20", form="10-Q"),
            # Un año despues la reexpresan: ese numero NO estaba disponible en 2023.
            _entry(None, "2023-03-31", 1250.0, "2024-04-22", form="10-K"),
        ])
    )
    out = sec.facts_to_observations(payload)
    row = out[out["concept"] == "assets"].iloc[0]
    assert row["value"] == 1000.0
    assert row["filed"] == pd.Timestamp("2023-04-20")


# ---------------------------------------------------------------------------
#  TTM
# ---------------------------------------------------------------------------


def _four_quarters(tag="Revenues", values=(100.0, 110.0, 120.0, 130.0)):
    periods = [
        ("2022-01-01", "2022-03-31", "2022-04-25"),
        ("2022-04-01", "2022-06-30", "2022-07-25"),
        ("2022-07-01", "2022-09-30", "2022-10-25"),
        ("2022-10-01", "2022-12-31", "2023-02-20"),
    ]
    return ("USD", [_entry(s, e, v, f) for (s, e, f), v in zip(periods, values)])


def test_ttm_suma_cuatro_trimestres_y_usa_la_presentacion_mas_tardia():
    payload = _facts(Revenues=_four_quarters())
    out = sec.facts_to_observations(payload)
    ttm = out[out["concept"] == "revenue"]
    assert len(ttm) == 1
    assert ttm["value"].iloc[0] == pytest.approx(460.0)
    # El TTM solo se pudo calcular cuando se presento el ultimo trimestre.
    assert ttm["filed"].iloc[0] == pd.Timestamp("2023-02-20")


def test_ttm_no_suma_ventanas_con_hueco():
    """Falta un trimestre: la ventana abarca 15 meses y se descarta."""
    payload = _facts(Revenues=("USD", [
        _entry("2022-01-01", "2022-03-31", 100.0, "2022-04-25"),
        _entry("2022-04-01", "2022-06-30", 110.0, "2022-07-25"),
        # falta Q3
        _entry("2022-10-01", "2022-12-31", 130.0, "2023-02-20"),
        _entry("2023-01-01", "2023-03-31", 140.0, "2023-04-25"),
    ]))
    out = sec.facts_to_observations(payload)
    assert out[out["concept"] == "revenue"].empty


def test_ttm_deriva_trimestres_desde_acumulados_del_año_fiscal():
    """Caso 10-Q real: se publica el acumulado, no el trimestre suelto."""
    payload = _facts(Revenues=("USD", [
        _entry("2022-01-01", "2022-03-31", 100.0, "2022-04-25"),   # 3 meses
        _entry("2022-01-01", "2022-06-30", 210.0, "2022-07-25"),   # 6 meses
        _entry("2022-01-01", "2022-09-30", 330.0, "2022-10-25"),   # 9 meses
        _entry("2022-01-01", "2022-12-31", 460.0, "2023-02-20"),   # 12 meses
    ]))
    quarters = sec.quarterly_segments(sec.extract_raw_facts(payload))
    quarters = quarters[quarters["concept"] == "revenue"].sort_values("end")
    assert list(quarters["value"]) == pytest.approx([100.0, 110.0, 120.0, 130.0])

    out = sec.facts_to_observations(payload)
    ttm = out[out["concept"] == "revenue"]
    # El anual (460) y la suma de los cuatro trimestres coinciden: una sola fila.
    assert ttm["value"].iloc[-1] == pytest.approx(460.0)


def test_respaldo_anual_cuando_no_hay_trimestres():
    """Emisor extranjero que solo presenta el año completo (20-F/40-F)."""
    payload = _facts(Revenues=("USD", [
        _entry("2022-01-01", "2022-12-31", 1000.0, "2023-03-15", form="40-F"),
        _entry("2023-01-01", "2023-12-31", 1200.0, "2024-03-15", form="40-F"),
    ]))
    out = sec.facts_to_observations(payload)
    ttm = out[out["concept"] == "revenue"].sort_values("period_end")
    assert list(ttm["value"]) == pytest.approx([1000.0, 1200.0])


# ---------------------------------------------------------------------------
#  La barrera contra el look-ahead
# ---------------------------------------------------------------------------


def test_as_of_no_deja_ver_lo_que_aun_no_se_habia_presentado():
    observations = pd.DataFrame({
        "cik": [1, 1, 2],
        "concept": ["assets", "assets", "assets"],
        "period_end": pd.to_datetime(["2023-03-31", "2023-06-30", "2023-03-31"]),
        "filed": pd.to_datetime(["2023-04-20", "2023-07-25", "2023-04-28"]),
        "value": [1000.0, 1100.0, 5000.0],
    })

    # El 30 de junio el cierre de junio existe como hecho contable, pero aun no
    # se ha presentado: el modelo NO puede verlo.
    visible = sec.as_of(observations, pd.Timestamp("2023-06-30"))
    assert visible.set_index("cik").loc[1, "assets"] == 1000.0

    despues = sec.as_of(observations, pd.Timestamp("2023-08-01"))
    assert despues.set_index("cik").loc[1, "assets"] == 1100.0


def test_as_of_antes_de_cualquier_presentacion_no_devuelve_nada():
    observations = pd.DataFrame({
        "cik": [1],
        "concept": ["assets"],
        "period_end": pd.to_datetime(["2023-03-31"]),
        "filed": pd.to_datetime(["2023-04-20"]),
        "value": [1000.0],
    })
    assert sec.as_of(observations, pd.Timestamp("2023-01-01")).empty
