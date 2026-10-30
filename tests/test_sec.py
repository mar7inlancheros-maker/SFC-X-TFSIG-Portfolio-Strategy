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


def test_en_un_mismo_periodo_gana_la_etiqueta_de_mayor_prioridad():
    """Dos etiquetas para el MISMO trimestre: se queda la de arriba de la lista."""
    payload = _facts(
        CashAndCashEquivalentsAtCarryingValue=("USD", [_entry(None, "2023-03-31", 500.0, "2023-04-20")]),
        CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents=("USD", [
            _entry(None, "2023-03-31", 9999.0, "2023-04-20"),
        ]),
    )
    elegido = sec.select_by_priority(sec.extract_raw_facts(payload))
    cash = elegido[elegido["concept"] == "cash"]
    assert len(cash) == 1
    assert cash["value"].iloc[0] == 500.0


def test_en_ventas_gana_el_total_y_no_un_desglose():
    """Gold.com: el desglose ASC 606 (618) tiene mas prioridad que el total (25.513)."""
    payload = _facts(
        RevenueFromContractWithCustomerExcludingAssessedTax=("USD", [
            _entry("2025-07-01", "2026-06-30", 618.0, "2026-08-20", form="10-K"),
        ]),
        Revenues=("USD", [
            _entry("2025-07-01", "2026-06-30", 25513.0, "2026-08-20", form="10-K"),
        ]),
    )
    elegido = sec.select_by_priority(sec.extract_raw_facts(payload))
    revenue = elegido[elegido["concept"] == "revenue"]
    assert len(revenue) == 1 and revenue["value"].iloc[0] == 25513.0
    assert revenue["tag"].iloc[0] == "us-gaap:Revenues"


def test_ventas_brutas_de_impuestos_solo_si_no_hay_otra():
    payload = _facts(
        RevenueFromContractWithCustomerExcludingAssessedTax=("USD", [
            _entry("2023-01-01", "2023-03-31", 800.0, "2023-04-20"),
        ]),
        RevenueFromContractWithCustomerIncludingAssessedTax=("USD", [
            _entry("2023-01-01", "2023-03-31", 1000.0, "2023-04-20"),
            _entry("2023-04-01", "2023-06-30", 1100.0, "2023-07-20"),
        ]),
    )
    revenue = sec.select_by_priority(sec.extract_raw_facts(payload))
    revenue = revenue[revenue["concept"] == "revenue"].sort_values("end")
    assert list(revenue["value"]) == [800.0, 1100.0]     # neta si existe; bruta solo como respaldo


def test_ventas_brutas_cuentan_si_el_total_vale_cero():
    """Cooper-Standard: `Revenues` = 0 y la bruta (sin neta) es la real."""
    payload = _facts(
        RevenueFromContractWithCustomerIncludingAssessedTax=("USD", [
            _entry("2023-01-01", "2023-12-31", 2815.9, "2024-02-20", form="10-K"),
        ]),
        Revenues=("USD", [_entry("2023-01-01", "2023-12-31", 0.0, "2024-02-20", form="10-K")]),
    )
    revenue = sec.select_by_priority(sec.extract_raw_facts(payload))
    assert revenue[revenue["concept"] == "revenue"]["value"].iloc[0] == 2815.9


def test_en_ventas_el_maximo_respeta_la_primera_publicacion():
    """El total re-expresado un ano despues no estaba disponible en su dia."""
    payload = _facts(
        RevenueFromContractWithCustomerExcludingAssessedTax=("USD", [
            _entry("2023-01-01", "2023-03-31", 500.0, "2023-04-20"),
        ]),
        Revenues=("USD", [
            _entry("2023-01-01", "2023-03-31", 900.0, "2023-04-20"),
            _entry("2023-01-01", "2023-03-31", 950.0, "2024-04-22", form="10-K"),
        ]),
    )
    elegido = sec.select_by_priority(sec.extract_raw_facts(payload))
    row = elegido[elegido["concept"] == "revenue"].iloc[0]
    assert row["value"] == 900.0 and row["filed"] == pd.Timestamp("2023-04-20")


def test_la_etiqueta_antigua_cubre_los_periodos_que_la_moderna_no_alcanza():
    """El caso ASC 606, que costaba la mitad de la cobertura de ingresos.

    Apple publica `RevenueFromContractWithCustomer...` desde 2017 y `Revenues`
    antes. Elegir UNA etiqueta para toda la empresa borraba la historia previa;
    elegir periodo a periodo la conserva.
    """
    payload = _facts(
        RevenueFromContractWithCustomerExcludingAssessedTax=("USD", [
            _entry("2018-01-01", "2018-03-31", 500.0, "2018-04-20"),
        ]),
        Revenues=("USD", [
            _entry("2016-01-01", "2016-03-31", 300.0, "2016-04-20"),
            _entry("2017-01-01", "2017-03-31", 400.0, "2017-04-20"),
        ]),
    )
    elegido = sec.select_by_priority(sec.extract_raw_facts(payload))
    revenue = elegido[elegido["concept"] == "revenue"].sort_values("end")
    assert len(revenue) == 3
    assert list(revenue["value"]) == [300.0, 400.0, 500.0]


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
            # Un ano despues la reexpresan: ese numero NO estaba disponible en 2023.
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
    return ("USD", [_entry(s, e, v, f) for (s, e, f), v in zip(periods, values, strict=True)])


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


def test_ttm_deriva_trimestres_desde_acumulados_del_ano_fiscal():
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
    """Emisor extranjero que solo presenta el ano completo (20-F/40-F)."""
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


# ---------------------------------------------------------------------------
#  Cache de dos niveles: crudas y observaciones versionadas
# ---------------------------------------------------------------------------


def _raw_de_prueba():
    payload = _facts(Revenues=_four_quarters())
    return sec.extract_raw_facts(payload)


def test_con_observaciones_en_cache_no_se_recalcula_el_ttm(tmp_path, monkeypatch):
    from sfc_tfsig.data import cache

    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    cache.write_frame(sec._raw_name(7), _raw_de_prueba())
    primera = sec.company_observations(7)

    llamadas = []
    monkeypatch.setattr(sec, "observations_from_raw",
                        lambda raw: llamadas.append(1) or pd.DataFrame())
    segunda = sec.company_observations(7)

    assert llamadas == []
    pd.testing.assert_frame_equal(primera.reset_index(drop=True),
                                  segunda.reset_index(drop=True))


def test_sin_observaciones_se_derivan_de_las_crudas_sin_red(tmp_path, monkeypatch):
    from sfc_tfsig.data import cache

    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    cache.write_frame(sec._raw_name(7), _raw_de_prueba())
    # Si se intentara bajar de EDGAR, este fallo lo delataria.
    monkeypatch.setattr(sec, "_session", lambda: (_ for _ in ()).throw(AssertionError("red")))

    obs = sec.company_observations(7)
    assert (obs["concept"] == "revenue").any()
    assert cache.exists(sec._observations_name(7))


def test_subir_la_version_invalida_las_observaciones_viejas(tmp_path, monkeypatch):
    """Cambiar la logica sin subir la version mezclaria observaciones viejas y
    codigo nuevo. Con la version en el nombre, eso no puede pasar."""
    from sfc_tfsig.data import cache

    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    cache.write_frame(sec._raw_name(7), _raw_de_prueba())
    sec.company_observations(7)
    vieja = sec._observations_name(7)

    monkeypatch.setattr(sec, "OBSERVATIONS_VERSION", "v-siguiente")
    assert sec._observations_name(7) != vieja
    assert not cache.exists(sec._observations_name(7))

    sec.company_observations(7)
    assert cache.exists(sec._observations_name(7))


def test_recalculo_en_paralelo_da_lo_mismo_que_en_serie_y_usa_la_cache_del_padre(tmp_path, monkeypatch):
    """Los procesos hijos escriben en la cache redirigida, no en la real."""
    from sfc_tfsig.data import cache

    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    ciks = list(range(100, 160))
    for c in ciks:
        cache.write_frame(sec._raw_name(c), _raw_de_prueba())
    assert sec.derive_missing_observations(ciks, workers=2, progress=False) == len(ciks)
    assert all(cache.exists(sec._observations_name(c)) for c in ciks)
    serie = sec.observations_from_raw(_raw_de_prueba())
    paralelo = cache.read_frame(sec._observations_name(ciks[0])).drop(columns="cik")
    pd.testing.assert_frame_equal(paralelo.reset_index(drop=True), serie.reset_index(drop=True))
    # Lo ya calculado no se repite.
    assert sec.derive_missing_observations(ciks, workers=2, progress=False) == 0

