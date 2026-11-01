"""El motor long/short entero, de la descarga al HTML, con datos sinteticos.

Sustituye red y disco (Yahoo, tipos de cambio, SEC, WRDS, fechas de
resultados, DCF, panel del multifactor) y corre lo mismo que `python
modelo.py`: analisis, plan, fichas, terminal y ficheros. Un `NameError` en el
HTML del plan paso 808 tests porque ninguno llegaba ahi; este si llega.

Si el motor empieza a leer algo nuevo de la red o del disco, hay que
sustituirlo aqui tambien o el test tocara la maquina real.
"""

from __future__ import annotations

import io
import zlib

import numpy as np
import pandas as pd
import pytest
import yaml
from rich.console import Console

from quant_engine import analysis
from quant_engine.data import loader, multifactor, wrds_data
from quant_engine.reporting import export, terminal
from quant_engine.settings import DEFAULT_YAML, build_settings, load_yaml
from quant_engine.valuation import engine as valuation_engine

LONGS, SHORTS = ("LONGA", "LONGB"), ("SHORTA", "SHORTB.TO")
PEERS = [f"PEER{i:02d}" for i in range(12)]
CAD = 0.72


def _ohlcv(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Paseo aleatorio con deriva propia por ticker: siempre el mismo para el mismo nombre."""
    rng = np.random.default_rng(zlib.crc32(ticker.encode()))
    idx = pd.bdate_range(start, end)
    drift = rng.normal(0.0003, 0.0004)
    close = 50 * np.exp(np.cumsum(rng.normal(drift, 0.018, len(idx))))
    spread = np.abs(rng.normal(0.0, 0.008, len(idx))) + 0.004
    return pd.DataFrame({"Open": close, "High": close * (1 + spread), "Low": close * (1 - spread),
                         "Close": close, "Volume": rng.integers(800_000, 3_000_000, len(idx)).astype(float)},
                        index=idx)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    tmp = tmp_path_factory.mktemp("qe_e2e")
    raw = load_yaml(DEFAULT_YAML)
    raw["scoring"]["reference_peers"] = PEERS
    raw["validation"].update({"months": 8, "min_names": 10})
    raw.setdefault("output", {})["directory"] = str(tmp / "out")
    cfg = tmp / "qe.yaml"
    cfg.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")

    mp.setattr(loader, "load_ohlcv", lambda tickers, start, end, **_: loader.MarketData(
        ohlcv={t: _ohlcv(t, pd.Timestamp(start), pd.Timestamp(end)) for t in dict.fromkeys(tickers)}))
    mp.setattr(loader, "load_fx", lambda currencies, start, end: (
        {c: pd.Series(CAD, index=pd.bdate_range(start, end)) for c in currencies}, {}))
    mp.setattr(loader, "load_sectors", lambda tickers: ({t: "Technology" for t in tickers},
                                                        {t: "test" for t in tickers}))
    mp.setattr(loader, "load_earnings_dates", lambda tickers, today, **_: {})
    mp.setattr(loader, "load_ex_dividend_dates", lambda tickers, today, **_: {})
    mp.setattr(wrds_data, "load", lambda tickers, start, as_of: wrds_data.WrdsData(available=False,
                                                                                    reason="test"))
    mp.setattr(valuation_engine, "run_valuations",
               lambda result, wrds, fx, as_of, tickers: {t: {"available": False, "reason": "test"} for t in tickers})
    mp.setattr(multifactor, "load", lambda *a, **k: {"available": False, "reason": "test"})

    settings = build_settings(",".join(LONGS), ",".join(SHORTS), yaml_path=cfg, lookback="3y")
    result = analysis.run_analysis(settings, run_robustness=False)
    report = Console(record=True, file=io.StringIO(), width=160)
    terminal.render(result, report)
    brief = Console(record=True, file=io.StringIO(), width=160)
    terminal.brief(result, brief)
    paths = export.export_all(result, report, tmp / "out", "test", [])
    yield result, report.export_text(), brief.export_text(), paths
    mp.undo()


def test_el_plan_cuadra(run):
    result, *_ = run
    tp = result.trade_plan
    assert tp["integrity_errors"] == []
    traded = tp["orders"][(tp["orders"]["target"] != 0) & (tp["orders"]["verdict"] != "HEDGE")]
    assert traded["open_date"].notna().all() and traded["exit_date"].notna().all()
    shorts = traded[traded["target"] < 0]
    assert shorts["time_stop_date"].notna().all() if len(shorts) else True


def test_la_salida_de_terminal_es_ascii_y_trae_la_senal(run):
    _, report, brief, _ = run
    for text in (report, brief):
        assert text.isascii(), [c for c in text if not c.isascii()][:5]
    assert "FINAL SIGNAL" in brief and "SIGNALS: WHAT TO DO, WHY, AND HOW" in brief


def test_los_ficheros_salen_completos(run):
    _, _, _, paths = run
    html = paths["trade_plan_html"].read_text(encoding="utf-8")
    assert "<h1>Trade Plan</h1>" in html and "Hard exit" in html
    orders = pd.read_csv(paths["orders"])
    assert {"open_date", "time_stop_date", "exit_date"} <= set(orders.columns)
