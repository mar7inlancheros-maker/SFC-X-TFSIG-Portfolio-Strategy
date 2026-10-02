"""Trazabilidad: el fingerprint solo cubre la config; el codigo tambien cuenta."""

from __future__ import annotations

import subprocess

import pandas as pd
import pytest

from sfc_tfsig import cli, orders as orders_mod, report as report_mod
from sfc_tfsig.config import code_revision, provenance_line


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "a.py").write_text("x = 1\n")
    _git(tmp_path, "add", "a.py")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_la_revision_es_el_hash_corto_del_commit(repo):
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    assert code_revision(repo) == head


def test_cambios_sin_commitear_marcan_dirty(repo):
    (repo / "a.py").write_text("x = 2\n")
    assert code_revision(repo).endswith("-dirty")


def test_fuera_de_git_no_revienta(tmp_path):
    assert code_revision(tmp_path / "no_existe") == "sin-git"


def test_la_procedencia_lleva_commit_y_versiones_de_cache():
    prov = cli.provenance()
    assert set(prov) == {"commit", "panel_version", "observations_version"}
    line = provenance_line(prov)
    assert prov["commit"] in line and prov["panel_version"] in line
    assert prov["observations_version"] in line


def test_la_hoja_de_ordenes_estampa_la_procedencia(cfg):
    prov = {"commit": "abc1234-dirty", "panel_version": "v9", "observations_version": "v7"}
    target = pd.DataFrame({"ticker": ["AAA"], "sector": ["Tech"], "price": [10.0],
                           "weight": [1.0], "score_composite": [1.0]})
    ordenes = orders_mod.build_orders(target, pd.Series(dtype="float64"), 1000.0, cfg)
    texto = orders_mod.render_orders(ordenes, target, 1000.0, cfg, pd.Timestamp("2026-09-30"),
                                     provenance=prov)
    assert "abc1234-dirty" in texto and "v9" in texto and "v7" in texto


def test_el_meta_de_los_artefactos_guarda_fingerprint_y_procedencia(tmp_path, monkeypatch):
    import json

    from sfc_tfsig.backtest import BacktestResult

    monkeypatch.setattr(report_mod, "REPORT_DIR", tmp_path)
    nav = pd.Series([1.0, 1.1], index=pd.date_range("2026-01-01", periods=2))
    result = BacktestResult(nav=nav, holdings=pd.DataFrame(), trades=pd.DataFrame(),
                            rebalances=pd.DataFrame(), config_fingerprint="f" * 16)
    prov = {"commit": "abc1234", "panel_version": "v9", "observations_version": "v7"}
    paths = report_mod.save_artifacts(result, provenance=prov)
    meta = json.loads(paths["meta"].read_text())
    assert meta["config_fingerprint"] == "f" * 16
    assert meta["commit"] == "abc1234"
