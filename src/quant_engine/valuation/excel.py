"""El DCF a un Excel con formulas vivas, con el formato de los modelos del club.

Inputs en azul, formulas en negro (la convencion de los modelos de PLTR y
CNL). La hoja DCF calcula con formulas exactamente lo mismo que
`dcf.value_vector`: si el analista cambia un supuesto o el selector de
escenario, el valor se recalcula. Sensibilidades, Monte Carlo y DCF inverso
van como valores calculados por el motor el dia de la corrida, y lo dicen: una
tabla de datos de Excel no se puede escribir desde Python.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

BLUE = Font(color="1F4E9E")
BOLD = Font(bold=True)
TITLE = Font(bold=True, size=14)
HEAD = PatternFill("solid", fgColor="E8EDF5")
KEY = PatternFill("solid", fgColor="FFF4D6")
THIN = Border(bottom=Side(style="thin", color="BBBBBB"))
PCT, MULT, MONEY, PRICE = "0.0%", "0.00", "#,##0.0;(#,##0.0)", "#,##0.00"


def _put(ws, cell: str, value, fmt: str | None = None, font=None, fill=None):
    c = ws[cell]
    c.value = value
    if fmt:
        c.number_format = fmt
    if font:
        c.font = font
    if fill:
        c.fill = fill
    return c


def _label(ws, row: int, text: str, bold: bool = False) -> None:
    c = ws.cell(row=row, column=1, value=text)
    if bold:
        c.font = BOLD


def write(v: dict, path: Path) -> Path:
    """Escribe el modelo de una empresa valorada con `engine.value_company`."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Inputs"
    cases = v["cases"]
    base = cases["base"]
    years = base.years
    cap, coc = v["capital"], v["cost_of_capital"]
    ccy = v["currency"]
    ws.column_dimensions["A"].width = 46
    for col in range(2, 4 + years):
        ws.column_dimensions[get_column_letter(col)].width = 12

    _put(ws, "A1", f"{v['ticker']} - automated DCF ({ccy} millions, except per share)", font=TITLE)
    _put(ws, "A2", f"Fiscal year {pd.Timestamp(v['fiscal_end']):%Y-%m-%d}; source {v.get('source', '')}. "
                   "Blue = input, black = formula.")
    _put(ws, "A4", "MARKET DATA & CAPITAL STRUCTURE", font=BOLD, fill=HEAD)
    rows = [("Share price (reporting currency)", cap.price, PRICE), ("Diluted shares (mm)", cap.shares, MONEY),
            ("Cash and short-term investments", cap.cash, MONEY), ("Debt incl. finance leases", cap.debt, MONEY),
            ("Minority interest", cap.minority, MONEY), ("Preferred stock", cap.preferred, MONEY)]
    for i, (name, val, fmt) in enumerate(rows, start=5):
        _label(ws, i, name)
        _put(ws, f"B{i}", float(val), fmt, BLUE)
    _label(ws, 11, "Market capitalisation")
    _put(ws, "B11", "=B5*B6", MONEY)
    # WACC
    _put(ws, "A13", "COST OF CAPITAL", font=BOLD, fill=HEAD)
    wacc_rows = [
        (14, "Risk-free rate", coc["rf"], PCT, True), (15, "Raw beta (regression vs benchmark)", coc["beta_raw"], MULT, True),
        (16, "Blume-adjusted beta = 0.67 x raw + 0.33", "=0.67*B15+0.33", MULT, False),
        (17, "Equity risk premium", coc["erp"], PCT, True), (18, "Cost of equity (CAPM)", "=B14+B16*B17", PCT, False),
        (19, f"Credit spread (synthetic rating {coc['rating']}, coverage {coc['coverage']:.1f}x)"
             if np.isfinite(coc["coverage"]) else f"Credit spread (synthetic rating {coc['rating']}, no interest)",
         coc["spread"], PCT, True),
        (20, "Pre-tax cost of debt", "=B14+B19", PCT, False), (21, "Marginal tax rate", coc["marginal_tax"], PCT, True),
        (22, "After-tax cost of debt", "=B20*(1-B21)", PCT, False),
        (23, "Equity weight E/(D+E)", "=B11/(B11+MAX(B8,0))", PCT, False), (24, "Debt weight", "=1-B23", PCT, False),
        (25, "WACC (CAPM, market weights)", "=B23*B18+B24*B22", PCT, False),
    ]
    for r, name, val, fmt, is_input in wacc_rows:
        _label(ws, r, name)
        _put(ws, f"B{r}", val, fmt, BLUE if is_input else None)
    ws["B25"].fill = KEY

    # Escenarios
    _put(ws, "A27", "SCENARIOS (1 = bear, 2 = base, 3 = bull)", font=BOLD, fill=HEAD)
    _label(ws, 28, "Selected scenario")
    _put(ws, "B28", 2, "0", BLUE, KEY)
    _put(ws, "C28", '=CHOOSE(B28,"Bear","Base","Bull")')
    row = 30
    year_cols = [get_column_letter(3 + t) for t in range(years)]
    for col, t in zip(year_cols, range(1, years + 1), strict=True):
        _put(ws, f"{col}{row - 1}", f"Year {t}", font=BOLD)
    blocks: dict[str, dict[str, int]] = {}
    for key in ("bear", "base", "bull"):
        a = cases[key]
        _put(ws, f"A{row}", f"{key.upper()} CASE", font=BOLD)
        row += 1
        refs = {}
        for name, arr in (("Revenue growth", a.growth), ("EBIT margin", a.margin), ("Cash tax rate", a.tax)):
            _label(ws, row, name)
            for col, x in zip(year_cols, arr, strict=True):
                _put(ws, f"{col}{row}", float(x), PCT, BLUE)
            refs[name] = row
            row += 1
        # Ajuste sobre el WACC CAPM de la hoja: con un override de WACC, el
        # ajuste lo recoge y la hoja sigue cuadrando con el motor.
        delta_wacc = a.wacc - coc["wacc"]
        for name, val, fmt in (("D&A % of revenue", a.da_pct, PCT), ("CapEx % of revenue", a.capex_pct, PCT),
                               ("Working capital % of new revenue", a.nwc_pct, PCT),
                               ("Max sales-to-capital", a.sales_to_capital, MULT),
                               ("Terminal growth", a.terminal_growth, PCT),
                               ("WACC adjustment vs CAPM", delta_wacc, PCT),
                               ("RONIC minus WACC", a.ronic - a.wacc, PCT)):
            _label(ws, row, name)
            _put(ws, f"B{row}", float(val), fmt, BLUE)
            refs[name] = row
            row += 1
        blocks[key] = refs
        row += 1

    # Hoja DCF con formulas
    d = wb.create_sheet("DCF")
    d.column_dimensions["A"].width = 40
    _put(d, "A1", f"{v['ticker']} - DCF ({ccy} millions)", font=TITLE)
    _put(d, "A2", "=\"Scenario: \"&Inputs!C28")
    hist = v["history"]
    last_rev = float(hist["revenue"].iloc[-1])
    cols = ["B"] + [get_column_letter(3 + t) for t in range(years)]
    for i, c in enumerate(cols):
        d.column_dimensions[c].width = 12
        label = f"FY{pd.Timestamp(v['fiscal_end']).year}A" if i == 0 else f"FY{pd.Timestamp(v['fiscal_end']).year + i}E"
        _put(d, f"{c}4", label, font=BOLD, fill=HEAD)
    sel = lambda name, col_idx: (  # noqa: E731
        f"CHOOSE(Inputs!$B$28,Inputs!{year_cols[col_idx]}{blocks['bear'][name]},"
        f"Inputs!{year_cols[col_idx]}{blocks['base'][name]},Inputs!{year_cols[col_idx]}{blocks['bull'][name]})")
    scal = lambda name: (  # noqa: E731
        f"CHOOSE(Inputs!$B$28,Inputs!$B${blocks['bear'][name]},Inputs!$B${blocks['base'][name]},"
        f"Inputs!$B${blocks['bull'][name]})")
    lines = ["Revenue", "  growth", "EBIT", "  margin", "Taxes on EBIT", "  tax rate", "NOPAT", "(+) D&A",
             "(-) CapEx", "(-) Increase in working capital", "(-) Extra capital to fund growth",
             "Unlevered free cash flow", "Discount period (mid-year)", "Discount factor", "PV of FCF"]
    r0 = 5
    R = {name: r0 + i for i, name in enumerate(lines)}
    for name, r in R.items():
        _label(d, r, name, bold=name in ("Revenue", "NOPAT", "Unlevered free cash flow"))
    _put(d, f"B{R['Revenue']}", last_rev, MONEY, BLUE)
    _put(d, f"B{R['EBIT']}", float(hist["ebit"].iloc[-1]), MONEY, BLUE)
    for i, c in enumerate(cols[1:]):
        prev = cols[i]
        _put(d, f"{c}{R['  growth']}", "=" + sel("Revenue growth", i), PCT)
        _put(d, f"{c}{R['Revenue']}", f"={prev}{R['Revenue']}*(1+{c}{R['  growth']})", MONEY)
        _put(d, f"{c}{R['  margin']}", "=" + sel("EBIT margin", i), PCT)
        _put(d, f"{c}{R['EBIT']}", f"={c}{R['Revenue']}*{c}{R['  margin']}", MONEY)
        _put(d, f"{c}{R['  tax rate']}", "=" + sel("Cash tax rate", i), PCT)
        _put(d, f"{c}{R['Taxes on EBIT']}", f"=MAX({c}{R['EBIT']},0)*{c}{R['  tax rate']}", MONEY)
        _put(d, f"{c}{R['NOPAT']}", f"={c}{R['EBIT']}-{c}{R['Taxes on EBIT']}", MONEY)
        _put(d, f"{c}{R['(+) D&A']}", f"={c}{R['Revenue']}*{scal('D&A % of revenue')}", MONEY)
        _put(d, f"{c}{R['(-) CapEx']}", f"={c}{R['Revenue']}*{scal('CapEx % of revenue')}", MONEY)
        _put(d, f"{c}{R['(-) Increase in working capital']}",
             f"=({c}{R['Revenue']}-{prev}{R['Revenue']})*{scal('Working capital % of new revenue')}", MONEY)
        net = (f"({c}{R['(-) CapEx']}-{c}{R['(+) D&A']}+{c}{R['(-) Increase in working capital']})")
        _put(d, f"{c}{R['(-) Extra capital to fund growth']}",
             f"=MAX(MAX({c}{R['Revenue']}-{prev}{R['Revenue']},0)/{scal('Max sales-to-capital')}-{net},0)", MONEY)
        _put(d, f"{c}{R['Unlevered free cash flow']}",
             f"={c}{R['NOPAT']}+{c}{R['(+) D&A']}-{c}{R['(-) CapEx']}-{c}{R['(-) Increase in working capital']}"
             f"-{c}{R['(-) Extra capital to fund growth']}", MONEY)
        _put(d, f"{c}{R['Discount period (mid-year)']}", i + 0.5, "0.0")
        _put(d, f"{c}{R['Discount factor']}", f"=(1+$B$23)^-{c}{R['Discount period (mid-year)']}", "0.0000")
        _put(d, f"{c}{R['PV of FCF']}", f"={c}{R['Unlevered free cash flow']}*{c}{R['Discount factor']}", MONEY)
    last = cols[-1]
    out = [
        (22, "WACC applied", f"=Inputs!B25+{scal('WACC adjustment vs CAPM')}", PCT),
        (23, "WACC (used above)", "=B22", PCT),
        (24, "Terminal growth", "=" + scal("Terminal growth"), PCT),
        (25, "RONIC (return on new capital)", f"=B22+{scal('RONIC minus WACC')}", PCT),
        (26, "Terminal FCF = NOPAT x (1+g) x (1 - g/RONIC)", f"={last}{R['NOPAT']}*(1+B24)*(1-B24/B25)", MONEY),
        (27, "Terminal value", "=B26/(B22-B24)", MONEY),
        (28, "PV of terminal value (mid-year)", f"=B27*(1+B22)^-({years}-0.5)", MONEY),
        (29, "Sum of PV of FCF", f"=SUM({cols[1]}{R['PV of FCF']}:{last}{R['PV of FCF']})", MONEY),
        (30, "Enterprise value", "=B28+B29", MONEY),
        (31, "  terminal value as % of EV", "=B28/B30", PCT),
        (32, "(+) Cash and short-term investments", "=Inputs!B7", MONEY),
        (33, "(-) Debt", "=-Inputs!B8", MONEY),
        (34, "(-) Minority interest", "=-Inputs!B9", MONEY),
        (35, "(-) Preferred stock", "=-Inputs!B10", MONEY),
        (36, "Equity value", "=SUM(B30,B32:B35)", MONEY),
        (37, "Value per share", "=B36/Inputs!B6", PRICE),
        (38, "Share price", "=Inputs!B5", PRICE),
        (39, "Upside / (downside)", "=B37/B38-1", PCT),
    ]
    _put(d, "A21", "VALUATION", font=BOLD, fill=HEAD)
    for r, name, f, fmt in out:
        _label(d, r, name, bold=r in (30, 36, 37, 39))
        _put(d, f"B{r}", f, fmt)
    d["B37"].fill = KEY

    # Resultados del motor (valores del dia)
    s = wb.create_sheet("Engine results")
    s.column_dimensions["A"].width = 44
    _put(s, "A1", "Computed by the engine on the run date (values, not formulas)", font=TITLE)
    r = 3
    sc = v["scenarios"]
    _put(s, f"A{r}", "SCENARIOS", font=BOLD, fill=HEAD)
    r += 1
    for k in ("bear", "base", "bull"):
        res = sc["cases"][k]
        _label(s, r, f"{k.title()} (weight {sc['weights'][k]:.0%})")
        _put(s, f"B{r}", res["per_share"], PRICE)
        _put(s, f"C{r}", res["upside"], PCT)
        r += 1
    _label(s, r, "Probability-weighted value", bold=True)
    _put(s, f"B{r}", v["expected"], PRICE, fill=KEY)
    _put(s, f"C{r}", v["expected_upside"], PCT)
    r += 2
    mc = v["monte_carlo"]
    if mc.get("n"):
        _put(s, f"A{r}", f"MONTE CARLO ({mc['n']:,} draws)", font=BOLD, fill=HEAD)
        r += 1
        for name, key in (("5th percentile", "p05"), ("Median", "p50"), ("95th percentile", "p95")):
            _label(s, r, name)
            _put(s, f"B{r}", mc[key], PRICE)
            r += 1
        _label(s, r, "Probability value > price")
        _put(s, f"B{r}", mc["prob_above_price"], PCT)
        r += 2
    rev = v["reverse"]
    _put(s, f"A{r}", "REVERSE DCF: what the price implies", font=BOLD, fill=HEAD)
    r += 1
    if rev.get("solved"):
        for name, key in (("Implied year-1 revenue growth", "implied_year1_growth"),
                          (f"Implied {rev['years']}-year revenue CAGR", "implied_cagr"),
                          (f"Base case {rev['years']}-year revenue CAGR", "base_cagr")):
            _label(s, r, name)
            _put(s, f"B{r}", rev[key], PCT)
            r += 1
    else:
        _put(s, f"A{r}", rev.get("note", ""))
        r += 1
    r += 1
    for title, grid, rfmt, cfmt in (("VALUE PER SHARE: WACC shift (rows) x terminal g shift (cols)",
                                     v["sens_wacc_g"], PCT, PCT),
                                    ("VALUE PER SHARE: growth shift (rows) x margin shift (cols)",
                                     v["sens_growth_margin"], PCT, PCT)):
        _put(s, f"A{r}", title, font=BOLD, fill=HEAD)
        r += 1
        for j, c in enumerate(grid.columns):
            _put(s, f"{get_column_letter(2 + j)}{r}", float(c), cfmt, BOLD)
        r += 1
        for idx, row_vals in grid.iterrows():
            _put(s, f"A{r}", float(idx), rfmt, BOLD)
            for j, x in enumerate(row_vals):
                _put(s, f"{get_column_letter(2 + j)}{r}", None if pd.isna(x) else float(x), PRICE)
            r += 1
        r += 1

    # Historia
    h = wb.create_sheet("History")
    h.column_dimensions["A"].width = 26
    _put(h, "A1", f"Annual financial statements ({ccy} millions) - {v.get('source', '')}", font=TITLE)
    for j, idx in enumerate(hist.index):
        _put(h, f"{get_column_letter(2 + j)}3", f"{pd.Timestamp(idx):%Y-%m-%d}", font=BOLD, fill=HEAD)
    for i, col in enumerate(hist.columns, start=4):
        _label(h, i, col)
        for j, x in enumerate(hist[col]):
            _put(h, f"{get_column_letter(2 + j)}{i}", None if pd.isna(x) else float(x), MONEY)

    # Notas
    n = wb.create_sheet("Notes")
    n.column_dimensions["A"].width = 140
    _put(n, "A1", "Where each assumption comes from, and what to check", font=TITLE)
    r = 3
    for title, items in (("Assumptions", v["notes"]), ("Checks", v["flags"] or ["no flags"])):
        _put(n, f"A{r}", title, font=BOLD, fill=HEAD)
        r += 1
        for item in items:
            c = n.cell(row=r, column=1, value=str(item))
            c.alignment = Alignment(wrap_text=True)
            r += 1
        r += 1
    for sheet in wb.worksheets:
        sheet.sheet_view.showGridLines = False
        for row_cells in sheet.iter_rows(min_row=4, max_row=4):
            for c in row_cells:
                c.border = THIN
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path
