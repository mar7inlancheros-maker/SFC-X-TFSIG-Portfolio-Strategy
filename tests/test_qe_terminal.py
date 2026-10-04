import io

import pandas as pd
from rich.console import Console

from quant_engine.reporting import terminal


def test_las_tablas_y_secciones_se_imprimen_en_ascii():
    """La consola de Windows usa cp1252 al redirigir: todo lo impreso va en ASCII."""
    c = Console(file=io.StringIO(), width=80)
    terminal.section(c, 14, "TRADE PLAN")
    c.print(terminal.table(pd.DataFrame({"entry": [1.0, 2.0]}, index=["AAPL", "TSLA"]),
                           title="Levels", index_name="Ticker"))
    out = c.file.getvalue()
    assert "AAPL" in out
    assert all(ord(ch) < 128 for ch in out)
