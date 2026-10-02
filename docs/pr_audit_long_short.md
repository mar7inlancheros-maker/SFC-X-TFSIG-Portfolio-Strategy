# PR: auditoría del motor long/short

## Cómo abrirlo

No había credenciales de GitHub en la máquina de la auditoría ni `gh`
instalado. Desde una terminal con acceso:

```bash
git push -u origin audit/long-short-review

# con GitHub CLI:
gh pr create --base main --head audit/long-short-review \
  --title "Auditoría del motor long/short (octubre de 2026)" \
  --body-file docs/pr_audit_long_short.md

# sin gh, abre la página de creación del PR y pega el texto de abajo:
# https://github.com/mar7inlancheros-maker/SFC-X-TFSIG-Portfolio-Strategy/compare/main...audit/long-short-review
```

No hay que mergearlo hasta que el comité lo revise.

---

## Auditoría del motor long/short (octubre de 2026)

Revisión del Quantitative Long/Short Research Engine (`src/quant_engine/`)
frente a las 35 secciones de su especificación original, con las correcciones
aprobadas por el comité.

- **Ningún valor de `config/quant_engine.yaml` ha cambiado.**
- No se toca `src/sfc_tfsig/` ni `strategy.toml`.
- El detalle de cada cambio está en `docs/audit_long_short_2026-10.md`: qué
  estaba mal, evidencia, cambio, test y cifras antes → después.

### Lo importante

- **`risk_parity` no era neutral en dólares (#12).** Era el default y tenía
  una neta de +0,23. Ahora hace ERC dentro de cada pata, con patas de
  100% / 100%. Su Sharpe en la trayectoria A baja de 1,26 a 1,10: parte del
  anterior venía de estar neto largo en un mercado alcista. La beta ex-ante
  queda en −0,25, porque la pata corta tiene más beta.
- **La sesión en curso entraba como cierre (#7).** Con el mercado abierto,
  la barra intradía de Yahoo movía precios de entrada, stops, VaR y backtest
  según la hora de la corrida. El 2 de octubre a las 12:00, NKE entraba a
  33,22 frente a su cierre de 35,15. Ahora los datos llegan hasta la última
  sesión cerrada.
- **El Modo A ya no se presenta como evidencia (#8).** La trayectoria
  hipotética (la cesta de hoy aplicada al pasado) va rotulada en terminal,
  txt, html, JSON y gráficos. El resumen final ya no cita su Sharpe ni el
  rango de la robustez.

### Cambios, un commit por número

- **Sin cambio de resultados:**
  - #23: ruff.
  - #21: ancho fijo del reporte interactivo. Antes, 429 celdas cortadas.
  - #22: pesos efectivos del Quant Score.
  - #16: aviso cuando el plan no queda neutral.
  - #8: rótulo del Modo A y resumen final.
- **Con cambio de resultados** (corridas antes y después en el log):
  - #7: sesión en curso.
  - #4: Sortino del backtest con la definición del motor. En `risk_parity`
    pasa de 2,10 a 1,97 antes de #12.
  - #15: el plan con `max_sharpe` ya no cae siempre a `equal_weight`.
  - #12: `risk_parity` por pata.
  - #6: con WRDS, acciones e interés corto ajustados por splits
    posteriores. **Probado solo con datos sintéticos: WRDS no estaba
    disponible.**
- **Fila de sensibilidad de `borrow_cost`** en la robustez (0,25% / 1% / 3%),
  sin cambiar el default.

### Métricas (default `risk_parity`, trayectoria A, hipotética)

| Etapa | Neta | Sharpe | Sortino | CAGR | Max DD | Rango robustez |
|---|---|---|---|---|---|---|
| Referencia `d49f706` | +0,234 | 1,263 | 2,095 | 33,36% | −17,3% | 0,815 – 1,395 |
| Final `f431572` | 0,000 | 1,102 | 1,685 | 29,33% | −21,1% | 0,471 – 1,395 |

### Para el comité

- **#26, alta:** el plan de operación emite órdenes (BUY / SELL SHORT,
  acciones, "do not trade") y el veredicto FLIP opera contra el research.
  Eso choca con las secciones 17 y 34. Hay una propuesta para dejarlo
  descriptivo.
- **#25:** la robustez se calcula sobre la trayectoria A. La propuesta es
  correrla sobre el Modo B cuando haya señales con fecha.
- **#27, #28 y #30:** entrada por defecto (la primera mitad va LONG),
  retornos anuales y mensuales sin reportar, mediana sin mostrar.
- **Parámetros, no aplicados:** `max_position` 0,35 frente al 20% de la
  especificación, coste de préstamo por nombre, construcción por defecto y
  constantes al YAML.
- **Pendiente con WRDS real:** verificar #6 (`comp.secd.ajexdi`).

### Pruebas

- `pytest`: 346 passed. Del motor, 129 (eran 104).
- `ruff --select E9,F`: limpio.
- `python main.py` interactivo con todas las opciones del menú: código 0.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
