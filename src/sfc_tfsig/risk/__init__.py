"""Gestion de riesgo: cuanto se puede perder, en que escenarios y por que.

- `var.py`           VaR y Expected Shortfall, y su backtest          [puro]
- `exposure.py`      covarianza, contribuciones, beta, liquidez       [puro]
- `garch.py`         GARCH(1,1) y Monte Carlo filtrado                [puro]
- `montecarlo.py`    bootstrap de la historia y simulacion parametrica [puro]
- `factor_model.py`  riesgo por sectores, estilos y especifico        [puro]
- `stress.py`        historicos, hipoteticos, inverso, VaR estresado  [puro]
- `history.py`       riesgo movil y calibracion del modelo de riesgo  [puro]
- `liquidity.py`     coste de liquidacion en crisis y capacidad       [puro]
- `limits.py`        limites de vigilancia y semaforo                 [puro]
- `analysis.py`      junta todo sobre una cartera concreta            [puro]

Nada de aqui cambia lo que se compra. Mide la cartera que el modelo ya decidio
y la compara con la politica de `config/risk.toml`.
"""
