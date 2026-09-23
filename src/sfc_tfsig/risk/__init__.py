"""Gestion de riesgo: cuanto se puede perder, en que escenarios y por que.

- `var.py`         VaR y Expected Shortfall, y su backtest          [puro]
- `exposure.py`    covarianza, contribuciones, beta, liquidez       [puro]
- `montecarlo.py`  bootstrap de la historia y simulacion parametrica [puro]
- `stress.py`      escenarios historicos, hipoteticos e inversos    [puro]
- `limits.py`      limites de vigilancia y semaforo                 [puro]
- `analysis.py`    junta todo sobre una cartera concreta            [puro]

Nada de aqui cambia lo que se compra. Mide la cartera que el modelo ya decidio
y la compara con la politica de `config/risk.toml`.
"""
