"""Lo unico de statsmodels que usa el motor: OLS con errores HAC.

No se importa `statsmodels.api`. Arrastra `statsmodels.tsa.statespace`, cuya
extension compilada bloquea el Control de aplicaciones de Windows en la maquina
del club ("Una directiva de Control de aplicaciones bloqueo este archivo"). Las
rutas directas de abajo no la cargan y dan exactamente el mismo OLS.
"""

from statsmodels.regression.linear_model import OLS  # noqa: F401
from statsmodels.tools.tools import add_constant  # noqa: F401
