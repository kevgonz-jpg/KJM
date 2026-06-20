"""
data_loader.py
==============
Carga y preprocesamiento de datos TES colombianos desde CSV.

Formato esperado (curvas_panel_historico_limpio.csv):
    fecha, r_2yr, r_3yr, r_5yr, r_7yr, r_10yr, r_15yr

Funciones exportadas
--------------------
load_tes_data(csv_path, ...)   → (yields, dates, maturities)
describe_data(yields, dates, maturities)

NOTA SOBRE IMPUTACIÓN
---------------------
Se restringe el índice a días hábiles con observaciones reales (BVC).
No se imputan fines de semana ni festivos.
"""

import numpy as np
import pandas as pd


# Columnas y plazos por defecto (formato del proyecto)
DEFAULT_YIELD_COLS = ["r_2yr", "r_3yr", "r_5yr", "r_7yr", "r_10yr", "r_15yr"]
DEFAULT_MATURITIES = np.array([2., 3., 5., 7., 10., 15.])
DEFAULT_DATE_COL   = "fecha"


# ═══════════════════════════════════════════════════════════════════════════════
# CARGA PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def load_tes_data(csv_path: str,
                  date_col:      str             = DEFAULT_DATE_COL,
                  yield_cols:    list[str] | None = None,
                  maturities_yr: np.ndarray | None = None,
                  start_date:    str | None = None,
                  end_date:      str | None = None) -> tuple:
    """
    Carga datos TES desde un CSV y devuelve arreglos listos para el modelo.

    Parámetros
    ----------
    csv_path      : ruta al archivo, e.g. 'datos/curvas_panel_historico_limpio.csv'
    date_col      : nombre de la columna de fecha (default: 'fecha')
    yield_cols    : lista de columnas de rendimiento.
                    Si None, usa ["r_2yr","r_3yr","r_5yr","r_7yr","r_10yr","r_15yr"]
    maturities_yr : plazos en años correspondientes a yield_cols.
                    Si None, usa [2, 3, 5, 7, 10, 15]
    start_date    : fecha de inicio 'YYYY-MM-DD' (inclusive), opcional
    end_date      : fecha de fin   'YYYY-MM-DD' (inclusive), opcional

    Retorna
    -------
    yields     : np.ndarray (T, N)   rendimientos (mismas unidades que el CSV)
    dates      : pd.DatetimeIndex    solo días con datos completos
    maturities : np.ndarray (N,)     plazos en años
    """
    if yield_cols is None:
        yield_cols = DEFAULT_YIELD_COLS
    if maturities_yr is None:
        maturities_yr = DEFAULT_MATURITIES

    df = pd.read_csv(csv_path, parse_dates=[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # Filtro de fechas
    if start_date:
        df = df[df[date_col] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df[date_col] <= pd.Timestamp(end_date)]

    # Solo filas con todos los plazos completos (sin NaN)
    df = df.dropna(subset=yield_cols).reset_index(drop=True)

    # Solo días hábiles (lunes–viernes)
    mask = df[date_col].dt.dayofweek < 5
    df   = df[mask].reset_index(drop=True)

    yields     = df[yield_cols].values.astype(float)
    dates      = pd.DatetimeIndex(df[date_col].values)
    maturities = np.asarray(maturities_yr, dtype=float)

    return yields, dates, maturities


# ═══════════════════════════════════════════════════════════════════════════════
# DESCRIPCIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def describe_data(yields: np.ndarray,
                  dates: pd.DatetimeIndex,
                  maturities: np.ndarray) -> None:
    """Imprime resumen estadístico de la muestra."""
    T, N = yields.shape
    print(f"\n{'=' * 52}")
    print(f"  Dataset TES Colombia")
    print(f"{'=' * 52}")
    print(f"  Periodos    : {T} días hábiles")
    print(f"  Plazos      : {N}  → {maturities.tolist()} años")
    print(f"  Fecha inicio: {str(dates[0])[:10]}")
    print(f"  Fecha fin   : {str(dates[-1])[:10]}")
    print(f"\n  {'Plazo':>7}  {'Media':>7}  {'Std':>7}  {'Min':>7}  {'Max':>7}")
    print(f"  {'─' * 42}")
    for i, m in enumerate(maturities):
        y = yields[:, i]
        print(f"  {m:>5.0f}Y  {y.mean():>7.3f}  {y.std():>7.3f}  "
              f"{y.min():>7.3f}  {y.max():>7.3f}")
    nan_count = np.isnan(yields).sum()
    if nan_count:
        print(f"\n  ⚠ NaN detectados: {nan_count} ({nan_count/yields.size*100:.1f}%)")
    else:
        print(f"\n  Sin valores faltantes")
