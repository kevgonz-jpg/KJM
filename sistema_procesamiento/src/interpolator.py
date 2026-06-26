# =============================================================================
# interpolator.py — M5: Interpolación a Nodos Estándar
# Proyecto: Curvas de Rendimiento Cero-Cupón TES · BVC 2015-2026
# =============================================================================
# Responsabilidad: recibir los N puntos spot discretos que produce el
# bootstrapper (en los tau exactos de los bonos activos) e interpolar
# a los 9 nodos fijos configurados en config.py.
#
# Método: interpolación lineal entre puntos discretos.
# Extrapolación: lineal usando los dos puntos extremos más cercanos.
# Límite: si el nodo está más de MAX_EXTRAPOLACION_ANOS fuera del rango
#         de bonos activos ese día, se registra NaN.
#
# Los Unsmoothed Fama-Bliss Yields son por definición puntos discretos
# sin suavizar. La interpolación lineal preserva esa característica.
# M5 puede reemplazarse por spline o Nelson-Siegel en el futuro sin
# tocar ningún otro módulo.
# =============================================================================

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def interpolar_curva(
    puntos_spot: list[dict],
    fecha: pd.Timestamp = None,
    verbose: bool = False
) -> dict:
    """
    Interpola los puntos spot discretos del bootstrapper a los nodos
    estándar definidos en config.NODOS_INTERPOLACION.

    Args:
        puntos_spot: lista de dicts producida por bootstrapper.bootstrap_dia()
                     Cada dict debe tener al menos 'tau' y 'r_spot'.
        fecha:       fecha del día (para el log) — opcional
        verbose:     imprime detalle del proceso

    Returns:
        dict con estructura:
        {
            'fecha':   Timestamp,
            'r_1yr':   float | NaN,
            'r_2yr':   float | NaN,
            ...
            'r_30yr':  float | NaN,
            'n_puntos_discretos': int,
            'tau_min': float,
            'tau_max': float,
            'advertencias': list[str]
        }
    """
    advertencias = []

    # Extraer arrays ordenados de tau y r_spot
    taus   = np.array([p['tau']   for p in puntos_spot])
    rspots = np.array([p['r_spot'] for p in puntos_spot])

    # Ordenar por tau ascendente (garantía adicional)
    orden  = np.argsort(taus)
    taus   = taus[orden]
    rspots = rspots[orden]

    tau_min = taus[0]
    tau_max = taus[-1]
    n_pts   = len(taus)

    if verbose:
        print(f"\n  Interpolando {n_pts} puntos discretos")
        print(f"  Rango disponible: [{tau_min:.3f}, {tau_max:.3f}] años")

    resultado = {
        'fecha':              fecha,
        'n_puntos_discretos': n_pts,
        'tau_min':            tau_min,
        'tau_max':            tau_max,
        'advertencias':       advertencias,
    }

    for nodo in config.NODOS_INTERPOLACION:
        nombre_col = f'r_{nodo}yr' if nodo != 1 else 'r_1yr'
        nombre_col = f'r_{nodo}yr'

        # ── Caso 1: nodo dentro del rango → interpolación lineal ────────────
        if tau_min <= nodo <= tau_max:
            valor = float(np.interp(nodo, taus, rspots))
            resultado[nombre_col] = valor

            if verbose:
                print(f"  {nodo:>4}Y  →  {valor:.4%}  [interpolado]")

        # ── Caso 2: nodo fuera del rango → extrapolación o NaN ──────────────
        else:
            distancia = min(
                abs(nodo - tau_min),
                abs(nodo - tau_max)
            )

            if distancia > config.MAX_EXTRAPOLACION_ANOS:
                # Demasiado lejos — NaN
                resultado[nombre_col] = np.nan
                msg = (f"Nodo {nodo}Y a {distancia:.2f} años fuera del rango "
                       f"[{tau_min:.2f}, {tau_max:.2f}] — NaN")
                advertencias.append(msg)
                if verbose:
                    print(f"  {nodo:>4}Y  →  NaN  "
                          f"[fuera de rango: {distancia:.2f}Y > "
                          f"{config.MAX_EXTRAPOLACION_ANOS}Y]")

            else:
                # Extrapolación lineal con los dos puntos extremos más cercanos
                if nodo < tau_min:
                    # Extrapolación hacia la izquierda — usar los 2 primeros
                    if n_pts >= 2:
                        t1, r1 = taus[0],  rspots[0]
                        t2, r2 = taus[1],  rspots[1]
                    else:
                        # Solo un punto — extrapolación plana
                        t1, r1 = taus[0], rspots[0]
                        t2, r2 = taus[0], rspots[0]
                else:
                    # Extrapolación hacia la derecha — usar los 2 últimos
                    if n_pts >= 2:
                        t1, r1 = taus[-2], rspots[-2]
                        t2, r2 = taus[-1], rspots[-1]
                    else:
                        t1, r1 = taus[-1], rspots[-1]
                        t2, r2 = taus[-1], rspots[-1]

                # Pendiente de la extrapolación lineal
                if abs(t2 - t1) > 1e-9:
                    pendiente = (r2 - r1) / (t2 - t1)
                    valor = r1 + pendiente * (nodo - t1)
                else:
                    valor = r1  # plana si los dos puntos son iguales

                resultado[nombre_col] = float(valor)

                msg = (f"Nodo {nodo}Y extrapolado linealmente "
                       f"({distancia:.2f}Y fuera de rango)")
                advertencias.append(msg)

                if verbose:
                    print(f"  {nodo:>4}Y  →  {valor:.4%}  "
                          f"[extrapolado, Δ={distancia:.2f}Y]")

    return resultado


def resultado_a_fila(resultado: dict) -> pd.Series:
    """
    Convierte el dict de resultado a una pd.Series lista para
    acumularse en el panel histórico final.

    Args:
        resultado: salida de interpolar_curva()

    Returns:
        pd.Series con índice:
        fecha | r_1yr | r_2yr | r_3yr | r_5yr | r_7yr |
        r_10yr | r_15yr | r_20yr | r_30yr
    """
    cols_nodos = [f'r_{n}yr' for n in config.NODOS_INTERPOLACION]
    datos = {'fecha': resultado['fecha']}
    for col in cols_nodos:
        datos[col] = resultado.get(col, np.nan)
    return pd.Series(datos)


# =============================================================================
# BLOQUE DE PRUEBA DIRECTO
# Ejecutar: python src/interpolator.py
# =============================================================================
if __name__ == '__main__':
    from data_loader import cargar_todo
    from preprocessor import preprocesar, obtener_dia
    from bootstrapper import bootstrap_dia

    df_op, df_est = cargar_todo()
    df_pre = preprocesar(df_op, df_est)
    df_dia = obtener_dia(df_pre, config.FECHA_PRUEBA)

    # Correr bootstrapper
    puntos, warnings_boot = bootstrap_dia(df_dia, verbose=False)

    if warnings_boot:
        for w in warnings_boot:
            print(f"  ⚠  {w}")

    # Interpolar
    print(f"\n{'=' * 60}")
    print(f"  INTERPOLATOR — {config.FECHA_PRUEBA}")
    print(f"{'=' * 60}")

    fecha = df_dia['fecha_neg'].iloc[0]
    resultado = interpolar_curva(puntos, fecha=fecha, verbose=True)

    # Mostrar resultado final
    print(f"\n{'=' * 60}")
    print(f"  VECTOR SPOT ESTANDARIZADO — {config.FECHA_PRUEBA}")
    print(f"{'=' * 60}")
    print(f"  {'Nodo':<8} {'Tasa spot':>10} {'Estado':>15}")
    print(f"  {'─'*8} {'─'*10} {'─'*15}")

    for nodo in config.NODOS_INTERPOLACION:
        col   = f'r_{nodo}yr'
        valor = resultado.get(col, np.nan)
        if np.isnan(valor):
            estado = 'NaN (fuera rango)'
            valor_str = '       NaN'
        elif any(str(nodo) in a for a in resultado['advertencias']):
            estado = 'extrapolado'
            valor_str = f'{valor:>10.4%}'
        else:
            estado = 'interpolado'
            valor_str = f'{valor:>10.4%}'
        print(f"  {nodo:<8} {valor_str} {estado:>15}")

    # Advertencias del interpolador
    if resultado['advertencias']:
        print(f"\n  Advertencias ({len(resultado['advertencias'])}):")
        for a in resultado['advertencias']:
            print(f"    ⚠  {a}")

    # Fila del panel
    fila = resultado_a_fila(resultado)
    print(f"\n  Fila del panel histórico:")
    print(f"  {fila.to_string()}")