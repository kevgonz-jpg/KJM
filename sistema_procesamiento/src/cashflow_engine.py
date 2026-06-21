# =============================================================================
# cashflow_engine.py — M3: Motor de Flujos de Caja
# Proyecto: Curvas de Rendimiento Cero-Cupón TES · BVC 2015-2026
# =============================================================================
# Responsabilidad: para cada bono en una fecha de negociación dada,
# generar el vector completo de flujos de caja futuros con sus tiempos
# exactos en años decimales desde la fecha de negociación.
#
# Convención:
#   - Valor nominal normalizado a 1
#   - Cupón anual = couponRate / 100
#   - Pago final  = 1 + couponRate / 100  (principal + último cupón)
#   - Los cupones se pagan en aniversarios exactos de dueDate
#   - Base 365 días
# =============================================================================

import pandas as pd
import numpy as np
from datetime import date
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def generar_fechas_cupon(
    fecha_neg: pd.Timestamp,
    due_date: pd.Timestamp,
    coupon_rate: float
) -> list[tuple[float, float]]:
    """
    Genera el calendario completo de flujos de caja futuros para un bono
    desde la fecha de negociación hasta su vencimiento.

    Los cupones caen en aniversarios de la fecha de vencimiento, contando
    hacia atrás desde dueDate. Por ejemplo, si dueDate es 2028-04-28,
    los cupones caen cada 28 de abril: 2027-04-28, 2026-04-28, etc.

    Args:
        fecha_neg:   fecha de negociación (pd.Timestamp)
        due_date:    fecha de vencimiento del bono (pd.Timestamp)
        coupon_rate: tasa cupón en porcentaje (ej: 6.0 para 6%)

    Returns:
        Lista de tuplas (t, CF) ordenada ascendentemente por t, donde:
            t  = tiempo en años decimales desde fecha_neg
            CF = monto del flujo de caja (normalizado, nominal = 1)

        Ejemplo para TFIT16280428 (cupón 6%, vence 2028-04-28)
        negociado el 2016-10-21:
            [(1.5..., 0.06),   ← cupón abril 2018
             (2.5..., 0.06),   ← cupón abril 2019
             ...
             (11.52, 1.06)]    ← pago final abril 2028
    """
    cupon = coupon_rate / 100.0
    flujos = []

    # Generar aniversarios hacia atrás desde dueDate
    # hasta encontrar fechas posteriores a fecha_neg
    anio_venc = due_date.year
    mes_venc  = due_date.month
    dia_venc  = due_date.day

    # Iterar desde el vencimiento hacia atrás buscando fechas de cupón
    # que sean estrictamente posteriores a fecha_neg
    anio_actual = anio_venc
    fechas_cupon = []

    while True:
        try:
            fecha_cupon = pd.Timestamp(anio_actual, mes_venc, dia_venc)
        except ValueError:
            # Día inválido para ese mes (ej: 29 feb en año no bisiesto)
            # Usar último día del mes
            import calendar
            ultimo_dia = calendar.monthrange(anio_actual, mes_venc)[1]
            fecha_cupon = pd.Timestamp(anio_actual, mes_venc, ultimo_dia)

        if fecha_cupon <= fecha_neg:
            # Ya estamos antes o en la fecha de negociación — parar
            break

        fechas_cupon.append(fecha_cupon)
        anio_actual -= 1

        # Seguridad: no retroceder más de 50 años
        if anio_venc - anio_actual > 50:
            break

    # Ordenar ascendentemente (del más cercano al más lejano)
    fechas_cupon.sort()

    # Construir lista de flujos
    for i, fecha_cupon in enumerate(fechas_cupon):
        t = (fecha_cupon - fecha_neg).days / config.BASE_DIAS
        es_ultimo = (fecha_cupon == due_date)

        if es_ultimo:
            # Pago final: principal + último cupón
            cf = 1.0 + cupon
        else:
            # Cupón intermedio
            cf = cupon

        flujos.append((t, cf))

    return flujos


def obtener_flujos_bono(fila: pd.Series) -> list[tuple[float, float]]:
    """
    Interfaz simplificada para obtener los flujos de una fila del
    DataFrame preprocesado.

    Args:
        fila: fila del DataFrame preprocesado con columnas
              fecha_neg, dueDate, couponRate, nemotecnico

    Returns:
        Lista de tuplas (t, CF) — mismo formato que generar_fechas_cupon()
    """
    return generar_fechas_cupon(
        fecha_neg   = fila['fecha_neg'],
        due_date    = fila['dueDate'],
        coupon_rate = fila['couponRate']
    )


def flujos_a_dataframe(
    flujos: list[tuple[float, float]],
    nemotecnico: str = ''
) -> pd.DataFrame:
    """
    Convierte la lista de flujos a DataFrame para visualización y debugging.

    Args:
        flujos:      salida de generar_fechas_cupon()
        nemotecnico: nombre del bono (opcional, para el encabezado)

    Returns:
        DataFrame con columnas: t_anos, flujo, tipo
    """
    rows = []
    for i, (t, cf) in enumerate(flujos):
        es_final = (i == len(flujos) - 1)
        rows.append({
            't_anos': round(t, 6),
            'flujo':  round(cf, 6),
            'tipo':   'FINAL (principal + cupón)' if es_final else 'cupón'
        })
    df = pd.DataFrame(rows)
    if nemotecnico:
        df.insert(0, 'nemotecnico', nemotecnico)
    return df


# -----------------------------------------------------------------------------
# BLOQUE DE PRUEBA DIRECTO
# Ejecutar: python src/cashflow_engine.py
#
# Validación manual para TFIT16280428 el 2016-10-21:
#   - dueDate: 2028-04-28, coupon: 6%
#   - Cupones futuros en: 2017-04-28, 2018-04-28, ..., 2027-04-28
#   - Pago final: 2028-04-28 (1.06)
#   - Tau total: (2028-04-28 - 2016-10-21).days / 365 = 11.526 años ✓
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_loader import cargar_todo
    from preprocessor import preprocesar, obtener_dia

    # Cargar y preprocesar
    df_op, df_est = cargar_todo()
    df_pre = preprocesar(df_op, df_est)
    df_dia = obtener_dia(df_pre, config.FECHA_PRUEBA)

    print(f"\n{'=' * 65}")
    print(f"  CASHFLOW ENGINE — validación para {config.FECHA_PRUEBA}")
    print(f"{'=' * 65}")

    # Mostrar flujos de cada bono activo ese día
    for _, fila in df_dia.iterrows():
        nemo   = fila['nemotecnico']
        flujos = obtener_flujos_bono(fila)
        df_f   = flujos_a_dataframe(flujos, nemo)

        print(f"\n  {nemo}  |  cupón: {fila['couponRate']}%  "
              f"|  vence: {fila['dueDate'].date()}"
              f"  |  tau: {fila['tau']:.4f} años"
              f"  |  {len(flujos)} flujos")
        print(df_f.to_string(index=False))

    # Validación específica TFIT16280428
    print(f"\n{'=' * 65}")
    print("  VALIDACIÓN MANUAL — TFIT16280428")
    print(f"{'=' * 65}")
    fila_ref = df_dia[df_dia['nemotecnico'] == 'TFIT16280428'].iloc[0]
    flujos_ref = obtener_flujos_bono(fila_ref)
    print(f"  Esperado: 12 flujos (11 cupones + 1 pago final)")
    print(f"  Obtenido: {len(flujos_ref)} flujos")
    print(f"  Último flujo: t={flujos_ref[-1][0]:.4f} años, "
          f"CF={flujos_ref[-1][1]:.4f}")
    print(f"  Tau desde preprocessor: {fila_ref['tau']:.4f} años")
    print(f"  ¿Coinciden? "
          f"{'✓' if abs(flujos_ref[-1][0] - fila_ref['tau']) < 0.001 else '✗ REVISAR'}")