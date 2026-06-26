# =============================================================================
# preprocessor.py — M2: Preprocesamiento
# Proyecto: Curvas de Rendimiento Cero-Cupón TES · BVC 2015-2026
# =============================================================================
# Responsabilidad: transformar los datos crudos en un DataFrame limpio,
# enriquecido y ordenado, listo para ser consumido por el bootstrapper.
#
# Operaciones en orden:
#   1. Calcular VWAP por (fecha_neg, nemotecnico)
#   2. Merge con info estática (dueDate, couponRate)
#   3. Calcular tau (plazo residual en años decimales)
#   4. Filtrar bonos vencidos (tau <= 0)
#   5. Aplicar filtro de estabilidad (umbral 30 días)
#   6. Ordenar por (fecha_neg, tau) ascendente
# =============================================================================

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def calcular_vwap(df_operaciones: pd.DataFrame) -> pd.DataFrame:
    """
    Colapsa todas las operaciones del día por bono en un único registro
    con tasa ponderada por volumen (VWAP) y volumen total acumulado.

    El volumen acumulado se conserva porque es necesario para el filtro
    de estabilidad: cuando dos bonos vencen dentro del umbral de 30 días,
    se elimina el de menor volumen HISTÓRICO total.

    Args:
        df_operaciones: DataFrame crudo de operaciones (530k filas)

    Returns:
        DataFrame con columnas:
            fecha_neg, nemotecnico, tasa_vwap, volumen_dia, volumen_historico
    """
    grp = df_operaciones.groupby(['fecha_neg', 'nemotecnico'])

    # Tasa ponderada por volumen
    tasa_vwap = (
        (df_operaciones['rate'] * df_operaciones['volume'])
        .groupby([df_operaciones['fecha_neg'], df_operaciones['nemotecnico']])
        .sum()
        / grp['volume'].sum()
    ).reset_index(name='tasa_vwap')

    ###########

    

    ###########

    # Volumen total del día por bono
    volumen_dia = grp['volume'].sum().reset_index(name='volumen_dia')

    # Merge VWAP + volumen día
    df_vwap = tasa_vwap.merge(volumen_dia, on=['fecha_neg', 'nemotecnico'])

    # Volumen histórico total por bono (para filtro de estabilidad)
    vol_historico = (
        df_operaciones.groupby('nemotecnico')['volume']
        .sum()
        .reset_index(name='volumen_historico')
    )
    df_vwap = df_vwap.merge(vol_historico, on='nemotecnico')

    return df_vwap


def enriquecer_con_estatica(
    df_vwap: pd.DataFrame,
    df_estatica: pd.DataFrame
) -> pd.DataFrame:
    """
    Une el VWAP con la información estática del bono (dueDate, couponRate).
    Calcula tau = plazo residual en años decimales usando dueDate del CSV
    estático como fuente autoritativa.

    Args:
        df_vwap:     DataFrame resultado de calcular_vwap()
        df_estatica: DataFrame de info estática cargado por data_loader

    Returns:
        DataFrame enriquecido con columnas adicionales:
            dueDate, couponRate, tau
    """
    cols_estatica = ['nemotecnico', 'dueDate', 'couponRate']
    df = df_vwap.merge(
        df_estatica[cols_estatica],
        on='nemotecnico',
        how='left'
    )

    # Calcular tau en años decimales
    df['tau'] = (df['dueDate'] - df['fecha_neg']).dt.days / config.BASE_DIAS

    # Fallback: si dueDate no está en el CSV estático, parsear el nemotécnico
    # (solo para bonos que no tienen match en el CSV — no debería ocurrir)
    mascara_sin_due = df['dueDate'].isna()
    if mascara_sin_due.any():
        n_sin_due = mascara_sin_due.sum()
        print(f"  ⚠  {n_sin_due} registros sin dueDate en CSV estático "
              f"— aplicando fallback por nemotécnico")
        sufijos = df.loc[mascara_sin_due, 'nemotecnico'].str[-6:]
        dias    = sufijos.str[0:2].astype(int)
        meses   = sufijos.str[2:4].astype(int)
        anios   = 2000 + sufijos.str[4:6].astype(int)
        due_fallback = pd.to_datetime({
            'year': anios, 'month': meses, 'day': dias
        })
        df.loc[mascara_sin_due, 'dueDate'] = due_fallback
        df.loc[mascara_sin_due, 'tau'] = (
            (due_fallback - df.loc[mascara_sin_due, 'fecha_neg']).dt.days
            / config.BASE_DIAS
        )

    return df


def filtrar_vencidos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina registros donde el bono ya venció en la fecha de negociación
    (tau <= 0). Estos registros son inconsistencias de datos.
    """
    n_antes = len(df)
    df = df[df['tau'] > 0].copy()
    n_eliminados = n_antes - len(df)
    if n_eliminados > 0:
        print(f"  ⚠  {n_eliminados} registros con tau <= 0 eliminados")
    return df


def aplicar_filtro_estabilidad(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada fecha de negociación, detecta pares de bonos cuyas fechas
    de vencimiento están separadas por menos de UMBRAL_ESTABILIDAD_DIAS días.
    En cada par conflictivo, elimina el bono con menor volumen histórico.

    Esto previene inestabilidad numérica en el bootstrapping: si dos bonos
    vencen casi al mismo tiempo, el 'tramo nuevo' que aporta el segundo es
    tan pequeño que la resolución de f se vuelve mal condicionada.

    La estrategia es eficiente: en lugar de la matriz O(N²) del script
    original, ordena por dueDate y compara solo pares adyacentes — suficiente
    porque el conflicto solo puede ocurrir entre vecinos en la línea de tiempo.

    Args:
        df: DataFrame enriquecido con tau y volumen_historico

    Returns:
        DataFrame con bonos conflictivos eliminados
    """
    umbral = config.UMBRAL_ESTABILIDAD_DIAS
    registros_a_eliminar = set()
    n_conflictos = 0

    for fecha, grupo in df.groupby('fecha_neg'):
        # Ordenar por dueDate ascendente para comparar adyacentes
        grupo_ord = grupo.sort_values('dueDate').reset_index()

        for i in range(len(grupo_ord) - 1):
            bono_actual   = grupo_ord.iloc[i]
            bono_siguiente = grupo_ord.iloc[i + 1]

            diferencia_dias = (
                bono_siguiente['dueDate'] - bono_actual['dueDate']
            ).days

            if diferencia_dias < umbral:
                n_conflictos += 1
                # Eliminar el de menor volumen histórico
                if bono_actual['volumen_historico'] >= bono_siguiente['volumen_historico']:
                    idx_eliminar = bono_siguiente['index']
                else:
                    idx_eliminar = bono_actual['index']
                registros_a_eliminar.add(idx_eliminar)

    if n_conflictos > 0:
        print(f"  ⚠  {n_conflictos} conflictos de estabilidad detectados "
              f"— {len(registros_a_eliminar)} registros eliminados")

    return df.drop(index=registros_a_eliminar).copy()


def ordenar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ordena el DataFrame por fecha_neg ascendente y tau ascendente dentro
    de cada fecha. Este orden es mandatorio para el bootstrapping:
    los bonos deben procesarse de menor a mayor madurez.
    """
    return df.sort_values(
        by=['fecha_neg', 'tau'],
        ascending=True
    ).reset_index(drop=True)


def preprocesar(
    df_operaciones: pd.DataFrame,
    df_estatica: pd.DataFrame
) -> pd.DataFrame:
    """
    Pipeline completo de preprocesamiento. Función principal del módulo.

    Args:
        df_operaciones: salida de data_loader.cargar_operaciones()
        df_estatica:    salida de data_loader.cargar_info_estatica()

    Returns:
        DataFrame limpio con columnas:
            fecha_neg | nemotecnico | tasa_vwap | volumen_dia |
            volumen_historico | dueDate | couponRate | tau
        Ordenado por (fecha_neg ASC, tau ASC).
        Listo para ser sliceado por fecha y enviado al bootstrapper.
    """
    print("=" * 55)
    print("  PREPROCESSOR — iniciando pipeline")
    print("=" * 55)

    print("\n[1/5] Calculando VWAP...")
    df = calcular_vwap(df_operaciones)
    print(f"       {len(df_operaciones):,} operaciones → {len(df):,} "
          f"registros VWAP (fecha, bono)")

    print("\n[2/5] Enriqueciendo con info estática y calculando tau...")
    df = enriquecer_con_estatica(df, df_estatica)

    print("\n[3/5] Filtrando bonos vencidos (tau <= 0)...")
    df = filtrar_vencidos(df)
    print(f"       Registros restantes: {len(df):,}")

    print("\n[4/5] Aplicando filtro de estabilidad (umbral "
          f"{config.UMBRAL_ESTABILIDAD_DIAS} días)...")
    df = aplicar_filtro_estabilidad(df)
    print(f"       Registros restantes: {len(df):,}")

    print("\n[5/5] Ordenando por (fecha_neg, tau)...")
    df = ordenar(df)

    # Estadísticas finales
    n_dias   = df['fecha_neg'].nunique()
    n_bonos  = df['nemotecnico'].nunique()
    tau_min  = df['tau'].min()
    tau_max  = df['tau'].max()

    print(f"\n{'=' * 55}")
    print(f"  Preprocesamiento completado.")
    print(f"  Días hábiles con datos: {n_dias:,}")
    print(f"  Bonos únicos activos:   {n_bonos}")
    print(f"  Tau mínimo:             {tau_min:.4f} años")
    print(f"  Tau máximo:             {tau_max:.4f} años")
    print(f"{'=' * 55}")

    return df


def obtener_dia(df_preprocesado: pd.DataFrame, fecha: str) -> pd.DataFrame:
    """
    Extrae el subconjunto de bonos activos para una fecha específica.
    Es la función que usa el orquestador en cada iteración del loop.

    Args:
        df_preprocesado: salida de preprocesar()
        fecha: string en formato 'YYYY-MM-DD'

    Returns:
        DataFrame filtrado para esa fecha, ordenado por tau ascendente.
        Retorna DataFrame vacío si la fecha no tiene datos.
    """
    fecha_dt = pd.to_datetime(fecha)
    return df_preprocesado[
        df_preprocesado['fecha_neg'] == fecha_dt
    ].reset_index(drop=True)


# -----------------------------------------------------------------------------
# BLOQUE DE PRUEBA DIRECTO
# Ejecutar: python src/preprocessor.py
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    from data_loader import cargar_todo

    df_op, df_est = cargar_todo()
    df_pre = preprocesar(df_op, df_est)

    # Validar fecha de prueba
    fecha_prueba = config.FECHA_PRUEBA
    df_dia = obtener_dia(df_pre, fecha_prueba)

    print(f"\n--- Bonos activos el {fecha_prueba} ---")
    cols_vista = ['nemotecnico', 'tasa_vwap', 'tau', 'couponRate', 'dueDate']
    print(df_dia[cols_vista].to_string(index=False))
    print(f"\nTotal bonos activos ese día: {len(df_dia)}")