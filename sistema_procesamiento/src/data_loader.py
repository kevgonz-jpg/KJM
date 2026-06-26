# =============================================================================
# data_loader.py — M1: Carga de Datos
# Proyecto: Curvas de Rendimiento Cero-Cupón TES · BVC 2015-2026
# =============================================================================
# Responsabilidad única: conectar a las 3 SQLite, cargar y concatenar el
# dataset completo de operaciones, y cargar la información estática de bonos.
# La carga se hace UNA SOLA VEZ al inicio del proceso — los datos completos
# se mantienen en memoria durante todo el barrido histórico.
# =============================================================================

import sqlite3
import pandas as pd
import sys
import os

# Permite importar config.py desde cualquier contexto
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def cargar_operaciones() -> pd.DataFrame:
    """
    Conecta a las 3 bases de datos SQLite y retorna un DataFrame único
    con todas las operaciones del período 2015-2026, ordenado por fecha
    ascendente.

    Returns:
        pd.DataFrame con columnas:
            fecha_neg   (datetime64)
            nemotecnico (str)
            time        (str)
            rate        (float)
            price       (float)
            volume      (float)
    """
    rutas = {
        '2015-2018': config.RUTA_DB_2015_2018,
        '2019-2022': config.RUTA_DB_2019_2022,
        '2023-2026': config.RUTA_DB_2023_2026,
    }

    fragmentos = []

    for periodo, ruta in rutas.items():
        try:
            conn = sqlite3.connect(ruta)
            df = pd.read_sql_query("SELECT * FROM operaciones", conn)
            conn.close()
            fragmentos.append(df)
            print(f"  ✓  {periodo}  →  {len(df):>7,} operaciones cargadas")
        except Exception as e:
            raise RuntimeError(
                f"[data_loader] Error cargando {periodo} desde {ruta}:\n{e}"
            )

    # Concatenar los tres períodos
    df_total = pd.concat(fragmentos, ignore_index=True)

    # Convertir fecha_neg a datetime — facilita todos los filtros posteriores
    df_total['fecha_neg'] = pd.to_datetime(df_total['fecha_neg'])

    # Ordenar ascendentemente por fecha, nemotécnico y hora
    df_total = df_total.sort_values(
        by=['fecha_neg', 'nemotecnico', 'time'],
        ascending=True
    ).reset_index(drop=True)

    # Persistir dataset consolidado para auditoría y reutilización externa.
    ruta_csv_operaciones = os.path.join(config.OUTPUTS_DIR, 'operaciones_consolidadas.csv')
    df_total.to_csv(ruta_csv_operaciones, index=False)

    print(f"\n  TOTAL  →  {len(df_total):>7,} operaciones")
    print(f"  Período: {df_total['fecha_neg'].min().date()} → "
          f"{df_total['fecha_neg'].max().date()}")
    print(f"  Bonos únicos: {df_total['nemotecnico'].nunique()}")
    print(f"  CSV exportado: {ruta_csv_operaciones}")

    return df_total


def cargar_info_estatica() -> pd.DataFrame:
    """
    Carga el CSV de información estática de los 29 bonos TFIT.
    Convierte dueDate e issueDate a datetime para facilitar
    el cálculo de tau en el preprocesador.

    Returns:
        pd.DataFrame con una fila por bono y columnas estructurales:
            nemotecnico, dueDate, couponRate, issueDate, etc.
    """
    try:
        df = pd.read_csv(config.RUTA_INFO_ESTATICA)
    except Exception as e:
        raise RuntimeError(
            f"[data_loader] Error cargando info estática:\n{e}"
        )

    # Convertir fechas a datetime
    df['dueDate']   = pd.to_datetime(df['dueDate'],   errors='coerce')
    df['issueDate'] = pd.to_datetime(df['issueDate'], errors='coerce')

    # couponRate viene como string tipo "7.25%" o como float — normalizar a float
    if df['couponRate'].dtype == object:
        df['couponRate'] = (
            df['couponRate']
            .str.replace('%', '', regex=False)
            .str.strip()
            .astype(float)
        )

    print(f"  ✓  Info estática  →  {len(df)} bonos cargados")
    print(f"  Vencimientos: {df['dueDate'].min().date()} → "
          f"{df['dueDate'].max().date()}")
    print(f"  Cupones: {df['couponRate'].min()}% – {df['couponRate'].max()}%")

    return df


def cargar_todo() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Función principal del módulo. Carga operaciones e info estática
    en una sola llamada y retorna ambos DataFrames listos para el
    preprocesador.

    Returns:
        tuple: (df_operaciones, df_estatica)

    Uso típico desde el orquestador:
        df_op, df_est = data_loader.cargar_todo()
    """
    print("=" * 55)
    print("  DATA_LOADER — cargando datos")
    print("=" * 55)

    print("\n[1/2] Operaciones de mercado:")
    df_operaciones = cargar_operaciones()

    print("\n[2/2] Información estática de bonos:")
    df_estatica = cargar_info_estatica()

    print("\n" + "=" * 55)
    print("  Carga completada exitosamente.")
    print("=" * 55)

    return df_operaciones, df_estatica


# -----------------------------------------------------------------------------
# BLOQUE DE PRUEBA DIRECTO
# Ejecutar este archivo directamente para validar la carga:
#   python src/data_loader.py
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    df_op, df_est = cargar_todo()

    print("\n--- Vista previa operaciones ---")
    print(df_op.head(3).to_string())

    print("\n--- Vista previa info estática ---")
    print(df_est[['nemotecnico', 'dueDate', 'couponRate']].to_string())