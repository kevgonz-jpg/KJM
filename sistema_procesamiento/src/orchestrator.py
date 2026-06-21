# =============================================================================
# orchestrator.py — M6: Orquestador Principal
# Proyecto: Curvas de Rendimiento Cero-Cupón TES · BVC 2015-2026
# =============================================================================
# Responsabilidad: coordinar el pipeline completo.
#
# Modo prueba:     procesa FECHA_PRUEBA, muestra resultado detallado.
# Modo producción: itera sobre todos los días hábiles disponibles,
#                  acumula resultados y exporta tres archivos CSV:
#                    1. curvas_panel_historico.csv    — nodos estándar (wide)
#                    2. curvas_discretas_historico.csv — puntos crudos
#                    3. log_bootstrapping.csv          — advertencias
#
# Diseño del loop:
#   - Carga datos UNA sola vez antes del loop (M1)
#   - Preprocesa TODO el dataset UNA sola vez (M2)
#   - Por cada fecha: slice → bootstrap (M4) → interpolar (M5) → acumular
#   - Convierte a DataFrame AL FINAL para evitar concatenaciones repetidas
# =============================================================================

import pandas as pd
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_loader   import cargar_todo
from preprocessor  import preprocesar, obtener_dia
from bootstrapper  import bootstrap_dia
from interpolator  import interpolar_curva, resultado_a_fila


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def _formatear_tiempo(segundos: float) -> str:
    """Convierte segundos a string legible mm:ss."""
    m = int(segundos // 60)
    s = int(segundos % 60)
    return f"{m:02d}:{s:02d}"


def _progreso(i: int, total: int, inicio: float, intervalo: int = 100):
    """Imprime barra de progreso cada `intervalo` días."""
    if i % intervalo == 0 or i == total - 1:
        pct      = (i + 1) / total * 100
        elapsed  = time.time() - inicio
        est_total = elapsed / (i + 1) * total if i > 0 else 0
        restante  = est_total - elapsed
        print(f"  [{i+1:>4}/{total}]  {pct:>5.1f}%  "
              f"elapsed: {_formatear_tiempo(elapsed)}  "
              f"restante: {_formatear_tiempo(restante)}")


# =============================================================================
# MODO PRUEBA
# =============================================================================

def correr_prueba(df_pre: pd.DataFrame, df_est: pd.DataFrame):
    """
    Procesa únicamente config.FECHA_PRUEBA con output detallado.
    Útil para validación visual antes del barrido completo.
    """
    fecha = config.FECHA_PRUEBA
    print(f"\n{'=' * 65}")
    print(f"  MODO PRUEBA — {fecha}")
    print(f"{'=' * 65}")

    df_dia = obtener_dia(df_pre, fecha)

    if df_dia.empty:
        print(f"  ✗  Sin datos para {fecha}")
        return None, None

    print(f"  Bonos activos: {len(df_dia)}")

    # Bootstrapping
    puntos, warns_boot = bootstrap_dia(df_dia, verbose=True)

    if not puntos:
        print(f"  ✗  Bootstrapping sin resultados")
        return None, None

    # Interpolación
    fecha_ts = df_dia['fecha_neg'].iloc[0]
    resultado = interpolar_curva(puntos, fecha=fecha_ts, verbose=True)

    # Mostrar fila del panel
    fila = resultado_a_fila(resultado)
    print(f"\n{'=' * 65}")
    print(f"  FILA DEL PANEL — {fecha}")
    print(f"{'=' * 65}")
    for col, val in fila.items():
        if col == 'fecha':
            print(f"  {col:<10} {str(val.date())}")
        elif pd.isna(val):
            print(f"  {col:<10} NaN")
        else:
            print(f"  {col:<10} {val:.4%}")

    # Mostrar advertencias
    todas_warns = warns_boot + resultado['advertencias']
    if todas_warns:
        print(f"\n  Advertencias ({len(todas_warns)}):")
        for w in todas_warns:
            print(f"    ⚠  {w}")

    return pd.DataFrame(puntos), fila


# =============================================================================
# MODO PRODUCCIÓN
# =============================================================================

def correr_produccion(df_pre: pd.DataFrame):
    """
    Itera sobre todos los días hábiles disponibles en df_pre.
    Acumula resultados en listas y construye los DataFrames al final.

    Returns:
        tuple: (df_panel, df_discreto, df_log)
    """
    fechas = sorted(df_pre['fecha_neg'].unique())
    total  = len(fechas)

    print(f"\n{'=' * 65}")
    print(f"  MODO PRODUCCIÓN — {total:,} días hábiles")
    print(f"{'=' * 65}\n")

    # Acumuladores — listas son mucho más eficientes que pd.concat en loop
    filas_panel    = []   # una fila por día — nodos estándar
    filas_discreto = []   # N filas por día — puntos crudos del bootstrapper
    filas_log      = []   # advertencias por día

    dias_ok           = 0
    dias_insuficiente = 0
    dias_sin_datos    = 0

    inicio = time.time()

    for i, fecha in enumerate(fechas):
        df_dia = obtener_dia(df_pre, pd.Timestamp(fecha))

        # Día sin datos después del preprocesamiento
        if df_dia.empty:
            dias_sin_datos += 1
            filas_log.append({
                'fecha':  fecha,
                'tipo':   'SIN_DATOS',
                'mensaje': 'Sin bonos activos tras preprocesamiento'
            })
            continue

        # Bootstrapping
        puntos, warns_boot = bootstrap_dia(df_dia, verbose=False)

        # Día insuficiente
        if len(puntos) < 2:
            dias_insuficiente += 1
            filas_log.append({
                'fecha':   fecha,
                'tipo':    'INSUFICIENTE',
                'mensaje': f'Solo {len(puntos)} punto(s) spot'
            })
            continue

        # Interpolación
        fecha_ts = df_dia['fecha_neg'].iloc[0]
        resultado = interpolar_curva(puntos, fecha=fecha_ts, verbose=False)

        # Acumular fila del panel principal
        fila = resultado_a_fila(resultado)
        filas_panel.append(fila.to_dict())

        # Acumular puntos discretos crudos
        for p in puntos:
            filas_discreto.append({
                'fecha':       p['fecha'],
                'nemotecnico': p['nemotecnico'],
                'tau':         p['tau'],
                'r_spot':      p['r_spot'],
                'f_forward':   p['f_forward'],
                'n_flujos_B':  p['n_flujos_B'],
            })

        # Acumular advertencias del día
        todas_warns = warns_boot + resultado['advertencias']
        for w in todas_warns:
            filas_log.append({
                'fecha':   fecha,
                'tipo':    'ADVERTENCIA',
                'mensaje': w
            })

        dias_ok += 1
        _progreso(i, total, inicio, intervalo=200)

    # ── Construir DataFrames finales ─────────────────────────────────────────
    print(f"\n  Construyendo DataFrames finales...")

    df_panel = pd.DataFrame(filas_panel)
    if not df_panel.empty:
        df_panel['fecha'] = pd.to_datetime(df_panel['fecha'])
        df_panel = df_panel.sort_values('fecha').reset_index(drop=True)

    df_discreto = pd.DataFrame(filas_discreto)
    if not df_discreto.empty:
        df_discreto['fecha'] = pd.to_datetime(df_discreto['fecha'])
        df_discreto = df_discreto.sort_values(
            ['fecha', 'tau']
        ).reset_index(drop=True)

    df_log = pd.DataFrame(filas_log) if filas_log else pd.DataFrame(
        columns=['fecha', 'tipo', 'mensaje']
    )

    # ── Resumen final ────────────────────────────────────────────────────────
    elapsed = time.time() - inicio
    print(f"\n{'=' * 65}")
    print(f"  RESUMEN DE PRODUCCIÓN")
    print(f"{'=' * 65}")
    print(f"  Días procesados:      {total:>6,}")
    print(f"  Días OK:              {dias_ok:>6,}")
    print(f"  Días insuficientes:   {dias_insuficiente:>6,}")
    print(f"  Días sin datos:       {dias_sin_datos:>6,}")
    print(f"  Tiempo total:         {_formatear_tiempo(elapsed)}")
    print(f"  Filas panel:          {len(df_panel):>6,}")
    print(f"  Filas discreto:       {len(df_discreto):>6,}")
    print(f"  Entradas log:         {len(df_log):>6,}")
    print(f"{'=' * 65}")

    return df_panel, df_discreto, df_log


def exportar(
    df_panel: pd.DataFrame,
    df_discreto: pd.DataFrame,
    df_log: pd.DataFrame
):
    """Exporta los tres DataFrames a CSV en config.OUTPUTS_DIR."""

    df_panel.to_csv(config.RUTA_PANEL_HISTORICO,   index=False)
    df_discreto.to_csv(config.RUTA_PANEL_DISCRETO, index=False)
    df_log.to_csv(config.RUTA_LOG_BOOTSTRAPPING,   index=False)

    print(f"\n  Archivos exportados:")
    print(f"  ✓  {config.RUTA_PANEL_HISTORICO}")
    print(f"  ✓  {config.RUTA_PANEL_DISCRETO}")
    print(f"  ✓  {config.RUTA_LOG_BOOTSTRAPPING}")


# =============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# =============================================================================

def main():
    print(f"\n{'=' * 65}")
    print(f"  ORCHESTRATOR — iniciando pipeline")
    print(f"  Modo: {config.MODO.upper()}")
    print(f"{'=' * 65}")

    # ── Carga y preprocesamiento (una sola vez) ──────────────────────────────
    df_op, df_est = cargar_todo()
    df_pre        = preprocesar(df_op, df_est)

    # ── Despachar según modo ─────────────────────────────────────────────────
    if config.MODO == 'prueba':
        df_discreto_prueba, fila_prueba = correr_prueba(df_pre, df_est)

        if df_discreto_prueba is not None:
            # Exportar solo los resultados de la fecha de prueba
            fecha_str = config.FECHA_PRUEBA.replace('-', '')
            ruta_disc = os.path.join(
                config.RUTA_CURVAS_DISCRETAS,
                f'curva_discreta_{fecha_str}.csv'
            )
            df_discreto_prueba.to_csv(ruta_disc, index=False)
            print(f"\n  ✓  Curva discreta exportada: {ruta_disc}")

    elif config.MODO == 'produccion':
        df_panel, df_discreto, df_log = correr_produccion(df_pre)
        exportar(df_panel, df_discreto, df_log)

    else:
        raise ValueError(
            f"[orchestrator] MODO inválido: '{config.MODO}'. "
            f"Usar 'prueba' o 'produccion'."
        )

    print(f"\n  Pipeline completado.\n")


# =============================================================================
# BLOQUE DE PRUEBA DIRECTO
# Ejecutar: python src/orchestrator.py
# =============================================================================
if __name__ == '__main__':
    main()
    