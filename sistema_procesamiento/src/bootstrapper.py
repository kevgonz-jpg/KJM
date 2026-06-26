# =============================================================================
# bootstrapper.py — M4: Motor de Bootstrapping Fama-Bliss
# Proyecto: Curvas de Rendimiento Cero-Cupón TES · BVC 2015-2026
# =============================================================================
# Responsabilidad: dado el DataFrame de bonos activos para UN día específico
# (ordenados por tau ascendente), extraer la curva spot cero-cupón discreta
# mediante el método iterativo de Fama-Bliss.
#
# Convención matemática: composición discreta anual, base 365.
#
# Clasificación estricta de flujos por bono i (núcleo de la corrección):
#   Categoría A — flujos_conocidos:   t <= T_anterior
#                 Se descuentan con el cursor (tasas ya calculadas).
#   Categoría B — flujos_tramo:       T_anterior < t < tau_i
#                 Caen en el tramo nuevo — dependen obligatoriamente de f.
#   Categoría C — flujo_final:        t == tau_i
#                 Pago final — depende de f.
#
# Regla de resolución:
#   Si B está vacío  → despeje algebraico exacto (una sola incógnita simple)
#   Si B no está vacío → solver numérico brentq (polinomio en f)
#
# Fórmula de consolidación spot (equivalencia geométrica exacta):
#   R_spot(Ti) = [(1+R_anterior)^T_anterior * (1+f)^DeltaT]^(1/Ti) - 1
# =============================================================================

import pandas as pd
import numpy as np
from scipy.optimize import brentq
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from cashflow_engine import obtener_flujos_bono


# =============================================================================
# FUNCIONES DE DESCUENTO
# =============================================================================

def factor_descuento_cursor(t: float, cursor: list[tuple[float, float]]) -> float:
    """
    Calcula el factor de descuento acumulado para un tiempo t usando
    EXCLUSIVAMENTE las tasas spot ya conocidas en el cursor.

    Solo debe llamarse con flujos de Categoría A (t <= T_anterior).
    Nunca debe usarse para flujos del tramo nuevo.

    Args:
        t:      tiempo en años — debe ser <= T_anterior (último nodo del cursor)
        cursor: lista de (T_nodo, R_spot) ordenada ascendentemente

    Returns:
        Factor de descuento (1 + R_spot)^t interpolado entre nodos conocidos
    """
    if not cursor:
        raise ValueError("[bootstrapper] Cursor vacío")

    # Caso exacto en un nodo
    for (T_nodo, R_nodo) in cursor:
        if abs(t - T_nodo) < 1e-9:
            return (1 + R_nodo) ** T_nodo

    # t menor que el primer nodo — extrapolación plana hacia atrás
    T0, R0 = cursor[0]
    if t <= T0:
        return (1 + R0) ** t

    # t entre dos nodos conocidos — extraer forward implícito del segmento
    for i in range(len(cursor) - 1):
        T_i,  R_i  = cursor[i]
        T_i1, R_i1 = cursor[i + 1]
        if T_i <= t <= T_i1:
            delta = T_i1 - T_i
            f_seg = ((1 + R_i1) ** T_i1 / (1 + R_i) ** T_i) ** (1 / delta) - 1
            return (1 + R_i) ** T_i * (1 + f_seg) ** (t - T_i)

    # t más allá del último nodo — extrapolación plana
    T_last, R_last = cursor[-1]
    return (1 + R_last) ** T_last * (1 + R_last) ** (t - T_last)


def vp_flujo_con_f(
    t: float,
    cf: float,
    f: float,
    T_anterior: float,
    R_anterior: float
) -> float:
    """
    Calcula el valor presente de un flujo que cae en el tramo activo
    (T_anterior, tau_i] usando el denominador compuesto:

        VP = CF / [(1 + R_anterior)^T_anterior * (1 + f)^(t - T_anterior)]

    Válido para flujos de Categoría B y C.

    Args:
        t:          tiempo del flujo en años (> T_anterior)
        cf:         monto del flujo de caja
        f:          tasa forward del tramo activo (incógnita)
        T_anterior: vencimiento del bono anterior (inicio del tramo)
        R_anterior: tasa spot acumulada hasta T_anterior

    Returns:
        Valor presente del flujo
    """
    delta_t = t - T_anterior
    denominador = (1 + R_anterior) ** T_anterior * (1 + f) ** delta_t
    return cf / denominador


# =============================================================================
# CLASIFICACIÓN DE FLUJOS (núcleo de la corrección)
# =============================================================================

def clasificar_flujos(
    flujos: list[tuple[float, float]],
    T_anterior: float,
    tau_i: float
) -> tuple[list, list, tuple]:
    """
    Clasifica los flujos de caja de un bono en tres categorías estrictas.

    Args:
        flujos:     lista de (t, CF) ordenada ascendentemente
        T_anterior: vencimiento del bono anterior (inicio del tramo nuevo)
        tau_i:      vencimiento del bono actual (fin del tramo nuevo)

    Returns:
        tuple:
            conocidos:  [(t, CF)] con t <= T_anterior  — Categoría A
            tramo:      [(t, CF)] con T_anterior < t < tau_i — Categoría B
            final:      (t, CF) con t == tau_i  — Categoría C
    """
    conocidos = []
    tramo     = []
    final     = None

    for t, cf in flujos:
        if t <= T_anterior + 1e-9:
            # Categoría A: antes o en el último nodo conocido
            conocidos.append((t, cf))
        elif abs(t - tau_i) < 1e-6:
            # Categoría C: flujo final
            final = (t, cf)
        else:
            # Categoría B: en el tramo nuevo pero antes del vencimiento
            tramo.append((t, cf))

    if final is None:
        # Seguridad: si no se detectó flujo final por tolerancia,
        # usar el último flujo de la lista
        final = flujos[-1]
        # Remover de tramo si estaba ahí
        tramo = [(t, cf) for t, cf in tramo
                 if abs(t - final[0]) > 1e-9]

    return conocidos, tramo, final


# =============================================================================
# MOTOR ITERATIVO PRINCIPAL
# =============================================================================

def bootstrap_dia(
    df_dia: pd.DataFrame,
    verbose: bool = False
) -> tuple[list[dict], list[str]]:
    """
    Ejecuta el bootstrapping de Fama-Bliss para un día específico.

    Procesa los bonos en orden ascendente de tau (garantizado por el
    preprocesador). Para cada bono:
      1. Clasifica flujos en A, B, C
      2. Descuenta A con el cursor
      3. Calcula vp_residual_puro = precio_sucio - vp_A
      4. Si B vacío → despeje algebraico de f
         Si B no vacío → solver numérico brentq
      5. Consolida R_spot con equivalencia geométrica exacta
      6. Actualiza el cursor

    Args:
        df_dia:  DataFrame del día ordenado por tau ascendente
        verbose: imprime proceso iterativo detallado

    Returns:
        puntos_spot:  lista de dicts con los puntos de la curva discreta
        advertencias: lista de warnings del proceso
    """
    puntos_spot  = []
    advertencias = []
    cursor       = []  # lista de (T_nodo, R_spot) — crece con cada bono

    fecha = df_dia['fecha_neg'].iloc[0]

    if verbose:
        print(f"\n  Bootstrapping para {fecha.date()} "
              f"— {len(df_dia)} bonos")
        print(f"  {'─' * 70}")

    for _, fila in df_dia.iterrows():
        nemo      = fila['nemotecnico']
        tau_i     = fila['tau']
        tasa_vwap = fila['tasa_vwap'] / 100.0
        coupon    = fila['couponRate'] / 100.0

        # Estado actual del cursor
        T_anterior = cursor[-1][0] if cursor else 0.0
        R_anterior = cursor[-1][1] if cursor else 0.0

        # Obtener flujos de caja
        flujos = obtener_flujos_bono(fila)
        if not flujos:
            msg = f"{nemo}: sin flujos — omitido"
            advertencias.append(msg)
            if verbose:
                print(f"  ⚠  {msg}")
            continue

        # ── PASO 1: Clasificar flujos en A, B, C ────────────────────────────
        conocidos, tramo, final = clasificar_flujos(flujos, T_anterior, tau_i)
        t_final, cf_final = final

        if verbose:
            print(f"\n  {nemo}  tau={tau_i:.4f}  "
                  f"T_ant={T_anterior:.4f}  ΔT={tau_i - T_anterior:.4f}")
            print(f"    Flujos A (conocidos): {len(conocidos)}  "
                  f"B (tramo): {len(tramo)}  C (final): 1")

        # ── PASO 2: Calcular precio sucio implícito desde VWAP ──────────────
        # El dataset tiene tasas VWAP, no precios. Reconstruimos el precio
        # sucio descontando todos los flujos a la tasa VWAP observada.
        precio_sucio = sum(
            cf / (1 + tasa_vwap) ** t
            for t, cf in flujos
        )

        # ── PASO 3: Descontar flujos Categoría A ────────────────────────────
        # Solo los flujos con t <= T_anterior usan el cursor.
        vp_A = 0.0
        error_en_A = False

        for t_k, cf_k in conocidos:
            try:
                fd   = factor_descuento_cursor(t_k, cursor)
                vp_A += cf_k / fd
            except Exception as e:
                msg = f"{nemo}: error descontando flujo A en t={t_k:.4f}: {e}"
                advertencias.append(msg)
                error_en_A = True
                break

        if error_en_A:
            if verbose:
                print(f"  ⚠  {nemo}: error en descuento A — omitido")
            continue

        # ── PASO 4: VP residual puro ─────────────────────────────────────────
        # Solo se restan los flujos correctamente descontados (Categoría A).
        # Los flujos B y C NO se tocan aquí — son la incógnita.
        vp_residual_puro = precio_sucio - vp_A

        if vp_residual_puro <= 0:
            msg = (f"{nemo}: VP residual puro negativo "
                   f"({vp_residual_puro:.6f}) — bono omitido")
            advertencias.append(msg)
            if verbose:
                print(f"  ⚠  {msg}")
            continue

        # ── PASO 5: Resolver f ───────────────────────────────────────────────
        delta_T = t_final - T_anterior

        if delta_T < 1e-9:
            msg = (f"{nemo}: ΔT ≈ 0 ({delta_T:.8f}) — omitido")
            advertencias.append(msg)
            if verbose:
                print(f"  ⚠  {msg}")
            continue

        f = None

        if not tramo:
            # ── Caso simple: B vacío → despeje algebraico exacto ────────────
            # vp_residual_puro = cf_final / [(1+R_ant)^T_ant * (1+f)^delta_T]
            # Despejando:
            # (1+f)^delta_T = cf_final / (vp_residual_puro * (1+R_ant)^T_ant)
            # f = cociente^(1/delta_T) - 1
            try:
                factor_conocido = (1 + R_anterior) ** T_anterior
                cociente = cf_final / (vp_residual_puro * factor_conocido)

                if cociente <= 0:
                    raise ValueError(f"cociente negativo: {cociente:.6f}")

                f = cociente ** (1.0 / delta_T) - 1.0

                if verbose:
                    print(f"    → Despeje algebraico: f={f:.6%}")

            except Exception as e:
                msg = f"{nemo}: despeje algebraico falló: {e} — omitido"
                advertencias.append(msg)
                if verbose:
                    print(f"  ⚠  {msg}")
                continue

        else:
            # ── Caso complejo: B no vacío → solver numérico obligatorio ─────
            # La ecuación es un polinomio en f — no tiene solución algebraica.
            # Igualamos:
            # vp_residual_puro = SUM_B[CF_b / denominador(f, t_b)]
            #                  + CF_final / denominador(f, t_final)
            # donde denominador(f, t) = (1+R_ant)^T_ant * (1+f)^(t - T_ant)

            if verbose:
                print(f"    → Solver numérico (B no vacío, "
                      f"{len(tramo)} flujos intermedios)")

            def ecuacion_f(f_val: float) -> float:
                # VP de flujos Categoría B descontados con f_val
                vp_B = sum(
                    vp_flujo_con_f(t, cf, f_val, T_anterior, R_anterior)
                    for t, cf in tramo
                )
                # VP del flujo final (Categoría C) descontado con f_val
                vp_C = vp_flujo_con_f(
                    t_final, cf_final, f_val, T_anterior, R_anterior
                )
                # La función debe ser cero cuando f_val es la solución
                return vp_B + vp_C - vp_residual_puro

            try:
                f = brentq(
                    ecuacion_f,
                    a=-0.15,
                    b=1.0,
                    xtol=1e-10,
                    rtol=1e-10,
                    maxiter=500
                )
                if verbose:
                    print(f"    → Solver convergió: f={f:.6%}")

            except Exception as e:
                msg = (f"{nemo}: solver numérico no convergió: {e} — omitido")
                advertencias.append(msg)
                if verbose:
                    print(f"  ⚠  {msg}")
                continue

        # ── PASO 6: Validar f ────────────────────────────────────────────────
        if f < config.F_MIN or f > config.F_MAX:
            msg = (f"{nemo}: f fuera de rango ({f:.4%}) — omitido")
            advertencias.append(msg)
            if verbose:
                print(f"  ⚠  {msg}")
            continue

        # ── PASO 7: Consolidar R_spot (equivalencia geométrica exacta) ──────
        # R_spot(Ti) = [(1+R_ant)^T_ant * (1+f)^DeltaT]^(1/Ti) - 1
        producto = (1 + R_anterior) ** T_anterior * (1 + f) ** delta_T
        R_spot_i = producto ** (1.0 / tau_i) - 1.0

        # ── PASO 8: Actualizar cursor y registrar punto ──────────────────────
        cursor.append((tau_i, R_spot_i))

        puntos_spot.append({
            'fecha':       fecha,
            'nemotecnico': nemo,
            'tau':         tau_i,
            'r_spot':      R_spot_i,
            'f_forward':   f,
            'T_anterior':  T_anterior,
            'R_anterior':  R_anterior,
            'delta_T':     delta_T,
            'n_flujos_B':  len(tramo),
        })

        if verbose:
            metodo = "algebraico" if not tramo else "numérico"
            print(f"    ✓  f={f:8.4%}  R_spot={R_spot_i:8.4%}  "
                  f"[{metodo}]")

    if len(puntos_spot) < 2:
        advertencias.append(
            f"Día {fecha.date()}: {len(puntos_spot)} punto(s) — INSUFICIENTE"
        )

    return puntos_spot, advertencias


# =============================================================================
# BLOQUE DE PRUEBA DIRECTO
# Ejecutar: python src/bootstrapper.py
# =============================================================================
if __name__ == '__main__':
    from data_loader import cargar_todo
    from preprocessor import preprocesar, obtener_dia

    df_op, df_est = cargar_todo()
    df_pre = preprocesar(df_op, df_est)
    df_dia = obtener_dia(df_pre, config.FECHA_PRUEBA)

    print(f"\n{'=' * 70}")
    print(f"  BOOTSTRAPPER — {config.FECHA_PRUEBA}")
    print(f"{'=' * 70}")

    puntos, warnings = bootstrap_dia(df_dia, verbose=True)

    if warnings:
        print(f"\n  Advertencias ({len(warnings)}):")
        for w in warnings:
            print(f"    ⚠  {w}")

    print(f"\n{'=' * 70}")
    print(f"  CURVA SPOT DISCRETA — {config.FECHA_PRUEBA}")
    print(f"{'=' * 70}")
    print(f"  {'Bono':<15} {'tau':>7} {'R_spot':>9} "
          f"{'f_forward':>10} {'método':>12} {'flujos_B':>8}")
    print(f"  {'─'*15} {'─'*7} {'─'*9} {'─'*10} {'─'*12} {'─'*8}")

    for p in puntos:
        metodo = "algebraico" if p['n_flujos_B'] == 0 else "numérico"
        print(f"  {p['nemotecnico']:<15} "
              f"{p['tau']:>7.3f} "
              f"{p['r_spot']:>9.4%} "
              f"{p['f_forward']:>10.4%} "
              f"{metodo:>12} "
              f"{p['n_flujos_B']:>8}")

    tasas = [p['r_spot'] for p in puntos]
    print(f"\n  Total puntos spot: {len(puntos)}")
    print(f"  Rango tasas spot:  {min(tasas):.4%} – {max(tasas):.4%}")
    print(f"  ¿Rango plausible TES 2016 (4%-16%)? "
          f"{'✓' if 0.04 <= min(tasas) and max(tasas) <= 0.16 else '✗ REVISAR'}")