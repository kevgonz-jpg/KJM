"""
run_kalman.py
=============
Ejecuta el pipeline completo DNS/DNSS frecuentista:
  1. Carga datos TES desde CSV
  2. Grid search adaptativo de τ_t (ventana rodante o expandiente)
  3. Construye H_arr(t) con los τ_t óptimos
  4. MLE multi-start (L-BFGS-B)
  5. KF + RTS Smoother
  6. Guarda resultados en outputs/

Uso
---
    python scripts/run_kalman.py --csv datos/curvas_panel_historico_limpio.csv --model DNS
    python scripts/run_kalman.py --csv datos/curvas_panel_historico_limpio.csv --model DNSS --r_mode grupos

Argumentos opcionales
---------------------
  --csv       : ruta al archivo CSV
  --model     : DNS | DNSS (default: DNS)
  --window    : tamaño ventana grid search en días (None = expandiente)
  --r_mode    : libre | grupos (default: libre)
  --n_starts  : intentos MLE (default: 6)
  --out_dir   : carpeta de salida (default: outputs/)
  --start     : fecha inicio YYYY-MM-DD (opcional)
  --end       : fecha fin   YYYY-MM-DD (opcional)
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import (
    load_tes_data, describe_data,
    rolling_tau_dns, rolling_tau_dnss,
    build_H_array, mle_estimate,
    kalman_filter, rts_smoother,
    compute_P0_stationary, residual_stats,
    generate_full_report,
)


def parse_args():
    p = argparse.ArgumentParser(description='DNS/DNSS Kalman Filter — TES Colombia')
    p.add_argument('--csv',      required=True,           help='Ruta al CSV de rendimientos')
    p.add_argument('--model',    default='DNS',           choices=['DNS', 'DNSS'])
    p.add_argument('--window',   default=None, type=int,  help='Ventana grid (None=expandiente)')
    p.add_argument('--r_mode',   default='libre',         choices=['libre', 'grupos'])
    p.add_argument('--n_starts', default=2,   type=int)
    p.add_argument('--out_dir',  default='outputs')
    p.add_argument('--start',    default=None,            help='Fecha inicio YYYY-MM-DD')
    p.add_argument('--end',      default=None,            help='Fecha fin   YYYY-MM-DD')
    return p.parse_args()


def main():
    args   = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    prefix = os.path.join(args.out_dir, f'{args.model}_KF')

    # ── 1. Datos ─────────────────────────────────────────────────────────────
    print("\n[1/5] Cargando datos TES…")
    yields, dates, maturities = load_tes_data(
        args.csv,
        start_date=args.start,
        end_date=args.end)
    describe_data(yields, dates, maturities)
    T, N = yields.shape
    k    = 3 if args.model == 'DNS' else 4

    # ── 2. Grid search τ_t ───────────────────────────────────────────────────
    print(f"\n[2/5] Grid search τ_t ({args.model}) …")
    tau1_grid = np.linspace(0.3, 5.0, 20)

    if args.model == 'DNS':
        tau1_t, sse_t, _ = rolling_tau_dns(
            yields, maturities, tau1_grid, args.window)
        tau2_t = None
    else:
        tau2_grid = np.linspace(0.3, 2.5, 10)
        tau1_t, tau2_t, sse_t = rolling_tau_dnss(
            yields, maturities, tau1_grid, tau2_grid, args.window, sep_min=0.30)

    print(f"  τ₁: media={tau1_t.mean():.3f}  std={tau1_t.std():.3f}"
          f"  rango=[{tau1_t.min():.3f},{tau1_t.max():.3f}]")
    if tau2_t is not None:
        print(f"  τ₂: media={tau2_t.mean():.3f}  std={tau2_t.std():.3f}")

    # ── 3. Matriz H_arr(t) ───────────────────────────────────────────────────
    print("\n[3/5] Construyendo H_arr(t) …")
    H_arr = build_H_array(maturities, tau1_t, tau2_t, args.model)

    # ── 4. MLE ───────────────────────────────────────────────────────────────
    print(f"\n[4/5] MLE multi-start (F diagonal, R {args.r_mode}) …")
    mu, F, Q, R_diag, best_ll, best_params, r_groups_log = mle_estimate(
        yields, H_arr, k, N,
        F_type     = 'diagonal',
        n_starts   = args.n_starts,
        verbose    = True,
        r_mode     = args.r_mode,
        maturities = maturities,
    )

    # ── 5. KF + RTS Smoother ─────────────────────────────────────────────────
    print("\n[5/5] Kalman Filter + RTS Smoother …")
    P0    = compute_P0_stationary(F, Q, 'diagonal')
    beta0 = mu.copy()

    beta_pred, beta_filt, P_pred, P_filt, innov, loglik = kalman_filter(
        yields, H_arr, F, Q, R_diag, mu, beta0, P0)

    beta_smooth, P_smooth, G_all = rts_smoother(
        beta_filt, P_filt, beta_pred, P_pred, F)

    fitted_smooth = np.array([(H_arr[t] @ beta_smooth[t]) for t in range(T)])
    residuals     = yields - fitted_smooth

    print("\n  Estadísticos de residuos (pb):")
    df_res = residual_stats(residuals, maturities, pb_scale=True)
    print(df_res.to_string(index=False))

    # ── Guardar ──────────────────────────────────────────────────────────────
    np.save(f'{prefix}_beta_filt.npy',   beta_filt)
    np.save(f'{prefix}_beta_smooth.npy', beta_smooth)
    np.save(f'{prefix}_beta_pred.npy',   beta_pred)
    np.save(f'{prefix}_residuals.npy',   residuals)
    np.save(f'{prefix}_tau1_t.npy',      tau1_t)
    if tau2_t is not None:
        np.save(f'{prefix}_tau2_t.npy', tau2_t)

    np.savez(f'{prefix}_params.npz',
             mu=mu, F=F, Q=Q, R_diag=R_diag,
             loglik=np.array([best_ll]))

    pd.DataFrame({'date': dates}).to_csv(f'{prefix}_dates.csv',      index=False)
    pd.DataFrame({'maturity': maturities}).to_csv(f'{prefix}_maturities.csv', index=False)

    print(f"\nResultados guardados en: {args.out_dir}/")
    print(f"  Log-lik final       : {loglik:.2f}")
    print(f"  Residuo RMSE prom.  : {df_res['RMSE'].mean():.3f} pb")

    # ── Reporte PDF ───────────────────────────────────────────────────────────
    print("\nGenerando reporte PDF…")
    generate_full_report(
        yields       = yields,
        dates        = dates,
        maturities   = maturities,
        model        = args.model,
        tau_mode     = 'grid_search',
        out_dir      = args.out_dir,
        beta_smooth  = beta_smooth,
        tau1_t       = tau1_t,
        tau2_t       = tau2_t,
        mle_params   = {
            'mu'     : mu,
            'F_diag' : np.diag(F),
            'Q_diag' : np.diag(Q),
            'R_diag' : R_diag,
            'loglik' : best_ll,
        },
        residuals_kf = residuals,
    )


if __name__ == '__main__':
    main()
