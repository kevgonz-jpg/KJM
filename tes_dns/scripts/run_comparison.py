"""
run_comparison.py
=================
Fase 4: Comparación out-of-sample KF vs Bayesiano.

Carga los resultados previos de run_kalman.py y run_bayesian.py
y evalúa el pronóstico a h = 1, 3, 6, 12 meses con ventana
expandiente. Genera tabla comparativa RMSE/MAE.

Uso
---
    python scripts/run_comparison.py \\
        --csv datos/curvas_panel_historico_limpio.csv \\
        --model DNS \\
        --out_dir outputs/

Requisitos previos
------------------
Haber corrido:
    python scripts/run_kalman.py   --csv ... --model DNS
    python scripts/run_bayesian.py --csv ... --model DNS
para que existan los archivos de parámetros en outputs/.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import (
    load_tes_data, make_priors,
    expanding_window_kf, expanding_window_bayesian,
    compute_metrics, summary_metrics, compare_models,
    generate_full_report,
)


def parse_args():
    p = argparse.ArgumentParser(description='Comparación OOS KF vs Bayesiano')
    p.add_argument('--csv',        required=True,           help='Ruta al CSV de rendimientos')
    p.add_argument('--model',      default='DNS',           choices=['DNS', 'DNSS'])
    p.add_argument('--tau_mode',   default='estatico',      choices=['estatico', 'ar1'])
    p.add_argument('--horizons',   default='1,3,6,12')
    p.add_argument('--min_train',  default=120,  type=int)
    p.add_argument('--refit_every',default=21,   type=int,  help='Re-estimar Bayes cada N periodos')
    p.add_argument('--n_iter',     default=800,  type=int)
    p.add_argument('--n_burnin',   default=400,  type=int)
    p.add_argument('--n_chains',   default=2,    type=int)
    p.add_argument('--out_dir',    default='outputs')
    p.add_argument('--start',      default=None)
    p.add_argument('--end',        default=None)
    return p.parse_args()


def load_kf_params(out_dir, model):
    """Carga parámetros MLE guardados por run_kalman.py."""
    prefix = os.path.join(out_dir, f'{model}_KF')
    try:
        npz    = np.load(f'{prefix}_params.npz')
        mu     = npz['mu']
        F      = npz['F']
        Q      = npz['Q']
        R_diag = npz['R_diag']
        tau1_t = np.load(f'{prefix}_tau1_t.npy')
        tau2_t = (np.load(f'{prefix}_tau2_t.npy')
                  if os.path.exists(f'{prefix}_tau2_t.npy') else None)
        return mu, F, Q, R_diag, tau1_t, tau2_t
    except FileNotFoundError as e:
        print(f"\n⚠  No se encontraron parámetros KF: {e}")
        print(f"   Corre primero: python scripts/run_kalman.py --csv <archivo.csv> --model {model}")
        return None


def main():
    args     = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    horizons = [int(h) for h in args.horizons.split(',')]
    prefix   = os.path.join(args.out_dir, f'{args.model}_OOS')

    # ── 1. Datos ─────────────────────────────────────────────────────────────
    print("\n[1/3] Cargando datos TES…")
    yields, dates, maturities = load_tes_data(
        args.csv,
        start_date=args.start,
        end_date=args.end)
    print(f"  T={len(yields)} | N={len(maturities)} | "
          f"horizons={horizons} | min_train={args.min_train}")

    # ── 2. KF OOS ────────────────────────────────────────────────────────────
    print("\n[2/3] OOS — Kalman Filter…")
    kf_result = load_kf_params(args.out_dir, args.model)
    if kf_result is None:
        sys.exit(1)
    mu, F, Q, R_diag, tau1_t, tau2_t = kf_result

    df_kf = expanding_window_kf(
        yields, maturities, dates,
        tau1_t      = tau1_t,
        mu_final    = mu,
        F_final     = F,
        Q_final     = Q,
        R_diag_final= R_diag,
        horizons    = horizons,
        min_train   = args.min_train,
        model       = args.model,
        tau2_t      = tau2_t,
        verbose     = True)

    df_kf.to_csv(f'{prefix}_KF_errors.csv', index=False)

    # ── 3. Bayesiano OOS ─────────────────────────────────────────────────────
    print("\n[3/3] OOS — Bayesiano (Gibbs) …")
    priors = make_priors(model=args.model, prior_tau_type='uniforme')

    df_bayes = expanding_window_bayesian(
        yields, maturities, dates, priors,
        horizons    = horizons,
        min_train   = args.min_train,
        refit_every = args.refit_every,
        n_iter      = args.n_iter,
        n_burnin    = args.n_burnin,
        n_chains    = args.n_chains,
        tau_mode    = args.tau_mode,
        verbose     = True)

    df_bayes.to_csv(f'{prefix}_Bayes_errors.csv', index=False)

    # ── Tabla comparativa ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  COMPARACIÓN OOS — {args.model}")
    print("=" * 60)

    df_cmp = compare_models(df_kf, df_bayes, pb_scale=True)
    print(df_cmp.to_string(index=False))
    df_cmp.to_csv(f'{prefix}_comparison.csv', index=False)

    # Detalle por plazo y horizonte
    met_kf    = compute_metrics(df_kf,    pb_scale=True)
    met_bayes = compute_metrics(df_bayes, pb_scale=True)

    print("\n  RMSE por horizonte y plazo (KF vs Bayesiano, pb):")
    for h in horizons:
        kf_h  = met_kf[met_kf['horizon'] == h][['maturity', 'RMSE']].rename(
            columns={'RMSE': f'RMSE_KF'})
        bay_h = met_bayes[met_bayes['horizon'] == h][['maturity', 'RMSE']].rename(
            columns={'RMSE': f'RMSE_Bayes'})
        tbl   = kf_h.merge(bay_h, on='maturity')
        tbl['ΔRMSE'] = tbl['RMSE_Bayes'] - tbl['RMSE_KF']
        tbl['Mejor'] = tbl['ΔRMSE'].apply(lambda x: 'Bayes' if x < 0 else 'KF')
        print(f"\n  h = {h} mes(es):")
        print(tbl.to_string(index=False))

    print(f"\n  Archivos guardados en: {args.out_dir}/")
    print(f"    {args.model}_OOS_KF_errors.csv")
    print(f"    {args.model}_OOS_Bayes_errors.csv")
    print(f"    {args.model}_OOS_comparison.csv")

    # ── Reporte PDF ───────────────────────────────────────────────────────────
    print("\nGenerando reporte PDF (comparación OOS)…")
    generate_full_report(
        yields       = yields,
        dates        = dates,
        maturities   = maturities,
        model        = args.model,
        tau_mode     = args.tau_mode,
        out_dir      = args.out_dir,
        df_kf        = df_kf,
        df_bayes     = df_bayes,
        horizons     = horizons,
    )


if __name__ == '__main__':
    main()
