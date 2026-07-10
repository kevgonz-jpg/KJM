"""
run_particlefilter.py
======================
Ejecuta el pipeline completo DNS/DNSS vía Particle Filter (τ_t dinámico):
  1. Carga datos TES desde CSV
  2. Warm-start del Kalman Filter (grid search + MLE, con cache en disco)
  3. Calibración AR(1) de la dinámica de log(τ_t)
  4. Particle Filter (Bootstrap SIR) sobre el estado aumentado (β_t, log τ_t)
  5. Diagnósticos (ESS, remuestreo, AIC/BIC)
  6. Guarda resultados en outputs/ y genera reporte PDF

El warm-start del KF es la parte lenta (grid search + MLE multi-start) y
NO depende de los hiperparámetros propios del PF (N_particulas, seed,
ess_threshold, q_ltau_scale, innovation_dist, ...). Por eso se cachea en
disco — re-correr este script cambiando solo argumentos del PF reutiliza
el warm-start ya calculado, en vez de recalcularlo desde cero. Usar
--force_refit_kf si se sabe que algo cambió externamente y se quiere
forzar el recálculo.

Uso
---
    python tes_dns\scripts\run_particlefilter.py --csv TES_colombia\curvas_panel_historico_limpio.csv --model DNS 

    python tes_dns\scripts\run_particlefilter.py \\
        --csv TES_colombia\curvas_panel_historico_limpio.csv \\
        --model DNSS \\
        --n_particles 5000 \\
        --innovation_dist student_t --student_t_df 5 \\
        --q_ltau_scale 2.0

    # Experimentación rápida reutilizando el warm-start ya cacheado:
    python scripts/run_particlefilter.py --csv ... --model DNS --n_particles 8000 --seed 7

Argumentos
----------
  --csv             : ruta al archivo CSV
  --model           : DNS | DNSS (default: DNS)
  --n_particles     : número de partículas del PF (default: 2000)
  --window          : ventana del grid search de warm-start, días (None=expandiente)
  --n_starts_kf     : intentos MLE del warm-start (default: 1)
  --r_mode          : libre | grupos, para el MLE de warm-start (default: libre)
  --sep_min         : separación mínima tau1 > tau2 + sep_min, solo DNSS (default: 0.30)
  --ess_threshold   : fracción de N bajo la cual se remuestrea (default: 0.5)
  --innovation_dist : gaussian | student_t (default: gaussian)
  --student_t_df    : grados de libertad si innovation_dist=student_t (default: 5.0)
  --q_ltau_scale    : escala sobre la varianza de innovación de log-tau calibrada (default: 1.0)
  --seed            : semilla del PF (default: 42)
  --force_refit_kf  : ignora el cache del warm-start y reestima desde cero
  --cache_dir       : carpeta del cache de warm-start (default: outputs/cache_kf_warmstart)
  --out_dir         : carpeta de salida (default: outputs/)
  --start           : fecha inicio YYYY-MM-DD (opcional)
  --end             : fecha fin   YYYY-MM-DD (opcional)
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import (
    load_tes_data, describe_data,
    fit_pf_model, residual_stats,
    generate_full_report,
)


def parse_args():
    p = argparse.ArgumentParser(description='DNS/DNSS Particle Filter (τ dinámico) — TES Colombia')
    p.add_argument('--csv',             required=True,             help='Ruta al CSV de rendimientos')
    p.add_argument('--model',           default='DNS',             choices=['DNS', 'DNSS'])
    p.add_argument('--n_particles',     default=20000, type=int)
    p.add_argument('--window',          default=None, type=int,    help='Ventana grid warm-start (None=expandiente)')
    p.add_argument('--n_starts_kf',     default=1,    type=int)
    p.add_argument('--r_mode',          default='libre',           choices=['libre', 'grupos'])
    p.add_argument('--sep_min',         default=0.30, type=float)
    p.add_argument('--ess_threshold',   default=0.5,  type=float)
    p.add_argument('--innovation_dist', default='gaussian',        choices=['gaussian', 'student_t'])
    p.add_argument('--student_t_df',    default=5.0,  type=float)
    p.add_argument('--q_ltau_scale',    default=1.0,  type=float)
    p.add_argument('--seed',            default=42,   type=int)
    p.add_argument('--force_refit_kf',  action='store_true')
    p.add_argument('--cache_dir',       default=None,              help='default: <out_dir>/cache_kf_warmstart')
    p.add_argument('--out_dir',         default='outputs')
    p.add_argument('--start',           default=None)
    p.add_argument('--end',             default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    prefix = os.path.join(args.out_dir, f'{args.model}_PF')
    cache_dir = args.cache_dir or os.path.join(args.out_dir, 'cache_kf_warmstart')

    # ── 1. Datos ─────────────────────────────────────────────────────────────
    print("\n[1/3] Cargando datos TES…")
    yields, dates, maturities = load_tes_data(
        args.csv,
        start_date=args.start,
        end_date=args.end)
    describe_data(yields, dates, maturities)

    # ── 2. Warm-start KF + Particle Filter ──────────────────────────────────
    print(f"\n[2/3] Particle Filter ({args.model}, N={args.n_particles} partículas, "
          f"innovaciones={args.innovation_dist}) …")
    out = fit_pf_model(
        yields, maturities, dates,
        model            = args.model,
        n_particles      = args.n_particles,
        window_size      = args.window,
        n_starts_kf      = args.n_starts_kf,
        r_mode           = args.r_mode,
        sep_min          = args.sep_min,
        ess_threshold    = args.ess_threshold,
        innovation_dist  = args.innovation_dist,
        student_t_df     = args.student_t_df,
        q_ltau_scale     = args.q_ltau_scale,
        seed             = args.seed,
        force_refit_kf   = args.force_refit_kf,
        cache_dir        = cache_dir,
        verbose          = True,
    )
    res_kf, res_pf = out['kf'], out['pf']

    # ── 3. Diagnósticos y guardado ───────────────────────────────────────────
    print("\n[3/3] Diagnósticos y guardado…")
    print(f"\n  Estadísticos de residuos (PF, pb):")
    df_res_pf = residual_stats(res_pf['residuals'], maturities, pb_scale=True)
    print(df_res_pf.to_string(index=False))
    df_res_pf.to_csv(f'{prefix}_residuals_stats.csv', index=False)

    print(f"\n  {'Métrica':<28} {'KF (warm-start)':>16} {'PF (τ dinámico)':>16}")
    print(f"  {'-'*60}")
    print(f"  {'RMSE total (pb)':<28} {res_kf['rmse_total']*100:>16.4f} {res_pf['rmse_total']*100:>16.4f}")
    print(f"  {'Log-lik':<28} {res_kf['loglik']:>16,.1f} {res_pf['loglik']:>16,.1f}")
    print(f"  {'AIC':<28} {res_kf['aic']:>16,.1f} {res_pf['aic']:>16,.1f}")
    print(f"  {'BIC':<28} {res_kf['bic']:>16,.1f} {res_pf['bic']:>16,.1f}")
    print(f"\n  Nota: el log-lik/AIC/BIC del PF es una estimación Monte Carlo (varianza")
    print(f"  finita en N); el del PF cuenta solo k={res_pf['k_nuevo']} parámetros NUEVOS")
    print(f"  de la dinámica de tau, no el sistema completo heredado del warm-start KF.")
    print(f"  ESS/N promedio: {res_pf['ess_t'].mean()/args.n_particles:.3f}"
          f"   Remuestreos: {res_pf['resampled_t'].mean()*100:.1f}% de los periodos")

    # Guardar arrays (mismo patrón que run_kalman.py)
    n_beta = 3 if args.model == 'DNS' else 4
    np.save(f'{prefix}_beta_filt.npy',  res_pf['x_filt_mean'][:, :n_beta])
    np.save(f'{prefix}_beta_std.npy',   res_pf['x_filt_std'][:, :n_beta])
    np.save(f'{prefix}_residuals.npy',  res_pf['residuals'])
    np.save(f'{prefix}_tau1_t.npy',     res_pf['tau1_t'])
    np.save(f'{prefix}_ess_t.npy',      res_pf['ess_t'])
    np.save(f'{prefix}_resampled_t.npy', res_pf['resampled_t'])
    if args.model == 'DNSS':
        np.save(f'{prefix}_tau2_t.npy', res_pf['tau2_t'])

    np.savez(f'{prefix}_params.npz',
             mu_beta=res_kf['mu'], F_beta_diag=np.diag(res_kf['F']),
             Q_beta_diag=np.diag(res_kf['Q']), R_diag=res_kf['R_diag'],
             loglik=np.array([res_pf['loglik']]),
             aic=np.array([res_pf['aic']]), bic=np.array([res_pf['bic']]))

    pd.DataFrame({'date': dates}).to_csv(f'{prefix}_dates.csv', index=False)
    pd.DataFrame({'maturity': maturities}).to_csv(f'{prefix}_maturities.csv', index=False)

    print(f"\nResultados guardados en: {args.out_dir}/")

    # ── Reporte PDF ───────────────────────────────────────────────────────────
    print("\nGenerando reporte PDF…")
    generate_full_report(
        yields       = yields,
        dates        = dates,
        maturities   = maturities,
        model        = args.model,
        tau_mode     = f'particle_filter ({args.innovation_dist})',
        out_dir      = args.out_dir,
        # KF (warm-start, para comparación y tablas)
        beta_smooth  = res_kf['beta_smoothed'],
        tau1_t       = res_kf['tau1_t'],
        tau2_t       = res_kf['tau2_t'],
        mle_params   = {
            'mu'     : res_kf['mu'],
            'F_diag' : np.diag(res_kf['F']),
            'Q_diag' : np.diag(res_kf['Q']),
            'R_diag' : res_kf['R_diag'],
            'loglik' : res_kf['loglik'],
            'k'      : res_kf['k'],
            'aic'    : res_kf['aic'],
            'bic'    : res_kf['bic'],
        },
        residuals_kf = res_kf['residuals'],
        # Particle Filter
        res_pf       = res_pf,
    )


if __name__ == '__main__':
    main()
