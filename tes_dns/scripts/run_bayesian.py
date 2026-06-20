"""
run_bayesian.py
===============
Ejecuta el pipeline completo DNS/DNSS Bayesiano (Gibbs sampler):
  1. Carga datos TES desde CSV
  2. Define priors
  3. Lanza cadenas en paralelo (multi-start)
  4. Diagnósticos MCMC (R-hat, ESS)
  5. Resumen de la posterior
  6. Guarda resultados en outputs/

Uso
---
    python scripts/run_bayesian.py --csv datos/curvas_panel_historico_limpio.csv --model DNS
    python scripts/run_bayesian.py --csv datos/curvas_panel_historico_limpio.csv --model DNSS \\
        --tau_mode ar1 --n_iter 2000 --n_burnin 1000

Argumentos
----------
  --csv       : ruta al archivo CSV
  --model     : DNS | DNSS (default: DNS)
  --tau_mode  : estatico | ar1 | rolling (default: estatico)
  --n_iter    : iteraciones totales por cadena (default: 1500)
  --n_burnin  : burn-in a descartar (default: 750)
  --n_chains  : cadenas paralelas (default: 4)
  --window    : tamaño ventana en días (solo tau_mode=rolling, default: 120)
  --step      : desplazamiento entre ventanas (default: 21)
  --r_mode    : libre | grupos (default: libre)
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
    make_priors, make_priors_ar1,
    run_chains_parallel, run_rolling_bayes,
    compute_diagnostics, print_diagnostics,
    residual_stats, get_H, effective_sample_size,
    generate_full_report,
)


def parse_args():
    p = argparse.ArgumentParser(description='DNS/DNSS Gibbs Sampler — TES Colombia')
    p.add_argument('--csv',      required=True,           help='Ruta al CSV de rendimientos')
    p.add_argument('--model',    default='DNS',           choices=['DNS', 'DNSS'])
    p.add_argument('--tau_mode', default='estatico',      choices=['estatico', 'ar1', 'rolling'])
    p.add_argument('--n_iter',   default=1500, type=int)
    p.add_argument('--n_burnin', default=750,  type=int)
    p.add_argument('--n_chains', default=4,    type=int)
    p.add_argument('--window',   default=120,  type=int,  help='Ventana días (rolling)')
    p.add_argument('--step',     default=21,   type=int,  help='Step entre ventanas (rolling)')
    p.add_argument('--r_mode',   default='libre',         choices=['libre', 'grupos'])
    p.add_argument('--out_dir',  default='outputs')
    p.add_argument('--start',    default=None)
    p.add_argument('--end',      default=None)
    return p.parse_args()


def posterior_summary(chains, model):
    k       = chains[0]['mu'].shape[1]
    names_b = ['beta0(nivel)', 'beta1(pend.)', 'beta2(curv.)', 'beta3(curv2)'][:k]

    all_tau1 = np.concatenate([ch['tau1'] for ch in chains])
    params   = {'tau1': all_tau1}
    if model == 'DNSS':
        params['tau2'] = np.concatenate([ch['tau2'] for ch in chains])
    for i, nb in enumerate(names_b):
        params[f'f({nb})']  = np.concatenate([ch['f_vec'][:, i] for ch in chains])
        params[f'mu({nb})'] = np.concatenate([ch['mu'][:, i]    for ch in chains])
        params[f'q({nb})']  = np.concatenate([ch['q_vec'][:, i] for ch in chains])

    rows = []
    for name, samp in params.items():
        rows.append({
            'Parámetro': name,
            'Media'    : round(float(samp.mean()), 5),
            'Std'      : round(float(samp.std()),  5),
            'IC2.5%'   : round(float(np.percentile(samp, 2.5)),  5),
            'Mediana'  : round(float(np.median(samp)),            5),
            'IC97.5%'  : round(float(np.percentile(samp, 97.5)), 5),
            'ESS'      : round(effective_sample_size(samp), 0),
        })
    return pd.DataFrame(rows)


def main():
    args   = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    prefix = os.path.join(args.out_dir, f'{args.model}_Bayes_{args.tau_mode}')

    # ── 1. Datos ─────────────────────────────────────────────────────────────
    print("\n[1/4] Cargando datos TES…")
    yields, dates, maturities = load_tes_data(
        args.csv,
        start_date=args.start,
        end_date=args.end)
    describe_data(yields, dates, maturities)

    # ── 2. Priors ─────────────────────────────────────────────────────────────
    print(f"\n[2/4] Definiendo priors ({args.model} | τ-modo: {args.tau_mode}) …")
    if args.tau_mode == 'ar1':
        priors = make_priors_ar1(
            model          = args.model,
            tau1_init      = 1.78,
            tau2_init      = 0.60 if args.model == 'DNSS' else None,
            prior_tau_type = 'uniforme',
        )
    else:
        priors = make_priors(
            model          = args.model,
            prior_tau_type = 'uniforme',
        )

    # ── 3. Estimación ─────────────────────────────────────────────────────────
    gibbs_kwargs = dict(r_mode=args.r_mode, tau_mode=args.tau_mode)

    if args.tau_mode == 'rolling':
        print(f"\n[3/4] Gibbs rolling (W={args.window}, step={args.step}) …")
        resultados = run_rolling_bayes(
            yields, maturities, dates, priors,
            n_iter      = args.n_iter,
            n_burnin    = args.n_burnin,
            window_size = args.window,
            step        = args.step,
            n_chains    = args.n_chains,
            verbose     = True,
            r_mode      = args.r_mode,
        )
        tau1_ventanas = np.array([r['tau1_pm'] for r in resultados])
        print(f"\n  τ₁ por ventana: media={tau1_ventanas.mean():.3f} "
              f"std={tau1_ventanas.std():.3f}")

        df_roll = pd.DataFrame([{
            't_start'   : r['t_start'],
            't_end'     : r['t_end'],
            'date_mid'  : str(r['dates_window'][len(r['dates_window']) // 2])[:10],
            'tau1_pm'   : r['tau1_pm'],
            'tau1_q025' : r['tau1_q025'],
            'tau1_q975' : r['tau1_q975'],
        } | ({'tau2_pm': r['tau2_pm'], 'tau2_q025': r['tau2_q025'],
              'tau2_q975': r['tau2_q975']} if args.model == 'DNSS' else {})
            for r in resultados
        ])
        df_roll.to_csv(f'{prefix}_rolling_tau.csv', index=False)
        print(f"  Guardado: {prefix}_rolling_tau.csv")
        return

    else:
        print(f"\n[3/4] Gibbs sampler ({args.n_chains} cadenas × "
              f"{args.n_iter} iters, burnin={args.n_burnin}) …")
        chains = run_chains_parallel(
            yields, maturities, priors,
            n_iter   = args.n_iter,
            n_burnin = args.n_burnin,
            n_chains = args.n_chains,
            verbose  = True,
            **gibbs_kwargs,
        )

    # ── 4. Diagnósticos ───────────────────────────────────────────────────────
    print("\n[4/4] Diagnósticos MCMC …")
    print_diagnostics(chains)

    df_diag = compute_diagnostics(chains)
    df_diag.to_csv(f'{prefix}_diagnostics.csv', index=False)

    df_post = posterior_summary(chains, args.model)
    print(f"\n{'Posterior':─<55}")
    print(df_post.to_string(index=False))
    df_post.to_csv(f'{prefix}_posterior.csv', index=False)

    # Residuos con media posterior
    tau1_pm = float(np.concatenate([ch['tau1'] for ch in chains]).mean())
    tau2_pm = (float(np.concatenate([ch['tau2'] for ch in chains]).mean())
               if args.model == 'DNSS' else None)
    H_pm    = get_H(maturities, tau1_pm, tau2_pm, args.model)
    beta_pm = np.mean([ch['betas'].mean(axis=0) for ch in chains], axis=0)
    resid   = yields - (H_pm @ beta_pm.T).T

    print("\n  Estadísticos de residuos (pb, con E[β|Y]):")
    df_res = residual_stats(resid, maturities, pb_scale=True)
    print(df_res.to_string(index=False))
    df_res.to_csv(f'{prefix}_residuals_stats.csv', index=False)

    for j, ch in enumerate(chains):
        np.save(f'{prefix}_chain{j+1}_tau1.npy',  ch['tau1'])
        np.save(f'{prefix}_chain{j+1}_betas.npy', ch['betas'])
        if args.model == 'DNSS':
            np.save(f'{prefix}_chain{j+1}_tau2.npy', ch['tau2'])

    print(f"\nResultados guardados en: {args.out_dir}/")

    # ── Reporte PDF ───────────────────────────────────────────────────────────
    print("\nGenerando reporte PDF…")
    generate_full_report(
        yields         = yields,
        dates          = dates,
        maturities     = maturities,
        model          = args.model,
        tau_mode       = args.tau_mode,
        out_dir        = args.out_dir,
        chains         = chains,
        df_diagnostics = df_diag,
        df_posterior   = df_post,
    )


if __name__ == '__main__':
    main()
