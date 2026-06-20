"""
forecasting.py
==============
Pronóstico out-of-sample DNS/DNSS con ventana expandiente.
Evalúa horizontes h = 1, 3, 6, 12 meses con RMSE y MAE.

IMPORTANTE
----------
Para pronóstico válido se usan estados PREDICHOS (β̂_{t|t-1}),
NO estados suavizados (β̂_{t|T}). Los suavizados incorporan
información futura → sobreestiman la calidad del pronóstico.

Funciones exportadas
--------------------
forecast_kf(...)        — pronóstico desde KF frecuentista
forecast_bayesian(...)  — pronóstico desde Gibbs posterior
expanding_window_kf(...)
expanding_window_bayesian(...)
compute_metrics(...)    — RMSE y MAE por horizonte y plazo
summary_metrics(...)    — tabla resumen compacta

Referencias
-----------
Diebold & Li (2006) — Sec. 5
Caldeira, Laurini & Portugal (2010)
"""

import numpy as np
import pandas as pd

from .nelson_siegel import get_H
from .kalman        import kalman_filter, rts_smoother, compute_P0_stationary


# ═══════════════════════════════════════════════════════════════════════════════
# PRONÓSTICO KF — h pasos adelante desde β_{T|T}
# ═══════════════════════════════════════════════════════════════════════════════

def forecast_kf(beta_filt_T: np.ndarray,
                mu: np.ndarray,
                F: np.ndarray,
                H_forecast: np.ndarray,
                horizons: list[int]) -> dict[int, np.ndarray]:
    """
    Propaga β_{T|T} h pasos adelante bajo el modelo de transición.

    Parámetros
    ----------
    beta_filt_T  : (k,)    estado filtrado al último periodo T
    mu           : (k,)    media incondicional
    F            : (k, k)  matriz de transición
    H_forecast   : (N, k)  cargas para el horizonte de pronóstico
                           (puede ser H fija si τ es estático)
    horizons     : lista de horizontes (en periodos)

    Retorna
    -------
    dict {h: y_hat(N,)}
    """
    y_hats = {}
    beta   = beta_filt_T.copy()
    for h in sorted(horizons):
        for _ in range(h):
            beta = mu + F @ (beta - mu)
        y_hats[h] = H_forecast @ beta
        beta = beta_filt_T.copy()          # reiniciar para siguiente horizonte
    return y_hats


def _propagate_beta(beta_0, mu, F, h):
    """Propaga β_0 exactamente h pasos."""
    b = beta_0.copy()
    for _ in range(h):
        b = mu + F @ (b - mu)
    return b


# ═══════════════════════════════════════════════════════════════════════════════
# PRONÓSTICO BAYESIANO — media posterior predictiva
# ═══════════════════════════════════════════════════════════════════════════════

def forecast_bayesian(chains: list,
                      maturities: np.ndarray,
                      horizons: list[int],
                      tau_mode: str = 'estatico') -> dict[int, np.ndarray]:
    """
    Pronóstico h-pasos adelante integrando sobre la posterior.

    Promedia E[y_{T+h}|Y_T] = H·E[β_{T+h}|Y_T] sobre todas
    las muestras MCMC de las cadenas.

    Parámetros
    ----------
    chains     : lista de cadenas (run_chain / run_chains_parallel)
    maturities : (N,)
    horizons   : list[int]
    tau_mode   : 'estatico' | 'ar1' | 'rolling'

    Retorna
    -------
    dict {h: y_hat(N,)}
    """
    model = chains[0]['model']
    all_yhats = {h: [] for h in horizons}

    for ch in chains:
        n_draw = ch['tau1'].shape[0]
        T_obs  = ch['betas'].shape[1]

        for s in range(n_draw):
            mu_s    = ch['mu'][s]
            f_s     = ch['f_vec'][s]
            F_s     = np.diag(f_s)
            tau1_s  = ch['tau1'][s]
            tau2_s  = ch['tau2'][s] if model == 'DNSS' else None

            # Estado filtrado al último periodo (última columna del FFBS)
            beta_T = ch['betas'][s, T_obs - 1, :]

            H_s = get_H(maturities, tau1_s, tau2_s, model)

            for h in horizons:
                beta_h = _propagate_beta(beta_T, mu_s, F_s, h)
                all_yhats[h].append(H_s @ beta_h)

    return {h: np.mean(all_yhats[h], axis=0) for h in horizons}


# ═══════════════════════════════════════════════════════════════════════════════
# VENTANA EXPANDIENTE — KF
# ═══════════════════════════════════════════════════════════════════════════════

def expanding_window_kf(yields: np.ndarray,
                        maturities: np.ndarray,
                        dates: np.ndarray,
                        tau1_t: np.ndarray,
                        mu_final: np.ndarray,
                        F_final: np.ndarray,
                        Q_final: np.ndarray,
                        R_diag_final: np.ndarray,
                        horizons: list[int] = [1, 3, 6, 12],
                        min_train: int = 120,
                        model: str = 'DNS',
                        tau2_t: np.ndarray | None = None,
                        verbose: bool = True) -> pd.DataFrame:
    """
    Evaluación out-of-sample con ventana expandiente para el KF.

    En cada periodo t se usa solo y_1:t para estimar β_{t|t},
    luego se propaga h pasos con los parámetros (μ, F) ya estimados
    sobre la muestra completa (parámetros fijos, estados actualizados).

    Retorna
    -------
    pd.DataFrame con columnas: date, maturity, horizon, y_real, y_hat, error
    """
    T, N      = yields.shape
    H_arr     = np.zeros((T, N, 3 if model == 'DNS' else 4))
    for t in range(T):
        t2 = tau2_t[t] if tau2_t is not None else None
        H_arr[t] = get_H(maturities, tau1_t[t], t2, model)

    P0    = compute_P0_stationary(F_final, Q_final, 'diagonal')
    beta0 = mu_final.copy()

    if verbose:
        print(f"OOS-KF: {T - min_train} pronósticos × {len(horizons)} horizontes")

    records = []
    for t in range(min_train, T - max(horizons)):
        # Re-filtrar hasta t (ventana expandiente)
        *_, beta_filt, _, _, _ = kalman_filter(
            yields[:t], H_arr[:t], F_final, Q_final,
            R_diag_final, mu_final, beta0, P0)

        beta_T = beta_filt[-1]

        for h in horizons:
            t_fut = t + h
            if t_fut >= T:
                continue
            tau1_f = tau1_t[t]
            tau2_f = tau2_t[t] if tau2_t is not None else None
            H_f    = get_H(maturities, tau1_f, tau2_f, model)
            beta_h = _propagate_beta(beta_T, mu_final, F_final, h)
            y_hat  = H_f @ beta_h
            y_real = yields[t_fut]
            for n, m in enumerate(maturities):
                records.append({
                    'date'    : dates[t_fut],
                    'maturity': m,
                    'horizon' : h,
                    'y_real'  : float(y_real[n]),
                    'y_hat'   : float(y_hat[n]),
                    'error'   : float(y_real[n] - y_hat[n]),
                })

    df = pd.DataFrame(records)
    if verbose and len(df):
        _print_oos_summary(df, 'KF')
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# VENTANA EXPANDIENTE — BAYESIANO
# ═══════════════════════════════════════════════════════════════════════════════

def expanding_window_bayesian(yields: np.ndarray,
                              maturities: np.ndarray,
                              dates: np.ndarray,
                              priors: dict,
                              horizons: list[int] = [1, 3, 6, 12],
                              min_train: int = 120,
                              refit_every: int = 21,
                              n_iter: int = 800,
                              n_burnin: int = 400,
                              n_chains: int = 2,
                              tau_mode: str = 'estatico',
                              verbose: bool = True,
                              **gibbs_kwargs) -> pd.DataFrame:
    """
    Evaluación OOS bayesiana con re-estimación periódica cada `refit_every` días.

    NOTA: re-estimar el Gibbs sampler en cada paso t es costoso.
    Se re-ajusta solo cada `refit_every` periodos y se reutilizan
    los parámetros intermedios.

    Retorna
    -------
    pd.DataFrame con las mismas columnas que expanding_window_kf
    """
    from .gibbs import run_chains_parallel   # import diferido — evita circularidad

    T, N  = yields.shape
    model = priors['model']

    if verbose:
        n_steps = len(range(min_train, T - max(horizons), refit_every))
        print(f"OOS-Bayes: ~{n_steps} re-estimaciones × {len(horizons)} horizontes")

    records    = []
    last_chains = None

    for t in range(min_train, T - max(horizons), refit_every):
        # Re-estimación Gibbs
        last_chains = run_chains_parallel(
            yields[:t], maturities, priors,
            n_iter=n_iter, n_burnin=n_burnin,
            n_chains=n_chains, verbose=False,
            tau_mode=tau_mode, **gibbs_kwargs)

        # Pronóstico posterior predictivo
        y_hats = forecast_bayesian(last_chains, maturities, horizons, tau_mode)

        for h in horizons:
            t_fut = t + h
            if t_fut >= T:
                continue
            y_hat  = y_hats[h]
            y_real = yields[t_fut]
            for n, m in enumerate(maturities):
                records.append({
                    'date'    : dates[t_fut],
                    'maturity': m,
                    'horizon' : h,
                    'y_real'  : float(y_real[n]),
                    'y_hat'   : float(y_hat[n]),
                    'error'   : float(y_real[n] - y_hat[n]),
                })

        if verbose:
            t_str = str(dates[t])[:10]
            print(f"  t={t} ({t_str})  tau1={last_chains[0]['tau1'].mean():.3f}")

    df = pd.DataFrame(records)
    if verbose and len(df):
        _print_oos_summary(df, 'Bayesiano')
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# MÉTRICAS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(df: pd.DataFrame,
                    pb_scale: bool = True) -> pd.DataFrame:
    """
    Calcula RMSE y MAE por (horizonte, plazo).

    Parámetros
    ----------
    df        : DataFrame de expanding_window_kf / expanding_window_bayesian
    pb_scale  : Si True, reporta en puntos básicos (×100)

    Retorna
    -------
    pd.DataFrame con columnas: horizon, maturity, RMSE, MAE, N_obs
    """
    scale = 100 if pb_scale else 1
    rows  = []
    for (h, m), g in df.groupby(['horizon', 'maturity']):
        e     = g['error'].values * scale
        rows.append({
            'horizon' : h,
            'maturity': m,
            'RMSE'    : float(np.sqrt(np.mean(e ** 2))),
            'MAE'     : float(np.mean(np.abs(e))),
            'N_obs'   : len(e),
        })
    return pd.DataFrame(rows).sort_values(['horizon', 'maturity']).reset_index(drop=True)


def summary_metrics(df: pd.DataFrame,
                    model_label: str = '',
                    pb_scale: bool = True) -> pd.DataFrame:
    """
    Tabla resumen: RMSE y MAE promediados sobre plazos, por horizonte.

    Útil para comparar KF vs Bayesiano en una sola línea por horizonte.
    """
    metrics = compute_metrics(df, pb_scale)
    rows    = []
    scale_unit = 'pb' if pb_scale else 'pp'
    for h, g in metrics.groupby('horizon'):
        rows.append({
            'Modelo'   : model_label,
            'Horizonte': h,
            f'RMSE_prom ({scale_unit})': round(g['RMSE'].mean(), 4),
            f'MAE_prom  ({scale_unit})': round(g['MAE'].mean(), 4),
            'RMSE_max' : round(g['RMSE'].max(), 4),
            'MAE_max'  : round(g['MAE'].max(), 4),
        })
    return pd.DataFrame(rows)


def compare_models(df_kf: pd.DataFrame,
                   df_bayes: pd.DataFrame,
                   pb_scale: bool = True) -> pd.DataFrame:
    """
    Tabla comparativa KF vs Bayesiano: RMSE relativo y diferencia MAE.
    """
    s_kf    = summary_metrics(df_kf,    'KF',       pb_scale)
    s_bayes = summary_metrics(df_bayes, 'Bayesiano', pb_scale)
    key     = f'RMSE_prom ({"pb" if pb_scale else "pp"})'
    key_mae = f'MAE_prom  ({"pb" if pb_scale else "pp"})'

    merged = s_kf.merge(s_bayes, on='Horizonte', suffixes=('_KF', '_Bayes'))
    merged['ΔRMSE']      = merged[f'{key}_Bayes'] - merged[f'{key}_KF']
    merged['ΔMAE']       = merged[f'{key_mae}_Bayes'] - merged[f'{key_mae}_KF']
    merged['Mejor']      = np.where(merged['ΔRMSE'] < 0, 'Bayesiano', 'KF')
    return merged[['Horizonte', f'{key}_KF', f'{key}_Bayes', 'ΔRMSE', 'ΔMAE', 'Mejor']]


# ── Helpers internos ──────────────────────────────────────────────────────────

def _print_oos_summary(df: pd.DataFrame, label: str) -> None:
    m = compute_metrics(df, pb_scale=True)
    print(f"\n  OOS-{label} — RMSE promedio por horizonte (pb):")
    for h, g in m.groupby('horizon'):
        print(f"    h={h:>2}m  RMSE={g['RMSE'].mean():.3f}  MAE={g['MAE'].mean():.3f}")
