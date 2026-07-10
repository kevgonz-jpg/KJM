"""
diagnostics.py
==============
Diagnósticos de convergencia MCMC (R-hat, ESS) y análisis de
residuos / innovaciones del filtro para modelos DNS/DNSS.

Funciones exportadas
--------------------
gelman_rubin(chains, key, idx)
effective_sample_size(series)
compute_diagnostics(chains)           → pd.DataFrame
print_diagnostics(chains)
ljung_box_pval(series, lags)
jarque_bera_pval(series)
residual_stats(residuals, maturities) → pd.DataFrame
"""

import numpy as np
import pandas as pd
from scipy import stats


# ═══════════════════════════════════════════════════════════════════════════════
# DIAGNÓSTICOS MCMC
# ═══════════════════════════════════════════════════════════════════════════════

def gelman_rubin(chains: list, key: str, idx: int | None = None) -> float:
    """
    Estadístico R-hat de Gelman-Rubin (1992).

    Parámetros
    ----------
    chains : lista de dicts (resultado de run_chain / run_chains_parallel)
    key    : nombre del parámetro en el dict (e.g. 'tau1', 'f_vec')
    idx    : índice del vector si el parámetro es multidimensional

    Retorna
    -------
    R-hat : float   (<1.05 = excelente, <1.10 = aceptable, ≥1.10 = no convergió)
    """
    series = []
    for ch in chains:
        s = ch[key] if idx is None else ch[key][:, idx]
        series.append(np.asarray(s, dtype=float))

    m = len(series)
    n = len(series[0])
    chain_means = np.array([s.mean() for s in series])
    chain_vars  = np.array([s.var(ddof=1) for s in series])
    grand_mean  = chain_means.mean()

    B       = n / (m - 1) * np.sum((chain_means - grand_mean) ** 2)
    W       = chain_vars.mean()
    var_hat = (n - 1) / n * W + B / n

    return float(np.sqrt(var_hat / W)) if W > 1e-15 else float('nan')


def effective_sample_size(series: np.ndarray) -> float:
    """
    ESS = n / (1 + 2·Σ ρₖ).

    Para la serie MCMC, suma las autocorrelaciones hasta que |ρₖ| < 0.05.
    """
    x   = np.asarray(series, dtype=float)
    n   = len(x)
    xc  = x - x.mean()
    var = xc.var()
    if var < 1e-15:
        return float(n)
    rho_sum = 0.0
    for k in range(1, min(100, n // 2)):
        rho_k = np.mean(xc[:n - k] * xc[k:]) / var
        if abs(rho_k) < 0.05:
            break
        rho_sum += rho_k
    return max(1.0, n / (1.0 + 2.0 * rho_sum))


def compute_diagnostics(chains: list) -> pd.DataFrame:
    """
    Tabla R-hat y ESS para todos los parámetros escalares y vectoriales.
    Incluye semáforo: OK (<1.05), ~ (<1.10), NO (≥1.10).

    Retorna
    -------
    pd.DataFrame con columnas: Parametro, Rhat, ESS_min, ESS_med, Converge
    """
    model = chains[0]['model']
    k     = chains[0]['mu'].shape[1]
    nombres_b = ['beta0(nivel)', 'beta1(pend.)', 'beta2(curv.)', 'beta3(curv2)'][:k]

    checks = [('tau1', None, 'tau1')]
    if model == 'DNSS':
        checks.append(('tau2', None, 'tau2'))
    for i, nb in enumerate(nombres_b):
        checks += [
            ('f_vec', i, f'f({nb})'),
            ('mu',    i, f'mu({nb})'),
            ('q_vec', i, f'q({nb})'),
        ]
    for n in range(min(chains[0]['r_vec'].shape[1], 8)):
        checks.append(('r_vec', n, f'r_vec[{n}]'))

    rows = []
    for key, idx, name in checks:
        rhat     = gelman_rubin(chains, key, idx)
        ess_vals = [effective_sample_size(
            ch[key] if idx is None else ch[key][:, idx]) for ch in chains]
        semaforo = 'OK' if rhat < 1.05 else ('~' if rhat < 1.10 else 'NO')
        rows.append({'Parametro': name, 'Rhat': round(rhat, 4),
                     'ESS_min': round(min(ess_vals), 0),
                     'ESS_med': round(np.mean(ess_vals), 0),
                     'Converge': semaforo})
    return pd.DataFrame(rows)


def print_diagnostics(chains: list) -> None:
    """Imprime tabla de diagnósticos MCMC con semáforos."""
    df = compute_diagnostics(chains)
    print(f"\n{'=' * 58}")
    print(f"  Diagnósticos MCMC — {chains[0]['model']}")
    print(f"{'=' * 58}")
    print(f"  {'Parámetro':<20} {'Rhat':>7} {'ESS_min':>8} {'ESS_med':>8} {'Conv.':>6}")
    print(f"  {'─' * 52}")
    for _, row in df.iterrows():
        flag = {'OK': 'OK', '~': '~~', 'NO': 'XX'}.get(row['Converge'], '??')
        print(f"  {row['Parametro']:<20} {row['Rhat']:>7.4f} "
              f"{row['ESS_min']:>8.0f} {row['ESS_med']:>8.0f}  {flag}")
    n_bad = (df['Converge'] != 'OK').sum()
    if n_bad == 0:
        print(f"\n  Todos los parámetros convergieron (Rhat < 1.05)")
    else:
        print(f"\n  {n_bad} parámetros con Rhat ≥ 1.05 — considerar más iteraciones")


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS ESTADÍSTICOS DE RESIDUOS / INNOVACIONES
# ═══════════════════════════════════════════════════════════════════════════════

def ljung_box_pval(series: np.ndarray, lags: int = 10) -> float:
    """
    p-valor del test Ljung-Box para autocorrelación.
    H₀: no hay autocorrelación hasta lag `lags`.
    p > 0.05 → no se rechaza H₀ (ruido blanco).
    """
    x   = np.asarray(series, dtype=float)
    T   = len(x)
    xc  = x - x.mean()
    xcs = xc / (xc.std() + 1e-15)
    lb_stat = T * (T + 2) * sum(
        np.corrcoef(xcs[:T - lag], xcs[lag:])[0, 1] ** 2 / (T - lag)
        for lag in range(1, lags + 1)
    )
    return float(1.0 - stats.chi2.cdf(lb_stat, df=lags))


def jarque_bera_pval(series: np.ndarray) -> float:
    """p-valor del test Jarque-Bera de normalidad."""
    return float(stats.jarque_bera(series).pvalue)


# ═══════════════════════════════════════════════════════════════════════════════
# AIC / BIC — genérico, model-agnostic (KF, Gibbs, PF)
# ═══════════════════════════════════════════════════════════════════════════════

def aic_bic(loglik: float, k: int, T: int) -> tuple:
    """
    Criterios de información de Akaike y Bayesiano.

        AIC = 2k − 2·log-lik
        BIC = k·ln(T) − 2·log-lik

    Parámetros
    ----------
    loglik : log-verosimilitud (o estimación SMC del log-lik marginal, en
             el caso del Particle Filter — ver advertencia abajo)
    k      : número de parámetros libres del modelo
    T      : número de observaciones (periodos)

    Advertencia (Particle Filter)
    ------------------------------
    Si `loglik` proviene de un PF, es una ESTIMACIÓN Monte Carlo con
    varianza finita en N (número de partículas), no un valor exacto como en
    el KF. El AIC/BIC heredan esa varianza. Para una conclusión robusta,
    recomputar con 3-5 semillas distintas y reportar el rango, no solo el
    punto estimado.

    Retorna
    -------
    (aic, bic) : float, float
    """
    aic = 2 * k - 2 * loglik
    bic = k * np.log(T) - 2 * loglik
    return float(aic), float(bic)


def residual_stats(residuals: np.ndarray,
                   maturities: np.ndarray,
                   pb_scale: bool = True) -> pd.DataFrame:
    """
    Estadísticos de residuos por plazo: RMSE, MAE, sesgo, JB, LB.

    Parámetros
    ----------
    residuals  : (T, N)
    maturities : (N,)
    pb_scale   : si True, reporta en puntos básicos (×100)

    Retorna
    -------
    pd.DataFrame con una fila por plazo
    """
    scale = 100 if pb_scale else 1
    T, N  = residuals.shape
    rows  = []
    for i, m in enumerate(maturities):
        r     = residuals[:, i] * scale
        rmse  = float(np.sqrt(np.mean(r ** 2)))
        mae   = float(np.mean(np.abs(r)))
        bias  = float(r.mean())
        jb_p  = jarque_bera_pval(r)
        lb_p  = ljung_box_pval(r)
        rows.append({
            'Plazo'  : f'{m:.0f}Y',
            'RMSE'   : round(rmse, 4),
            'MAE'    : round(mae, 4),
            'Sesgo'  : round(bias, 4),
            'JB_pval': round(jb_p, 4),
            'LB_pval': round(lb_p, 4),
            'Normal' : 'Sí' if jb_p > 0.05 else 'No',
            'Blanco' : 'Sí' if lb_p > 0.05 else 'No',
        })
    return pd.DataFrame(rows)
