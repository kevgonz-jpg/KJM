"""
particle_filter.py
===================
Bootstrap Particle Filter (SIR) para los modelos DNS/DNSS, con τ_t
formulado como parte del ESTADO AUMENTADO en lugar de un hiperparámetro
fijo obtenido vía grid search.

A diferencia de kalman.py (τ_t fijo, insumo externo del grid search de
grid_search.py) y de gibbs.py (τ estático o AR(1) latente, pero muestreado
con MH dentro de un esquema bayesiano completo), aquí τ_t evoluciona como
un AR(1) en escala logarítmica dentro del propio filtro, y la ecuación de
medición se vuelve no lineal en el estado aumentado x_t = (β_t, log τ_t).
Esto motiva un Particle Filter en vez de un Extended Kalman Filter: no se
necesita linealizar Λ(τ) en cada paso, solo evaluarla por partícula.

El warm-start de (μ, F, Q, R) reutiliza íntegramente kalman.py + grid_search.py
— no se reestima nada de cero con un método distinto. Lo único nuevo que el
PF introduce son los hiperparámetros de la dinámica de τ_t (calibrados por
OLS sobre la serie de grid_search.py), más los hiperparámetros propios del
filtro de partículas (N, ess_threshold, sep_min, innovation_dist, ...).

Funciones exportadas
--------------------
calibrate_tau_ar1(tau_t, ...)              — calibración OLS de la dinámica de τ
predicted_yields(maturities, beta, ltau, model)
propagate_beta / propagate_tau / enforce_tau_order
gaussian_loglik / student_t_loglik
systematic_resample
run_particle_filter(yields, maturities, model, n_particles, params, config, ...)
fit_model_kf_cached(...)                   — warm-start del KF con cache en disco
fit_pf_model(...)                          — pipeline completo (función principal)

Referencias
-----------
Gordon, Salmond & Smith (1993)      — bootstrap PF original
Kim, Shephard & Chib (1998)         — PF para volatilidad estocástica
Pitt & Shephard (1999)              — auxiliary particle filter
Doucet, de Freitas & Gordon (2001)  — tutorial de SMC
Andrieu, Doucet & Holenstein (2010) — Particle MCMC (extensión futura)
"""

import os
import sys
import time
import hashlib
import pickle

import numpy as np
from scipy.special import gammaln

from .nelson_siegel import get_H
from .grid_search import rolling_tau_dns, rolling_tau_dnss
from .kalman import (kalman_filter, build_H_array, compute_P0_stationary,
                     mle_estimate)
from .diagnostics import aic_bic


# ═══════════════════════════════════════════════════════════════════════════════
# CALIBRACIÓN DE LA DINÁMICA AR(1) DE τ
# ═══════════════════════════════════════════════════════════════════════════════

def calibrate_tau_ar1(tau_t: np.ndarray,
                      floor_phi: float = 0.80,
                      cap_phi: float = 0.999) -> tuple:
    """
    Calibra (mu_ltau, phi_ltau, q_ltau) vía OLS sobre log(tau_t), usando la
    serie ya producida por rolling_tau_dns/rolling_tau_dnss (grid_search.py)
    como si fuera "data observada" de τ.

    Advertencia metodológica: tau_t del grid search es ya una versión
    filtrada/discretizada (ventana móvil + grid discreto) del proceso
    continuo subyacente. Su varianza empírica de innovaciones SUBESTIMA
    la verdadera varianza de innovación de τ — es un límite inferior, no
    un valor exacto. Por eso fit_pf_model() expone `q_ltau_scale` para
    poder inflarla.

    Retorna
    -------
    (mu_ltau, phi_ltau, q_ltau) : float, float, float
    """
    ltau = np.log(np.asarray(tau_t, dtype=float))
    y, x = ltau[1:], ltau[:-1]
    xm, ym = x.mean(), y.mean()
    phi = np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2)
    phi = float(np.clip(phi, floor_phi, cap_phi))
    intercept = ym - phi * xm
    mu_ltau = intercept / (1 - phi)
    resid = y - (intercept + phi * x)
    q_ltau = float(resid.var(ddof=1))
    return mu_ltau, phi, q_ltau


# ═══════════════════════════════════════════════════════════════════════════════
# NÚCLEO MATEMÁTICO DEL PF (vectorizado sobre el eje de partículas)
# ═══════════════════════════════════════════════════════════════════════════════

def predicted_yields(maturities: np.ndarray, beta: np.ndarray,
                     ltau: np.ndarray, model: str) -> np.ndarray:
    """y_hat (N_particulas, N_madurez) para un lote completo de partículas."""
    tau1 = np.exp(ltau[:, 0])
    m = maturities[None, :]
    mt1 = m / tau1[:, None]
    e1 = np.exp(-mt1)
    c1 = (1.0 - e1) / mt1
    y_hat = beta[:, 0:1] * 1.0 + beta[:, 1:2] * c1 + beta[:, 2:3] * (c1 - e1)
    if model == 'DNSS':
        tau2 = np.exp(ltau[:, 1])
        mt2 = m / tau2[:, None]
        e2 = np.exp(-mt2)
        c2 = (1.0 - e2) / mt2
        y_hat = y_hat + beta[:, 3:4] * (c2 - e2)
    return y_hat


def propagate_beta(beta_prev, mu_beta, F_diag, sqrtQ_diag, rng):
    """Un paso del AR(1) diagonal de los factores beta."""
    noise = rng.standard_normal(beta_prev.shape) * sqrtQ_diag[None, :]
    return mu_beta[None, :] + (beta_prev - mu_beta[None, :]) * F_diag[None, :] + noise


def propagate_tau(ltau_prev, mu_ltau, phi_ltau, sqrtq_ltau, rng):
    """Un paso del AR(1) de log-tau (independiente por cada tau)."""
    noise = rng.standard_normal(ltau_prev.shape) * sqrtq_ltau[None, :]
    return mu_ltau[None, :] + (ltau_prev - mu_ltau[None, :]) * phi_ltau[None, :] + noise


def enforce_tau_order(ltau, sep_min, tau_floor=0.05):
    """
    Restricción direccional para DNSS: tau1 (hump largo) > tau2 (hump corto)
    + sep_min — la misma restricción de identificación de
    grid_search.rolling_tau_dnss, pero aplicada PARTÍCULA POR PARTÍCULA en
    cada periodo (no solo descartando pares inválidos de un grid estático).
    Se ordena por magnitud (resuelve el label-switching por construcción) y
    se empuja hacia abajo el menor si la separación no se cumple.
    """
    tau = np.exp(ltau)
    tau_hi = np.maximum(tau[:, 0], tau[:, 1])
    tau_lo = np.minimum(tau[:, 0], tau[:, 1])
    deficit = np.maximum(sep_min - (tau_hi - tau_lo), 0.0)
    tau_lo_new = np.maximum(tau_lo - deficit, tau_floor)
    return np.log(np.column_stack([tau_hi, tau_lo_new]))


def gaussian_loglik(y_t, y_hat, R_diag):
    """log p(y_t | particula) bajo innovaciones Gaussianas independientes."""
    resid = y_t[None, :] - y_hat
    return -0.5 * np.sum(resid ** 2 / R_diag[None, :] + np.log(2 * np.pi * R_diag[None, :]), axis=1)


def student_t_loglik(y_t, y_hat, R_diag, nu):
    """
    log p(y_t | particula) bajo innovaciones Student-t independientes
    (colas pesadas). La escala se calibra para que Var coincida con R_diag
    cuando nu > 2. Relevante directamente para el hallazgo de colas pesadas
    en los QQ-plots de diagnostics.residual_stats.
    """
    resid = y_t[None, :] - y_hat
    scale2 = R_diag * (nu - 2.0) / nu
    z2 = resid ** 2 / scale2[None, :]
    c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi * scale2)
    return np.sum(c[None, :] - ((nu + 1) / 2) * np.log(1 + z2 / nu), axis=1)


def _normalize_log_weights(logw):
    mx = np.max(logw)
    w = np.exp(logw - mx)
    s = w.sum()
    return w / s, mx + np.log(s)


def _ess(w):
    """Effective Sample Size = 1 / sum(w_i^2), con w normalizado."""
    return 1.0 / np.sum(w ** 2)


def systematic_resample(w, rng):
    """Remuestreo sistemático — menor varianza que el multinomial puro para el mismo N."""
    Np = len(w)
    positions = (rng.random() + np.arange(Np)) / Np
    cumsum = np.cumsum(w)
    cumsum[-1] = 1.0
    return np.searchsorted(cumsum, positions)


# ═══════════════════════════════════════════════════════════════════════════════
# BARRA DE PROGRESO (sin dependencias externas)
# ═══════════════════════════════════════════════════════════════════════════════

def _format_eta(seconds):
    seconds = max(0, seconds)
    if seconds >= 60:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{seconds:.1f}s"


def _pf_progress_bar(t, T, t0_wall, ess_ratio, n_resampled, bar_width=28):
    """
    Barra de una sola línea (se sobreescribe con \\r). El KF es cerrado y
    rapidísimo; el PF escala con N_particulas y puede tardar de segundos a
    minutos — de ahí que convenga monitorear progreso, ESS y ETA en vivo.
    """
    frac = (t + 1) / T
    elapsed = time.time() - t0_wall
    rate = (t + 1) / elapsed if elapsed > 0 else 0.0
    eta = (T - (t + 1)) / rate if rate > 0 else float('nan')
    filled = int(bar_width * frac)
    bar = '#' * filled + '-' * (bar_width - filled)
    msg = (f"\r  [{bar}] {frac*100:5.1f}%  t={t+1}/{T}  "
           f"ESS/N={ess_ratio:4.2f}  remuestreos={n_resampled}  "
           f"{rate:,.0f} it/s  transcurrido={_format_eta(elapsed)}  ETA={_format_eta(eta)}   ")
    sys.stdout.write(msg)
    sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════════════════════
# LOOP PRINCIPAL — SEQUENTIAL IMPORTANCE RESAMPLING (SIR)
# ═══════════════════════════════════════════════════════════════════════════════

def run_particle_filter(yields, maturities, model, n_particles, params, config,
                        show_progress=True, progress_every=None):
    """
    Bootstrap Particle Filter (SIR con remuestreo adaptativo por ESS) sobre
    el estado aumentado x_t = (beta_t, log-tau_t).

    Para cada periodo t: (1) predicción — propagar cada partícula con la
    transición AR(1); (2) verosimilitud incremental p(y_t|x_t^(i));
    (3) actualización de pesos w_t ∝ w_{t-1}·p(y_t|x_t); (4) log-verosimilitud
    marginal acumulada (estimación SMC); (5) remuestreo condicional por ESS.

    Disciplina OOS: se guardan por separado x_pred_mean (β̂_{t|t-1}, ANTES
    de ver y_t) y x_filt_mean (β̂_{t|t}, DESPUÉS) — la misma distinción que
    forecasting.py aplica para el KF y el Gibbs (β predichos, no suavizados,
    para evaluación fuera de muestra).

    Parámetros
    ----------
    yields, maturities, model : igual convención que el resto del paquete
    n_particles : int
        Número de partículas N. Determina la varianza de Monte Carlo del
        log-lik y de los estados estimados. Punto de partida razonable:
        1000-5000; validar con un análisis de sensibilidad.
    params : dict
        mu_beta, F_beta_diag, Q_beta_diag : dinámica de beta (warm-start del KF)
        mu_ltau, phi_ltau, q_ltau         : dinámica AR(1) de log(tau) —
                                            arrays shape (1,) en DNS, (2,) en DNSS
        R_diag                             : varianza de medición por plazo
    config : dict
        seed             : semilla (CRÍTICO — el PF tiene varianza de Monte
                           Carlo finita en N, a diferencia del KF)
        ess_threshold    : fracción de N bajo la cual se dispara el remuestreo
        sep_min          : separación mínima tau1 > tau2 + sep_min (solo DNSS)
        innovation_dist  : 'gaussian' | 'student_t'
        student_t_df     : grados de libertad si innovation_dist=='student_t'
    show_progress : bool
        Barra de progreso de una sola línea con ETA y velocidad (it/s).
    progress_every : int | None
        Cada cuántos periodos actualizar la barra (None → automático, ~200
        actualizaciones en total).

    Retorna
    -------
    dict con un esquema de claves compatible con el resultado del KF
    (beta_filtered, tau1_t, tau2_t, residuals, fitted, rmse_total, ...) más
    diagnósticos propios del PF: ess_t, resampled_t, x_filt_mean, x_filt_std,
    loglik, aic, bic.
    """
    T, Nm = yields.shape
    n_beta = 3 if model == 'DNS' else 4
    n_tau = 1 if model == 'DNS' else 2
    rng = np.random.default_rng(config.get('seed', 0))

    mu_beta = params['mu_beta']; F_diag = params['F_beta_diag']
    sqrtQ = np.sqrt(params['Q_beta_diag'])
    mu_ltau = np.atleast_1d(params['mu_ltau']); phi_ltau = np.atleast_1d(params['phi_ltau'])
    sqrtq_ltau = np.sqrt(np.atleast_1d(params['q_ltau']))
    R_diag = params['R_diag']
    sep_min = config.get('sep_min', 0.30)
    ess_thr = config.get('ess_threshold', 0.5) * n_particles
    dist = config.get('innovation_dist', 'gaussian')
    nu = config.get('student_t_df', 5.0)

    P0_beta = params['Q_beta_diag'] / np.maximum(1 - F_diag ** 2, 1e-6)
    beta = mu_beta[None, :] + rng.standard_normal((n_particles, n_beta)) * np.sqrt(P0_beta)[None, :]
    var0_ltau = np.atleast_1d(params['q_ltau']) / np.maximum(1 - phi_ltau ** 2, 1e-6)
    ltau = mu_ltau[None, :] + rng.standard_normal((n_particles, n_tau)) * np.sqrt(var0_ltau)[None, :]
    if model == 'DNSS':
        ltau = enforce_tau_order(ltau, sep_min)
    w = np.full(n_particles, 1.0 / n_particles)

    kdim = n_beta + n_tau
    x_filt_mean = np.zeros((T, kdim)); x_filt_std = np.zeros((T, kdim))
    x_pred_mean = np.zeros((T, kdim))
    fitted_pred = np.zeros((T, Nm)); fitted_filt = np.zeros((T, Nm))
    ess_t = np.zeros(T); resampled_t = np.zeros(T, dtype=bool)
    loglik = 0.0

    if progress_every is None:
        progress_every = max(1, T // 200)
    t0_wall = time.time()
    n_resampled_so_far = 0

    for t in range(T):
        beta_pred = propagate_beta(beta, mu_beta, F_diag, sqrtQ, rng)
        ltau_pred = propagate_tau(ltau, mu_ltau, phi_ltau, sqrtq_ltau, rng)
        if model == 'DNSS':
            ltau_pred = enforce_tau_order(ltau_pred, sep_min)

        y_hat_pred = predicted_yields(maturities, beta_pred, ltau_pred, model)
        x_t = np.hstack([beta_pred, ltau_pred])
        fitted_pred[t] = w @ y_hat_pred
        x_pred_mean[t] = w @ x_t

        ll_i = (gaussian_loglik(yields[t], y_hat_pred, R_diag) if dist == 'gaussian'
                else student_t_loglik(yields[t], y_hat_pred, R_diag, nu))
        logw_unnorm = np.log(w + 1e-300) + ll_i
        w_new, log_sum = _normalize_log_weights(logw_unnorm)
        loglik += log_sum

        fitted_filt[t] = w_new @ y_hat_pred
        x_filt_mean[t] = w_new @ x_t
        var_filt = w_new @ (x_t ** 2) - x_filt_mean[t] ** 2
        x_filt_std[t] = np.sqrt(np.maximum(var_filt, 0))

        ess_val = _ess(w_new); ess_t[t] = ess_val
        if ess_val < ess_thr:
            idx = systematic_resample(w_new, rng)
            beta, ltau = beta_pred[idx], ltau_pred[idx]
            w = np.full(n_particles, 1.0 / n_particles)
            resampled_t[t] = True
            n_resampled_so_far += 1
        else:
            beta, ltau = beta_pred, ltau_pred
            w = w_new
            resampled_t[t] = False

        if show_progress and ((t + 1) % progress_every == 0 or t == T - 1):
            _pf_progress_bar(t, T, t0_wall, ess_val / n_particles, n_resampled_so_far)

    if show_progress:
        sys.stdout.write('\n'); sys.stdout.flush()

    # ── Empaquetado en esquema compatible con kalman_filter / generate_full_report ──
    tau1_t = np.exp(x_filt_mean[:, n_beta])
    tau2_t = np.exp(x_filt_mean[:, n_beta + 1]) if model == 'DNSS' else None
    beta_filtered = x_filt_mean[:, :n_beta]

    H_arr_pf = build_H_array(maturities, tau1_t, tau2_t, model)
    resid = yields - fitted_filt
    rmse_total = np.sqrt(np.mean(resid ** 2)); mae_total = np.mean(np.abs(resid))
    rmse_mat = np.sqrt(np.mean(resid ** 2, axis=0)); mae_mat = np.mean(np.abs(resid), axis=0)

    k_nuevo_tau = 3 * n_tau   # mu_ltau, phi_ltau, q_ltau por cada tau (lo que el PF agrega)
    aic_pf, bic_pf = aic_bic(loglik, k_nuevo_tau, T)

    return dict(
        model=model, n_particles=n_particles, tau1_t=tau1_t, tau2_t=tau2_t,
        H_arr=H_arr_pf, R_diag=R_diag,
        beta_filtered=beta_filtered, beta_smoothed=beta_filtered,   # no hay smoother — ver docstring
        x_filt_mean=x_filt_mean, x_filt_std=x_filt_std, x_pred_mean=x_pred_mean,
        fitted=fitted_filt, fitted_pred=fitted_pred, residuals=resid,
        loglik=loglik, aic=aic_pf, bic=bic_pf, k_nuevo=k_nuevo_tau,
        rmse_total=rmse_total, mae_total=mae_total,
        rmse_mat=rmse_mat, mae_mat=mae_mat,
        ess_t=ess_t, resampled_t=resampled_t,
        maturities=maturities, dates=None, yields=yields,
        innovation_dist=dist, student_t_df=nu, sep_min=sep_min,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# WARM-START DEL KF CON CACHE EN DISCO (experimentación rápida)
# ═══════════════════════════════════════════════════════════════════════════════

CACHE_DIR_DEFAULT = "outputs/cache_kf_warmstart"


def _kf_cache_key(yields, maturities, model, tau1_grid, tau2_grid, window_size, n_starts, r_mode):
    """Hash determinista de todo lo que afecta el resultado del KF warm-start."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(yields).tobytes())
    h.update(np.ascontiguousarray(maturities).tobytes())
    h.update(model.encode())
    h.update(np.ascontiguousarray(tau1_grid).tobytes())
    if tau2_grid is not None:
        h.update(np.ascontiguousarray(tau2_grid).tobytes())
    h.update(str(window_size).encode())
    h.update(str(n_starts).encode())
    h.update(r_mode.encode())
    return h.hexdigest()[:16]


def fit_model_kf_cached(yields, maturities, dates, model='DNS',
                        tau1_grid=None, tau2_grid=None, window_size=None,
                        n_starts=4, r_mode='libre', sep_min=0.30,
                        verbose=True, force_refit=False,
                        cache_dir=CACHE_DIR_DEFAULT):
    """
    Warm-start del KF (grid_search + kalman.mle_estimate + kalman_filter)
    con cache en disco — pensado para que `fit_pf_model()` no tenga que
    re-correr la parte lenta (grid search + MLE multi-start) cada vez que
    se quiere experimentar con hiperparámetros del PF (N_particulas, seed,
    q_ltau_scale, ess_threshold, ...), que en sí mismos no cambian en nada
    el resultado del KF.

    force_refit=True ignora el cache y reestima de cero. El cache se
    invalida automáticamente si cambian los datos o cualquier argumento
    que afecte al resultado (ver `_kf_cache_key`).

    Retorna
    -------
    dict : mismo esquema que produce internamente esta función — model,
    tau1_t, tau2_t, H_arr, mu, F, Q, R_diag, beta_filtered, beta_smoothed
    (= filtrado, no hay RTS aquí — usar kalman.rts_smoother aparte si se
    necesita para otros fines), fitted, residuals, loglik, aic, bic,
    rmse_total, mae_total, rmse_mat, mae_mat, maturities, dates, yields.
    """
    os.makedirs(cache_dir, exist_ok=True)
    T, N = yields.shape
    k = 3 if model == 'DNS' else 4

    if tau1_grid is None:
        tau1_grid = np.linspace(0.3, 5.0, 35)
    if tau2_grid is None and model == 'DNSS':
        tau2_grid = np.linspace(0.15, 3.0, 20)

    key = _kf_cache_key(yields, maturities, model, tau1_grid, tau2_grid, window_size, n_starts, r_mode)
    path = os.path.join(cache_dir, f"kf_warmstart_{model}_{key}.pkl")

    if not force_refit and os.path.exists(path):
        with open(path, 'rb') as f:
            res_kf = pickle.load(f)
        if verbose:
            print(f"  [KF cache] HIT  -- usando warm-start guardado ({path})")
            print(f"             RMSE={res_kf['rmse_total']*100:.4f}pb  log-lik={res_kf['loglik']:,.2f}")
        res_kf['dates'] = dates
        return res_kf

    if verbose:
        print(f"  [KF cache] MISS -- ajustando warm-start desde cero "
              f"(grid search + MLE, esto es lo lento)...")

    t0 = time.time()
    if model == 'DNS':
        tau1_t, sse_t, _ = rolling_tau_dns(yields, maturities, tau1_grid, window_size)
        tau2_t = None
    else:
        tau1_t, tau2_t, sse_t = rolling_tau_dnss(
            yields, maturities, tau1_grid, tau2_grid, window_size, sep_min=sep_min)

    H_arr = build_H_array(maturities, tau1_t, tau2_t, model)
    mu, F, Q, R_diag, best_ll, best_params, r_groups_log = mle_estimate(
        yields, H_arr, k, N, F_type='diagonal', n_starts=n_starts,
        verbose=verbose, r_mode=r_mode, maturities=maturities)

    P0 = compute_P0_stationary(F, Q, 'diagonal')
    beta0 = mu.copy()
    beta_pred, beta_filt, P_pred, P_filt, innov, loglik = kalman_filter(
        yields, H_arr, F, Q, R_diag, mu, beta0, P0)

    fitted = np.einsum('tni,ti->tn', H_arr, beta_filt)
    resid = yields - fitted
    rmse_total = np.sqrt(np.mean(resid ** 2)); mae_total = np.mean(np.abs(resid))
    rmse_mat = np.sqrt(np.mean(resid ** 2, axis=0)); mae_mat = np.mean(np.abs(resid), axis=0)

    k_kf = k + k + k + N   # mu + F_diag + Q_diag + R_diag (libre); ver build_latex_report para 'grupos'
    aic_kf, bic_kf = aic_bic(loglik, k_kf, T)

    if verbose:
        print(f"  [KF warm-start] RMSE={rmse_total*100:.4f}pb  log-lik={loglik:,.2f}"
              f"  ({time.time()-t0:.1f}s total)")

    res_kf = dict(
        model=model, tau1_t=tau1_t, tau2_t=tau2_t, H_arr=H_arr,
        mu=mu, F=F, Q=Q, R_diag=R_diag, k=k_kf,
        beta_filtered=beta_filt, beta_smoothed=beta_filt,
        P_filtered=P_filt, P_predicted=P_pred, beta_predicted=beta_pred,
        innovations=innov, fitted=fitted, residuals=resid,
        loglik=loglik, aic=aic_kf, bic=bic_kf,
        rmse_total=rmse_total, mae_total=mae_total, rmse_mat=rmse_mat, mae_mat=mae_mat,
        maturities=maturities, dates=dates, yields=yields,
    )
    with open(path, 'wb') as f:
        pickle.dump(res_kf, f)
    if verbose:
        print(f"  [KF cache] guardado en {path}")
    return res_kf


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL — fit_pf_model()
# ═══════════════════════════════════════════════════════════════════════════════

def fit_pf_model(
    yields, maturities, dates,
    model='DNS',
    n_particles=2000,
    window_size=None,
    n_starts_kf=4,
    r_mode='libre',
    tau1_grid=None, tau2_grid=None,
    sep_min=0.30,
    ess_threshold=0.5,
    innovation_dist='gaussian',
    student_t_df=5.0,
    q_ltau_scale=1.0,
    seed=42,
    kf_warmstart=None,
    force_refit_kf=False,
    cache_dir=CACHE_DIR_DEFAULT,
    show_progress=True,
    verbose=True,
):
    """
    Ajuste DNS/DNSS vía Particle Filter con tau_t dinámico. Función de alto
    nivel, análoga en espíritu a `mle_estimate` + `kalman_filter` para el
    KF, o a `run_chains_parallel` para el Gibbs sampler.

    Orquesta: (1) warm-start del KF —con cache en disco para experimentación
    rápida—, (2) calibración AR(1) de log-tau, (3) ejecución del PF.

    Parámetros
    ----------
    yields, maturities, dates : igual convención que el resto del paquete
    model            : 'DNS' | 'DNSS'
    n_particles      : número de partículas N (ver docstring de
                       run_particle_filter)
    window_size      : ventana del grid search de warm-start (None=expandiente)
    n_starts_kf      : multi-start del MLE de warm-start
    r_mode           : 'libre' | 'grupos' — pasado al MLE de warm-start
    tau1_grid, tau2_grid : grids de warm-start (misma semántica que
                       grid_search.rolling_tau_dns/dnss)
    sep_min          : separación mínima tau1 > tau2 + sep_min (DNSS),
                       aplicada partícula-por-partícula en CADA periodo
    ess_threshold    : fracción de N bajo la cual se dispara el remuestreo
    innovation_dist  : 'gaussian' | 'student_t' — relajar la gaussianidad
                       de la ecuación de medición (colas pesadas)
    student_t_df     : grados de libertad si innovation_dist=='student_t'
    q_ltau_scale     : factor multiplicativo sobre la varianza de innovación
                       de log-tau calibrada por OLS. scale->0 colapsa tau_t
                       a casi-fijo (chequeo de consistencia contra el KF);
                       scale>1 permite más movimiento del sugerido por la
                       serie del grid search (que tiende a subestimar la
                       verdadera variabilidad — ver calibrate_tau_ar1).
    seed             : semilla del PF (CRÍTICO — varianza de Monte Carlo
                       finita en N, a diferencia del KF determinista)
    kf_warmstart     : dict opcional con un resultado ya calculado de
                       fit_model_kf_cached / un KF ya corrido — evita
                       recalcular el warm-start.
    force_refit_kf   : ignora el cache del warm-start y reestima de cero.
    cache_dir        : carpeta del cache de warm-start.
    show_progress    : barra de progreso del PF (ver run_particle_filter).
    verbose          : imprime progreso de cada etapa.

    Retorna
    -------
    dict con dos llaves: 'kf' (resultado del warm-start) y 'pf' (resultado
    del Particle Filter) — cada uno con esquema de claves compatible con
    report.generate_full_report().
    """
    yields = np.asarray(yields, dtype=float)
    maturities = np.asarray(maturities, dtype=float)
    assert model in ('DNS', 'DNSS')

    if tau1_grid is None:
        tau1_grid = np.linspace(0.3, 5.0, 35)
    if tau2_grid is None and model == 'DNSS':
        tau2_grid = np.linspace(0.15, 3.0, 20)

    # ── 1. Warm-start del KF (con cache) ─────────────────────────────────────
    if kf_warmstart is not None:
        res_kf = kf_warmstart
        if verbose:
            print("[1/3] Usando KF warm-start provisto por el usuario")
    else:
        if verbose:
            print("[1/3] Obteniendo KF warm-start (cache si está disponible)...")
        res_kf = fit_model_kf_cached(
            yields, maturities, dates, model=model,
            tau1_grid=tau1_grid, tau2_grid=tau2_grid, window_size=window_size,
            n_starts=n_starts_kf, r_mode=r_mode, sep_min=sep_min,
            verbose=verbose, force_refit=force_refit_kf, cache_dir=cache_dir)

    # ── 2. Calibración AR(1) de log-tau ──────────────────────────────────────
    if verbose:
        print("[2/3] Calibrando dinámica AR(1) de tau...")
    mu_lt1, phi_lt1, q_lt1 = calibrate_tau_ar1(res_kf['tau1_t'])
    mu_ltau = [mu_lt1]; phi_ltau = [phi_lt1]; q_ltau = [q_lt1 * q_ltau_scale]
    if model == 'DNSS':
        mu_lt2, phi_lt2, q_lt2 = calibrate_tau_ar1(res_kf['tau2_t'])
        mu_ltau.append(mu_lt2); phi_ltau.append(phi_lt2); q_ltau.append(q_lt2 * q_ltau_scale)
    if verbose:
        print(f"    tau1: mu={np.exp(mu_lt1):.3f}  phi={phi_lt1:.4f}  sqrt(q)={np.sqrt(q_lt1*q_ltau_scale):.4f}")
        if model == 'DNSS':
            print(f"    tau2: mu={np.exp(mu_lt2):.3f}  phi={phi_lt2:.4f}  sqrt(q)={np.sqrt(q_lt2*q_ltau_scale):.4f}")

    # ── 3. Particle Filter sobre el estado aumentado ─────────────────────────
    if verbose:
        print(f"[3/3] Particle Filter ({model}, N={n_particles} partículas)...")
    params_pf = dict(mu_beta=res_kf['mu'], F_beta_diag=np.diag(res_kf['F']),
                     Q_beta_diag=np.diag(res_kf['Q']),
                     mu_ltau=np.array(mu_ltau), phi_ltau=np.array(phi_ltau),
                     q_ltau=np.array(q_ltau), R_diag=res_kf['R_diag'])
    config_pf = dict(seed=seed, ess_threshold=ess_threshold, sep_min=sep_min,
                     innovation_dist=innovation_dist, student_t_df=student_t_df)

    t0 = time.time()
    res_pf = run_particle_filter(yields, maturities, model, n_particles,
                                 params_pf, config_pf, show_progress=show_progress)
    res_pf['dates'] = dates
    if verbose:
        print(f"    RMSE PF={res_pf['rmse_total']*100:.4f}pb  log-lik={res_pf['loglik']:,.2f}"
              f"  AIC={res_pf['aic']:,.1f}  BIC={res_pf['bic']:,.1f}"
              f"  ESS/N medio={res_pf['ess_t'].mean()/n_particles:.3f}"
              f"  remuestreo={res_pf['resampled_t'].mean()*100:.1f}%  ({time.time()-t0:.2f}s)")

    return dict(kf=res_kf, pf=res_pf)
