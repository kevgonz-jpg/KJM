"""
gibbs.py
========
Gibbs sampler bayesiano para DNS/DNSS con:
  - FFBS (Carter & Kohn 1994) para β₁:T
  - Posteriors conjugadas para F, μ, Q, R
  - MH para τ (modo estático o AR(1))
  - Ventana rodante (run_rolling_bayes)
  - Multi-cadena paralela

Funciones exportadas
--------------------
ffbs(mu, f_vec, q_vec, r_vec, H, yields)
kf_loglik(mu, f_vec, q_vec, r_vec, H, yields)
sample_F_mu_diagonal(...)
sample_Q(...)
sample_R(...)  /  sample_R_groups(...)
sample_tau_MH_dns(...)  /  sample_tau_MH_dnss(...)
make_priors(...)  /  make_priors_ar1(...)
initialize_chain(...)
run_chain(...)
run_chains_parallel(...)
run_rolling_bayes(...)

Referencias
-----------
Carter & Kohn (1994)
Caldeira, Laurini & Portugal (2010)
Çakmaklı (2013)
Roberts & Rosenthal (2001)  — tasa de aceptación MH
"""

import time
import numpy as np
from scipy.stats import invgamma, truncnorm
from joblib import Parallel, delayed

from .nelson_siegel import ns_loadings, nss_loadings, get_H, ols_init_tau

# ── Numba opcional ─────────────────────────────────────────────────────────────
try:
    from numba import njit as _njit
    _NUMBA = True
except ImportError:
    _NUMBA = False
    def _njit(**kw):
        return lambda f: f


# ═══════════════════════════════════════════════════════════════════════════════
# FFBS — Forward Filter Backward Sampler  (Carter & Kohn 1994)
# ═══════════════════════════════════════════════════════════════════════════════

if _NUMBA:
    from numba import njit as _njit_real

    @_njit_real(cache=True)
    def _ffbs_diag_numba(mu, f_vec, q_vec, r_vec, H, yields):
        """FFBS con F diagonal. Compilado JIT. Válido para k=3 (DNS) y k=4 (DNSS)."""
        T_obs, N_obs = yields.shape
        k = len(mu)
        beta_filt = np.zeros((T_obs, k))
        P_filt    = np.zeros((T_obs, k, k))
        Ik        = np.eye(k)

        beta_hat = mu.copy()
        P = np.zeros((k, k))
        for i in range(k):
            P[i, i] = q_vec[i] / max(1.0 - f_vec[i] ** 2, 1e-6)

        # ── FORWARD ──────────────────────────────────────────────────────────
        for t in range(T_obs):
            beta_p = np.zeros(k)
            for i in range(k):
                beta_p[i] = mu[i] + f_vec[i] * (beta_hat[i] - mu[i])

            P_p = np.zeros((k, k))
            for i in range(k):
                for j in range(k):
                    P_p[i, j] = f_vec[i] * P[i, j] * f_vec[j]
            for i in range(k):
                P_p[i, i] += q_vec[i]
            P_p = 0.5 * (P_p + P_p.T)

            v  = yields[t] - H @ beta_p
            HP = H @ P_p
            S  = HP @ H.T
            for n in range(N_obs):
                S[n, n] += r_vec[n]
            S = 0.5 * (S + S.T)

            K        = np.linalg.solve(S, HP).T
            beta_hat = beta_p + K @ v

            Rmat = np.zeros((N_obs, N_obs))
            for n in range(N_obs):
                Rmat[n, n] = r_vec[n]
            IKH = Ik - K @ H
            P   = IKH @ P_p @ IKH.T + K @ Rmat @ K.T
            P   = 0.5 * (P + P.T)
            beta_filt[t] = beta_hat
            P_filt[t]    = P.copy()

        # ── BACKWARD ──────────────────────────────────────────────────────────
        betas = np.zeros((T_obs, k))
        PT    = P_filt[T_obs - 1] + 1e-10 * Ik
        L     = np.linalg.cholesky(PT)
        betas[T_obs - 1] = beta_filt[T_obs - 1] + L @ np.random.randn(k)

        for t in range(T_obs - 2, -1, -1):
            Pt  = P_filt[t]
            bft = beta_filt[t]
            Pt1p = np.zeros((k, k))
            for i in range(k):
                for j in range(k):
                    Pt1p[i, j] = f_vec[i] * Pt[i, j] * f_vec[j]
            for i in range(k):
                Pt1p[i, i] += q_vec[i]
            Pt1p = 0.5 * (Pt1p + Pt1p.T)

            PtFT = np.zeros((k, k))
            for i in range(k):
                for j in range(k):
                    PtFT[i, j] = Pt[i, j] * f_vec[j]
            Gt = np.linalg.solve(Pt1p.T, PtFT.T).T

            b_pred = np.zeros(k)
            for i in range(k):
                b_pred[i] = mu[i] + f_vec[i] * (bft[i] - mu[i])

            mu_s = bft + Gt @ (betas[t + 1] - b_pred)
            Ps   = Pt - Gt @ Pt1p @ Gt.T
            Ps   = 0.5 * (Ps + Ps.T) + 1e-10 * Ik

            L        = np.linalg.cholesky(Ps)
            betas[t] = mu_s + L @ np.random.randn(k)

        return betas

    def _ffbs_python(mu, f_vec, q_vec, r_vec, H, yields):
        """FFBS Python puro (fallback sin Numba)."""
        T_obs, N_obs = yields.shape
        k = len(mu)
        Ik = np.eye(k)
        F_mat = np.diag(f_vec); Q_mat = np.diag(q_vec); R_mat = np.diag(r_vec)
        beta_filt = np.zeros((T_obs, k))
        P_filt    = np.zeros((T_obs, k, k))
        beta_hat  = mu.copy()
        P         = np.diag(q_vec / np.maximum(1 - f_vec ** 2, 1e-6))

        for t in range(T_obs):
            beta_p = mu + F_mat @ (beta_hat - mu)
            P_p    = F_mat @ P @ F_mat.T + Q_mat
            P_p    = 0.5 * (P_p + P_p.T)
            v  = yields[t] - H @ beta_p
            HP = H @ P_p
            S  = HP @ H.T + R_mat
            K  = np.linalg.solve(S, HP).T
            beta_hat = beta_p + K @ v
            IKH = Ik - K @ H
            P   = IKH @ P_p @ IKH.T + K @ R_mat @ K.T
            P   = 0.5 * (P + P.T)
            beta_filt[t] = beta_hat; P_filt[t] = P

        betas = np.zeros((T_obs, k))
        betas[-1] = beta_filt[-1] + np.linalg.cholesky(P_filt[-1] + 1e-10 * Ik) @ np.random.randn(k)

        for t in range(T_obs - 2, -1, -1):
            Pt   = P_filt[t]; bft = beta_filt[t]
            Pt1p = F_mat @ Pt @ F_mat.T + Q_mat
            Pt1p = 0.5 * (Pt1p + Pt1p.T)
            Gt   = np.linalg.solve(Pt1p.T, (Pt @ F_mat.T).T).T
            b_pred = mu + F_mat @ (bft - mu)
            mu_s = bft + Gt @ (betas[t + 1] - b_pred)
            Ps   = Pt - Gt @ Pt1p @ Gt.T
            Ps   = 0.5 * (Ps + Ps.T) + 1e-10 * Ik
            betas[t] = mu_s + np.linalg.cholesky(Ps) @ np.random.randn(k)

        return betas

    def ffbs(mu, f_vec, q_vec, r_vec, H, yields):
        """Wrapper público FFBS — despacha a Numba o Python."""
        return _ffbs_diag_numba(
            np.asarray(mu,    np.float64), np.asarray(f_vec, np.float64),
            np.asarray(q_vec, np.float64), np.asarray(r_vec, np.float64),
            np.asarray(H,     np.float64), np.asarray(yields, np.float64))

else:
    def ffbs(mu, f_vec, q_vec, r_vec, H, yields):
        return _ffbs_python(
            np.asarray(mu,    np.float64), np.asarray(f_vec, np.float64),
            np.asarray(q_vec, np.float64), np.asarray(r_vec, np.float64),
            np.asarray(H,     np.float64), np.asarray(yields, np.float64))

    def _ffbs_python(mu, f_vec, q_vec, r_vec, H, yields):
        T_obs, N_obs = yields.shape
        k = len(mu)
        Ik = np.eye(k)
        F_mat = np.diag(f_vec); Q_mat = np.diag(q_vec); R_mat = np.diag(r_vec)
        beta_filt = np.zeros((T_obs, k))
        P_filt    = np.zeros((T_obs, k, k))
        beta_hat  = mu.copy()
        P         = np.diag(q_vec / np.maximum(1 - f_vec ** 2, 1e-6))

        for t in range(T_obs):
            beta_p = mu + F_mat @ (beta_hat - mu)
            P_p    = F_mat @ P @ F_mat.T + Q_mat
            P_p    = 0.5 * (P_p + P_p.T)
            v  = yields[t] - H @ beta_p
            S  = H @ P_p @ H.T + R_mat
            K  = np.linalg.solve(S, H @ P_p).T
            beta_hat = beta_p + K @ v
            IKH = Ik - K @ H
            P   = IKH @ P_p @ IKH.T + K @ R_mat @ K.T
            P   = 0.5 * (P + P.T)
            beta_filt[t] = beta_hat; P_filt[t] = P

        betas = np.zeros((T_obs, k))
        betas[-1] = beta_filt[-1] + np.linalg.cholesky(P_filt[-1] + 1e-10 * Ik) @ np.random.randn(k)

        for t in range(T_obs - 2, -1, -1):
            Pt   = P_filt[t]; bft = beta_filt[t]
            Pt1p = F_mat @ Pt @ F_mat.T + Q_mat
            Pt1p = 0.5 * (Pt1p + Pt1p.T)
            Gt   = np.linalg.solve(Pt1p.T, (Pt @ F_mat.T).T).T
            b_pred = mu + F_mat @ (bft - mu)
            mu_s = bft + Gt @ (betas[t + 1] - b_pred)
            Ps   = Pt - Gt @ Pt1p @ Gt.T
            Ps   = 0.5 * (Ps + Ps.T) + 1e-10 * Ik
            betas[t] = mu_s + np.linalg.cholesky(Ps) @ np.random.randn(k)

        return betas


# ═══════════════════════════════════════════════════════════════════════════════
# LOG-VEROSIMILITUD MARGINAL KF (para MH de τ)
# ═══════════════════════════════════════════════════════════════════════════════

def kf_loglik(mu, f_vec, q_vec, r_vec, H, yields) -> float:
    """
    Log-verosimilitud marginal p(Y|τ, θ) vía KF con F diagonal.
    Integra β₁:T analíticamente — necesaria para el paso MH de τ.
    """
    T_obs, N_obs = yields.shape
    k = len(mu)
    LOG2PI   = np.log(2.0 * np.pi)
    F_mat    = np.diag(f_vec); Q_mat = np.diag(q_vec); R_mat = np.diag(r_vec)
    beta_hat = mu.copy()
    P        = np.diag(q_vec / np.maximum(1 - f_vec ** 2, 1e-6))
    Ik       = np.eye(k)
    log_lik  = 0.0

    for t in range(T_obs):
        beta_p = mu + F_mat @ (beta_hat - mu)
        P_p    = F_mat @ P @ F_mat.T + Q_mat
        P_p    = 0.5 * (P_p + P_p.T)
        v      = yields[t] - H @ beta_p
        S      = H @ P_p @ H.T + R_mat
        S      = 0.5 * (S + S.T)

        L       = np.linalg.cholesky(S + 1e-10 * np.eye(N_obs))
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        z       = np.linalg.solve(L, v)
        quad    = z @ z
        log_lik += -0.5 * (N_obs * LOG2PI + log_det + quad)

        K        = np.linalg.solve(S, H @ P_p).T
        beta_hat = beta_p + K @ v
        IKH      = Ik - K @ H
        P        = IKH @ P_p @ IKH.T + K @ R_mat @ K.T
        P        = 0.5 * (P + P.T)

    return float(log_lik)


# ═══════════════════════════════════════════════════════════════════════════════
# POSTERIORS CONJUGADAS
# ═══════════════════════════════════════════════════════════════════════════════

def sample_F_mu_diagonal(betas, f_vec, q_vec,
                          prior_f_mean, prior_f_var,
                          prior_mu_mean, prior_mu_var):
    """
    Muestrea F (diagonal) y μ de sus posteriors conjugadas.
    Secuencial por factor i: μᵢ | fᵢ → fᵢ | μᵢ.
    """
    k      = len(f_vec)
    f_new  = f_vec.copy()
    mu_new = np.zeros(k)
    bt     = betas[1:, :]
    bt_lag = betas[:-1, :]
    T_eff  = bt.shape[0]

    for i in range(k):
        qi = q_vec[i]; fi = f_new[i]

        # Posterior de μᵢ | fᵢ
        lhs    = bt[:, i] - fi * bt_lag[:, i]
        Xi     = 1.0 - fi
        prec_m = 1.0 / prior_mu_var[i] + Xi ** 2 * T_eff / qi
        mean_m = (prior_mu_mean[i] / prior_mu_var[i] + Xi / qi * lhs.sum()) / prec_m
        var_m  = 1.0 / prec_m
        mu_new[i] = np.random.normal(mean_m, np.sqrt(max(var_m, 1e-15)))

        # Posterior de fᵢ | μᵢ_new  (Normal truncada en (-1, 1))
        mui    = mu_new[i]
        ai_t   = bt[:, i]    - mui
        ai_lag = bt_lag[:, i] - mui
        s1     = np.sum(ai_lag ** 2)
        s2     = np.sum(ai_t * ai_lag)
        prec_f = 1.0 / prior_f_var[i] + s1 / qi
        mean_f = (prior_f_mean[i] / prior_f_var[i] + s2 / qi) / prec_f
        var_f  = 1.0 / prec_f; std_f = np.sqrt(max(var_f, 1e-15))
        a_tr   = (-1.0 - mean_f) / std_f
        b_tr   = ( 1.0 - mean_f) / std_f
        f_new[i] = truncnorm.rvs(a_tr, b_tr, loc=mean_f, scale=std_f)

    return f_new, mu_new


def sample_Q(betas, mu, f_vec, prior_a, prior_b):
    """Muestrea Q diagonal de Inverse-Gamma conjugada."""
    k     = len(f_vec)
    q_new = np.zeros(k)
    bt    = betas[1:, :]; bt_lag = betas[:-1, :]
    T_eff = bt.shape[0]
    for i in range(k):
        eta      = bt[:, i] - mu[i] - f_vec[i] * (bt_lag[:, i] - mu[i])
        a_post   = prior_a + T_eff / 2.0
        b_post   = prior_b + np.sum(eta ** 2) / 2.0
        q_new[i] = invgamma.rvs(a_post, scale=b_post)
    return q_new


def sample_R(yields, betas, H, prior_a, prior_b, obs_weights=None):
    """
    Muestrea R diagonal de Inverse-Gamma conjugada.
    obs_weights: (N,) reduce prior_b efectivo para plazos con mayor peso (Opción C).
    """
    T_obs, N_obs = yields.shape
    r_new  = np.zeros(N_obs)
    fitted = (H @ betas.T).T
    resid  = yields - fitted

    for n in range(N_obs):
        sse    = np.sum(resid[:, n] ** 2)
        a_post = prior_a + T_obs / 2.0
        w_n    = obs_weights[n] if obs_weights is not None else 1.0
        b_post = prior_b / max(w_n, 1e-6) + sse / 2.0
        r_new[n] = invgamma.rvs(a_post, scale=b_post)
    return r_new


def sample_R_groups(yields, betas, H, prior_a, prior_b,
                    maturities, group_thresholds=(3.0, 10.0)):
    """
    Opción A: Muestrea R con estructura por grupos (3 parámetros).
    corto ≤ th1, medio ≤ th2, largo > th2.
    """
    T_obs, N_obs = yields.shape
    th1, th2     = group_thresholds
    fitted       = (H @ betas.T).T
    resid        = yields - fitted
    r_new        = np.zeros(N_obs)

    for lo, hi in [(0, th1), (th1, th2), (th2, np.inf)]:
        mask = (maturities > lo) & (maturities <= hi)
        if not mask.any():
            continue
        sse_g  = np.sum(resid[:, mask] ** 2)
        ng     = mask.sum()
        a_post = prior_a + T_obs * ng / 2.0
        b_post = prior_b + sse_g / 2.0
        r_new[mask] = invgamma.rvs(a_post, scale=b_post)

    return r_new


# ═══════════════════════════════════════════════════════════════════════════════
# METROPOLIS-HASTINGS PARA τ
# ═══════════════════════════════════════════════════════════════════════════════

def _log_prior_tau(tau1, tau2, prior_tau):
    """Evalúa log-prior de τ (uniforme o normal)."""
    kind = prior_tau[0]
    if kind == 'uniforme':
        lo, hi = prior_tau[1], prior_tau[2]
        in1 = lo < tau1 < hi
        in2 = (tau2 is None) or (lo < tau2 < hi)
        return 0.0 if (in1 and in2) else -np.inf
    elif kind == 'normal':
        mu1, sd1 = prior_tau[1], prior_tau[2]
        lp = -0.5 * ((tau1 - mu1) / sd1) ** 2
        if tau2 is not None:
            mu2, sd2 = prior_tau[3], prior_tau[4]
            lp += -0.5 * ((tau2 - mu2) / sd2) ** 2
        return lp
    return 0.0


def sample_tau_MH_dns(tau1, yields, mu, f_vec, q_vec, r_vec,
                       maturities, prop_std, tau_bounds, prior_tau):
    """MH 1D para τ₁ (DNS) en log-escala. Retorna (tau1_new, loglik_new, accepted)."""
    lo, hi = tau_bounds
    H_curr       = ns_loadings(maturities, tau1)
    loglik_curr  = kf_loglik(mu, f_vec, q_vec, r_vec, H_curr, yields)
    lp_curr      = _log_prior_tau(tau1, None, prior_tau)

    tau1_p = np.exp(np.log(tau1) + prop_std * np.random.randn())

    if not (lo < tau1_p < hi):
        return tau1, loglik_curr, False

    H_prop      = ns_loadings(maturities, tau1_p)
    loglik_prop = kf_loglik(mu, f_vec, q_vec, r_vec, H_prop, yields)
    lp_prop     = _log_prior_tau(tau1_p, None, prior_tau)
    log_alpha   = (loglik_prop - loglik_curr) + (lp_prop - lp_curr)

    if np.isfinite(log_alpha) and np.log(np.random.rand()) < log_alpha:
        return tau1_p, loglik_prop, True
    return tau1, loglik_curr, False


def sample_tau_MH_dnss(tau1, tau2, yields, mu, f_vec, q_vec, r_vec,
                        maturities, prop_std, tau_bounds, prior_tau, sep_min=0.15):
    """MH 2D para (τ₁, τ₂) (DNSS). Retorna (tau1, tau2, loglik, accepted)."""
    lo, hi = tau_bounds
    H_curr       = nss_loadings(maturities, tau1, tau2)
    loglik_curr  = kf_loglik(mu, f_vec, q_vec, r_vec, H_curr, yields)
    lp_curr      = _log_prior_tau(tau1, tau2, prior_tau)

    tau1_p = np.exp(np.log(tau1) + prop_std[0] * np.random.randn())
    tau2_p = np.exp(np.log(tau2) + prop_std[1] * np.random.randn())

    if not (lo < tau1_p < hi) or not (lo < tau2_p < hi) \
       or abs(tau1_p - tau2_p) < sep_min:
        return tau1, tau2, loglik_curr, False

    H_prop      = nss_loadings(maturities, tau1_p, tau2_p)
    loglik_prop = kf_loglik(mu, f_vec, q_vec, r_vec, H_prop, yields)
    lp_prop     = _log_prior_tau(tau1_p, tau2_p, prior_tau)
    log_alpha   = (loglik_prop - loglik_curr) + (lp_prop - lp_curr)

    if np.isfinite(log_alpha) and np.log(np.random.rand()) < log_alpha:
        return tau1_p, tau2_p, loglik_prop, True
    return tau1, tau2, loglik_curr, False


# ═══════════════════════════════════════════════════════════════════════════════
# PRIORS
# ═══════════════════════════════════════════════════════════════════════════════

def make_priors(model='DNS', prior_tau_type='uniforme',
                tau1_mu=1.78, tau1_sd=0.40,
                tau2_mu=0.60, tau2_sd=0.08,
                tau_lo=0.15,  tau_hi=5.0):
    """
    Construye el diccionario de priors para el Gibbs sampler.

    prior_tau_type : 'uniforme' → prior plano en (tau_lo, tau_hi)
                     'normal'   → Normal informativo sobre τ
    """
    k = 3 if model == 'DNS' else 4

    prior_f_mean  = np.array([0.90, 0.90, 0.90, 0.90])[:k]
    prior_f_var   = np.array([0.10, 0.10, 0.15, 0.15])[:k] ** 2
    prior_mu_mean = np.array([5.0, -2.0, 0.0, 0.0])[:k]
    prior_mu_var  = np.full(k, 25.0)

    if prior_tau_type == 'uniforme':
        prior_tau = ('uniforme', tau_lo, tau_hi)
    else:
        if model == 'DNS':
            prior_tau = ('normal', tau1_mu, tau1_sd, None, None)
        else:
            prior_tau = ('normal', tau1_mu, tau1_sd, tau2_mu, tau2_sd)

    return {
        'prior_f_mean' : prior_f_mean,
        'prior_f_var'  : prior_f_var,
        'prior_mu_mean': prior_mu_mean,
        'prior_mu_var' : prior_mu_var,
        'prior_Q_a'    : 0.01,
        'prior_Q_b'    : 0.01,
        'prior_R_a'    : 0.01,
        'prior_R_b'    : 0.01,
        'prior_tau'    : prior_tau,
        'tau_bounds'   : (tau_lo, tau_hi),
        'model'        : model,
        'k'            : k,
    }


def make_priors_ar1(model, tau1_init, tau2_init=None,
                    phi_x_mean=0.95, phi_x_sd=0.05,
                    sigma_x_a=2.0,   sigma_x_b=0.01,
                    **kwargs):
    """Extiende make_priors() con hiperparámetros para el AR(1) de τ."""
    priors = make_priors(model, **kwargs)
    priors['tau_ar1'] = {
        'mu_x1'      : np.log(tau1_init),
        'mu_x2'      : np.log(tau2_init) if tau2_init else None,
        'phi_x_mean' : phi_x_mean,
        'phi_x_var'  : phi_x_sd ** 2,
        'sigma_x_a'  : sigma_x_a,
        'sigma_x_b'  : sigma_x_b,
    }
    return priors


# ═══════════════════════════════════════════════════════════════════════════════
# INICIALIZACIÓN Y CADENA
# ═══════════════════════════════════════════════════════════════════════════════

def initialize_chain(yields, maturities, model, tau1_init, tau2_init, seed):
    """Inicializa vía OLS two-step (Diebold-Li)."""
    np.random.seed(seed)
    H  = get_H(maturities, tau1_init, tau2_init, model)
    k  = H.shape[1]
    B  = np.linalg.lstsq(H, yields.T, rcond=None)[0].T   # (T, k)

    mu0 = B.mean(axis=0)
    b_lag = B[:-1]; b_curr = B[1:]
    f0  = np.clip(
        [np.corrcoef(b_lag[:, i], b_curr[:, i])[0, 1] for i in range(k)],
        -0.95, 0.95)
    f0 = np.array(f0)

    eta = b_curr - mu0 - f0 * (b_lag - mu0)
    q0  = np.var(eta, axis=0, ddof=1).clip(1e-6)
    eps = yields - (H @ B.T).T
    r0  = np.var(eps, axis=0, ddof=1).clip(1e-6)

    return {'mu': mu0, 'f_vec': f0, 'q_vec': q0, 'r_vec': r0,
            'tau1': tau1_init, 'tau2': tau2_init, 'betas': B.copy()}


def run_chain(yields, maturities, priors, n_iter, n_burnin, seed,
              tau1_init=None, tau2_init=None,
              prop_std_init=None, adapt_mh=True,
              obs_weights=None, r_mode='libre',
              group_thresholds=(3.0, 10.0),
              tau_mode='estatico',
              tau_ar1_prop_std=0.05,
              verbose=False):
    """
    Gibbs sampler DNS/DNSS — una cadena.

    tau_mode : 'estatico'  — τ global via MH
               'ar1'       — τ_t AR(1) latente, MH por periodo

    Retorna dict con muestras post burn-in.
    """
    np.random.seed(seed)
    model    = priors['model']
    k        = priors['k']
    T_obs, N = yields.shape

    if tau1_init is None:
        tau1_init, tau2_init = ols_init_tau(yields, maturities, model)
    elif tau2_init is None and model == 'DNSS':
        _, tau2_init = ols_init_tau(yields, maturities, model)

    state  = initialize_chain(yields, maturities, model,
                              tau1_init,
                              tau2_init if model == 'DNSS' else 1.0, seed)
    mu     = state['mu'].copy()
    f_vec  = state['f_vec'].copy()
    q_vec  = state['q_vec'].copy()
    r_vec  = state['r_vec'].copy()
    tau1   = state['tau1']
    tau2   = state['tau2'] if model == 'DNSS' else None
    betas  = state['betas'].copy()

    # ── Storage ───────────────────────────────────────────────────────────────
    n_store = n_iter - n_burnin
    store = {
        'mu'   : np.zeros((n_store, k)),
        'f_vec': np.zeros((n_store, k)),
        'q_vec': np.zeros((n_store, k)),
        'r_vec': np.zeros((n_store, N)),
        'tau1' : np.zeros(n_store),
        'betas': np.zeros((n_store, T_obs, k)),
    }
    if model == 'DNSS':
        store['tau2'] = np.zeros(n_store)
    if tau_mode == 'ar1':
        store['tau1_seq']  = np.zeros((n_store, T_obs))
        store['phi_x1']    = np.zeros(n_store)
        store['sigma_x1']  = np.zeros(n_store)
        if model == 'DNSS':
            store['tau2_seq'] = np.zeros((n_store, T_obs))

    # ── MH prop_std y AR(1) ────────────────────────────────────────────────────
    if prop_std_init is None:
        prop_std = np.array([0.08, 0.06]) if model == 'DNSS' else np.array([0.08])
    else:
        prop_std = np.atleast_1d(prop_std_init).copy()

    # Inicialización AR(1)
    if tau_mode == 'ar1':
        ar1_cfg = priors.get('tau_ar1', {})
        mu_x1   = ar1_cfg.get('mu_x1', np.log(tau1))
        phi_x1  = ar1_cfg.get('phi_x_mean', 0.95)
        sig_x1  = np.sqrt(ar1_cfg.get('sigma_x_b', 0.01) /
                          max(ar1_cfg.get('sigma_x_a', 2.0) - 1, 0.1))
        x_seq   = np.full((T_obs, 1), np.log(tau1))
        n_tau   = 1
        mu_x    = np.array([mu_x1])
        phi_x   = np.array([phi_x1])
        sigma_x = np.array([sig_x1])
        if model == 'DNSS':
            mu_x2  = ar1_cfg.get('mu_x2', np.log(tau2))
            x_seq  = np.column_stack([x_seq, np.full((T_obs, 1), np.log(tau2))])
            n_tau  = 2
            mu_x   = np.append(mu_x, mu_x2)
            phi_x  = np.append(phi_x, phi_x1)
            sigma_x = np.append(sigma_x, sig_x1)
        phi_x_mean = ar1_cfg.get('phi_x_mean', 0.95)
        phi_x_var  = ar1_cfg.get('phi_x_var',  0.05 ** 2)
        sigma_x_a  = ar1_cfg.get('sigma_x_a',  2.0)
        sigma_x_b  = ar1_cfg.get('sigma_x_b',  0.01)

    n_mh_acc = 0; n_mh_tot = 0
    adapt_win = max(1, n_burnin // 5)
    t0_time   = time.time()

    for s in range(n_iter):
        # ── Determinar H ──────────────────────────────────────────────────────
        if tau_mode == 'ar1':
            tau1_med = float(np.median(np.exp(x_seq[:, 0])))
            tau2_med = float(np.median(np.exp(x_seq[:, 1]))) if model == 'DNSS' else None
        else:
            tau1_med = tau1; tau2_med = tau2

        H = get_H(maturities, tau1_med,
                  tau2_med if model == 'DNSS' else None, model)

        # ── Bloque 1: FFBS ────────────────────────────────────────────────────
        betas = ffbs(mu, f_vec, q_vec, r_vec, H, yields)

        # ── Bloque 2: F, μ ────────────────────────────────────────────────────
        f_vec, mu = sample_F_mu_diagonal(
            betas, f_vec, q_vec,
            priors['prior_f_mean'], priors['prior_f_var'],
            priors['prior_mu_mean'], priors['prior_mu_var'])

        # ── Bloque 3: Q ───────────────────────────────────────────────────────
        q_vec = sample_Q(betas, mu, f_vec, priors['prior_Q_a'], priors['prior_Q_b'])

        # ── Bloque 4: R ───────────────────────────────────────────────────────
        if r_mode == 'grupos':
            r_vec = sample_R_groups(yields, betas, H, priors['prior_R_a'],
                                    priors['prior_R_b'], maturities, group_thresholds)
        else:
            r_vec = sample_R(yields, betas, H, priors['prior_R_a'],
                             priors['prior_R_b'], obs_weights)

        # ── Bloque 5: τ ───────────────────────────────────────────────────────
        if tau_mode == 'estatico':
            if model == 'DNS':
                tau1, _, accepted = sample_tau_MH_dns(
                    tau1, yields, mu, f_vec, q_vec, r_vec,
                    maturities, prop_std[0], priors['tau_bounds'], priors['prior_tau'])
            else:
                tau1, tau2, _, accepted = sample_tau_MH_dnss(
                    tau1, tau2, yields, mu, f_vec, q_vec, r_vec,
                    maturities, prop_std, priors['tau_bounds'], priors['prior_tau'])
            n_mh_tot += 1; n_mh_acc += int(accepted)

        elif tau_mode == 'ar1':
            x_seq, acc_t = _sample_tau_ar1_MH(
                x_seq, betas, yields, r_vec, model, maturities,
                phi_x, mu_x, sigma_x, tau_ar1_prop_std)
            n_mh_acc += acc_t; n_mh_tot += 1; accepted = acc_t > 0

            phi_x   = _sample_phi_x(x_seq, mu_x, sigma_x, phi_x_mean, phi_x_var)
            sigma_x = _sample_sigma_x(x_seq, mu_x, phi_x, sigma_x_a, sigma_x_b)
            tau1 = float(np.exp(x_seq[:, 0]).mean())
            if model == 'DNSS':
                tau2 = float(np.exp(x_seq[:, 1]).mean())

        # ── Adaptación MH ────────────────────────────────────────────────────
        if adapt_mh and tau_mode == 'estatico' and s < n_burnin \
           and (s + 1) % adapt_win == 0:
            rate   = n_mh_acc / max(n_mh_tot, 1)
            target = 0.40 if model == 'DNS' else 0.28
            if   rate < target - 0.12: prop_std *= 0.70
            elif rate > target + 0.15: prop_std *= 1.35
            prop_std = np.clip(prop_std, 0.005, 0.60)
            n_mh_acc = 0; n_mh_tot = 0

        # ── Guardar post burn-in ───────────────────────────────────────────────
        if s >= n_burnin:
            idx = s - n_burnin
            store['mu'][idx]    = mu
            store['f_vec'][idx] = f_vec
            store['q_vec'][idx] = q_vec
            store['r_vec'][idx] = r_vec
            store['tau1'][idx]  = tau1
            store['betas'][idx] = betas
            if model == 'DNSS':
                store['tau2'][idx] = tau2
            if tau_mode == 'ar1':
                store['tau1_seq'][idx] = np.exp(x_seq[:, 0])
                store['phi_x1'][idx]   = phi_x[0]
                store['sigma_x1'][idx] = sigma_x[0]
                if model == 'DNSS':
                    store['tau2_seq'][idx] = np.exp(x_seq[:, 1])

        if verbose and (s + 1) % max(1, n_iter // 8) == 0:
            elapsed = time.time() - t0_time
            eta     = (n_iter - s - 1) / max((s + 1) / elapsed, 1e-6)
            rate_s  = n_mh_acc / max(n_mh_tot, 1) * 100
            tau_str = (f"tau1={tau1:.3f}" if model == 'DNS'
                       else f"tau1={tau1:.3f} tau2={tau2:.3f}")
            print(f"  [{s + 1:4d}/{n_iter}] {tau_str}  MH:{rate_s:.0f}%  ETA:{eta:.0f}s")

    store['mh_final_rate']  = n_mh_acc / max(n_mh_tot, 1)
    store['prop_std_final'] = prop_std.copy()
    store['model']          = model
    store['priors']         = priors
    store['tau_mode']       = tau_mode
    return store


# ── AR(1) auxiliares ──────────────────────────────────────────────────────────

def _ar1_loglik_local(x_t, betas_t, yields_t, r_vec, model, maturities):
    """Log-lik local p(y_t | x_t, β_t, r) para un periodo t."""
    if model == 'DNS':
        tau1_t = np.exp(x_t[0])
        H_t    = ns_loadings(maturities, tau1_t)
    else:
        tau1_t = np.exp(x_t[0]); tau2_t = np.exp(x_t[1])
        if abs(tau1_t - tau2_t) < 0.15:
            return -np.inf
        H_t = nss_loadings(maturities, tau1_t, tau2_t)
    resid = yields_t - H_t @ betas_t
    return float(-0.5 * np.sum(resid ** 2 / r_vec + np.log(2 * np.pi * r_vec)))


def _sample_tau_ar1_MH(x_seq, betas, yields, r_vec, model, maturities,
                        phi_x, mu_x, sigma_x, prop_std_t=0.05):
    """MH individual por periodo para x_t = log(τ_t)."""
    T    = len(x_seq)
    n_x  = x_seq.shape[1]
    x_new = x_seq.copy()
    n_acc = 0

    phi_arr = np.atleast_1d(phi_x)   * np.ones(n_x)
    mu_arr  = np.atleast_1d(mu_x)    * np.ones(n_x)
    sig_arr = np.atleast_1d(sigma_x) * np.ones(n_x)

    for t in range(T):
        x_curr = x_new[t].copy()
        x_prop = x_curr + prop_std_t * np.random.randn(n_x)

        ll_curr = _ar1_loglik_local(x_curr, betas[t], yields[t], r_vec, model, maturities)
        ll_prop = _ar1_loglik_local(x_prop, betas[t], yields[t], r_vec, model, maturities)
        if not np.isfinite(ll_prop):
            continue

        def _lp(x_t, t_idx):
            lp = 0.0
            for j in range(n_x):
                if t_idx > 0:
                    eta_t = x_t[j] - (mu_arr[j] + phi_arr[j] * (x_new[t_idx - 1, j] - mu_arr[j]))
                    lp += -0.5 * eta_t ** 2 / sig_arr[j] ** 2
                if t_idx < T - 1:
                    eta_t1 = x_new[t_idx + 1, j] - (mu_arr[j] + phi_arr[j] * (x_t[j] - mu_arr[j]))
                    lp += -0.5 * eta_t1 ** 2 / sig_arr[j] ** 2
            return lp

        log_alpha = (ll_prop - ll_curr) + (_lp(x_prop, t) - _lp(x_curr, t))
        if np.isfinite(log_alpha) and np.log(np.random.rand()) < log_alpha:
            x_new[t] = x_prop
            n_acc    += 1

    return x_new, n_acc / T


def _sample_phi_x(x_seq, mu_x, sigma_x, phi_x_mean, phi_x_var):
    """Muestrea φ_x de Normal truncada conjugada."""
    n_x     = x_seq.shape[1]
    phi_new = np.zeros(n_x)
    mu_arr  = np.atleast_1d(mu_x) * np.ones(n_x)
    sig_arr = np.atleast_1d(sigma_x) * np.ones(n_x)
    for j in range(n_x):
        a_t   = x_seq[1:, j]  - mu_arr[j]
        a_lag = x_seq[:-1, j] - mu_arr[j]
        s2    = sig_arr[j] ** 2
        prec  = 1.0 / phi_x_var + np.sum(a_lag ** 2) / s2
        mean  = (phi_x_mean / phi_x_var + np.sum(a_t * a_lag) / s2) / prec
        std   = 1.0 / np.sqrt(prec)
        a_tr, b_tr = (0.0 - mean) / std, (1.0 - mean) / std
        phi_new[j] = truncnorm.rvs(a_tr, b_tr, loc=mean, scale=std)
    return phi_new


def _sample_sigma_x(x_seq, mu_x, phi_x, sigma_x_a, sigma_x_b):
    """Muestrea σ_x de Inverse-Gamma conjugada."""
    T_eff  = len(x_seq) - 1
    n_x    = x_seq.shape[1]
    sig_new = np.zeros(n_x)
    mu_arr  = np.atleast_1d(mu_x) * np.ones(n_x)
    phi_arr = np.atleast_1d(phi_x) * np.ones(n_x)
    for j in range(n_x):
        eta = x_seq[1:, j] - mu_arr[j] - phi_arr[j] * (x_seq[:-1, j] - mu_arr[j])
        a_post = sigma_x_a + T_eff / 2.0
        b_post = sigma_x_b + np.sum(eta ** 2) / 2.0
        sig_new[j] = np.sqrt(invgamma.rvs(a_post, scale=b_post))
    return sig_new


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-CADENA PARALELA
# ═══════════════════════════════════════════════════════════════════════════════

def run_chains_parallel(yields, maturities, priors, n_iter, n_burnin,
                        n_chains=4, prop_std_init=None,
                        obs_weights=None, r_mode='libre',
                        group_thresholds=(3.0, 10.0),
                        tau_mode='estatico',
                        tau_ar1_prop_std=0.05,
                        n_jobs=-1, verbose=True):
    """
    Lanza n_chains cadenas en paralelo (joblib).
    Cada cadena parte con τ ligeramente perturbado.
    """
    model = priors['model']
    tau1_base, tau2_base = ols_init_tau(yields, maturities, model)
    pertubs  = np.linspace(-0.10, 0.10, n_chains)
    t1_inits = [tau1_base * np.exp(p) for p in pertubs]
    t2_inits = [(tau2_base * np.exp(p * 0.8)) if model == 'DNSS' else None
                for p in pertubs]

    if verbose:
        print(f"Gibbs sampler {model} [{tau_mode}] — {n_chains} cadenas")
        print(f"  n_iter={n_iter} | n_burnin={n_burnin} | "
              f"muestras/cadena={n_iter - n_burnin}")
    t0 = time.time()

    chains = Parallel(n_jobs=min(n_chains, n_jobs) if n_jobs != -1 else n_chains)(
        delayed(run_chain)(
            yields, maturities, priors, n_iter, n_burnin,
            seed=42 + j,
            tau1_init=t1_inits[j], tau2_init=t2_inits[j],
            prop_std_init=prop_std_init, adapt_mh=True,
            obs_weights=obs_weights, r_mode=r_mode,
            group_thresholds=group_thresholds,
            tau_mode=tau_mode,
            tau_ar1_prop_std=tau_ar1_prop_std,
            verbose=False,
        )
        for j in range(n_chains)
    )

    if verbose:
        print(f"\nCompletado en {time.time() - t0:.1f}s")
        for j, ch in enumerate(chains):
            rate_str = f"{ch['mh_final_rate'] * 100:.1f}%"
            print(f"  Cadena {j + 1}: MH={rate_str}  tau1~{ch['tau1'].mean():.3f}")
    return chains


# ═══════════════════════════════════════════════════════════════════════════════
# MODO VENTANA RODANTE
# ═══════════════════════════════════════════════════════════════════════════════

def run_rolling_bayes(yields, maturities, dates, priors,
                      n_iter, n_burnin, window_size, step=None,
                      n_chains=2, verbose=True, **kwargs):
    """
    Gibbs sampler en ventanas rodantes: τ cambia entre ventanas.

    Parámetros
    ----------
    window_size : int  tamaño de cada ventana (periodos)
    step        : int  desplazamiento entre ventanas (None → window_size//2)
    **kwargs    : se pasan a run_chains_parallel

    Retorna
    -------
    list de dicts por ventana con claves 't_start', 't_end',
    'dates_window', 'tau1_pm', 'tau1_q025', 'tau1_q975', etc.
    """
    T = len(yields)
    if step is None:
        step = max(1, window_size // 2)
    model = priors['model']

    starts = list(range(0, T - window_size + 1, step))
    if verbose:
        print(f"Modo ventana: {len(starts)} ventanas | W={window_size} | step={step}")

    resultados = []
    for i, t0 in enumerate(starts):
        t1    = t0 + window_size
        y_win = yields[t0:t1]
        d_win = dates[t0:t1]

        if verbose:
            print(f"  Ventana {i + 1}/{len(starts)}: t=[{t0},{t1})  "
                  f"{str(d_win[0])[:10]}→{str(d_win[-1])[:10]}")

        chains_w = run_chains_parallel(
            y_win, maturities, priors, n_iter, n_burnin,
            n_chains=n_chains, verbose=False, **kwargs)

        tau1_samp = np.concatenate([ch['tau1'] for ch in chains_w])
        tau2_samp = (np.concatenate([ch['tau2'] for ch in chains_w])
                     if model == 'DNSS' else None)

        win_res = {
            'chains'       : chains_w,
            't_start'      : t0,
            't_end'        : t1,
            'dates_window' : d_win,
            'tau1_pm'      : float(tau1_samp.mean()),
            'tau1_q025'    : float(np.percentile(tau1_samp, 2.5)),
            'tau1_q975'    : float(np.percentile(tau1_samp, 97.5)),
            'tau2_pm'      : float(tau2_samp.mean())            if tau2_samp is not None else None,
            'tau2_q025'    : float(np.percentile(tau2_samp, 2.5)) if tau2_samp is not None else None,
            'tau2_q975'    : float(np.percentile(tau2_samp, 97.5)) if tau2_samp is not None else None,
        }
        resultados.append(win_res)

    if verbose:
        print("\nResumen τ₁ por ventana:")
        for r in resultados:
            print(f"  [{r['t_start']:3d},{r['t_end']:3d})  "
                  f"tau1={r['tau1_pm']:.3f} "
                  f"IC95%=[{r['tau1_q025']:.3f},{r['tau1_q975']:.3f}]")
    return resultados
