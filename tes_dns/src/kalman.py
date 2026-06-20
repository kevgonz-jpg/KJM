"""
kalman.py
=========
Filtro de Kalman (con H_t variable), RTS Smoother, y estimación MLE
para los modelos DNS/DNSS frecuentistas con τ adaptativo.

Funciones exportadas
--------------------
kalman_filter(yields, H_arr, F, Q, R_diag, mu, beta0, P0)
rts_smoother(beta_filt, P_filt, beta_pred, P_pred, F)
compute_P0_stationary(F, Q, F_type)
build_H_array(maturities, tau1_t, tau2_t, model)
pack_params / unpack_params   (serialización para scipy.optimize)
neg_loglik(...)               (objetivo negativo para L-BFGS-B)
mle_estimate(...)             (multi-start MLE)
build_R_groups(...)           (Opción A: R estructurada por grupos de plazo)

Referencias
-----------
Hamilton (1994) Cap. 13
Diebold & Li (2006)
Caldeira, Laurini & Portugal (2010)
Çakmaklı (2013)
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve_discrete_lyapunov

from .nelson_siegel import ns_loadings, nss_loadings

# ── Numba opcional ─────────────────────────────────────────────────────────────
try:
    from numba import njit as _njit
    _NUMBA = True
except ImportError:
    _NUMBA = False
    def _njit(**kw):
        return lambda f: f


# ═══════════════════════════════════════════════════════════════════════════════
# FILTRO DE KALMAN
# ═══════════════════════════════════════════════════════════════════════════════

if _NUMBA:
    from numba import njit as _njit_real

    @_njit_real(cache=True)
    def _kf_core_numba(yields, H_arr, F, Q, R_diag, mu, beta0, P0):
        """Kalman Filter con H_t variable. JIT compilado."""
        T_obs, N_obs = yields.shape
        k_dim = F.shape[0]
        Ik     = np.eye(k_dim)
        LOG2PI = np.log(2.0 * np.pi)

        beta_pred_all = np.zeros((T_obs, k_dim))
        beta_filt_all = np.zeros((T_obs, k_dim))
        P_pred_all    = np.zeros((T_obs, k_dim, k_dim))
        P_filt_all    = np.zeros((T_obs, k_dim, k_dim))
        innov_all     = np.zeros((T_obs, N_obs))
        loglik        = 0.0

        beta = beta0.copy()
        P    = P0.copy()

        for t in range(T_obs):
            # Predicción
            beta_p = mu + F @ (beta - mu)
            P_p    = F @ P @ F.T + Q
            P_p    = 0.5 * (P_p + P_p.T)
            beta_pred_all[t] = beta_p
            P_pred_all[t]    = P_p

            # Innovación
            Ht  = H_arr[t]
            v   = yields[t] - Ht @ beta_p
            HP  = Ht @ P_p
            S   = HP @ Ht.T
            for n in range(N_obs):
                S[n, n] += R_diag[n]
            S = 0.5 * (S + S.T)

            # Log-det vía Cholesky
            L_ch    = np.linalg.cholesky(S + 1e-10 * np.eye(N_obs))
            log_det = 2.0 * np.sum(np.log(np.diag(L_ch)))
            z       = np.linalg.solve(L_ch, v)
            quad    = np.dot(z, z)
            loglik += -0.5 * (N_obs * LOG2PI + log_det + quad)
            innov_all[t] = v

            # Ganancia de Kalman
            Z  = np.linalg.solve(L_ch, HP)
            Kt = np.linalg.solve(L_ch.T, Z)
            K  = Kt.T                   # (k, N)

            beta = beta_p + K @ v

            # Forma Joseph (numéricamente estable)
            Rmat = np.zeros((N_obs, N_obs))
            for n in range(N_obs):
                Rmat[n, n] = R_diag[n]
            IKH = Ik - K @ Ht
            P   = IKH @ P_p @ IKH.T + K @ Rmat @ K.T
            P   = 0.5 * (P + P.T)

            beta_filt_all[t] = beta
            P_filt_all[t]    = P

        return beta_pred_all, beta_filt_all, P_pred_all, P_filt_all, innov_all, loglik

    def _kf_core_python(yields, H_arr, F, Q, R_diag, mu, beta0, P0):
        """Versión Python puro (fallback)."""
        T_obs, N_obs = yields.shape
        k_dim = F.shape[0]
        Ik    = np.eye(k_dim)
        R_mat = np.diag(R_diag)
        LOG2PI = np.log(2.0 * np.pi)

        beta_pred_all = np.zeros((T_obs, k_dim))
        beta_filt_all = np.zeros((T_obs, k_dim))
        P_pred_all    = np.zeros((T_obs, k_dim, k_dim))
        P_filt_all    = np.zeros((T_obs, k_dim, k_dim))
        innov_all     = np.zeros((T_obs, N_obs))
        loglik        = 0.0

        beta = beta0.copy()
        P    = P0.copy()

        for t in range(T_obs):
            beta_p = mu + F @ (beta - mu)
            P_p    = F @ P @ F.T + Q
            P_p    = 0.5 * (P_p + P_p.T)
            beta_pred_all[t] = beta_p
            P_pred_all[t]    = P_p

            Ht = H_arr[t]
            v  = yields[t] - Ht @ beta_p
            S  = Ht @ P_p @ Ht.T + R_mat
            S  = 0.5 * (S + S.T)

            L       = np.linalg.cholesky(S + 1e-10 * np.eye(N_obs))
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            z       = np.linalg.solve(L, v)
            quad    = z @ z
            loglik += -0.5 * (N_obs * LOG2PI + log_det + quad)
            innov_all[t] = v

            K    = np.linalg.solve(S, Ht @ P_p).T
            beta = beta_p + K @ v
            IKH  = Ik - K @ Ht
            P    = IKH @ P_p @ IKH.T + K @ R_mat @ K.T
            P    = 0.5 * (P + P.T)
            beta_filt_all[t] = beta
            P_filt_all[t]    = P

        return beta_pred_all, beta_filt_all, P_pred_all, P_filt_all, innov_all, loglik

    def kalman_filter(yields, H_arr, F, Q, R_diag, mu, beta0, P0):
        """Despacha a Numba o Python puro según disponibilidad."""
        return _kf_core_numba(
            np.asarray(yields, np.float64), H_arr, F, Q,
            R_diag, mu, beta0, P0)

else:
    # Sin Numba — sólo Python puro

    def kalman_filter(yields, H_arr, F, Q, R_diag, mu, beta0, P0):
        """Kalman Filter con H_t variable — Python puro."""
        T_obs, N_obs = yields.shape
        k_dim = F.shape[0]
        Ik    = np.eye(k_dim)
        R_mat = np.diag(R_diag)
        LOG2PI = np.log(2.0 * np.pi)

        beta_pred_all = np.zeros((T_obs, k_dim))
        beta_filt_all = np.zeros((T_obs, k_dim))
        P_pred_all    = np.zeros((T_obs, k_dim, k_dim))
        P_filt_all    = np.zeros((T_obs, k_dim, k_dim))
        innov_all     = np.zeros((T_obs, N_obs))
        loglik        = 0.0

        beta = beta0.copy()
        P    = P0.copy()

        for t in range(T_obs):
            beta_p = mu + F @ (beta - mu)
            P_p    = F @ P @ F.T + Q
            P_p    = 0.5 * (P_p + P_p.T)
            beta_pred_all[t] = beta_p
            P_pred_all[t]    = P_p

            Ht = H_arr[t]
            v  = yields[t] - Ht @ beta_p
            S  = Ht @ P_p @ Ht.T + R_mat
            S  = 0.5 * (S + S.T)

            L       = np.linalg.cholesky(S + 1e-10 * np.eye(N_obs))
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            z       = np.linalg.solve(L, v)
            quad    = z @ z
            loglik += -0.5 * (N_obs * LOG2PI + log_det + quad)
            innov_all[t] = v

            K    = np.linalg.solve(S, Ht @ P_p).T
            beta = beta_p + K @ v
            IKH  = Ik - K @ Ht
            P    = IKH @ P_p @ IKH.T + K @ R_mat @ K.T
            P    = 0.5 * (P + P.T)
            beta_filt_all[t] = beta
            P_filt_all[t]    = P

        return beta_pred_all, beta_filt_all, P_pred_all, P_filt_all, innov_all, loglik


# ═══════════════════════════════════════════════════════════════════════════════
# RTS SMOOTHER
# ═══════════════════════════════════════════════════════════════════════════════

def rts_smoother(beta_filt, P_filt, beta_pred, P_pred, F):
    """
    Rauch-Tung-Striebel Smoother.

    Retorna
    -------
    beta_smooth : (T, k)
    P_smooth    : (T, k, k)
    G_all       : (T, k, k)  ganancias suavizadoras
    """
    T, k   = beta_filt.shape
    beta_s = np.zeros_like(beta_filt)
    P_s    = np.zeros_like(P_filt)
    G_all  = np.zeros_like(P_filt)

    beta_s[-1] = beta_filt[-1]
    P_s[-1]    = P_filt[-1]

    for t in range(T - 2, -1, -1):
        try:
            G = P_filt[t] @ F.T @ np.linalg.inv(P_pred[t + 1])
        except np.linalg.LinAlgError:
            G = np.zeros((k, k))
        beta_s[t] = beta_filt[t] + G @ (beta_s[t + 1] - beta_pred[t + 1])
        dP        = P_s[t + 1] - P_pred[t + 1]
        P_s[t]    = P_filt[t] + G @ dP @ G.T
        P_s[t]    = 0.5 * (P_s[t] + P_s[t].T)
        G_all[t]  = G

    return beta_s, P_s, G_all


# ═══════════════════════════════════════════════════════════════════════════════
# UTILIDADES DEL MODELO
# ═══════════════════════════════════════════════════════════════════════════════

def compute_P0_stationary(F: np.ndarray, Q: np.ndarray, F_type: str) -> np.ndarray:
    """Covarianza inicial estacionaria via ecuación de Lyapunov discreta."""
    k = F.shape[0]
    if F_type == 'diagonal':
        f_vec = np.diag(F)
        q_vec = np.diag(Q)
        return np.diag(q_vec / np.maximum(1 - f_vec ** 2, 1e-6))
    try:
        return solve_discrete_lyapunov(F, Q)
    except Exception:
        return np.eye(k) * 1.0


def build_H_array(maturities, tau1_t, tau2_t, model):
    """
    Construye H_arr (T, N, k) desde las secuencias tau_t.
    tau2_t puede ser None o array de NaN para DNS.
    """
    T_obs = len(tau1_t)
    k     = 3 if model == 'DNS' else 4
    N     = len(maturities)
    H_arr = np.zeros((T_obs, N, k))
    for t in range(T_obs):
        if model == 'DNS':
            H_arr[t] = ns_loadings(maturities, tau1_t[t])
        else:
            H_arr[t] = nss_loadings(maturities, tau1_t[t], tau2_t[t])
    return H_arr


# ═══════════════════════════════════════════════════════════════════════════════
# OPCIÓN A: R estructurada por grupos de plazos
# ═══════════════════════════════════════════════════════════════════════════════

def build_R_groups(maturities, r_groups, group_thresholds=(3.0, 10.0)):
    """
    Construye R_diag (N,) con tres varianzas de medición (corto/medio/largo).

    Parámetros
    ----------
    r_groups         : (3,) varianzas [σ²_corto, σ²_medio, σ²_largo]
    group_thresholds : (th1, th2) en años
    """
    th1, th2 = group_thresholds
    R_diag   = np.empty(len(maturities))
    for n, m in enumerate(maturities):
        if   m <= th1: R_diag[n] = r_groups[0]
        elif m <= th2: R_diag[n] = r_groups[1]
        else:          R_diag[n] = r_groups[2]
    return R_diag


def _R_groups_from_log(log_r_groups, maturities, group_thresholds):
    return build_R_groups(maturities, np.exp(log_r_groups), group_thresholds)


# ═══════════════════════════════════════════════════════════════════════════════
# SERIALIZACIÓN DE PARÁMETROS PARA EL OPTIMIZADOR
# ═══════════════════════════════════════════════════════════════════════════════

def pack_params(mu, F, Q_diag, R_diag, k, F_type,
                r_mode='libre', r_groups_log=None):
    """Empaqueta (μ, F, Q_diag, R_diag) → vector para scipy.optimize."""
    p = list(mu)

    if F_type == 'diagonal':
        f_vec = np.clip(np.diag(F), -0.994, 0.994)
        p += list(np.arctanh(f_vec / 0.995))
    elif F_type == 'triangular':
        for i in range(k):
            for j in range(i + 1):
                val = F[i, j]
                p.append(np.arctanh(np.clip(val, -0.994, 0.994) / 0.995) if i == j
                         else float(val))
    else:
        p += list(F.flatten())

    p += list(np.log(np.maximum(Q_diag, 1e-10)))

    if r_mode == 'libre':
        p += list(np.log(np.maximum(R_diag, 1e-10)))
    else:
        p += list(r_groups_log)

    return np.array(p)


def unpack_params(params, k, N, F_type,
                  r_mode='libre', maturities=None,
                  group_thresholds=(3.0, 10.0)):
    """Desempaqueta vector → (μ, F, Q, Q_diag, R_diag, r_groups_log)."""
    idx = 0
    mu  = params[idx:idx + k]; idx += k

    F = np.zeros((k, k))
    if F_type == 'diagonal':
        x = params[idx:idx + k]; idx += k
        np.fill_diagonal(F, 0.995 * np.tanh(x))
    elif F_type == 'triangular':
        n_tri = k * (k + 1) // 2
        x = params[idx:idx + n_tri]; idx += n_tri
        p = 0
        for i in range(k):
            for j in range(i + 1):
                F[i, j] = (0.995 * np.tanh(x[p]) if i == j else x[p])
                p += 1
    else:
        n_full = k * k
        F = params[idx:idx + n_full].reshape(k, k); idx += n_full

    Q_diag = np.exp(params[idx:idx + k]); idx += k

    if r_mode == 'libre':
        R_diag       = np.exp(params[idx:idx + N])
        r_groups_log = None
    else:
        r_groups_log = params[idx:idx + 3]
        R_diag       = _R_groups_from_log(r_groups_log, maturities, group_thresholds)

    return mu, F, np.diag(Q_diag), Q_diag, R_diag, r_groups_log


def _random_init_params(k, N, F_type, seed, r_mode='libre'):
    np.random.seed(seed)
    mu = np.random.randn(k) * 2.0

    if F_type == 'diagonal':
        x_f = np.arctanh(np.random.uniform(0.85, 0.98, k) / 0.995)
    elif F_type == 'triangular':
        n_tri = k * (k + 1) // 2
        x_f = np.zeros(n_tri); p = 0
        for i in range(k):
            for j in range(i + 1):
                x_f[p] = (np.arctanh(np.random.uniform(0.85, 0.98) / 0.995)
                          if i == j else np.random.randn() * 0.1)
                p += 1
    else:
        x_f = np.random.randn(k * k) * 0.1

    log_q = np.random.uniform(-4, -2, k)
    log_r = np.random.uniform(-5, -2, 3 if r_mode == 'grupos' else N)
    return np.concatenate([mu, x_f, log_q, log_r])


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN OBJETIVO Y MLE
# ═══════════════════════════════════════════════════════════════════════════════

def neg_loglik(params, yields, H_arr, k, N, F_type,
               obs_weights=None,
               r_mode='libre', maturities=None,
               group_thresholds=(3.0, 10.0)):
    """
    Función objetivo: −log-verosimilitud del KF.

    obs_weights (Opción C): pondera plazos en la log-lik escalando R_diag.
    r_mode='grupos' (Opción A): solo 3 parámetros de varianza de medición.
    """
    mu, F, Q, Q_diag, R_diag, _ = unpack_params(
        params, k, N, F_type, r_mode, maturities, group_thresholds)

    if obs_weights is not None:
        w      = np.asarray(obs_weights, dtype=float)
        R_diag = R_diag / np.maximum(w, 1e-6)

    P0    = compute_P0_stationary(F, Q, F_type)
    beta0 = mu.copy()
    try:
        *_, loglik = kalman_filter(yields, H_arr, F, Q, R_diag, mu, beta0, P0)
    except Exception:
        return 1e15
    return -loglik if np.isfinite(loglik) else 1e15


def mle_estimate(yields, H_arr, k, N, F_type='diagonal',
                 n_starts=5, verbose=True,
                 obs_weights=None,
                 r_mode='libre', maturities=None,
                 group_thresholds=(3.0, 10.0)):
    """
    Estimación MLE multi-start vía L-BFGS-B.

    Retorna
    -------
    mu, F, Q, R_diag_final, best_loglik, best_params, r_groups_log
    """
    best_loglik = -np.inf
    best_params = None

    if verbose:
        modo_r = f"R-{r_mode}"
        modo_w = "ponderada" if obs_weights is not None else "uniforme"
        print(f"  MLE multi-start ({n_starts} intentos) — "
              f"F:'{F_type}'  R:{modo_r}  log-lik:{modo_w}")

    for seed in range(n_starts):
        try:
            p0  = _random_init_params(k, N, F_type, seed, r_mode)
            res = minimize(
                neg_loglik, p0,
                args=(yields, H_arr, k, N, F_type,
                      obs_weights, r_mode, maturities, group_thresholds),
                method='L-BFGS-B',
                options={'maxiter': 800, 'ftol': 1e-11, 'gtol': 1e-7})
            ll = -res.fun
            if ll > best_loglik:
                best_loglik = ll
                best_params = res.x
            if verbose:
                ok = 'OK' if res.success else '~'
                print(f"    Start {seed + 1}: log-lik={ll:,.2f} ({ok})")
        except Exception as ex:
            if verbose:
                print(f"    Start {seed + 1}: falló ({ex})")

    if best_params is None:
        raise RuntimeError("Todos los intentos MLE fallaron")

    mu, F, Q, Q_diag, R_diag, r_groups_log = unpack_params(
        best_params, k, N, F_type, r_mode, maturities, group_thresholds)

    # Si se usaron pesos, el KF final debe ver R_diag/w
    if obs_weights is not None:
        w             = np.asarray(obs_weights, dtype=float)
        R_diag_final  = R_diag / np.maximum(w, 1e-6)
    else:
        R_diag_final = R_diag

    if verbose:
        eigvals = np.abs(np.linalg.eigvals(F))
        print(f"\n  Mejor log-lik = {best_loglik:,.2f}")
        print(f"  F estable: {np.all(eigvals < 1)}  "
              f"(max|λ|={eigvals.max():.4f})")
        print(f"  μ        = {np.round(mu, 3)}")
        print(f"  diag(F)  = {np.round(np.diag(F), 3)}")
        print(f"  √Q_diag  = {np.round(np.sqrt(Q_diag), 5)}")
        if r_mode == 'grupos':
            r_g  = np.exp(r_groups_log)
            th1, th2 = group_thresholds
            print(f"  R grupos : corto(≤{th1}Y)={np.sqrt(r_g[0])*100:.3f}pb  "
                  f"medio(≤{th2}Y)={np.sqrt(r_g[1])*100:.3f}pb  "
                  f"largo(>{th2}Y)={np.sqrt(r_g[2])*100:.3f}pb")
        else:
            print(f"  √R*100   = {np.round(np.sqrt(R_diag)*100, 3)}")

    return mu, F, Q, R_diag_final, best_loglik, best_params, r_groups_log
