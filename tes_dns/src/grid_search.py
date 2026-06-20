"""
grid_search.py
==============
Búsqueda adaptativa de τ_t (o τ₁_t, τ₂_t para DNSS) mediante
Rolling/Expanding Grid Search sobre perfil SSE via el Residual Maker.

Eficiencia: M_τ se precomputa una vez. Para cada (t, τ) solo se
evalúa tr(M_τ Σ_t) en O(N²) vía einsum.

Funciones exportadas
--------------------
rolling_tau_dns(yields, maturities, tau1_grid, window_size)
rolling_tau_dnss(yields, maturities, tau1_grid, tau2_grid, window_size, sep_min)

Referencias
-----------
Diebold & Li (2006)  — ventana expandiente
Caldeira et al. (2010) — ventana rodante
"""

import numpy as np

from .nelson_siegel import ns_loadings, nss_loadings


# ── Utilidades internas ────────────────────────────────────────────────────────

def _residual_maker(H: np.ndarray) -> np.ndarray | None:
    """
    M = I − H(H'H)⁻¹H'  (proyector al complemento ortogonal de col(H)).
    Retorna None si H está mal condicionada.
    """
    N, k = H.shape
    HtH  = H.T @ H
    if np.linalg.cond(HtH) > 1e12:
        return None
    try:
        return np.eye(N) - H @ np.linalg.solve(HtH, H.T)
    except np.linalg.LinAlgError:
        return None


def _build_YYt_cumsum(yields: np.ndarray) -> np.ndarray:
    """
    Sumas acumuladas de yₛ yₛ'. Retorna (T+1, N, N).
    Permite obtener Σ_t = YYt[t1] − YYt[t0] en O(1) por ventana.
    """
    T, N = yields.shape
    YYt  = np.zeros((T + 1, N, N))
    for t in range(T):
        yt        = yields[t]
        YYt[t + 1] = YYt[t] + np.outer(yt, yt)
    return YYt


# ═══════════════════════════════════════════════════════════════════════════════
# GRID SEARCH — DNS (1 parámetro de forma)
# ═══════════════════════════════════════════════════════════════════════════════

def rolling_tau_dns(yields: np.ndarray,
                    maturities: np.ndarray,
                    tau1_grid: np.ndarray,
                    window_size: int | None = None) -> tuple:
    """
    Búsqueda adaptativa de τ₁_t para DNS.

    Parámetros
    ----------
    window_size : int|None
        None → ventana expandiente (causal, recomendada para pronóstico).
        int  → ventana rodante de los últimos W periodos.

    Retorna
    -------
    tau1_t     : (T,)      secuencia óptima de τ₁
    sse_t      : (T,)      SSE mínimo por periodo
    sse_grid_t : (T, G1)   perfil SSE completo (para diagnóstico)
    """
    T, N = yields.shape
    G1   = len(tau1_grid)

    # Precompute Mᵥ por cada candidato τ
    M_list = []
    for tau1 in tau1_grid:
        H  = ns_loadings(maturities, tau1)
        Mv = _residual_maker(H)
        M_list.append(Mv if Mv is not None else np.eye(N))
    M_arr = np.stack(M_list, axis=0)       # (G1, N, N)

    YYt_cum    = _build_YYt_cumsum(yields)
    default    = G1 // 2
    tau1_t     = np.zeros(T)
    sse_t      = np.zeros(T)
    sse_grid_t = np.full((T, G1), np.nan)

    for t in range(T):
        if window_size is None:
            Sigma = YYt_cum[t + 1]; n_win = t + 1
        else:
            t0    = max(0, t - window_size + 1)
            Sigma = YYt_cum[t + 1] - YYt_cum[t0]; n_win = t - t0 + 1

        if n_win < 5:                      # mínimo robusto: k+2 = 5
            tau1_t[t] = tau1_grid[default]; continue

        # sse_vec[g] = tr(M_arr[g] @ Σ) — vectorizado
        sse_vec           = np.einsum('gij,ji->g', M_arr, Sigma)
        sse_grid_t[t]     = sse_vec
        best_g            = int(np.argmin(sse_vec))
        tau1_t[t]         = tau1_grid[best_g]
        sse_t[t]          = sse_vec[best_g]

    return tau1_t, sse_t, sse_grid_t


# ═══════════════════════════════════════════════════════════════════════════════
# GRID SEARCH — DNSS (2 parámetros de forma)
# ═══════════════════════════════════════════════════════════════════════════════

def rolling_tau_dnss(yields: np.ndarray,
                     maturities: np.ndarray,
                     tau1_grid: np.ndarray,
                     tau2_grid: np.ndarray,
                     window_size: int | None = None,
                     sep_min: float = 0.30) -> tuple:
    """
    Búsqueda adaptativa de (τ₁_t, τ₂_t) para DNSS.

    Restricción de identificación: τ₁ > τ₂ + sep_min.
    τ₁ controla siempre el hump lento (largo plazo).
    τ₂ controla siempre el hump rápido (corto plazo).

    Retorna
    -------
    tau1_t : (T,)
    tau2_t : (T,)
    sse_t  : (T,)
    """
    T, N   = yields.shape
    G1, G2 = len(tau1_grid), len(tau2_grid)

    # Precompute Mᵥ para pares (τ₁, τ₂) válidos
    M_arr = np.zeros((G1, G2, N, N))
    valid = np.zeros((G1, G2), dtype=bool)

    for i, tau1 in enumerate(tau1_grid):
        for j, tau2 in enumerate(tau2_grid):
            if tau1 <= tau2 + sep_min:
                continue
            H  = nss_loadings(maturities, tau1, tau2)
            Mv = _residual_maker(H)
            if Mv is not None:
                M_arr[i, j] = Mv
                valid[i, j] = True

    valid_pairs = np.argwhere(valid)
    if len(valid_pairs) == 0:
        raise ValueError(
            "No hay pares (τ₁, τ₂) válidos — revisa los rangos de tau1_grid y tau2_grid")

    mid  = len(valid_pairs) // 2
    di, dj = valid_pairs[mid]

    YYt_cum = _build_YYt_cumsum(yields)
    tau1_t  = np.zeros(T)
    tau2_t  = np.zeros(T)
    sse_t   = np.zeros(T)

    for t in range(T):
        if window_size is None:
            Sigma = YYt_cum[t + 1]; n_win = t + 1
        else:
            t0    = max(0, t - window_size + 1)
            Sigma = YYt_cum[t + 1] - YYt_cum[t0]; n_win = t - t0 + 1

        if n_win < 6:
            tau1_t[t] = tau1_grid[di]; tau2_t[t] = tau2_grid[dj]; continue

        sse_grid = np.einsum('ghjk,kj->gh', M_arr, Sigma)
        sse_grid[~valid] = np.inf

        flat         = int(np.argmin(sse_grid.ravel()))
        bi, bj       = np.unravel_index(flat, (G1, G2))
        tau1_t[t]    = tau1_grid[bi]
        tau2_t[t]    = tau2_grid[bj]
        sse_t[t]     = sse_grid[bi, bj]

    return tau1_t, tau2_t, sse_t
