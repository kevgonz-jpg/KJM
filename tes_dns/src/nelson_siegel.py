"""
nelson_siegel.py
================
Cargas (factor loadings) para modelos Nelson-Siegel (NS) y
Nelson-Siegel-Svensson (NSS), más inicialización OLS de τ.

Funciones exportadas
--------------------
ns_loadings(maturities, tau1)             → (N, 3)
nss_loadings(maturities, tau1, tau2)      → (N, 4)
get_H(maturities, tau1, tau2, model)      → (N, k)
ols_init_tau(yields, maturities, model)   → (tau1, tau2|None)

Referencias
-----------
Nelson & Siegel (1987)
Svensson (1994)
Diebold & Li (2006)
"""

import numpy as np


# ── Cargas ─────────────────────────────────────────────────────────────────────

def ns_loadings(maturities: np.ndarray, tau1: float) -> np.ndarray:
    """
    Cargas Nelson-Siegel. Shape (N, 3).

    Columnas: [nivel=1, pendiente=load2, curvatura=load3]
    Hump de curvatura en m* ≈ 1.793 * tau1 años.
    """
    m  = np.asarray(maturities, dtype=float)
    mt = m / tau1
    e  = np.exp(-mt)
    c  = (1.0 - e) / mt
    return np.column_stack([np.ones_like(m), c, c - e])


def nss_loadings(maturities: np.ndarray, tau1: float, tau2: float) -> np.ndarray:
    """
    Cargas Nelson-Siegel-Svensson. Shape (N, 4).

    Columnas: [nivel, pendiente, curvatura1, curvatura2]
    """
    m   = np.asarray(maturities, dtype=float)
    mt1 = m / tau1;  mt2 = m / tau2
    e1  = np.exp(-mt1);  e2 = np.exp(-mt2)
    c1  = (1.0 - e1) / mt1
    return np.column_stack([np.ones_like(m), c1, c1 - e1, (1.0 - e2) / mt2 - e2])


def get_H(maturities: np.ndarray,
          tau1: float,
          tau2: float | None = None,
          model: str = 'DNS') -> np.ndarray:
    """
    Wrapper unificado. Devuelve (N, 3) para DNS, (N, 4) para DNSS.
    """
    if model == 'DNS':
        return ns_loadings(maturities, tau1)
    if tau2 is None:
        raise ValueError("tau2 es requerido para DNSS")
    return nss_loadings(maturities, tau1, tau2)


# ── Inicialización OLS de τ ────────────────────────────────────────────────────

def ols_init_tau(
    yields: np.ndarray,
    maturities: np.ndarray,
    model: str = 'DNS',
    tau1_grid: np.ndarray | None = None,
    tau2_grid: np.ndarray | None = None,
) -> tuple[float, float | None]:
    """
    Busca τ inicial vía Grid OLS (mínimo SSE sobre toda la muestra).
    Sirve como punto de arranque para el KF o el Gibbs sampler.

    Retorna
    -------
    (tau1_best, tau2_best)  — tau2_best es None para DNS
    """
    if tau1_grid is None:
        tau1_grid = np.linspace(0.5, 4.0, 12)
    if tau2_grid is None:
        tau2_grid = np.linspace(0.15, 2.0, 8)

    best_sse = np.inf
    best_t1  = tau1_grid[len(tau1_grid) // 2]
    best_t2  = tau2_grid[len(tau2_grid) // 4] if model == 'DNSS' else None

    for t1 in tau1_grid:
        candidates = [None] if model == 'DNS' else tau2_grid
        for t2 in candidates:
            if model == 'DNSS' and abs(t1 - t2) < 0.15:
                continue
            H = get_H(maturities, t1, t2, model)
            try:
                B   = np.linalg.lstsq(H, yields.T, rcond=None)[0]
                sse = float(np.sum((yields.T - H @ B) ** 2))
            except Exception:
                continue
            if sse < best_sse:
                best_sse = sse
                best_t1  = t1
                if model == 'DNSS':
                    best_t2 = t2

    return best_t1, best_t2
