# DNS/DNSS — TES Colombia
Estimación y pronóstico de curvas de rendimiento TES mediante
**Dynamic Nelson-Siegel (DNS)** y **Nelson-Siegel-Svensson (DNSS)**,
con enfoques frecuentista (Kalman Filter + MLE) y bayesiano (Gibbs sampler + FFBS).

---

## Estructura del proyecto

```
tes_dns/
│
├── src/                        # Paquete principal — importable
│   ├── __init__.py             # Re-exporta todo
│   ├── nelson_siegel.py        # Cargas NS/NSS, get_H, OLS init τ
│   ├── kalman.py               # Filtro KF, RTS Smoother, MLE L-BFGS-B
│   ├── grid_search.py          # Grid search adaptativo τ_t (SSE via Residual Maker)
│   ├── gibbs.py                # Gibbs sampler: FFBS, posteriors conjugadas, MH τ
│   ├── diagnostics.py          # R-hat, ESS, Ljung-Box, Jarque-Bera
│   ├── forecasting.py          # OOS ventana expandiente, RMSE/MAE
│   └── data_loader.py          # Carga TES desde CSV, índice días hábiles
│
├── scripts/                    # Ejecución batch por CLI
│   ├── run_kalman.py           # Pipeline KF completo
│   ├── run_bayesian.py         # Pipeline Gibbs completo
│   └── run_comparison.py       # Fase 4: comparación OOS
│
├── notebooks/                  # (tus notebooks existentes — sin cambios)
│   ├── DNS_DNSS_KalmanFilter_Adaptativo.ipynb
│   └── DNS_DNSS_Bayesiano_Gibbs_v2.ipynb
│
└── outputs/                    # Resultados generados (.npy, .csv)
```

---

## Instalación de dependencias

```bash
pip install numpy scipy pandas joblib matplotlib
# Opcional pero recomendado para velocidad:
pip install numba
```

---

## Formato del CSV

El archivo CSV debe tener la siguiente estructura:

```
fecha,r_2yr,r_3yr,r_5yr,r_7yr,r_10yr,r_15yr
2015-01-05,4.52,4.81,5.10,5.34,5.67,6.01
...
```

- Columna de fecha: `fecha`
- Columnas de rendimiento: `r_2yr`, `r_3yr`, `r_5yr`, `r_7yr`, `r_10yr`, `r_15yr`
- Solo se usan filas de días hábiles con datos completos (sin NaN)

---

## Uso desde scripts (CLI)

### 1. Kalman Filter frecuentista

```bash
# DNS con R libre, ventana expandiente de τ
python scripts/run_kalman.py --csv datos/curvas_panel_historico_limpio.csv --model DNS

# DNSS con R por grupos de plazo, ventana rodante de 120 días
python scripts/run_kalman.py \
    --csv datos/curvas_panel_historico_limpio.csv \
    --model DNSS \
    --r_mode grupos \
    --window 120 \
    --n_starts 8
```

### 2. Gibbs sampler Bayesiano

```bash
# DNS τ estático, 4 cadenas paralelas
python scripts/run_bayesian.py --csv datos/curvas_panel_historico_limpio.csv --model DNS

# DNS τ AR(1), más iteraciones
python scripts/run_bayesian.py \
    --csv datos/curvas_panel_historico_limpio.csv \
    --model DNS \
    --tau_mode ar1 \
    --n_iter 3000 \
    --n_burnin 1500

# DNSS τ ventana rodante (W=120, step=21)
python scripts/run_bayesian.py \
    --csv datos/curvas_panel_historico_limpio.csv \
    --model DNSS \
    --tau_mode rolling \
    --window 120 \
    --step 21
```

### 3. Comparación out-of-sample (Fase 4)

```bash
# Primero correr run_kalman y run_bayesian, luego:
python scripts/run_comparison.py \
    --csv datos/curvas_panel_historico_limpio.csv \
    --model DNS \
    --horizons 1,3,6,12 \
    --min_train 120 \
    --refit_every 21
```

---

## Uso desde notebooks (importando src/)

```python
import sys
sys.path.insert(0, '../')   # si el notebook está en notebooks/

from src import (
    load_tes_data, get_H,
    rolling_tau_dns, build_H_array,
    mle_estimate, kalman_filter, rts_smoother,
    make_priors, run_chains_parallel,
    print_diagnostics, compute_metrics,
)

# Cargar datos
yields, dates, maturities = load_tes_data('datos/curvas_panel_historico_limpio.csv')

# Grid search τ_t
tau1_grid = np.linspace(0.3, 5.0, 20)
tau1_t, _, _ = rolling_tau_dns(yields, maturities, tau1_grid, window_size=120)

# MLE
H_arr = build_H_array(maturities, tau1_t, None, 'DNS')
mu, F, Q, R_diag, ll, *_ = mle_estimate(yields, H_arr, k=3, N=len(maturities))

# Gibbs
priors = make_priors(model='DNS')
chains = run_chains_parallel(yields, maturities, priors, n_iter=1500, n_burnin=750)
print_diagnostics(chains)
```

---

## Decisiones de diseño clave

| Decisión | Elección | Justificación |
|---|---|---|
| Estructura F | Diagonal | Caldeira et al. (2010), Çakmaklı (2013) |
| τ en Gibbs | MH log-escala | Exploración simétrica en escala natural |
| τ AR(1) prop_std | 0.15 (recomendado) | Exploración adecuada; 0.04 es demasiado estrecho |
| Ventana rodante | `run_rolling_bayes` directo | `comparar_modos_tau` no expone `step` |
| OOS forecasting | β̂_{t\|t-1} (predichos) | β̂_{t\|T} (suavizados) incorporan información futura |
| Identificación DNSS | τ₁ > τ₂ + sep_min | Constraint direccional, no simétrico |
| Imputación faltantes | NO | Solo días hábiles con datos reales BVC |

---

## Referencias

- Nelson & Siegel (1987)
- Svensson (1994)
- Diebold & Li (2006)
- Caldeira, Laurini & Portugal (2010)
- Çakmaklı (2013)
- Carter & Kohn (1994)
- Hamilton (1994) Cap. 13
