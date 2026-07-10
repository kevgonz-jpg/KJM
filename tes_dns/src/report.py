"""
report.py
=========
Genera un reporte PDF académico completo con los resultados
del pipeline DNS/DNSS, compilando un archivo LaTeX desde Python.

Flujo:
    1. Guarda figuras (.png) en outputs/figures/
    2. Genera outputs/reporte_dns.tex  con tablas y rutas a figuras
    3. Llama a pdflatex (dos veces para TOC/referencias)
    4. El PDF final queda en outputs/reporte_dns.pdf

Funciones exportadas
--------------------
plot_yield_curve(yields, dates, maturities, ...)  → path PNG
plot_factors(betas, dates, model, beta_std, ...)   → path PNG
plot_tau(tau1_t, dates, tau2_t, ..., bandas/comparación) → path PNG
plot_residuals(residuals, maturities, ...)         → path PNG
plot_residuals_heatmap(residuals, maturities, ...) → path PNG
plot_diagnostics_mcmc(chains, ...)                 → path PNG
plot_diagnostics_pf(res_pf, ...)                   → path PNG
plot_oos_comparison(df_kf, df_bayes, horizons, ...)→ path PNG
build_latex_report(config)                         → path TEX
compile_pdf(tex_path)                              → path PDF | None
generate_full_report(...)                          → path PDF | None
"""

import os
import subprocess
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from datetime import datetime


# ── Estilo global de figuras ───────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'      : 'serif',
    'font.size'        : 9,
    'axes.titlesize'   : 10,
    'axes.labelsize'   : 9,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.grid'        : True,
    'grid.alpha'       : 0.3,
    'grid.linestyle'   : '--',
    'figure.dpi'       : 150,
    'savefig.dpi'      : 150,
    'savefig.bbox'     : 'tight',
    'savefig.facecolor': 'white',
})

COLORES = ['#2C5F8A', '#C0392B', '#27AE60', '#8E44AD', '#E67E22', '#16A085']


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURAS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_yield_curve(yields, dates, maturities,
                     n_curves=8, out_dir='outputs/figures',
                     filename='fig_yield_curves.png') -> str:
    """
    Muestra n_curves curvas de rendimiento seleccionadas uniformemente
    más la curva media de la muestra.
    """
    _ensure_dir(out_dir)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Panel izquierdo: curvas seleccionadas
    ax = axes[0]
    idx = np.linspace(0, len(yields) - 1, n_curves, dtype=int)
    cmap = plt.cm.Blues(np.linspace(0.35, 0.95, n_curves))
    for i, t in enumerate(idx):
        label = str(dates[t])[:10] if i in [0, n_curves // 2, n_curves - 1] else None
        ax.plot(maturities, yields[t], color=cmap[i], lw=1.4, label=label)
    ax.plot(maturities, yields.mean(axis=0), 'k--', lw=1.8, label='Media muestral')
    ax.set_xlabel('Plazo (años)')
    ax.set_ylabel('Rendimiento (%)')
    ax.set_title('Curvas de rendimiento TES seleccionadas')
    ax.legend(fontsize=7, loc='lower right')

    # Panel derecho: serie temporal por plazo
    ax = axes[1]
    for i, m in enumerate(maturities):
        ax.plot(dates, yields[:, i], lw=0.8,
                color=COLORES[i % len(COLORES)], label=f'{m:.0f}Y')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Rendimiento (%)')
    ax.set_title('Series de tiempo por plazo')
    ax.legend(fontsize=7, ncol=3, loc='upper right')
    fig.tight_layout()

    path = os.path.join(out_dir, filename)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_factors(betas, dates, model='DNS',
                 beta_std=None,
                 out_dir='outputs/figures',
                 filename='fig_factors.png') -> str:
    """
    Grafica los factores latentes (nivel, pendiente, curvatura).

    Parámetros
    ----------
    beta_std : (T, k) | None
        Desviación estándar de cada factor en cada periodo (p.ej.
        sqrt(diag(P_smooth)) del KF, o x_filt_std del Particle Filter).
        Si se provee, se dibuja una banda de confianza al 95%
        (beta ± 1.96·std) detrás de la línea, igual que en el notebook
        del Particle Filter.
    """
    _ensure_dir(out_dir)
    k      = betas.shape[1]
    labels = ['Nivel (\\u03b2\\u2081)', 'Pendiente (\\u03b2\\u2082)',
              'Curvatura (\\u03b2\\u2083)', 'Curvatura2 (\\u03b2\\u2084)'][:k]
    colores_f = COLORES[:k]

    fig, axes = plt.subplots(k, 1, figsize=(12, 2.2 * k), sharex=True)
    if k == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        if beta_std is not None:
            lo = betas[:, i] - 1.96 * beta_std[:, i]
            hi = betas[:, i] + 1.96 * beta_std[:, i]
            ax.fill_between(dates, lo, hi, color=colores_f[i], alpha=0.18,
                             label='IC 95%' if i == 0 else None)
        ax.plot(dates, betas[:, i], color=colores_f[i], lw=1.0)
        ax.axhline(0, color='gray', lw=0.6, ls=':')
        ax.set_ylabel(labels[i], fontsize=8)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        if i == 0 and beta_std is not None:
            ax.legend(fontsize=7, loc='upper right')

    axes[-1].set_xlabel('Fecha')
    fig.suptitle(f'Factores latentes — {model}', y=1.01)
    fig.tight_layout()

    path = os.path.join(out_dir, filename)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_tau(tau1_t, dates, tau2_t=None, model='DNS',
            tau1_band=None, tau2_band=None,
            tau1_compare=None, tau2_compare=None, label_compare=None,
            out_dir='outputs/figures',
            filename='fig_tau.png') -> str:
    """
    Grafica la evolución temporal de tau1 (y tau2 si es DNSS).

    Parámetros
    ----------
    tau1_band, tau2_band : (lo, hi) | None
        Tupla de arrays (T,) con la banda de confianza (p.ej. percentiles
        de las partículas del PF, o tau1 ± 1.96·std en escala log
        exponenciada). Se dibuja como sombra detrás de la línea principal.
    tau1_compare, tau2_compare : (T,) | None
        Serie adicional para comparar contra la principal en el mismo
        panel (p.ej. tau1_t del grid search del KF superpuesto sobre el
        tau1_t dinámico del Particle Filter, o viceversa).
    label_compare : str | None
        Etiqueta de la serie de comparación (p.ej. 'KF (grid search)').
    """
    _ensure_dir(out_dir)
    n_panels = 2 if (tau2_t is not None and model == 'DNSS') else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    if tau1_band is not None:
        axes[0].fill_between(dates, tau1_band[0], tau1_band[1],
                             color=COLORES[0], alpha=0.18, label='IC 95%')
    axes[0].plot(dates, tau1_t, color=COLORES[0], lw=1.0, label=r'$\tau_1$')
    if tau1_compare is not None:
        axes[0].plot(dates, tau1_compare, color='#7f7f7f', lw=0.9, ls='--',
                     label=label_compare or 'Comparación')
    axes[0].set_ylabel(r'$\tau_1$')
    axes[0].set_title(f'Parametro de forma $\\tau_1$ — {model}')
    if tau1_band is not None or tau1_compare is not None:
        axes[0].legend(fontsize=8, loc='upper right')

    if n_panels == 2:
        if tau2_band is not None:
            axes[1].fill_between(dates, tau2_band[0], tau2_band[1],
                                 color=COLORES[1], alpha=0.18, label='IC 95%')
        axes[1].plot(dates, tau2_t, color=COLORES[1], lw=1.0, label=r'$\tau_2$')
        if tau2_compare is not None:
            axes[1].plot(dates, tau2_compare, color='#7f7f7f', lw=0.9, ls='--',
                         label=label_compare or 'Comparación')
        axes[1].set_ylabel(r'$\tau_2$')
        axes[1].set_title(r'Parametro de forma $\tau_2$')
        if tau2_band is not None or tau2_compare is not None:
            axes[1].legend(fontsize=8, loc='upper right')

    axes[-1].set_xlabel('Fecha')
    fig.tight_layout()

    path = os.path.join(out_dir, filename)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_residuals(residuals, maturities, dates,
                   out_dir='outputs/figures',
                   filename='fig_residuals.png') -> str:
    """
    Panel de diagnóstico de residuos:
    - Serie temporal por plazo
    - QQ-plot agregado
    - RMSE por plazo (barras)
    """
    _ensure_dir(out_dir)
    N   = len(maturities)
    res_pb = residuals * 100  # puntos básicos

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(3, N, figure=fig, hspace=0.45, wspace=0.35)

    # Fila 0: series temporales
    for i, m in enumerate(maturities):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(dates, res_pb[:, i], lw=0.6, color=COLORES[i % len(COLORES)])
        ax.axhline(0, color='k', lw=0.6, ls='--')
        ax.set_title(f'{m:.0f}Y', fontsize=8)
        ax.set_xlabel('')
        if i == 0:
            ax.set_ylabel('Residuo (pb)', fontsize=7)

    # Fila 1: histogramas con curva normal
    from scipy import stats as _stats
    for i, m in enumerate(maturities):
        ax   = fig.add_subplot(gs[1, i])
        r    = res_pb[:, i]
        ax.hist(r, bins=30, density=True, color=COLORES[i % len(COLORES)],
                alpha=0.6, edgecolor='none')
        xr   = np.linspace(r.min(), r.max(), 200)
        ax.plot(xr, _stats.norm.pdf(xr, r.mean(), r.std()), 'k-', lw=1.2)
        ax.set_title(f'{m:.0f}Y', fontsize=8)
        if i == 0:
            ax.set_ylabel('Densidad', fontsize=7)

    # Fila 2: QQ-plot
    for i, m in enumerate(maturities):
        ax = fig.add_subplot(gs[2, i])
        r  = res_pb[:, i]
        (osm, osr), (slope, intercept, _) = _stats.probplot(r)
        ax.scatter(osm, osr, s=3, alpha=0.5, color=COLORES[i % len(COLORES)])
        ax.plot(osm, slope * np.array(osm) + intercept, 'k-', lw=1.0)
        ax.set_title(f'{m:.0f}Y', fontsize=8)
        if i == 0:
            ax.set_ylabel('Cuantiles muestrales', fontsize=7)
        ax.set_xlabel('Cuantiles teóricos', fontsize=7)

    fig.suptitle('Diagnóstico de residuos de medicion (pb)', y=1.01)
    path = os.path.join(out_dir, filename)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_residuals_heatmap(residuals, maturities, dates,
                           out_dir='outputs/figures',
                           filename='fig_residuals_heatmap.png') -> str:
    """
    Mapa de calor de residuos de medición: plazos (eje Y) vs tiempo (eje X).
    Rojo = el modelo SUBESTIMA el rendimiento observado (residuo > 0,
    y_real > y_hat). Azul = el modelo SOBREESTIMA (residuo < 0).

    Complementa a plot_residuals(): aquella muestra series/histogramas/QQ
    por plazo; esta muestra de un vistazo si hay periodos o tramos de la
    curva donde el error se concentra sistemáticamente (p.ej. alrededor de
    quiebres estructurales conocidos como COVID o cambios de gobierno).
    """
    from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
    _ensure_dir(out_dir)
    res_pb = residuals * 100
    T = len(dates)

    fig, ax = plt.subplots(figsize=(14, 5))
    vmax = np.percentile(np.abs(res_pb), 97)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = LinearSegmentedColormap.from_list('bwr_soft', ['#2166ac', '#f7f7f7', '#d6604d'], N=256)
    im = ax.imshow(res_pb.T, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')

    n_t = min(10, T // 40 + 1)
    t_ticks = np.linspace(0, T - 1, n_t, dtype=int)
    ax.set_xticks(t_ticks)
    ax.set_xticklabels([str(dates[i])[:7] for i in t_ticks], rotation=45, ha='right')
    ax.set_yticks(range(len(maturities)))
    ax.set_yticklabels([f'{m:.0f}Y' for m in maturities])
    ax.set_xlabel('Fecha'); ax.set_ylabel('Plazo')
    ax.set_title('Residuos de medición (pb)\n[rojo = subestima, azul = sobreestima]')
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label='Residuo (pb)')
    plt.tight_layout()

    path = os.path.join(out_dir, filename)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_diagnostics_pf(res_pf, out_dir='outputs/figures',
                        filename='fig_pf_diagnostics.png') -> str:
    """
    Diagnóstico propio del Particle Filter: Effective Sample Size (ESS) a
    lo largo del tiempo y frecuencia de remuestreo — análogo en espíritu a
    los traceplots/R-hat del Gibbs sampler, pero para degeneración de
    partículas en vez de convergencia de cadenas MCMC.
    """
    _ensure_dir(out_dir)
    dates  = res_pf['dates']
    ess_t  = res_pf['ess_t']
    N      = res_pf['n_particles']
    resamp = res_pf['resampled_t']
    model  = res_pf['model']

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                             gridspec_kw=dict(height_ratios=[2, 1]))
    axes[0].plot(dates, ess_t / N, color=COLORES[0], lw=0.8)
    axes[0].axhline(0.5, color='gray', lw=0.8, ls='--', label='Umbral remuestreo (ESS/N=0.5)')
    axes[0].fill_between(dates, 0, ess_t / N, alpha=0.10, color=COLORES[0])
    axes[0].set_ylabel('ESS / N'); axes[0].set_ylim(0, 1.02); axes[0].legend(fontsize=8)
    axes[0].set_title(f'Effective Sample Size — {model} (N={N} partículas)')

    axes[1].scatter(np.array(dates)[resamp], np.ones(int(resamp.sum())),
                    s=3, color=COLORES[1], marker='|')
    axes[1].set_yticks([])
    axes[1].set_title(f'Periodos con remuestreo ({resamp.mean()*100:.1f}% del total)')
    axes[1].set_xlabel('Fecha')
    fig.tight_layout()

    path = os.path.join(out_dir, filename)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_diagnostics_mcmc(chains, out_dir='outputs/figures',
                           filename='fig_mcmc.png') -> str:
    """
    Traceplots y ACF de tau1, f_vec[0], mu[0] para diagnóstico MCMC.
    """
    _ensure_dir(out_dir)
    model   = chains[0]['model']
    params  = [('tau1', None, r'$\tau_1$'),
               ('f_vec', 0,   r'$f_1$ (nivel)'),
               ('mu',    0,   r'$\mu_1$ (nivel)'),
               ('q_vec', 0,   r'$q_1$')]

    n_par = len(params)
    fig, axes = plt.subplots(n_par, 2, figsize=(12, 2.5 * n_par))

    for row, (key, idx, label) in enumerate(params):
        # Traceplot
        ax_tr = axes[row, 0]
        for j, ch in enumerate(chains):
            s = ch[key] if idx is None else ch[key][:, idx]
            ax_tr.plot(s, lw=0.5, alpha=0.8, color=COLORES[j % len(COLORES)],
                       label=f'Cadena {j+1}')
        ax_tr.set_ylabel(label, fontsize=8)
        ax_tr.set_title('Traceplot' if row == 0 else '')
        if row == 0:
            ax_tr.legend(fontsize=7)

        # ACF (primera cadena)
        ax_ac = axes[row, 1]
        s     = chains[0][key] if idx is None else chains[0][key][:, idx]
        s_c   = s - s.mean()
        var   = s_c.var()
        lags  = min(50, len(s) // 4)
        acf   = [1.0] + [float(np.mean(s_c[:len(s)-k] * s_c[k:]) / (var + 1e-15))
                         for k in range(1, lags + 1)]
        ax_ac.bar(range(lags + 1), acf, color=COLORES[0], alpha=0.7, width=0.8)
        ax_ac.axhline(1.96 / np.sqrt(len(s)), color='r', ls='--', lw=0.8)
        ax_ac.axhline(-1.96 / np.sqrt(len(s)), color='r', ls='--', lw=0.8)
        ax_ac.set_ylim(-0.3, 1.05)
        ax_ac.set_title('ACF (cadena 1)' if row == 0 else '')
        ax_ac.set_ylabel(label, fontsize=8)

    axes[-1, 0].set_xlabel('Iteracion')
    axes[-1, 1].set_xlabel('Lag')
    fig.suptitle(f'Diagnosticos MCMC — {model}', y=1.01)
    fig.tight_layout()

    path = os.path.join(out_dir, filename)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_oos_comparison(df_kf, df_bayes, horizons,
                        maturities,
                        out_dir='outputs/figures',
                        filename='fig_oos.png') -> str:
    """
    RMSE por horizonte y plazo — KF vs Bayesiano, barras agrupadas.
    """
    from .forecasting import compute_metrics
    _ensure_dir(out_dir)

    met_kf    = compute_metrics(df_kf,    pb_scale=True)
    met_bayes = compute_metrics(df_bayes, pb_scale=True)

    n_h   = len(horizons)
    fig, axes = plt.subplots(1, n_h, figsize=(4 * n_h, 4), sharey=False)
    if n_h == 1:
        axes = [axes]

    x     = np.arange(len(maturities))
    width = 0.35

    for ax, h in zip(axes, horizons):
        r_kf  = met_kf[met_kf['horizon'] == h].set_index('maturity')['RMSE']
        r_bay = met_bayes[met_bayes['horizon'] == h].set_index('maturity')['RMSE']
        vals_kf  = [r_kf.get(m, np.nan) for m in maturities]
        vals_bay = [r_bay.get(m, np.nan) for m in maturities]

        ax.bar(x - width/2, vals_kf,  width, label='KF',        color=COLORES[0], alpha=0.85)
        ax.bar(x + width/2, vals_bay, width, label='Bayesiano',  color=COLORES[1], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{m:.0f}Y' for m in maturities], fontsize=8)
        ax.set_title(f'h = {h} mes(es)')
        ax.set_ylabel('RMSE (pb)')
        ax.legend(fontsize=8)

    fig.suptitle('Comparacion out-of-sample — RMSE por horizonte y plazo', y=1.02)
    fig.tight_layout()

    path = os.path.join(out_dir, filename)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_posterior_tau(chains, out_dir='outputs/figures',
                       filename='fig_posterior_tau.png') -> str:
    """Densidad posterior de tau1 (y tau2 si DNSS) con líneas por cadena."""
    from scipy.stats import gaussian_kde
    _ensure_dir(out_dir)
    model   = chains[0]['model']
    n_tau   = 2 if model == 'DNSS' else 1
    fig, axes = plt.subplots(1, n_tau, figsize=(6 * n_tau, 4))
    if n_tau == 1:
        axes = [axes]

    for col, key, label in [('tau1', r'$\tau_1$', 0),
                              ('tau2', r'$\tau_2$', 1)][:n_tau]:
        ax   = axes[label]
        samp = np.concatenate([ch[col] for ch in chains])
        kde  = gaussian_kde(samp)
        xr   = np.linspace(samp.min(), samp.max(), 300)
        ax.fill_between(xr, kde(xr), alpha=0.25, color=COLORES[label])
        ax.plot(xr, kde(xr), color=COLORES[label], lw=1.8)
        ax.axvline(float(np.median(samp)), color='k', ls='--', lw=1.0,
                   label=f'Mediana={np.median(samp):.3f}')
        ax.axvline(float(np.percentile(samp, 2.5)), color='gray', ls=':', lw=0.8)
        ax.axvline(float(np.percentile(samp, 97.5)), color='gray', ls=':', lw=0.8,
                   label='IC 95%')
        ax.set_xlabel(col if col == 'tau1' else col)
        ax.set_title(f'Posterior de {col} ({label})')
        ax.set_ylabel('Densidad')
        ax.legend(fontsize=8)

    fig.suptitle(f'Distribuciones posteriores — {model}', y=1.01)
    fig.tight_layout()

    path = os.path.join(out_dir, filename)
    fig.savefig(path)
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# UTILIDADES PARA TABLAS LATEX
# ═══════════════════════════════════════════════════════════════════════════════

def _df_to_latex(df: pd.DataFrame, caption: str, label: str,
                 fmt: dict | None = None) -> str:
    """Convierte un DataFrame a tabla LaTeX con booktabs.

    Escapa caracteres especiales de LaTeX (_, %, &, #, ^, ~) tanto en los
    encabezados de columna como en las celdas de tipo str/object, para
    evitar errores de compilación con nombres como 'JB_pval' o '95%'
    (bug presente en versiones anteriores: un guion bajo sin escapar fuera
    de modo matemático corta la tabla con errores en cascada "Missing $").
    """
    cols   = df.columns.tolist()
    n_cols = len(cols)
    align  = 'l' + 'r' * (n_cols - 1)

    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\small',
        rf'\caption{{{caption}}}',
        rf'\label{{{label}}}',
        rf'\begin{{tabular}}{{{align}}}',
        r'\toprule',
    ]
    # Encabezado (escapado)
    header = ' & '.join(_escape_latex(str(c)) for c in cols) + r' \\'
    lines += [header, r'\midrule']

    # Filas
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            val = row[c]
            if fmt and c in fmt:
                cells.append(fmt[c].format(val))
            elif isinstance(val, float):
                cells.append(f'{val:.4f}')
            elif isinstance(val, (int, np.integer)):
                cells.append(str(val))
            else:
                cells.append(_escape_latex(str(val)))
        lines.append(' & '.join(cells) + r' \\')

    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def _escape_latex(s: str) -> str:
    """Escapa caracteres especiales LaTeX en strings comunes."""
    for ch, rep in [('_', r'\_'), ('%', r'\%'), ('&', r'\&'),
                    ('#', r'\#'), ('^', r'\^{}'), ('~', r'\~{}')]:
        s = s.replace(ch, rep)
    return s


# ═══════════════════════════════════════════════════════════════════════════════
# GENERADOR DEL .TEX
# ═══════════════════════════════════════════════════════════════════════════════

def build_latex_report(config: dict) -> str:
    """
    Genera el archivo .tex completo con todos los resultados.

    config dict keys
    ----------------
    model          : 'DNS' | 'DNSS'
    tau_mode       : 'estatico' | 'ar1' | 'rolling'
    out_dir        : carpeta de salida
    figures        : dict de rutas a figuras (relativas al .tex)
    df_residuals   : pd.DataFrame de residual_stats()
    df_diagnostics : pd.DataFrame de compute_diagnostics()  (Bayesiano)
    df_posterior   : pd.DataFrame de posterior_summary()    (Bayesiano)
    df_comparison  : pd.DataFrame de compare_models()       (OOS)
    mle_params     : dict {mu, F_diag, Q_diag, R_diag, loglik, k, aic, bic}  (KF)
    pf_params      : dict {n_particles, rmse_total, loglik, aic, bic,
                           k_nuevo, ess_mean, resampled_pct}             (PF)
    df_residuals_pf: pd.DataFrame de residual_stats() para el PF
    yields_shape   : (T, N)
    maturities     : np.ndarray
    date_start     : str
    date_end       : str
    horizons       : list[int]
    author         : str  (default: 'Kevin & Juan David')
    """
    model     = config.get('model', 'DNS')
    tau_mode  = config.get('tau_mode', 'estatico')
    out_dir   = config.get('out_dir', 'outputs')
    author    = config.get('author', 'Kevin \\& Juan David')
    figs      = config.get('figures', {})
    T, N      = config.get('yields_shape', (0, 0))
    mats      = config.get('maturities', np.array([]))
    d_start   = config.get('date_start', '')
    d_end     = config.get('date_end', '')
    horizons  = config.get('horizons', [1, 3, 6, 12])
    today     = datetime.today()
    _MESES_ES = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio',
                'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre']
    today_str = f"{today.day} de {_MESES_ES[today.month - 1]} de {today.year}"

    mats_str  = ', '.join(f'{m:.0f}' for m in mats)
    model_full = ('Dynamic Nelson-Siegel (DNS)'
                  if model == 'DNS'
                  else 'Dynamic Nelson-Siegel-Svensson (DNSS)')

    # ── Preámbulo ─────────────────────────────────────────────────────────────
    preamble = textwrap.dedent(r"""
    \documentclass[11pt, a4paper]{article}
    \usepackage[T1]{fontenc}
    \usepackage[utf8]{inputenc}
    \usepackage[spanish]{babel}
    \usepackage{mathpazo}
    \usepackage{microtype}
    \usepackage{amsmath, amssymb, bm}
    \usepackage[margin=2.8cm]{geometry}
    \usepackage{fancyhdr}
    \usepackage{xcolor}
    \usepackage{tcolorbox}
    \usepackage{booktabs}
    \usepackage{graphicx}
    \usepackage{hyperref}
    \usepackage{parskip}
    \usepackage{caption}
    \tcbuselibrary{most}

    \definecolor{azulsuave}{RGB}{70,100,150}
    \definecolor{azulclaro}{RGB}{230,238,250}
    \definecolor{verdeclaro}{RGB}{230,248,236}
    \definecolor{verdeoscuro}{RGB}{40,110,70}
    \definecolor{amaclaro}{RGB}{255,250,225}
    \definecolor{amaoscuro}{RGB}{160,120,20}
    \definecolor{grisclaro}{RGB}{245,245,248}
    \definecolor{grisoscuro}{RGB}{80,80,90}
    \definecolor{rojoclaro}{RGB}{252,235,235}
    \definecolor{rojooscuro}{RGB}{160,40,40}

    \newtcolorbox{resultado}[1][]{
      colback=verdeclaro, colframe=verdeoscuro,
      fonttitle=\bfseries, title={#1}, boxrule=0.5pt,
      left=4pt, right=4pt, top=3pt, bottom=3pt}
    \newtcolorbox{nota}[1][]{
      colback=amaclaro, colframe=amaoscuro,
      fonttitle=\bfseries, title={#1}, boxrule=0.5pt,
      left=4pt, right=4pt, top=3pt, bottom=3pt}
    \newtcolorbox{cuidado}[1][]{
      colback=rojoclaro, colframe=rojooscuro,
      fonttitle=\bfseries, title={#1}, boxrule=0.5pt,
      left=4pt, right=4pt, top=3pt, bottom=3pt}

    \hypersetup{colorlinks=true, linkcolor=azulsuave,
                urlcolor=azulsuave, citecolor=azulsuave}
    \captionsetup{font=small, labelfont=bf}

    \pagestyle{fancy}
    \fancyhf{}
    \fancyhead[L]{\small\color{grisoscuro}\textit{Proyecto KJM --- TES Colombia}}
    \fancyhead[R]{\small\color{grisoscuro}\textit{Curva de Rendimientos """ + model + r"""}}
    \fancyfoot[C]{\thepage}
    \renewcommand{\headrulewidth}{0.4pt}
    """)

    # ── Portada ───────────────────────────────────────────────────────────────
    portada = textwrap.dedent(rf"""
    \begin{{document}}

    \begin{{titlepage}}
    \centering
    \vspace*{{2cm}}
    {{\LARGE\bfseries\color{{azulsuave}}
      Estimaci\'on de la Curva de Rendimientos\\[0.4em]
      de TES Colombia}}\\[1.2em]
    {{\large Modelo: {model_full}}}\\[0.5em]
    {{\large M\'etodo $\tau$: {_escape_latex(tau_mode)}}}\\[2em]
    \textcolor{{azulsuave}}{{\rule{{10cm}}{{0.8pt}}}}\\[1.5em]
    {{\large {author}}}\\[0.5em]
    {{\large Proyecto KJM --- Universidad}}\\[0.5em]
    {{\large {today_str}}}\\[2em]
    \textcolor{{grisoscuro}}{{\small
      Muestra: {d_start} -- {d_end}\\
      Observaciones: {T} d\'ias h\'abiles $\times$ {N} plazos
      ({mats_str} a\~nos)
    }}
    \end{{titlepage}}

    \tableofcontents
    \newpage
    """)

    # ── Sección 1: Descripción de los datos ───────────────────────────────────
    sec_datos = textwrap.dedent(rf"""
    \section{{Descripci\'on de los Datos}}

    La muestra cubre bonos TES colombianos en los plazos
    {mats_str} a\~nos, con observaciones diarias (d\'ias h\'abiles BVC)
    desde {d_start} hasta {d_end}, para un total de $T = {T}$ periodos
    y $N = {N}$ plazos.

    \begin{{nota}}[Convenci\'on de unidades]
    Los rendimientos est\'an expresados en porcentaje anual.
    Los residuos y RMSE se reportan en puntos b\'asicos (pb = \% $\times$ 100).
    \end{{nota}}
    """)

    if 'fig_yield_curves' in figs:
        sec_datos += textwrap.dedent(rf"""
    \begin{{figure}}[htbp]
    \centering
    \includegraphics[width=\linewidth]{{{figs['fig_yield_curves']}}}
    \caption{{Curvas de rendimiento TES seleccionadas (izquierda)
              y series de tiempo por plazo (derecha).}}
    \label{{fig:yield_curves}}
    \end{{figure}}
    """)

    # ── Sección 2: Modelo ─────────────────────────────────────────────────────
    if model == 'DNS':
        eq_H = (r"y_t(m) = \beta_{1t} + \beta_{2t}"
                r"\left(\frac{1-e^{-m/\tau}}{m/\tau}\right)"
                r"+ \beta_{3t}\left(\frac{1-e^{-m/\tau}}{m/\tau}"
                r"- e^{-m/\tau}\right) + \varepsilon_t(m)")
    else:
        eq_H = (r"y_t(m) = \beta_{1t} + \beta_{2t}"
                r"\frac{1-e^{-m/\tau_1}}{m/\tau_1}"
                r"+ \beta_{3t}\left(\frac{1-e^{-m/\tau_1}}{m/\tau_1}"
                r"- e^{-m/\tau_1}\right)"
                r"+ \beta_{4t}\left(\frac{1-e^{-m/\tau_2}}{m/\tau_2}"
                r"- e^{-m/\tau_2}\right) + \varepsilon_t(m)")

    sec_modelo = textwrap.dedent(rf"""
    \section{{Especificaci\'on del Modelo}}

    \subsection{{Ecuaci\'on de Medici\'on}}

    \begin{{equation}}
    {eq_H}
    \end{{equation}}

    donde $\boldsymbol{{\beta}}_t = (\beta_{{1t}}, \beta_{{2t}}, \beta_{{3t}})^\top$
    son los factores latentes de nivel, pendiente y curvatura respectivamente,
    y $\varepsilon_t(m) \sim \mathcal{{N}}(0, \sigma^2_m)$.

    \subsection{{Ecuaci\'on de Transici\'on}}

    \begin{{equation}}
    \boldsymbol{{\beta}}_t - \boldsymbol{{\mu}} =
    \mathbf{{F}}(\boldsymbol{{\beta}}_{{t-1}} - \boldsymbol{{\mu}}) + \boldsymbol{{\eta}}_t,
    \qquad \boldsymbol{{\eta}}_t \sim \mathcal{{N}}(\mathbf{{0}}, \mathbf{{Q}})
    \end{{equation}}

    donde $\boldsymbol{{\mu}}$ es la media incondicional de los factores y
    $\mathbf{{F}}$ es diagonal (Caldeira et al., 2010; \c{{C}}akmaklı, 2013).
    """)

    # ── Sección 3: Resultados KF ──────────────────────────────────────────────
    sec_kf = r'\section{Resultados --- Kalman Filter (Frecuentista)}' + '\n\n'

    mle_p = config.get('mle_params')
    if mle_p:
        mu_v    = np.atleast_1d(mle_p.get('mu', []))
        f_v     = np.atleast_1d(mle_p.get('F_diag', []))
        q_v     = np.atleast_1d(mle_p.get('Q_diag', []))
        r_v     = np.atleast_1d(mle_p.get('R_diag', []))
        ll_val  = mle_p.get('loglik', float('nan'))
        k       = len(mu_v)
        bnames  = [r'$\beta_1$', r'$\beta_2$', r'$\beta_3$', r'$\beta_4$'][:k]

        rows_mle = []
        for i in range(k):
            rows_mle.append({
                'Factor'    : bnames[i],
                r'$\mu_i$'  : f'{mu_v[i]:.4f}',
                r'$f_i$'    : f'{f_v[i]:.4f}',
                r'$\sqrt{q_i}$ (pb)': f'{np.sqrt(q_v[i])*100:.4f}',
            })
        df_mle = pd.DataFrame(rows_mle)

        sec_kf += textwrap.dedent(rf"""
    \subsection{{Par\'ametros MLE}}

    Log-verosimilitud marginal (KF): $\hat{{\ell}} = {ll_val:,.2f}$.
    """)
        if 'aic' in mle_p and 'bic' in mle_p:
            sec_kf += textwrap.dedent(rf"""
    AIC $= {mle_p['aic']:,.2f}$, \quad BIC $= {mle_p['bic']:,.2f}$
    \quad ($k = {mle_p.get('k', '?')}$ par\'ametros libres).
    """)
        sec_kf += textwrap.dedent(rf"""
    {_df_to_latex(df_mle,
                  'Parametros MLE: media incondicional, persistencia y ruido de transicion.',
                  'tab:mle_params')}

    \medskip
    \noindent Desviaciones est\'andar de medicion $\sqrt{{R_{{mm}}}}$ (pb):
    {', '.join(f'{np.sqrt(r)*100:.3f}' for r in r_v)}.
    """)

    if 'fig_tau' in figs:
        sec_kf += textwrap.dedent(rf"""
    \subsection{{Evoluci\'on de $\tau$}}

    \begin{{figure}}[htbp]
    \centering
    \includegraphics[width=\linewidth]{{{figs['fig_tau']}}}
    \caption{{Evoluci\'on temporal del par\'ametro de forma $\tau$
              estimado via grid search adaptativo.}}
    \label{{fig:tau}}
    \end{{figure}}
    """)

    if 'fig_factors' in figs:
        sec_kf += textwrap.dedent(rf"""
    \subsection{{Factores Latentes (Estados Suavizados)}}

    \begin{{figure}}[htbp]
    \centering
    \includegraphics[width=\linewidth]{{{figs['fig_factors']}}}
    \caption{{Factores latentes estimados via RTS Smoother.
              $\beta_1$: nivel, $\beta_2$: pendiente, $\beta_3$: curvatura.}}
    \label{{fig:factors}}
    \end{{figure}}
    """)

    # ── Sección 4: Diagnóstico de residuos ────────────────────────────────────
    sec_resid = r'\section{Diagn\'ostico de Residuos}' + '\n\n'

    df_res = config.get('df_residuals')
    if df_res is not None:
        sec_resid += _df_to_latex(
            df_res,
            'Estadisticos de residuos de medicion por plazo (pb).',
            'tab:residuals') + '\n\n'

    if 'fig_residuals' in figs:
        sec_resid += textwrap.dedent(rf"""
    \begin{{figure}}[htbp]
    \centering
    \includegraphics[width=\linewidth]{{{figs['fig_residuals']}}}
    \caption{{Diagn\'ostico de residuos: series temporales (fila 1),
              histogramas con curva normal (fila 2), QQ-plots (fila 3).}}
    \label{{fig:residuals}}
    \end{{figure}}
    """)

    if 'fig_residuals_heatmap' in figs:
        sec_resid += textwrap.dedent(rf"""
    \begin{{figure}}[htbp]
    \centering
    \includegraphics[width=\linewidth]{{{figs['fig_residuals_heatmap']}}}
    \caption{{Mapa de calor de residuos de medici\'on: plazos (eje vertical)
              vs.\ tiempo (eje horizontal). Rojo indica que el modelo
              subestima el rendimiento observado; azul, que lo sobreestima.
              \'Util para detectar tramos o periodos donde el error se
              concentra sistem\'aticamente (p.\,ej.\ quiebres estructurales).}}
    \label{{fig:residuals_heatmap}}
    \end{{figure}}
    """)

    # ── Sección 5: Resultados Bayesianos ──────────────────────────────────────
    sec_bayes = ''
    df_diag   = config.get('df_diagnostics')
    df_post   = config.get('df_posterior')

    if df_diag is not None or df_post is not None:
        sec_bayes = r'\section{Resultados --- Gibbs Sampler (Bayesiano)}' + '\n\n'

        if df_diag is not None:
            sec_bayes += r'\subsection{Convergencia MCMC (R-hat y ESS)}' + '\n\n'
            sec_bayes += _df_to_latex(
                df_diag,
                'Diagnosticos de convergencia MCMC. R-hat $< 1.05$ indica convergencia.',
                'tab:diagnostics') + '\n\n'

        if 'fig_mcmc' in figs:
            sec_bayes += textwrap.dedent(rf"""
    \begin{{figure}}[htbp]
    \centering
    \includegraphics[width=\linewidth]{{{figs['fig_mcmc']}}}
    \caption{{Traceplots (izquierda) y ACF (derecha) de los par\'ametros
              principales. L\'ineas punteadas: banda de significancia al 5\%.}}
    \label{{fig:mcmc}}
    \end{{figure}}
    """)

        if df_post is not None:
            sec_bayes += r'\subsection{Distribuciones Posteriores}' + '\n\n'
            sec_bayes += _df_to_latex(
                df_post,
                'Resumen de la distribucion posterior: media, std e IC 95\\%.',
                'tab:posterior') + '\n\n'

        if 'fig_posterior_tau' in figs:
            sec_bayes += textwrap.dedent(rf"""
    \begin{{figure}}[htbp]
    \centering
    \includegraphics[width=0.8\linewidth]{{{figs['fig_posterior_tau']}}}
    \caption{{Densidades posteriores del par\'ametro de forma $\tau$.
              L\'inea discontinua: mediana. Punteadas: IC 95\%.}}
    \label{{fig:posterior_tau}}
    \end{{figure}}
    """)

    # ── Sección PF: Resultados Particle Filter ────────────────────────────────
    sec_pf = ''
    pf_p   = config.get('pf_params')

    if pf_p is not None:
        sec_pf = r'\section{Resultados --- Particle Filter ($\tau_t$ Din\'amico)}' + '\n\n'
        sec_pf += textwrap.dedent(rf"""
    A diferencia del Kalman Filter (donde $\tau_t$ es un insumo fijo
    proveniente del grid search adaptativo) y del Gibbs sampler (donde
    $\tau$ es est\'atico o sigue un AR(1) latente muestreado v\'ia
    Metropolis-Hastings), aqu\'i $\tau_t$ se modela como parte del
    \textbf{{estado aumentado}} $x_t = (\boldsymbol{{\beta}}_t, \log\tau_t)$,
    evolucionando como un AR(1) en escala logar\'itmica. Esto vuelve la
    ecuaci\'on de medici\'on no lineal en el estado, lo que motiva un
    \textit{{Particle Filter}} (Gordon, Salmond \& Smith, 1993) en vez de un
    Extended Kalman Filter.

    \begin{{nota}}[Advertencia metodol\'ogica]
    La log-verosimilitud del PF es una \textbf{{estimaci\'on Monte Carlo}}
    con varianza finita en $N$ (n\'umero de part\'iculas) -- no es un valor
    exacto como en el KF. El AIC/BIC aqu\'i reportados heredan esa varianza;
    para una conclusi\'on robusta conviene recalcular con 3--5 semillas
    distintas y reportar el rango.
    \end{{nota}}

    \subsection{{Ajuste y Criterios de Informaci\'on}}

    \begin{{itemize}}
    \item N\'umero de part\'iculas: $N = {pf_p.get('n_particles', '?')}$
    \item RMSE total: ${pf_p.get('rmse_total', float('nan'))*100:.4f}$ pb
    \item Log-verosimilitud (estimaci\'on SMC): $\hat{{\ell}}_{{PF}} = {pf_p.get('loglik', float('nan')):,.2f}$
    \item AIC $= {pf_p.get('aic', float('nan')):,.2f}$, BIC $= {pf_p.get('bic', float('nan')):,.2f}$
          \quad ($k = {pf_p.get('k_nuevo', '?')}$ -- solo los hiperpar\'ametros
          \textit{{nuevos}} de la din\'amica de $\tau$; ver nota abajo)
    \item ESS/N promedio: ${pf_p.get('ess_mean', float('nan')):.3f}$
          \quad Remuestreos: ${pf_p.get('resampled_pct', float('nan'))*100:.1f}\%$ de los periodos
    \end{{itemize}}

    \begin{{nota}}[Conteo de $k$ para AIC/BIC]
    El AIC/BIC del PF aqu\'i reportado cuenta \'unicamente
    $k=3$ por cada $\tau$ ($\mu_{{\log\tau}}$, $\phi_{{\log\tau}}$,
    $q_{{\log\tau}}$) -- los par\'ametros que el PF efectivamente
    \textit{{a\~nade}} sobre el warm-start del KF (cuyos $\mu, F, Q, R$
    no se reestiman). No es directamente comparable, bajo la misma
    convenci\'on de conteo, al AIC/BIC del KF de la Secci\'on anterior
    salvo que se re-cuente $k$ incluyendo todos los par\'ametros del
    sistema heredados del warm-start.
    \end{{nota}}
    """)

        if 'fig_tau_pf' in figs:
            sec_pf += textwrap.dedent(rf"""
    \subsection{{Evoluci\'on Din\'amica de $\tau_t$}}

    \begin{{figure}}[htbp]
    \centering
    \includegraphics[width=\linewidth]{{{figs['fig_tau_pf']}}}
    \caption{{Evoluci\'on de $\tau_t$ estimada por el Particle Filter
              (l\'inea s\'olida, con banda de confianza al 95\% sobre la
              nube de part\'iculas), comparada contra la serie del grid
              search adaptativo del Kalman Filter (l\'inea discontinua gris).}}
    \label{{fig:tau_pf}}
    \end{{figure}}
    """)

        if 'fig_factors_pf' in figs:
            sec_pf += textwrap.dedent(rf"""
    \subsection{{Factores Latentes (Filtrados, PF)}}

    \begin{{figure}}[htbp]
    \centering
    \includegraphics[width=\linewidth]{{{figs['fig_factors_pf']}}}
    \caption{{Factores latentes filtrados por el Particle Filter (media
              ponderada de part\'iculas, con banda de confianza al 95\%).
              A diferencia de los factores del KF (Secci\'on anterior,
              estados \textit{{suavizados}} v\'ia RTS), estos son estados
              \textit{{filtrados}} $\hat{{\boldsymbol{{\beta}}}}_{{t|t}}$ --
              el PF no implementa a\'un un \textit{{particle smoother}}.}}
    \label{{fig:factors_pf}}
    \end{{figure}}
    """)

        if 'fig_residuals_heatmap_pf' in figs:
            sec_pf += textwrap.dedent(rf"""
    \subsection{{Residuos de Medici\'on (PF)}}

    \begin{{figure}}[htbp]
    \centering
    \includegraphics[width=\linewidth]{{{figs['fig_residuals_heatmap_pf']}}}
    \caption{{Mapa de calor de residuos de medici\'on del Particle Filter.
              Misma convenci\'on de color que la Secci\'on de diagn\'ostico
              de residuos del KF.}}
    \label{{fig:residuals_heatmap_pf}}
    \end{{figure}}
    """)

        df_res_pf = config.get('df_residuals_pf')
        if df_res_pf is not None:
            sec_pf += _df_to_latex(
                df_res_pf,
                'Estadisticos de residuos de medicion por plazo, Particle Filter (pb).',
                'tab:residuals_pf') + '\n\n'

        if 'fig_pf_diagnostics' in figs:
            sec_pf += textwrap.dedent(rf"""
    \subsection{{Diagn\'ostico de Degeneraci\'on de Part\'iculas}}

    \begin{{figure}}[htbp]
    \centering
    \includegraphics[width=\linewidth]{{{figs['fig_pf_diagnostics']}}}
    \caption{{Effective Sample Size (ESS) normalizado por $N$ a lo largo
              del tiempo (panel superior) y periodos en los que se
              dispar\'o el remuestreo sistem\'atico (panel inferior).
              ESS/N bajo indica degeneraci\'on de pesos.}}
    \label{{fig:pf_diagnostics}}
    \end{{figure}}
    """)

    # ── Sección 6: Comparación OOS ────────────────────────────────────────────
    sec_oos = ''
    df_cmp  = config.get('df_comparison')

    if df_cmp is not None:
        sec_oos = textwrap.dedent(r"""
    \section{Comparaci\'on Out-of-Sample}

    \begin{nota}[Nota metodol\'ogica]
    El pron\'ostico usa estados predichos $\hat{\boldsymbol{\beta}}_{t|t-1}$,
    no estados suavizados $\hat{\boldsymbol{\beta}}_{t|T}$, para garantizar
    que no se incorpora informaci\'on futura en la evaluaci\'on.
    \end{nota}

    """)
        sec_oos += _df_to_latex(
            df_cmp,
            f'Comparacion OOS: RMSE promedio (pb) por horizonte. '
            f'Horizontes h = {", ".join(str(h) for h in horizons)} mes(es).',
            'tab:oos_comparison') + '\n\n'

        if 'fig_oos' in figs:
            sec_oos += textwrap.dedent(rf"""
    \begin{{figure}}[htbp]
    \centering
    \includegraphics[width=\linewidth]{{{figs['fig_oos']}}}
    \caption{{RMSE out-of-sample (pb) por horizonte y plazo.
              KF frecuentista vs.\ Gibbs bayesiano.}}
    \label{{fig:oos}}
    \end{{figure}}
    """)

    # ── Sección 7: Referencias ────────────────────────────────────────────────
    sec_refs = textwrap.dedent(r"""
    \section{Referencias}

    \begin{itemize}
    \item Nelson, C.\ \& Siegel, A.\ (1987). Parsimonious modeling of yield curves.
          \textit{Journal of Business}, 60(4), 473--489.
    \item Svensson, L.\ (1994). Estimating and interpreting forward interest rates.
          \textit{NBER Working Paper}, 4871.
    \item Diebold, F.\ \& Li, C.\ (2006). Forecasting the term structure of
          government bond yields.
          \textit{Journal of Econometrics}, 130(2), 337--364.
    \item Caldeira, J., Laurini, M.\ \& Portugal, M.\ (2010).
          Bayesian inference applied to dynamic Nelson-Siegel model.
          \textit{Brazilian Review of Econometrics}.
    \item \c{C}akmaklı, C.\ (2013). Bayesian semiparametric dynamic Nelson-Siegel model.
          \textit{Working Paper}.
    \item Carter, C.\ \& Kohn, R.\ (1994). On Gibbs sampling for state space models.
          \textit{Biometrika}, 81(3), 541--553.
    \item Hamilton, J.\ (1994). \textit{Time Series Analysis}. Princeton University Press.
    \end{itemize}

    \end{document}
    """)

    # ── Ensamblar ─────────────────────────────────────────────────────────────
    content = (preamble + portada + sec_datos + sec_modelo +
               sec_kf + sec_resid + sec_bayes + sec_pf + sec_oos + sec_refs)

    tex_path = os.path.join(out_dir, 'reporte_dns.tex')
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(content)

    return tex_path


# ═══════════════════════════════════════════════════════════════════════════════
# COMPILACIÓN PDF
# ═══════════════════════════════════════════════════════════════════════════════

def compile_pdf(tex_path: str, n_runs: int = 2) -> str | None:
    """
    Compila el .tex con pdflatex (dos pasadas para TOC).

    Retorna la ruta al PDF o None si falló.
    Requiere pdflatex en el PATH (MiKTeX en Windows, TeX Live en Linux/Mac).
    """
    tex_dir  = os.path.dirname(os.path.abspath(tex_path))
    tex_file = os.path.basename(tex_path)
    pdf_path = tex_path.replace('.tex', '.pdf')

    cmd = ['pdflatex', '-interaction=nonstopmode', tex_file]

    for run in range(n_runs):
        result = subprocess.run(
            cmd, cwd=tex_dir,
            capture_output=True, text=True,
            encoding='utf-8', errors='replace')
        if result.returncode != 0 and run == n_runs - 1:
            print(f"\n  [pdflatex] Error en compilación (run {run+1}):")
            # Mostrar solo las líneas de error relevantes
            for line in result.stdout.split('\n'):
                if line.startswith('!') or 'Error' in line:
                    print(f"    {line}")
            return None

    if os.path.exists(pdf_path):
        # Limpiar auxiliares
        for ext in ['.aux', '.log', '.toc', '.out']:
            aux = tex_path.replace('.tex', ext)
            if os.path.exists(aux):
                os.remove(aux)
        return pdf_path
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def generate_full_report(
    yields, dates, maturities, model, tau_mode,
    out_dir       = 'outputs',
    # KF
    beta_smooth   = None,
    P_smooth      = None,
    tau1_t        = None,
    tau2_t        = None,
    mle_params    = None,
    residuals_kf  = None,
    # Particle Filter
    res_pf        = None,
    # Bayesiano
    chains        = None,
    df_diagnostics= None,
    df_posterior  = None,
    # OOS
    df_kf         = None,
    df_bayes      = None,
    horizons      = None,
    # Meta
    author        = 'Kevin \\& Juan David',
) -> str | None:
    """
    Función de alto nivel: genera figuras, .tex y compila el PDF.

    Llámala al final de run_kalman.py, run_bayesian.py, run_particlefilter.py
    o run_comparison.py con los objetos ya disponibles en memoria.

    Parámetros nuevos
    ------------------
    P_smooth : (T, k, k) | None
        Covarianza suavizada del KF (rts_smoother). Si se provee junto con
        beta_smooth, plot_factors() dibuja bandas de confianza al 95% sobre
        los factores latentes (sqrt de la diagonal de P_smooth).
    res_pf   : dict | None
        Resultado del Particle Filter, tal como lo devuelve
        particle_filter.fit_pf_model()['pf'] (o run_particle_filter()
        directamente). Si se provee, se genera una sección completa de
        resultados PF: evolución de tau con bandas (comparada contra
        tau1_t/tau2_t del KF si también se proveen), factores filtrados
        con bandas, heatmap de residuos, diagnóstico de ESS/remuestreo,
        tabla de residuos y AIC/BIC.

    Retorna la ruta al PDF generado, o None si pdflatex no está disponible.
    """
    if horizons is None:
        horizons = [1, 3, 6, 12]

    fig_dir = os.path.join(out_dir, 'figures')
    figs    = {}

    print("\n  Generando figuras…")

    # Curvas de rendimiento (siempre)
    figs['fig_yield_curves'] = os.path.relpath(
        plot_yield_curve(yields, dates, maturities, out_dir=fig_dir),
        out_dir).replace('\\', '/')

    # Tau (KF)
    if tau1_t is not None:
        figs['fig_tau'] = os.path.relpath(
            plot_tau(tau1_t, dates, tau2_t, model, out_dir=fig_dir),
            out_dir).replace('\\', '/')

    # Factores latentes (KF smoother), con banda de confianza si hay P_smooth
    if beta_smooth is not None:
        beta_std_kf = None
        if P_smooth is not None:
            k_dim = beta_smooth.shape[1]
            beta_std_kf = np.sqrt(np.stack(
                [P_smooth[:, i, i] for i in range(k_dim)], axis=1))
        figs['fig_factors'] = os.path.relpath(
            plot_factors(beta_smooth, dates, model, beta_std=beta_std_kf,
                         out_dir=fig_dir),
            out_dir).replace('\\', '/')

    # Residuos (KF) — series/histograma/QQ + heatmap plazos vs tiempo
    if residuals_kf is not None:
        figs['fig_residuals'] = os.path.relpath(
            plot_residuals(residuals_kf, maturities, dates, out_dir=fig_dir),
            out_dir).replace('\\', '/')
        figs['fig_residuals_heatmap'] = os.path.relpath(
            plot_residuals_heatmap(residuals_kf, maturities, dates, out_dir=fig_dir),
            out_dir).replace('\\', '/')

    # MCMC
    if chains is not None:
        figs['fig_mcmc'] = os.path.relpath(
            plot_diagnostics_mcmc(chains, out_dir=fig_dir),
            out_dir).replace('\\', '/')
        figs['fig_posterior_tau'] = os.path.relpath(
            plot_posterior_tau(chains, out_dir=fig_dir),
            out_dir).replace('\\', '/')

    # ── Particle Filter ───────────────────────────────────────────────────────
    pf_params      = None
    df_residuals_pf = None

    if res_pf is not None:
        n_beta = 3 if res_pf['model'] == 'DNS' else 4
        x_filt_mean, x_filt_std = res_pf['x_filt_mean'], res_pf['x_filt_std']
        pf_dates = res_pf.get('dates', dates)

        # Tau con banda de confianza (percentil log-normal vía std en log-escala)
        ltau1_m, ltau1_s = x_filt_mean[:, n_beta], x_filt_std[:, n_beta]
        tau1_band = (np.exp(ltau1_m - 1.96 * ltau1_s), np.exp(ltau1_m + 1.96 * ltau1_s))
        tau2_band = None
        tau2_pf = None
        if res_pf['model'] == 'DNSS':
            ltau2_m, ltau2_s = x_filt_mean[:, n_beta + 1], x_filt_std[:, n_beta + 1]
            tau2_band = (np.exp(ltau2_m - 1.96 * ltau2_s), np.exp(ltau2_m + 1.96 * ltau2_s))
            tau2_pf = res_pf['tau2_t']

        figs['fig_tau_pf'] = os.path.relpath(
            plot_tau(res_pf['tau1_t'], pf_dates, tau2_pf, res_pf['model'],
                     tau1_band=tau1_band, tau2_band=tau2_band,
                     tau1_compare=tau1_t, tau2_compare=tau2_t,
                     label_compare='KF (grid search)',
                     out_dir=fig_dir, filename='fig_tau_pf.png'),
            out_dir).replace('\\', '/')

        figs['fig_factors_pf'] = os.path.relpath(
            plot_factors(x_filt_mean[:, :n_beta], pf_dates, res_pf['model'],
                        beta_std=x_filt_std[:, :n_beta],
                        out_dir=fig_dir, filename='fig_factors_pf.png'),
            out_dir).replace('\\', '/')

        figs['fig_residuals_heatmap_pf'] = os.path.relpath(
            plot_residuals_heatmap(res_pf['residuals'], maturities, pf_dates,
                                   out_dir=fig_dir, filename='fig_residuals_heatmap_pf.png'),
            out_dir).replace('\\', '/')

        figs['fig_pf_diagnostics'] = os.path.relpath(
            plot_diagnostics_pf(res_pf, out_dir=fig_dir),
            out_dir).replace('\\', '/')

        from .diagnostics import residual_stats
        df_residuals_pf = residual_stats(res_pf['residuals'], maturities, pb_scale=True)

        pf_params = dict(
            n_particles=res_pf['n_particles'], rmse_total=res_pf['rmse_total'],
            loglik=res_pf['loglik'], aic=res_pf['aic'], bic=res_pf['bic'],
            k_nuevo=res_pf['k_nuevo'],
            ess_mean=float(res_pf['ess_t'].mean() / res_pf['n_particles']),
            resampled_pct=float(res_pf['resampled_t'].mean()),
        )

    # OOS
    if df_kf is not None and df_bayes is not None:
        figs['fig_oos'] = os.path.relpath(
            plot_oos_comparison(df_kf, df_bayes, horizons, maturities,
                                out_dir=fig_dir),
            out_dir).replace('\\', '/')

    # Tablas de comparación OOS
    df_comparison = None
    if df_kf is not None and df_bayes is not None:
        from .forecasting import compare_models
        df_comparison = compare_models(df_kf, df_bayes, pb_scale=True)

    # Estadísticos de residuos (KF)
    df_residuals = None
    if residuals_kf is not None:
        from .diagnostics import residual_stats
        df_residuals = residual_stats(residuals_kf, maturities, pb_scale=True)

    print("  Generando .tex…")
    tex_path = build_latex_report({
        'model'         : model,
        'tau_mode'      : tau_mode,
        'out_dir'       : out_dir,
        'figures'       : figs,
        'df_residuals'  : df_residuals,
        'df_residuals_pf': df_residuals_pf,
        'df_diagnostics': df_diagnostics,
        'df_posterior'  : df_posterior,
        'df_comparison' : df_comparison,
        'mle_params'    : mle_params,
        'pf_params'     : pf_params,
        'yields_shape'  : yields.shape,
        'maturities'    : maturities,
        'date_start'    : str(dates[0])[:10],
        'date_end'      : str(dates[-1])[:10],
        'horizons'      : horizons,
        'author'        : author,
    })
    print(f"  .tex guardado: {tex_path}")

    print("  Compilando PDF con pdflatex…")
    pdf_path = compile_pdf(tex_path)

    if pdf_path:
        print(f"  PDF generado: {pdf_path}")
    else:
        print("  pdflatex no disponible o error. El .tex está listo para compilar manualmente.")

    return pdf_path
