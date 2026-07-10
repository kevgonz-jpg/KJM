"""
src — Paquete de estimación DNS/DNSS para TES colombianos
==========================================================
Módulos disponibles:

    nelson_siegel  — cargas NS/NSS, get_H, inicialización OLS de τ
    kalman         — Filtro de Kalman, RTS Smoother, MLE multi-start
    grid_search    — Grid search adaptativo de τ (ventana rodante/expandiente)
    gibbs          — Gibbs sampler Bayesiano con Carter-Kohn FFBS
    particle_filter— Bootstrap Particle Filter con τ_t dinámico (estado aumentado)
    diagnostics    — R-hat, ESS, Ljung-Box, Jarque-Bera, AIC/BIC, tabla de residuos
    forecasting    — Pronóstico OOS, ventana expandiente, RMSE/MAE
    data_loader    — Carga TES desde CSV, índice de días hábiles
"""

from .nelson_siegel import ns_loadings, nss_loadings, get_H, ols_init_tau
from .kalman        import (kalman_filter, rts_smoother,
                            compute_P0_stationary, build_H_array,
                            mle_estimate, build_R_groups)
from .grid_search   import rolling_tau_dns, rolling_tau_dnss
from .gibbs         import (ffbs, kf_loglik,
                            sample_F_mu_diagonal, sample_Q, sample_R, sample_R_groups,
                            make_priors, make_priors_ar1,
                            initialize_chain, run_chain,
                            run_chains_parallel, run_rolling_bayes)
from .particle_filter import (calibrate_tau_ar1, predicted_yields,
                            propagate_beta, propagate_tau, enforce_tau_order,
                            gaussian_loglik, student_t_loglik, systematic_resample,
                            run_particle_filter, fit_model_kf_cached, fit_pf_model)
from .diagnostics   import (gelman_rubin, effective_sample_size,
                            compute_diagnostics, print_diagnostics,
                            ljung_box_pval, jarque_bera_pval, residual_stats,
                            aic_bic)
from .forecasting   import (forecast_kf, forecast_bayesian,
                            expanding_window_kf, expanding_window_bayesian,
                            compute_metrics, summary_metrics, compare_models)
from .data_loader   import load_tes_data, describe_data
from .report        import (generate_full_report, build_latex_report, compile_pdf,
                            plot_yield_curve, plot_factors, plot_tau,
                            plot_residuals, plot_residuals_heatmap,
                            plot_diagnostics_mcmc, plot_diagnostics_pf,
                            plot_oos_comparison, plot_posterior_tau)

__all__ = [
    # nelson_siegel
    'ns_loadings', 'nss_loadings', 'get_H', 'ols_init_tau',
    # kalman
    'kalman_filter', 'rts_smoother', 'compute_P0_stationary',
    'build_H_array', 'mle_estimate', 'build_R_groups',
    # grid_search
    'rolling_tau_dns', 'rolling_tau_dnss',
    # gibbs
    'ffbs', 'kf_loglik',
    'sample_F_mu_diagonal', 'sample_Q', 'sample_R', 'sample_R_groups',
    'make_priors', 'make_priors_ar1',
    'initialize_chain', 'run_chain', 'run_chains_parallel', 'run_rolling_bayes',
    # particle_filter
    'calibrate_tau_ar1', 'predicted_yields',
    'propagate_beta', 'propagate_tau', 'enforce_tau_order',
    'gaussian_loglik', 'student_t_loglik', 'systematic_resample',
    'run_particle_filter', 'fit_model_kf_cached', 'fit_pf_model',
    # diagnostics
    'gelman_rubin', 'effective_sample_size',
    'compute_diagnostics', 'print_diagnostics',
    'ljung_box_pval', 'jarque_bera_pval', 'residual_stats', 'aic_bic',
    # forecasting
    'forecast_kf', 'forecast_bayesian',
    'expanding_window_kf', 'expanding_window_bayesian',
    'compute_metrics', 'summary_metrics', 'compare_models',
    # data_loader
    'load_tes_data', 'describe_data',
    # report
    'generate_full_report', 'build_latex_report', 'compile_pdf',
    'plot_yield_curve', 'plot_factors', 'plot_tau',
    'plot_residuals', 'plot_residuals_heatmap',
    'plot_diagnostics_mcmc', 'plot_diagnostics_pf',
    'plot_oos_comparison', 'plot_posterior_tau',
]
