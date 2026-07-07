# Comparación entre enfoques frecuentista y bayesiano en modelos Nelson-Siegel dinámicos para TES en Colombia

### Comparing frequentist and bayesian approaches in dynamic Nelson-Siegel models for Colombian TES

**Autores:**
* **Kevin H. Gonzalez-Duarte** (Estudiante Estadística & Ing. Sistemas, UNAL - Med, kgonzalezd@unal.edu.co)
* **Juan D. Garro-Arboleda** (Estudiante Ingeniería Administrativa & Estadística, UNAL - Med, jgarro@unal.edu.co)
* **Monica A. Arango-Arango** (Profesora Asociada, UNAL - Med, marango@unal.edu.co)
* **René Iral-Palomino** (Profesor Asociado, UNAL - Med, riral@unal.edu.co)
* **Julián Saavedra-Echavarría** (Estudiante de Maestría en Analítica, UNAL - Med, jsaavedrae@unal.edu.co)

---

## Resumen

Los Títulos de Tesorería (TES) constituyen la base del mercado de capitales colombiano y son el principal referente de tasas libres de riesgo. Modelar adecuadamente su curva de rendimientos es fundamental para la valoración de activos y la gestión de riesgos. Entre los enfoques más utilizados se encuentran los modelos de la familia Nelson-Siegel; sin embargo, la mayoría de aplicaciones en la literatura asume que el parámetro de decaimiento ($\tau$), que controla la forma de la curva, permanece constante en el tiempo. Esta simplificación ignora la naturaleza dinámica de los mercados financieros y puede limitar la capacidad de ajuste de los modelos.

Para cubrir esta brecha empírica, este estudio contrasta la estimación de la curva cero-cupón mediante modelos de espacio de estados, evaluando metodologías frecuentistas frente a estimaciones bayesianas puras. Se construye un panel histórico de tasas spot mediante el método de bootstrapping de Fama-Bliss en seis nodos de vencimiento (entre 2 y 15 años). Sobre este panel, se extraen los factores latentes aplicando los modelos Dinámico de Nelson-Siegel (DNS) y Nelson-Siegel-Svensson (DNSS) bajo dos paradigmas: (1) un enfoque frecuentista basado en el Filtro de Kalman, optimizando los parámetros $\tau$ a través de una búsqueda de grilla en ventana rodante, y (2) un enfoque Bayesiano mediante el algoritmo de Muestreo de Gibbs, que estima conjuntamente los factores $\beta$ y $\tau$. El desempeño de las metodologías se evalúa a través de métricas de bondad de ajuste. La hipótesis central sugiere que la estimación dinámica de $\tau$ reduce significativamente los errores de ajuste. Se identifica que el enfoque Bayesiano ofrezca una ventaja al capturar mejor la incertidumbre durante periodos volátiles, mientras que el Filtro de Kalman destaca por su eficiencia computacional y ajuste competitivo. Este estudio aporta evidencia empírica a favor de los modelos dinámicos, proveyendo herramientas robustas para la valoración de activos, gestión de riesgos y toma de decisiones en Colombia.

**Palabras clave:** Renta fija, Curva de rendimientos, Nelson-Siegel, Modelos dinámicos, Decaimiento dinámico.

---

## Abstract

The Treasury Bonds (TES) are the basis of the Colombian capital market and are the main reference for risk-free rates. Properly modeling their yield curve is fundamental for asset valuation and risk management. Among the most widely used approaches are the models of the Nelson-Siegel family; however, most applications in the literature assume that the decay parameter ($\tau$), which controls the shape of the curve, remains constant over time. This simplification ignores the dynamic nature of financial markets and may limit the models' fitting ability.

To fill this empirical gap, this study contrasts the estimation of the zero-coupon curve through state-space models, evaluating frequentist methodologies against pure Bayesian estimates. A historical panel of spot rates is constructed using the Fama-Bliss bootstrapping method at six maturity nodes (between 2 and 15 years). On this panel, latent factors are extracted by applying the Dynamic Nelson-Siegel (DNS) and Nelson-Siegel-Svensson (DNSS) models under two paradigms: (1) a frequentist approach based on the Kalman Filter, optimizing the $\tau$ parameters through a grid search in a rolling window, and (2) a Bayesian approach using the Gibbs Sampling algorithm, which jointly estimates the $\beta$ and $\tau$ factors. The performance of the methodologies is evaluated through goodness-of-fit metrics. The central hypothesis suggests that the dynamic estimation of $\tau$ significantly reduces fitting errors. It is identified that the Bayesian approach will offer an advantage in better capturing uncertainty during volatile periods, while the Kalman Filter stands out for its computational efficiency and competitive fit. This study provides empirical evidence in favor of dynamic models, providing robust tools for asset valuation, risk management, and decision-making in Colombia.

**Keywords:** Fixed income, Yield curve, Nelson-Siegel, Dynamic models, Dynamic decay.

---

## Referencias

* Dhayalkar, S. R. (2025). Particle filter made simple: A step-by-step beginner-friendly guide. *arXiv*.
* Diebold, F. X., & Li, C. (2006). Forecasting the term structure of government bond yields. *Journal of Econometrics*, 130, 337-364.
* Fama, E. F., & Bliss, R. R. (1987). The information in long-maturity forward rates. *The American Economic Review*, 77(4), 680-692.
* Nelson, C. R., & Siegel, A. F. (1987). Parsimonious modeling of yield curves. *The Journal of Business*, 60(4), 473-489.
* Svensson, L. E. O. (1995). Estimating forward interest rates with the extended Nelson & Siegel method. *Penning- & Valutapolitik*, 3, 13-26.

---

## Tareas y Ejecución

| fecha | b0 | b1 | b2 | b3 | tao1 | tao2 | res_2y | res_3y | res_5y | res_7y | res_10y | res_15y |
|-------|----|----|----|----|------|------|--------|--------|--------|--------|---------|---------|

Para iniciar la consola del notebook:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned; .\Open-NotebookConsole.ps1
```