# Contexto de Conversación Inicial — Proyecto TES Curvas de Rendimiento
### Documento de continuidad para nuevas conversaciones con Claude

---

## propósito de este documento

Este archivo captura todo el razonamiento, decisiones técnicas, correcciones y acuerdos
tomados durante la conversación de diseño inicial del proyecto. Debe cargarse como
contexto en cualquier conversación nueva para que Claude pueda retomar sin pérdida
de información.

---

## 1. Quién es el usuario y qué está construyendo

Juan David Garro es estudiante universitario (semestre 11) trabajando en un proyecto
de semillero de investigación. El objetivo es construir un algoritmo que genere un
panel histórico completo de curvas de rendimiento cero-cupón para los bonos soberanos
TES colombianos (TFIT) negociados en la BVC entre 2015 y 2026, usando el método de
bootstrapping de Fama-Bliss. El destino final de las curvas es alimentar modelos
econométricos o de machine learning.

---

## 2. Decisiones de configuración confirmadas

| Parámetro | Decisión | Justificación |
|---|---|---|
| Entorno de ejecución | VSCode local + Windows 11 | Mayor control, mejor DX, arquitectura modular más natural |
| Convención matemática | Composición discreta anual | Convención nativa del mercado colombiano TES |
| Base de días | 365 | Confirmado para todos los instrumentos |
| Frecuencia de cupón | Anual vencido | Uniforme para todos los bonos TFIT |
| Umbral de estabilidad | 30 días | Balance entre estabilidad numérica y cobertura de bonos |
| Nodos de interpolación | 1, 2, 3, 5, 7, 10, 15, 20, 30 años | Nodos estándar curva TES colombiana |
| Formato panel final | Wide | Una fila por fecha, una columna por nodo — óptimo para ML |
| Fecha de validación | Modular vía `config.py` — valor inicial `'2016-10-21'` | Existe scatter plot de referencia para esa fecha en el script original |
| Granularidad | Modular — modo prueba (1 día) y modo producción (2,736 días) | Un solo parámetro `MODO` en `config.py` controla el comportamiento |

---

## 3. Decisiones sobre los outputs del algoritmo

El algoritmo produce tres outputs:

**Output 1 — Panel principal (curvas interpoladas):**
- Archivo: `outputs/curvas_panel_historico.csv`
- Formato: wide — una fila por fecha, 10 columnas
- Columnas: `fecha | r_1yr | r_2yr | r_3yr | r_5yr | r_7yr | r_10yr | r_15yr | r_20yr | r_30yr`
- Tasas expresadas en decimales (0.05 = 5%)

**Output 2 — Panel de puntos discretos crudos:**
- Archivo: `outputs/curvas_discretas_historico.csv`
- Contiene los N puntos spot antes de interpolación — uno por bono activo por día
- Columnas: `fecha | nemotecnico | tau | r_spot_discreto`
- Permite auditoría, detección de anomalías y cambio futuro del método de interpolación

**Output 3 — Log de diagnóstico:**
- Archivo: `outputs/log_bootstrapping.csv`
- Registra: bonos omitidos por filtro 30 días, VP residuales negativos, tasas fuera de rango, días INSUFICIENTE

---

## 4. Arquitectura de módulos — estado actual

| Módulo | Archivo | Estado |
|---|---|---|
| M0 | `src/config.py` | **Pendiente — es el primer archivo a construir** |
| M1 | `src/data_loader.py` | Pendiente |
| M2 | `src/preprocessor.py` | Pendiente |
| M3 | `src/cashflow_engine.py` | Pendiente |
| M4 | `src/bootstrapper.py` | Pendiente — núcleo más complejo |
| M5 | `src/interpolator.py` | Pendiente |
| M6 | `src/orchestrator.py` | Pendiente |

El código existente en `integrador_procesador.py` (Google Colab) cubre hasta
el scatter plot de yields VWAP — equivalente a M1 + parte de M2. Todo lo que
viene después es nuevo.

---

## 5. Correcciones matemáticas críticas ya incorporadas

### Corrección 1 — Fórmula de consolidación spot (CRÍTICA)

**Fórmula INCORRECTA** (descartada — solo válida en composición continua):
$$R_{\text{spot}} = \frac{r_{\text{anterior}} \cdot T_{\text{anterior}} + f_{\text{actual}} \cdot \Delta T_{\text{actual}}}{T_i}$$

**Fórmula CORRECTA** (equivalencia geométrica exacta en composición discreta):
$$R_{\text{spot}}(T_i) = \left[ (1 + r_{\text{anterior}})^{T_{\text{anterior}}} \times (1 + f_{\text{actual}})^{\Delta T_{\text{actual}}} \right]^{\frac{1}{T_i}} - 1$$

Esta corrección fue introducida por feedback externo y está incorporada en el
informe estratégico desde la Sección 2.5. Todo el código del bootstrapper
debe usar exclusivamente la fórmula geométrica.

---

## 6. Fundamentos matemáticos acordados

### Ecuación fundamental de precio
$$P_{\text{sucio}} = \sum_{t} \frac{CF_t}{(1 + r_t)^{t}}$$

### Denominador compuesto para fechas no alineadas (broken dates)
$$VP = \frac{CF}{(1 + r_{\text{anterior}})^{T_{\text{anterior}}} \times (1 + f)^{\Delta T}}$$

### Inicialización del ancla (primer bono sin cupones intermedios)
$$B_1 = \frac{P_{\text{sucio}}}{1 + \text{couponRate}/100} \quad \Rightarrow \quad r_{\text{spot},1} = B_1^{-1/\tau_1} - 1$$

### Supuesto step-function de Fama-Bliss
La tasa forward $f$ es constante en el intervalo $[T_{i-1}, T_i]$. Esto implica
que cualquier cupón que caiga dentro de ese intervalo se descuenta usando
el denominador compuesto de arriba.

---

## 7. Decisiones de diseño y sus justificaciones

**VWAP en lugar de tasa de cierre:**
El mercado TES es OTC e ilíquido en algunos instrumentos. El VWAP da más peso
a operaciones institucionales (mayor volumen) que a operaciones retail. Es la
práctica estándar en renta fija.

**`dueDate` del CSV estático como fuente autoritativa:**
El parsing del nemotécnico (extraer fecha de los últimos 6 dígitos de TFIT...)
es frágil y asume convención de nomenclatura constante en 11 años. El CSV
`bonos_info_estatica.csv` contiene la fecha oficial de la API BVC. El parsing
se mantiene solo como fallback de contingencia.

**Umbral de 30 días:**
Garantiza que el tramo nuevo de cada bono sea al menos 30/365 ≈ 0.082 años,
manteniendo la ecuación de resolución de $f$ bien condicionada numéricamente.
El script original usaba 370 días pero era solo exploratorio — el umbral
operativo del algoritmo es 30 días.

**Interpolación lineal y no spline:**
Los Unsmoothed Fama-Bliss Yields son por definición puntos discretos sin suavizar.
La interpolación lineal preserva esa característica. M5 puede reemplazarse por
spline cúbico o Nelson-Siegel en el futuro sin tocar ningún otro módulo.

**Composición discreta sobre continua:**
Los TES colombianos se cotizan en convención discreta anual base 365. Usar
continua requeriría conversiones innecesarias y viola la coherencia con el
mercado. El promedio ponderado lineal de tasas solo es válido en continua —
en discreta se debe usar la equivalencia geométrica (ver Sección 5).

---

## 8. Estructura del entorno local configurado

```
C:\proyectos\TES_curvas\
│
├── datos/
│   ├── BVC_Bonos_2015_2018.db      (32 MB)
│   ├── BVC_Bonos_2019_2022.db      (19 MB)
│   ├── BVC_Bonos_2023_2026.db      (12 MB)
│   └── bonos_info_estatica.csv     (4 KB)
│
├── src/                            (6 archivos .py vacíos — pendientes)
├── notebooks/
├── outputs/
│   └── curva_discreta/
├── tests/
├── docs/                           (documentación del proyecto)
│
├── verificar_entorno.py
├── .gitignore
├── requirements.txt
└── README.md
```

**Entorno virtual:** `.venv` creado y activado con `Set-ExecutionPolicy RemoteSigned`
aplicado para habilitar scripts en PowerShell.

**Dependencias instaladas:** pandas, numpy, scipy, plotly, matplotlib, seaborn,
ipykernel, jupyter.

**Intérprete VSCode:** configurado en `.venv` vía `Python: Select Interpreter`.

---

## 9. Archivos de documentación en docs/

| Archivo | Contenido |
|---|---|
| `informe_estrategico_curvas_TES.md` | Plan estratégico completo — 10 secciones — arquitectura, matemáticas, itinerario, riesgos |
| `checklist_entorno_local_TES.md` | Guía paso a paso de configuración del entorno local |
| `contexto_dataset_bonos_bvc.md` | Descripción del dataset de operaciones — estructura, cobertura, consideraciones |
| `contexto_info_estatica_bonos.md` | Descripción del CSV de información estática de los 29 bonos |
| `proceso_general_deconstruccion.md` | Documento de proceso general del algoritmo |
| `proceso_especifico_curva_discreta.md` | Documento de proceso específico del bootstrapping |

---

## 10. Estado del proyecto al cerrar esta conversación

- Entorno local configurado y funcional ✓
- Documentación estratégica completa y congelada ✓
- Tres preguntas finales respondidas y registradas ✓
- Corrección matemática de fórmula spot incorporada ✓
- **Próximo paso inmediato: construir `src/config.py` (M0)**

El script `verificar_entorno.py` debe mostrar todos los `✓` antes de
arrancar con el código. Una vez confirmado, el orden de construcción es
M0 → M1 → M2 → M3 → M4 → M5 → M6, validando cada módulo antes de avanzar
al siguiente.

---

## 11. Tono y estilo de trabajo acordado

- Juan David está aprendiendo buenas prácticas de desarrollo en paralelo al proyecto.
  Explicar el "por qué" de cada decisión, no solo el "qué".
- Antes de programar cualquier módulo nuevo, confirmar si hay feedback pendiente.
- El código se entrega módulo por módulo — nunca todo junto.
- Cada módulo debe tener comentarios que expliquen la lógica, no solo el código.
- Ante cualquier duda matemática o de diseño, detener y discutir antes de implementar.

---

*Documento generado al cierre de la conversación de diseño inicial.*
*Proyecto Semillero · Curvas de Rendimiento TES · BVC 2015–2026*
