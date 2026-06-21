# Informe de Handoff — Fase de Experimentación
## Proyecto: Curvas de Rendimiento Cero-Cupón TES · BVC 2015–2026

---

## 1. Contexto del proyecto

Se está construyendo un pipeline en Python que construye curvas de rendimiento
cero-cupón para bonos soberanos colombianos TES (TFIT) negociados en la BVC
entre 2015 y 2026, usando el método de bootstrapping de Fama-Bliss.

El pipeline está parcialmente construido. Los módulos M0 a M4 están funcionales
y validados. La fase de experimentación busca inspeccionar los resultados
intermedios del bootstrapper antes de continuar con M5 (interpolación) y M6
(orquestador histórico).

---

## 2. Convenciones matemáticas — CRÍTICO leer antes de experimentar

**Convención dominante:** composición discreta anual, base 365 días.

**Fórmula de consolidación spot** (equivalencia geométrica exacta):
$$R_{spot}(T_i) = \left[(1 + r_{anterior})^{T_{anterior}} \times (1 + f_{actual})^{\Delta T}\right]^{1/T_i} - 1$$

> ⚠️ La fórmula de promedio ponderado lineal
> $R_{spot} = (r_{ant} \cdot T_{ant} + f \cdot \Delta T) / T_i$
> está **descartada** — solo es válida en composición continua.
> Todo el código usa exclusivamente la equivalencia geométrica.

**Denominador compuesto para broken dates:**
$$VP = \frac{CF}{(1 + r_{anterior})^{T_{anterior}} \times (1 + f)^{\Delta T}}$$

**Clasificación estricta de flujos por bono i:**

| Categoría | Condición | Descuento |
|---|---|---|
| A — conocidos | $t \leq T_{anterior}$ | Cursor de tasas ya calculadas |
| B — tramo nuevo | $T_{anterior} < t < \tau_i$ | Depende de $f$ (incógnita) |
| C — flujo final | $t = \tau_i$ | Depende de $f$ (incógnita) |

**Regla de resolución:**
- B vacío → despeje algebraico exacto
- B no vacío → solver numérico `brentq` (polinomio en f, sin solución algebraica)

---

## 3. Estructura del proyecto

```
C:\proyectos\TES_curvas\
│
├── datos/
│   ├── BVC_Bonos_2015_2018.db      (281,969 operaciones)
│   ├── BVC_Bonos_2019_2022.db      (150,210 operaciones)
│   ├── BVC_Bonos_2023_2026.db      ( 98,728 operaciones)
│   └── bonos_info_estatica.csv     (29 bonos — info estructural)
│
├── src/
│   ├── config.py           M0 — parámetros y rutas ✓
│   ├── data_loader.py      M1 — carga de SQLite    ✓
│   ├── preprocessor.py     M2 — VWAP, tau, filtros ✓
│   ├── cashflow_engine.py  M3 — calendarios flujos ✓
│   ├── bootstrapper.py     M4 — motor Fama-Bliss   ✓
│   ├── interpolator.py     M5 — pendiente
│   └── orchestrator.py     M6 — pendiente
│
└── notebooks/              ← aquí van los experimentos
```

---

## 4. Cómo invocar el pipeline desde un notebook

### 4.1 Setup inicial (primera celda del notebook)

```python
import sys
import os

# Agregar src/ al path para poder importar los módulos
sys.path.insert(0, os.path.join(os.getcwd(), '..', 'src'))

# Importar módulos
import config
from data_loader import cargar_todo
from preprocessor import preprocesar, obtener_dia
from cashflow_engine import obtener_flujos_bono, flujos_a_dataframe
from bootstrapper import bootstrap_dia

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

> Si el notebook está en `notebooks/`, el path `../src` apunta correctamente
> a la carpeta de módulos.

### 4.2 Carga y preprocesamiento (hacer UNA sola vez por sesión)

```python
# Carga completa — ~5 segundos
df_op, df_est = cargar_todo()

# Preprocesamiento completo — ~10 segundos
df_pre = preprocesar(df_op, df_est)

print(f"Registros preprocesados: {len(df_pre):,}")
print(f"Días hábiles: {df_pre['fecha_neg'].nunique():,}")
```

### 4.3 Obtener datos de un día específico

```python
fecha = '2016-10-21'  # o cualquier fecha hábil del período
df_dia = obtener_dia(df_pre, fecha)
print(df_dia[['nemotecnico', 'tasa_vwap', 'tau', 'couponRate', 'dueDate']])
```

### 4.4 Obtener flujos de caja de un bono

```python
fila = df_dia[df_dia['nemotecnico'] == 'TFIT16280428'].iloc[0]
flujos = obtener_flujos_bono(fila)
df_flujos = flujos_a_dataframe(flujos, 'TFIT16280428')
print(df_flujos)
```

### 4.5 Correr el bootstrapper para un día

```python
puntos, warnings = bootstrap_dia(df_dia, verbose=True)

# Convertir a DataFrame para análisis
df_curva = pd.DataFrame(puntos)
print(df_curva[['nemotecnico', 'tau', 'r_spot', 'f_forward', 'n_flujos_B']])

# Advertencias
for w in warnings:
    print(f"⚠ {w}")
```

---

## 5. Estructura del DataFrame de salida del bootstrapper

Cada elemento de `puntos` es un dict con estas claves:

| Campo | Tipo | Descripción |
|---|---|---|
| `fecha` | Timestamp | Fecha de negociación |
| `nemotecnico` | str | Identificador del bono |
| `tau` | float | Plazo residual en años decimales |
| `r_spot` | float | Tasa spot cero-cupón calculada (decimal, ej: 0.065) |
| `f_forward` | float | Tasa forward marginal del tramo (decimal) |
| `T_anterior` | float | Vencimiento del bono anterior (inicio del tramo) |
| `R_anterior` | float | Tasa spot acumulada hasta T_anterior |
| `delta_T` | float | Duración del tramo nuevo en años |
| `n_flujos_B` | int | Número de flujos intermedios en el tramo (B) |

---

## 6. Resultado validado para 2016-10-21

El bootstrapper produce 8 puntos spot para esta fecha:

| Bono | tau (años) | tasa VWAP | R_spot esperado |
|---|---|---|---|
| TFIT06211118 | 2.085 | 6.264% | ~6.x% |
| TFIT06110919 | 2.890 | 6.481% | ~6.x% |
| TFIT15240720 | 3.759 | 6.633% | ~6.x% |
| TFIT10040522 | 5.537 | 6.764% | ~6.x% |
| TFIT16240724 | 7.762 | 6.928% | ~7.x% |
| TFIT15260826 | 9.852 | 7.192% | ~7.x% |
| TFIT16280428 | 11.526 | 7.378% | ~7.x% |
| TFIT16180930 | 13.918 | 7.328% | ~7.x% |

Los R_spot exactos emergen del bootstrapping. Son los valores a validar
en la experimentación.

---

## 7. Preguntas abiertas para la fase de experimentación

Estas son las preguntas que se busca responder antes de continuar
con M5 (interpolación):

1. **Plausibilidad de la curva:** ¿Los R_spot son monótonamente crecientes
   con el plazo? ¿Están en el rango histórico TES 2016 (4%–16%)?

2. **Coherencia VWAP vs spot:** ¿Los R_spot son sistemáticamente menores
   que las tasas VWAP correspondientes? (Deben serlo — el spot elimina
   el efecto cupón que infla el yield).

3. **Métodos usados:** ¿Cuántos bonos requirieron solver numérico vs
   despeje algebraico? ¿Hubo advertencias?

4. **Visualización:** Scatter de R_spot vs tau superpuesto sobre el
   scatter de tasa_vwap vs tau — ¿la curva spot queda por debajo
   de los yields?

5. **Estabilidad:** Correr el bootstrapper para varias fechas y verificar
   que la curva no produce valores aberrantes.

---

## 8. Lo que NO está implementado aún

- **M5 interpolator:** la interpolación lineal a nodos fijos
  [1, 2, 3, 5, 7, 10, 15, 20, 30 años] no existe todavía.
  El bootstrapper produce puntos en los tau exactos de los bonos,
  no en nodos estándar.

- **M6 orchestrator:** el barrido histórico completo (2,736 días)
  no está implementado. Solo se puede correr fecha por fecha.

- **Outputs a CSV:** no hay escritura a disco todavía — todo vive
  en memoria durante la sesión.

---

## 9. Parámetros configurables en config.py

```python
MODO                   = 'prueba'      # 'prueba' | 'produccion'
FECHA_PRUEBA           = '2016-10-21'  # fecha de validación
UMBRAL_ESTABILIDAD_DIAS = 30           # filtro de estabilidad
NODOS_INTERPOLACION    = [1,2,3,5,7,10,15,20,30]  # años
BASE_DIAS              = 365
F_MIN                  = -0.05         # tasa forward mínima válida
F_MAX                  =  0.50         # tasa forward máxima válida
MAX_EXTRAPOLACION_ANOS =  5.0
```

---

*Documento generado para handoff a fase de experimentación.*
*Proyecto Semillero · Curvas de Rendimiento TES · BVC 2015–2026*
