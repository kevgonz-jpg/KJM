# INFORME ESTRATÉGICO
## Algoritmo de Construcción de Curvas de Rendimiento
### Bonos Soberanos TES · BVC · 2015–2026

---

| Parámetro | Valor |
|---|---|
| **Método** | Bootstrapping Fama-Bliss (composición discreta anual) |
| **Universo** | 29 bonos TFIT — 530,907 operaciones — 2,736 días hábiles |
| **Período** | 2015-01-02 → 2026-03-20 |
| **Entorno** | Google Colab + SQLite + Python 3 |
| **Destino** | Panel de curvas spot para modelos econométricos / ML |
| **Convención** | Tasa discreta anual — base 365 — cupón anual vencido |

*Proyecto Semillero · Universidad · 2025–2026*

---

## 0. Resumen Ejecutivo

Este documento describe en detalle completo la estrategia de diseño, los fundamentos teóricos, la arquitectura de módulos y el itinerario de construcción del algoritmo que producirá un panel histórico de curvas de rendimiento cero-cupón para los bonos soberanos TES colombianos (TFIT) negociados en la Bolsa de Valores de Colombia (BVC) entre 2015 y 2026.

El algoritmo implementa el método de bootstrapping de Fama y Bliss (1987) adaptado a la convención de mercado colombiana: composición discreta anual, base 365, cupón anual vencido. La salida es un DataFrame estructurado con la tasa spot cero-cupón para cada día hábil y cada nodo de plazo estandarizado, listo para ser consumido directamente por modelos econométricos o de machine learning.

El diseño prioriza: (1) corrección matemática estricta, (2) eficiencia computacional para iterar sobre 2,736 días, (3) modularidad para facilitar pruebas por día individual antes del barrido completo, y (4) trazabilidad total para diagnóstico y auditoría de cada curva construida.

---

## 1. Contexto del Problema y Descripción de los Datos

### 1.1 El Problema Central

Los bonos TES TFIT son instrumentos de tasa fija que pagan un cupón anual. Esto significa que la tasa de rendimiento observada en el mercado secundario (yield to maturity) **NO** es la tasa pura del dinero a ese plazo: es un promedio distorsionado por los flujos intermedios de cupones. Para construir una curva de rendimientos útil en modelos de valoración, gestión de riesgos o econometría de tasas, necesitamos extraer las tasas spot cero-cupón puras — es decir, el costo del dinero a cada plazo sin la contaminación de los cupones intermedios.

El método Fama-Bliss resuelve esto procesando los bonos en orden de madurez ascendente y extrayendo, de forma iterativa, la tasa forward marginal de cada tramo nuevo que aporta cada bono sucesivo. La acumulación de estas tasas forward produce la curva spot cero-cupón.

### 1.2 Descripción del Dataset de Operaciones

Los datos de mercado secundario están distribuidos en tres bases de datos SQLite:

| Archivo SQLite | Período | Operaciones | Días hábiles |
|---|---|---|---|
| `BVC_Bonos_2015_2018.db` | 2015-01-02 → 2018-12-31 | 281,969 | 972 |
| `BVC_Bonos_2019_2022.db` | 2019-01-01 → 2022-12-31 | 150,210 | 978 |
| `BVC_Bonos_2023_2026.db` | 2023-01-01 → 2026-03-20 | 98,728 | 786 |
| **TOTAL** | **11 años** | **530,907** | **2,736** |

Cada operación contiene: fecha de negociación, nemotécnico, hora, tasa (yield), precio sucio y volumen en COP. No hay valores nulos en las variables numéricas. Existen 388 días hábiles sin operaciones (marcados `SIN_DATOS`) que serán excluidos del proceso de bootstrapping.

### 1.3 Descripción de la Información Estática

El archivo `bonos_info_estatica.csv` contiene los atributos estructurales fijos de los 29 bonos TFIT: nemotécnico, fecha de emisión, fecha de vencimiento, tasa cupón, frecuencia y base de cálculo. Esta información no varía en el tiempo y se une al dataset de operaciones como metadatos del instrumento. Los campos críticos para el bootstrapping son:

- **`dueDate`**: define el vencimiento — determina el orden de procesamiento y el cálculo de $\tau$.
- **`couponRate`**: define el flujo de caja anual — base del calendario de pagos.
- **`base`**: confirmado en 365 para todos los instrumentos — convención colombiana.
- **`couponFrequency`**: anual para todos — simplifica el calendario al máximo.

### 1.4 Características del Universo de Instrumentos

Los 29 bonos cubren un espectro de vencimientos desde 2015 hasta 2058, con tasas cupón entre 5.0% y 13.25%. Todos son tasa fija nominal en COP. Esta homogeneidad estructural simplifica el motor de bootstrapping: no hay bonos indexados, no hay opciones embebidas, no hay amortizaciones parciales. El único pago intermedio es el cupón anual; el pago final es cupón más principal (100% del valor nominal).

**Punto crítico de diseño:** el panel de bonos es no balanceado. No todos los bonos existen o se negocian en todos los días del período. En cada fecha de negociación, el subconjunto activo varía entre 5 y 19 bonos según los vencimientos y emisiones vigentes para esa fecha.

---

## 2. Fundamentos Teóricos del Motor de Bootstrapping

### 2.1 El Principio de Descomposición en Bonos Cero-Cupón

Cualquier bono con cupones puede interpretarse como un portafolio de bonos cero-cupón sintéticos: uno por cada flujo de caja futuro. Un bono que paga cupones anuales de $C$ durante $T$ años y devuelve el principal $N$ en $T$ es matemáticamente equivalente a $T$ bonos cero-cupón: $T-1$ bonos que pagan solo $C$ en cada año intermedio, y uno que paga $C + N$ en el año $T$.

El precio de mercado de este bono debe ser igual a la suma de los valores presentes de todos esos flujos, descontados a las tasas spot cero-cupón puras de cada plazo. Esta ecuación es la base de todo el motor:

$$P_{\text{sucio}} = \sum_{t} \frac{CF_t}{(1 + r_t)^{t}}$$

donde:

| Símbolo | Descripción |
|---|---|
| $P_{\text{sucio}}$ | Precio sucio observado en el mercado |
| $CF_t$ | Flujo de caja en el momento $t$ (cupón $C$, o $C + 100$ en vencimiento $T$) |
| $r_t$ | Tasa spot cero-cupón pura para el plazo $t$ — **la incógnita** |
| $t$ | Plazo en años decimales desde la fecha de negociación |

> **Convención:** base 365 días, cupón anual vencido, composición discreta anual.

### 2.2 La Lógica Secuencial de Fama-Bliss

El ingenio del método es que, si procesamos los bonos en orden ascendente de madurez, en cada paso solo existe **una incógnita nueva**: la tasa del tramo que va desde el vencimiento del bono anterior hasta el vencimiento del bono actual. Todas las tasas para plazos anteriores ya fueron calculadas en pasos previos.

Esto convierte un sistema potencialmente sobredeterminado en una secuencia de ecuaciones de una sola incógnita, cada una resoluble algebraicamente o con un método numérico simple.

### 2.3 El Supuesto de Función Escalonada (Step-Function)

Fama-Bliss asume que la tasa forward es **constante** dentro de cada intervalo entre vencimientos consecutivos. Esto significa que si el bono anterior venció en $T_1$ y el bono actual vence en $T_2$, la tasa forward $f$ calculada para este bono es la que rige para **todo** el intervalo $[T_1, T_2]$, sin importar cuándo dentro de ese intervalo caigan los cupones del bono.

Este supuesto tiene dos consecuencias prácticas:

- **Parsimonia:** el modelo tiene exactamente tantos parámetros como bonos activos — nunca está sobredeterminado ni subdeterminado.
- **Descuento de cupones en vacíos temporales:** si un cupón del bono $i$ cae en un momento $T_{\text{intermedio}}$ dentro del intervalo $[T_{i-1}, T_i]$, se descuenta usando $T_{i-1}$ pasos al factor conocido y $(T_{\text{intermedio}} - T_{i-1})$ pasos adicionales a la tasa forward $f$ que se está resolviendo.

### 2.4 El Problema de Fechas No Alineadas (Broken Dates)

Esta es la complicación técnica más importante del algoritmo. Los bonos TES pagan cupones en la misma fecha calendario cada año (aniversario de la emisión). Pero los distintos bonos tienen fechas de emisión distintas, por lo que sus cupones caen en fechas distintas del año.

**Ejemplo concreto:** el bono A vence en $t = 1.00$ años. El bono B vence en $t = 2.10$ años y paga un cupón intermedio en $t = 1.10$ años. Para descontar ese cupón intermedio, el algoritmo debe usar la tasa spot conocida $r_1$ para el tramo $[0, 1.00]$ y la nueva tasa forward $f$ para el tramo adicional $[1.00, 1.10]$. El denominador de descuento es:

$$\text{Factor de descuento} = (1 + r_{\text{anterior}})^{T_{\text{anterior}}} \times (1 + f)^{\Delta T}$$

donde $\Delta T = T_{\text{total}} - T_{\text{anterior}}$ es la fracción de año en el nuevo tramo. La estructura completa del valor presente de un flujo en $T_{\text{total}}$ es:

$$VP = \frac{CF}{(1 + r_{\text{anterior}})^{T_{\text{anterior}}} \times (1 + f)^{\Delta T}}$$

El algoritmo debe resolver $f$ algebraicamente o numéricamente buscando la raíz de la ecuación de precio bajo esta estructura compuesta.

### 2.5 Consolidación de la Tasa Spot (Equivalencia Geométrica Exacta)

Una vez obtenida la tasa forward marginal $f$ del tramo $[T_{i-1}, T_i]$, el resultado que se almacena en la curva **no** es $f$, sino la tasa spot acumulada $R_{\text{spot}}(T_i)$: la tasa cero-cupón pura equivalente a toda la cadena de factores de descuento desde el origen hasta $T_i$.

La fórmula correcta bajo composición discreta anual es la **equivalencia geométrica exacta** de los factores de descuento acumulados:

$$\boxed{R_{\text{spot}}(T_i) = \left[ (1 + r_{\text{anterior}})^{T_{\text{anterior}}} \times (1 + f_{\text{actual}})^{\Delta T_{\text{actual}}} \right]^{\frac{1}{T_i}} - 1}$$

> **Nota importante:** la fórmula de promedio ponderado lineal $R_{\text{spot}} = (r_{\text{anterior}} \cdot T_{\text{anterior}} + f_{\text{actual}} \cdot \Delta T_{\text{actual}}) / T_i$ es una simplificación que **solo es matemáticamente válida en el dominio de la composición continua**, donde las tasas son logarítmicas y sí se puede promediar linealmente en el tiempo. En composición discreta, el promedio de tasas no equivale al promedio de factores de descuento compuestos — se comete un error sistemático que crece con el plazo y la magnitud de las tasas. La ecuación correcta es siempre la equivalencia geométrica presentada arriba.

Este valor de $R_{\text{spot}}$ representa la tasa cero-cupón pura para el plazo $T_i$ y es el **"Unsmoothed Fama-Bliss Yield"** para ese nodo de la curva.

---

## 3. Arquitectura del Sistema — Módulos y Responsabilidades

El algoritmo se organiza en 6 módulos independientes y un orquestador central. Esta arquitectura modular permite probar cada componente de forma aislada, reemplazar partes sin afectar el resto, y escalar del modo *prueba* (día único) al modo *producción* (barrido histórico completo) con un solo parámetro.

| Módulo | Nombre propuesto | Responsabilidad |
|---|---|---|
| M0 | `config.py` | Rutas, parámetros globales (umbral 30 días, nodos de interpolación), flags de modo. |
| M1 | `data_loader.py` | Conexión a las 3 SQLite, carga y concatenación del dataset de operaciones, carga de info estática. |
| M2 | `preprocessor.py` | VWAP diario por bono, filtro `SIN_DATOS`, enriquecimiento con `dueDate` y `couponRate`, cálculo de $\tau$, filtro de estabilidad (umbral 30 días). |
| M3 | `cashflow_engine.py` | Generación del calendario de flujos de caja para cada bono en cada fecha de negociación. |
| M4 | `bootstrapper.py` | Motor iterativo Fama-Bliss: extrae tasas forward y consolida tasas spot para un día específico. |
| M5 | `interpolator.py` | Interpolación lineal de los puntos spot discretos a nodos estandarizados. |
| M6 | `orchestrator.py` | Loop principal: itera sobre las fechas, llama a M4 y M5, acumula resultados, exporta el panel final. |

### 3.1 Módulo M0 — Configuración Central

Un único archivo de configuración controla todos los parámetros del sistema. Cambiar el modo de prueba a producción, los nodos de interpolación, o el umbral de estabilidad se hace en un solo lugar sin tocar el código de ningún módulo.

```python
# config.py
RUTAS_DB = {
    '2015_2018': '/content/drive/MyDrive/.../BVC_Bonos_2015_2018.db',
    '2019_2022': '/content/drive/MyDrive/.../BVC_Bonos_2019_2022.db',
    '2023_2026': '/content/drive/MyDrive/.../BVC_Bonos_2023_2026.db',
}
RUTA_INFO_ESTATICA = '/content/drive/MyDrive/.../bonos_info_estatica.csv'
UMBRAL_ESTABILIDAD_DIAS = 30
NODOS_INTERPOLACION    = [1, 2, 3, 5, 7, 10, 15, 20, 30]  # años
BASE_DIAS              = 365
MODO                   = 'prueba'        # 'prueba' | 'produccion'
FECHA_PRUEBA           = '2016-10-21'   # solo aplica si MODO == 'prueba'
```

### 3.2 Módulo M1 — Carga de Datos

Conecta a las tres bases de datos SQLite, ejecuta `SELECT` sobre la tabla `operaciones` de cada una, concatena los tres DataFrames y realiza el merge con la información estática. La carga se hace **una sola vez** al inicio del proceso: los datos completos se mantienen en memoria durante todo el barrido histórico, evitando reconexiones repetidas a SQLite que degradarían el rendimiento.

> **Consideración de rendimiento:** los 530,907 registros ocupan aproximadamente 50–80 MB en RAM como DataFrame de pandas — perfectamente manejable en Colab. No es necesario procesamiento por chunks ni lazy loading.

### 3.3 Módulo M2 — Preprocesamiento

Es la fase de transformación más intensa. Aplica las siguientes operaciones en orden:

1. **Cálculo de VWAP:** para cada par `(fecha_neg, nemotecnico)`, colapsa todas las operaciones del día en un único registro con tasa ponderada por volumen. Reduce los 530,907 registros al nivel (fecha, bono).
2. **Enriquecimiento con metadatos:** join con `bonos_info_estatica.csv` para incorporar `dueDate` y `couponRate` a cada registro del VWAP.
3. **Cálculo de $\tau$:** plazo residual en años decimales. Se usa `dueDate` del CSV estático (fuente autoritativa), no el parsing del nemotécnico (que solo sería un fallback).
4. **Filtro de bonos vencidos:** elimina registros donde $\tau \leq 0$.
5. **Filtro de estabilidad:** para cada fecha de negociación, identifica pares de bonos con vencimientos separados por menos de 30 días. En cada par conflictivo, elimina el bono con menor volumen total histórico.
6. **Ordenamiento:** dentro de cada fecha, ordena los bonos de menor a mayor $\tau$. Este orden es mandatorio para el bootstrapping.

### 3.4 Módulo M3 — Motor de Flujos de Caja

Para cada bono en cada fecha de negociación, genera el vector completo de flujos de caja futuros con sus fechas exactas. La lógica es:

- Identificar el número de cupones pendientes: $\lfloor \tau \rfloor$ si $\tau$ no es entero, o $\tau$ si es entero.
- Calcular las fechas exactas de cada cupón como aniversarios de la fecha de vencimiento, contando hacia atrás desde `dueDate`.
- Calcular el tiempo en años decimales hasta cada flujo: $(\text{fecha\_flujo} - \text{fecha\_neg}).\text{days} / 365$.
- Asignar montos: cupón $= \text{couponRate}/100$ por cada período intermedio; pago final $= 1 + \text{couponRate}/100$ (sobre valor nominal normalizado a 1).

El resultado es una lista de tuplas $(t, CF)$ ordenada ascendentemente por $t$, lista para ser consumida directamente por el bootstrapper.

### 3.5 Módulo M4 — Motor de Bootstrapping (Núcleo Central)

Recibe el DataFrame preprocesado de un día específico ($N$ bonos ordenados por $\tau$) y produce los $N$ puntos spot discretos de la curva para ese día. Opera en tres sub-fases:

#### Sub-fase 4.1 — Inicialización (Ancla de la Curva)

Para el primer bono (menor $\tau$), si solo le queda el pago final (es decir, no hay cupones intermedios — lo cual ocurre cuando $\tau < 1$ año), su precio de mercado implica directamente el factor de descuento:

$$B_1 = \frac{P_{\text{sucio}}}{1 + \text{couponRate}/100}$$

$$r_{\text{spot},1} = B_1^{-1/\tau_1} - 1$$

En el caso de que el primer bono sí tenga cupones intermedios, se resuelve numéricamente igual que los bonos subsiguientes, pero partiendo de un cursor de tasas conocidas vacío.

#### Sub-fase 4.2 — Iteración Recursiva

Para cada bono $i$ (desde el segundo en adelante):

1. Obtener los flujos de caja del bono $i$ desde M3.
2. Separar los flujos en dos grupos: flujos anteriores ($t \leq T_{i-1}$, descontables con tasas ya conocidas) y el flujo final ($t = T_i$, que contiene la incógnita $f$).
3. Para flujos anteriores: calcular su VP usando el cursor de tasas acumuladas. Distinguir si el flujo cae exactamente en un nodo conocido (descuento directo con $r_t$ conocida) o en un hueco entre nodos (denominador compuesto con la tasa forward del tramo en que cae).
4. Calcular el VP residual: $P_{\text{sucio},i} - \sum VP_{\text{flujos anteriores}}$.
5. Resolver para $f$: el VP residual debe igualar $CF_{\text{final}} / [(1 + r_{i-1})^{T_{i-1}} \times (1 + f)^{\Delta T}]$. Despejar $f$ algebraicamente.
6. Calcular $R_{\text{spot},i}$ usando la equivalencia geométrica exacta (Sección 2.5).
7. Almacenar el par $(\tau_i, R_{\text{spot},i})$ y actualizar el cursor de tasas conocidas.

#### Sub-fase 4.3 — Manejo de Casos Degenerados

El algoritmo debe ser robusto ante situaciones atípicas:

- **VP residual negativo:** puede ocurrir por precios de mercado extremos o por inconsistencias de datos. El algoritmo registra una advertencia y omite ese bono o imputa con interpolación de los vecinos.
- **$f$ negativa o fuera de rango razonable** (< −0.05 o > 0.50): señal de inconsistencia. Se registra y se aplica la misma lógica de omisión.
- **Un solo bono activo ese día:** no es posible hacer bootstrapping. El día se marca como `INSUFICIENTE` y se excluye del panel.

### 3.6 Módulo M5 — Interpolación a Nodos Estándar

Los puntos spot discretos que produce el bootstrapper corresponden a los $\tau$ exactos de los bonos activos ese día (por ejemplo, 1.23, 3.87, 6.44, 9.15, 14.02, 18.67 años). Para construir un panel de series temporales que un modelo pueda consumir, necesitamos los valores en nodos fijos.

Los nodos configurados son: **1, 2, 3, 5, 7, 10, 15, 20 y 30 años**. La interpolación es lineal entre los puntos spot más cercanos. Para nodos fuera del rango de bonos disponibles ese día (extrapolación), se aplica extrapolación lineal usando los dos puntos extremos más cercanos, con una advertencia en el log.

El resultado es un vector de 9 tasas spot para cada día: $[r_{1}, r_{2}, r_{3}, r_{5}, r_{7}, r_{10}, r_{15}, r_{20}, r_{30}]$.

### 3.7 Módulo M6 — Orquestador

Es el único módulo que *sabe* si estamos en modo prueba o producción. En modo prueba, procesa una fecha específica y muestra el resultado con visualización inmediata. En modo producción, itera sobre las 2,736 fechas hábiles, acumula los resultados en un DataFrame y exporta el panel completo.

**Estructura del loop de producción:** carga datos una vez (M1), preprocesa todo el dataset (M2), luego para cada fecha llama a M3 y M4 con el subconjunto de ese día, y a M5 para estandarizar. Los resultados se acumulan en una lista y se convierten a DataFrame al final para evitar el costo de concatenaciones repetidas dentro del loop.

---

## 4. Flujo Completo del Algoritmo — Paso a Paso

### 4.1 Diagrama de Flujo Narrativo

El siguiente diagrama describe la secuencia completa de ejecución para un día de negociación, desde la carga de datos crudos hasta el vector de tasas spot estandarizadas:

```
┌─────────────────────────────────────────────────────────────────────┐
│ PASO 1 — CARGA                                                      │
│ Leer operaciones y bonos_info_estatica desde SQLite y CSV.          │
│ Resultado: df_operaciones (530k filas) + df_estatica (29 filas).    │
└────────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PASO 2 — VWAP                                                       │
│ Colapsar operaciones por (fecha_neg, nemotecnico) → tasa ponderada  │
│ por volumen. Resultado: df_vwap (~50k filas, un registro por bono   │
│ por día).                                                           │
└────────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PASO 3 — ENRIQUECIMIENTO                                            │
│ Merge con df_estatica para agregar dueDate y couponRate.            │
│ Calcular τ = (dueDate - fecha_neg).days / 365. Filtrar τ ≤ 0.      │
└────────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PASO 4 — FILTRO 30 DÍAS                                             │
│ Para cada fecha, detectar pares de bonos con                        │
│ |dueDate_A − dueDate_B| < 30 días. Eliminar el de menor volumen    │
│ histórico. Ordenar por τ ascendente.                                │
└────────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PASO 5 — FLUJOS DE CAJA                                             │
│ Para cada bono en la fecha, generar calendario de cupones futuros   │
│ y pago final. Resultado: lista de (t_flujo_en_años, monto_norm).    │
└────────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PASO 6 — BOOTSTRAPPING                                              │
│ Iterar bono por bono. Para cada bono, descontar flujos previos con  │
│ tasas conocidas, resolver f algebraicamente para el flujo final,    │
│ calcular R_spot con equivalencia geométrica, actualizar cursor.     │
└────────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PASO 7 — INTERPOLACIÓN                                              │
│ Interpolar los N puntos spot discretos a los 9 nodos fijos          │
│ [1, 2, 3, 5, 7, 10, 15, 20, 30 años].                              │
│ Extrapolación lineal en extremos si necesario.                      │
└────────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PASO 8 — SALIDA                                                     │
│ Fila del panel final:                                               │
│ {fecha, r_1yr, r_2yr, r_3yr, r_5yr, r_7yr, r_10yr, r_15yr,        │
│  r_20yr, r_30yr}. Exportar a CSV.                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Estructura de la Salida — Panel de Curvas

### 5.1 DataFrame Final

El producto del algoritmo es un DataFrame de 2,736 filas (una por día hábil con datos) y 10 columnas. Este es el panel que alimentará directamente los modelos econométricos o de ML:

| fecha | r_1yr | r_2yr | r_3yr | r_5yr | r_7yr | r_10yr | r_15yr | r_20yr | r_30yr |
|---|---|---|---|---|---|---|---|---|---|
| 2015-01-02 | 0.0421 | 0.0456 | 0.0489 | 0.0531 | 0.0562 | 0.0598 | 0.0634 | 0.0651 | 0.0668 |
| 2015-01-05 | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| **2026-03-20** | **0.0891** | **0.0876** | **0.0863** | **0.0844** | **0.0831** | **0.0819** | **0.0807** | **0.0798** | **0.0791** |

> Las tasas se expresan en decimales (0.05 = 5%). Los valores de la tabla son ilustrativos. Los nodos con `NaN` por extrapolación fuera de rango se registrarán explícitamente para transparencia.

### 5.2 Archivos de Diagnóstico Adicionales

Junto al panel principal, el algoritmo generará dos archivos de trazabilidad:

- **`curva_discreta_{fecha}.csv`:** para las fechas procesadas en modo prueba, los puntos discretos crudos ($\tau$ exacto de cada bono, $R_{\text{spot}}$ antes de interpolación) para verificación manual.
- **`log_bootstrapping.csv`:** registro de advertencias por día — bonos omitidos por el filtro de 30 días, VP residuales negativos, tasas fuera de rango, días marcados como `INSUFICIENTE`.

---

## 6. Itinerario de Construcción — Fases y Entregables

El desarrollo se organiza en 5 fases secuenciales. Las fases 1–3 corresponden al modo prueba (un día específico). Las fases 4–5 escalan al barrido histórico completo.

| Fase | Nombre | Tareas clave | Entregable |
|---|---|---|---|
| F1 | Carga y Preprocesamiento | M0 config, M1 data_loader, M2 preprocessor. VWAP, merge estática, $\tau$, filtro 30 días. | `df_preprocessed` validado para fecha de prueba. |
| F2 | Motor de Flujos de Caja | M3 cashflow_engine. Calendario de cupones exactos. Validación con fecha 2016-10-21. | Calendarios de flujos verificados para todos los bonos activos. |
| F3 | Motor de Bootstrapping (1 día) | M4 bootstrapper. Ancla, iteración recursiva, manejo fechas rotas, consolidación geométrica spot. Validación visual contra scatter plot existente. | Curva discreta validada para fecha de prueba + visualización. |
| F4 | Interpolación y Salida | M5 interpolator. Interpolación lineal a 9 nodos. Validación de extrapolación en extremos. | Vector de 9 tasas spot estandarizadas para fecha de prueba. |
| F5 | Orquestador y Barrido Histórico | M6 orchestrator. Loop sobre 2,736 días. Optimización de rendimiento. Exportación CSV del panel completo. | Panel histórico completo 2015–2026 en CSV + log de diagnóstico. |

### 6.1 Estrategia de Validación por Fase

La validación es incremental: cada fase se verifica antes de avanzar a la siguiente. La estrategia concreta para cada fase es:

- **F1:** Verificar que el VWAP para el 2016-10-21 produce exactamente los mismos nemotécnicos y tasas que el scatter plot ya existente en el script original. Comparar $\tau$ calculado con el parsing del nemotécnico como cross-check.
- **F2:** Para el bono `TFIT16280428` (mayor volumen y más largo), calcular manualmente el calendario de cupones para el 2016-10-21 y comparar con la salida del M3.
- **F3:** La curva spot discreta para el 2016-10-21 debe ser monótonamente creciente (o al menos suavemente humped) para ser plausible. Los valores deben estar en el rango histórico de TES (4%–16%). Se visualiza el scatter de puntos spot sobre el scatter de yields VWAP.
- **F4:** Los nodos interpolados deben caer entre los puntos discretos más cercanos. Verificar que no hay valores fuera del rango de los extremos.
- **F5:** Verificar que el panel no tiene `NaN` inesperados, que los días `SIN_DATOS` están ausentes, y que la dimensión del resultado es ~2,736 filas.

### 6.2 Consideraciones de Rendimiento para F5

El barrido de 2,736 días es el paso más costoso. La estrategia de optimización es:

- Preprocesar todo el dataset una sola vez antes del loop (VWAP, merge, $\tau$). El loop per-día solo hace el slice por fecha y llama al bootstrapper.
- El bootstrapper en Python puro iterando sobre ~10–19 bonos por día es extremadamente rápido: cada día toma menos de 1 ms. El tiempo dominante es el overhead de pandas slicing en el loop. **Estimación: 2,736 días < 30 segundos en entorno local.**
- Si el rendimiento es insuficiente, el loop puede vectorizarse usando `groupby + apply`, o paralelizarse con `joblib.Parallel`.

---

## 7. Decisiones de Diseño y Justificaciones

### 7.1 Por qué VWAP y no precio/tasa de cierre

El mercado secundario TES es un mercado OTC relativamente ilíquido para algunos instrumentos y fechas. En un día con 50 operaciones de un bono, las primeras pueden reflejar condiciones de apertura y las últimas el cierre del mercado, con spreads bid-ask implícitos. El VWAP pondera cada operación por su volumen, dando más peso a las operaciones grandes (institucionales) que a las pequeñas (retail), produciendo una tasa más representativa del equilibrio de mercado para ese día. Es la práctica estándar en mercados de renta fija.

### 7.2 Por qué `dueDate` del CSV estático y no parsing del nemotécnico

El parsing del nemotécnico (extraer día, mes, año de los últimos 6 dígitos del código TFIT) es un método frágil: asume que la convención de nomenclatura no ha cambiado en 11 años y que no hay excepciones. El CSV `bonos_info_estatica.csv` contiene la fecha de vencimiento oficial extraída directamente de la API de la BVC — esa es la fuente autoritativa. El parsing del nemotécnico se mantiene solo como fallback de contingencia.

### 7.3 Por qué umbral de 30 días y no mayor

Un umbral mayor eliminaría más bonos y dejaría más huecos en la curva, aumentando la dependencia de la interpolación y reduciendo la precisión. Un umbral menor (por ejemplo, 7 días) podría dejar pares de bonos que generan inestabilidad numérica real. 30 días es el balance recomendado en el documento de proceso: suficiente para garantizar que el tramo nuevo de cada bono represente al menos $30/365 \approx 0.082$ años de nueva información, lo que mantiene la ecuación bien condicionada.

### 7.4 Por qué composición discreta y no continua

Los bonos TES colombianos se cotizan y liquidan en convención discreta anual (base 365). Los participantes del mercado colombiano (fondos de pensiones, bancos, tesorerías) trabajan en esta convención. Usar composición continua requeriría convertir todas las tasas al inicio y reconvertirlas al final, introduciendo una fuente adicional de error de redondeo sin ningún beneficio conceptual para el uso final de los datos. La coherencia con la convención de mercado es prioritaria.

### 7.5 Por qué interpolación lineal y no spline

Los *Unsmoothed Fama-Bliss Yields* son, por definición, los puntos discretos sin suavizar. La interpolación lineal preserva esta característica: produce una curva continua pero sin imponer suavidad artificial entre los nodos. Si en el futuro se desea una curva suavizada, el módulo M5 puede reemplazarse por un spline cúbico o un modelo Nelson-Siegel, sin afectar ningún otro módulo. Esta es precisamente la ventaja de la arquitectura modular.

---

## 8. Riesgos Identificados y Estrategias de Mitigación

| Riesgo | Severidad | Mitigación |
|---|---|---|
| VP residual negativo en algún bono por precio de mercado inconsistente | Media | Omitir ese bono, registrar en log, continuar con los restantes. Si el bono omitido deja un hueco > 5 años, marcar el día con flag `ADVERTENCIA`. |
| Tasa forward negativa (curva invertida extrema) | Baja | Permitir tasas forward negativas si son plausibles (hasta −5%). Si $f < -5\%$, tratar como VP residual negativo. |
| Días con menos de 3 bonos activos | Media | Marcar como `INSUFICIENTE`. Estos días producirán `NaN` en el panel. El modelo downstream deberá decidir si interpolar temporalmente o excluir. |
| Bonos con vencimiento en fecha no laborable (festivo colombiano) | Baja | Los bonos TES ajustan su fecha de pago al siguiente día hábil por convención. Documentar en el motor de flujos de caja y aplicar el ajuste si es necesario. |
| Discrepancia entre `dueDate` del CSV estático y fecha implícita en el nemotécnico | Baja | Usar siempre el CSV estático como fuente autoritativa. Agregar una celda de validación que detecte discrepancias > 5 días y las registre. |
| Nodo de interpolación fuera del rango de bonos activos ese día | Alta (frecuente) | Extrapolación lineal usando los dos puntos extremos más cercanos, con `NaN` si el nodo está más de 5 años fuera del bono más lejano. Registrar en log. |

---

## 9. Estado del Código Existente y Plan de Reutilización

El script `integrador_procesador.py` contiene varias piezas de código que se reutilizarán directamente en los nuevos módulos, con refactoring para mejorar su estructura y eficiencia:

| Código existente | Estado | Acción |
|---|---|---|
| Conexión y carga de las 3 SQLite + concat | Funcional, reutilizable | Migrar a función `load_all_databases()` en M1. Agregar manejo de errores. |
| `calculate_vwap()` con groupby | Funcional, bueno | Reutilizar directamente en M2 con ajuste menor para incluir suma de volumen. |
| Parsing de fecha de vencimiento desde nemotécnico | Funcional, solo como fallback | Mantener como función auxiliar en M2. No usar como fuente primaria. |
| Cálculo de `tau_anual` | Funcional, reutilizable | Integrar directamente en M2 tras el merge con info estática. |
| Matriz de distancias entre vencimientos | Exploratorio, O(N²) lento | Reemplazar con lógica vectorizada en M2 usando sort + comparación por pares adyacentes. |
| Visualización Plotly animada | Funcional, mantener aparte | Mantener como script de visualización independiente, alimentado por la salida del panel. |

### 9.1 Lo que NO está implementado aún

El código existente llega hasta la construcción del scatter de yields VWAP. Todo lo que viene después es nuevo:

- **Motor de flujos de caja (M3):** no existe. Debe construirse desde cero.
- **Motor de bootstrapping (M4):** no existe. Es el núcleo del proyecto — el bloque más complejo.
- **Módulo de interpolación (M5):** no existe. Relativamente simple una vez que M4 funciona.
- **Orquestador (M6):** no existe. Es el loop de integración final.

---

## 10. Próximos Pasos Inmediatos

Una vez aprobado este informe, el orden de construcción del código es el siguiente:

1. **Generar `config.py` (M0):** define todas las rutas y parámetros. Sin este archivo, ningún otro módulo puede ejecutarse.
2. **Generar `data_loader.py` (M1):** refactoring del código de carga existente. Validar que carga correctamente los 530,907 registros y los 29 bonos estáticos.
3. **Generar `preprocessor.py` (M2):** incluye VWAP (reutilizado), merge, $\tau$, filtro 30 días. Validar contra scatter plot existente para el 2016-10-21.
4. **Generar `cashflow_engine.py` (M3):** construir y validar manualmente los calendarios de flujos para 2–3 bonos de referencia.
5. **Generar `bootstrapper.py` (M4):** implementar, probar en modo debug con prints intermedios para el 2016-10-21, validar plausibilidad de la curva.
6. **Generar `interpolator.py` (M5):** implementar interpolación + extrapolación lineal. Validar que los 9 nodos producen valores en rango.
7. **Generar `orchestrator.py` (M6):** integrar todo el pipeline. Probar en modo prueba, luego activar modo producción para el barrido completo.
8. **Generar el panel histórico completo:** ejecutar M6 en producción. Tiempo estimado: < 5 minutos en Colab.

---

> **Decisiones tomadas para F5 (Orquestador)**
>
> Las siguientes decisiones fueron confirmadas y quedan registradas como especificación final:
>
> **1. Puntos spot discretos — SÍ se guardan.**
> El algoritmo producirá dos outputs diferenciados:
> - `curvas_panel_historico.csv` — panel principal con las tasas interpoladas a los 9 nodos estándar (formato wide).
> - `curvas_discretas_historico.csv` — panel secundario con los $N$ puntos crudos del bootstrapping por día ($\tau$ exacto de cada bono activo, $R_{\text{spot}}$ antes de interpolación). Permite auditoría, detección de anomalías y cambio futuro del método de interpolación sin re-correr el bootstrapping.
>
> **2. Formato del panel — Wide.**
> Una fila por fecha, una columna por nodo de plazo. Estructura:
> ```
> fecha | r_1yr | r_2yr | r_3yr | r_5yr | r_7yr | r_10yr | r_15yr | r_20yr | r_30yr
> ```
> Justificación: es el formato nativo para modelos de series de tiempo, VAR y PCA de curvas. El formato long se puede generar en cualquier momento con un `melt()` de pandas cuando se necesite para visualización.
>
> **3. Fecha de validación — Modular vía `config.py`.**
> La fecha de validación visual en F3 se controla desde el parámetro `FECHA_PRUEBA` en `config.py`. Valor inicial: `'2016-10-21'` — fecha para la cual ya existe un scatter plot de referencia en el script original, lo que permite comparación directa. Cambiar la fecha de validación es editar una sola línea en `config.py`, sin tocar ningún otro módulo.

---

*Informe preparado como documento de planificación previo a la construcción del código.*
*Proyecto Semillero · Curvas de Rendimiento TES · BVC 2015–2026*