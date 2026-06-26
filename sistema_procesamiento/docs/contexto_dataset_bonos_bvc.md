# Contexto del Dataset — Bonos Soberanos TES (TFIT) · BVC · 2015–2026

---

## Descripción general

El dataset contiene operaciones de mercado secundario de bonos soberanos colombianos
denominados TES (Títulos de Tesorería), identificados con el prefijo `TFIT`, negociados
en la Bolsa de Valores de Colombia (BVC) en el segmento A2A (Agente a Agente).

Cubre un período de aproximadamente 11 años, desde el 2 de enero de 2015 hasta el
20 de marzo de 2026, con un total de **530,907 operaciones** distribuidas en
**2,736 días hábiles** de negociación.

---

## Estructura del dataset

Cada registro representa una operación individual ejecutada en el mercado secundario.
Las variables disponibles son las siguientes:

| Variable | Tipo | Descripción |
|---|---|---|
| `fecha_neg` | TEXT (YYYY-MM-DD) | Fecha de negociación |
| `nemotecnico` | TEXT | Identificador único del bono (e.g. `TFIT16280428`) |
| `time` | TEXT | Hora de la operación |
| `rate` | REAL | Tasa de rendimiento negociada (yield) |
| `price` | REAL | Precio sucio de la operación |
| `volume` | REAL | Volumen negociado en pesos colombianos |

El dataset no presenta valores nulos en ninguna de sus variables numéricas
(`rate`, `price`, `volume`), lo que indica que todas las operaciones registradas
están completas.

---

## Cobertura por período

| Período | Operaciones | Días hábiles | Bonos únicos |
|---|---|---|---|
| 2015 – 2018 | 281,969 | 972 | 16 |
| 2019 – 2022 | 150,210 | 978 | 15 |
| 2023 – 2026 | 98,728 | 786 | 19 |
| **Total** | **530,907** | **2,736** | **29** |

La mayor concentración de operaciones se encuentra en el período 2015–2018, lo que
refleja una mayor actividad en el mercado secundario de TES durante esos años.
El número de bonos únicos varía entre períodos — de 15 a 19 — lo que es esperado
dado que los TES tienen fechas de vencimiento definidas y nuevas emisiones entran
al mercado de forma periódica.

---

## Universo de instrumentos

El dataset cubre **29 bonos TFIT únicos** a lo largo de todo el período. El nemotécnico
de cada bono codifica información sobre su estructura: el prefijo `TFIT` identifica
el instrumento como un TES de tasa fija, seguido de dígitos que contienen información
sobre la tasa cupón y la fecha de vencimiento. Por ejemplo, `TFIT16280428` corresponde
a un bono con cupón del 16% que vence el 28 de abril de 2028.

No todos los bonos se negocian en todos los días hábiles — la presencia de un bono
en el dataset para una fecha dada indica que efectivamente hubo operaciones ese día,
no simplemente que el bono existía.

---

## Completitud y días sin datos

De los **3,124 días calendario consultados** (días hábiles del período):

- **2,736 días** tienen al menos una operación registrada → `COMPLETO`
- **388 días** no presentaron operaciones → `SIN_DATOS`

Los días `SIN_DATOS` corresponden principalmente a festivos del mercado colombiano
o días con baja liquidez en los que ningún bono TFIT registró operaciones en el
segmento A2A. Estos días están identificados en el log de progreso y pueden
excluirse o imputarse según la estrategia de modelado que se adopte.

---

## Consideraciones para los procesos siguientes

**Limpieza y preprocesamiento:**
El dataset no requiere imputación de valores nulos en las variables numéricas.
Sin embargo, se recomienda verificar la consistencia de los registros duplicados
potenciales en la llave primaria (`fecha_neg`, `nemotecnico`, `time`), ya que
operaciones ejecutadas en el mismo segundo podrían generar colisiones. También
es importante definir el tratamiento de los 388 días `SIN_DATOS` — si se incluyen
como filas con valores cero, se excluyen del análisis, o se interpolan.

**Análisis exploratorio:**
La variable `rate` (tasa de rendimiento) es probablemente la más relevante para
análisis de curva de rendimientos, dado que refleja el precio del dinero en
distintos plazos. La combinación de `fecha_neg` + `nemotecnico` permite construir
series de tiempo por instrumento o por plazo al vencimiento. El volumen puede
usarse como proxy de liquidez para ponderar observaciones.

**Construcción del modelo:**
La naturaleza del dataset es de panel — múltiples instrumentos observados en
múltiples fechas — lo que permite tanto modelos por serie individual como modelos
que exploten la estructura transversal de la curva de rendimientos. Los 388 días
sin datos generan irregularidades en las series temporales que deben considerarse
en la especificación del modelo, especialmente si se usan rezagos o diferencias.
La variación en el número de bonos activos por período implica que el panel es
no balanceado.

---

## Almacenamiento

Los datos están distribuidos en tres bases de datos SQLite independientes,
una por período, almacenadas en Google Drive:

| Archivo | Período |
|---|---|
| `BVC_Bonos_2015_2018.db` | 2015-01-02 → 2018-12-31 |
| `BVC_Bonos_2019_2022.db` | 2019-01-01 → 2022-12-31 |
| `BVC_Bonos_2023_2026.db` | 2023-01-01 → 2026-03-20 |

Cada base de datos contiene dos tablas: `operaciones` con los datos de mercado
y `log_progreso` con el registro de días procesados. Para trabajar con el dataset
completo se recomienda unir las tres tablas `operaciones` mediante `pd.concat()`
o mediante una consulta `ATTACH` en SQLite, como se hace en el script
`integrador_procesador.py`.

