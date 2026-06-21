# Contexto del Dataset — Información Estática de Bonos TFIT · BVC · 2015–2026

---

## Descripción general

Este dataset complementa los datos de operaciones de mercado secundario con la
información estructural de cada uno de los 29 bonos TFIT únicos que componen
el universo de instrumentos negociados en la BVC entre 2015 y 2026. Contiene
una fila por bono y 15 variables que describen sus características de emisión,
condiciones financieras y estado actual.

---

## Estructura del dataset

| Variable | Tipo | Descripción |
|---|---|---|
| `nemotecnico` | TEXT | Identificador del bono en el mercado (llave de unión con operaciones) |
| `symbol` | TEXT | Igual al nemotécnico — identificador interno de la BVC |
| `code` | TEXT | Código de la serie de emisión |
| `isin` | TEXT | Código internacional de identificación del valor (ISIN) |
| `securityClass` | TEXT | Clase del instrumento — todos son "TITULOS TES" |
| `issueDate` | TEXT (YYYY-MM-DD) | Fecha de emisión del bono |
| `dueDate` | TEXT (YYYY-MM-DD) | Fecha de vencimiento del bono |
| `couponRate` | TEXT | Tasa cupón nominal anual en porcentaje |
| `couponFrequency` | TEXT | Frecuencia de pago del cupón |
| `typeRate` | TEXT | Tipo de tasa — todos son "Nominal" |
| `base` | TEXT | Base de cálculo de días — todos usan 365 |
| `issuedValue` | REAL | Monto total emitido en pesos colombianos |
| `minimumLot` | REAL | Lote mínimo de negociación en pesos colombianos |
| `currency` | TEXT | Moneda — todos en COP |
| `status` | TEXT | Estado del bono — "Activo" o vacío si ya venció |

---

## Características del universo de instrumentos

Los 29 bonos cubren un rango de emisión que va desde 2005 hasta 2025, y fechas
de vencimiento que van desde 2015 hasta 2058 — un horizonte de más de 40 años.
Todos son títulos de tasa fija nominal, denominados en pesos colombianos, con
pago de cupón anual y base de 365 días, lo que hace el universo homogéneo en
sus condiciones estructurales.

**Tasas cupón:** oscilan entre 5.0% y 13.25%, con la mayoría de los bonos más
recientes ubicados en rangos más altos (11%–13.25%) reflejando el entorno de
tasas de los años 2022–2025. Los bonos emitidos entre 2012 y 2020 concentran
tasas entre 5.75% y 7.75%.

**Estado:** 18 bonos están activos a la fecha de extracción. Los 11 restantes
ya vencieron — sus fechas de vencimiento van desde 2015 hasta 2022. Estos bonos
vencidos tienen campos como `issuedValue`, `minimumLot` e `isin` vacíos, ya que
la BVC no conserva esa información para instrumentos fuera de circulación.

**Monto emitido:** entre los bonos activos con información disponible, el monto
emitido varía ampliamente — desde COP 1,000 millones (TFIT34130358) hasta
COP 741,800 millones (TFIT16280428), lo que refleja diferencias significativas
en liquidez esperada entre instrumentos.

**Lote mínimo:** uniforme en COP 500 millones para todos los bonos activos.

---

## Relación con el dataset de operaciones

La variable `nemotecnico` es la llave de unión entre este dataset y el de
operaciones de mercado secundario. Con ella se pueden calcular variables
derivadas clave para el modelado, como el plazo al vencimiento en cada fecha
de negociación (`dueDate - fecha_neg`), la duración modificada, o agrupar
operaciones por cohorte de emisión.

---

## Consideraciones para los procesos siguientes

**Limpieza:** los 11 bonos vencidos tienen campos vacíos en `issuedValue`,
`minimumLot` e `isin`. Dependiendo del modelo, estos valores pueden imputarse
con cero, dejarse como nulos, o simplemente excluirse del análisis si solo
se trabaja con bonos activos.

**Análisis exploratorio:** la combinación de `issueDate`, `dueDate` y
`couponRate` permite construir la curva de rendimientos en cualquier fecha
histórica del dataset de operaciones, cruzando el plazo residual de cada
bono con la tasa negociada ese día.

**Construcción del modelo:** `couponRate` y `dueDate` son variables
estructurales que no cambian en el tiempo — se unen al dataset de operaciones
como atributos fijos del instrumento, no como series temporales.

