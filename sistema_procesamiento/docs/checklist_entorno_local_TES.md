# ✅ Checklist — Entorno Local para Proyecto Curvas TES
### Windows 11 + VSCode + Python

---

## FASE 1 — Verificar y preparar Python

### 1.1 Confirmar que Python está bien instalado

Abre una terminal de Windows (busca `cmd` o `PowerShell` en el menú inicio) y ejecuta:

```bash
python --version
```

**Resultado esperado:** algo como `Python 3.10.x` o superior.

Si ves un error o una versión menor a 3.9, descarga e instala Python desde
https://www.python.org/downloads/ — marca la casilla **"Add Python to PATH"**
durante la instalación. Eso es crítico.

```bash
# También verifica que pip (gestor de paquetes) está disponible:
pip --version
```

---

## FASE 2 — Instalar y configurar VSCode

### 2.1 Instalar VSCode (si no lo tienes)

Descarga desde: https://code.visualstudio.com/

### 2.2 Instalar extensiones esenciales

Abre VSCode. Ve al panel de extensiones (ícono de bloques en la barra izquierda,
o `Ctrl+Shift+X`) e instala estas cuatro:

| Extensión | Para qué sirve |
|---|---|
| **Python** (Microsoft) | Soporte completo de Python: autocompletado, linting, debugging |
| **Pylance** (Microsoft) | Motor de análisis de tipos — mejora el autocompletado enormemente |
| **Jupyter** (Microsoft) | Permite abrir y correr notebooks `.ipynb` dentro de VSCode |
| **GitLens** (opcional) | Control de versiones mejorado — útil pero no crítico ahora |

---

## FASE 3 — Estructura de carpetas del proyecto

### 3.1 Crear la carpeta raíz del proyecto

Elige una ubicación limpia en tu máquina. Recomendación:

```
C:\proyectos\TES_curvas\
```

> Evita rutas con espacios o caracteres especiales (tildes, ñ).
> NO uses el escritorio ni Documentos — son rutas largas y problemáticas.

### 3.2 Estructura de carpetas a crear

Dentro de `TES_curvas\`, crea exactamente esta estructura:

```
TES_curvas/
│
├── datos/                        ← aquí van las 3 bases de datos SQLite
│   ├── BVC_Bonos_2015_2018.db
│   ├── BVC_Bonos_2019_2022.db
│   ├── BVC_Bonos_2023_2026.db
│   └── bonos_info_estatica.csv
│
├── src/                          ← aquí va todo el código fuente
│   ├── config.py                 ← M0: parámetros y rutas
│   ├── data_loader.py            ← M1: carga de SQLite
│   ├── preprocessor.py           ← M2: VWAP, tau, filtros
│   ├── cashflow_engine.py        ← M3: calendario de flujos
│   ├── bootstrapper.py           ← M4: motor Fama-Bliss
│   ├── interpolator.py           ← M5: interpolación a nodos
│   └── orchestrator.py           ← M6: loop principal
│
├── notebooks/                    ← para exploración y validación visual
│   ├── 01_exploracion_datos.ipynb
│   ├── 02_validacion_bootstrapper.ipynb
│   └── 03_visualizacion_curvas.ipynb
│
├── outputs/                      ← salidas del algoritmo
│   ├── curvas_panel_historico.csv
│   ├── log_bootstrapping.csv
│   └── curva_discreta/           ← subcarpeta para curvas por día (modo prueba)
│
├── tests/                        ← pruebas unitarias (opcional pero recomendado)
│   ├── test_cashflow_engine.py
│   └── test_bootstrapper.py
│
├── .venv/                        ← entorno virtual Python (se crea en Fase 4)
├── .gitignore                    ← archivos a excluir de git
├── requirements.txt              ← lista de dependencias
└── README.md                     ← descripción del proyecto
```

> Puedes crear todas estas carpetas manualmente en el explorador de Windows,
> o con los comandos de la Fase 3.3 abajo.

### 3.3 Crear la estructura desde PowerShell (opcional — más rápido)

Abre PowerShell, navega a `C:\proyectos\` y ejecuta:

```powershell
# Crear carpeta raíz y moverse a ella
mkdir TES_curvas
cd TES_curvas

# Crear subcarpetas
mkdir datos
mkdir src
mkdir notebooks
mkdir outputs
mkdir outputs\curva_discreta
mkdir tests

# Crear archivos vacíos iniciales
New-Item requirements.txt -ItemType File
New-Item README.md -ItemType File
New-Item .gitignore -ItemType File
New-Item src\config.py -ItemType File
New-Item src\data_loader.py -ItemType File
New-Item src\preprocessor.py -ItemType File
New-Item src\cashflow_engine.py -ItemType File
New-Item src\bootstrapper.py -ItemType File
New-Item src\interpolator.py -ItemType File
New-Item src\orchestrator.py -ItemType File
```

---

## FASE 4 — Crear el entorno virtual (MUY IMPORTANTE)

Un entorno virtual aísla las dependencias de este proyecto de los demás
proyectos Python de tu máquina. Es la práctica estándar y evita conflictos
de versiones.

### 4.1 Crear el entorno virtual

Desde PowerShell, dentro de `C:\proyectos\TES_curvas\`:

```powershell
python -m venv .venv
```

Esto crea la carpeta `.venv\` con una copia aislada de Python.

### 4.2 Activar el entorno virtual

```powershell
.venv\Scripts\activate
```

Sabrás que está activo porque el prompt cambia a:

```
(.venv) PS C:\proyectos\TES_curvas>
```

> **Importante:** debes activar el entorno virtual CADA VEZ que abras
> una terminal nueva para trabajar en este proyecto.

### 4.3 Configurar VSCode para usar el entorno virtual

1. Abre la carpeta del proyecto en VSCode: `Archivo → Abrir carpeta → TES_curvas`
2. Presiona `Ctrl+Shift+P` para abrir la paleta de comandos
3. Escribe `Python: Select Interpreter`
4. Selecciona el intérprete que dice `.venv` — algo como
   `Python 3.x.x ('.venv': venv)`

A partir de aquí, VSCode usará automáticamente el entorno virtual para
todos los archivos del proyecto.

---

## FASE 5 — Instalar las dependencias del proyecto

### 5.1 Instalar los paquetes necesarios

Con el entorno virtual activado en PowerShell:

```powershell
pip install pandas numpy scipy plotly matplotlib seaborn ipykernel jupyter
```

| Paquete | Uso en el proyecto |
|---|---|
| `pandas` | Manipulación de DataFrames — columna vertebral del proyecto |
| `numpy` | Operaciones numéricas y matemáticas del bootstrapper |
| `scipy` | Resolución numérica de ecuaciones (fallback si el despeje algebraico falla) |
| `plotly` | Visualizaciones interactivas (ya usado en el script original) |
| `matplotlib` | Visualizaciones estáticas para validación |
| `seaborn` | Scatter plots de validación (ya usado en el script original) |
| `ipykernel` | Permite que VSCode use el entorno virtual en notebooks Jupyter |
| `jupyter` | Soporte de notebooks dentro de VSCode |

### 5.2 Guardar las dependencias en requirements.txt

```powershell
pip freeze > requirements.txt
```

Esto genera el archivo `requirements.txt` con todas las versiones exactas.
Cualquier colaborador puede replicar el entorno exacto con:

```powershell
pip install -r requirements.txt
```

---

## FASE 6 — Bajar los datos de Google Drive a local

### 6.1 Descargar los archivos desde Google Drive

Desde Google Drive, descarga manualmente estos 4 archivos a la carpeta
`C:\proyectos\TES_curvas\datos\`:

```
BVC_Bonos_2015_2018.db
BVC_Bonos_2019_2022.db
BVC_Bonos_2023_2026.db
bonos_info_estatica.csv
```

> Los archivos `.db` son bases de datos SQLite — se abren directamente
> con Python, no necesitas ningún programa adicional.

### 6.2 Verificar que los archivos están completos

Abre PowerShell y ejecuta:

```powershell
# Verificar tamaños de los archivos .db (deben ser varios MB cada uno)
Get-ChildItem C:\proyectos\TES_curvas\datos\
```

---

## FASE 7 — Configurar .gitignore

### 7.1 Contenido del .gitignore

Abre el archivo `.gitignore` en VSCode y pega esto:

```
# Entorno virtual
.venv/

# Datos pesados (no se suben a git — van en Drive)
datos/*.db
datos/*.csv

# Outputs generados
outputs/

# Archivos de Python compilado
__pycache__/
*.pyc
*.pyo

# Configuraciones locales de VSCode
.vscode/settings.json

# Jupyter checkpoints
.ipynb_checkpoints/
```

> Esto evita que subas accidentalmente las bases de datos (500MB+)
> o el entorno virtual a git.

---

## FASE 8 — Verificar que todo funciona

### 8.1 Prueba de humo — crear un script de verificación

Crea un archivo `verificar_entorno.py` en la raíz del proyecto y pega esto:

```python
# verificar_entorno.py
import sys
import sqlite3
import pandas as pd
import numpy as np
import scipy
import plotly
import matplotlib

print("=" * 50)
print("VERIFICACIÓN DEL ENTORNO")
print("=" * 50)
print(f"Python:     {sys.version}")
print(f"pandas:     {pd.__version__}")
print(f"numpy:      {np.__version__}")
print(f"scipy:      {scipy.__version__}")
print(f"plotly:     {plotly.__version__}")
print(f"matplotlib: {matplotlib.__version__}")
print()

# Verificar acceso a las bases de datos
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATOS_DIR = os.path.join(BASE_DIR, 'datos')

dbs = [
    'BVC_Bonos_2015_2018.db',
    'BVC_Bonos_2019_2022.db',
    'BVC_Bonos_2023_2026.db',
]

print("BASES DE DATOS:")
for db in dbs:
    path = os.path.join(DATOS_DIR, db)
    if os.path.exists(path):
        conn = sqlite3.connect(path)
        n = pd.read_sql("SELECT COUNT(*) as n FROM operaciones", conn).iloc[0,0]
        conn.close()
        print(f"  ✓  {db}  →  {n:,} operaciones")
    else:
        print(f"  ✗  {db}  →  ARCHIVO NO ENCONTRADO")

csv_path = os.path.join(DATOS_DIR, 'bonos_info_estatica.csv')
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print(f"\nINFO ESTÁTICA:")
    print(f"  ✓  bonos_info_estatica.csv  →  {len(df)} bonos")
else:
    print(f"\n  ✗  bonos_info_estatica.csv  →  ARCHIVO NO ENCONTRADO")

print()
print("=" * 50)
print("Si todos los ítems muestran ✓, el entorno está listo.")
print("=" * 50)
```

### 8.2 Correr la verificación

En la terminal integrada de VSCode (`Ctrl+ñ` o `Terminal → Nueva terminal`):

```powershell
python verificar_entorno.py
```

**Resultado esperado:**
```
==================================================
VERIFICACIÓN DEL ENTORNO
==================================================
Python:     3.11.x ...
pandas:     2.x.x
numpy:      1.x.x
...

BASES DE DATOS:
  ✓  BVC_Bonos_2015_2018.db   →  281,969 operaciones
  ✓  BVC_Bonos_2019_2022.db   →  150,210 operaciones
  ✓  BVC_Bonos_2023_2026.db   →   98,728 operaciones

INFO ESTÁTICA:
  ✓  bonos_info_estatica.csv  →  29 bonos

==================================================
Si todos los ítems muestran ✓, el entorno está listo.
==================================================
```

---

## FASE 9 — Flujo de trabajo diario (cómo trabajar en este proyecto)

Una vez que el entorno está configurado, el flujo de trabajo para cada
sesión de desarrollo es este:

```
1. Abrir VSCode
       ↓
2. Abrir la carpeta TES_curvas (File → Open Folder)
       ↓
3. Abrir terminal integrada (Ctrl+ñ)
       ↓
4. Activar entorno virtual:   .venv\Scripts\activate
       ↓
5. Trabajar en los archivos src/*.py
       ↓
6. Validar en notebooks/  (abrir el .ipynb correspondiente,
   seleccionar kernel .venv, correr celdas)
       ↓
7. Cuando termines, cerrar VSCode normalmente.
   El entorno virtual se desactiva solo al cerrar la terminal.
```

---

## RESUMEN — Orden de ejecución de esta checklist

| # | Tarea | Tiempo estimado |
|---|---|---|
| 1 | Verificar Python (`python --version`) | 2 min |
| 2 | Instalar extensiones de VSCode | 5 min |
| 3 | Crear estructura de carpetas | 5 min |
| 4 | Crear y activar entorno virtual | 3 min |
| 5 | Configurar intérprete en VSCode | 2 min |
| 6 | Instalar dependencias con pip | 5 min |
| 7 | Descargar datos de Google Drive | 10 min (depende de internet) |
| 8 | Configurar .gitignore | 2 min |
| 9 | Correr script de verificación | 2 min |
| **TOTAL** | **Entorno listo para programar** | **~35 min** |

---

*Una vez que el script de verificación muestre todos los ✓,
el entorno está 100% listo y podemos arrancar con config.py (M0).*
