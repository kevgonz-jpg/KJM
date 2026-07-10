# Guía de Configuración — Colaborador
## Proyecto: Curvas de Rendimiento Cero-Cupón TES · BVC 2015–2026

---

## ⚠️ Importante: qué SÍ y qué NO viaja por GitHub

El `.gitignore` del proyecto excluye deliberadamente archivos pesados o
regenerables. Esto significa que clonar el repo **no es suficiente** para
ejecutar el proyecto — faltan piezas que hay que añadir manualmente.

| Carpeta/archivo | ¿En GitHub? | Acción del colaborador |
|---|---|---|
| `src/*.py` | ✓ Sí | Ya viene en el clone |
| `notebooks/*.ipynb` | ✓ Sí | Ya viene en el clone |
| `docs/*.md` | ✓ Sí | Ya viene en el clone |
| `requirements.txt` | ✓ Sí | Ya viene en el clone |
| `.venv/` | ✗ No | Debe crearlo localmente (Paso 2) |
| `datos/*.db`, `datos/*.csv` | ✗ No | Debe conseguirlos aparte (Paso 4) |
| `outputs/` | ✗ No | Se regenera corriendo el pipeline (Paso 5) |

---

## Paso 1 — Clonar y ubicar el subproyecto

Si el repositorio contiene otras partes además de este proyecto:

```bash
git clone <url-del-repositorio>
cd <nombre-repo>/TES_curvas
```

> Ajusta la ruta según cómo esté organizado el repo. Lo importante es
> terminar parado dentro de la carpeta que contiene `src/`, `datos/`, etc.

---

## Paso 2 — Crear y activar el entorno virtual

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

Si PowerShell bloquea la activación con un error de políticas de ejecución:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

El prompt debe mostrar `(.venv)` al inicio una vez activado.

---

## Paso 3 — Instalar dependencias exactas

El `requirements.txt` fue generado con `pip freeze`, así que instala
las versiones exactas usadas en el desarrollo original:

```bash
pip install -r requirements.txt
```

> Si alguna versión falla por incompatibilidad con su sistema operativo
> o versión de Python, puede instalar sin versiones fijas:
> `pip install pandas numpy scipy plotly matplotlib seaborn ipykernel jupyter`

---

## Paso 4 — Conseguir los datos (el paso que NO está en git)

Los 4 archivos de datos pesan en total ~65 MB y fueron excluidos
deliberadamente del repositorio. Debes compartirlos a tu compañero
por un canal aparte — Google Drive, un zip por correo, USB, etc.

Los archivos necesarios son:

```
datos/
├── BVC_Bonos_2015_2018.db
├── BVC_Bonos_2019_2022.db
├── BVC_Bonos_2023_2026.db
├── bonos_info_estatica.csv
├── TIB.csv
└── IBR.csv
```

Tu compañero debe colocarlos exactamente en la carpeta `datos/` dentro
de su copia local del proyecto — las rutas en `config.py` son relativas
a la raíz del proyecto, así que funcionan sin importar en qué computador
estén, siempre que los archivos estén en `datos/`.

---

## Paso 5 — Verificar que todo está en orden

Con el `.venv` activo y los datos en su lugar:

```bash
python verificar_entorno.py
```

Debe mostrar todos los `✓`. Si algo falla, el mensaje de error indica
exactamente qué archivo falta o qué paquete no se instaló.

---

## Paso 6 — Regenerar los outputs (si los necesita)

Los archivos de `outputs/` tampoco viajan por git — son resultados
derivados, no fuente de verdad. Se regeneran corriendo el pipeline:

```bash
python src/orchestrator.py
```

> Verificar primero en `src/config.py` si `MODO = 'prueba'` o
> `'produccion'`. El modo producción genera el panel histórico
> completo (2,736 días) en menos de 2 minutos.

---

## Resumen — los 5 comandos esenciales

```bash
git clone <url>
cd TES_curvas
python -m venv .venv && .venv\Scripts\activate    # Windows
pip install -r requirements.txt
python verificar_entorno.py
```

*(sustituir la activación por `source .venv/bin/activate` en macOS/Linux)*

---

## Recomendación

Vale la pena pegar el contenido de esta guía directamente en el
`README.md` del proyecto — actualmente está vacío. Así cualquier
futuro colaborador (incluido tu yo del futuro en otra máquina) tiene
las instrucciones sin necesidad de preguntarte.