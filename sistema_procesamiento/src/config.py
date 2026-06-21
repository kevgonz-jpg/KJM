# =============================================================================
# config.py — M0: Configuración Central
# Proyecto: Curvas de Rendimiento Cero-Cupón TES · BVC 2015-2026
# =============================================================================
# Este es el único archivo que tiene conciencia del entorno y las rutas.
# Todos los demás módulos importan sus parámetros desde aquí.
# Para cambiar cualquier parámetro del sistema, este es el único lugar
# donde hay que tocar algo.
# =============================================================================

import os

# -----------------------------------------------------------------------------
# 1. ENTORNO
# -----------------------------------------------------------------------------
# Controla el modo de ejecución del algoritmo.
# 'prueba'     → procesa solo FECHA_PRUEBA, muestra resultados intermedios
# 'produccion' → itera sobre todos los días hábiles disponibles (2,736 días)

MODO = 'produccion'
FECHA_PRUEBA = '2016-10-21'  # fecha de validación visual — modular

# -----------------------------------------------------------------------------
# 2. RUTAS
# -----------------------------------------------------------------------------
# Todas las rutas se construyen relativamente desde la ubicación de este
# archivo. Esto garantiza que el proyecto funciona sin importar dónde esté
# la carpeta TES_curvas en el sistema de archivos.

# Raíz del proyecto — carpeta que contiene src/, datos/, outputs/, etc.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Carpeta de datos
DATOS_DIR = os.path.join(BASE_DIR, 'datos')

# Bases de datos SQLite
RUTA_DB_2015_2018 = os.path.join(DATOS_DIR, 'BVC_Bonos_2015_2018.db')
RUTA_DB_2019_2022 = os.path.join(DATOS_DIR, 'BVC_Bonos_2019_2022.db')
RUTA_DB_2023_2026 = os.path.join(DATOS_DIR, 'BVC_Bonos_2023_2026.db')

# Información estática de bonos
RUTA_INFO_ESTATICA = os.path.join(DATOS_DIR, 'bonos_info_estatica.csv')

# Carpeta de outputs
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')

# Archivos de salida
RUTA_PANEL_HISTORICO   = os.path.join(OUTPUTS_DIR, 'curvas_panel_historico.csv')
RUTA_PANEL_DISCRETO    = os.path.join(OUTPUTS_DIR, 'curvas_discretas_historico.csv')
RUTA_LOG_BOOTSTRAPPING = os.path.join(OUTPUTS_DIR, 'log_bootstrapping.csv')
RUTA_CURVAS_DISCRETAS  = os.path.join(OUTPUTS_DIR, 'curva_discreta')  # subcarpeta

# -----------------------------------------------------------------------------
# 3. PARÁMETROS DEL ALGORITMO
# -----------------------------------------------------------------------------

# Umbral de estabilidad numérica entre vencimientos consecutivos (días).
# Si dos bonos vencen con menos de este umbral de diferencia, se elimina
# el de menor volumen histórico para evitar inestabilidad en la resolución
# de la tasa forward.
UMBRAL_ESTABILIDAD_DIAS = 30

# Nodos de interpolación estándar (en años).
# La curva spot discreta se interpola a estos plazos fijos para generar
# un panel de series temporales homogéneo consumible por modelos de ML.
NODOS_INTERPOLACION = [1, 2, 3, 5, 7, 10, 15, 20, 30]

# Base de cálculo de días — uniforme para todos los bonos TFIT.
BASE_DIAS = 365

# Rango de tasas forward considerado plausible.
# Fuera de este rango se trata como inconsistencia de datos.
F_MIN = -0.05   # −5%
F_MAX =  0.50   # 50%

# Máxima distancia en años fuera del rango de bonos activos
# para permitir extrapolación. Más allá de esto se registra NaN.
MAX_EXTRAPOLACION_ANOS = 5.0

# -----------------------------------------------------------------------------
# 4. VALIDACIÓN DE RUTAS AL IMPORTAR
# -----------------------------------------------------------------------------
# Verifica que los archivos críticos existen en el momento en que cualquier
# módulo importa config. Falla rápido con un mensaje claro en lugar de
# producir errores crípticos más adelante.

def _validar_rutas():
    archivos_criticos = {
        'DB 2015-2018':   RUTA_DB_2015_2018,
        'DB 2019-2022':   RUTA_DB_2019_2022,
        'DB 2023-2026':   RUTA_DB_2023_2026,
        'Info estática':  RUTA_INFO_ESTATICA,
    }
    errores = []
    for nombre, ruta in archivos_criticos.items():
        if not os.path.exists(ruta):
            errores.append(f"  ✗  {nombre}: {ruta}")

    if errores:
        raise FileNotFoundError(
            "\n[config.py] Archivos críticos no encontrados:\n" +
            "\n".join(errores) +
            "\n\nVerifica que los datos están en la carpeta 'datos/' del proyecto."
        )

    # Crear carpeta de outputs y subcarpeta si no existen
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(RUTA_CURVAS_DISCRETAS, exist_ok=True)

_validar_rutas()

# -----------------------------------------------------------------------------
# 5. RESUMEN DE CONFIGURACIÓN ACTIVA (visible al importar en modo prueba)
# -----------------------------------------------------------------------------

if MODO == 'prueba':
    print("=" * 55)
    print("  CONFIG.PY — configuración activa")
    print("=" * 55)
    print(f"  Modo:              {MODO}")
    print(f"  Fecha de prueba:   {FECHA_PRUEBA}")
    print(f"  Umbral estabilidad:{UMBRAL_ESTABILIDAD_DIAS} días")
    print(f"  Nodos:             {NODOS_INTERPOLACION}")
    print(f"  Base días:         {BASE_DIAS}")
    print(f"  Rango f válido:    [{F_MIN}, {F_MAX}]")
    print(f"  Raíz proyecto:     {BASE_DIR}")
    print("=" * 55)