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