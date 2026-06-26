
# 📔 Lección: El Comando de "La Llave Volátil"
**Proyecto:** Curvas de Rendimiento TES (BVC 2015–2026)

---

## 🛠️ El Comando de Prueba
python -c "import sys; sys.path.insert(0, 'src'); import config"

## 🧩 Desglose Técnico del Proceso

**1. python -c**
Ejecuta código directamente en la terminal sin necesidad de un archivo físico. Crea un proceso temporal para probar el sistema.

**2. import sys**
Carga el módulo que permite a Python ver y modificar su entorno de búsqueda de archivos.

**3. sys.path.insert(0, 'src')**
Crea la "Llave Maestra". Añade la carpeta 'src/' al mapa de búsqueda con prioridad máxima para que Python encuentre tus módulos.

**4. import config**
Accede a la Central de Configuración (M0) para validar que el archivo existe y no tiene errores de código.

---

## 🚌 Analogía de la Logística TES

**La Llave (sys.path)**
Es el permiso de acceso. Sin ella, Python no tiene permitido entrar a la carpeta 'src/' para buscar tus scripts de procesamiento.

**La Central (config.py)**
Es el tablero de despacho. No guarda los bonos, pero tiene las direcciones exactas de cada base de datos SQLite y archivos CSV.

**El GPS (os.path)**
Es la herramienta técnica que construye rutas que funcionan igual en cualquier sistema operativo (Windows o Linux).

---

## 💡 Aprendizajes Críticos

**Llave Temporal**
El acceso creado en la terminal muere cuando el comando termina. El Orquestador (M6) debe crear esta llave de forma permanente al arrancar el programa.

**Rutas de Código vs Rutas de Datos**
sys.path se usa para encontrar archivos de Python (.py). En cambio, os.path se usa dentro de la configuración para encontrar archivos de datos (.db).

**Escudo de Errores**
Este comando activa la validación de rutas, impidiendo que el algoritmo arranque si falta alguna base de datos de los 11 años de historial.

---
*Nota: Estructura modular verificada. Listo para la fase de Carga de Datos (M1).*

# 📔 Lección: El Cargador de Datos (M1) y la Arquitectura Profesional
**Proyecto:** Curvas de Rendimiento TES (BVC 2015–2026)

---

## 🏗️ Conceptos de Arquitectura Escalable

En esta fase, el proyecto deja de ser un script suelto y se convierte en un sistema desacoplado. Estos son los pilares aplicados en el Módulo M1:

**1. Desacoplamiento (Decoupling)**
El módulo `data_loader.py` no sabe "dónde" están los archivos en tu disco duro. Simplemente confía en que el "donante" (`config.py`) le entregará las coordenadas correctas. Esto permite que, si mueves las bases de datos de carpeta, solo tengas que actualizar un archivo (`config.py`) y no todo el código de carga.

**2. El Principio de "Donación" de Suministros**
A través de `import config`, el Módulo M1 accede al diccionario de rutas. Las variables como `config.RUTA_DB_2015_2018` son los insumos que permiten al operario (M1) encontrar los almacenes de datos (SQLite) sin tener rutas "quemadas" (hardcoded) en el código.



**3. Estrategia de Carga en Memoria**
Dado que el dataset completo (~530,907 registros) ocupa poco espacio en RAM (aprox. 80 MB), el diseño opta por cargar todo una sola vez al inicio. Esto elimina la necesidad de conectarse y desconectarse de la base de datos miles de veces durante el barrido histórico, acelerando drásticamente el algoritmo.

---

## 🛠️ Detalles Técnicos de Robustez

**Protocolo de "Falla Rápido" (Fail-Fast)**
El uso de bloques `try...except` con `RuntimeError` asegura que, si una base de datos está corrupta o falta, el programa se detenga con un mensaje humano explicativo en lugar de un error críptico de Python.

**Normalización de Datos (Data Integrity)**
El cargador no solo trae los datos, los "cura" para los matemáticos:
* **Conversión de Fechas**: Transforma strings en objetos `datetime64`, permitiendo filtros temporales precisos.
* **Limpieza de Cupones**: Normaliza tasas como "7.25%" a valores numéricos (0.0725) para poder usarlos en las fórmulas de valoración.
* **Ordenamiento Mandatorio**: Organiza las operaciones por fecha, nemotécnico y hora, garantizando que el Bootstrapper siempre procese la información en la secuencia cronológica correcta.

---

## 💡 Aprendizajes de Ingeniería

* **Independencia del Módulo**: Al incluir `sys.path.insert` dentro del archivo, el `data_loader.py` se vuelve capaz de habilitar su propio mapa de código, permitiéndole funcionar tanto si lo llamas solo como si lo llama el Orquestador.
* **Uso de SQLite**: Se prefiere sobre CSV para los datos históricos porque garantiza integridad (tipos de datos fijos) y permite realizar consultas rápidas antes de cargar todo a memoria.

---
*Estado del Sistema: Datos cargados y normalizados. El camión de suministros ha llegado a la terminal.*

# LECCIÓN: EL TRUCO DEL "HÍBRIDO" EN PYTHON (__name__ == "__main__")

1. ¿QUÉ ES ESTO?
Cada vez que corres un archivo en Python, el programa crea una variable invisible llamada __name__. 
Este bloque es un interruptor que detecta cómo estás usando el archivo.

2. LOS DOS COMPORTAMIENTOS:
- MODO SCRIPT: Si corres el archivo directamente (python mi_archivo.py), __name__ vale "__main__".
- MODO LIBRERÍA: Si otro archivo importa este código (import mi_archivo), __name__ vale "mi_archivo".

3. ESTRUCTURA RECOMENDADA:

def mi_funcion_util():
    print("Esto es una herramienta")

if __name__ == "__main__":
    # TODO LO QUE ESTÉ AQUÍ ADENTRO:
    # - Solo se ejecuta si le das al botón de "Play" a este archivo.
    # - No molesta a otros archivos cuando importan tus funciones.
    # - Es el lugar ideal para hacer pruebas (testing) o ejemplos de uso.
    print("Ejecutando pruebas locales...")
    mi_funcion_util()

4. ¿POR QUÉ ES "BRILLANTE"?
Porque te permite tener "lo mejor de los dos mundos": puedes tener un archivo que sirve como una caja de herramientas para otros proyectos, pero que al mismo tiempo tiene su propio laboratorio de pruebas adentro sin que ambos se estorben. 

Es el estándar de oro en ingeniería de software para mantener el código limpio y profesional.