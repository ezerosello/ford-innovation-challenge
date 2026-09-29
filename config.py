"""
Configuración central del proyecto.
Todo lo que sea una "decisión" (rutas, días de corte, etc.) vive acá,
así se puede cambiar en un solo lugar y queda documentado.
"""
from pathlib import Path

# --- Rutas ---------------------------------------------------------------
RAIZ = Path(__file__).parent
CARPETA_DATOS = RAIZ / "data"
CARPETA_SALIDAS = RAIZ / "salidas"
CARPETA_SALIDAS.mkdir(exist_ok=True)



def buscar_excel():
    """Devuelve el .xlsx que esté en data/, sin depender del nombre exacto
    (las tildes o sufijos como ' (1)' suelen cambiar al descargar el archivo)."""
    archivos = sorted(p for p in CARPETA_DATOS.glob("*.xlsx")
                      if not p.name.startswith("~$"))   # ~$ = archivo temporal de Excel abierto
    if not archivos:
        raise FileNotFoundError(
            f"No encontré ningún .xlsx en {CARPETA_DATOS}\n"
            "Copiá ahí el Excel del dataset de Ford y volvé a correr.")
    if len(archivos) > 1:
        print(f"Aviso: hay {len(archivos)} archivos .xlsx en data/; uso {archivos[0].name}")
    return archivos[0]


# Copias intermedias (leer el Excel tarda ~30 s; estas cargan en 1 s)
RUTA_CRUDO = CARPETA_SALIDAS / "01_crudo.pkl"
RUTA_LIMPIO = CARPETA_SALIDAS / "02_limpio.pkl"
RUTA_TABLA_VIN = CARPETA_SALIDAS / "03_tabla_vin.pkl"
RUTA_TRAIN = CARPETA_SALIDAS / "04_train.pkl"
RUTA_TEST = CARPETA_SALIDAS / "04_test.pkl"
RUTA_POST_CORTE = CARPETA_SALIDAS / "04_post_corte.pkl"

# --- Decisiones de negocio / validación ------------------------------------
# Ford mencionó una "medida de corte" cerca del día 260. En los datos se ve:
# después del día 260 casi no hay calibraciones (3 en ~5.000 unidades).
# Esos días son otro "régimen" y no sirven para entrenar ni para evaluar.
DIA_MEDIDA_CORTE = 260

# Split cronológico: entrenamos con el pasado y evaluamos con el "futuro".
DIA_FIN_TRAIN = 200          # train = días 1..200
# test = días 201..DIA_MEDIDA_CORTE

# Semilla para que todo lo aleatorio sea reproducible
SEMILLA = 42

# --- Paso 2: modelo ----------------------------------------------------------
# Dentro del train, los últimos 20 días se usan como "validación": sirven para
# decidir cuándo dejar de entrenar y qué variante de modelo elegir, SIN mirar
# el test. El test se usa una sola vez, al final, para el número que se reporta.
DIA_FIN_AJUSTE = 180         # ajuste = días 1..180, validación = 181..200

# Capacidad de inspección: hoy Ford revisa el 5% de las unidades
PRESUPUESTO_INSPECCION = 0.05

RUTA_MODELO = CARPETA_SALIDAS / "modelo_final.pkl"
RUTA_RESULTADOS = CARPETA_SALIDAS / "05_comparacion_modelos.csv"
RUTA_RANKING = CARPETA_SALIDAS / "06_ranking_test.csv"
