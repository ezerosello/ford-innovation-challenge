"""
PASO 1a — INGESTA
Objetivo: traer el Excel del QLS a una tabla de pandas, sin tocar nada.

Idea teórica: separar "leer" de "limpiar". Si mañana Ford manda los datos
en otro formato (CSV, base de datos), solo cambia este archivo; el resto
del pipeline no se entera.
"""
import pandas as pd
from config import buscar_excel, RUTA_CRUDO


def leer_excel(ruta=None) -> pd.DataFrame:
    ruta = ruta or buscar_excel()
    print(f"Leyendo {ruta.name} (tarda ~30 segundos la primera vez)...")
    # La fila 1 del Excel es una descripción en texto de cada columna.
    # La fila 2 tiene los nombres "de sistema". Por eso header=1.
    return pd.read_excel(ruta, header=1)


def diccionario_de_columnas(ruta=None) -> pd.DataFrame:
    """Devuelve una tabla nombre_columna -> descripción (la fila 1 del Excel).
    Sirve para entender qué significa cada variable."""
    ruta = ruta or buscar_excel()
    desc = pd.read_excel(ruta, header=None, nrows=2)
    return pd.DataFrame({"columna": desc.iloc[1].values,
                         "descripcion": desc.iloc[0].values})


def cargar_crudo(forzar_relectura: bool = False) -> pd.DataFrame:
    """Lee el Excel una sola vez y guarda una copia rápida (.pkl).
    Las siguientes veces carga la copia en 1 segundo."""
    if RUTA_CRUDO.exists() and not forzar_relectura:
        return pd.read_pickle(RUTA_CRUDO)
    df = leer_excel()
    df.to_pickle(RUTA_CRUDO)
    return df


if __name__ == "__main__":
    df = cargar_crudo()
    print(df.shape)
    print(diccionario_de_columnas().to_string())
