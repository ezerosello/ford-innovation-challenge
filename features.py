"""
PASO 1c — TABLA POR VEHÍCULO (feature engineering)
Objetivo: pasar de "1 fila = 1 defecto" a "1 fila = 1 vehículo (VIN)",
porque la pregunta de negocio es por vehículo: ¿esta unidad va a necesitar
calibración en la Inspección Adicional?

Cada columna que construimos acá es una "feature": un número que resume
algo del historial QLS de esa unidad. Los modelos solo ven estos números.

Grupos de features:
  A. Catálogo descompuesto (versión / mercado / año-modelo)
  B. Volumen de defectos (cuántos, qué tan variados)
  C. Tiempos (cuánto tardaron las reparaciones, cuántos días estuvo en QLS)
  D. "Qué componente y qué área": conteos por VRT (área), VFG (grupo de función)
     y zona que detectó el defecto -> es la pista que nos dio Ford.
"""
import pandas as pd
from config import RUTA_TABLA_VIN

# Columnas que NO pueden entrar al modelo:
#  - vin: es un identificador, no una característica.
#  - y: es la respuesta.
#  - componente_auditoria: solo existe si la unidad fue calibrada -> sería
#    "hacer trampa" (data leakage): el modelo aprendería a mirar la respuesta.
#  - dia_*: son fechas; las usamos para ordenar y separar train/test, no para
#    predecir (un modelo de árboles no sabe extrapolar a días que nunca vio).
#  - catalogo: lo usamos descompuesto en sus caracteres (ver abajo).
NO_FEATURES = ["y", "componente_auditoria", "dia_salida_qls", "dia_primer_defecto",
               "catalogo"]


def _conteos_por_categoria(df, columna, prefijo, min_apariciones):
    """Para cada VIN, cuántos defectos tuvo en cada valor de `columna`.
    Solo se crean columnas para valores con suficientes apariciones
    (los muy raros no le enseñan nada al modelo y agregan ruido)."""
    frec = df[columna].value_counts()
    valores = frec[frec >= min_apariciones].index
    sub = df[df[columna].isin(valores)]
    tabla = sub.groupby(["vin", columna]).size().unstack(fill_value=0)
    tabla.columns = [f"{prefijo}_{v}".replace(" ", "_") for v in tabla.columns]
    return tabla


def construir_tabla_vin(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("vin")

    # --- Identidad y respuesta ---
    base = pd.DataFrame({
        "y": g["y"].first(),
        "componente_auditoria": g["componente_auditoria"].first(),
        "catalogo": g["catalogo"].first(),
        # Fecha de referencia de la unidad: el último registro en QLS
        # (≈ cuando sale hacia Gate Release). Es el momento en que predecimos.
        "dia_salida_qls": g["dia_insp"].max().astype(int),
        "dia_primer_defecto": g["dia_insp"].min().astype(int),
    })

    # --- A. Catálogo descompuesto ---
    # Ford dijo que están "subcategorizando" el código. Cada carácter parece
    # codificar algo distinto (versión, mercado, año-modelo).
    cat = base["catalogo"].astype(str)
    base["cat_pos2"] = cat.str[1]
    base["cat_pos3"] = cat.str[2]
    base["cat_pos4"] = cat.str[3]
    base["cat_pos23"] = cat.str[1:3]

    # --- B. Volumen de defectos ---
    base["n_defectos"] = g.size()
    base["n_componentes_distintos"] = g["componente"].nunique()
    base["n_ccc_distintos"] = g["ccc"].nunique()
    base["n_vfg_distintos"] = g["vfg"].nunique()
    base["n_areas_vrt_distintas"] = g["vrt"].nunique()
    base["n_puntos_control"] = g["cp"].nunique()
    base["n_inspectores"] = g["inspector"].nunique()
    base["n_reparadores"] = g["reparador"].nunique()
    rep_comp = df.groupby(["vin", "componente"]).size()
    base["max_repeticiones_mismo_componente"] = rep_comp.groupby("vin").max()
    base["n_componentes_repetidos"] = (rep_comp > 1).groupby("vin").sum()

    # --- C. Tiempos ---
    base["demora_rep_media_h"] = g["demora_reparacion_h"].mean()
    base["demora_rep_max_h"] = g["demora_reparacion_h"].max()
    base["dias_en_qls"] = base["dia_salida_qls"] - base["dia_primer_defecto"]

    # --- D. Qué componente / qué área ---
    partes = [
        _conteos_por_categoria(df, "vrt", "area_vrt", min_apariciones=100),
        _conteos_por_categoria(df, "vfg", "grupo_vfg", min_apariciones=300),
        _conteos_por_categoria(df, "cp_zona", "zona", min_apariciones=100),
    ]
    tabla = base.join(pd.concat(partes, axis=1))
    cols_conteo = [c for p in partes for c in p.columns]
    tabla[cols_conteo] = tabla[cols_conteo].fillna(0).astype(int)
    return tabla


def columnas_features(tabla: pd.DataFrame):
    """Devuelve (numéricas, categóricas) que sí pueden entrar al modelo."""
    candidatas = [c for c in tabla.columns if c not in NO_FEATURES]
    categoricas = [c for c in candidatas if c.startswith("cat_pos")]
    numericas = [c for c in candidatas if c not in categoricas]
    return numericas, categoricas


def cargar_tabla_vin(df_limpio: pd.DataFrame = None, forzar: bool = False) -> pd.DataFrame:
    if RUTA_TABLA_VIN.exists() and not forzar:
        return pd.read_pickle(RUTA_TABLA_VIN)
    from clean import cargar_limpio
    tabla = construir_tabla_vin(df_limpio if df_limpio is not None else cargar_limpio())
    tabla.to_pickle(RUTA_TABLA_VIN)
    return tabla


if __name__ == "__main__":
    t = cargar_tabla_vin(forzar=True)
    num, cat = columnas_features(t)
    print(t.shape, "| numéricas:", len(num), "| categóricas:", cat)
