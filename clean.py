"""
PASO 1b — LIMPIEZA
Objetivo: dejar la tabla de defectos (1 fila = 1 defecto registrado en el QLS)
consistente y cómoda de usar. Todavía NO agregamos por vehículo.

Qué hace, en orden:
  1. Renombra columnas a nombres cortos sin tildes ni espacios.
  2. Normaliza textos (espacios sobrantes, vacíos -> nulo).
  3. Convierte fechas "DIA_12" -> 12 y horas (fracción de día) -> horas.
  4. Crea la variable objetivo y = 1 si la unidad quedó CALIBRADA.
  5. Elimina filas duplicadas exactas.
  6. Elimina columnas que no aportan información (constantes o casi vacías).
  7. Valida supuestos: cada VIN tiene un solo resultado y un solo catálogo.
"""
import numpy as np
import pandas as pd
from config import RUTA_LIMPIO

NOMBRES = {
    "VIN": "vin",
    "INSPECTOR": "inspector",
    "Hora Inspección": "hora_insp",
    "Fecha Inspección": "fecha_insp",
    "CP": "cp",                                  # Collection Point: dónde se detectó
    "Sec.CP": "sec_cp",
    "CP Grupo Trabajo Reporta": "cp_grupo",
    "CP Zona Reporta": "cp_zona",                # zona/área que detectó el defecto
    "Componente Inspección": "componente",       # QUÉ pieza tuvo el defecto
    "UC Nombre Incidencia": "incidencia",        # qué le pasó (roce, falla, etc.)
    "UC Nombre Tipo Incidencia": "tipo_incidencia",
    "UC Nombre Posicion A": "pos_a",
    "UC Nombre Posicion B": "pos_b",
    "UC Nombre Posicion C": "pos_c",
    "UC Nombre Grupo Posicion C": "grupo_pos_c",
    "UC Nombre Posición D": "pos_d",
    "UC Nombre Grupo Posición D": "grupo_pos_d",
    "UC Posición Arbitraria": "pos_arbitraria",
    "CCC": "ccc",    # Customer Concern Code: falla específica (nivel más fino)
    "VFG": "vfg",    # Vehicle Function Group: grupo de función (asientos, tablero...)
    "VRT": "vrt",    # Variable Reduction Team: área/equipo funcional (nivel más grueso)
    "UC Nombre PUL a Reparar": "pul_a_reparar",  # equipo asignado a reparar
    "Fecha Reparación": "fecha_rep",
    "Hora Reparación": "hora_rep",
    "Código de Reparador": "reparador",
    "Rep PUL": "rep_pul",
    "Rep Parte Causal": "rep_parte",
    "Rep.incid.": "rep_incidencia",
    "Rep.Tipo Incid.": "rep_tipo_incidencia",
    "Rep.PosA": "rep_pos_a",
    "Rep.PosB": "rep_pos_b",
    "Rep.PosC": "rep_pos_c",
    "Rep.Grp.PosC": "rep_grupo_pos_c",
    "Rep Nombre Posición D": "rep_pos_d",
    "Rep Nombre Grupo Posición D": "rep_grupo_pos_d",
    "Rep.Pos.arbit": "rep_pos_arbitraria",
    "Rep Respuesta a Pregunta Remplazar": "rep_reemplazo",
    "Rep Respuesta a Pregunta Desensamblar": "rep_desensamble",
    "Código de Catálogo": "catalogo",            # versión + mercado de destino
    "Auditoría Adicional": "auditoria",          # OK / CALIBRADA  (el resultado)
    "Componente Auditoría Adicional": "componente_auditoria",  # qué se calibró
}

# Columnas que usan como "reloj": las necesitamos sí o sí
COLUMNAS_OBLIGATORIAS = ["vin", "fecha_insp", "hora_insp", "catalogo", "auditoria"]


def _dia_a_numero(serie: pd.Series) -> pd.Series:
    """'DIA_12' -> 12. Si viene vacío, queda nulo."""
    return pd.to_numeric(serie.astype("string").str.replace("DIA_", "", regex=False),
                         errors="coerce").astype("Int64")


def limpiar(df: pd.DataFrame, max_nulos: float = 0.95, verbose: bool = True) -> pd.DataFrame:
    reporte = {}
    df = df.copy()
    filas_iniciales = len(df)

    # 1. Renombrar (quitando espacios sobrantes de los encabezados por las dudas)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns=NOMBRES)
    faltan = [c for c in COLUMNAS_OBLIGATORIAS if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas esperadas: {faltan}")

    # 2. Normalizar textos: 'FC ' y 'FC' tienen que ser lo mismo
    for col in df.columns:
        if col in ("hora_insp", "hora_rep"):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            s = df[col].astype("string").str.strip()
            df[col] = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    # En componente_auditoria, '-' significa "calibrada pero sin componente informado"
    df["componente_auditoria"] = df["componente_auditoria"].replace({"-": "SIN_INFORMAR"})

    # 3. Tiempos. Excel guarda la hora como fracción del día (0.5 = 12:00 hs)
    df["dia_insp"] = _dia_a_numero(df["fecha_insp"])
    df["dia_rep"] = _dia_a_numero(df["fecha_rep"])
    df["hora_insp"] = pd.to_numeric(df["hora_insp"], errors="coerce") * 24
    df["hora_rep"] = pd.to_numeric(df["hora_rep"], errors="coerce") * 24
    # "Reloj continuo" en horas desde el día 0: permite restar tiempos fácilmente
    df["t_insp_h"] = df["dia_insp"].astype(float) * 24 + df["hora_insp"]
    df["t_rep_h"] = df["dia_rep"].astype(float) * 24 + df["hora_rep"]
    df["demora_reparacion_h"] = (df["t_rep_h"] - df["t_insp_h"]).clip(lower=0)
    df = df.drop(columns=["fecha_insp", "fecha_rep"])

    # 4. Variable objetivo
    df["y"] = (df["auditoria"] == "CALIBRADA").astype(int)

    # 5. Duplicados exactos (el mismo defecto cargado dos veces)
    antes = len(df)
    df = df.drop_duplicates()
    reporte["duplicados_eliminados"] = antes - len(df)

    # 6. Columnas sin información
    protegidas = set(COLUMNAS_OBLIGATORIAS) | {"y", "componente_auditoria", "dia_insp",
                                                "t_insp_h", "dia_rep", "t_rep_h"}
    constantes = [c for c in df.columns
                  if c not in protegidas and df[c].nunique(dropna=True) <= 1]
    # "Casi constantes": el valor dominante cubre más del 99.9% de las filas no nulas
    casi_constantes = [c for c in df.columns
                       if c not in protegidas and c not in constantes
                       and df[c].value_counts(normalize=True).iloc[0] > 0.999]
    casi_vacias = [c for c in df.columns
                   if c not in protegidas and df[c].isna().mean() > max_nulos]
    a_borrar = sorted(set(constantes + casi_constantes + casi_vacias))
    df = df.drop(columns=a_borrar)
    reporte["columnas_eliminadas"] = a_borrar

    # 7. Validaciones: si algo de esto falla, nuestro entendimiento del dato está mal
    por_vin = df.groupby("vin")
    assert (por_vin["auditoria"].nunique() == 1).all(), "Hay VINs con más de un resultado"
    assert (por_vin["catalogo"].nunique() == 1).all(), "Hay VINs con más de un catálogo"
    assert df["auditoria"].isin(["OK", "CALIBRADA"]).all(), "Valores raros en 'auditoria'"

    reporte["filas"] = f"{filas_iniciales} -> {len(df)}"
    reporte["vins"] = df["vin"].nunique()
    reporte["tasa_calibrada_por_vin"] = round(por_vin["y"].first().mean(), 4)

    if verbose:
        print("\n=== Reporte de limpieza ===")
        for k, v in reporte.items():
            print(f"  {k}: {v}")
    return df


def cargar_limpio(df_crudo: pd.DataFrame = None, forzar: bool = False) -> pd.DataFrame:
    if RUTA_LIMPIO.exists() and not forzar:
        return pd.read_pickle(RUTA_LIMPIO)
    from ingest import cargar_crudo
    df = limpiar(df_crudo if df_crudo is not None else cargar_crudo())
    df.to_pickle(RUTA_LIMPIO)
    return df


if __name__ == "__main__":
    cargar_limpio(forzar=True)
