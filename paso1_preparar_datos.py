"""
PASO 1 COMPLETO: Excel crudo -> datos listos para modelar.

Uso:
    python paso1_preparar_datos.py            (usa copias guardadas si existen)
    python paso1_preparar_datos.py --forzar   (rehace todo desde el Excel)
"""
import sys
import pandas as pd
from config import CARPETA_SALIDAS
from ingest import cargar_crudo, diccionario_de_columnas
from clean import limpiar
from features import construir_tabla_vin, columnas_features
from split import guardar_split
from config import RUTA_LIMPIO, RUTA_TABLA_VIN

forzar = "--forzar" in sys.argv

# 1a. Ingesta
crudo = cargar_crudo(forzar_relectura=forzar)
print(f"Excel leído: {crudo.shape[0]:,} filas x {crudo.shape[1]} columnas")
diccionario_de_columnas().to_csv(CARPETA_SALIDAS / "diccionario_columnas.csv", index=False)

# 1b. Limpieza
limpio = limpiar(crudo)
limpio.to_pickle(RUTA_LIMPIO)

# 1c. Tabla por vehículo
tabla = construir_tabla_vin(limpio)
tabla.to_pickle(RUTA_TABLA_VIN)
num, cat = columnas_features(tabla)
print(f"\n=== Tabla por VIN ===\n  {len(tabla):,} vehículos | "
      f"{len(num)} features numéricas + {len(cat)} categóricas")

# 1d. Split
train, test, post = guardar_split(tabla)

# Mini-exploración para entender el problema (no es parte del pipeline)
print("\n=== Tasa de calibración por período de 20 días ===")
periodo = ((tabla["dia_salida_qls"] - 1) // 20) * 20 + 1
print((tabla.groupby(periodo)["y"].mean() * 100).round(1).to_string())

print("\n=== Tasa de calibración por 3er carácter del catálogo (días <= 260) ===")
pre = tabla[tabla["dia_salida_qls"] <= 260]
resumen = pre.groupby("cat_pos3")["y"].agg(tasa="mean", unidades="size")
resumen["tasa"] = (resumen["tasa"] * 100).round(1)
print(resumen.sort_values("unidades", ascending=False).to_string())

print("\nListo. Archivos en:", CARPETA_SALIDAS)
