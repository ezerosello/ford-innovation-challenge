"""
PASO 1d — SPLIT CRONOLÓGICO
Objetivo: separar los vehículos en "pasado" (train) y "futuro" (test).

Por qué cronológico y no al azar: en planta el modelo se entrena con lo
que ya pasó y se usa sobre autos nuevos. Si mezcláramos al azar, el modelo
vería autos de la misma semana que los que después evalúa, y el resultado
sería optimista (mediríamos algo que en la vida real no va a pasar).

Tres grupos:
  - train:      días 1 .. DIA_FIN_TRAIN
  - test:       días DIA_FIN_TRAIN+1 .. DIA_MEDIDA_CORTE
  - post_corte: después de la medida de corte. Casi no hay calibraciones,
                así que no sirve para medir el modelo; lo guardamos aparte
                para mostrar el cambio de régimen (monitoreo / drift).
"""
import pandas as pd
from config import (DIA_FIN_TRAIN, DIA_MEDIDA_CORTE,
                    RUTA_TRAIN, RUTA_TEST, RUTA_POST_CORTE)


def split_cronologico(tabla: pd.DataFrame, verbose: bool = True):
    dia = tabla["dia_salida_qls"]
    train = tabla[dia <= DIA_FIN_TRAIN]
    test = tabla[(dia > DIA_FIN_TRAIN) & (dia <= DIA_MEDIDA_CORTE)]
    post = tabla[dia > DIA_MEDIDA_CORTE]

    if verbose:
        print("\n=== Split cronológico ===")
        for nombre, parte in [("train", train), ("test", test), ("post_corte", post)]:
            print(f"  {nombre:11s} días {parte['dia_salida_qls'].min():>3}-"
                  f"{parte['dia_salida_qls'].max():<3}  unidades={len(parte):>6}  "
                  f"calibradas={parte['y'].sum():>5}  tasa={parte['y'].mean():.1%}")
    return train, test, post


def guardar_split(tabla: pd.DataFrame):
    train, test, post = split_cronologico(tabla)
    train.to_pickle(RUTA_TRAIN)
    test.to_pickle(RUTA_TEST)
    post.to_pickle(RUTA_POST_CORTE)
    return train, test, post


def cargar_split():
    return (pd.read_pickle(RUTA_TRAIN), pd.read_pickle(RUTA_TEST),
            pd.read_pickle(RUTA_POST_CORTE))


if __name__ == "__main__":
    from features import cargar_tabla_vin
    guardar_split(cargar_tabla_vin())
