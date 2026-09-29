"""
PASO 3 — DASHBOARD
Genera salidas/dashboard.html: una página web que se abre con doble clic,
sin servidor y sin instalar nada más.

Qué muestra (lo que pide la ficha de Ford en la sección 5):
  1. Unidades priorizadas por día: el 5% con mayor riesgo, marcado para revisar.
  2. Qué revisar en cada auto: los componentes (VFG) y el área (VRT) que más
     probablemente haya que calibrar, según su versión/mercado.
  3. Por qué el modelo lo eligió: las variables que más empujaron su riesgo.
  4. Monitoreo del proceso: gráfico de control de la tasa de calibración.
  5. Evidencia de que funciona: curva de captura y comparación con el azar.

Uso:  python paso3_dashboard.py   (requiere haber corrido los pasos 1 y 2)
"""
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

import model  # noqa: F401  (necesario para poder abrir el modelo guardado)
from config import (RUTA_LIMPIO, RUTA_MODELO, RUTA_RESULTADOS, CARPETA_SALIDAS,
                    PRESUPUESTO_INSPECCION, DIA_FIN_TRAIN, DIA_MEDIDA_CORTE)
from split import cargar_split
from model import metricas, intervalo_lift

RUTA_TEMPLATE = Path(__file__).parent / "dashboard_template.html"
RUTA_DASHBOARD = CARPETA_SALIDAS / "dashboard.html"

for requerido in [RUTA_MODELO, RUTA_RESULTADOS]:
    if not requerido.exists():
        raise SystemExit("Primero corré: python paso2_modelo.py")

train, test, post = cargar_split()
limpio = pd.read_pickle(RUTA_LIMPIO)
with open(RUTA_MODELO, "rb") as f:
    modelo = pickle.load(f)
tabla_vin = pd.concat([train, test, post])
evaluar = pd.concat([test, post])          # días que muestra el dashboard (201 en adelante)

# ============================================================ 1. Predicciones
print("Calculando riesgo de cada unidad...")
prob = modelo.predict_proba(evaluar)

# ============================================================ 2. ¿Por qué?
# XGBoost puede descomponer cada predicción en "cuánto sumó cada variable"
# (valores SHAP). Agrupamos las columnas one-hot del catálogo en su variable
# original para que la explicación sea legible.
X = modelo._matriz(evaluar)[modelo.columnas_]
contrib = modelo.xgb_.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
contrib = pd.DataFrame(contrib, columns=modelo.columnas_, index=evaluar.index)
grupo_de = {c: c.split("=")[0] for c in modelo.columnas_}
contrib_grupo = contrib.T.groupby(pd.Series(grupo_de)).sum().T

# Descripción legible de cada VFG: los componentes más frecuentes en el QLS
desc_vfg = (limpio.groupby("vfg")["componente"]
            .agg(lambda s: ", ".join(s.value_counts().index[:2].str.lower())))
area_vfg = limpio.groupby("vfg")["vrt"].agg(lambda s: s.value_counts().index[0])

NOMBRES = {
    "n_defectos": "defectos registrados en QLS",
    "n_componentes_distintos": "componentes distintos con defecto",
    "n_ccc_distintos": "fallas (CCC) distintas",
    "n_vfg_distintos": "grupos de función (VFG) con defecto",
    "n_areas_vrt_distintas": "áreas (VRT) con defecto",
    "n_puntos_control": "puntos de control que lo detectaron",
    "n_inspectores": "inspectores distintos",
    "n_reparadores": "reparadores distintos",
    "max_repeticiones_mismo_componente": "máx. veces que se repitió un componente",
    "n_componentes_repetidos": "componentes con defecto repetido",
    "demora_rep_media_h": "demora media de reparación (h)",
    "demora_rep_max_h": "demora máxima de reparación (h)",
    "dias_en_qls": "días en QLS",
    "score_anomalia": "rareza del historial (Isolation Forest)",
    "cat_pos2": "catálogo, letra 2",
    "cat_pos3": "catálogo, letra 3",
    "cat_pos4": "catálogo, letra 4",
    "cat_pos23": "catálogo, letras 2 y 3",
}


def nombre_legible(var):
    if var in NOMBRES:
        return NOMBRES[var]
    if var.startswith("area_vrt_"):
        return f"defectos en área VRT {var[9:]}"
    if var.startswith("grupo_vfg_"):
        v = var[10:]
        return f"defectos en VFG {v} ({desc_vfg.get(v, 's/d')})"
    if var.startswith("zona_"):
        return f"defectos detectados en zona {var[5:].replace('_', ' ')}"
    return var


def etiqueta_con_valor(var, fila):
    if var == "score_anomalia":
        return NOMBRES[var]
    if var.startswith("cat_pos"):
        return f"{nombre_legible(var)}: {fila[var]}"
    valor = fila[var]
    if isinstance(valor, (int, np.integer)) or float(valor).is_integer():
        valor = int(valor)
    else:
        valor = round(float(valor), 1)
    nombre = nombre_legible(var)
    if valor == 0 and nombre.startswith("defectos"):
        return f"sin {nombre}"   # la ausencia de defectos también puede sumar riesgo
    return f"{nombre}: {str(valor).replace('.', ',')}"


vocabulario, indice_vocab = [], {}


def id_factor(texto):
    if texto not in indice_vocab:
        indice_vocab[texto] = len(vocabulario)
        vocabulario.append(texto)
    return indice_vocab[texto]


# ============================================================ 3. ¿Qué revisar?
# Entre los autos calibrados del train: ¿qué componente se calibró según la
# versión/mercado? Se combina de lo general a lo particular (letra 2, letra 3,
# letras 2-3, catálogo completo). Cuantos más autos hay en un grupo, más se
# confía en ese grupo; si son pocos, se usa el nivel más general.
calibrados = train[(train["y"] == 1) & (train["componente_auditoria"] != "SIN_INFORMAR")]
frec_global = calibrados["componente_auditoria"].value_counts(normalize=True)


def distribucion_componentes(fila, fuerza=20):
    p = frec_global.copy()
    for nivel in ["cat_pos2", "cat_pos3", "cat_pos23", "catalogo"]:
        grupo = calibrados[calibrados[nivel] == fila[nivel]]["componente_auditoria"]
        if len(grupo):
            conteo = grupo.value_counts().reindex(p.index, fill_value=0)
            p = (conteo + fuerza * p) / (len(grupo) + fuerza)
    return p.sort_values(ascending=False)


catalogos = tabla_vin.drop_duplicates("catalogo").set_index("catalogo")
catalogos["catalogo"] = catalogos.index
sugerencias = {c: list(distribucion_componentes(f).index[:3]) for c, f in catalogos.iterrows()}

cal_test = test[test["y"] == 1]
acierto_modelo = np.mean([r.componente_auditoria in sugerencias[r.catalogo]
                          for r in cal_test.itertuples()])
acierto_frecuentes = np.mean(cal_test["componente_auditoria"].isin(frec_global.index[:3]))

componentes_info = {}
for v in set(frec_global.index) | set(tabla_vin["componente_auditoria"].dropna()):
    componentes_info[v] = {"desc": desc_vfg.get(v), "area": area_vfg.get(v)}

# ============================================================ 4. Unidades
unidades = []
for i, (vin, fila) in enumerate(evaluar.iterrows()):
    top = contrib_grupo.loc[vin].sort_values(ascending=False).head(3)
    factores = [[id_factor(etiqueta_con_valor(var, fila)), round(float(c), 3)]
                for var, c in top.items() if c > 0]
    real = fila["componente_auditoria"] if fila["y"] == 1 else None
    unidades.append([vin, int(fila["dia_salida_qls"]), fila["catalogo"],
                     round(float(prob[i]), 4), int(fila["y"]), real, factores])

# ============================================================ 5. Monitoreo
# Gráfico de control p: para cada día, la tasa de calibración se compara con
# los últimos 30 días "normales". Si cae fuera de ±3 desvíos, es alarma, y ese
# día no se usa como referencia (para que la alarma no se "normalice" sola).
print("Armando gráfico de control...")
por_dia = tabla_vin.groupby("dia_salida_qls")["y"].agg(k="sum", n="size")
esperada = pd.Series(prob, index=evaluar.index).groupby(evaluar["dia_salida_qls"]).mean()
referencia, control = [], []
for dia, r in por_dia.iterrows():
    punto = {"d": int(dia), "n": int(r.n), "tasa": round(r.k / r.n, 4),
             "p0": None, "lcl": None, "ucl": None, "alarma": 0,
             "esperada": round(float(esperada[dia]), 4) if dia in esperada.index else None}
    if r.n >= 50:
        base_k = sum(k for k, _ in referencia[-30:])
        base_n = sum(n for _, n in referencia[-30:])
        if base_n >= 1000:
            p0 = base_k / base_n
            s = np.sqrt(p0 * (1 - p0) / r.n)
            punto.update(p0=round(p0, 4), lcl=round(max(0.0, p0 - 3 * s), 4),
                         ucl=round(p0 + 3 * s, 4))
            punto["alarma"] = int(not (p0 - 3 * s <= r.k / r.n <= p0 + 3 * s))
        if not punto["alarma"]:
            referencia.append((r.k, r.n))
    control.append(punto)
alarmas = [p["d"] for p in control if p["alarma"]]
print(f"  Días en alarma: {alarmas}")

# ============================================================ 6. Evidencia
y_test = test["y"].values
p_test = prob[:len(test)]
m = metricas(y_test, p_test)
lo, hi = intervalo_lift(y_test, p_test)
orden = np.argsort(-p_test)
acum = np.cumsum(y_test[orden]) / y_test.sum()
captura = [round(float(acum[max(0, int(len(acum) * q / 100) - 1)]), 4) if q else 0.0
           for q in range(0, 101)]
sel = pd.DataFrame({"dia": test["dia_salida_qls"].values, "p": p_test, "y": y_test})
sel["rank"] = sel.groupby("dia")["p"].rank(ascending=False, method="first")
sel["cupo"] = sel.groupby("dia")["p"].transform(
    lambda s: max(1, int(round(len(s) * PRESUPUESTO_INSPECCION))))
diaria = sel[sel["rank"] <= sel["cupo"]]

comparacion = pd.read_csv(RUTA_RESULTADOS, index_col=0)
importancia = pd.read_csv(CARPETA_SALIDAS / "07_importancia_variables.csv",
                          index_col=0).iloc[:, 0]

datos = {
    "generado": datetime.now().strftime("%d/%m/%Y %H:%M"),
    "variante": modelo.descripcion(),
    "presupuesto": PRESUPUESTO_INSPECCION,
    "finTrain": DIA_FIN_TRAIN,
    "corte": DIA_MEDIDA_CORTE,
    "kpi": {
        "tasaAzar": round(float(y_test.mean()), 4),
        "precisionModelo": round(float(m["precision_top5"]), 4),
        "lift": round(float(m["lift_top5"]), 2),
        "liftBajo": round(float(lo), 2), "liftAlto": round(float(hi), 2),
        "prAuc": round(float(m["PR-AUC"]), 3),
        "diariaRevisados": int(len(diaria)),
        "diariaPrecision": round(float(diaria["y"].mean()), 4),
        "aciertoComponentes": round(float(acierto_modelo), 3),
        "aciertoFrecuentes": round(float(acierto_frecuentes), 3),
    },
    "comparacion": [{"modelo": n, "precision": round(float(r["precision_top5"]), 4),
                     "lift": round(float(r["lift_top5"]), 2)}
                    for n, r in comparacion.iterrows()],
    "captura": captura,
    "importancia": [{"nombre": nombre_legible(v), "valor": round(float(x), 5)}
                    for v, x in importancia.head(10).items()],
    "control": control,
    "componentes": componentes_info,
    "sugerencias": sugerencias,
    "factores": vocabulario,
    "unidades": unidades,
}

html = RUTA_TEMPLATE.read_text(encoding="utf-8")
html = html.replace("/*__DATOS__*/null", json.dumps(datos, ensure_ascii=False))
RUTA_DASHBOARD.write_text(html, encoding="utf-8")
print(f"\nDashboard generado: {RUTA_DASHBOARD}")
print("Abrilo con doble clic (se ve en cualquier navegador).")
