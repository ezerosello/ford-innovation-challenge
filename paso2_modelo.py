"""
PASO 2 COMPLETO: entrenar, comparar, elegir y evaluar el modelo.

Uso:  python paso2_modelo.py   (requiere haber corrido el paso 1)

Flujo:
  1. Train (días 1-200) se parte en AJUSTE (1-180) y VALIDACIÓN (181-200).
  2. Se entrenan varias variantes (qué información usa cada una) en AJUSTE
     y se comparan en VALIDACIÓN. Se elige una.
  3. Todas se reentrenan con el train completo y se miden en TEST (201-260).
     El test no se usó para ninguna decisión: es la "prueba de fuego".
  4. Se generan: tabla comparativa, ranking de autos, gráficos, y un chequeo
     sobre el período posterior a la medida de corte.
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

from config import (DIA_FIN_AJUSTE, PRESUPUESTO_INSPECCION, SEMILLA, CARPETA_SALIDAS,
                    RUTA_TRAIN, RUTA_MODELO, RUTA_RESULTADOS, RUTA_RANKING)
from split import cargar_split
from features import columnas_features
from model import (ModeloCalidad, metricas, metricas_azar, intervalo_lift,
                   importancia_por_permutacion)

pd.set_option("display.width", 200)

# ---------------------------------------------------------------- 0. Datos
if not RUTA_TRAIN.exists():
    raise SystemExit("Primero corré: python paso1_preparar_datos.py")
train, test, post_corte = cargar_split()
ajuste = train[train["dia_salida_qls"] <= DIA_FIN_AJUSTE]
validacion = train[train["dia_salida_qls"] > DIA_FIN_AJUSTE]
print(f"Ajuste: {len(ajuste):,} autos | Validación: {len(validacion):,} | Test: {len(test):,}")

# ---------------------------------------------------------------- 1. Variantes
# Cada variante responde una pregunta: "¿cuánto aporta este bloque de información?"
VARIANTES = {
    "Solo catálogo":                     dict(usar_catalogo=True,  usar_historial=False, usar_anomalia=False),
    "Solo historial QLS":                dict(usar_catalogo=False, usar_historial=True,  usar_anomalia=False),
    "Historial + anomalía":              dict(usar_catalogo=False, usar_historial=True,  usar_anomalia=True),
    "Catálogo + historial":              dict(usar_catalogo=True,  usar_historial=True,  usar_anomalia=False),
    "Propuesta completa (cat+hist+IF)":  dict(usar_catalogo=True,  usar_historial=True,  usar_anomalia=True),
}

print("\n=== Etapa A: comparar variantes en VALIDACIÓN (días 181-200) ===")
en_validacion, n_arboles = {}, {}
for nombre, opciones in VARIANTES.items():
    m = ModeloCalidad(**opciones).fit(ajuste, df_validacion=validacion)
    en_validacion[nombre] = metricas(validacion["y"], m.predict_proba(validacion))
    n_arboles[nombre] = m.n_arboles_
    print(f"  {nombre:34s} PR-AUC={en_validacion[nombre]['PR-AUC']:.3f}  "
          f"lift@5%={en_validacion[nombre]['lift_top5']:.2f}  (árboles={m.n_arboles_})")

# Regla de elección: el mejor PR-AUC en validación. Si otra variante más simple
# (menos bloques de información) queda a menos de 3% de la mejor, elegimos la simple:
# menos piezas = menos cosas que se pueden romper y más fácil de explicar.
mejor_pr = max(r["PR-AUC"] for r in en_validacion.values())
candidatas = [n for n, r in en_validacion.items() if r["PR-AUC"] >= 0.97 * mejor_pr]
complejidad = {n: sum(VARIANTES[n].values()) for n in VARIANTES}
elegida = min(candidatas, key=lambda n: (complejidad[n], -en_validacion[n]["PR-AUC"]))
print(f"\n  -> Variante elegida: {elegida}")

# ---------------------------------------------------------------- 2. Test
print("\n=== Etapa B: reentrenar con train completo (días 1-200) y medir en TEST ===")
filas, scores_test, modelos = [], {}, {}
azar = metricas_azar(test["y"])
filas.append({"modelo": "Azar (método actual)", **azar})

# Isolation Forest solo, sin etiquetas: ¿"raro" equivale a "necesita calibración"?
hist_cols, _ = columnas_features(train)
iso = IsolationForest(n_estimators=300, random_state=SEMILLA).fit(train[hist_cols].astype(float))
scores_test["Solo Isolation Forest"] = -iso.score_samples(test[hist_cols].astype(float))
filas.append({"modelo": "Solo Isolation Forest (sin etiquetas)",
              **metricas(test["y"], scores_test["Solo Isolation Forest"])})

for nombre, opciones in VARIANTES.items():
    m = ModeloCalidad(**opciones).fit(train, n_arboles=n_arboles[nombre])
    modelos[nombre] = m
    scores_test[nombre] = m.predict_proba(test)
    filas.append({"modelo": nombre, **metricas(test["y"], scores_test[nombre])})

tabla = pd.DataFrame(filas).set_index("modelo")
tabla.to_csv(RUTA_RESULTADOS)
print(tabla.round(3).to_string())

final = modelos[elegida]
p_final = scores_test[elegida]
lo, hi = intervalo_lift(test["y"], p_final)
r = tabla.loc[elegida]
print(f"\n=== Modelo final: {elegida} ===")
print(f"  Revisando el {PRESUPUESTO_INSPECCION:.0%} de los autos:")
print(f"    al azar:     {azar['precision_top5']:.1%} de los revisados necesita calibración")
print(f"    con modelo:  {r['precision_top5']:.1%}  (lift {r['lift_top5']:.2f}x, "
      f"intervalo 95%: {lo:.2f}x - {hi:.2f}x)")
with open(RUTA_MODELO, "wb") as f:
    pickle.dump(final, f)

# ---------------------------------------------------------------- 3. Ranking
# En planta el ranking se arma por día: cada día se revisa el 5% con mayor riesgo.
ranking = test[["dia_salida_qls", "catalogo", "y", "componente_auditoria"]].copy()
ranking["prob_calibracion"] = p_final
ranking["prioridad_en_el_dia"] = (ranking.groupby("dia_salida_qls")["prob_calibracion"]
                                  .rank(ascending=False, method="first").astype(int))
cupo = ranking.groupby("dia_salida_qls")["y"].transform(
    lambda s: max(1, int(round(len(s) * PRESUPUESTO_INSPECCION))))
ranking["revisar"] = ranking["prioridad_en_el_dia"] <= cupo
ranking = ranking.sort_values(["dia_salida_qls", "prioridad_en_el_dia"])
ranking.to_csv(RUTA_RANKING)
sel = ranking[ranking["revisar"]]
print(f"\n  Selección diaria del 5% en test: {len(sel):,} autos revisados, "
      f"{sel['y'].mean():.1%} necesitaban calibración (azar: {test['y'].mean():.1%})")

# ---------------------------------------------------------------- 4. Gráficos
def curva_captura(y, score):
    orden = np.argsort(-score)
    capturadas = np.cumsum(np.asarray(y)[orden]) / np.sum(y)
    revisadas = np.arange(1, len(y) + 1) / len(y)
    return revisadas * 100, capturadas * 100

fig, ax = plt.subplots(figsize=(7, 5))
for nombre, estilo in [(elegida, "-"), ("Solo historial QLS", "--"), ("Solo Isolation Forest", ":")]:
    if nombre in scores_test:
        x, yv = curva_captura(test["y"].values, scores_test[nombre])
        ax.plot(x, yv, estilo, label=nombre, linewidth=2)
ax.plot([0, 100], [0, 100], color="gray", label="Azar (método actual)")
ax.axvline(PRESUPUESTO_INSPECCION * 100, color="red", alpha=0.4)
ax.set_xlabel("% de autos revisados (ordenados por riesgo)")
ax.set_ylabel("% de calibraciones encontradas")
ax.set_title("Curva de captura en test (días 201-260)")
ax.set_xlim(0, 50); ax.set_ylim(0, 70); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(CARPETA_SALIDAS / "grafico_curva_captura.png", dpi=150)

print("\n  Calculando importancia de variables (tarda 1-2 minutos)...")
imp = importancia_por_permutacion(final, test, n_repeticiones=2)
imp.to_csv(CARPETA_SALIDAS / "07_importancia_variables.csv")
top = imp.head(12)[::-1]
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(top.index, top.values)
ax.set_xlabel("Caída de PR-AUC al desordenar la variable")
ax.set_title(f"Variables que más usa el modelo\n({elegida})")
fig.tight_layout(); fig.savefig(CARPETA_SALIDAS / "grafico_importancia.png", dpi=150)
print(imp.head(8).round(4).to_string())

# ---------------------------------------------------------------- 5. Post-corte
print("\n=== Chequeo post medida de corte (días 261+) ===")
p_post = final.predict_proba(post_corte)
print(f"  El modelo espera {p_post.mean():.1%} de calibraciones; la realidad fue "
      f"{post_corte['y'].mean():.1%}.")
print("  -> El proceso cambió y el modelo no lo sabe: por eso hace falta monitoreo y reentrenamiento.")

# ---------------------------------------------------------------- 6. MLflow (opcional)
try:
    import mlflow
    with mlflow.start_run(run_name=elegida):
        mlflow.log_params({"variante": elegida, "n_arboles": final.n_arboles_,
                           "presupuesto": PRESUPUESTO_INSPECCION})
        mlflow.log_metrics({k.replace("-", "_"): float(v) for k, v in r.items()})
        for archivo in ["grafico_curva_captura.png", "grafico_importancia.png"]:
            mlflow.log_artifact(str(CARPETA_SALIDAS / archivo))
    print("\nCorrida registrada en MLflow (ver con: mlflow ui)")
except ImportError:
    print("\n(MLflow no está instalado: se omite el registro de la corrida)")

print("\nListo. Resultados en:", CARPETA_SALIDAS)
