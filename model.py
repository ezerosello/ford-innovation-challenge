"""
PASO 2 — MODELO
Implementa la propuesta del equipo como un pipeline de 3 etapas:

  (1) Codificación del catálogo: letras -> columnas 0/1 (one-hot encoding).
  (2) Isolation Forest (NO supervisado): mira el historial QLS de cada auto y le
      asigna un "score de anomalía" = qué tan raro es respecto de lo habitual.
      No usa la respuesta (y) para nada.
  (3) XGBoost (supervisado): recibe las features + el score de anomalía como una
      columna más, y aprende de la respuesta histórica (OK / CALIBRADA) a
      devolver una probabilidad de calibración por auto.

La clase ModeloCalidad permite prender/apagar cada bloque de información
(catálogo, historial, anomalía). Así podemos medir cuánto aporta cada uno.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from config import SEMILLA, PRESUPUESTO_INSPECCION
from features import columnas_features

COLUMNAS_HISTORIAL, COLUMNAS_CATALOGO = None, None  # se completan al primer uso


def _grupos_de_columnas(df):
    global COLUMNAS_HISTORIAL, COLUMNAS_CATALOGO
    if COLUMNAS_HISTORIAL is None:
        COLUMNAS_HISTORIAL, COLUMNAS_CATALOGO = columnas_features(df)
    return COLUMNAS_HISTORIAL, COLUMNAS_CATALOGO


class ModeloCalidad:
    def __init__(self, usar_catalogo=True, usar_historial=True, usar_anomalia=True,
                 min_unidades_categoria=30):
        if not (usar_catalogo or usar_historial or usar_anomalia):
            raise ValueError("El modelo necesita al menos un bloque de información")
        self.usar_catalogo = usar_catalogo
        self.usar_historial = usar_historial
        self.usar_anomalia = usar_anomalia
        self.min_unidades_categoria = min_unidades_categoria
        self.n_arboles_ = None

    # ---------- (1) One-hot del catálogo ----------
    def _aprender_categorias(self, df):
        """Guarda qué letras existen en el train. Letras que aparecen en menos de
        `min_unidades_categoria` autos se ignoran (no hay evidencia suficiente)."""
        _, cat_cols = _grupos_de_columnas(df)
        self.categorias_ = {}
        for c in cat_cols:
            frec = df[c].value_counts()
            self.categorias_[c] = list(frec[frec >= self.min_unidades_categoria].index)

    def _one_hot(self, df):
        partes = {}
        for c, valores in self.categorias_.items():
            for v in valores:
                partes[f"{c}={v}"] = (df[c] == v).astype(int).values
        return pd.DataFrame(partes, index=df.index)

    # ---------- (2) Isolation Forest ----------
    def _entrenar_isolation_forest(self, df):
        hist_cols, _ = _grupos_de_columnas(df)
        self.isolation_forest_ = IsolationForest(n_estimators=300, random_state=SEMILLA)
        self.isolation_forest_.fit(df[hist_cols].astype(float))

    def score_anomalia(self, df):
        """Más alto = más raro. (score_samples devuelve lo contrario, por eso el -)."""
        hist_cols, _ = _grupos_de_columnas(df)
        return -self.isolation_forest_.score_samples(df[hist_cols].astype(float))

    # ---------- Armado de la matriz que ve XGBoost ----------
    def _matriz(self, df):
        hist_cols, _ = _grupos_de_columnas(df)
        bloques = []
        if self.usar_historial:
            bloques.append(df[hist_cols].astype(float))
        if self.usar_catalogo:
            bloques.append(self._one_hot(df))
        if self.usar_anomalia:
            bloques.append(pd.DataFrame({"score_anomalia": self.score_anomalia(df)},
                                        index=df.index))
        return pd.concat(bloques, axis=1)

    def _nuevo_xgboost(self, n_arboles, early_stopping):
        return XGBClassifier(
            n_estimators=n_arboles,
            learning_rate=0.03,     # pasos chicos: aprende despacio y generaliza mejor
            max_depth=4,            # árboles poco profundos: reglas simples
            min_child_weight=20,    # no crear reglas con muy pocos autos
            subsample=0.8,          # cada árbol ve el 80% de los autos...
            colsample_bytree=0.8,   # ...y el 80% de las columnas (reduce sobreajuste)
            reg_lambda=5.0,
            eval_metric="aucpr",
            early_stopping_rounds=early_stopping,
            random_state=SEMILLA,
            n_jobs=-1,
        )

    # ---------- (3) Entrenamiento completo ----------
    def fit(self, df, df_validacion=None, n_arboles=None):
        """Si se pasa df_validacion, XGBoost usa "early stopping": sigue agregando
        árboles mientras mejore en validación y se detiene cuando deja de mejorar.
        Si se pasa n_arboles, entrena exactamente esa cantidad."""
        self._aprender_categorias(df)
        if self.usar_anomalia:
            self._entrenar_isolation_forest(df)
        X = self._matriz(df)
        self.columnas_ = list(X.columns)
        if df_validacion is not None:
            self.xgb_ = self._nuevo_xgboost(2000, early_stopping=150)
            self.xgb_.fit(X, df["y"], eval_set=[(self._matriz(df_validacion), df_validacion["y"])],
                          verbose=False)
            self.n_arboles_ = self.xgb_.best_iteration + 1
        else:
            self.n_arboles_ = n_arboles or 300
            self.xgb_ = self._nuevo_xgboost(self.n_arboles_, early_stopping=None)
            self.xgb_.fit(X, df["y"], verbose=False)
        return self

    def predict_proba(self, df):
        X = self._matriz(df)[self.columnas_]
        return self.xgb_.predict_proba(X)[:, 1]

    def descripcion(self):
        partes = [n for n, u in [("catálogo", self.usar_catalogo),
                                 ("historial QLS", self.usar_historial),
                                 ("score anomalía", self.usar_anomalia)] if u]
        return " + ".join(partes)


# ============================ EVALUACIÓN ===================================

def metricas(y, score, presupuesto=PRESUPUESTO_INSPECCION):
    """Métricas pensadas para el problema real: tenemos capacidad para revisar
    solo el `presupuesto` (5%) de los autos. ¿Qué tan bien elegimos ese 5%?"""
    y = np.asarray(y)
    n = max(1, int(round(len(y) * presupuesto)))
    elegidos = np.argsort(-np.asarray(score))[:n]
    precision = y[elegidos].mean()
    return {
        "PR-AUC": average_precision_score(y, score),
        "ROC-AUC": roc_auc_score(y, score),
        "precision_top5": precision,             # de los que revisamos, % que calibró
        "recall_top5": y[elegidos].sum() / y.sum(),  # de todos los que calibraron, % que encontramos
        "lift_top5": precision / y.mean(),       # cuántas veces mejor que el azar
    }


def metricas_azar(y, presupuesto=PRESUPUESTO_INSPECCION):
    """Lo que se espera del método actual (muestreo aleatorio)."""
    base = np.mean(y)
    return {"PR-AUC": base, "ROC-AUC": 0.5, "precision_top5": base,
            "recall_top5": presupuesto, "lift_top5": 1.0}


def intervalo_lift(y, score, n_boot=500, presupuesto=PRESUPUESTO_INSPECCION):
    """Bootstrap: re-muestreamos el test muchas veces para ver cuánto varía el
    lift por pura suerte. Si el intervalo no incluye 1.0, la mejora es real."""
    rng = np.random.default_rng(SEMILLA)
    y, score = np.asarray(y), np.asarray(score)
    lifts = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        lifts.append(metricas(y[i], score[i], presupuesto)["lift_top5"])
    return np.percentile(lifts, [2.5, 97.5])


def importancia_por_permutacion(modelo, df, n_repeticiones=3):
    """¿Qué variables usa de verdad el modelo? Desordenamos una variable a la
    vez (rompemos su relación con la respuesta) y medimos cuánto empeora el
    PR-AUC. Si empeora mucho, esa variable importaba."""
    rng = np.random.default_rng(SEMILLA)
    y = df["y"].values
    base = average_precision_score(y, modelo.predict_proba(df))
    hist_cols, cat_cols = _grupos_de_columnas(df)
    variables = []
    if modelo.usar_historial or modelo.usar_anomalia:
        variables += hist_cols   # el score de anomalía se calcula a partir de estas
    if modelo.usar_catalogo:
        variables += cat_cols
    resultados = {}
    for v in variables:
        caidas = []
        for _ in range(n_repeticiones):
            copia = df.copy()
            copia[v] = rng.permutation(copia[v].values)
            caidas.append(base - average_precision_score(y, modelo.predict_proba(copia)))
        resultados[v] = np.mean(caidas)
    return pd.Series(resultados).sort_values(ascending=False)
