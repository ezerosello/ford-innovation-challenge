# FIC III — Desafío 3: Data-Driven Predictive Quality

**Equipo Algoritmos & Caballos:** Nicolas Cardone, Ezequiel Rosello, Valentin Provost

Herramienta que, a partir del historial QLS de cada vehículo, estima qué unidades tienen más probabilidad de necesitar calibración en la **Inspección Adicional**. Con eso arma un ranking diario para que el cupo de inspección (5% de las unidades) se use donde más rinde, en lugar de elegirlo al azar como se hace hoy.

El programa tiene tres pasos, que se corren en orden:

| Paso | Script | Qué produce | Tiempo aprox. |
|---|---|---|---|
| 1. Preparar datos | `paso1_preparar_datos.py` | Tabla limpia, una fila por vehículo, separada en train/test | 30–60 s (la primera vez) |
| 2. Modelo | `paso2_modelo.py` | Modelo entrenado, comparación de variantes, ranking y gráficos | 2–4 min |
| 3. Dashboard | `paso3_dashboard.py` | Página web `dashboard.html` | menos de 1 min |

---

## 1. Instalación

### Requisitos

- **Python 3.10 o superior, de 64 bits.** Recomendado: Python 3.12 desde [python.org](https://www.python.org/downloads/) → "Windows installer (64-bit)". Durante la instalación, tildar **"Add python.exe to PATH"**.
- El Excel del dataset de Ford (`Dataset_QLS_InspecciónAdicional.xlsx`).

> Python de 32 bits **no sirve**: XGBoost no tiene versión para 32 bits en Windows y scikit-learn no se instala.
> Para verificar: `py -3.12 -c "import struct; print(struct.calcsize('P')*8)"` tiene que imprimir `64`.

### Instalar las librerías (una sola vez)

Desde la carpeta `fic_qls`:

```
py -3.12 -m pip install -r requirements.txt
```

En Mac/Linux, reemplazar `py -3.12` por `python3` en todos los comandos.

### Ubicar el Excel

Copiar el Excel dentro de la carpeta `data/`. El nombre no importa mientras termine en `.xlsx` (recomendado: `dataset.xlsx`, sin tildes).

---

## 2. Uso rápido

```
py -3.12 paso1_preparar_datos.py
py -3.12 paso2_modelo.py
py -3.12 paso3_dashboard.py
```

Después abrir `salidas/dashboard.html` con doble clic. No necesita servidor ni conexión a internet.

- Cada paso usa lo que dejó el anterior en `salidas/`. Si se modifica un paso, hay que volver a correr ese paso y los siguientes.
- El paso 1 guarda una copia rápida del Excel. Para rehacer todo desde el Excel original: `py -3.12 paso1_preparar_datos.py --forzar`.

---

## 3. Estructura del proyecto

```
fic_qls/
├── data/                       <- acá va el Excel
├── salidas/                    <- todo lo que genera el programa
├── config.py                   <- rutas y decisiones (días de corte, cupo, etc.)
├── ingest.py                   <- paso 1a: lectura del Excel
├── clean.py                    <- paso 1b: limpieza
├── features.py                 <- paso 1c: tabla por vehículo
├── split.py                    <- paso 1d: separación cronológica
├── paso1_preparar_datos.py     <- corre todo el paso 1
├── model.py                    <- pipeline del modelo y métricas
├── paso2_modelo.py             <- corre todo el paso 2
├── paso3_dashboard.py          <- calcula los datos del dashboard
├── dashboard_template.html     <- diseño de la página (el paso 3 le inserta los datos)
└── requirements.txt
```

**Por qué está dividido así:** cada archivo tiene una sola responsabilidad. Si Ford entregara los datos en otro formato, solo cambiaría `ingest.py`; si se quisiera cambiar el día de corte o el cupo de inspección, solo `config.py`. Esto es lo que hace que la solución sea mantenible y reproducible, y no un notebook que se corrió una vez.

---

## 4. Paso 1 — Preparación de datos

**Objetivo:** pasar del Excel crudo (una fila por defecto registrado) a una tabla donde cada fila es un vehículo, lista para entrenar un modelo.

### 1a. Ingesta (`ingest.py`)

Lee el Excel tal como viene. La primera fila del Excel es una descripción de cada columna y la segunda tiene los nombres; por eso se lee con `header=1`. También guarda esas descripciones en `salidas/diccionario_columnas.csv`, útil para entender qué significa cada variable.

### 1b. Limpieza (`clean.py`)

Sigue siendo una fila por defecto. Lo que hace:

| Acción | Por qué |
|---|---|
| Renombra columnas a nombres cortos sin tildes (`Componente Inspección` → `componente`) | Más cómodo y sin errores de tipeo |
| Quita espacios sobrantes en los textos (`"FC "` → `"FC"`) | Si no, el mismo valor se cuenta como dos distintos |
| Convierte `DIA_12` → `12` y las horas (fracción del día en Excel) → horas reales | Para poder ordenar y restar tiempos |
| Calcula la demora de cada reparación en horas | Es una característica del proceso |
| Crea la variable objetivo `y` = 1 si la unidad quedó CALIBRADA | Es lo que el modelo tiene que predecir |
| Elimina 477 filas duplicadas exactas | El mismo defecto cargado dos veces inflaría los conteos |
| Elimina 6 columnas casi vacías o constantes | Una columna que no varía no le enseña nada al modelo |
| Valida supuestos (cada VIN tiene un solo resultado y un solo catálogo) | Si algo no cumple, el programa se detiene con un error claro |

Resultado: 195.331 defectos de 59.681 vehículos; 10,2% de los vehículos quedaron CALIBRADOS.

### 1c. Tabla por vehículo (`features.py`)

La decisión de inspeccionar se toma por auto, no por defecto. Por eso cada vehículo se resume en una fila de números llamados **features**. Son 74, en cuatro grupos:

| Grupo | Ejemplos | Idea |
|---|---|---|
| Catálogo descompuesto | letra 2, letra 3, letra 4, letras 2 y 3 | El código de catálogo combina versión, mercado y año-modelo; separado en letras, cada parte aporta por su cuenta |
| Volumen de defectos | cantidad de defectos, componentes distintos, componentes repetidos | Cuánto "trabajo" tuvo la unidad en el QLS |
| Tiempos | demora media y máxima de reparación, días en QLS | Cómo fluyó por el proceso |
| Qué componente y qué área | defectos por área (VRT), por grupo de función (VFG) y por zona que lo detectó | La pista de Ford: qué componente falló y a qué área pertenece |

**Columnas que nunca entran al modelo** (listadas en `NO_FEATURES`):

- `componente_auditoria`: solo tiene valor si la unidad fue calibrada. Usarla sería mirar la respuesta (*data leakage*).
- Las fechas: se usan para ordenar y separar, no para predecir. Un modelo de árboles no sabe extrapolar a días que nunca vio.
- `y`: es la respuesta.

**Momento de la predicción:** el último registro de la unidad en el QLS, antes del Gate Release. Todas las features usan solo información disponible en ese momento.

### 1d. Split cronológico (`split.py`)

| Grupo | Días | Unidades | Tasa de calibración | Uso |
|---|---|---|---|---|
| Train | 1–200 | 41.772 | 12,0% | Entrenar |
| Test | 201–260 | 12.834 | 8,3% | Medir el resultado final |
| Post-corte | 261–284 | 5.075 | 0,1% | Mostrar el cambio de régimen (monitoreo) |

**Por qué cronológico y no al azar:** en planta el modelo se entrena con el pasado y se usa sobre autos nuevos. Mezclar al azar mediría algo que en la realidad no va a pasar y daría un resultado optimista.

**Por qué el post-corte va aparte:** Ford mencionó una medida de corte cerca del día 260. En los datos se confirma: desde el día 261 casi no hay calibraciones. Incluir esos días en el test haría que las métricas no tengan sentido.

### Archivos que genera

`01_crudo.pkl`, `02_limpio.pkl`, `03_tabla_vin.pkl`, `04_train.pkl`, `04_test.pkl`, `04_post_corte.pkl`, `diccionario_columnas.csv`.

---

## 5. Paso 2 — Modelo

**Objetivo:** entrenar el modelo propuesto, medir cuánto aporta cada parte y evaluar el resultado final sobre datos que el modelo nunca vio.

### El pipeline (`model.py`)

Cada vehículo pasa por tres etapas:

1. **Codificación del catálogo (one-hot).** Un modelo solo entiende números. "La letra 3 del catálogo es F" se convierte en una columna que vale 1 o 0.
2. **Isolation Forest (no supervisado).** Mira el historial QLS sin saber si la unidad calibró o no, y le asigna un **score de anomalía**: qué tan distinta es del comportamiento habitual. La idea es que una unidad rara se puede aislar del resto con pocos cortes al azar.
3. **XGBoost (supervisado).** Recibe todas las features más el score de anomalía como una columna adicional, y aprende de los resultados históricos (OK / CALIBRADA). Funciona como cientos de árboles de decisión chicos, donde cada uno corrige los errores de los anteriores. Devuelve una **probabilidad de calibración** por unidad.

El acople entre los dos modelos es literalmente una columna: la salida del Isolation Forest es una entrada más de XGBoost.

### Tres conjuntos, tres roles

| Conjunto | Días | Para qué se usa |
|---|---|---|
| Ajuste | 1–180 | El modelo aprende acá |
| Validación | 181–200 | Decidir cuándo dejar de agregar árboles (*early stopping*) y qué variante elegir |
| Test | 201–260 | Se usa una sola vez, al final. Ninguna decisión se tomó mirándolo, por eso su resultado es creíble |

**Early stopping:** XGBoost agrega árboles mientras mejora en validación y se detiene cuando deja de mejorar. Si se siguieran agregando, el modelo empezaría a memorizar ruido del pasado (sobreajuste).

### Comparación de variantes (ablation)

Para no suponer qué aporta cada parte, se entrena el modelo prendiendo y apagando cada bloque de información:

- Solo catálogo
- Solo historial QLS
- Historial + score de anomalía
- Catálogo + historial
- Propuesta completa (catálogo + historial + Isolation Forest)
- Solo Isolation Forest, sin etiquetas (¿"raro" equivale a "necesita calibración"?)

**Regla de elección, definida de antemano:** gana el mejor PR-AUC en validación. Si una variante más simple queda a menos de 3% de la mejor, se elige la simple: menos piezas, menos cosas que se pueden romper y más fácil de explicar. Después se reentrena la elegida con todo el train (días 1–200) y se mide en test.

### Métricas

| Métrica | Qué responde |
|---|---|
| Precisión en el 5% | De las unidades que revisamos, ¿qué porcentaje necesitaba calibración? |
| Recall en el 5% | De todas las que necesitaban calibración, ¿qué porcentaje encontramos? |
| Lift | ¿Cuántas veces mejor que elegir al azar? |
| PR-AUC | Calidad general del ranking; adecuada cuando los positivos son pocos (~10%) |
| Intervalo de confianza del lift (bootstrap) | Se re-muestrea el test 500 veces. Si el intervalo no incluye 1,0, la mejora no es suerte |
| Importancia por permutación | Se desordena una variable a la vez y se mide cuánto empeora el modelo. Cuanto más empeora, más dependía de ella |

### Resultados de referencia (test, días 201–260)

| | Azar (hoy) | Modelo |
|---|---|---|
| Precisión en el 5% revisado | 8,3% | 12,5%–13,7% |
| Lift | 1,0x | 1,5x–1,65x (intervalo 95%: ~1,2x a ~1,9x) |
| Selección diaria del 5% | 8,3% | 12,2%–12,7% |

Lo que muestran las variantes:

- **Isolation Forest solo rinde peor que el azar** (~0,9x): raro no es lo mismo que riesgoso.
- **El historial QLS solo no tiene señal** (~1,05x).
- **La mejora viene del catálogo** (versión/mercado). Las variables más importantes son letras del catálogo.
- Agregar historial o score de anomalía al catálogo no mejora el resultado en este dataset. La arquitectura queda preparada para cuando haya parámetros de proceso reales (tiempos de ciclo, mediciones), que el dataset ficticio no incluye.
- **Post-corte:** el modelo espera ~10,5% de calibraciones y la realidad es 0,1%. El proceso cambió y el modelo no se enteró: por eso hace falta monitoreo y reentrenamiento (paso 3).

> Los números pueden variar un poco entre computadoras por diferencias de versión de XGBoost/scikit-learn (ver "Problemas frecuentes"). Las conclusiones no cambian.

### Archivos que genera

| Archivo | Contenido |
|---|---|
| `05_comparacion_modelos.csv` | Métricas de cada variante en test |
| `06_ranking_test.csv` | Probabilidad y prioridad diaria de cada unidad del test |
| `07_importancia_variables.csv` | Importancia de cada variable |
| `grafico_curva_captura.png` | % de calibraciones encontradas vs % de autos revisados |
| `grafico_importancia.png` | Variables que más usa el modelo |
| `modelo_final.pkl` | El modelo entrenado |

MLflow es opcional: si está instalado (`py -3.12 -m pip install mlflow`), cada corrida queda registrada con sus parámetros y métricas, y se consulta con `mlflow ui`.

---

## 6. Paso 3 — Dashboard

**Objetivo:** presentar las predicciones de forma accionable para el equipo de calidad, como pide la sección 5 de la ficha técnica.

`paso3_dashboard.py` calcula todos los datos y los inserta en `dashboard_template.html`, generando `salidas/dashboard.html`. Es un solo archivo, sin dependencias: se puede mandar por mail o abrir en la presentación sin internet.

Para verlo: doble clic en `salidas/dashboard.html`, o desde la terminal `start salidas/dashboard.html`. Si se abre en un editor de código, usar clic derecho → "Abrir con" → Chrome o Edge.

También genera `salidas/08_unidades_a_revisar.csv`: la lista completa de unidades a revisar de todos los días (201 en adelante), con prioridad, probabilidad, qué revisar primero, por qué y el resultado real. Está separada por `;` para que Excel en español la abra directamente en columnas.

### Qué muestra

**Día de producción.** Selector de día (también con las flechas del teclado) y buscador de VIN. Para el día elegido: unidades producidas, cupo de revisión, cuántas de las elegidas necesitaban calibración y cuántas se esperaban al azar.

**Lista priorizada.** Unidades del día de mayor a menor riesgo; las filas amarillas son el cupo del 5%. Para cada unidad:

- **Qué revisar primero:** los 3 componentes (VFG), con su área (VRT), que más se calibraron históricamente en esa versión y mercado. Se calculan combinando de lo general a lo particular (letra 2 → letra 3 → letras 2 y 3 → catálogo completo): cuantos más autos hay en un grupo, más se confía en él. En test, el componente real estuvo entre los 3 sugeridos en ~45% de los casos, contra ~40% sugiriendo siempre los 3 más frecuentes.
- **Por qué subió su riesgo:** las 3 variables que más empujaron la probabilidad hacia arriba, calculadas con valores SHAP (XGBoost descompone cada predicción en "cuánto sumó cada variable").
- **Resultado real:** solo para validar la demo. En planta ese dato se conoce después de la inspección.

**Monitoreo del proceso.** Gráfico de control p de la tasa diaria de calibración. Cada día se compara con los 30 días normales anteriores (rango de ±3 desvíos). Si cae afuera, es alarma, y ese día no se usa como referencia para que la alarma no se "normalice" sola. Resultado: ninguna falsa alarma antes del día 260, y alarma en todos los días con datos suficientes desde el día 261. **El sistema habría detectado la medida de corte el mismo día.** Hacer clic en un punto lleva a ese día.

**Todo el período.** Totales de todos los días (cuántas unidades revisadas necesitaban calibración contra lo esperable al azar), una tabla con el resumen de cada día (clic en una fila para ir a ese día) y el botón **Descargar lista completa (CSV)** con todas las unidades a revisar de todos los días.

La selección se hace **día por día** y no sobre el total: los autos llegan a diario y se inspeccionan entre 0 y 5 días después del Gate Release, así que no se puede esperar al final del período para elegir el 5% de mayor riesgo. La lista total es la suma de las listas diarias.

**¿Rinde más que el azar?** Curva de captura, precisión del modelo contra el azar y tabla de variantes.

**Qué mira el modelo.** Importancia de variables.

---

## 7. Decisiones clave (para defender ante el jurado)

| Decisión | Justificación |
|---|---|
| Una fila por vehículo | La decisión de inspeccionar es por auto |
| Split cronológico | Simula el uso real: entrenar con el pasado, predecir el futuro |
| Excluir el post-corte de la evaluación | Es otro régimen del proceso; se usa para el monitoreo |
| Validación separada del test | Todas las decisiones se toman sin mirar el test |
| Comparación de variantes | Cada componente del modelo se justifica con números, no con supuestos |
| Métricas en el 5% | Reflejan la capacidad real de inspección de la planta |
| PR-AUC en vez de accuracy | Con 10% de positivos, un modelo que dice "todo OK" tendría 90% de accuracy sin servir para nada |
| Pandas en vez de PySpark | 195.000 filas entran en memoria; PySpark es el camino para escalar a millones de registros |
| Monitoreo con gráfico de control | El proceso cambia (día 260) y el modelo tiene que enterarse |

---

## 8. Problemas frecuentes

| Error | Causa | Solución |
|---|---|---|
| Falla `pip install` compilando scipy/scikit-learn | Python de 32 bits o muy viejo | Instalar Python 3.12 de 64 bits (ver Instalación) |
| `FileNotFoundError` con la ruta del Excel | El Excel no está en `data/` o el nombre no coincide | Copiarlo a `data/`. Si tu `config.py` tiene una línea `RUTA_EXCEL = ...`, el nombre tiene que coincidir exactamente con esa línea |
| `ImportError: cannot import name 'DIA_FIN_AJUSTE'` | `config.py` es de una versión anterior | Usar el `config.py` actual del proyecto |
| "Primero corré: python paso1..." | Falta correr un paso anterior | Correr los pasos en orden |
| Mis números difieren un poco de los de referencia | Distintas versiones de XGBoost/scikit-learn | Normal. Para que el equipo tenga siempre el mismo resultado, fijar versiones: `py -3.12 -m pip freeze > requirements-lock.txt` y usar esa computadora para la presentación |
| El dashboard se ve con otra tipografía | Sin internet no carga la fuente | Es solo estético; todo funciona igual |

### Alternativa sin instalar nada: Google Colab

Subir el zip del proyecto y el Excel al panel de archivos de Colab, y ejecutar en una celda:

```
!unzip -o fic_qls.zip && cd fic_qls && mv ../*.xlsx data/
%cd fic_qls
!python paso1_preparar_datos.py
!python paso2_modelo.py
!python paso3_dashboard.py
```

Después descargar `salidas/dashboard.html` desde el panel de archivos. Colab borra los archivos al cerrar la sesión.

---

## 9. Limitaciones y próximos pasos

- **Datos ficticios:** el dataset fue generado por Ford para el desafío y no incluye parámetros de proceso (tiempos de ciclo, mediciones). Con datos reales, el historial y el score de anomalía podrían aportar más.
- **Mejora del modelo (acotada):** alrededor de 1,5x el azar. Es real (el intervalo de confianza no incluye 1,0) pero moderada.
- **Validación con una sola ventana:** la elección de variante usa 20 días de validación, lo que la hace sensible al azar. Próximo paso: validación con varias ventanas móviles.
- **Reentrenamiento:** en producción, el modelo debería reentrenarse periódicamente con una ventana móvil de datos recientes, y siempre que el gráfico de control marque una alarma sostenida.
