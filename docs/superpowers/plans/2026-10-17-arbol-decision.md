# Árbol de decisión para `paga_comision` — Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Entrenar un `DecisionTreeClassifier` sobre el pipeline de preprocesamiento existente, ajustarlo para maximizar la función de ganancia del enunciado y generar `casas_entrega.csv`.

**Architecture:** Todo vive en `primer_parcial/primer_parcial.ipynb`, en una sección nueva `# Modelado: árbol de decisión` agregada al final (después de la celda de exportación de CSV procesados). El modelo es `Pipeline([('preprocesamiento', preprocesamiento), ('modelo', DecisionTreeClassifier(...))])`, de modo que el preprocesamiento (incluido el `TargetEncoder` de `l3` y el recorte de `RecortarMaximo`) se ajusta solo con los datos de entrenamiento de cada split/fold. Los "tests" son celdas con `assert` dentro del notebook: si algo se rompe, la ejecución se frena.

**Tech Stack:** Python, pandas 3.0.5, scikit-learn 1.9.0 (`DecisionTreeClassifier`, `GridSearchCV`, `StratifiedKFold`, `make_scorer`, `cross_val_predict`, `plot_tree`), matplotlib.

**Spec:** `primer_parcial/2026-2C-Primer-Parcial.pdf` (enunciado) + doc de `sklearn.tree.DecisionTreeClassifier`.

## Global Constraints

- Ganancia: **+US$ 100** por cada casa predicha `paga` que efectivamente paga; **−US$ 50** por cada casa predicha `paga` que no paga. Predecir `no paga` vale 0.
- Evaluación: muestra de testing del **30%** (`test_size=0.3`, `stratify=y`, `random_state=42`). Referencia del enunciado: ~US$ 166.000 en test; aprobado si la ganancia sobre `casas_predecir` supera **US$ 500.000**.
- Entregable: `casas_entrega.csv` con los `id` de `casas_predecir.csv` predichos como `paga` (una sola columna `id`). Los ids **no** pueden ser de entrenamiento.
- `random_state=42` en todo lo que tenga azar (split, folds, árbol).
- El test (30%) se usa **una sola vez por modelo** para reportar ganancia: ni hiperparámetros ni umbral se eligen mirando test.
- `y` es 0/1 (1 = paga), ya definido en la celda del pipeline. `X`, `y`, `X_predecir`, `preprocesamiento` vienen de esa celda.
- No modificar las celdas existentes de limpieza, EDA ni pipeline.

## Review Focus

1. **NaN en `precio_usd` llegando al árbol**: el pipeline deja NaN a propósito. `DecisionTreeClassifier` los acepta desde sklearn 1.3 (con `splitter='best'`); si alguien cambia a `splitter='random'` o a otra versión, falla. Test: Task 2 verifica que `fit` funciona con NaN presentes.
2. **Umbral elegido con el test** (fuga): el umbral sale de `cross_val_predict` sobre train. Test: Task 4 asserta que el umbral se calculó con un array del largo de `X_train`.
3. **Entregable con ids de entrenamiento o con filas perdidas**: Test: Task 6, asserts del enunciado + que se predijo sobre las 41.054 filas.
4. **Categorías de `l1`/`l2` que no están en train** (aparecen en predecir o en un fold): los encoders ya usan `handle_unknown`; Test: Task 6 predice sobre `X_predecir` completo sin error.
5. **Árbol sin límite que memoriza** (ganancia en train ≫ test): Test: Task 2 imprime ambas; Task 3 asserta que el árbol ajustado mejora al base en test.

---

### Task 1: Función de ganancia y split train/test

**Files:**
- Modify: `primer_parcial/primer_parcial.ipynb` (agregar al final: markdown `# Modelado: árbol de decisión` + celda de código)

**Interfaces:**
- Consumes: `X`, `y` (celda del pipeline).
- Produces: `ganancia(y_true, y_pred) -> float`, `scorer_ganancia` (`make_scorer(ganancia)`), `X_train, X_test, y_train, y_test`, `cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.

- [ ] **Step 1: Escribir la celda de tests de `ganancia` (antes de definirla)**

```python
assert ganancia(np.array([1, 1, 0]), np.array([1, 0, 0])) == 100   # 1 acierto
assert ganancia(np.array([0, 0]), np.array([1, 1])) == -100         # 2 falsos positivos
assert ganancia(np.array([1, 0]), np.array([1, 1])) == 50           # 100 - 50
assert ganancia(np.array([1, 1]), np.array([0, 0])) == 0            # no predecir paga no cuesta
```

- [ ] **Step 2: Correr la celda → falla con `NameError: name 'ganancia' is not defined`**

- [ ] **Step 3: Implementar, en una celda arriba de los tests**

`ganancia(y_true, y_pred) -> float`: `100 * TP - 50 * FP` (con `y` en 0/1). Debajo: `scorer_ganancia = make_scorer(ganancia)`, el `train_test_split` según Global Constraints y `cv`. Imports nuevos en esta celda: `DecisionTreeClassifier`, `plot_tree`, `train_test_split`, `GridSearchCV`, `StratifiedKFold`, `cross_val_predict`, `make_scorer`, `Pipeline` ya está.

- [ ] **Step 4: Correr → pasa; agregar y correr asserts del split**

```python
assert len(X_test) == round(len(X) * 0.3)
assert abs(y_train.mean() - y_test.mean()) < 0.005    # estratificado
assert X_train.index.intersection(X_test.index).empty
```

Markdown breve antes de la celda explicando la función de ganancia y por qué accuracy no sirve (89% sin predecir nada).

---

### Task 2: Árbol base (sin ajustar)

**Files:**
- Modify: `primer_parcial/primer_parcial.ipynb` (markdown `## Modelo base` + 1 celda)

**Interfaces:**
- Consumes: Task 1.
- Produces: `modelo_base` (Pipeline ajustado), `ganancia_base_test: float`.

- [ ] **Step 1: Celda con el modelo y sus asserts**

`modelo_base = Pipeline([('preprocesamiento', preprocesamiento), ('modelo', DecisionTreeClassifier(random_state=42))])`, `fit(X_train, y_train)`. Imprimir ganancia en train y en test, profundidad (`modelo_base[-1].get_depth()`) y cantidad de hojas.

```python
assert X_train['title'].notna().any()  # el pipeline recibe X crudo
assert np.isnan(modelo_base[:-1].transform(X_train)['precio_usd']).any()  # hay NaN y el árbol igual entrenó
assert ganancia_base_test > 0
```

- [ ] **Step 2: Correr → pasa.** Esperado: ganancia en train mucho más alta que en test (sobreajuste). Markdown corto interpretando eso: motiva la Task 3.

---

### Task 3: Búsqueda de hiperparámetros con validación cruzada

**Files:**
- Modify: `primer_parcial/primer_parcial.ipynb` (markdown `## Ajuste de hiperparámetros` + 1–2 celdas)

**Interfaces:**
- Consumes: Task 1 (`scorer_ganancia`, `cv`), Task 2 (`ganancia_base_test`).
- Produces: `busqueda` (`GridSearchCV` ajustado), `mejores_params: dict`, `modelo_ajustado = busqueda.best_estimator_`, `ganancia_ajustada_test: float`.

- [ ] **Step 1: Celda de búsqueda**

`GridSearchCV(pipeline_arbol, param_grid, scoring=scorer_ganancia, cv=cv, n_jobs=-1)` con `fit(X_train, y_train)`. Grilla (96 combinaciones × 5 folds):

```python
param_grid = {
    'modelo__max_depth': [4, 6, 8, 10, 12, None],
    'modelo__min_samples_leaf': [1, 10, 50, 100],
    'modelo__criterion': ['gini', 'entropy'],
    'modelo__class_weight': [None, 'balanced'],
}
```

Mostrar las 10 mejores filas de `cv_results_` (params, `mean_test_score`, `std_test_score`) y `best_params_`.

- [ ] **Step 2: Evaluar en test y asserts**

```python
assert ganancia_ajustada_test > ganancia_base_test
assert busqueda.n_splits_ == 5
```

- [ ] **Step 3: Correr → pasa.** Si el mejor `max_depth` cae en un borde de la grilla (4 o 12), extender la grilla hacia ese lado y repetir. Markdown: qué hiperparámetros ganaron y qué significan (citar la doc).

---

### Task 4: Umbral de decisión para la ganancia

**Por qué:** `predict` usa umbral 0,5, pero con +100/−50 conviene predecir `paga` cuando `p > 1/3` (100·p − 50·(1−p) > 0). Se busca el mejor umbral con probabilidades fuera de fold sobre train.

**Files:**
- Modify: `primer_parcial/primer_parcial.ipynb` (markdown `## Umbral de decisión` + 1 celda + gráfico)

**Interfaces:**
- Consumes: Task 3 (`modelo_ajustado`, `cv`), Task 1 (`ganancia`).
- Produces: `umbral: float`, `ganancia_umbral_test: float`.

- [ ] **Step 1: Celda**

`proba_oof = cross_val_predict(modelo_ajustado, X_train, y_train, cv=cv, method='predict_proba')[:, 1]`. Recorrer `np.arange(0.05, 0.95, 0.01)`, calcular `ganancia(y_train, proba_oof >= u)`, quedarse con el máximo. Graficar ganancia vs umbral con líneas en 0,5 y 1/3. Evaluar en test: `ganancia(y_test, modelo_ajustado.predict_proba(X_test)[:, 1] >= umbral)`.

```python
assert len(proba_oof) == len(X_train)       # el umbral NO se eligió con test
assert 0 < umbral < 1
```

- [ ] **Step 2: Correr → pasa.** Tabla final en markdown: base / ajustado / ajustado + umbral, en test, contra la referencia de US$ 166.000. Si el umbral no mejora en test, quedarse con 0,5 y decirlo.

---

### Task 5: Interpretación del árbol

**Files:**
- Modify: `primer_parcial/primer_parcial.ipynb` (markdown `## ¿Qué aprendió el árbol?` + 2 celdas)

**Interfaces:**
- Consumes: `modelo_ajustado`.
- Produces: nada que usen otras tasks.

- [ ] **Step 1: Importancia de variables**: `pd.Series(modelo_ajustado[-1].feature_importances_, index=modelo_ajustado[:-1].get_feature_names_out())`, top 15 en barras horizontales. Assert: `abs(importancias.sum() - 1) < 1e-9`.
- [ ] **Step 2: Árbol dibujado**: `plot_tree(..., max_depth=3, feature_names=..., class_names=['no paga', 'paga'], filled=True, proportion=True)`.
- [ ] **Step 3: Correr → pasa.** Markdown contrastando con las conclusiones del EDA (se espera `precio_usd`, `surface_covered`, `bathrooms`, `l3` arriba).

---

### Task 6: Modelo final y `casas_entrega.csv`

**Files:**
- Modify: `primer_parcial/primer_parcial.ipynb` (markdown `## Entregable` + 1 celda)
- Create: `primer_parcial/casas_entrega.csv`

**Interfaces:**
- Consumes: `mejores_params` (Task 3), `umbral` (Task 4), `X`, `y`, `X_predecir`, `df_casas_predecir`, `df_casas_entrenamiento`.
- Produces: `modelo_final`, `ids_paga`, archivo `casas_entrega.csv`.

- [ ] **Step 1: Celda**

`modelo_final = clone(modelo_ajustado).fit(X, y)` (reentrena con todo el entrenamiento: 70% + 30%). `proba_predecir = modelo_final.predict_proba(X_predecir)[:, 1]`, `ids_paga = X_predecir.index[proba_predecir >= umbral]`, `pd.DataFrame(index=ids_paga).to_csv('casas_entrega.csv', index_label='id')`. Imprimir cuántas casas y qué porcentaje se predicen `paga`.

```python
assert len(proba_predecir) == len(df_casas_predecir) == 41054
assert ids_paga.isin(df_casas_predecir['id']).all()          # ids de predecir
assert not ids_paga.isin(df_casas_entrenamiento['id']).any() # ninguno de entrenamiento
assert ids_paga.is_unique and len(ids_paga) > 0
releido = pd.read_csv('casas_entrega.csv')
assert list(releido.columns) == ['id'] and len(releido) == len(ids_paga)
```

(`df_casas_*` tienen `id` como columna; el índice lo pone solo el pipeline.)

- [ ] **Step 2: Correr → pasa.** Sanity: el % predicho `paga` debe ser del orden del % que el modelo predice `paga` en test (si difiere mucho, revisar antes de entregar).

---

### Verificación final

- [ ] Correr el notebook completo de punta a punta. **Bloqueante conocido:** la celda de nulos del EDA falla con `NameError: df_eda` hasta que se agregue la celda que crea `df_eda` y `tasa_global` (pendiente de decisión del usuario). Mientras tanto, verificar con el script temporal que inyecta esa definición solo en memoria y corre todas las celdas de código con los CSV locales; esperado: `TODO OK` y `casas_entrega.csv` generado.
- [ ] Commit solo si el usuario lo pide.
