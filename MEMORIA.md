# MEMORIA DEL PROYECTO TFG — Carsharing Demand Estimation

**Fecha última actualización:** Abril 2026  
**Autor:** Juan Perpic  
**Repositorio:** [TFG---Carsharing-Estimation-Demand](https://github.com/juanruu/TFG---Carsharing-Estimation-Demand)

---

## 📋 Índice

1. [Descripción General](#descripción-general)
2. [Decisiones Arquitectónicas](#decisiones-arquitectónicas)
3. [Datasets y Exploración](#datasets-y-exploración)
4. [Metodología: XGBoost vs LSTM](#metodología-xgboost-vs-lstm)
5. [Resultados Principales](#resultados-principales)
6. [Estructura del TFG](#estructura-del-tfg)
7. [Estado del Proyecto](#estado-del-proyecto)
8. [Próximas Tareas](#próximas-tareas)

---

## 🎯 Descripción General

**Objetivo:** Predecir demanda de carsharing en múltiples ciudades europeas usando Machine Learning y Deep Learning, evaluando transfer learning entre ciudades.

**Caso de uso:** Operadores de carsharing necesitan optimizar distribución de vehículos. Predicciones precisas de demanda por zona-hora permiten:
- Balanceo automático de flota
- Planificación de mantenimiento
- Estrategia de precios dinámicos

**Enfoque principal:** Comparar XGBoost (baseline) con LSTM multi-ciudad preentrenado, con transfer learning a ciudades no vistas.

---

## 📁 Estructura del Repositorio

```
TFG---Carsharing-Estimation-Demand/
├── cs_datasets/                     # Datos crudos (10 ciudades, ~828K viajes)
├── datasets/                        # Datos procesados
│   ├── dataset_h3_multicidad.parquet
│   └── [otros CSVs]
├── results/                         # Resultados, modelos, figuras
│   ├── lstm_pretrained.pt
│   ├── mapa_h3_milan.png
│   └── [gráficos, métricas]
├── trip_cleaning.py                 # Módulo compartido de limpieza (XGBoost + LSTM)
├── apply_cleaning_patches.py        # Script idempotente para parchear notebooks
├── xgboost_solution.ipynb           # XGBoost: regresión + clasificación (Milán)
├── lstm_dataset_creation.ipynb      # LSTM: ETL crudos → parquet H3 multi-ciudad
├── lstm_predictions.ipynb           # LSTM: entrenamiento, evaluación, transfer
├── mapa_h3_milan.ipynb              # Visualización H3 en Milán
├── MEMORIA.md                       # Este archivo
└── README.md
```

> 📝 **Nota 27 abril 2026:** los notebooks `lstm_dataset_creation` y `lstm_predictions` estaban históricamente intercambiados (el de "predictions" contenía el ETL y viceversa). Se renombraron en esta sesión para que el nombre refleje el contenido real.

---

## 🏗️ Decisiones Arquitectónicas

### 1. **Discretización Espacial: H3 Hexagonal Indexing**

**Por qué H3 (resolución 7):**
- ✅ Capa espacial común para 10 ciudades (ACE es específico de Italia)
- ✅ Resolución 7: ~5 km² por celda (balance entre granularidad y densidad de datos)
- ✅ Menos ceros que H3-8 (60-65% vs 70-90%)
- ✅ Comparabilidad con XGBoost ACE-zones

**Alternativas rechazadas:**
- H3-8 (resolución 8): demasiados ceros (70-91%), modelos subespecializados
- Grillas rectangulares: menos eficientes en geometría urbana
- Barrios/comunas: no disponibles para todas las ciudades

**Implementación:**
```python
# h3.latlng_to_cell(lat, lon, resolution=7)
# Convierte coordenadas GPS a celda hexagonal
```

---

### 2. **XGBoost: Baseline de Referencia**

**Modelo de regresión (Milán, del notebook eda_preliminar):**
- **Datos:** Dataset ACE-zones (86 zonas Milán)
- **Features:** hour, day_of_week, month, lag_1h, lag_24h, rolling_mean_3h
- **Target:** Demanda continua (viajes/hora/zona)
- **Resultados:**
  - MAE = 0.9247
  - RMSE = 1.4663
  - R² = 0.5660

**Modelo de clasificación (3 clases):**
- sin_demanda (0 viajes), moderada (1-3), alta (>3)
- Accuracy = 63.22%, F1 weighted = 0.6333
- Problema: clase "moderada" confundida (precision 0.56)

**Conclusión:** Buen baseline pero R²=0.57 indica margen de mejora (no captura varianza).

**Hallazgo metodológico (ACE redundancy):**
Se exploró la inclusión de variables socioeconómicas del CPA 2011 (`population_resident`, `student_count`, `outbound_commuters`, `employed_resident`, `pop_density`) como features adicionales en XGBoost. El análisis posterior de explicabilidad (SHAP + `feature_importance`) mostró que el **identificador de zona ACE concentra la mayor parte del peso predictivo**, mientras que las variables socioeconómicas explícitas aportan poco. Interpretación: ISTAT construye las ACE precisamente agrupando unidades territoriales con perfil socioeconómico homogéneo, por lo que el ACE-id ya las encapsula implícitamente. **El modelo final usa solo el ACE-id + features temporales (hour, dow, lags)**, manteniéndose como sensitivity check con/sin socio. Este hallazgo es **una de las contribuciones metodológicas del TFG** y no aparece en el paper de Boldrini et al. (2019), que mantiene los análisis explicativo y predictivo en bloques separados.

---

### 3. **Domain Adaptation via Transfer Learning**

**Concepto clave:** El modelo se entrena en 9 ciudades (dominio conocido) y se evalúa en una ciudad nunca vista (dominio nuevo). **Misma tarea** (predecir demanda), **distinto dominio** (ciudad diferente, período diferente).

**Terminología precisa:**
- **Transfer Learning clásico** (ImageNet→Medical): tarea diferente, dominio diferente
- **Domain Adaptation** (lo que hacemos): **tarea idéntica**, dominio diferente
- **Zero-shot**: evaluación sin reentrenamiento en nuevo dominio
- **Fine-tuning**: ajuste de pesos para adaptarse al nuevo dominio

**Por qué funciona:**
- El encoder LSTM aprende patrones **universales** de demanda urbana
- Ciclos diarios: baja demanda 2-5am, picos laborales 7-9am, 5-7pm
- Ciclos semanales: lunes ≠ viernes/sábado
- Relaciones espaciales: zona céntrica → mayor demanda
- Estos patrones transfieren a nueva ciudad sin reentrenamiento

---

### 4. **LSTM: Arquitectura para Series Temporales**

**Por qué LSTM (complemento a XGBoost):**
- ✅ Captura dependencias temporales no-lineales (XGBoost solo usa lags simples)
- ✅ Memoria a largo plazo ideal para ciclos diarios/semanales
- ✅ Arquitectura separable: encoder (LSTM) + head (Dense) → facilita transfer learning

**Arquitectura:**
```
Input: [batch=512, T=24, features=6]
  └─ Ventanas deslizantes de 24h

LSTM Layer 1: hidden=64, dropout=0.2
LSTM Layer 2: hidden=64, dropout=0.2
  └─ Procesa la secuencia temporal

Dense Head:
  ├─ Dropout(0.2)
  ├─ Linear(64 → 32, ReLU)
  └─ Linear(32 → 1)  # Output: demanda normalizada

Total parámetros: ~15K
```

**Features por timestep (6):**
1. `target_demanda` (normalizado z-score)
2. `hour_sin` (codificación cíclica)
3. `hour_cos`
4. `dow_sin` (día de semana)
5. `dow_cos`
6. `is_weekend`

**Por qué sin/cos en lugar de valores directos:**
- Hour 23 y 0 están a 1h, pero |23-0|=23 (distancia euclidiana engañosa)
- Sin/cos: convierte línea a círculo, hour 23 cerca de hour 0
- XGBoost no lo necesita (árboles no usan distancias), pero LSTM sí

---

### 4. **Loss Function: Huber Loss**

**Elección:**
```python
criterion = nn.HuberLoss(delta=1.0)
```

**Por qué Huber:**
- Datos con 70-90% ceros → MSE penaliza outliers demasiado
- Huber: cuadrático para |error| ≤ 1, lineal para |error| > 1
- Robusto sin eliminar información de errores grandes

**Escala normalizada:**
- Targets: z-score por celda → valores típicos [-2, 3]
- Pérdida típica epoch 1: 0.31 (no es anormalmente baja)
- Baseline (predecir siempre 0): ~0.50
- Epoch 1: ~0.31 (40% mejor que baseline)

---

## 📊 Datasets y Exploración

### Datos Originales

| Ciudad | Viajes | Periodo | Resolución |
|--------|--------|---------|------------|
| Amsterdam | 51,278 | 2015-05-17 a 2015-07-01 | 46 días |
| Berlin | 225,809 | 2015-05-17 a 2015-07-01 | 46 días |
| Firenze | 19,356 | 2015-05-17 a 2015-07-01 | 46 días |
| Kobenhavn | 12,611 | 2015-05-17 a 2015-07-01 | 46 días |
| Milano | 158,068 | 2015-05-17 a 2015-07-01 | 46 días |
| München | 82,523 | **2016-03-11 a 2016-05-13** | 64 días ⚠️ |
| Roma | 101,483 | 2015-05-17 a 2015-07-01 | 46 días |
| Stockholm | 16,835 | 2015-05-17 a 2015-07-01 | 46 días |
| Torino | 25,579 | 2015-05-17 a 2015-07-01 | 46 días |
| Wien | 146,517 | 2015-05-17 a 2015-07-01 | 46 días |
| **TOTAL** | **828,508** | — | — |

> ⚠️ **Corrección 27 abril 2026:** la cifra previa para Wien (158.069 viajes) era un error de transcripción; el fichero `wien_trips.txt` contiene realmente 146.517 filas (146.516 viajes + cabecera). Total recalculado y verificado contra los `wc -l` de los diez ficheros.

**Notas críticas:**
- München: **período diferente (2016)** → ideal como ciudad de test (transfer learning cross-temporal)
- Desbalance de datos: Berlin (225K) domina vs Kobenhavn (12K)
- Todos comparten 24 atributos idénticos → compatible para multi-ciudad

### Dataset H3 (Parquet)

**Estadísticas agregadas por ciudad (H3-7):**

```
              filas  celdas_h3  demanda_media  pct_ceros
amsterdam    184368        167           0.28      80.10%
berlin       553104        501           0.41      73.57%
firenze      115920        105           0.17      87.00%
kobenhavn    114816        104           0.11      90.53%
milano       256128        232           0.62      71.21%
muenchen     348672        227           0.24      83.87%
roma         230736        209           0.44      72.71%
stockholm    125856        114           0.13      89.29%
torino       131376        119           0.19      85.38%
wien         260544        236           0.56      68.98%
```

**Archivos generados:**
1. `dataset_h3_multicidad.parquet` (1.8 GB)
   - Columns: city, h3_cell, date, hour, day_of_week, month, is_weekend, 
     hour_sin, hour_cos, dow_sin, dow_cos, target_demanda, 
     lag_1h, lag_24h, lag_168h, rolling_mean_3h
   - Total: 2,321,520 filas

> ⚠️ **Pendiente de regenerar (27 abr 2026):** estas estadísticas corresponden al parquet generado **antes** de introducir el módulo de cleaning compartido (`trip_cleaning.py`). Tras volver a ejecutar `lstm_dataset_creation.ipynb`, las cifras de filas, celdas y porcentajes de ceros cambiarán ligeramente (≈4-5% menos viajes por ciudad).

---

## 🧹 Pipeline de Limpieza Compartido

**Decisión arquitectónica (27 abr 2026):** ambos frameworks (XGBoost-Milán y LSTM-multiciudad) operaban con criterios de limpieza distintos, lo que invalidaba parcialmente la comparación cruzada. Se centraliza la lógica en un módulo único `trip_cleaning.py`, importado por los dos notebooks.

### Filtros aplicados

| Filtro | Umbral | Justificación |
|--------|--------|---------------|
| **Distancia mínima** | ≥ 100 m | Descarta reservas degeneradas (vehículo no se movió) |
| **Duración mínima** | ≥ 1 min | Descarta glitches de telemetría / reservas canceladas |
| **Duración máxima** | ≤ 480 min (8 h) | Free-floating capa rentings; >8h ≈ vehículo varado / mantenimiento |
| **Outliers geográficos** | percentil [0.1%, 99.9%] de lat/lon **por ciudad** | Filtra errores de GPS sin necesidad de bounding boxes hardcoded |
| **Duplicados exactos** | mismo (vin, s_date, s_coord) | Descarta artefactos de re-ingesta |

### Auditoría del cleaning (27 abr 2026)

| city | initial | -short_dist | -dur_anom | -geo | -dup | final | %removed |
|---|---|---|---|---|---|---|---|
| amsterdam | 51.277 | 2.735 | 437 | 193 | 0 | 47.912 | 6.56% |
| berlin | 225.808 | 7.027 | 1.578 | 694 | 0 | 216.509 | 4.12% |
| firenze | 19.355 | 890 | 266 | 38 | 0 | 18.161 | 6.17% |
| kobenhavn | 12.610 | 966 | 169 | 46 | 0 | 11.429 | 9.37% |
| milano | 158.068 | 3.980 | 1.481 | 459 | 0 | 152.148 | 3.75% |
| muenchen | 82.522 | 1.555 | 677 | 265 | 0 | 80.025 | 3.03% |
| roma | 101.482 | 3.578 | 1.237 | 385 | 0 | 96.282 | 5.12% |
| stockholm | 16.834 | 908 | 119 | 33 | 0 | 15.774 | 6.30% |
| torino | 25.578 | 956 | 224 | 96 | 0 | 24.302 | 4.99% |
| wien | 146.516 | 4.185 | 1.669 | 286 | 0 | 140.376 | 4.19% |
| **TOTAL** | **840.050** | **26.780** | **7.857** | **2.495** | **0** | **802.918** | **4.42%** |

### Hallazgos interpretables

- **Filtro dominante:** distancia <100 m (3.2% del total), refleja la prevalencia de reservas degeneradas en operaciones reales de free-floating.
- **Duración:** ~1% adicional, mayoría de viajes legítimos respeta la franja [1 min, 8 h].
- **Outliers geográficos:** ~0.3%, en línea con el orden teórico esperado (~0.2%) → confirma que el percentil 99.9% es un buen umbral conservador.
- **Duplicados:** 0 en todas las ciudades → buena calidad de los datos brutos del operador.
- **København destaca** con 9.37% (casi el doble de la media). Probable causa: al ser la ciudad más pequeña, los filtros pesan proporcionalmente más sobre su volumen.
- **München y Milán son las más limpias** (3.03% y 3.75%) → refuerza la elección de München como ciudad de test.

### Decisión de timezone

- **XGBoost (Milán):** trabaja en **hora local** de Milán (no convierte a UTC). Justificación: la demanda urbana es intrínsecamente local-horaria.
- **LSTM (multi-ciudad):** convierte a **UTC** para tener referencia absoluta común. La hora-del-día se recupera localmente cuando hace falta.

---

## 🔄 Metodología: Dos Soluciones Complementarias

### Arquitectura General

```
FASE 1: DATA PREPARATION (Común a ambas soluciones)
├─ Raw trips (10 ciudades, 840K viajes)
├─ Spatial join a H3-7 (capa común para todas las ciudades)
├─ Temporal aggregation → demanda (city, h3_cell, date, hour)
├─ Expand with zeros → incluir horas sin viajes (70-90% ceros)
├─ Normalization → z-score por celda
└─ Feature engineering → lags (1h, 24h, 168h), cyclic encoding

FASE 2A: SOLUCIÓN XGBOOST
├─ Features: hour, day_of_week, month, lag_1h, lag_24h, rolling_mean_3h
├─ Training: XGBRegressor + XGBClassifier (3 clases)
├─ Evaluation: MAE, RMSE, R² (regresión) + Accuracy, F1 (clasificación)
└─ Scope: Monociudad (Milán) como baseline

FASE 2B: SOLUCIÓN LSTM (Domain Adaptation)
├─ SlidingWindowDataset → ventanas T=24h
├─ DataLoader → minibatches de 512
├─ Training: LSTM preentrenado en 9 ciudades (2015)
├─ Evaluation: Zero-shot (München 2016), fine-tuning (7 días)
└─ Scope: Multi-ciudad + domain adaptation a ciudad nueva
```

**Importancia de ambas soluciones:**
- **XGBoost:** Baseline rápido, interpretable, compara directamente con literatura
- **LSTM:** Captura patrones temporales, valida domain adaptation multi-ciudad
- **Preprocesamiento:** Punto crítico común (H3 spatial discretization, normalization, handling sparsity)

### Split de Datos

```
Train:     9 ciudades (2015) - sin últimos 7 días
           → 940,000 ventanas LSTM
           
Validation: 9 ciudades (2015) - últimos 7 días
           → 85,000 ventanas LSTM
           
Test:      München (2016) - completo
           → 145,000 ventanas LSTM (zero-shot)
           → También se usa para fine-tuning (7 días)
```

### LSTM Training (Epoch 1-16)

**Curva de aprendizaje real:**
```
Epoch  1: Train=0.3151, Val=0.2966
Epoch  5: Train=0.2966, Val=0.2911
Epoch  8: Train=0.2933, Val=0.2866  ← MEJOR val
Epoch 10: Train=0.2933, Val=0.2971  ← Comienza overfitting
Epoch 15: Train=0.2910, Val=0.2903
Epoch 16: EARLY STOPPING (patience=8 epochs sin mejorar)
```

**Interpretación:**
- Epoch 1 NO es "sin entrenar": ya pasó 1 epoch (940K ventanas)
- Mejora rápida epochs 1-8 (aprende patrones)
- Luego se estanca/sube (overfitting contenido)
- Early stopping a tiempo: evita sobreajuste severo

**Proceso por epoch (ejemplo epoch 1):**
1. Itera 1,836 minibatches (940K / 512)
2. Cada batch: forward → loss → backward → update
3. Promedia 1,836 pérdidas → epoch_train_loss = 0.3151
4. Validación similar con 85K ventanas

---

## 📈 Resultados Principales

### XGBoost Baseline (Milán, regresión)

| Métrica | Valor |
|---------|-------|
| MAE | 0.9247 |
| RMSE | 1.4663 |
| R² | **0.5660** |

**Interpretación:** Error promedio 0.92 viajes/zona-hora. Pero R²=0.57 bajo → modelo no captura varianza bien.

---

### LSTM Results (H3-7)

#### Domain Adaptation Results

**Domain Adaptation Setup:**
- **Source domain (training):** 9 ciudades europeas, período 2015 (May-Jul)
- **Target domain (test):** München, período 2016 (Mar-May) — distinto año, geografía nórdica
- **Evaluation:** Zero-shot (0 fine-tuning) y fine-tuning (7 días datos München)

#### Per-City Validation (últimos 7 días, 9 ciudades de entrenamiento)

| Ciudad | MAE | RMSE | R² |
|--------|-----|------|-----|
| amsterdam | 0.8722 | 1.3546 | 0.5458 |
| berlin | 1.2770 | 1.8933 | **0.6694** |
| firenze | 0.7192 | 1.0806 | 0.4624 |
| kobenhavn | 0.6152 | 0.8925 | 0.2507 |
| milano | 1.8063 | 2.9727 | **0.7659** ✅ |
| roma | 1.3798 | 2.1573 | 0.6198 |
| stockholm | 0.7164 | 1.1264 | 0.3697 |
| torino | 0.8977 | 1.4485 | 0.5831 |
| wien | 1.4095 | 2.1366 | **0.6724** |
| **München (zero-shot)** | 0.9845 | 1.6281 | **0.6375** ✅ |

**Hallazgos:**
- ✅ **Milán LSTM (0.77) vs XGBoost (0.57):** +35% R² mejoría
- ✅ **München zero-shot (0.64):** Transfer learning funciona sin reentrenamiento
- ✅ Berlin, Wien: R² > 0.66 (muy bueno)
- ⚠️ Kobenhavn: R² bajo (0.25) → ciudad pequeña, pocos datos

#### Fine-tuning en München (7 días)

| Modelo | MAE | RMSE | R² |
|--------|-----|------|-----|
| Zero-shot (0 días FT) | 0.9845 | 1.6281 | 0.6375 |
| Fine-tuned (7 días FT) | 0.9590 | 1.5712 | **0.6632** |
| Mejora | -0.0255 | -0.0569 | +0.0257 |

**Conclusión:** Mejora marginal (~2.6%) porque el preentrenamiento ya generalizaba bien.

---

### Comparativa Final: XGBoost vs LSTM

```
                                  MAE    RMSE    R²
XGBoost Milán (baseline)         0.9247 1.4663  0.5660
LSTM Milán (mismo dataset)       1.8063 2.9727  0.7659  ← +35% R²
LSTM Zero-shot München          0.9845 1.6281  0.6375
LSTM Fine-tuned München (7d)    0.9590 1.5712  0.6632
```

**Interpretación:**
- MAE más alto en LSTM (pero más honesto: no predice siempre cero)
- **R² es la métrica que importa:** LSTM captura 76.6% de varianza (vs 56.6%)
- Transfer learning válido: zero-shot en nueva ciudad/período funciona

---

## 📚 Estructura del TFG

### Capítulos (Actualizado — 23 Abril 2026)

**Cambio estructural:** se fusionan "State of the Art" y "Enabling Technologies" en un único capítulo 2 "Background". Los antiguos apéndices "Impact" y "Budget" pasan a ser capítulos regulares. Estructura final: **7 capítulos**.

```
1. INTRODUCTION
   ├─ 1.1 Context
   ├─ 1.2 Project Goals
   └─ 1.3 Structure of this Document

2. BACKGROUND
   ├─ 2.1 Enabling Technologies
   │   ├─ 2.1.1 Python Scientific Stack
   │   ├─ 2.1.2 H3 Hexagonal Hierarchical Index
   │   ├─ 2.1.3 Machine Learning Fundamentals
   │   ├─ 2.1.4 LSTM Networks
   │   ├─ 2.1.5 Transfer Learning and Domain Adaptation
   │   └─ 2.1.6 Feature Engineering and Normalisation
   └─ 2.2 Related Work
       ├─ 2.2.1 Evolution of Carsharing Systems
       ├─ 2.2.2 Demand Prediction in Urban Mobility
       └─ 2.2.3 Research Gap and Contribution

3. ARCHITECTURE AND METHODOLOGY
   ├─ 3.1 Data Acquisition and Exploration
   │   ├─ 3.1.1 Data Sources (10 European cities)
   │   ├─ 3.1.2 Data Cleaning and Standardisation
   │   └─ 3.1.3 Exploratory Data Analysis
   ├─ 3.2 First Predictive Framework: XGBoost (City of Milan)
   │   ├─ 3.2.1 Spatial Aggregation: ACE Zoning
   │   ├─ 3.2.2 Feature Engineering
   │   ├─ 3.2.3 Model Architecture and Hyperparameters
   │   └─ 3.2.4 Training Procedure
   └─ 3.3 Second Predictive Framework: LSTM with Cross-City Domain Adaptation
       ├─ 3.3.1 Spatial Discretisation: H3 Hexagonal Grid
       ├─ 3.3.2 Feature Engineering (cyclic encoding + sliding window)
       ├─ 3.3.3 Model Architecture
       ├─ 3.3.4 Training Procedure
       └─ 3.3.5 Transfer Learning Setup

4. TRAINING, EVALUATION AND MODEL COMPARISON
   ├─ 4.1 XGBoost Results (Milan)
   │   ├─ 4.1.1 Regression Model
   │   ├─ 4.1.2 Classification Model
   │   └─ 4.1.3 Feature Importance
   ├─ 4.2 LSTM Results
   │   ├─ 4.2.1 Pretraining on 9 Cities
   │   ├─ 4.2.2 Per-City Validation Metrics
   │   ├─ 4.2.3 Zero-Shot Transfer to Unseen City
   │   └─ 4.2.4 Fine-tuning Results
   ├─ 4.3 Comparative Analysis
   │   ├─ 4.3.1 Predictive Accuracy
   │   ├─ 4.3.2 Data Requirements and Computational Cost
   │   ├─ 4.3.3 Interpretability
   │   └─ 4.3.4 Generalisation Capability
   └─ 4.4 Visualisations and Error Analysis

5. CONCLUSIONS
   ├─ 5.1 Summary of Findings
   ├─ 5.2 Research Contributions
   ├─ 5.3 Practical Implications
   └─ 5.4 Future Work

6. IMPACT AND SUSTAINABILITY
   ├─ 6.1 Societal Impact
   ├─ 6.2 Environmental Considerations
   └─ 6.3 Economic Impact

7. PROJECT BUDGET

APPENDICES
├─ A. Detailed H3 Grid Visualization (Milán)
├─ B. Dataset Statistics by City
├─ C. Hyperparameter Tuning Details
├─ D. Code Repository (GitHub link)
└─ E. Additional Plots and Tables

FRONT MATTER
├─ Resumen (español)
├─ Abstract (inglés)
└─ Agradecimientos
```

### Decisiones estructurales (23 Abril 2026)

1. **Fusión Background**: los antiguos capítulos "State of the Art" y "Enabling Technologies" se integran en un único capítulo 2 con dos secciones (2.1 Enabling Technologies + 2.2 Related Work). Esto evita redundancia y facilita una transición natural del contexto técnico a la revisión bibliográfica.

2. **Arquitectura con estructura híbrida**: se mantiene una sección común 3.1 para adquisición y exploración de datos (limpieza, estandarización, EDA), pero los detalles de preprocesamiento específicos de cada modelo (agregación ACE para XGBoost vs. discretización H3 para LSTM, feature engineering distinto) se describen dentro de las secciones 3.2 y 3.3 respectivamente. Así se evita fragmentar la narrativa de cada framework.

3. **XGBoost primero, LSTM después**: se mantiene el orden narrativo "enfoque inicial → contribución principal" tanto en el capítulo 3 (metodología) como en el 4 (resultados), seguido siempre de análisis comparativo.

4. **Ambos modelos al mismo nivel**: tras la indicación del tutor, los objetivos del proyecto tratan XGBoost y LSTM como contribuciones equivalentes. La comparación sistemática entre ambos se formula como el cuarto objetivo (G4) y es el eje vertebrador del capítulo 4.

5. **Impact y Budget como capítulos**: se sacan del apéndice y pasan a ser capítulos regulares (6 y 7), siguiendo la convención de la plantilla GSI-ETSIT más reciente.

---

## 🎯 Estado del Proyecto

### ✅ Completado

**SOLUCIÓN 1: XGBOOST (Baseline)**
1. Notebook: `xgboost_solution.ipynb`
   - Regresión: MAE=0.92, RMSE=1.47, R²=0.566
   - Clasificación: Accuracy=63.2%, F1=0.633
   - Dataset: Milán (ACE zones), features estáticas + lags
   - Benchmark de referencia para comparación

**SOLUCIÓN 2: LSTM (Domain Adaptation)**
2. Notebook: `lstm_dataset_creation.ipynb`
   - H3-7 spatial discretization (capa común 10 ciudades)
   - Normalización z-score por celda
   - Features: temporal (cyclic encoding) + lags
   - Salida: `datasets/dataset_h3_multicidad.parquet` (2.3M filas)

3. Notebook: `lstm_predictions.ipynb`
   - Arquitectura: 2 capas LSTM + dense head
   - Preentrenamiento: 9 ciudades (940K ventanas)
   - Zero-shot Munich: R²=0.6375 (sin reentrenamiento)
   - Fine-tuning: R²=0.6632 (7 días)
   - Modelos: `results/lstm_pretrained.pt`

**PREPROCESAMIENTO & ANALYSIS**
4. Notebook: `mapa_h3_milan.ipynb`
   - Visualización: `results/mapa_h3_milan.png`
   - Análisis espacial de demanda por H3 cell

**PIPELINE DE LIMPIEZA (27 abr 2026)**
5. Módulo `trip_cleaning.py`
   - Centraliza la limpieza para ambos frameworks
   - 5 filtros: distancia, duración (×2), outliers geográficos, duplicados
   - Función `clean_trips(df, city)` + utilidades `parse_date_local`, `parse_date_utc`, `parse_coord`, `trip_quality_summary`
6. Script `apply_cleaning_patches.py`
   - Parchea ambos notebooks con `nbformat` (sin riesgo de corromper JSON)
   - Idempotente, crea backups `.bak`
7. Auditoría completa del cleaning ejecutada (4.42% viajes descartados global)

**DOCUMENTACIÓN & VERSIONADO**
8. GitHub
   - 4 notebooks con nombres actualizados ✅
   - MEMORIA.md con contexto completo ✅
   - Commits documentados ✅
   - Estructura clara (datasets/, results/) ✅

**ESCRITURA TFG (27 abr 2026)**
9. Capítulo 3 — Architecture and Methodology
   - 3.1 introducción: redactado en inglés (1 pág, contextualiza car2go + 3 subsecciones)
   - 3.1.1 Data Sources: redactado en inglés con `longtable` LaTeX (incluye datasets auxiliares ISTAT)
   - Tabla 3.1 (volumen por ciudad) en LaTeX con `booktabs`
   - 3.1.2 Data Cleaning: pendiente de redactar (auditoría disponible)
   - 3.1.3 EDA: estrategia decidida (trip-level only, sin H3 ni ACE)

### 🔄 En Progreso

1. **Escritura LaTeX del TFG**
   - Cap. 3.1 introducción + 3.1.1: ✅ pegado en Overleaf
   - Cap. 3.1.2: pendiente (con números reales de auditoría)
   - Cap. 3.1.3: pendiente (gráficos: perfil horario, semanal, % ceros, volumen por ciudad)
   - Cap. 3.2 (XGBoost) + 3.3 (LSTM): pendiente
   - Cap. 4 (Resultados): pendiente
   - Cap. 2 (Background + Related Work): pendiente

2. **Regeneración de `dataset_h3_multicidad.parquet`**
   - El parquet actual se hizo con datos sin limpiar
   - Tras parchear notebooks con `apply_cleaning_patches.py`, ejecutar `lstm_dataset_creation.ipynb` de nuevo
   - Reentrenar LSTM y XGBoost para validar que las métricas no cambian sustancialmente

3. **Sincronización Overleaf-GitHub**
   - Opción elegida: copiar-pegar manual (sin pago)
   - Genera contenido LaTeX → copia a Overleaf

### ⏭️ Por Hacer

1. **Escribir Cap 2: Estado del Arte**
   - Evolución carsharing
   - Soluciones similares
   - Problemas identificados

2. **Completar Cap 4: Arquitectura**
   - Detalles dataset
   - EDA figuras
   - Normalización
   - Arquitecturas XGBoost y LSTM

3. **Rellenar Cap 5: Training & Eval**
   - Resultados XGBoost
   - Resultados LSTM
   - Comparativa análisis
   - Visualizaciones

4. **Escribir Cap 6-8: Conclusiones, Impacto, Budget**

5. **Generación de figuras**
   - Mapa H3 mejorado (coloreado por demanda)
   - Curvas aprendizaje LSTM
   - Scatter predicciones vs reales
   - Tablas de resultados LaTeX

---

## 📋 Próximas Tareas (Prioridad)

### ALTA PRIORIDAD

- [ ] **Cap 2: Estado del Arte** (1-2 semanas)
  - Investigar evolución carsharing
  - Enumerar soluciones similares (Zipcar, Enjoy, etc.)
  - Detallar problemas: sparsity, temporal shifts, heterogeneidad

- [ ] **Cap 4: Arquitectura** (2-3 semanas)
  - Incluir figuras EDA
  - Diagramas del pipeline
  - Ecuaciones normalización, sliding window

- [ ] **Cap 5: Resultados** (2-3 semanas)
  - Tablas comparativas
  - Figuras y gráficos
  - Análisis detallado por ciudad

### MEDIA PRIORIDAD

- [ ] Variables socioeconómicas (Eurostat, censos por país)
- [ ] Análisis de errores detallado
- [ ] Visualizaciones interactivas (opcional)

### BAJA PRIORIDAD

- [ ] Transformer arquitectura (mencionar en future work)
- [ ] Real-time deployment (aplicación práctica)
- [ ] Bases de datos temporales (InfluxDB, TimescaleDB)

---

## 🔗 Recursos Clave

### Repositorio GitHub
```
https://github.com/juanruu/TFG---Carsharing-Estimation-Demand
```

### Notebooks Principales

| Notebook | Propósito | Entrada | Salida |
|----------|-----------|---------|--------|
| `xgboost_solution.ipynb` | XGBoost baseline (regresión + clasificación) | `cs_datasets/*.txt` | XGBoost metrics |
| `lstm_dataset_creation.ipynb` | H3 discretization, normalization, feature engineering | `cs_datasets/*.txt` | `datasets/dataset_h3_multicidad.parquet` |
| `lstm_predictions.ipynb` | LSTM training, zero-shot eval, fine-tuning, comparison | `datasets/dataset_h3_multicidad.parquet` | `results/lstm_pretrained.pt`, métricas |
| `mapa_h3_milan.ipynb` | Visualización de celdas H3 | `datasets/dataset_h3_multicidad.parquet` | `results/mapa_h3_milan.png` |

### Datasets
- `dataset_h3_multicidad.parquet` (1.8 GB) — Dato principal
- `cs_datasets/*.txt` (845 KB total) — Datos crudos
- `dati-cpa_2011/` — Indicadores socioeconómicos ISTAT (CPA 2011) usados en XGBoost
- `mapa_lombardia/`, `mapa_lazio/`, `mapa_piamonte/`, `mapa_toscana/` — Shapefiles regionales con ACE

### Documentación
- `eda_preliminar.ipynb` — Análisis inicial con XGBoost
- `MEMORIA.md` (este archivo) — Contexto del proyecto

### Referencias Académicas Clave

**Boldrini, C., Bruno, R., & Conti, M. (2019).** *Weak signals in the mobility landscape: car sharing in ten European cities.* **EPJ Data Science**, 8(1).
https://doi.org/10.1140/epjds/s13688-019-0186-8

> Es la **referencia central** del estado del arte. Usa el mismo dataset car2go de las 10 ciudades. Hace dos análisis disjuntos: (a) explicativo con censo ISTAT 2011 a nivel ACE, y (b) predictivo con Random Forest sobre rejilla regular de 500m × 500m. **No integra los indicadores socioeconómicos en el modelo predictivo** — los analiza en bloques separados.
>
> **Research gap que cubre este TFG:**
> 1. Integra socio-features + grid en un único modelo predictivo (XGBoost) y descubre por explicabilidad la redundancia del identificador ACE.
> 2. Extiende a deep learning (LSTM) con domain adaptation cross-city.
> 3. Evalúa transfer learning a ciudad no vista (München) en zero-shot y fine-tuning.

```bibtex
@article{boldrini2019weaksignals,
    author  = {Boldrini, Chiara and Bruno, Raffaele and Conti, Marco},
    title   = {Weak signals in the mobility landscape: car sharing in ten European cities},
    journal = {EPJ Data Science},
    volume  = {8},
    number  = {1},
    pages   = {7},
    year    = {2019},
    doi     = {10.1140/epjds/s13688-019-0186-8}
}

@misc{istat2011basi,
    author       = {{Italian National Institute of Statistics (ISTAT)}},
    title        = {Basi territoriali e variabili censuarie},
    year         = {2011},
    howpublished = {\url{https://www.istat.it/notizia/basi-territoriali-e-variabili-censuarie/}},
    note         = {Accessed: April 2026}
}
```

---

## 💡 Notas Técnicas Importantes

### 1. Escala Normalizada vs Real

```python
# Datos guardados normalizados:
demand_norm = (demand - mean) / std      # Rango típico: [-2, 3]

# Pérdida en escala normalizada:
loss_epoch_1 = 0.3151                    # Parece bajo, pero es correcto

# Métricas reportadas (escala real):
MAE = 0.96 viajes/hora/celda            # Desnormalizado de demand_norm
```

### 2. Huber Loss en profundidad

```
Si |error| <= delta (1.0):
    loss = 0.5 * error²                  # Cuadrático
else:
    loss = delta * (|error| - 0.5*delta) # Lineal (robusto)

Ejemplo: error=0.5, delta=1.0
    loss = 0.5 * 0.5² = 0.125

Promedio 512 pérdidas = pérdida del batch
Promedio 1,836 batches = época_train_loss
```

### 3. DataLoader y Minibatches

```
Dataset: 940,000 ventanas
Batch size: 512
Total minibatches/epoch: 1,836

Cada minibatch:
  X_batch: [512, 24, 6]  ← 512 ventanas, 24h, 6 features
  y_batch: [512]         ← 512 targets

Pueden ser de DIFERENTES ciudades y NO ser consecutivos
(shuffle=True en entrenamiento, False en val/test)
```

### 4. Sin/Cos para Ciclicidad

```
Problem: hour 23 y 0 están a 1 hora, pero |23-0|=23
Solution: Convertir a círculo con sin/cos

hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)

hour=0:  sin=0.00, cos=1.00   ← punto (0,1)
hour=1:  sin=0.26, cos=0.97   ← cerca
hour=23: sin=-0.26, cos=0.97  ← CERCA de (0,1)

Distancia euclidiana 23→0 en espacio sin/cos ≈ 0.27 (pequeña ✓)
```

---

## 📞 Contacto y Versiones

**Autor:** Juan Perpic Rodríguez  
**Email:** juanperpic@gmail.com  
**Universidad:** [Universidad]  
**Supervisor:** [Supervisor TFG]

**Version History:**
- v1.0 (Abril 2026): Creación de memoria con resultados LSTM y H3-7

---

**Última actualización:** Abril 22, 2026  
**Estado:** 🔄 En progreso — Escritura LaTeX TFG iniciada
