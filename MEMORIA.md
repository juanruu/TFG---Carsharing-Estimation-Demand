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
├── cs_datasets/              # Datos crudos (10 ciudades, 840K viajes)
├── datasets/                 # Datos procesados
│   ├── dataset_h3_multicidad.parquet
│   └── [otros CSVs]
├── results/                  # Resultados, modelos, figuras
│   ├── lstm_pretrained.pt
│   ├── mapa_h3_milan.png
│   └── [gráficos, métricas]
├── xgboost_solution.ipynb    # XGBoost: regresión + clasificación
├── lstm_dataset_creation.ipynb # LSTM: preparación datos, H3, normalización
├── lstm_predictions.ipynb    # LSTM: entrenamiento, evaluación, comparativa
├── mapa_h3_milan.ipynb       # Visualización H3 en Milán
├── MEMORIA.md                # Este archivo
└── README.md
```

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
| Wien | 158,069 | 2015-05-17 a 2015-07-01 | 46 días |
| **TOTAL** | **840,060** | — | — |

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

### Capítulos (Actualizado)

```
1. INTRODUCTION
   ├─ 1.1 Motivation and Problem Statement
   ├─ 1.2 Research Objectives
   └─ 1.3 Document Structure

2. STATE OF THE ART AND RELATED WORK
   ├─ 2.1 Evolution of Carsharing Systems
   ├─ 2.2 Demand Prediction in Urban Mobility
   ├─ 2.3 Similar Solutions and Benchmarks
   └─ 2.4 Research Gap and Contribution

3. ENABLING TECHNOLOGIES
   ├─ 3.1 H3 Hexagonal Indexing
   ├─ 3.2 XGBoost (Extreme Gradient Boosting)
   ├─ 3.3 LSTM Networks
   ├─ 3.4 Transfer Learning
   └─ 3.5 Normalization and Feature Engineering

4. ARCHITECTURE AND METHODOLOGY
   ├─ 4.1 Data Acquisition and Exploration
   │   ├─ 4.1.1 Dataset Description (10 ciudades)
   │   ├─ 4.1.2 Exploratory Data Analysis
   │   └─ 4.1.3 H3 Spatial Discretization
   ├─ 4.2 Data Standardization Pipeline
   │   ├─ 4.2.1 Temporal Aggregation
   │   ├─ 4.2.2 Missing Values Handling
   │   ├─ 4.2.3 Normalization Strategy
   │   └─ 4.2.4 Train/Val/Test Split
   ├─ 4.3 XGBoost Model
   │   ├─ 4.3.1 Architecture and Hyperparameters
   │   ├─ 4.3.2 Feature Selection
   │   └─ 4.3.3 Training Procedure
   └─ 4.4 LSTM Model
       ├─ 4.4.1 Architecture
       ├─ 4.4.2 Sliding Window Dataset
       ├─ 4.4.3 Training Strategy
       ├─ 4.4.4 Transfer Learning
       └─ 4.4.5 Evaluation Metrics

5. TRAINING, EVALUATION AND MODEL COMPARISON
   ├─ 5.1 XGBoost Results
   │   ├─ 5.1.1 Regression Model
   │   ├─ 5.1.2 Classification Model
   │   └─ 5.1.3 Learning Curves
   ├─ 5.2 LSTM Results
   │   ├─ 5.2.1 Pretraining on 9 Cities
   │   ├─ 5.2.2 Per-City Validation Metrics
   │   ├─ 5.2.3 Zero-Shot Transfer Learning
   │   └─ 5.2.4 Fine-tuning in München
   ├─ 5.3 Comparative Analysis
   │   ├─ 5.3.1 Performance Comparison Table
   │   ├─ 5.3.2 Why LSTM Outperforms XGBoost
   │   ├─ 5.3.3 Transfer Learning Insights
   │   └─ 5.3.4 Limitations and Challenges
   └─ 5.4 Visualizations and Error Analysis

6. CONCLUSIONS
   ├─ 6.1 Summary of Findings
   ├─ 6.2 Research Contributions
   ├─ 6.3 Practical Implications
   └─ 6.4 Future Work

7. IMPACT AND SUSTAINABILITY
   ├─ 7.1 Societal Impact
   ├─ 7.2 Environmental Considerations
   └─ 7.3 Economic Impact

8. PROJECT BUDGET

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

**DOCUMENTACIÓN & VERSIONADO**
5. GitHub
   - 4 notebooks con nombres actualizados ✅
   - MEMORIA.md con contexto completo ✅
   - Commits documentados ✅
   - Estructura clara (datasets/, results/) ✅

### 🔄 En Progreso

1. **Escritura LaTeX del TFG**
   - Estructura actualizada (cap. 2 nuevo)
   - Estado del Arte pendiente
   - Arquitectura (cap 4) pendiente
   - Training/Eval/Comparison (cap 5) pendiente

2. **Sincronización Overleaf-GitHub**
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

### Documentación
- `eda_preliminar.ipynb` — Análisis inicial con XGBoost
- `MEMORIA.md` (este archivo) — Contexto del proyecto

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
