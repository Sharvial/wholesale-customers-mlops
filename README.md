# wholesale-customers-mlops
Proyecto de MLOps para la segmentación de clientes utilizando el conjunto de datos Wholesale Customers.
# Wholesale Customers MLOps

## Componentes del proyecto sergio

### 1. Training

Se implementó el entrenamiento de un modelo K-Means para realizar agrupamiento de clientes mayoristas.

Configuración utilizada:

- Modelo: K-Means
- Número de clusters: 3
- Random state: 42
- N_init: 10

El entrenamiento registra en MLflow los parámetros, métricas y artefactos generados.

### 2. Evaluation

Se implementó una evaluación independiente del modelo entrenado.

Debido a que se trata de un modelo de clustering no supervisado, se utilizan:

- Silhouette Score: mayor valor indica una mejor separación de los clusters.
- Davies-Bouldin Score: menor valor indica una mejor separación de los clusters.

Las métricas de evaluación son registradas en MLflow.

### 3. Best Candidate

Se implementó la selección del mejor candidato utilizando:

1. Mayor Silhouette Score.
2. Menor Davies-Bouldin Score como criterio secundario.

### 4. Model Registry

El mejor modelo seleccionado se registra en MLflow Model Registry.

Modelo registrado:

- Nombre: `WholesaleCustomersKMeans`
- Versión: `1`

### 5. Monitoring

Se implementó monitoreo de diferentes aspectos del modelo y sus datos.

#### Data Drift

Se comparan los datos de referencia con los datos actuales utilizando la prueba estadística Kolmogorov-Smirnov.

Configuración:

- Método: Kolmogorov-Smirnov
- Threshold: 0.05

El sistema registra en MLflow:

- Estadístico KS.
- P-value.
- Drift por variable.
- Cantidad de variables con drift.
- Drift general.

#### Model Performance

Se evalúa el desempeño del modelo utilizando:

- Silhouette Score.
- Davies-Bouldin Score.

Los resultados son registrados en MLflow.

#### System Metrics

Se implementó monitoreo de recursos del sistema mediante `psutil`.

Se registran:

- Uso de CPU.
- Uso de memoria RAM.
- Memoria disponible.
- Uso de disco.
- Tiempo de ejecución.

### 6. Retrain Trigger

Se implementó un mecanismo de activación de reentrenamiento basado en la detección de Data Drift.

Cuando se detecta drift:

```text
Data Drift detectado
        ↓
Retrain Required = 1
        ↓
Retrain Trigger = TRUE
        ↓
Reason = data_drift