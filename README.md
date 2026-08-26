## 1. Business Problem & 2. Dataset

El objetivo principal de este sistema es descubrir estructuras y perfiles ocultos de clientes de un distribuidor mayorista basándose exclusivamente en su comportamiento de compra anual. Para cumplir con la restricción de aprendizaje puramente no supervisado, el modelo segmenta a los clientes sin utilizar categorías previas conocidas.

El dataset original contiene el gasto anual (en unidades monetarias) en seis categorías de productos. Las variables categóricas de clasificación (`Channel` y `Region`) fueron descartadas intencionalmente para evitar sesgos en el cálculo de distancias del algoritmo. Las características utilizadas son:
* **Fresh:** Gasto anual en productos frescos.
* **Milk:** Gasto anual en productos lácteos.
* **Grocery:** Gasto anual en abarrotes.
* **Frozen:** Gasto anual en productos congelados.
* **Detergents_Paper:** Gasto anual en detergentes y papel.
* **Delicassen:** Gasto anual en productos delicatessen.

## 6. Data Ingestion & Quality Gates

La ingesta de datos está automatizada y protegida por un script de validación (`data_quality.py`) que actúa como barrera de entrada al sistema. Antes de cualquier procesamiento, los datos deben superar compuertas de calidad estrictas:
* Validación de estructura y tipos de datos numéricos.
* Detección de valores nulos o faltantes.
* Confirmación de ausencia de registros duplicados.
* Verificación de integridad matemática (ausencia de gastos con valores negativos).

## 7. Training & Feature Engineering

El modelo base (Baseline) utiliza el algoritmo K-Means. Para garantizar que la lógica de experimentación sea idéntica a la de producción, las transformaciones matemáticas se encapsularon en un Pipeline reutilizable de Scikit-Learn con los siguientes pasos:
* **Log1p Transformation:** Se aplica para mitigar la asimetría positiva extrema (skewness) de los gastos, soportando de forma segura la existencia de ceros reales en compras no realizadas.
* **RobustScaler:** Se utiliza para escalar las variables utilizando la mediana y el rango intercuartílico, limitando el efecto gravitacional de los clientes "ballena" sin necesidad de eliminar estos valiosos registros.
* **Configuración del Modelo:** Se seleccionó k=3 clústeres como la configuración óptima tras evaluar el Método del Codo (Inercia) y el coeficiente Silhouette.

## 12. Results (Interpretación de Clústeres)

El algoritmo descubrió de forma no supervisada tres perfiles distintos de clientes mayoristas:
* **Clúster 0 (Minimarkets y Tiendas de Conveniencia):** Clientes con un gasto fuertemente concentrado en abarrotes, leche y detergentes.
* **Clúster 1 (Restaurantes y Cafeterías):** Clientes cuyo gasto está dominado casi en su totalidad por productos frescos y de consumo diario.
* **Clúster 2 (Compradores Pequeños o Diversificados):** Clientes que mantienen un gasto moderado y balanceado de manera uniforme en todas las categorías de productos.