import pandas as pd
import os
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

def train_model(data_path, model_output_path):
    print(f"Cargando datos procesados desde: {data_path}")
    df = pd.read_csv(data_path)

    # 1. CLUSTERING (Entrenamiento del Baseline)
    # Probaremos con k=3 clústeres como punto de partida
    k = 3
    print(f" Entrenando modelo K-Means con k={k}...")
    
    # random_state=42 asegura la reproducibilidad
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(df)

    # 2. MÉTRICAS
    sil_score = silhouette_score(df, labels)
    db_score = davies_bouldin_score(df, labels)

    print("\n--- Resultados de Evaluación (Métricas) ---")
    print(f"Silhouette Score: {sil_score:.3f} (Idealmente más cerca a 1)")
    print(f"Davies-Bouldin Score: {db_score:.3f} (Idealmente más cerca a 0)")

    # 3. Guardar el artefacto (Modelo)
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    
    print(f"\nModelo entrenado y guardado EXITOSAMENTE en: {model_output_path}")

if __name__ == "__main__":
    # Rutas basadas en la estructura del proyecto
    INPUT_DATA = "data/processed/wholesale_customers_scaled.csv"
    MODEL_OUTPUT = "models/baseline_kmeans.pkl"
    
    train_model(INPUT_DATA, MODEL_OUTPUT)