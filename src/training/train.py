import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score


EXPERIMENT_NAME = "wholesale-customers-clustering"
REGISTERED_MODEL_NAME = "WholesaleCustomersKMeans"


def train_model(data_path, model_output_path):

    print(f"Cargando datos procesados desde: {data_path}")
    df = pd.read_csv(data_path)

    # ---------------------------------------------------------
    # 1. ENTRENAMIENTO
    # ---------------------------------------------------------

    k = 3
    random_state = 42
    n_init = 10

    print(f"Entrenando modelo K-Means con k={k}...")

    model = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=n_init
    )

    labels = model.fit_predict(df)

    # ---------------------------------------------------------
    # 2. EVALUACIÓN
    # ---------------------------------------------------------

    sil_score = silhouette_score(df, labels)
    db_score = davies_bouldin_score(df, labels)

    print("\n--- Resultados de Evaluación ---")
    print(f"Silhouette Score: {sil_score:.3f}")
    print(f"Davies-Bouldin Score: {db_score:.3f}")

    # ---------------------------------------------------------
    # 3. GUARDAR MODELO LOCAL
    # ---------------------------------------------------------

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

    joblib.dump(model, model_output_path)

    print(f"\nModelo guardado en: {model_output_path}")

    # ---------------------------------------------------------
    # 4. MLFLOW TRACKING
    # ---------------------------------------------------------

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"kmeans-k{k}"):

        mlflow.log_params({
            "model": "KMeans",
            "n_clusters": k,
            "random_state": random_state,
            "n_init": n_init
        })

        mlflow.log_metrics({
            "silhouette_score": sil_score,
            "davies_bouldin_score": db_score
        })

        mlflow.log_artifact(
            model_output_path,
            artifact_path="model"
        )

        mlflow.sklearn.log_model(
            model,
            name="kmeans_model"
        )

        run_id = mlflow.active_run().info.run_id

        print("\n--- MLflow Tracking ---")
        print(f"Experimento: {EXPERIMENT_NAME}")
        print(f"Run ID: {run_id}")
        print("Parámetros registrados.")
        print("Métricas registradas.")
        print("Modelo registrado como artefacto.")

    # ---------------------------------------------------------
    # 5. BUSCAR BEST CANDIDATE
    # ---------------------------------------------------------

    print("\n--- Selección del Best Candidate ---")

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id]
    )

    if runs.empty:
        print("No se encontraron ejecuciones.")
        return

    # Ordenamos:
    # 1. Silhouette más alto = mejor
    # 2. Davies-Bouldin más bajo = mejor
    best_run = runs.sort_values(
        by=[
            "metrics.silhouette_score",
            "metrics.davies_bouldin_score"
        ],
        ascending=[
            False,
            True
        ]
    ).iloc[0]

    best_run_id = best_run["run_id"]
    best_silhouette = best_run["metrics.silhouette_score"]
    best_db = best_run["metrics.davies_bouldin_score"]

    print(f"Best Candidate Run ID: {best_run_id}")
    print(f"Best Silhouette Score: {best_silhouette:.6f}")
    print(f"Best Davies-Bouldin Score: {best_db:.6f}")

    # ---------------------------------------------------------
    # 6. MODEL REGISTRY
    # ---------------------------------------------------------

    print("\n--- Model Registry ---")

    best_model_uri = f"runs:/{best_run_id}/kmeans_model"

    try:

        model_version = mlflow.register_model(
            model_uri=best_model_uri,
            name=REGISTERED_MODEL_NAME
        )

        print(f"Modelo registrado en Model Registry.")
        print(f"Nombre: {REGISTERED_MODEL_NAME}")
        print(f"Versión: {model_version.version}")

    except Exception as e:

        print("\nEl modelo ya puede estar registrado.")
        print(f"Detalle: {e}")


if __name__ == "__main__":

    INPUT_DATA = "data/processed/wholesale_customers_scaled.csv"
    MODEL_OUTPUT = "models/baseline_kmeans.pkl"

    train_model(
        INPUT_DATA,
        MODEL_OUTPUT
    )