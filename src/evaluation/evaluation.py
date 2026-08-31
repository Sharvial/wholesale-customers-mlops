import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.metrics import silhouette_score, davies_bouldin_score


EXPERIMENT_NAME = "wholesale-customers-clustering"


def evaluate_model(
    model_uri,
    data_path
):
    """
    Evalúa un modelo K-Means ya entrenado utilizando
    las métricas apropiadas para clustering.
    """

    print(f"Cargando datos desde: {data_path}")
    df = pd.read_csv(data_path)

    print(f"Cargando modelo desde: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)

    # Generar predicciones
    labels = model.predict(df)

    # Calcular métricas de desempeño
    sil_score = silhouette_score(df, labels)
    db_score = davies_bouldin_score(df, labels)

    print("\n--- Model Performance ---")
    print(f"Silhouette Score: {sil_score:.6f}")
    print(f"Davies-Bouldin Score: {db_score:.6f}")

    # Registrar resultados en MLflow
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(
        run_name="model-performance-evaluation"
    ):

        # Parámetros
        mlflow.log_param(
            "evaluation_type",
            "clustering"
        )

        mlflow.log_param(
            "model_type",
            "KMeans"
        )

        mlflow.log_param(
            "model_uri",
            model_uri
        )

        # Métricas de desempeño
        mlflow.log_metric(
            "silhouette_score",
            sil_score
        )

        mlflow.log_metric(
            "davies_bouldin_score",
            db_score
        )

        # Tags
        mlflow.set_tag(
            "evaluation_type",
            "model_performance"
        )

        mlflow.set_tag(
            "model_evaluation_status",
            "completed"
        )

        run_id = mlflow.active_run().info.run_id

        print("\n--- MLflow Evaluation ---")
        print(f"Run ID: {run_id}")
        print("Métricas de desempeño registradas.")
        print("Evaluación completada correctamente.")


if __name__ == "__main__":

    MODEL_URI = (
    "models:/WholesaleCustomersKMeans/1"
)

    # Datos utilizados para la evaluación
    DATA_PATH = (
        "data/processed/wholesale_customers_scaled.csv"
    )

    evaluate_model(
        MODEL_URI,
        DATA_PATH
    )