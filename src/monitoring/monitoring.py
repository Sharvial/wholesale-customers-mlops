import pandas as pd
import mlflow
from scipy.stats import ks_2samp


EXPERIMENT_NAME = "wholesale-customers-clustering"


def calculate_data_drift(
    reference_path,
    current_path,
    threshold=0.05
):
    """
    Compara los datos de referencia contra los datos actuales
    utilizando la prueba Kolmogorov-Smirnov.
    """

    print("Cargando datos de referencia...")
    reference_df = pd.read_csv(reference_path)

    print("Cargando datos actuales...")
    current_df = pd.read_csv(current_path)

    # Verificar que ambos conjuntos tengan las mismas columnas
    if list(reference_df.columns) != list(current_df.columns):
        raise ValueError(
            "Los datos de referencia y actuales no tienen "
            "las mismas columnas."
        )

    results = []

    for column in reference_df.columns:

        reference_values = reference_df[column].dropna()
        current_values = current_df[column].dropna()

        statistic, p_value = ks_2samp(
            reference_values,
            current_values
        )

        drift_detected = p_value < threshold

        results.append({
            "feature": column,
            "ks_statistic": statistic,
            "p_value": p_value,
            "drift_detected": drift_detected
        })

    results_df = pd.DataFrame(results)

    overall_drift = results_df["drift_detected"].any()

    print("\n--- Data Drift ---")

    for _, row in results_df.iterrows():

        print(
            f"{row['feature']}: "
            f"KS={row['ks_statistic']:.4f}, "
            f"p-value={row['p_value']:.4f}, "
            f"Drift={row['drift_detected']}"
        )

    print(
        f"\nDrift general detectado: {overall_drift}"
    )

    return results_df, overall_drift


def monitor_data_drift(
    reference_path,
    current_path,
    threshold=0.05
):
    """
    Ejecuta Data Drift, registra los resultados en MLflow
    y determina si se requiere reentrenamiento.
    """

    mlflow.set_experiment(EXPERIMENT_NAME)

    results_df, overall_drift = calculate_data_drift(
        reference_path,
        current_path,
        threshold
    )

    # Crear un Run específico para Monitoring
    with mlflow.start_run(run_name="data-drift-monitoring"):

        # ---------------------------------------------------------
        # 1. PARÁMETROS DEL MONITOREO
        # ---------------------------------------------------------

        mlflow.log_param(
            "drift_method",
            "Kolmogorov-Smirnov"
        )

        mlflow.log_param(
            "drift_threshold",
            threshold
        )

        # ---------------------------------------------------------
        # 2. MÉTRICAS POR VARIABLE
        # ---------------------------------------------------------

        for _, row in results_df.iterrows():

            feature = row["feature"]

            mlflow.log_metric(
                f"{feature}_ks_statistic",
                row["ks_statistic"]
            )

            mlflow.log_metric(
                f"{feature}_p_value",
                row["p_value"]
            )

            mlflow.log_metric(
                f"{feature}_drift",
                int(row["drift_detected"])
            )

        # ---------------------------------------------------------
        # 3. MÉTRICAS GENERALES
        # ---------------------------------------------------------

        drift_count = int(
            results_df["drift_detected"].sum()
        )

        mlflow.log_metric(
            "drifted_features_count",
            drift_count
        )

        mlflow.log_metric(
            "total_features",
            len(results_df)
        )

        mlflow.log_metric(
            "overall_drift",
            int(overall_drift)
        )

        # ---------------------------------------------------------
        # 4. RETRAIN TRIGGER
        # ---------------------------------------------------------

        retrain_required = int(overall_drift)

        mlflow.log_metric(
            "retrain_required",
            retrain_required
        )

        # ---------------------------------------------------------
        # 5. TAGS DE MONITOREO
        # ---------------------------------------------------------

        mlflow.set_tag(
            "monitoring_type",
            "data_drift"
        )

        mlflow.set_tag(
            "drift_status",
            "DRIFT_DETECTED"
            if overall_drift
            else "NO_DRIFT"
        )

        mlflow.set_tag(
            "retrain_trigger",
            "TRUE"
            if overall_drift
            else "FALSE"
        )

        mlflow.set_tag(
            "retrain_reason",
            "data_drift"
            if overall_drift
            else "none"
        )

        # ---------------------------------------------------------
        # 6. INFORMACIÓN DEL RUN
        # ---------------------------------------------------------

        run_id = mlflow.active_run().info.run_id

        print("\n--- MLflow Monitoring ---")
        print(f"Run ID: {run_id}")
        print(f"Variables con Drift: {drift_count}")
        print(f"Drift general: {overall_drift}")
        print(f"Retrain requerido: {bool(retrain_required)}")
        print(
            f"Razón del Retrain: "
            f"{'data_drift' if overall_drift else 'none'}"
        )
        print("Resultados registrados en MLflow.")


if __name__ == "__main__":

    REFERENCE_DATA = (
        "data/processed/wholesale_customers_scaled.csv"
    )

    CURRENT_DATA = (
        "data/processed/wholesale_customers_current.csv"
    )

    monitor_data_drift(
        REFERENCE_DATA,
        CURRENT_DATA
    )