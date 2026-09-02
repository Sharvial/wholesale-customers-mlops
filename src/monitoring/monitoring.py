import pandas as pd
import numpy as np
import joblib
import mlflow

from scipy.stats import ks_2samp
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


EXPERIMENT_NAME = "wholesale-customers-clustering"


# ============================================================
# 1. DATA DRIFT
# ============================================================

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


# ============================================================
# 2. MONITOREO DE CLUSTERS
# ============================================================

def calculate_cluster_monitoring(
    reference_path,
    current_path,
    model_path,
    distribution_threshold=0.20,
    centroid_threshold=1.0,
    silhouette_drop_threshold=0.10
):
    """
    Monitorea el comportamiento del clustering comparando
    los datos de referencia contra los datos actuales.

    Se evalúan:

    - Distribución de los clusters.
    - Movimiento de centroides.
    - Silhouette Score.
    - Degradación del modelo.
    """

    print("\n--- Cluster Monitoring ---")

    reference_df = pd.read_csv(reference_path)
    current_df = pd.read_csv(current_path)

    model = joblib.load(model_path)

    # --------------------------------------------------------
    # Predicciones con el modelo de referencia
    # --------------------------------------------------------

    reference_labels = model.predict(reference_df)

    # --------------------------------------------------------
    # Nuevo K-Means sobre los datos actuales
    # --------------------------------------------------------

    current_model = KMeans(
        n_clusters=model.n_clusters,
        random_state=42,
        n_init=10
    )

    current_labels = current_model.fit_predict(current_df)

    # --------------------------------------------------------
    # Alinear clusters
    #
    # Los números de cluster pueden cambiar de orden.
    # Por ejemplo:
    # Cluster 0 de referencia puede corresponder al
    # Cluster 2 del modelo actual.
    # --------------------------------------------------------

    distance_matrix = np.linalg.norm(
        model.cluster_centers_[:, np.newaxis, :]
        - current_model.cluster_centers_[np.newaxis, :, :],
        axis=2
    )

    reference_indices, current_indices = linear_sum_assignment(
        distance_matrix
    )

    cluster_mapping = {
        current_cluster: reference_cluster
        for reference_cluster, current_cluster
        in zip(reference_indices, current_indices)
    }

    aligned_current_labels = np.array([
        cluster_mapping[label]
        for label in current_labels
    ])

    # --------------------------------------------------------
    # Distribución de clusters
    # --------------------------------------------------------

    reference_distribution = (
        pd.Series(reference_labels)
        .value_counts(normalize=True)
        .sort_index()
    )

    current_distribution = (
        pd.Series(aligned_current_labels)
        .value_counts(normalize=True)
        .sort_index()
    )

    distribution_df = pd.DataFrame({
        "reference": reference_distribution,
        "current": current_distribution
    }).fillna(0)

    distribution_df["absolute_change"] = (
        distribution_df["current"]
        - distribution_df["reference"]
    ).abs()

    max_distribution_change = (
        distribution_df["absolute_change"].max()
    )

    # --------------------------------------------------------
    # Movimiento de centroides
    # --------------------------------------------------------

    aligned_current_centers = np.zeros_like(
        current_model.cluster_centers_
    )

    for reference_cluster, current_cluster in zip(
        reference_indices,
        current_indices
    ):
        aligned_current_centers[reference_cluster] = (
            current_model.cluster_centers_[current_cluster]
        )

    centroid_distances = np.linalg.norm(
        model.cluster_centers_
        - aligned_current_centers,
        axis=1
    )

    mean_centroid_movement = centroid_distances.mean()
    max_centroid_movement = centroid_distances.max()

    # --------------------------------------------------------
    # Silhouette Score
    # --------------------------------------------------------

    reference_silhouette = silhouette_score(
        reference_df,
        reference_labels
    )

    current_silhouette = silhouette_score(
        current_df,
        aligned_current_labels
    )

    silhouette_relative_drop = (
        (reference_silhouette - current_silhouette)
        / reference_silhouette
        if reference_silhouette != 0
        else 0
    )

    # --------------------------------------------------------
    # Detección de degradación
    # --------------------------------------------------------

    distribution_degradation = (
        max_distribution_change
        > distribution_threshold
    )

    centroid_degradation = (
        max_centroid_movement
        > centroid_threshold
    )

    silhouette_degradation = (
        silhouette_relative_drop
        > silhouette_drop_threshold
    )

    model_degradation = (
        distribution_degradation
        or centroid_degradation
        or silhouette_degradation
    )

    # --------------------------------------------------------
    # Resultados
    # --------------------------------------------------------

    print("\nDistribución de clusters:")

    for cluster, row in distribution_df.iterrows():

        print(
            f"Cluster {cluster}: "
            f"Referencia={row['reference']:.4f}, "
            f"Actual={row['current']:.4f}, "
            f"Cambio={row['absolute_change']:.4f}"
        )

    print(
        f"\nCambio máximo en distribución: "
        f"{max_distribution_change:.4f}"
    )

    print("\nMovimiento de centroides:")

    for cluster, distance in enumerate(centroid_distances):

        print(
            f"Cluster {cluster}: "
            f"{distance:.4f}"
        )

    print(
        f"\nMovimiento promedio de centroides: "
        f"{mean_centroid_movement:.4f}"
    )

    print(
        f"Movimiento máximo de centroides: "
        f"{max_centroid_movement:.4f}"
    )

    print(
        f"\nSilhouette referencia: "
        f"{reference_silhouette:.4f}"
    )

    print(
        f"Silhouette actual: "
        f"{current_silhouette:.4f}"
    )

    print(
        f"Caída relativa del Silhouette: "
        f"{silhouette_relative_drop:.4f}"
    )

    print(
        f"\nDegradación por distribución: "
        f"{distribution_degradation}"
    )

    print(
        f"Degradación por centroides: "
        f"{centroid_degradation}"
    )

    print(
        f"Degradación por Silhouette: "
        f"{silhouette_degradation}"
    )

    print(
        f"\nDegradación general del modelo: "
        f"{model_degradation}"
    )

    return {
        "distribution_df": distribution_df,
        "max_distribution_change": max_distribution_change,
        "mean_centroid_movement": mean_centroid_movement,
        "max_centroid_movement": max_centroid_movement,
        "reference_silhouette": reference_silhouette,
        "current_silhouette": current_silhouette,
        "silhouette_relative_drop": silhouette_relative_drop,
        "distribution_degradation": distribution_degradation,
        "centroid_degradation": centroid_degradation,
        "silhouette_degradation": silhouette_degradation,
        "model_degradation": model_degradation
    }


# ============================================================
# 3. MONITOREO COMPLETO + MLFLOW
# ============================================================

def monitor_data_drift(
    reference_path,
    current_path,
    model_path="models/baseline_kmeans.pkl",
    threshold=0.05,
    distribution_threshold=0.20,
    centroid_threshold=1.0,
    silhouette_drop_threshold=0.10
):
    """
    Ejecuta Data Drift y monitoreo de clustering.

    Importante:
    Data Drift NO implica automáticamente reentrenamiento.

    El reentrenamiento se activa cuando existe degradación
    o inestabilidad del modelo.
    """

    mlflow.set_experiment(EXPERIMENT_NAME)

    # --------------------------------------------------------
    # Data Drift
    # --------------------------------------------------------

    results_df, overall_drift = calculate_data_drift(
        reference_path,
        current_path,
        threshold
    )

    # --------------------------------------------------------
    # Cluster Monitoring
    # --------------------------------------------------------

    cluster_results = calculate_cluster_monitoring(
        reference_path,
        current_path,
        model_path,
        distribution_threshold,
        centroid_threshold,
        silhouette_drop_threshold
    )

    model_degradation = cluster_results["model_degradation"]

    # --------------------------------------------------------
    # Regla de retraining
    # --------------------------------------------------------

    retrain_required = model_degradation

    if model_degradation:
        retrain_reason = "model_degradation"
    elif overall_drift:
        retrain_reason = "data_drift_review"
    else:
        retrain_reason = "none"

    # --------------------------------------------------------
    # MLflow
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name="data-drift-and-cluster-monitoring"
    ):

        # Parámetros

        mlflow.log_param(
            "drift_method",
            "Kolmogorov-Smirnov"
        )

        mlflow.log_param(
            "drift_threshold",
            threshold
        )

        mlflow.log_param(
            "distribution_threshold",
            distribution_threshold
        )

        mlflow.log_param(
            "centroid_movement_threshold",
            centroid_threshold
        )

        mlflow.log_param(
            "silhouette_drop_threshold",
            silhouette_drop_threshold
        )

        # Métricas de Data Drift

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

        # Métricas de clustering

        mlflow.log_metric(
            "max_cluster_distribution_change",
            cluster_results["max_distribution_change"]
        )

        mlflow.log_metric(
            "mean_centroid_movement",
            cluster_results["mean_centroid_movement"]
        )

        mlflow.log_metric(
            "max_centroid_movement",
            cluster_results["max_centroid_movement"]
        )

        mlflow.log_metric(
            "reference_silhouette",
            cluster_results["reference_silhouette"]
        )

        mlflow.log_metric(
            "current_silhouette",
            cluster_results["current_silhouette"]
        )

        mlflow.log_metric(
            "silhouette_relative_drop",
            cluster_results["silhouette_relative_drop"]
        )

        mlflow.log_metric(
            "model_degradation",
            int(model_degradation)
        )

        mlflow.log_metric(
            "retrain_required",
            int(retrain_required)
        )

        # Tags

        mlflow.set_tag(
            "monitoring_type",
            "data_drift_and_clustering"
        )

        mlflow.set_tag(
            "drift_status",
            "DRIFT_DETECTED"
            if overall_drift
            else "NO_DRIFT"
        )

        mlflow.set_tag(
            "model_status",
            "DEGRADED"
            if model_degradation
            else "STABLE"
        )

        mlflow.set_tag(
            "retrain_trigger",
            "TRUE"
            if retrain_required
            else "FALSE"
        )

        mlflow.set_tag(
            "retrain_reason",
            retrain_reason
        )

        # Información del Run

        run_id = mlflow.active_run().info.run_id

        print("\n========================================")
        print("      MLFLOW MONITORING")
        print("========================================")

        print(f"Run ID: {run_id}")
        print(f"Variables con Drift: {drift_count}")
        print(f"Drift general: {overall_drift}")
        print(f"Degradación del modelo: {model_degradation}")
        print(f"Retrain requerido: {retrain_required}")
        print(f"Razón: {retrain_reason}")
        print("Resultados registrados en MLflow.")


# ============================================================
# 4. EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    REFERENCE_DATA = (
        "data/processed/wholesale_customers_scaled.csv"
    )

    CURRENT_DATA = (
        "data/processed/wholesale_customers_current.csv"
    )

    MODEL_PATH = (
        "models/baseline_kmeans.pkl"
    )

    monitor_data_drift(
        REFERENCE_DATA,
        CURRENT_DATA,
        MODEL_PATH
    )