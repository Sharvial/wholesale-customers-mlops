import time
import os
import mlflow
import psutil


EXPERIMENT_NAME = "wholesale-customers-clustering"


def collect_system_metrics():

    print("\n--- System Metrics ---")

    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)

    # Memoria RAM
    memory = psutil.virtual_memory()
    memory_percent = memory.percent

    # Memoria disponible en MB
    memory_available_mb = memory.available / (1024 ** 2)

    # Uso de disco
    disk = psutil.disk_usage("/")
    disk_percent = disk.percent

    print(f"CPU: {cpu_percent:.2f}%")
    print(f"Memoria RAM: {memory_percent:.2f}%")
    print(f"Memoria disponible: {memory_available_mb:.2f} MB")
    print(f"Disco utilizado: {disk_percent:.2f}%")

    return {
        "cpu_percent": cpu_percent,
        "memory_percent": memory_percent,
        "memory_available_mb": memory_available_mb,
        "disk_percent": disk_percent
    }


def monitor_system_metrics():

    mlflow.set_experiment(EXPERIMENT_NAME)

    start_time = time.time()

    metrics = collect_system_metrics()

    execution_time = time.time() - start_time

    with mlflow.start_run(
        run_name="system-metrics-monitoring"
    ):

        # Registrar métricas del sistema
        mlflow.log_metric(
            "cpu_percent",
            metrics["cpu_percent"]
        )

        mlflow.log_metric(
            "memory_percent",
            metrics["memory_percent"]
        )

        mlflow.log_metric(
            "memory_available_mb",
            metrics["memory_available_mb"]
        )

        mlflow.log_metric(
            "disk_percent",
            metrics["disk_percent"]
        )

        mlflow.log_metric(
            "execution_time_seconds",
            execution_time
        )

        # Parámetros
        mlflow.log_param(
            "monitoring_type",
            "system_metrics"
        )

        mlflow.log_param(
            "operating_system",
            os.name
        )

        # Tags
        mlflow.set_tag(
            "monitoring_type",
            "system_metrics"
        )

        mlflow.set_tag(
            "monitoring_status",
            "completed"
        )

        run_id = mlflow.active_run().info.run_id

        print("\n--- MLflow System Monitoring ---")
        print(f"Run ID: {run_id}")
        print("Métricas del sistema registradas.")
        print("Monitoreo completado correctamente.")


if __name__ == "__main__":

    monitor_system_metrics()