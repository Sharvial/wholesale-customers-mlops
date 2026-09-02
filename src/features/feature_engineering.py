import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from sklearn.compose import ColumnTransformer 
import joblib

def build_feature_pipeline():
    """
    Construye un pipeline de Scikit-Learn reutilizable para producción.
    Empaqueta las decisiones tomadas en el EDA.
    """
    # Decisión 1: Transformación Logarítmica para corregir la asimetría (skewness)
    # Usamos log1p (log(x + 1)) por si hay valores de gasto en 0
    log_transformer = FunctionTransformer(np.log1p, validate=True)
    
    # Decisión 2: RobustScaler para manejar los valores atípicos (outliers)
    robust_scaler = RobustScaler()
    
    # Unimos ambos pasos en un solo Pipeline
    pipeline = Pipeline(steps=[
        ('log_transform', log_transformer),
        ('scaler', robust_scaler)
    ])
    
    return pipeline

def process_data(input_path, output_path):
    print(f"Cargando datos validados desde: {input_path}")
    df = pd.read_csv(input_path)
    
    # Para el clustering de gastos, descartamos variables categóricas de clasificación previa
    cols_to_drop = ['Channel', 'Region']
    df_features = df.drop(columns=cols_to_drop)
    
    # Construimos y entrenamos el pipeline
    print("Aplicando Feature Pipeline (Log1p + RobustScaler)...")
    pipeline = build_feature_pipeline()
    
    # Transformamos los datos
    df_processed_array = pipeline.fit_transform(df_features)
    
    # Reconstruimos el DataFrame para guardarlo
    df_processed = pd.DataFrame(df_processed_array, columns=df_features.columns)
    
    # Guardamos los datos procesados
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_processed.to_csv(output_path, index=False)
    
    print(f"Transformación exitosa. Datos listos para MLflow guardados en: {output_path}")
    print(f"Dimensiones finales: {df_processed.shape}")
    
    return pipeline

if __name__ == "__main__":
    # Definimos rutas según la estructura del repositorio
    INPUT_DATA = "data/raw/wholesale_customers.csv"
    OUTPUT_DATA = "data/processed/wholesale_customers_scaled.csv"
    PIPELINE_OUTPUT = "models/feature_pipeline.pkl"
    
    # 1. Ejecutamos el procesamiento y capturamos el pipeline entrenado
    trained_pipeline = process_data(INPUT_DATA, OUTPUT_DATA)
    
    # 2. Guardamos el pipeline de preprocesamiento para que Sharon lo use en la API
    os.makedirs(os.path.dirname(PIPELINE_OUTPUT), exist_ok=True)
    joblib.dump(trained_pipeline, PIPELINE_OUTPUT)
    
    print(f"Pipeline exportado EXITOSAMENTE para la API en: {PIPELINE_OUTPUT}")
