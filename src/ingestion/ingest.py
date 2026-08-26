import pandas as pd
import os

def download_data():
    # URL directa al dataset de Wholesale Customers de UCI
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"
    
    # Ruta donde guardaremos los datos crudos (bronze)
    output_dir = "data/raw"
    output_path = os.path.join(output_dir, "wholesale_customers.csv")
    
    # Crear el directorio si no existe (por seguridad)
    os.makedirs(output_dir, exist_ok=True)
    
    print("Iniciando la descarga del dataset de Wholesale Customers...")
    try:
        # Leer directamente desde la web
        df = pd.read_csv(url)
        
        # Guardar localmente sin el índice
        df.to_csv(output_path, index=False)
        print(f"Dataset descargado exitosamente.")
        print(f"Guardado en: {output_path}")
        print(f"Dimensiones iniciales: {df.shape}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    download_data()