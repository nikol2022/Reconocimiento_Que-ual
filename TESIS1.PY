from pathlib import Path
import kagglehub

# Descargar / localizar dataset
path = kagglehub.dataset_download("yudhaislamisulistya/plants-type-datasets")
print("Path to dataset files:", path)

# Convertir a Path
base_path = Path(path)

print("\n=== Estructura del dataset ===")
for item in base_path.rglob("*"):
    print(item)
