import os
import subprocess
import sys

# Rango de 1 a 69
# Each image will be processed with every optimizer, repeated NUM_RUNS=30 times
# inside main_main.py (required for Wilcoxon / Friedman tests).

for i in range(2, 8):
    # Formatea el número a dos dígitos (01, 02, ..., 69)
    image_name = f"{i:02d}.jpeg"

    print(f"\n{'='*60}")
    print(f"--- Procesando: {image_name} ---")
    print(f"{'='*60}")

    # Ejecuta el comando: python main_main.py XX.jpg
    result = subprocess.run([sys.executable, "main_main.py", image_name])

    # Si quieres detener el proceso si ocurre un error en alguna imagen:
    if result.returncode != 0:
        print(f"Error procesando {image_name}. Deteniendo...")
        break

print("\n*** Todas las imágenes han sido procesadas. ***")
print("Los CSV por algoritmo se encuentran en:")
print("  enhanced_results/<image_name>/metrics_csv/<ALG>_metrics.csv")
print("Cada archivo contiene las 30 ejecuciones independientes de ese algoritmo sobre la imagen.")
