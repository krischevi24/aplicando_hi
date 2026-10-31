import os
import subprocess

# Rango de 1 a 69
for i in range(1, 70): # 70
    # Formatea el número a dos dígitos (01, 02, ..., 69)
    image_name = f"{i:02d}.jpg"

    print(f"\n--- Procesando: {image_name} ---")

    # Ejecuta el comando: python main_main.py XX.jpg
    result = subprocess.run(["python", "main_main.py", image_name])

    # Si quieres detener el proceso si ocurre un error en alguna imagen:
    if result.returncode != 0:
        print(f"Error procesando {image_name}. Deteniendo...")
        break
