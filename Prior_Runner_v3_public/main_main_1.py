import cv2
import numpy as np
import os
import csv
import time
import argparse
from scipy.ndimage import minimum_filter
from Region import Region, clip_region
from QuadTree import QuadTree, get_leaves
from Metrics import calculate_metrics

from Optimizers import lissajou_optimize,ga_optimize, pso_optimize, cso_optimize, sinecosine_optimize, gasdi_optimize, nro_optimize,qio_optimize, kla_optimize

def process_image(image_path):
    # Obtener el directorio del script para resolver rutas relativas correctamente
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Si la ruta proporcionada no es absoluta, la busca desde el directorio del script
    if not os.isabs(image_path):
        full_image_path = os.path.join(script_dir, image_path)
    else:
        full_image_path = image_path

    # Verificar existencia
    if not os.path.exists(full_image_path):
        print(f"Error: La imagen '{image_path}' no se encontró en: {full_image_path}")
        return

    print(f"¡Imagen encontrada con éxito!: {full_image_path}")
    # Aquí va tu código de procesamiento (ej. cv2.imread(full_image_path))


# --- 1. Funciones Centrales del Algoritmo de Mejora ---

def fitness(region, gray):
    """Calcula el fitness (mu / (sigma^2 + phi)) para la estimación de la luz atmosférica."""
    sub = gray[region.y:region.y + region.h, region.x:region.x + region.w]
    if sub.size == 0:
        return 0
    mu = np.mean(sub)
    sigma2 = np.var(sub)
    phi = 1e-6
    # Maximizar: brillante (mu alto) y uniforme (sigma2 bajo)
    return mu / (sigma2 + phi)

def get_atm_light(img, region):
    """Estima la Luz Atmosférica (A) promediando los píxeles en la región óptima."""
    sub = img[region.y:region.y + region.h, region.x:region.x + region.w]
    if sub.size == 0:
        return np.array([255.0, 255.0, 255.0], dtype=np.float32)
    A = np.mean(sub, axis=(0, 1)).astype(np.float32)
    return A

def get_transmission(hazy_float, A_float, omega=0.95, size=15):
    """Estima el mapa de transmisión (t) usando el Dark Channel Prior."""
    norm = hazy_float / A_float
    norm_min = np.min(norm, axis=2)
    raw_dark_norm = minimum_filter(norm_min, size=size, mode='nearest')
    return raw_dark_norm

def guided_filter(guide, src, r=60, eps=1e-3):
    """Filtro guiado para refinar el mapa de transmisión."""
    mean_I = cv2.boxFilter(guide, cv2.CV_32F, (r, r))
    mean_p = cv2.boxFilter(src, cv2.CV_32F, (r, r))
    mean_Ip = cv2.boxFilter(guide * src, cv2.CV_32F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(guide * guide, cv2.CV_32F, (r, r))
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_32F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (r, r))
    q = mean_a * guide + mean_b
    return q

def recover(hazy_float, t, A_float, t0=0.1):
    """Recupera la imagen sin velo (dehazing) y la invierte para mejorar la iluminación."""
    t = np.maximum(t, t0)
    res = np.empty(hazy_float.shape, dtype=np.float32)
    # Ecuación de dehazing: J = (I - A) / max(t, t0) + A
    for i in range(3):
        res[:, :, i] = (hazy_float[:, :, i] - A_float[i]) / t + A_float[i]
    res = np.clip(res, 0, 1)
    
    # Invertir de nuevo para el realce de baja luz: 255 - J
    enhanced = (res * 255).astype(np.uint8)
    enhanced = 255 - enhanced 
    return enhanced

# --- 2. Función Principal de Experimento ---

def enhance_with_optimizer(input_path, output_dir, optimizer_name, optimizer_func, save_image=True):
    """Corre el algoritmo de mejora con un optimizador específico.
    
    Parameters
    ----------
    save_image : bool
        If True, writes the enhanced image to disk. Set to False for intermediate
        statistical runs when only metrics are required.
    """
    orig = cv2.imread(input_path)
    if orig is None:
        print(f"Error: Falló la carga de la imagen {input_path}.")
        return

    # Preparación de la "Imagen con velo (Hazy)" (Inversión)
    start_time = time.time()
    hazy = 255 - orig
    hazy_float = hazy.astype(np.float32) / 255.0
    gray = cv2.cvtColor(hazy, cv2.COLOR_BGR2GRAY)
    
    # 1. Segmentación con QuadTree
    qt = QuadTree(0, 0, hazy.shape[1], hazy.shape[0], gray)
    leaves = []
    get_leaves(qt, leaves)
    if not leaves:
        print(f"Advertencia ({optimizer_name}): No se encontraron nodos hoja del QuadTree.")
        return

    # 2. Estimación de Luz Atmosférica (A) usando el Optimizador
    As = []
    # Ejecuta el optimizador 10 veces para una estimación más robusta
    for _ in range(10): 
        best_region = optimizer_func(leaves, gray, fitness)
        A_i = get_atm_light(hazy, best_region)
        As.append(A_i)
    
    A = np.mean(As, axis=0)
    A_float = A / 255.0
    
    # 3. Estimación del Mapa de Transmisión (t)
    avg_bright = np.mean(gray)
    patch_size = 9 if avg_bright > 180 else (7 if avg_bright > 80 else 3)

    raw_dark_norm = get_transmission(hazy_float, A_float, omega=0.95, size=patch_size)
    
    # Suavizado de t con Operaciones Morfológicas y Filtro Guiado
    dark_uint = (raw_dark_norm * 255).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    morph_dark = cv2.morphologyEx(dark_uint, cv2.MORPH_OPEN, kernel)
    morph_dark_float = morph_dark.astype(np.float32) / 255.0
    raw_t = 1 - 0.95 * morph_dark_float
    
    guide = cv2.cvtColor(hazy_float, cv2.COLOR_BGR2GRAY)
    r = max(15, max(hazy.shape[:2]) // 10)
    t = guided_filter(guide, raw_t, r=r, eps=1e-3)
    
    # 4. Recuperación de Imagen y Finalización
    enhanced_image = recover(hazy_float, t, A_float)
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Guardar la imagen mejorada (only when requested)
    if save_image:
        output_path = os.path.join(output_dir, f"enhanced_{optimizer_name}.jpg")
        cv2.imwrite(output_path, enhanced_image)
    
    # Calcular y guardar métricas
    metrics = calculate_metrics(orig, enhanced_image)
    metrics['Optimizer'] = optimizer_name
    metrics['Time (s)'] = elapsed_time
    
    return metrics

# --- 3. Ejecución de Experimentos ---
# Number of independent repetitions required for Wilcoxon signed-rank
# and Friedman ranking statistical tests.
NUM_RUNS = 30

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procesador de imágenes")
    parser.add_argument("image", type=str, help="Nombre o ruta de la imagen")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 🔹 Reasignación con el argumento analizado por argparse:
    INPUT_IMAGE = os.path.join(script_dir, args.image)

    # Verifica si la imagen existe
    if not os.path.exists(INPUT_IMAGE):
        print(f"Error: La imagen de entrada '{INPUT_IMAGE}' no se encontró. ¡Asegúrate de colocarla en el mismo directorio!")
    else:
        # Extraer el nombre base de la imagen sin extensión (p. ej., "foto1.jpg" -> "foto1")
        image_name = os.path.splitext(os.path.basename(args.image))[0]

        # Diccionario de optimizadores a probar (all enabled for ranking tests)
        optimizers = {
            "LDE": lissajou_optimize,
            "GA": ga_optimize,
            "PSO": pso_optimize,
            "CSO": cso_optimize,
            "SineCosine": sinecosine_optimize,
            "GASDI": gasdi_optimize,
            "NRO": nro_optimize,
            "QIO": qio_optimize,
            "KLA": kla_optimize,
        }

        # Crear estructura de directorio de salida específica para la imagen actual:
        # enhanced_results / <image_name> /
        OUTPUT_BASE_DIR = os.path.join("enhanced_results", image_name)
        CSV_DIR = os.path.join(OUTPUT_BASE_DIR, "metrics_csv")

        os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
        os.makedirs(CSV_DIR, exist_ok=True)

        all_metrics = []  # will hold every run of every optimizer

        for name, func in optimizers.items():
            print(f"\n=== Optimizador: {name}  |  Imagen: {image_name}  |  {NUM_RUNS} ejecuciones ===")

            # Crear subcarpeta para el optimizador dentro del directorio de la imagen
            optimizer_output_dir = os.path.join(OUTPUT_BASE_DIR, name)
            os.makedirs(optimizer_output_dir, exist_ok=True)

            run_metrics_list = []

            for run in range(1, NUM_RUNS + 1):
                print(f"  -> Run {run}/{NUM_RUNS} ...", end=" ", flush=True)
                try:
                    # Save the enhanced image only on the final run (keeps disk usage low)
                    save_img = (run == NUM_RUNS)
                    metrics = enhance_with_optimizer(
                        INPUT_IMAGE, optimizer_output_dir, name, func,
                        save_image=save_img
                    )
                    if metrics:
                        metrics = dict(metrics)  # shallow copy
                        metrics["Image"] = image_name
                        metrics["Run"] = run
                        run_metrics_list.append(metrics)
                        all_metrics.append(metrics)
                        print(f"OK  (SSIM={metrics.get('SSIM', 'n/a')})")
                    else:
                        print("FAILED (no metrics returned)")
                except Exception as e:
                    print(f"ERROR: {e}")

            # Guardar CSV individual del algoritmo (todas las runs de esta imagen)
            if run_metrics_list:
                csv_path = os.path.join(CSV_DIR, f"{name}_metrics.csv")
                fieldnames = list(run_metrics_list[0].keys())
                # Ensure consistent column order
                preferred = ["Image", "Run", "Optimizer", "Time (s)",
                             "SSIM", "PSNR", "MSE", "Entropy", "Entropy_Gain",
                             "Avg_Gradient", "Edge_Intensity", "QF",
                             "FADE_proxy", "FADE_proxy_Imp", "CII"]
                fieldnames = [c for c in preferred if c in fieldnames] + \
                             [c for c in fieldnames if c not in preferred]

                with open(csv_path, "w", newline="") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(run_metrics_list)
                print(f"  CSV del algoritmo '{name}' guardado: {csv_path}  ({len(run_metrics_list)} filas)")

        # Guardar todas las métricas juntas para esta imagen (todas las runs + todos los algoritmos)
        if all_metrics:
            combined_csv_path = os.path.join(
                OUTPUT_BASE_DIR, f"{image_name}_combined_metrics_summary.csv"
            )
            fieldnames = list(all_metrics[0].keys())
            preferred = ["Image", "Run", "Optimizer", "Time (s)",
                         "SSIM", "PSNR", "MSE", "Entropy", "Entropy_Gain",
                         "Avg_Gradient", "Edge_Intensity", "QF",
                         "FADE_proxy", "FADE_proxy_Imp", "CII"]
            fieldnames = [c for c in preferred if c in fieldnames] + \
                         [c for c in fieldnames if c not in preferred]

            with open(combined_csv_path, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_metrics)
            print(f"\nResumen combinado ({len(all_metrics)} filas) guardado en: {combined_csv_path}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Procesador de imágenes")
#     parser.add_argument("image", type=str, help="Nombre o ruta de la imagen")
#     args = parser.parse_args()
#
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#
#     # 🔹 Reasignación con el argumento analizado por argparse:
#     INPUT_IMAGE = os.path.join(script_dir, args.image)
#
#     # Verifica si la imagen existe
#     if not os.path.exists(INPUT_IMAGE):
#         print(f"Error: La imagen de entrada '{INPUT_IMAGE}' no se encontró. ¡Asegúrate de colocarla en el mismo directorio!")
#     else:
#
#         # Diccionario de optimizadores a probar
#         optimizers = {
#             "LDE":lissajou_optimize,
#     #
#         }
#
# #            optimizers = {
# #             "LDE":lissajou_optimize,
# #             "GA": ga_optimize,
# #             "PSO": pso_optimize,
# #             "CSO": cso_optimize,
# #             "SineCosine": sinecosine_optimize,
# #             "GASDI": gasdi_optimize,
# #             "NRO": nro_optimize,
# #             "QIO": qio_optimize,       #
# #             "KLA": kla_optimize,        #
# #         }
# #
#         # Crear directorios de salida
#         OUTPUT_BASE_DIR = "enhanced_results"
#         CSV_DIR = os.path.join(OUTPUT_BASE_DIR, "metrics_csv")
#
#         os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
#         os.makedirs(CSV_DIR, exist_ok=True)
#
#         all_metrics = []
#
#         for name, func in optimizers.items():
#             print(f"\n--- Ejecutando Experimento con {name} ---")
#
#             # Crear subcarpeta para el optimizador
#             optimizer_output_dir = os.path.join(OUTPUT_BASE_DIR, name)
#             print(optimizer_output_dir)
#             os.makedirs(optimizer_output_dir, exist_ok=True)
#
#             # Ejecutar y obtener métricas
#             try:
#                 metrics = enhance_with_optimizer(INPUT_IMAGE, optimizer_output_dir, name, func)
#                 if metrics:
#                     all_metrics.append(metrics)
#                     print(f"Resultados de {name} guardados en {optimizer_output_dir}.")
#
#                     # Guardar métricas en CSV individual
#                     csv_path = os.path.join(CSV_DIR, f"{name}_metrics.csv")
#                     fieldnames = metrics.keys()
#                     with open(csv_path, 'w', newline='') as csvfile:
#                         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#                         writer.writeheader()
#                         writer.writerow(metrics)
#                     print(f"Métricas de {name} guardadas en {csv_path}")
#             except Exception as e:
#                 print(f"Error al ejecutar {name}: {e}")
#
#         # Guardar todas las métricas juntas (opcional, para una tabla comparativa)
#         if all_metrics:
#             combined_csv_path = os.path.join(OUTPUT_BASE_DIR, "combined_metrics_summary.csv")
#             fieldnames = all_metrics[0].keys()
#             with open(combined_csv_path, 'w', newline='') as csvfile:
#                 writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#                 writer.writeheader()
#                 writer.writerows(all_metrics)
#             print(f"\nResumen de todas las métricas guardado en: {combined_csv_path}")
