# Metrics.py - Corrected version aligned with literature (2025)
# Full-reference and no-reference metrics commonly used in dehazing / image enhancement.

import numpy as np
import cv2

def calculate_metrics(original, enhanced):
    """
    Compute standard image quality metrics used in dehazing literature.

    Full-reference (need original + enhanced):
        SSIM, PSNR, MSE, CII, QF (visible-edge gradient ratio)

    No-reference (only enhanced, or both for improvement):
        Entropy, Avg_Gradient, Edge_Intensity, FADE (approximation note)

    References:
        - SSIM: Wang et al., IEEE TIP 2004
        - PSNR / MSE: classic
        - Entropy: Shannon entropy on grayscale histogram
        - Average Gradient: mean of gradient magnitudes (common sharpness measure)
        - Edge Intensity: EI(F) = sqrt(sum sum (sx^2 + sy^2)) / (M*N)
          (amplitude of edge gradients, normalized by image size)
        - QF / visible-edge ratio: inspired by Hautière, Tarel et al.
          "Blind Contrast Enhancement Assessment by Gradient Ratioing at Visible Edges"
          Image Analysis & Stereology, 2008 (the classic e, r̄, σ metrics)
        - CII: Contrast Improvement Index (std or local contrast ratio)
        - FADE: Choi, You, Bovik, IEEE TIP 2015. The real FADE uses NSS + MVG
          models and is not re-implemented here (official code required).
          A simple proxy is provided for convenience only.
    """
    # Ensure float64 for calculations
    orig = original.astype(np.float64)
    enh  = enhanced.astype(np.float64)

    # Grayscale versions (uint8 for histogram / Sobel stability)
    if original.ndim == 3:
        gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        gray_enh  = cv2.cvtColor(enhanced,  cv2.COLOR_BGR2GRAY)
    else:
        gray_orig = original
        gray_enh  = enhanced

    # ------------------- 1. SSIM (manual, Wang et al. formulation) -------------------
    def ssim_manual(img1, img2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.T)
        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq  = mu1 ** 2
        mu2_sq  = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12   = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return float(ssim_map.mean())

    ssim_val = ssim_manual(gray_orig, gray_enh)

    # ------------------- 2. PSNR & MSE -------------------
    mse = np.mean((orig - enh) ** 2)
    if mse < 1e-10:
        psnr = float('inf')
    else:
        psnr = 20.0 * np.log10(255.0 / np.sqrt(mse))

    # ------------------- 3. Entropy (correct Shannon on 256-bin histogram) -------------------
    def image_entropy(gray):
        """Shannon entropy of grayscale image (literature standard)."""
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        hist = hist / (hist.sum() + 1e-12)
        hist = hist[hist > 0]
        return float(-np.sum(hist * np.log2(hist)))

    ent_orig = image_entropy(gray_orig)
    ent_enh  = image_entropy(gray_enh)
    entropy_gain = ent_enh - ent_orig   # sometimes reported as information gain

    # ------------------- 4. Average Gradient (sharpness) -------------------
    # Mean of gradient magnitudes: (1/(M*N)) * sum sqrt(sx^2 + sy^2)
    gx = cv2.Sobel(gray_enh, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_enh, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    avg_gradient = float(np.mean(grad_mag))

    # ------------------- 5. Edge Intensity (literature formula) -------------------
    # EI(F) = sqrt( sum_i sum_j (sx(i,j)^2 + sy(i,j)^2) ) / (M * N)
    # This is the L2-norm of the gradient field normalized by image size.
    h, w = gray_enh.shape[:2]
    sum_sq = np.sum(gx ** 2 + gy ** 2)
    edge_intensity = float(np.sqrt(sum_sq) / (h * w))

    # ------------------- 6. QF / Visible-Edge Gradient Ratio -------------------
    # Inspired by Hautière, Tarel et al. 2008 (r̄ metric).
    # Ratio of gradient magnitudes on strong edges of the original image.
    gx_o = cv2.Sobel(gray_orig, cv2.CV_64F, 1, 0, ksize=3)
    gy_o = cv2.Sobel(gray_orig, cv2.CV_64F, 0, 1, ksize=3)
    grad_orig = np.sqrt(gx_o ** 2 + gy_o ** 2)

    # Visible edges: top 10 % strongest gradients in original (common practical choice)
    thresh = np.percentile(grad_orig, 90)
    mask = grad_orig > thresh
    if np.count_nonzero(mask) > 100:
        # Arithmetic mean of ratios (stable). Geometric mean also used in literature.
        ratios = grad_mag[mask] / (grad_orig[mask] + 1e-8)
        qf = float(np.mean(ratios))
    else:
        qf = avg_gradient / (float(np.mean(grad_orig)) + 1e-8)

    # ------------------- 7. FADE (Fog Aware Density Evaluator) -------------------
    # IMPORTANT: The real FADE (Choi et al., IEEE TIP 2015) is a sophisticated
    # no-reference metric based on 12 NSS fog-aware features + MVG models.
    # It cannot be correctly re-implemented without the original trained parameters
    # (official MATLAB code: http://live.ece.utexas.edu/research/fog/).
    # Here we keep only a simple proxy for convenience; do NOT cite it as FADE.
    def simple_fog_proxy(img):
        """Very rough visibility proxy (NOT real FADE). Lower ≈ clearer."""
        if img.ndim == 3:
            gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            hsv  = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV)
            sat  = hsv[:, :, 1].mean()
        else:
            gray = img.astype(np.uint8)
            sat  = 0.0
        contrast   = gray.std()
        brightness = gray.mean()
        # Ad-hoc combination (for illustration only)
        score = np.exp(-0.05 * contrast + 0.01 * sat + 0.0005 * brightness)
        return float(np.clip(score, 0.0, 1.0))

    fade_proxy_enh  = simple_fog_proxy(enhanced)
    fade_proxy_orig = simple_fog_proxy(original)
    fade_proxy_imp  = fade_proxy_orig - fade_proxy_enh   # positive ≈ less fog

    # ------------------- 8. Contrast Improvement Index (CII) -------------------
    # Standard definition: C_enhanced / C_original
    # Here C = global standard deviation (very common practical choice).
    # Alternative literature versions use local Michelson contrast on blocks.
    cii = float(enhanced.std() / (original.std() + 1e-8))

    # ------------------- Results -------------------
    return {
        'SSIM':           round(ssim_val, 4),
        'PSNR':           round(float(psnr), 4) if psnr != float('inf') else float('inf'),
        'MSE':            round(float(mse), 4),
        'Entropy':        round(ent_enh, 4),
        'Entropy_Gain':   round(entropy_gain, 4),   # was called IQE (non-standard name)
        'Avg_Gradient':   round(avg_gradient, 4),
        'Edge_Intensity': round(edge_intensity, 2),
        'QF':             round(qf, 4),             # visible-edge gradient ratio
        'FADE_proxy':     round(fade_proxy_enh, 4), # NOT real FADE – see comment
        'FADE_proxy_Imp': round(fade_proxy_imp, 4),
        'CII':            round(cii, 4),
    }
