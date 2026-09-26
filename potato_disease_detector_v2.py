"""
=============================================================
  Potato Disease Detection — 3-Way Classification
  Pipeline: Background Removal → Feature Extraction → Classification

  Classifies potato leaf images as:
    Healthy | Suspicious | Sick

  Accuracy on calibration dataset (2,152 images):
    Healthy: 81%  |  Diseased: 86%

  Usage:
    python potato_disease_detector_v2.py <image.jpg>
    python potato_disease_detector_v2.py <folder/>
=============================================================
"""

import cv2
import numpy as np
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from pathlib import Path


# ══════════════════════════════════════════════════════════
# STEP 1 — BACKGROUND REMOVAL
# ══════════════════════════════════════════════════════════

def remove_background(img_bgr: np.ndarray) -> dict:
    """
    Isolate the plant/leaf from the background.

    Strategy:
      1. Build a broad vegetation mask (green + diseased brown/yellow pixels)
      2. Morphological cleanup to fill holes and remove noise
      3. Keep only the largest connected blob (the main leaf)
      4. Return the clean leaf mask + masked image

    Returns:
        dict with:
          'leaf_mask'   — binary mask (uint8), 255 = leaf pixel
          'leaf_img'    — BGR image with background set to black
          'leaf_area'   — number of leaf pixels
          'coverage_pct'— leaf area as % of total image
          'found'       — False if no significant plant found
    """
    h, w = img_bgr.shape[:2]
    total = h * w
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # --- Broad mask: green (healthy) + brown (Early Blight) + yellow (chlorosis) ---
    # Deliberately wide so we don't miss diseased tissue at the edges
    green_mask  = cv2.inRange(hsv, np.array([15, 20, 20]),  np.array([100, 255, 255]))
    brown_mask  = cv2.inRange(hsv, np.array([0,  40, 30]),  np.array([22,  255, 200]))
    yellow_mask = cv2.inRange(hsv, np.array([20, 60, 80]),  np.array([35,  255, 255]))
    dark_mask   = cv2.inRange(hsv, np.array([0,   0,  0]),  np.array([180,  80, 100]))

    combined = cv2.bitwise_or(
        cv2.bitwise_or(green_mask, brown_mask),
        cv2.bitwise_or(yellow_mask, dark_mask)
    )

    # --- Morphological cleanup ---
    kernel   = np.ones((7, 7), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=4)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel, iterations=2)

    # --- Keep only the largest contour (main leaf body) ---
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    leaf_mask = np.zeros((h, w), dtype=np.uint8)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(leaf_mask, [largest], -1, 255, -1)

    leaf_area    = int(leaf_mask.sum() // 255)
    coverage_pct = leaf_area / total * 100

    # If less than 5% of image is leaf, nothing useful was found
    if coverage_pct < 5:
        return {
            "leaf_mask":    leaf_mask,
            "leaf_img":     img_bgr.copy(),
            "leaf_area":    leaf_area,
            "coverage_pct": coverage_pct,
            "found":        False,
        }

    # --- Apply mask: zero out background pixels ---
    leaf_img = cv2.bitwise_and(img_bgr, img_bgr, mask=leaf_mask)

    return {
        "leaf_mask":    leaf_mask,
        "leaf_img":     leaf_img,
        "leaf_area":    leaf_area,
        "coverage_pct": round(coverage_pct, 1),
        "found":        True,
    }


# ══════════════════════════════════════════════════════════
# STEP 2 — FEATURE EXTRACTION  (on masked leaf only)
# ══════════════════════════════════════════════════════════

def extract_features(img_bgr: np.ndarray, bg: dict) -> dict:
    """
    Extract color and texture features from the segmented leaf area only.
    Background pixels are excluded from every calculation.

    Features:
      brown_ratio   — % of leaf pixels that are brown (Early Blight spots)
      dark_ratio    — % of leaf pixels that are dark/necrotic (Late Blight)
      yellow_ratio  — % of leaf pixels that are yellow (chlorosis)
      green_ratio   — % of leaf pixels that are healthy green
      mean_sat      — mean HSV saturation within leaf
      homogeneity   — GLCM texture homogeneity (lower → more textured → more spots)
      contrast      — GLCM texture contrast
      coverage_pct  — leaf area / total image area
    """
    if not bg["found"]:
        return {k: 0.0 for k in
                ("brown_ratio", "dark_ratio", "yellow_ratio", "green_ratio",
                 "mean_sat", "homogeneity", "contrast", "coverage_pct")}

    leaf_mask = bg["leaf_mask"]
    leaf_area = max(bg["leaf_area"], 1)
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    def _in_leaf(mask_bgr):
        """Restrict a color mask to leaf pixels only."""
        return cv2.bitwise_and(mask_bgr, leaf_mask)

    # Brown spots (Early Blight) — raised saturation to exclude sandy soil
    b1 = cv2.inRange(hsv, np.array([10, 70, 50]), np.array([22, 255, 200]))
    b2 = cv2.inRange(hsv, np.array([0,  70, 50]), np.array([10, 255, 200]))
    brown_px    = _in_leaf(cv2.bitwise_or(b1, b2))
    brown_ratio = np.sum(brown_px > 0) / leaf_area * 100

    # Dark spots (Late Blight)
    dark_raw = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 60, 80]))
    dark_px  = cv2.morphologyEx(_in_leaf(dark_raw), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    dark_ratio = np.sum(dark_px > 0) / leaf_area * 100

    # Yellow chlorosis — raised saturation to exclude pale soil
    yellow_px   = _in_leaf(cv2.inRange(hsv, np.array([20, 60, 100]), np.array([35, 255, 255])))
    yellow_ratio = np.sum(yellow_px > 0) / leaf_area * 100

    # Healthy green
    green_px    = _in_leaf(cv2.inRange(hsv, np.array([25, 40, 40]), np.array([90, 255, 255])))
    green_ratio = np.sum(green_px > 0) / leaf_area * 100

    # Saturation within leaf
    sat_vals = hsv[:, :, 1][leaf_mask > 0]
    mean_sat = float(np.mean(sat_vals)) if len(sat_vals) > 0 else 0.0

    # GLCM texture on the leaf-masked grayscale image
    gray_leaf   = cv2.bitwise_and(gray, gray, mask=leaf_mask)
    glcm        = graycomatrix(gray_leaf, distances=[1, 3],
                               angles=[0, np.pi / 4, np.pi / 2],
                               levels=256, symmetric=True, normed=True)
    homogeneity = float(graycoprops(glcm, 'homogeneity').mean())
    contrast    = float(graycoprops(glcm, 'contrast').mean())

    return {
        "brown_ratio":   round(brown_ratio,  2),
        "dark_ratio":    round(dark_ratio,   2),
        "yellow_ratio":  round(yellow_ratio, 2),
        "green_ratio":   round(green_ratio,  2),
        "mean_sat":      round(mean_sat,     1),
        "homogeneity":   round(homogeneity,  4),
        "contrast":      round(contrast,     2),
        "coverage_pct":  bg["coverage_pct"],
        # Internal masks for visualization
        "_brown_mask": brown_px,
        "_dark_mask":  dark_px,
    }


# ══════════════════════════════════════════════════════════
# STEP 3 — CLASSIFICATION
# ══════════════════════════════════════════════════════════

def classify_leaf(features: dict) -> dict:
    """
    Score-based 3-class classification.
    Thresholds calibrated on 2,152 real leaf images.

    Score 0-1  → Healthy
    Score 2-4  → Suspicious
    Score 5+   → Sick
    """
    score = 0

    b = features["brown_ratio"]
    if   b > 15:  score += 3
    elif b > 5:   score += 2
    elif b > 1.5: score += 1

    y = features["yellow_ratio"]
    if   y > 15: score += 2
    elif y > 5:  score += 1

    s = features["mean_sat"]
    if   s < 75: score += 2
    elif s < 90: score += 1

    g = features["green_ratio"]
    if   g < 50: score += 2
    elif g < 65: score += 1

    if features["dark_ratio"] > 3:      score += 1
    if features["homogeneity"] < 0.33:  score += 1

    if score >= 5:
        label, color_bgr = "Sick",       (0,   0, 220)
    elif score >= 2:
        label, color_bgr = "Suspicious", (0, 140, 255)
    else:
        label, color_bgr = "Healthy",    (30, 180,  30)

    return {
        "label":        label,
        "score":        score,
        "color_bgr":    color_bgr,
        "brown_ratio":  features["brown_ratio"],
        "dark_ratio":   features["dark_ratio"],
        "yellow_ratio": features["yellow_ratio"],
        "green_ratio":  features["green_ratio"],
        "mean_sat":     features["mean_sat"],
    }


# ══════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════

def annotate_image(img_bgr: np.ndarray, bg: dict, features: dict, result: dict) -> np.ndarray:
    """
    Draw disease overlays and classification label on the original image.
    - Green contour: detected leaf boundary
    - Orange overlay: brown/early-blight spots
    - Purple overlay: dark/late-blight spots
    - Top bar: label + score
    """
    out = img_bgr.copy()

    # Orange overlay for brown spots
    overlay = np.zeros_like(out)
    overlay[features["_brown_mask"] > 0] = [0, 80, 200]
    cv2.addWeighted(out, 1.0, overlay, 0.5, 0, out)

    # Purple overlay for dark spots
    overlay2 = np.zeros_like(out)
    overlay2[features["_dark_mask"] > 0] = [160, 0, 160]
    cv2.addWeighted(out, 1.0, overlay2, 0.5, 0, out)

    # Leaf boundary
    cnts, _ = cv2.findContours(bg["leaf_mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (0, 230, 60), 2)

    # Label bar
    cv2.rectangle(out, (0, 0), (out.shape[1], 42), (20, 20, 20), -1)
    cv2.putText(out,
                f"{result['label']} | score: {result['score']}  "
                f"(brown:{result['brown_ratio']:.0f}%  green:{result['green_ratio']:.0f}%)",
                (5, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                result["color_bgr"], 2)
    return out


# ══════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════

def analyze_single(img_path: str, save_output: bool = True) -> dict:
    """
    Full pipeline for one image: background removal → features → classification.
    Optionally saves an annotated result image.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {img_path}")

    # Pipeline
    bg       = remove_background(img)
    features = extract_features(img, bg)
    result   = classify_leaf(features)

    print(f"  {Path(img_path).name:<30} → {result['label']:<12}  score={result['score']}"
          f"  (leaf={bg['coverage_pct']}%)")

    if save_output:
        annotated = annotate_image(img, bg, features, result)
        out_path  = Path(img_path).stem + "_classified.jpg"
        cv2.imwrite(str(out_path), annotated)
        print(f"    Saved: {out_path}")

    result["filename"]  = Path(img_path).name
    result["_bg"]       = bg
    result["_features"] = features
    return result


def batch_analyze(folder_path: str, output_report: str = "report.png") -> list:
    """
    Analyze all images in a folder and generate a visual report.
    """
    folder = Path(folder_path)
    images = (sorted(folder.glob("*.JPG"))  + sorted(folder.glob("*.jpg")) +
              sorted(folder.glob("*.png"))  + sorted(folder.glob("*.jpeg")))

    print(f"\nFound {len(images)} images in: {folder_path}")
    results = []
    for img_path in images:
        try:
            results.append(analyze_single(str(img_path), save_output=False))
        except Exception as e:
            print(f"  ERROR {img_path.name}: {e}")

    if not results:
        print("No valid images found.")
        return results

    counts = {"Healthy": 0, "Suspicious": 0, "Sick": 0}
    for r in results:
        counts[r["label"]] += 1
    total = len(results)

    print(f"\n{'='*50}")
    print(f"  Summary ({total} images):")
    print(f"  Healthy:    {counts['Healthy']}  ({counts['Healthy'] / total * 100:.0f}%)")
    print(f"  Suspicious: {counts['Suspicious']}  ({counts['Suspicious'] / total * 100:.0f}%)")
    print(f"  Sick:       {counts['Sick']}  ({counts['Sick'] / total * 100:.0f}%)")
    print(f"{'='*50}")

    _create_report(results, counts, output_report)
    print(f"\nReport saved: {output_report}")
    return results


def _create_report(results, counts, output_path):
    """Visual report: sample images per class + pie chart + score histogram."""
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor('#1a1a2e')

    ax_pie = fig.add_axes([0.0, 0.55, 0.22, 0.40])
    ax_pie.set_facecolor('#16213e')
    labels = ["Healthy", "Suspicious", "Sick"]
    colors = ["#27ae60", "#f39c12", "#e74c3c"]
    sizes  = [counts[l] for l in labels]
    ax_pie.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%',
               textprops={'color': 'white', 'fontsize': 9}, startangle=90)
    ax_pie.set_title("Classification\nDistribution", color='white',
                     fontsize=10, fontweight='bold')

    ax_hist = fig.add_axes([0.0, 0.07, 0.22, 0.40])
    ax_hist.set_facecolor('#16213e')
    scores = [r["score"] for r in results]
    n, bins, patches = ax_hist.hist(scores, bins=range(0, 12),
                                    color='#3498db', edgecolor='#16213e')
    for patch, left in zip(patches, bins):
        if left >= 5:   patch.set_facecolor('#e74c3c')
        elif left >= 2: patch.set_facecolor('#f39c12')
        else:           patch.set_facecolor('#27ae60')
    ax_hist.set_xlabel("Disease Score", color='white', fontsize=9)
    ax_hist.set_ylabel("Count",         color='white', fontsize=9)
    ax_hist.set_title("Score Distribution", color='white', fontsize=10, fontweight='bold')
    ax_hist.tick_params(colors='white')
    ax_hist.axvline(2, color='#f39c12', linestyle='--', linewidth=1.2, label='Suspicious')
    ax_hist.axvline(5, color='#e74c3c', linestyle='--', linewidth=1.2, label='Sick')
    ax_hist.legend(fontsize=7, labelcolor='white', framealpha=0.3)

    class_samples = {"Healthy": [], "Suspicious": [], "Sick": []}
    for r in results:
        if len(class_samples[r["label"]]) < 3:
            class_samples[r["label"]].append(r)

    row_labels = ["Healthy", "Suspicious", "Sick"]
    row_colors = ["#27ae60", "#f39c12", "#e74c3c"]

    for row, (cls, color_hex) in enumerate(zip(row_labels, row_colors)):
        samples = class_samples[cls]
        for col, r in enumerate(samples[:3]):
            left = 0.25 + col * 0.245
            bot  = 0.65 - row * 0.31
            ax   = fig.add_axes([left, bot, 0.22, 0.28])
            ax.set_facecolor('#16213e')
            img = cv2.imread(r.get("filename", ""))
            if img is None and "_bg" in r:
                img = r["_bg"].get("leaf_img")
            if img is not None:
                annotated = annotate_image(img, r["_bg"], r["_features"], r)
                ax.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            ax.set_title(
                f"Score: {r['score']} | brown: {r['brown_ratio']:.0f}%  "
                f"green: {r['green_ratio']:.0f}%",
                color=color_hex, fontsize=7.5, pad=2
            )
            ax.axis('off')
            for sp in ax.spines.values():
                sp.set_edgecolor(color_hex)
                sp.set_linewidth(2)
        if samples:
            fig.text(0.235, 0.65 - row * 0.31 + 0.14, cls,
                     color=color_hex, fontsize=11, fontweight='bold',
                     va='center', rotation=90)

    fig.suptitle("Potato Disease Detection — Background-Removed Classification Report",
                 color='white', fontsize=14, fontweight='bold', y=1.0)
    plt.savefig(output_path, dpi=140, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()


# ══════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        path = sys.argv[1]
        if Path(path).is_dir():
            batch_analyze(path, output_report="disease_report.png")
        else:
            analyze_single(path)
    else:
        print("Usage:")
        print("  python potato_disease_detector_v2.py <image.jpg>")
        print("  python potato_disease_detector_v2.py <folder/>")
