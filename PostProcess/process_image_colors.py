# ----------------------------------------------------------
# THEME COLOR EXTRACTOR (High Quality - Spotify/Apple level)
# ----------------------------------------------------------
from tqdm import tqdm
from rembg import remove
from PIL import Image
import io
import colorsys
from sklearn.cluster import KMeans
from skimage.color import deltaE_ciede2000, rgb2lab
import numpy as np
import os
import warnings
import glob
import json
import argparse

warnings.filterwarnings("ignore")


presets = {
    'David (Chapter 15)': [204, 148, 18]
}


# ----------------------------------------------------------
# Background Removal
# ----------------------------------------------------------

def remove_background(path):
    """
    Uses U2-Net (rembg) to remove background.
    Returns a PIL Image with RGBA.
    """
    with open(path, "rb") as f:
        data = f.read()
    output = remove(data)
    return Image.open(io.BytesIO(output))


# ----------------------------------------------------------
# Load useful pixels (exclude transparent)
# ----------------------------------------------------------

def load_pixels(img):
    """
    Converts to RGBA and removes transparent pixels 
    generated after segmentation.
    """
    img = img.convert("RGBA")
    arr = np.array(img)
    mask = arr[:, :, 3] > 10
    pixels = arr[:, :, :3][mask]
    return pixels.reshape(-1, 3)


# ----------------------------------------------------------
# KMeans Clustering
# ----------------------------------------------------------

def extract_clusters(pixels, k=10):
    """
    Runs high-quality KMeans to get color clusters.
    Returns (cluster_center, frequency).
    """
    kmeans = KMeans(n_clusters=k, n_init="auto")
    kmeans.fit(pixels)

    centers = kmeans.cluster_centers_.astype(int)
    counts = np.bincount(kmeans.labels_)

    freq = counts / counts.sum()
    return list(zip(centers, freq))


# ----------------------------------------------------------
# Color Utility Functions
# ----------------------------------------------------------

def luminance(c):
    r, g, b = c
    return 0.299 * r + 0.587 * g + 0.114 * b


def saturation(c):
    r, g, b = [x / 255 for x in c]
    return colorsys.rgb_to_hsv(r, g, b)[1]


def is_greyish(c, th=18):
    return max(c) - min(c) < th


def is_near_white(c):
    return luminance(c) > 235


def is_near_black(c):
    return luminance(c) < 25


# ----------------------------------------------------------
# Perceptual Delta-E (CIEDE2000)
# ----------------------------------------------------------

def cie2000(c1, c2):
    # convert RGB 0–255 → 0–1
    color1 = np.array([[c1]], dtype=np.uint8) / 255.0
    color2 = np.array([[c2]], dtype=np.uint8) / 255.0

    lab1 = rgb2lab(color1)
    lab2 = rgb2lab(color2)

    # deltaE returns a 1×1 array → extract float
    return float(deltaE_ciede2000(lab1, lab2))


# Background color to maximize contrast (for dark UI)
UI_BG = (15, 15, 15)


# ----------------------------------------------------------
# Color Scoring Formula
# ----------------------------------------------------------

def score_color(c, freq):
    """
    Scores color based on saturation, mid brightness,
    cluster frequency, and contrast with UI background.
    """
    lum = luminance(c)
    sat = saturation(c)
    contrast = cie2000(c, UI_BG)

    return (
            sat * 0.55 +  # vivid colors win
            (1 - abs(lum - 140) / 140) * 0.25 +  # mid brightness
            freq * 0.15 +  # importance in the image
            contrast * 0.05  # UI contrast
    )


# ----------------------------------------------------------
# Main Function: Best Theme Color
# ----------------------------------------------------------

def best_theme_color(path):
    # 1. Remove background
    img = remove_background(path)
    # print("Removed background")

    # 2. Extract meaningful pixels
    pixels = load_pixels(img)
    # print("Extract pixels")

    # 3. Cluster colors
    clusters = extract_clusters(pixels, k=10)
    # print("Custer colors")

    # 4. Filter & score colors
    candidates = []
    for c, freq in clusters:
        c = tuple(c)

        # rejection rules (high quality)
        if is_greyish(c): continue
        if is_near_white(c): continue
        if is_near_black(c): continue
        if saturation(c) < 0.15: continue
        if luminance(c) < 30 or luminance(c) > 225: continue

        score = score_color(c, freq)
        candidates.append((score, c))

    # print("Candidates found")

    # fallback: most common cluster
    if not candidates:
        clusters.sort(key=lambda x: x[1], reverse=True)
        return tuple(clusters[0][0])

    # return highest scoring
    return max(candidates)[1]


def extract_image_scheme(path):

    files = glob.glob(f"{path}/*.png")

    data = {}
    for file in tqdm(files, desc="Extracting"):
        try:
            filename = os.path.basename(file).split("/")[-1]
            image_preset = presets.get(filename.split("/")[-1])
            if image_preset:
                color = image_preset
            else:
                color = best_theme_color(file)
            # print(f"\n{file} : ", color)
            data[filename] = (int(color[0]), int(color[1]), int(color[2]))
        except Exception as e:
            print(f"\n{file} : {e}")

    with open(f"{path}/colorMap.json", 'w') as f:
        json.dump(data, f)

    print(f"Image extraction completed")


# ----------------------------------------------------------
# CLI
# ----------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ColorScheme")

    parser.add_argument("--path", type=str, default="Default", help="Image Paths")
    args = parser.parse_args()

    path = args.path
    extract_image_scheme(path)


