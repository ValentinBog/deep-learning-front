from __future__ import annotations
import os, sys, csv, random, traceback
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2

# ====== AJUSTA AQUÍ TU RUTA RAÍZ ======
# Estructura esperada:
# ROOT/
#   F0/0-1/...imagenes...
#   F1/...
#   ...
ROOT = r"C:\Proyectos\FSF\archive\archive\Dataset\Dataset"

# Clases de entrada y sufijo de salida
IN_CLASSES = ["F0", "F1", "F2", "F3", "F4"]
OUT_SUFFIX = "N"                 # F0 -> F0N
SAVE_EXT   = ".png"              # extensión de salida

# Preproceso
REMOVE_OVERLAYS_TOP    = 0.05    # recorte superior (para overlays globales)
REMOVE_OVERLAYS_BOTTOM = 0.05    # recorte inferior
SECTOR_MIN_AREA_RATIO  = 0.25    # área mínima del sector respecto a la imagen
P_LOW, P_HIGH          = 1, 99   # percentiles normalización robusta
USE_CLAHE              = False   # activar si hay contraste muy variable
CLAHE_CLIP             = 1.5
CLAHE_TILE             = 8
FINAL_SIZE             = 256     # salida cuadrada para modelos
# Limpieza de anotaciones
BAND_TOP_PAD           = 10
BAND_SIDE_PAD          = 14
TOP_FRAC_INPAINT       = 0.26

# Extensiones admitidas (incluye mayúsculas)
IMG_EXTS = {
    ".png",".jpg",".jpeg",".bmp",".tif",".tiff",".dcm",
    ".PNG",".JPG",".JPEG",".BMP",".TIF",".TIFF",".DCM"
}

# Barras de progreso y DICOM (opcionales)
try:
    from tqdm.notebook import tqdm
    HAVE_TQDM = True
except Exception:
    HAVE_TQDM = False

try:
    import pydicom
    HAVE_PYDICOM = True
except Exception:
    HAVE_PYDICOM = False

ROOT = Path(ROOT)
ROOT.mkdir(parents=True, exist_ok=True)
print("ROOT:", ROOT)


def _win_extended_path(p: Path) -> str:
    """
    En Windows, añade \\?\ para soportar rutas largas/Unicode.
    En otros SO, devuelve str normal.
    """
    s = str(p)
    if os.name == "nt":
        s = os.path.normpath(s)
        if not s.startswith("\\\\?\\"):
            s = "\\\\?\\" + s
    return s

def _imread_robust_gray(p: Path):
    """
    Lectura robusta en GRIS:
    1) cv2.imread (rápido)
    2) np.fromfile + cv2.imdecode (soporta rutas largas/unicode)
    """
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        return img.astype(np.float32)
    try:
        xp = _win_extended_path(p)
        data = np.fromfile(xp, dtype=np.uint8)
        if data.size > 0:
            img2 = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
            if img2 is not None:
                return img2.astype(np.float32)
    except Exception:
        pass
    return None

def imwrite_robust(dst_path: Path, img_u8: np.ndarray) -> bool:
    """
    Escritura robusta: imencode -> tofile con \\?\ en Windows.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if not img_u8.flags['C_CONTIGUOUS']:
        img_u8 = np.ascontiguousarray(img_u8)
    if img_u8.dtype != np.uint8:
        raise ValueError("imwrite_robust espera img uint8")

    ext = (dst_path.suffix or ".png").lower()
    ok, buf = cv2.imencode(ext, img_u8)
    if not ok:
        return False
    try:
        xp = _win_extended_path(dst_path)
        buf.tofile(xp)
        return True
    except Exception:
        return False


def load_image_any(path: Path) -> np.ndarray:
    ext = path.suffix.lower()

    if ext == ".dcm":
        if not HAVE_PYDICOM:
            raise RuntimeError("pydicom no está instalado, pero se encontró un .dcm")
        ds = pydicom.dcmread(_win_extended_path(path))
        img = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        inter = float(getattr(ds, "RescaleIntercept", 0.0))
        img = img * slope + inter
        if getattr(ds, "PhotometricInterpretation", "").upper() == "MONOCHROME1":
            img = img.max() - img
        return img

    img = _imread_robust_gray(path)
    if img is None:
        try:
            size = path.stat().st_size
        except Exception:
            size = -1
        raise RuntimeError(f"No se pudo leer (ruta larga/unicode/corrupción). Tamaño={size} bytes: {path}")
    return img


def remove_overlays_by_cropping(img: np.ndarray, top: float=0.05, bottom: float=0.05):
    h, w = img.shape
    y0 = int(h*top); y1 = int(h*(1-bottom))
    if y1 <= y0: return img
    return img[y0:y1, :]

def _normalize01(img):
    img = img.astype(np.float32)
    mn, mx = float(img.min()), float(img.max())
    if mx <= mn: return np.zeros_like(img, np.float32)
    return (img - mn) / (mx - mn + 1e-6)

def _largest_contour_mask(img01, thr=0.02):
    u8 = (img01 > thr).astype(np.uint8)*255
    u8 = cv2.medianBlur(u8, 5)
    cnts,_ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.ones_like(u8)
    cnt = max(cnts, key=cv2.contourArea)
    mask = np.zeros_like(u8)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
    return mask

def _first_last_cols_rows(mask):
    H,W = mask.shape
    cols = np.where(mask.sum(axis=0) > H*0.25)[0]
    rows = np.where(mask.sum(axis=1) > W*0.25)[0]
    if cols.size==0: cols = np.array([0, W-1])
    if rows.size==0: rows = np.array([0, H-1])
    return int(cols.min()), int(cols.max()), int(rows.min()), int(rows.max())

def crop_ultrasound_sector(img: np.ndarray, min_area_ratio: float=0.25):
    norm = _normalize01(img)
    thr = (norm > 0.02).astype(np.uint8)*255
    thr = cv2.medianBlur(thr, 5)
    cnts,_ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        h, w = img.shape
        return img, np.ones((h,w), np.uint8)*255
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    h, w = img.shape
    if area < min_area_ratio * (h*w):
        return img, thr
    x,y,wc,hc = cv2.boundingRect(cnt)
    crop = img[y:y+hc, x:x+wc]
    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
    mask = mask[y:y+hc, x:x+wc]
    return crop, mask

def robust_minmax(img: np.ndarray, p_low=1, p_high=99):
    lo = np.percentile(img, p_low); hi = np.percentile(img, p_high)
    if hi <= lo:
        lo, hi = float(img.min()), float(img.max())
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + 1e-6)
    return img.astype(np.float32)

def optional_clahe(img01: np.ndarray, clip=1.5, tile=8):
    u8 = np.clip(img01*255.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    out = clahe.apply(u8).astype(np.float32) / 255.0
    return out

def resize_letterbox(img01: np.ndarray, size=256):
    h, w = img01.shape
    scale = min(size / w, size / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img01, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype=np.float32)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas


def clean_ultrasound_annotations(img_gray,
                                 BAND_TOP_PAD=10,
                                 BAND_SIDE_PAD=22,    # un poco más ancho
                                 TOP_FRAC_INPAINT=0.35):  # miramos más alto para atrapar la T
    """
    Limpia overlays de ecografía:
      - Zonas FUERA del sector -> negro (sin inpaint) para evitar 'barridos'
      - Dentro del sector: inpaint de texto/líneas finas en franja superior
      - Detección explícita de icono 'T' por template matching + CC
    Devuelve uint8.
    """
    import cv2, numpy as np

    # A float32 y [0,1]
    img = img_gray.astype(np.float32, copy=False)
    mn, mx = float(img.min()), float(img.max())
    if mx <= mn:
        img01 = np.zeros_like(img, np.float32)
    else:
        img01 = (img - mn) / (mx - mn + 1e-6)

    H, W = img01.shape

    # ---- máscara del sector (abanico) ----
    thr = (img01 > 0.02).astype(np.uint8) * 255
    thr = cv2.medianBlur(thr, 5)
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sector_mask = np.zeros_like(thr)
    if cnts:
        cnt = max(cnts, key=cv2.contourArea)
        cv2.drawContours(sector_mask, [cnt], -1, 255, thickness=cv2.FILLED)
    else:
        sector_mask[:] = 255

    # Límites del sector
    cols = np.where(sector_mask.sum(axis=0) > H * 0.25)[0]
    rows = np.where(sector_mask.sum(axis=1) > W * 0.25)[0]
    cL, cR = (int(cols.min()), int(cols.max())) if cols.size else (0, W - 1)
    rT, rB = (int(rows.min()), int(rows.max())) if rows.size else (0, H - 1)

    # ---- 1) Zonas fuera del sector -> negro (evita artefactos de inpaint) ----
    u8_img = np.clip(img01 * 255.0, 0, 255).astype(np.uint8)
    outside = (sector_mask == 0)
    u8_img[outside] = 0  # negro directo

    # ---- 2) Bandas de margen dentro/adyacentes al sector (top/left/right) ----
    band = np.zeros_like(sector_mask, np.uint8)
    top_end = max(0, rT - BAND_TOP_PAD)
    if top_end > 0: band[:top_end, :] = 255
    left_end = max(0, cL + BAND_SIDE_PAD)
    if left_end > 0: band[:, :left_end] = 255
    right_start = min(W - 1, cR - BAND_SIDE_PAD)
    if right_start < W - 1: band[:, right_start:] = 255

    # ---- 3) Franja superior dentro del sector: texto/puntos/T -> inpaint ----
    frac_h = int((rB - rT + 1) * TOP_FRAC_INPAINT)
    y0, y1 = max(0, rT), min(H, rT + frac_h)

    # trabajamos con ROI en uint8
    roi_mask = (sector_mask[y0:y1, :] > 0).astype(np.uint8)
    roi_u8   = (u8_img[y0:y1, :] * roi_mask).astype(np.uint8)

    # realzar altas frecuencias (median en uint8)
    med  = cv2.medianBlur(roi_u8, 7)
    hf16 = roi_u8.astype(np.int16) - med.astype(np.int16)
    hf   = np.clip(hf16, 0, 255).astype(np.uint8)

    nz = hf[hf > 0]
    t  = max(10, float(np.percentile(nz, 80)) if nz.size else 10.0)
    fine = (hf > t).astype(np.uint8) * 255
    fine = cv2.morphologyEx(
        fine, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1
    )

    # ---- 4) Detector explícito de 'T' (template matching + CC) ----
    # Kernel sencillo de T (blanco sobre negro), reescalado a varios tamaños
    base_T = np.array([
        [0,1,1,1,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
    ], dtype=np.uint8) * 255

    T_mask_total = np.zeros_like(fine)
    for k in (0.9, 1.1, 1.3, 1.6):
        Tk = cv2.resize(base_T, (int(base_T.shape[1]*k), int(base_T.shape[0]*k)),
                        interpolation=cv2.INTER_NEAREST)
        if Tk.size == 0: 
            continue
        res = cv2.matchTemplate(roi_u8, Tk, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= 0.60)  # umbral de similitud
        for (yy, xx) in zip(*loc):
            # caja del match
            yy2 = min(yy + Tk.shape[0], fine.shape[0])
            xx2 = min(xx + Tk.shape[1], fine.shape[1])
            T_mask_total[yy:yy2, xx:xx2] = 255

    # Afinar con CC para evitar falsos positivos muy grandes/pequeños
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(T_mask_total, connectivity=8)
    T_mask = np.zeros_like(T_mask_total)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w    = stats[i, cv2.CC_STAT_WIDTH]
        h    = stats[i, cv2.CC_STAT_HEIGHT]
        if 20 <= area <= 2000 and 6 >= w >= 4 and h >= 6:  # heurístico "T" pequeña
            T_mask[labels == i] = 255

    # ---- 5) Máscara final de inpaint ----
    inpaint_mask = band.copy()
    inpaint_mask[y0:y1, :] = np.maximum(inpaint_mask[y0:y1, :], fine)
    inpaint_mask[y0:y1, :] = np.maximum(inpaint_mask[y0:y1, :], T_mask)

    # Inpaint sólo en regiones dentro/adyacentes al sector
    cleaned = cv2.inpaint(u8_img, inpaint_mask, 3, cv2.INPAINT_TELEA)

    # fuera del sector nos aseguramos que siga negro
    cleaned[outside] = 0
    return cleaned


def preprocess_single_image(path: Path) -> np.ndarray:
    """
    Carga -> recorta overlays -> limpia anotaciones (T, métricas, texto)
    -> recorta sector -> normaliza -> (CLAHE) -> letterbox -> (H,W,1) float32 [0,1]
    """
    img = load_image_any(path)
    img = remove_overlays_by_cropping(img, top=REMOVE_OVERLAYS_TOP, bottom=REMOVE_OVERLAYS_BOTTOM)
    img = clean_ultrasound_annotations(img,
                                       BAND_TOP_PAD=BAND_TOP_PAD,
                                       BAND_SIDE_PAD=BAND_SIDE_PAD,
                                       TOP_FRAC_INPAINT=TOP_FRAC_INPAINT)
    img, _ = crop_ultrasound_sector(img, min_area_ratio=SECTOR_MIN_AREA_RATIO)
    img = robust_minmax(img, P_LOW, P_HIGH)
    if USE_CLAHE:
        img = optional_clahe(img, CLAHE_CLIP, CLAHE_TILE)
    img = resize_letterbox(img, FINAL_SIZE)
    return np.expand_dims(img, -1)  # (H,W,1) float32 [0,1]


ERROR_LOG = ROOT / "errores_lectura_o_escritura.csv"
if ERROR_LOG.exists():
    ERROR_LOG.unlink()

def append_error(row):
    header_needed = not ERROR_LOG.exists()
    with open(ERROR_LOG, "a", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        if header_needed:
            wr.writerow(["clase", "paciente", "path", "error"])
        wr.writerow(row)

def collect_images(folder: Path) -> List[Path]:
    files = []
    for ext in IMG_EXTS:
        files.extend(folder.rglob(f"*{ext}"))
    return sorted(set(files))

def process_nested_structure(root: Path,
                             classes: List[str],
                             out_suffix: str = "N") -> None:
    assert root.exists(), f"No existe ROOT: {root}"

    for cls in classes:
        in_cls  = root / cls
        out_cls = root / f"{cls}{out_suffix}"
        if not in_cls.exists():
            print(f"[ADVERTENCIA] Falta carpeta de clase: {in_cls} (se omite)")
            continue
        out_cls.mkdir(parents=True, exist_ok=True)

        # subcarpetas de pacientes (p.ej., '0-1', '0-2', etc.)
        patient_dirs = [d for d in in_cls.iterdir() if d.is_dir()]
        if not patient_dirs:
            print(f"[ADVERTENCIA] No se hallaron subcarpetas de pacientes en {in_cls}.")
            patient_dirs = [in_cls]

        it1 = tqdm(patient_dirs, desc=f"{cls}: pacientes") if HAVE_TQDM else patient_dirs

        for pdir in it1:
            rel_patient = pdir.name
            out_patient = out_cls / rel_patient
            out_patient.mkdir(parents=True, exist_ok=True)

            imgs = [p for p in pdir.glob("**/*") if p.suffix in IMG_EXTS]
            if not imgs:
                continue

            it2 = tqdm(imgs, desc=f"{cls}/{rel_patient}", leave=False) if HAVE_TQDM else imgs

            for src in it2:
                try:
                    img = preprocess_single_image(src)             # (H,W,1) float32 [0,1]
                    u8  = np.clip(img[...,0] * 255.0, 0, 255).astype(np.uint8)

                    try:
                        rel = src.relative_to(pdir)
                        save_path = (out_patient / rel).with_suffix(SAVE_EXT)
                    except Exception:
                        save_path = (out_patient / src.name).with_suffix(SAVE_EXT)

                    ok = imwrite_robust(save_path, u8)
                    if not ok:
                        raise RuntimeError("imwrite_robust no pudo escribir el archivo")
                except Exception as e:
                    print(f"[ERROR] {src}: {e}")
                    append_error([cls, rel_patient, str(src), str(e)])

# Ejecutar procesamiento por lotes
process_nested_structure(ROOT, IN_CLASSES, OUT_SUFFIX)
print("✅ Listo. Imágenes limpias en F0N..F4N con subcarpetas por paciente.")
if ERROR_LOG.exists():
    print("⚠️ Revisa el log de errores:", ERROR_LOG)


import os, glob, random
import cv2
import matplotlib.pyplot as plt

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def show_random_before_after(root_dir, class_prefix, out_suffix="N", n_samples=20, verbose_missing=15):
    """
    Muestra pares aleatorios antes/después.
    Estructura esperada:
        root_dir/F0/...         (originales)
        root_dir/F0N/...        (procesadas)
    Coincidencia por:
        1) misma ruta relativa
        2) mismo nombre dentro del mismo paciente (búsqueda recursiva)
        3) mismo 'stem' dentro del mismo paciente (por si cambia la extensión)
    """
    base_path   = os.path.join(root_dir, class_prefix)           # p.ej. .../F0
    output_base = os.path.join(root_dir, class_prefix + out_suffix)  # p.ej. .../F0N

    if not os.path.isdir(base_path):
        print(f"❌ No existe carpeta originales: {base_path}")
        return
    if not os.path.isdir(output_base):
        print(f"❌ No existe carpeta procesadas: {output_base}")
        return

    pairs, missing = [], []

    for subdir, _, files in os.walk(base_path):
        for fname in files:
            if not fname.lower().endswith(IMG_EXTS):
                continue
            orig_path = os.path.join(subdir, fname)
            rel_path  = os.path.relpath(orig_path, base_path)      # p.ej. "0-1\\IM-0001-0007.jpg"

            # 1) intento: misma ruta relativa en F0N
            cand1 = os.path.join(output_base, rel_path)
            if os.path.exists(cand1):
                pairs.append((orig_path, cand1))
                continue

            # 2) si no, buscamos dentro del mismo paciente (primer directorio)
            parts = rel_path.split(os.sep)
            patient_dir = parts[0] if len(parts) > 1 else ""
            if not patient_dir:
                # si la imagen está directamente bajo F0
                search_root = output_base
            else:
                search_root = os.path.join(output_base, patient_dir)

            # 2a) mismo nombre exacto (por si solo cambió la subcarpeta)
            exact_matches = glob.glob(os.path.join(search_root, "**", fname), recursive=True)
            if exact_matches:
                pairs.append((orig_path, exact_matches[0]))
                continue

            # 2b) mismo "stem" (IM-xxx) con cualquier extensión
            stem = os.path.splitext(fname)[0]
            glob_matches = glob.glob(os.path.join(search_root, "**", stem + ".*"), recursive=True)
            # Filtra a extensiones de imagen
            glob_matches = [p for p in glob_matches if os.path.splitext(p)[1].lower() in IMG_EXTS]
            if glob_matches:
                pairs.append((orig_path, glob_matches[0]))
            else:
                missing.append(rel_path)

    if not pairs:
        print("⚠️ No se encontraron pares de imágenes antes/después.")
        if missing:
            print("Ejemplos sin match (hasta {verbose_missing}):")
            for r in missing[:verbose_missing]:
                print("  -", r)
        return

    print(f"✅ Encontrados {len(pairs)} pares. Mostrando {min(n_samples, len(pairs))} aleatorios.")
    samples = random.sample(pairs, min(n_samples, len(pairs)))

    plt.figure(figsize=(12, len(samples)*2))
    for i, (before_path, after_path) in enumerate(samples):
        bgr_before = cv2.imread(before_path)
        bgr_after  = cv2.imread(after_path)
        if bgr_before is None or bgr_after is None:
            continue
        before = cv2.cvtColor(bgr_before, cv2.COLOR_BGR2RGB)
        after  = cv2.cvtColor(bgr_after,  cv2.COLOR_BGR2RGB)

        plt.subplot(len(samples), 2, 2*i+1)
        plt.imshow(before); plt.axis("off")
        plt.title(f"ANTES: {os.path.basename(before_path)}", fontsize=9)

        plt.subplot(len(samples), 2, 2*i+2)
        plt.imshow(after); plt.axis("off")
        plt.title(f"DESPUÉS: {os.path.basename(after_path)}", fontsize=9)

    plt.tight_layout()
    plt.show()


from pathlib import Path
import numpy as np
import cv2, os
from tqdm import tqdm

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def list_images_any(root: Path):
    files = []
    for ext in IMG_EXTS:
        files.extend(root.rglob(f"*{ext}"))
    # únicos + ordenados
    return sorted({p for p in files if p.is_file()})

def imread_gray_robust(path: Path):
    """
    Lectura robusta en escala de grises que tolera rutas con acentos/espacios.
    """
    try:
        # Método robusto: np.fromfile + cv2.imdecode
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback normal
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        return img
    except Exception:
        return None


def detectar_sector(img_u8):
    """
    Estima el 'abanico' del ultrasonido: todo lo > 2 se considera activo,
    cierre + contorno mayor.
    """
    thr = (img_u8 > 2).astype(np.uint8) * 255
    thr = cv2.medianBlur(thr, 5)
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sector = np.zeros_like(img_u8, np.uint8)
    if cnts:
        cv2.drawContours(sector, [max(cnts, key=cv2.contourArea)], -1, 255, thickness=cv2.FILLED)
    return sector

def pseudomask_grabcut(img_u8, sector_mask, shrink=0.15, iters=6):
    """
    img_u8: uint8 (grayscale)
    sector_mask: 0/255 (uint8)
    Devuelve máscara binaria 0/255 (uint8).
    """
    # Asegurar uint8
    if img_u8.dtype != np.uint8:
        img_u8 = np.clip(img_u8, 0, 255).astype(np.uint8)
    if sector_mask.dtype != np.uint8:
        sector_mask = (sector_mask > 0).astype(np.uint8) * 255

    h, w = img_u8.shape
    # -> Convertir a BGR (grabCut requiere CV_8UC3)
    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)

    # Bounding del sector
    ys, xs = np.where(sector_mask > 0)
    if ys.size == 0:
        return np.zeros_like(img_u8, dtype=np.uint8)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    # Rectángulo interior probable de foreground
    dh = int((y1 - y0) * max(0.0, min(0.45, shrink)))
    dw = int((x1 - x0) * max(0.0, min(0.45, shrink)))
    x, y = max(x0 + dw, 0), max(y0 + dh, 0)
    rw = max(1, (x1 - x0) - 2 * dw)
    rh = max(1, (y1 - y0) - 2 * dh)

    # Inicialización de máscara para grabCut
    gc_mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)  # probable fondo
    gc_mask[sector_mask == 0] = cv2.GC_BGD              # fondo seguro
    gc_mask[y:y+rh, x:x+rw] = cv2.GC_PR_FGD             # foreground probable

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(img_bgr, gc_mask, (x, y, rw, rh), bgdModel, fgdModel, iters, cv2.GC_INIT_WITH_RECT)
    except cv2.error as e:
        # Fallback conservador: si grabCut falla, devolvemos sector_mask umbralado suave
        # (no es ideal, pero evita romper el batch)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fallback = cv2.morphologyEx(sector_mask, cv2.MORPH_CLOSE, k, iterations=1)
        return fallback

    # Binarizar salida (FGD o PR_FGD -> 1)
    mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    mask = cv2.bitwise_and(mask, sector_mask)

    # Limpieza ligera
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    mask = cv2.medianBlur(mask, 5)

    return mask


def generar_pseudoetiquetas_batch(origen_root: str, destino_root: str, clases=("F2N","F3N","F4N"), overwrite=False):
    origen_root  = Path(origen_root)
    destino_root = Path(destino_root)
    destino_root.mkdir(parents=True, exist_ok=True)

    for cls in clases:
        in_dir  = origen_root / cls
        out_dir = destino_root / cls
        if not in_dir.exists():
            print(f"⚠️ No existe carpeta de origen: {in_dir}")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)

        files = list_images_any(in_dir)
        print(f"📁 {cls}: encontrados {len(files)} archivos de imagen")

        for src in tqdm(files, desc=f"Generando pseudo-máscaras desde {cls}"):
            rel = src.relative_to(in_dir)
            dst = (out_dir / rel).with_suffix(".png")
            if dst.exists() and not overwrite:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)

            img = imread_gray_robust(src)
            if img is None:
                print(f"[WARN] No pude leer: {src}")
                continue
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)

            sector = detectar_sector(img)
            mask   = pseudomask_grabcut(img, sector, shrink=0.15, iters=6)

            # filtro por área relativa al sector (evitar máscaras vacías o saturadas)
            area_sector = int((sector > 0).sum())
            area_mask   = int((mask   > 0).sum())
            ratio = (area_mask / max(1, area_sector)) if area_sector else 0.0
            if ratio < 0.10 or ratio > 0.95:
                # marcar como 0 para revisión posterior
                mask[:] = 0

            ok = cv2.imwrite(str(dst), mask)
            if not ok:
                print(f"[WARN] No pude escribir: {dst}")

# Ejecuta aquí con tus rutas reales
ORIGEN  = r"C:\Proyectos\FSF\archive\archive\Dataset\Dataset"
DESTINO = r"C:\Proyectos\FSF\archive\archive\Dataset\Dataset\MasksPseudo"

# generar_pseudoetiquetas_batch(ORIGEN, DESTINO, clases=("F0N","F1N","F2N","F3N","F4N"), overwrite=False)


import os
if os.path.exists("splits_paths.json"):
    os.remove("splits_paths.json")
    print("splits_paths.json eliminado")



# ============================================================
# 1. Re-importar librerías necesarias
# ============================================================
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
import numpy as np
import cv2
from sklearn.model_selection import train_test_split


# ============================================================
# 2. Volver a definir el dataset (por si el kernel se reinició)
# ============================================================
class LiverUSMultiTaskDataset(Dataset):
    def __init__(self, 
                 images_root: str,
                 masks_root: str,
                 classes=("F0N","F1N","F2N","F3N","F4N"),
                 transforms=None):
        
        self.images_root = Path(images_root)
        self.masks_root  = Path(masks_root)
        self.classes     = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transforms  = transforms

        self.samples = []

        for cls in self.classes:
            img_dir  = self.images_root / cls
            mask_dir = self.masks_root  / cls

            if not img_dir.exists():
                print(f"⚠️ No existe carpeta: {img_dir}")
                continue

            for img_path in img_dir.rglob("*"):
                if img_path.suffix.lower() not in [".png",".jpg",".jpeg",".bmp",".tif",".tiff"]:
                    continue

                rel = img_path.relative_to(img_dir)
                mask_path = (mask_dir / rel).with_suffix(".png")

                if not mask_path.exists():
                    continue

                label_idx = self.class_to_idx[cls]
                self.samples.append((img_path, mask_path, label_idx))

        print(f"Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label_idx = self.samples[idx]

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        img = np.stack([img, img, img], axis=0)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)

        return torch.from_numpy(img), torch.from_numpy(mask), torch.tensor(label_idx, dtype=torch.long)


# ============================================================
# 3. Cargar el dataset
# ============================================================
IMAGES_ROOT = r"C:\Proyectos\FSF\archive\archive\Dataset\Dataset"
MASKS_ROOT  = r"C:\Proyectos\FSF\archive\archive\Dataset\Dataset\MasksPseudo"

dataset = LiverUSMultiTaskDataset(
    images_root=IMAGES_ROOT,
    masks_root=MASKS_ROOT,
    classes=("F0N","F1N","F2N","F3N","F4N")
)

labels = np.array([s[2] for s in dataset.samples])
indices = np.arange(len(dataset))


# ============================================================
# 4. HACER EL SPLIT ESTRATIFICADO SI NO EXISTE ARCHIVO (o recargar si existe)
# ============================================================
# import os

# if os.path.exists("splits_paths.json"):
#     print("Cargando splits desde archivo JSON...")

#     import json
#     with open("splits_paths.json", "r") as f:
#         splits = json.load(f)

#     # reconstruir índices
#     path_to_idx = { str(s[0]): i for i, s in enumerate(dataset.samples) }

#     train_idx = np.array([path_to_idx[p] for p in splits["train"]])
#     val_idx   = np.array([path_to_idx[p] for p in splits["val"]])
#     test_idx  = np.array([path_to_idx[p] for p in splits["test"]])

# else:
#     print("Generando splits estratificados...")

#     train_idx, temp_idx, y_train, y_temp = train_test_split(
#         indices,
#         labels,
#         test_size=0.30,
#         random_state=42,
#         stratify=labels
#     )

#     val_idx, test_idx, y_val, y_test = train_test_split(
#         temp_idx,
#         y_temp,
#         test_size=0.50,
#         random_state=42,
#         stratify=y_temp
#     )

#     # Guardar splits por PATHS para reproducibilidad robusta
#     train_paths = [str(dataset.samples[i][0]) for i in train_idx]
#     val_paths   = [str(dataset.samples[i][0]) for i in val_idx]
#     test_paths  = [str(dataset.samples[i][0]) for i in test_idx]

#     splits = {
#         "train": train_paths,
#         "val":   val_paths,
#         "test":  test_paths
#     }

#     import json
#     with open("splits_paths.json", "w") as f:
#         json.dump(splits, f)

#     print("Splits guardados en splits_paths.json ✔")


# print("Split sizes → Train:", len(train_idx), "Val:", len(val_idx), "Test:", len(test_idx))


# # ============================================================
# # 5. CREAR DATASETS Y LOADERS
# # ============================================================
# train_dataset = Subset(dataset, train_idx)
# val_dataset   = Subset(dataset, val_idx)
# test_dataset  = Subset(dataset, test_idx)

# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
# val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False)
# test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False)

# print("Dataloaders creados ✔")


# from sklearn.model_selection import train_test_split
# from torch.utils.data import Subset, DataLoader
# import numpy as np

# # Dataset completo
# dataset = LiverUSMultiTaskDataset(
#     images_root=IMAGES_ROOT,
#     masks_root=MASKS_ROOT,
#     classes=("F0N","F1N","F2N","F3N","F4N")
# )

# # Sacamos las etiquetas de cada muestra
# labels = np.array([s[2] for s in dataset.samples])  # s[2] es label_idx en 0..4
# indices = np.arange(len(dataset))

# # Proporciones
# train_frac = 0.7
# val_frac   = 0.15
# test_frac  = 0.15

# # 1) train vs temp (val+test) estratificado
# train_idx, temp_idx, y_train, y_temp = train_test_split(
#     indices,
#     labels,
#     test_size=(1.0 - train_frac),
#     random_state=42,
#     stratify=labels
# )

# # 2) val vs test estratificado dentro de temp
# val_size_rel = val_frac / (val_frac + test_frac)  # 0.5 si es 15/15
# val_idx, test_idx, y_val, y_test = train_test_split(
#     temp_idx,
#     y_temp,
#     test_size=(1.0 - val_size_rel),
#     random_state=42,
#     stratify=y_temp
# )

# print("Total:", len(dataset))
# print("Train:", len(train_idx), "Val:", len(val_idx), "Test:", len(test_idx))



import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp

device = "cuda" if torch.cuda.is_available() else "cpu"
device

import numpy as np
import cv2
from torch.utils.data import Subset, DataLoader

# ============================================================
# 1. Función genérica para validar una máscara de hígado
# ============================================================
def is_valid_liver_mask(mask, min_ratio=0.02, max_ratio=0.90):
    """
    Evalúa si una máscara binaria parece una segmentación razonable del hígado.

    Parámetros
    ----------
    mask : np.ndarray o torch.Tensor
        Máscara binaria (0/255 o 0/1), shape (H,W) o (1,H,W).
    min_ratio : float
        Proporción mínima de píxeles activos (área_mask / total_pixeles).
    max_ratio : float
        Proporción máxima de píxeles activos (para evitar máscaras casi todo blanco).

    Returns
    -------
    valid : bool
        True si la máscara se considera válida, False si es sospechosa (muy vacía o muy llena).
    ratio : float
        Porcentaje de píxeles activos en la máscara.
    """
    # Convertir torch → numpy si hace falta
    if "torch" in str(type(mask)):
        mask = mask.detach().cpu().numpy()

    mask = np.array(mask)

    # Quitar canal si viene como (1,H,W)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]

    # Normalizar a 0/1 si viene en 0/255
    if mask.dtype != np.bool_ and mask.max() > 1:
        mask_bin = (mask > 127).astype(np.uint8)
    else:
        mask_bin = (mask > 0).astype(np.uint8)

    total = mask_bin.size
    area_mask = np.count_nonzero(mask_bin)
    ratio = area_mask / max(1, total)

    valid = (ratio >= min_ratio) and (ratio <= max_ratio)
    return valid, ratio

def validate_inference_image_from_mask(pred_mask, min_ratio=0.02, max_ratio=0.90, verbose=True):
    """
    Función pensada para INFERENCIA:
    Dado una máscara binaria predicha por tu modelo (UNet++),
    devuelve si la imagen parece válida como ultrasonido hepático
    según el área de la máscara.

    Úsalo así:
        seg_logits, cls_logits = model(img.unsqueeze(0))
        seg_prob = torch.sigmoid(seg_logits)
        seg_bin  = (seg_prob > 0.5).float()
        valid, ratio = validate_inference_image_from_mask(seg_bin[0])

    Returns
    -------
    valid : bool
    ratio : float
    """
    valid, ratio = is_valid_liver_mask(pred_mask, min_ratio=min_ratio, max_ratio=max_ratio)
    if verbose:
        if valid:
            print(f"✅ Imagen válida para hígado (ratio máscara = {ratio:.4f})")
        else:
            print(f"⚠️ Imagen sospechosa / fuera de patrón (ratio máscara = {ratio:.4f})")
    return valid, ratio


class UNetPP_VGG16_MultiTask(nn.Module):
    def __init__(self, 
                 num_classes_seg=1,    # segmentación binaria
                 num_classes_cls=5,    # F0..F4
                 encoder_name="vgg16",
                 encoder_weights="imagenet"):
        super().__init__()

        # UNet++ base
        self.unetpp = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,           # ya convertimos gray->3 canales en el Dataset
            classes=num_classes_seg,
            activation=None         # logits
        )

        self.encoder = self.unetpp.encoder
        self.decoder = self.unetpp.decoder
        self.seg_head = self.unetpp.segmentation_head

        # canales del bottleneck del encoder (último feature map)
        enc_channels = self.encoder.out_channels  # p.ej. [3,64,128,256,512,512]
        bottleneck_channels = enc_channels[-1]

        # cabeza de clasificación F0–F4
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bottleneck_channels, num_classes_cls)
        )

    def forward(self, x):
        # encoder: lista de feature maps
        features = self.encoder(x)
        bottleneck = features[-1]

        # decoder -> segmentación
        decoder_output = self.decoder(features)
        seg_logits = self.seg_head(decoder_output)  # (B,1,H,W)

        # clasificación a partir del bottleneck
        cls_logits = self.cls_head(bottleneck)      # (B,5)

        return seg_logits, cls_logits


model = UNetPP_VGG16_MultiTask().to(device)
print("Modelo listo en", device)


import cv2
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

def preprocess_image(path, size=256):
    # leer en escala de grises
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0

    # redimensionar
    img_resized = cv2.resize(img, (size, size))

    # convertir a 3 canales
    img_rgb = np.stack([img_resized, img_resized, img_resized], axis=0)

    # tensor
    return torch.tensor(img_rgb).float().unsqueeze(0)  # (1,3,H,W)

def infer_image(path):
    img_tensor = preprocess_image(path).to(device)

    with torch.no_grad():
        seg_logits, cls_logits = model(img_tensor)

        seg_prob = torch.sigmoid(seg_logits)[0,0].cpu().numpy()
        seg_bin  = (seg_prob > 0.5).astype(np.uint8)

        pred_class = cls_logits.argmax(dim=1).item()

    return seg_prob, seg_bin, pred_class


def mostrar_resultado(path):
    seg_prob, seg_bin, pred_class = infer_image(path)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.title("Imagen original")
    plt.imshow(img, cmap="gray")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("Probabilidad máscara")
    plt.imshow(seg_prob, cmap="jet")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title(f"Clasificación F{pred_class}")
    plt.imshow(seg_bin, cmap="gray")
    plt.axis("off")

    plt.show()


print("Funciones de inferencia listas ✔")


import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from tqdm.auto import tqdm

# --------------------------------------------------------
# Cargar el mejor modelo guardado
# --------------------------------------------------------
model = UNetPP_VGG16_MultiTask().to(device)
model.load_state_dict(torch.load("best_unetpp_vgg16_multitask.pth", map_location=device))
model.eval()

all_preds = []
all_labels = []

dice_scores = []  # opcional, para segmentación


def dice_coef(pred_mask, true_mask, eps=1e-6):
    """
    pred_mask y true_mask son arrays 0/1
    """
    inter = np.sum(pred_mask * true_mask)
    union = np.sum(pred_mask) + np.sum(true_mask)
    return (2 * inter + eps) / (union + eps)


# --------------------------------------------------------
# Recorrer test_loader
# --------------------------------------------------------
# with torch.no_grad():
#     for imgs, masks, labels in tqdm(test_loader, desc="Evaluando TEST"):
#         imgs   = imgs.to(device)
#         masks  = masks.to(device)
#         labels = labels.to(device)

#         seg_logits, cls_logits = model(imgs)

#         # ------------------------------
#         # CLASIFICACIÓN
#         # ------------------------------
#         preds = cls_logits.argmax(dim=1)

#         all_preds.extend(preds.cpu().numpy().tolist())
#         all_labels.extend(labels.cpu().numpy().tolist())

#         # ------------------------------
#         # DICE para segmentación (opcional)
#         # ------------------------------
#         seg_probs = torch.sigmoid(seg_logits)
#         seg_bin   = (seg_probs > 0.5).float()

#         seg_bin_np = seg_bin.cpu().numpy()
#         true_mask_np = masks.cpu().numpy()

#         for p_m, t_m in zip(seg_bin_np, true_mask_np):
#             dice_scores.append(dice_coef(p_m[0], t_m[0]))  # quitar canal C=1


# --------------------------------------------------------
# MÉTRICAS DE CLASIFICACIÓN
# --------------------------------------------------------
# print("===================================")
# print("🔍 Accuracy total en TEST:")
# test_accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
# print(f"{test_accuracy:.4f}")

# print("\n===================================")
# print("🔍 Reporte clasificación por clase:")
# print(classification_report(all_labels, all_preds, target_names=["F0","F1","F2","F3","F4"]))

# print("===================================")
# print("🔍 Matriz de confusión:")
# cm = confusion_matrix(all_labels, all_preds)
# print(cm)

# --------------------------------------------------------
# MÉTRICAS DE SEGMENTACIÓN
# --------------------------------------------------------
# print("\n===================================")
# print("🔍 Dice promedio (segmentación):")
# print(np.mean(dice_scores))



import os
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from collections import Counter

# Usaremos las mismas etapas F0..F4
CLASS_NAMES = ["F0", "F1", "F2", "F3", "F4"]

# ============================================================
# 1) Preprocesar UNA imagen usando TU pipeline completo
#    (resize, limpieza, sector, normalización, letterbox)
# ============================================================
def preprocess_for_model(img_path: Path) -> torch.Tensor:
    """
    Usa preprocess_single_image (ya definido en tu código) para:
      - carga robusta
      - recorte de overlays
      - limpieza de anotaciones (T, texto, métricas)
      - recorte del sector de ultrasonido
      - normalización robusta
      - (CLAHE opcional)
      - letterbox a FINAL_SIZE

    Devuelve tensor (1,3,H,W) float32 en [0,1] listo para el modelo.
    """
    img_hw1 = preprocess_single_image(img_path)          # (H,W,1) float32 [0,1]
    img_hwc = np.repeat(img_hw1, 3, axis=-1)             # (H,W,3)
    img_chw = np.transpose(img_hwc, (2, 0, 1))           # (3,H,W)
    tensor  = torch.from_numpy(img_chw).float().unsqueeze(0)  # (1,3,H,W)
    return tensor


# ============================================================
# 2) Inferencia para UNA imagen:
#    - preprocesar
#    - modelo → máscara + clase
#    - filtrar según máscara (validate_inference_image_from_mask)
# ============================================================
def infer_single_image_patient(
    img_path: Path,
    min_ratio: float = 0.02,
    max_ratio: float = 0.90,
    verbose: bool = False,
):
    """
    Pipeline completo por imagen:
      1) Preprocesa con preprocess_single_image
      2) Pasa por el modelo multitarea → seg_logits, cls_logits
      3) Binariza máscara y la filtra con validate_inference_image_from_mask
      4) Si la máscara es válida:
            devuelve (True, pred_idx, probs)
         Si NO:
            devuelve (False, None, None)
    """
    img_path = Path(img_path)

    # 1) Preprocesamiento
    try:
        img_tensor = preprocess_for_model(img_path).to(device)
    except Exception as e:
        if verbose:
            print(f"   [SKIP] No se pudo procesar {img_path.name}: {e}")
        return False, None, None

    # 2) Forward del modelo
    model.eval()
    with torch.no_grad():
        seg_logits, cls_logits = model(img_tensor)

        # 3) Máscara predicha
        seg_prob = torch.sigmoid(seg_logits)[0, 0].cpu().numpy()      # (H,W)
        seg_bin  = (seg_prob > 0.5).astype(np.uint8)                   # (H,W) 0/1

        # 4) Filtrar imagen según calidad de máscara (ya definido en tu código)
        valid, ratio = validate_inference_image_from_mask(
            seg_bin,
            min_ratio=min_ratio,
            max_ratio=max_ratio,
            verbose=verbose
        )

        if not valid:
            # Imagen considerada basura / fuera de patrón hepático
            return False, None, None

        # 5) Clasificación F0..F4
        probs = F.softmax(cls_logits, dim=1)[0].cpu().numpy()         # (5,)
        pred_idx = int(np.argmax(probs))

    return True, pred_idx, probs


# ============================================================
# 3) Inferencia para una CARPETA de un paciente:
#    - organiza imágenes
#    - aplica todo el pipeline
#    - agrega predicciones válidas y devuelve resumen
# ============================================================
def infer_patient_folder(
    folder_path: str,
    min_ratio: float = 0.02,
    max_ratio: float = 0.90,
    verbose: bool = True,
):
    """
    Dada una carpeta con imágenes de UN paciente:
      - Aplica tu pipeline completo (resize, limpieza, máscara, filtrado, modelo)
      - Devuelve:
          * total_images       : nº total de archivos de imagen encontrados
          * used_images        : nº de imágenes usadas (máscara válida)
          * discarded_images   : nº de imágenes descartadas por máscara mala
          * per_image_results  : lista de dicts con predicción por imagen
          * per_class_counts   : conteo de clases F0..F4 en imágenes válidas
          * patient_probs      : prob. promedio por clase (F0..F4)
          * patient_stage      : etapa de fibrosis final sugerida (F0..F4)
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"No existe la carpeta: {folder_path}")

    # extensiones aceptadas (bajamos a minúsculas)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm"}

    # imágenes dentro de la carpeta (no recursivo para evitar mezclar pacientes)
    image_paths = sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts],
        key=lambda p: p.name
    )

    if verbose:
        print(f"\n📂 Paciente: {folder}")
        print(f"   Imágenes encontradas: {len(image_paths)}")

    all_probs = []
    all_preds = []
    per_image_results = []
    used = 0
    discarded = 0

    for img_path in image_paths:
        if verbose:
            print(f" - Procesando {img_path.name}...")

        valid, pred_idx, probs = infer_single_image_patient(
            img_path,
            min_ratio=min_ratio,
            max_ratio=max_ratio,
            verbose=verbose
        )

        if not valid:
            discarded += 1
            if verbose:
                print("   → Imagen descartada (máscara no válida / imagen basura)")
            per_image_results.append({
                "filename": img_path.name,
                "used": False,
                "pred_class": None,
                "probs": None,
            })
            continue

        used += 1
        all_preds.append(pred_idx)
        all_probs.append(probs)

        per_image_results.append({
            "filename": img_path.name,
            "used": True,
            "pred_class": CLASS_NAMES[pred_idx],
            "probs": probs.tolist(),
        })

    total = len(image_paths)

    # Si no quedó ninguna imagen válida, devolvemos resumen vacío
    if used == 0:
        if verbose:
            print("\n⚠️ Ninguna imagen válida tras el filtrado. No se puede estimar fibrosis para este paciente.")
        return {
            "total_images": total,
            "used_images": used,
            "discarded_images": discarded,
            "per_image_results": per_image_results,
            "per_class_counts": {},
            "patient_probs": None,
            "patient_stage": None,
        }

    # Conteo por clase (solo imágenes válidas)
    counts = Counter(all_preds)
    per_class_counts = {CLASS_NAMES[k]: v for k, v in counts.items()}

    # Probabilidades promedio por clase (agregación suave)
    all_probs = np.stack(all_probs, axis=0)   # (N_valid, 5)
    mean_probs = all_probs.mean(axis=0)       # (5,)
    patient_idx = int(np.argmax(mean_probs))
    patient_stage = CLASS_NAMES[patient_idx]

    if verbose:
        print("\n✅ Resumen del paciente:")
        print(f"   Imágenes totales:     {total}")
        print(f"   Imágenes usadas:      {used}")
        print(f"   Imágenes descartadas: {discarded}")
        print(f"   Conteo por clase en imágenes válidas:")
        for cls, c in per_class_counts.items():
            print(f"      - {cls}: {c}")
        print(f"   Probabilidades promedio (F0..F4): {np.round(mean_probs, 3)}")
        print(f"   🔍 Etapa de fibrosis final sugerida: {patient_stage}")

    return {
        "total_images": total,
        "used_images": used,
        "discarded_images": discarded,
        "per_image_results": per_image_results,
        "per_class_counts": per_class_counts,
        "patient_probs": mean_probs,
        "patient_stage": patient_stage,
    }


# ============================================================
# EJEMPLO DE USO
# ============================================================
# Cambia la ruta por la carpeta del paciente que quieras evaluar.
# Por ejemplo, una subcarpeta dentro de F4N o F2N:
#
#   r"C:\Proyectos\FSF\archive\archive\Dataset\Dataset\F4N\4-22"
#
# resultado = infer_patient_folder(
#     r"C:\Proyectos\FSF\archive\archive\Dataset\Dataset\F4N\4-22",
#     min_ratio=0.02,
#     max_ratio=0.90,
#     verbose=True
# )
#
# print("\nEtapa final estimada para el paciente:", resultado["patient_stage"])


# ============================================================
# EJEMPLO DE USO
# ============================================================
# Cambia la ruta por la carpeta del paciente que quieras evaluar.
# Por ejemplo, una subcarpeta dentro de F4N o F2N:
#
#   r"C:\Proyectos\FSF\archive\archive\Dataset\Dataset\F4N\4-22"
#
resultado = infer_patient_folder(
     r"/home/valentin/Bureau/DEEP LEARNING PROJET/FRONTGIT/deep-learning-front/uploads/4-7",
     min_ratio=0.02,
     max_ratio=0.90,
     verbose=True
 )

print("\nEtapa final estimada para el paciente:", resultado["patient_stage"])