import numpy as np
import cv2

BLACK_THRESH = 8

def hex_to_bgr(hex_color):
    h = hex_color.lstrip("#")
    return np.array([int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)], dtype=np.uint8)


def bgr_luminance(colors_bgr):
    # Relative luminance proxy in BGR order
    b = colors_bgr[:, 0].astype(np.float32)
    g = colors_bgr[:, 1].astype(np.float32)
    r = colors_bgr[:, 2].astype(np.float32)
    return 0.114 * b + 0.587 * g + 0.299 * r


def extend_palette(base_hex, target_n):
    base_bgr = np.array([hex_to_bgr(h) for h in base_hex], dtype=np.uint8)

    # Sort base palette by brightness so interpolation preserves lightness order
    lum = bgr_luminance(base_bgr)
    order = np.argsort(lum)
    base_bgr = base_bgr[order]

    n_base = len(base_bgr)
    if target_n <= n_base:
        # Evenly sample from sorted base colors if target is smaller/equal
        idx = np.linspace(0, n_base - 1, target_n).round().astype(int)
        return base_bgr[idx]

    # Interpolate in Lab for smoother perceptual extension
    base_lab = cv2.cvtColor(base_bgr.reshape(1, n_base, 3), cv2.COLOR_BGR2LAB).reshape(n_base, 3).astype(np.float32)
    x_base = np.linspace(0.0, 1.0, n_base)
    x_new = np.linspace(0.0, 1.0, target_n)

    new_lab = np.zeros((target_n, 3), dtype=np.float32)
    for c in range(3):
        new_lab[:, c] = np.interp(x_new, x_base, base_lab[:, c])

    new_bgr = cv2.cvtColor(new_lab.reshape(1, target_n, 3).astype(np.uint8), cv2.COLOR_LAB2BGR).reshape(target_n, 3)
    return new_bgr


def sample_source_colors(img_bgr, max_k=12, black_thresh=8):
    pixels_u8 = img_bgr.reshape(-1, 3)
    non_black_mask = np.any(pixels_u8 > black_thresh, axis=1)
    non_black_pixels = pixels_u8[non_black_mask]

    # If everything is black, return one black color
    if non_black_pixels.shape[0] == 0:
        return np.array([[0, 0, 0]], dtype=np.uint8)

    pixels = non_black_pixels.astype(np.float32)
    unique_colors = np.unique(non_black_pixels, axis=0)
    k = min(max_k, max(2, unique_colors.shape[0]))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.2)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )
    centers = centers.astype(np.uint8)

    lum = bgr_luminance(centers)
    order = np.argsort(lum)
    return centers[order]


def remap_to_palette_by_rank(img_bgr, base_palette_hex):
    # Preserve original black pixels
    black_mask = np.all(img_bgr <= BLACK_THRESH, axis=2)

    source_colors = sample_source_colors(img_bgr, max_k=12, black_thresh=BLACK_THRESH)
    n_source = source_colors.shape[0]

    n_target = n_source
    palette_colors = extend_palette(base_palette_hex, n_target)

    src_lum = bgr_luminance(source_colors)
    pal_lum = bgr_luminance(palette_colors)

    src_order = np.argsort(src_lum)
    pal_order = np.argsort(pal_lum)

    ranked_src = source_colors[src_order]
    ranked_pal = palette_colors[pal_order]

    if ranked_pal.shape[0] != ranked_src.shape[0]:
        idx = np.linspace(0, ranked_pal.shape[0] - 1, ranked_src.shape[0]).round().astype(int)
        ranked_pal = ranked_pal[idx]

    pixels = img_bgr.reshape(-1, 3).astype(np.int32)
    src = ranked_src.astype(np.int32)

    d2 = ((pixels[:, None, :] - src[None, :, :]) ** 2).sum(axis=2)
    nearest_idx = np.argmin(d2, axis=1)

    remapped_pixels = ranked_pal[nearest_idx]
    remapped = remapped_pixels.reshape(img_bgr.shape).astype(np.uint8)

    # Force original black pixels to remain black
    remapped[black_mask] = np.array([0, 0, 0], dtype=np.uint8)

    return remapped, ranked_src, ranked_pal