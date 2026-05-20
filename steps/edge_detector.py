import numpy as np
from scipy.ndimage import convolve
from collections import deque


def GaussSmoothing(I, N, Sigma):
    if N % 2 == 0:
        N += 1
    half = N // 2
    x, y = np.mgrid[-half:half + 1, -half:half + 1]
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * Sigma ** 2))
    kernel /= kernel.sum()
    return convolve(I.astype(np.float64), kernel, mode='reflect')


def ImageGradient(S):
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float64)
    Ky = np.array([[ 1,  2,  1],
                   [ 0,  0,  0],
                   [-1, -2, -1]], dtype=np.float64)

    Gx = convolve(S, Kx, mode='reflect')
    Gy = convolve(S, Ky, mode='reflect')

    Mag = np.sqrt(Gx ** 2 + Gy ** 2)
    Theta = np.degrees(np.arctan2(Gy, Gx)) % 180
    return Mag, Theta


def FindThreshold(Mag, percentageOfNonEdge):
    sorted_vals = np.sort(Mag.flatten())
    idx = int(np.clip(percentageOfNonEdge * len(sorted_vals), 0, len(sorted_vals) - 1))
    T_high = sorted_vals[idx]
    T_low = 0.5 * T_high
    return T_low, T_high


def NonmaximaSupress(Mag, Theta):
    rows, cols = Mag.shape
    Mag_nms = np.zeros_like(Mag)

    angle = Theta % 180
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            ang = angle[r, c]
            curr = Mag[r, c]
            if (0 <= ang < 22.5) or (157.5 <= ang < 180):
                n1, n2 = Mag[r, c - 1], Mag[r, c + 1] # 0°  horizontal
            elif 22.5 <= ang < 67.5:
                n1, n2 = Mag[r - 1, c + 1], Mag[r + 1, c - 1] # 45° diagonal
            elif 67.5 <= ang < 112.5:
                n1, n2 = Mag[r - 1, c], Mag[r + 1, c] # 90° vertical
            else:
                n1, n2 = Mag[r - 1, c - 1], Mag[r + 1, c + 1] # 135° diagonal
            if curr >= n1 and curr >= n2:
                Mag_nms[r, c] = curr

    return Mag_nms


def EdgeLinking(Mag_nms, T_low, T_high):
    rows, cols = Mag_nms.shape
    strong = Mag_nms >= T_high
    weak = (Mag_nms >= T_low) & ~strong

    E = np.zeros((rows, cols), dtype=bool)
    visited = np.zeros((rows, cols), dtype=bool)

    queue = deque()
    for r, c in np.argwhere(strong):
        visited[r, c] = True
        E[r, c] = True
        queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if not visited[nr, nc] and weak[nr, nc]:
                        visited[nr, nc] = True
                        E[nr, nc] = True
                        queue.append((nr, nc))

    return E


def canny_edge_detector(gray_img, N=5, Sigma=1.0, percentageOfNonEdge=0.90):
    I = gray_img.astype(np.float64)
    S = GaussSmoothing(I, N, Sigma)
    Mag, Theta = ImageGradient(S)
    T_low, T_high = FindThreshold(Mag, percentageOfNonEdge)
    Mag_nms = NonmaximaSupress(Mag, Theta)
    edges = EdgeLinking(Mag_nms, T_low, T_high)
    return edges
