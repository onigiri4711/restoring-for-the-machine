"""Image degradations. Convention: float32 RGB in [0, 1], shape (H, W, 3).
Every function takes and returns float images in [0, 1]; degrade() clips once at the end."""
import numpy as np
import cv2


def add_gaussian_noise(img, sigma, rng):
    """Additive white Gaussian noise. `sigma` is on the 0-255 scale (e.g. 15, 25, 50)."""
    # Zero-mean Gaussian noise with the same shape as the image. "White" = each value is drawn
    # independently of its neighbours (flat power spectrum). sigma is given on the 0-255 scale,
    # so divide by 255 to match our [0, 1] images. Working in float also avoids uint8 overflow.
    noise = rng.normal(0.0, sigma / 255.0, img.shape)
    return img + noise                       # no clipping here: degrade() clips once


def add_poisson_noise(img, peak, rng):
    """Shot noise: y = Poisson(peak * x) / peak. Lower `peak` = fewer photons = noisier."""
    # peak * x = expected photon count per pixel; sampling a Poisson count models the randomness
    # of photon arrival. Dividing by peak maps it back to [0, 1]. Var[y] = x / peak, so the noise
    # depends on the signal: SNR = sqrt(peak * x), which is worse in dark regions.
    return rng.poisson(peak * img) / peak


def add_salt_pepper(img, amount, rng):
    """Set a fraction `amount` of pixels to black (0) or white (1), equal odds, all three channels together."""
    # Per-PIXEL corruption (all 3 channels together), which is how dead/stuck sensor pixels behave.
    # A 2-D mask indexes the first two axes, so sp_noisy[mask] = 1.0 sets R, G and B at once.
    sp_noisy = img.copy()
    h, w = img.shape[:2]
    corrupt = rng.random((h, w)) < amount                     # which pixels get corrupted
    salt = rng.integers(0, 2, size=(h, w)).astype(bool)       # half white, half black
    sp_noisy[corrupt & salt] = 1.0
    sp_noisy[corrupt & ~salt] = 0.0
    return sp_noisy


def gaussian_blur(img, sigma, rng=None):
    """Gaussian blur with std `sigma` pixels, reflect borders."""
    # Kernel size (0, 0) lets OpenCV choose it from sigma (about 6*sigma covers +-3 sigma).
    # Reflect borders so the image edges don't darken.
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT)


def jpeg_compress(img, quality, rng=None):
    """Round-trip through JPEG at `quality` (1-100). cv2 wants uint8 BGR in, gives uint8 BGR out."""
    # The one place we leave our convention: OpenCV's JPEG codec needs uint8 BGR,
    # so convert float RGB [0, 1] -> uint8 BGR, encode/decode, then convert back.
    u8_bgr = (np.clip(img, 0, 1) * 255).round().astype(np.uint8)[:, :, ::-1]
    ok, buf = cv2.imencode(".jpg", u8_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return decoded[:, :, ::-1].astype(np.float32) / 255.0


def to_grayscale(img, method="luma", rng=None):
    """3-channel grey image. method: 'luma' (BT.601 weights) or 'average'."""
    r, g, b = img[..., 0], img[..., 1], img[..., 2]          # RGB order
    if method == "luma":
        grey = 0.299 * r + 0.587 * g + 0.114 * b             # BT.601: eye is most sensitive to green
    elif method == "average":
        grey = (r + g + b) / 3.0
    else:
        raise ValueError(f"unknown method: {method}")
    return np.stack([grey, grey, grey], axis=-1)              # back to 3 channels


def color_cast(img, strength, rng=None):
    """White-balance error: gains (1 + s, 1, 1 - s) on (R, G, B). strength=0 -> identity."""
    gains = np.array([1 + strength, 1.0, 1 - strength], dtype=np.float32)   # RGB order: warmer image
    return img * gains                                         # broadcasts over the channel axis


def low_light(img, level, rng):
    """Darken + gamma, then add shot and read noise. level=0 -> identity, level->1 -> very dark and noisy."""
    # 1. darken + gamma (level 0 -> gain 1, gamma 1 -> identity).
    #    Model: gain * x**gamma (a gamma curve, then an exposure scale).
    gain = 1.0 - 0.8 * level
    gamma = 1.0 + 2.0 * level
    x = gain * np.power(img, gamma)
    if level == 0:
        return x                                               # no noise at level 0 -> exact identity

    # 2. shot noise: fewer photons as level rises
    peak = 200.0 * (1.0 - level) + 5.0
    x = rng.poisson(x * peak) / peak

    # 3. read noise: signal-independent Gaussian, grows with level (already on the [0, 1] scale)
    read_sigma = 0.01 + 0.04 * level
    return x + rng.normal(0.0, read_sigma, x.shape)


# name -> (function, severities from mild to severe)
DEGRADATIONS = {
    "gaussian":    (add_gaussian_noise, [5, 15, 25, 35, 50]),
    "poisson":     (add_poisson_noise,  [100, 50, 20, 10, 5]),
    "salt_pepper": (add_salt_pepper,    [0.01, 0.03, 0.05, 0.1, 0.2]),
    "blur":        (gaussian_blur,      [0.5, 1.0, 1.5, 2.0, 3.0]),
    "jpeg":        (jpeg_compress,      [90, 70, 50, 30, 10]),
    "color_cast":  (color_cast,         [0.1, 0.2, 0.3, 0.4, 0.5]),
    "low_light":   (low_light,          [0.2, 0.4, 0.6, 0.8, 0.9]),
}


def degrade(img, name, severity, rng):
    """The one entry point other notebooks use. Clips to [0, 1] once, here."""
    fn, _ = DEGRADATIONS[name]
    out = fn(img.astype(np.float32), severity, rng=rng)
    return np.clip(out, 0, 1).astype(np.float32)