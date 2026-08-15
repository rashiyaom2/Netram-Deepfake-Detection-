import cv2
import numpy as np

def analyze_moire_frequency(image_bgr: np.ndarray, face_bbox: tuple) -> tuple:
    """
    Analyzes high-frequency periodic moire patterns caused by digital screen subpixel grids.
    Returns (detected: bool, confidence: float, details: str).
    """
    if face_bbox is None or image_bgr is None or image_bgr.size == 0:
        return False, 0.0, ""
    
    fx, fy, fw, fh = face_bbox
    h, w = image_bgr.shape[:2]
    
    # Clip ROI to image
    x1 = max(0, fx)
    y1 = max(0, fy)
    x2 = min(w, fx + fw)
    y2 = min(h, fy + fh)
    if x2 - x1 < 32 or y2 - y1 < 32:
        return False, 0.0, ""
        
    roi = image_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # High-pass filter via Laplacian to isolate high-frequency textures
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    
    # 2D FFT
    F = np.fft.fft2(lap)
    Fshift = np.fft.fftshift(F)
    mag = np.abs(Fshift)
    
    # Mask DC component (center)
    cy, cx = mag.shape[0] // 2, mag.shape[1] // 2
    r = min(cy, cx) // 4
    if r > 0:
        cv2.circle(mag, (cx, cy), r, 0, -1)
        
    # Periodic screen grids cause distinct, sharp high-frequency harmonic peaks
    peak_val = np.max(mag)
    mean_val = np.mean(mag) + 1e-6
    std_val = np.std(mag) + 1e-6
    papr = peak_val / mean_val  # Peak to Average Power Ratio
    
    # Also check high-frequency energy ratio
    high_freq_score = (peak_val - mean_val) / std_val
    
    # Synthetic / camera capture moire creates very high PAPR (> 18.0) or high_freq_score (> 8.0)
    if papr > 15.0 or high_freq_score > 7.5:
        conf = float(np.clip(0.50 + (papr / 80.0) * 0.45, 0.50, 0.95))
        return True, conf, f"High-frequency screen moire grid interference detected (PAPR: {papr:.1f})."
        
    return False, 0.0, ""

# Test synthetic moire
img = np.full((480, 640, 3), 50, dtype=np.uint8)
for y in range(160, 320, 4):
    img[y, 240:400] = 210
for x in range(240, 400, 4):
    img[160:320, x] = 210

det, conf, details = analyze_moire_frequency(img, (240, 160, 160, 160))
print("Moire test on synthetic grid:", det, conf, details)

# Test nominal face
img_nominal = np.full((480, 640, 3), 50, dtype=np.uint8)
cv2.circle(img_nominal, (320, 240), 75, (180, 160, 140), -1)
det_nom, conf_nom, _ = analyze_moire_frequency(img_nominal, (245, 165, 150, 150))
print("Moire test on nominal smooth face:", det_nom, conf_nom)
