import cv2
import numpy as np

def analyze_screen_bezel_and_edges(image_bgr: np.ndarray, face_bbox: tuple) -> tuple:
    """
    Robust geometric & edge-profile detector for smartphone/tablet/monitor screens enclosing a face.
    Returns (detected: bool, confidence: float, bbox: Optional[tuple], face_enclosed: bool, details: str).
    """
    if face_bbox is None or image_bgr is None or image_bgr.size == 0:
        return False, 0.0, None, False, ""
        
    fx, fy, fw, fh = face_bbox
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Expand ROI around face to capture the potential device boundaries
    roi_x1 = max(0, int(fx - fw * 1.5))
    roi_y1 = max(0, int(fy - fh * 1.5))
    roi_x2 = min(w, int(fx + fw * 2.5))
    roi_y2 = min(h, int(fy + fh * 2.5))
    
    face_area = fw * fh
    frame_area = h * w
    
    # 2. Contour & Polygon Analysis with multiple adaptive thresholds
    for (th1, th2) in [(20, 60), (30, 90), (50, 150), (15, 45)]:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, th1, th2)
        
        # Connect rectangular segments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < face_area * 0.9 or area > frame_area * 0.98:
                continue
                
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            
            # Rectangular or rounded-corner rectangular profile (4 to 12 vertices)
            if 4 <= len(approx) <= 12:
                bx, by, bw, bh = cv2.boundingRect(approx)
                if bw <= 0 or bh <= 0:
                    continue
                    
                # Check aspect ratio (typical smartphones/tablets/screens: 1.2 to 2.8)
                aspect = float(bh) / float(bw) if bw > 0 else 0.0
                inv_aspect = float(bw) / float(bh) if bh > 0 else 0.0
                max_aspect = max(aspect, inv_aspect)
                
                if 1.15 <= max_aspect <= 3.2:
                    # Check overlap with face
                    ox1 = max(bx, fx)
                    oy1 = max(by, fy)
                    ox2 = min(bx + bw, fx + fw)
                    oy2 = min(by + bh, fy + fh)
                    
                    if ox2 > ox1 and oy2 > oy1:
                        overlap_area = (ox2 - ox1) * (oy2 - oy1)
                        if overlap_area / float(face_area) >= 0.50:
                            # Verify rectangular fill extent
                            rect_area = bw * bh
                            extent = float(area) / float(rect_area) if rect_area > 0 else 0.0
                            if extent >= 0.50:
                                conf = float(np.clip(0.60 + (extent * 0.35), 0.65, 0.95))
                                return True, conf, (bx, by, bw, bh), True, f"Physical device rectangular bezel enclosing face detected (aspect: {max_aspect:.2f}, extent: {extent:.2f})."
                                
    # 3. Vertical & Horizontal Edge Projection Profile Analysis (Flanking parallel borders)
    # When a phone is held vertically, strong vertical edges flank the face on left and right
    roi_gray = gray[roi_y1:roi_y2, roi_x1:roi_x2]
    if roi_gray.shape[0] > 40 and roi_gray.shape[1] > 40:
        sobel_v = np.abs(cv2.Sobel(roi_gray, cv2.CV_32F, 1, 0, ksize=3))
        # Sum along vertical columns to find strong vertical line borders
        v_proj = np.sum(sobel_v, axis=0) / (sobel_v.shape[0] + 1e-6)
        
        # Relative face coordinates in ROI
        r_fx1 = fx - roi_x1
        r_fx2 = fx + fw - roi_x1
        
        # Look for prominent peaks to the left of face and to the right of face
        left_region = v_proj[:max(1, r_fx1)]
        right_region = v_proj[min(len(v_proj) - 1, r_fx2):]
        
        if len(left_region) > 5 and len(right_region) > 5:
            left_peak = np.max(left_region)
            right_peak = np.max(right_region)
            mean_energy = np.mean(v_proj) + 1e-6
            
            left_ratio = left_peak / mean_energy
            right_ratio = right_peak / mean_energy
            
            if left_ratio > 3.0 and right_ratio > 3.0:
                conf = float(np.clip(0.55 + min(left_ratio, right_ratio) * 0.05, 0.60, 0.90))
                return True, conf, (roi_x1, roi_y1, roi_x2 - roi_x1, roi_y2 - roi_y1), True, f"Symmetric physical device edge borders flanking face detected (edge ratio: {left_ratio:.1f}/{right_ratio:.1f})."
                
    return False, 0.0, None, False, ""

# Test on synthetic phone
img = np.full((480, 640, 3), 30, dtype=np.uint8)
px, py, pw, ph = 220, 40, 200, 400
cv2.rectangle(img, (px, py), (px + pw, py + ph), (10, 10, 10), -1)
cv2.rectangle(img, (px, py), (px + pw, py + ph), (220, 220, 220), 4)
cv2.rectangle(img, (px + 6, py + 6), (px + pw - 6, py + ph - 6), (80, 80, 80), -1)
cv2.circle(img, (320, 240), 50, (190, 170, 150), -1)

det, conf, bbox, enc, details = analyze_screen_bezel_and_edges(img, (270, 190, 100, 100))
print("Synthetic phone test:", det, conf, bbox, enc, details)

# Test on nominal face
img_nom = np.full((480, 640, 3), 40, dtype=np.uint8)
cv2.circle(img_nom, (320, 240), 80, (180, 160, 140), -1)
det_nom, conf_nom, _, _, _ = analyze_screen_bezel_and_edges(img_nom, (240, 160, 160, 160))
print("Nominal face test:", det_nom, conf_nom)
