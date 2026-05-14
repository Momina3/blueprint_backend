import cv2
import numpy as np


def validate_blueprint(image_path):
    try:
        img = cv2.imread(image_path)

        if img is None:
            return False, "Unreadable image"

        h, w, _ = img.shape

        # ============================================
        # Basic size check
        # ============================================
        if h < 300 or w < 300:
            return False, "Image too small"

        score = 0

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ============================================
        # 1. White / light background ratio
        # ============================================
        white_ratio = np.sum(gray > 210) / gray.size

        if white_ratio > 0.45:
            score += 2
        elif white_ratio > 0.30:
            score += 1

        # ============================================
        # 2. Detect edges + straight lines
        # ============================================
        edges = cv2.Canny(gray, 50, 150)

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=60,
            minLineLength=40,
            maxLineGap=10
        )

        line_count = 0 if lines is None else len(lines)

        if line_count > 50:
            score += 2
        elif line_count > 20:
            score += 1

        # ============================================
        # 3. Horizontal / Vertical line dominance
        # ============================================
        hv_lines = 0

        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l[0]

                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

                if angle < 10 or angle > 80:
                    hv_lines += 1

        if hv_lines > 25:
            score += 2
        elif hv_lines > 12:
            score += 1

        # ============================================
        # 4. Reject colorful real photos softly
        # ============================================
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = np.mean(hsv[:, :, 1])

        if saturation < 90:
            score += 1
        elif saturation > 140:
            score -= 1

        # ============================================
        # 5. Texture / noise check
        # ============================================
        lap = cv2.Laplacian(gray, cv2.CV_64F).var()

        if lap < 900:
            score += 1
        elif lap > 1800:
            score -= 1

        # ============================================
        # Debug prints
        # ============================================
        print("----- BLUEPRINT VALIDATION -----")
        print("Resolution   :", w, "x", h)
        print("White Ratio  :", round(white_ratio, 3))
        print("Lines Found  :", line_count)
        print("HV Lines     :", hv_lines)
        print("Saturation   :", round(saturation, 2))
        print("Texture Var  :", round(lap, 2))
        print("Final Score  :", score)
        print("--------------------------------")

        # ============================================
        # Final decision
        # ============================================
        if score >= 5:
            return True, "Blueprint detected"

        elif score >= 3:
            return True, "Low confidence blueprint accepted"

        return False, "Please upload blueprint / floorplan image"

    except Exception as e:
        return False, str(e)