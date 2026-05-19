import cv2
import numpy as np


class BlueprintValidator:

    def __init__(self):
        self.min_score = 6

    def validate(self, image_path):
        img = cv2.imread(image_path)

        if img is None:
            return False, "Unreadable image", 0.0

        h, w, _ = img.shape

        if h < 300 or w < 300:
            return False, "Image too small", 0.0

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

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

        white_ratio = np.mean(gray > 210)
        saturation = np.mean(hsv[:, :, 1])
        texture_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        edge_density = np.sum(edges > 0) / edges.size

        entropy = self._entropy(gray)

        if saturation > 180 and texture_var > 2200:
            return False, "Real photo detected", 0.0

        if edge_density < 0.03:
            return False, "Too few edges", 0.0

        if white_ratio < 0.12:
            return False, "Not blueprint-like", 0.0

        hv_ratio = self._horizontal_vertical_ratio(lines)

        score = 0

        if white_ratio > 0.45:
            score += 2
        elif white_ratio > 0.30:
            score += 1

        if line_count > 80:
            score += 2
        elif line_count > 40:
            score += 1

        if hv_ratio > 0.5:
            score += 2
        elif hv_ratio > 0.3:
            score += 1

        if texture_var < 800:
            score += 2
        elif texture_var < 1500:
            score += 1
        elif texture_var > 2500:
            score -= 2

        if entropy > 7.5:
            score -= 2
        elif entropy < 6.5:
            score += 1

        confidence = min(score / 8.0, 1.0)

        if score >= self.min_score:
            return True, "Blueprint detected", confidence

        return False, "Not a valid blueprint", confidence

    def _entropy(self, gray):
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.sum()
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))

    def _horizontal_vertical_ratio(self, lines):
        if lines is None:
            return 0

        hv = 0
        total = len(lines)

        for l in lines:
            x1, y1, x2, y2 = l[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

            if angle < 10 or angle > 80:
                hv += 1

        return hv / total if total > 0 else 0


# =========================================================
# ✅ WRAPPER (PUT THIS OUTSIDE CLASS)
# =========================================================

validator = BlueprintValidator()

def validate_blueprint(image_path):
    return validator.validate(image_path)