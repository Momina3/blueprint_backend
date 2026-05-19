import cv2
import numpy as np


class BlueprintValidator:

    def __init__(self):
        # stricter threshold
        self.min_score = 7

    def validate(self, image_path):

        try:
            img = cv2.imread(image_path)

            if img is None:
                return False, "Unreadable image", 0.0

            h, w, _ = img.shape

            # reject tiny images
            if h < 300 or w < 300:
                return False, "Image too small", 0.0

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # edges
            edges = cv2.Canny(gray, 50, 150)

            # line detection
            lines = cv2.HoughLinesP(
                edges,
                1,
                np.pi / 180,
                threshold=60,
                minLineLength=40,
                maxLineGap=10
            )

            line_count = 0 if lines is None else len(lines)

            # features
            white_ratio = np.mean(gray > 210)
            saturation = np.mean(hsv[:, :, 1])
            texture_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            edge_density = np.mean(edges > 0)

            entropy = self._entropy(gray)

            hv_ratio = self._hv_ratio(lines)

            rectangle_count = self._rectangle_count(edges)

            score = 0

            # ----------------------------
            # WHITE BACKGROUND
            # ----------------------------
            if white_ratio > 0.50:
                score += 2
            elif white_ratio > 0.35:
                score += 1

            # ----------------------------
            # STRUCTURE / LINES
            # ----------------------------
            if line_count > 100:
                score += 2
            elif line_count > 50:
                score += 1

            # ----------------------------
            # HORIZONTAL/VERTICAL WALLS
            # ----------------------------
            if hv_ratio > 0.65:
                score += 2
            elif hv_ratio > 0.45:
                score += 1

            # ----------------------------
            # TEXTURE
            # photos are highly textured
            # blueprints are cleaner
            # ----------------------------
            if texture_var < 800:
                score += 2
            elif texture_var < 1500:
                score += 1
            elif texture_var > 2500:
                score -= 2

            # ----------------------------
            # SATURATION
            # blueprints usually low color
            # ----------------------------
            if saturation < 25:
                score += 1
            elif saturation > 60:
                score -= 2

            # ----------------------------
            # EDGE DENSITY
            # ----------------------------
            if edge_density < 0.03:
                score += 1
            elif edge_density > 0.15:
                score -= 2

            # ----------------------------
            # ENTROPY
            # ----------------------------
            if entropy < 6.5:
                score += 1
            elif entropy > 7.5:
                score -= 2

            # ----------------------------
            # RECTANGLES / ROOM STRUCTURE
            # ----------------------------
            if rectangle_count > 20:
                score += 2
            elif rectangle_count > 10:
                score += 1

            # confidence
            confidence = max(min(score / 10.0, 1.0), 0.0)

            # ----------------------------
            # FINAL DECISION
            # ----------------------------
            if score >= self.min_score:
                return True, "Blueprint detected", confidence

            return False, "Not a valid blueprint", confidence

        except Exception as e:
            return False, str(e), 0.0

    # =========================================================
    # ENTROPY
    # =========================================================
    def _entropy(self, gray):

        hist = cv2.calcHist(
            [gray],
            [0],
            None,
            [256],
            [0, 256]
        )

        hist = hist.ravel()

        hist = hist / (hist.sum() + 1e-7)

        hist = hist[hist > 0]

        entropy = -np.sum(hist * np.log2(hist))

        return entropy

    # =========================================================
    # HORIZONTAL / VERTICAL LINE RATIO
    # =========================================================
    def _hv_ratio(self, lines):

        if lines is None:
            return 0.0

        hv = 0

        for l in lines:

            x1, y1, x2, y2 = l[0]

            angle = abs(
                np.degrees(
                    np.arctan2(y2 - y1, x2 - x1)
                )
            )

            horizontal = angle < 10
            vertical = abs(angle - 90) < 10

            if horizontal or vertical:
                hv += 1

        return hv / len(lines)

    # =========================================================
    # RECTANGLE COUNT
    # =========================================================
    def _rectangle_count(self, edges):

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        rectangles = 0

        for cnt in contours:

            area = cv2.contourArea(cnt)

            if area < 100:
                continue

            approx = cv2.approxPolyDP(
                cnt,
                0.02 * cv2.arcLength(cnt, True),
                True
            )

            if len(approx) == 4:
                rectangles += 1

        return rectangles


# =========================================================
# SINGLETON
# =========================================================
validator = BlueprintValidator()


def validate_blueprint(image_path):
    return validator.validate(image_path)


# =========================================================
# TEST
# =========================================================
if __name__ == "__main__":

    image_path = "test.jpg"

    valid, message, confidence = validate_blueprint(image_path)

    print("\n========================")
    print("VALIDATION RESULT")
    print("========================")
    print(f"Valid      : {valid}")
    print(f"Message    : {message}")
    print(f"Confidence : {confidence:.2f}")