"""
Document Preprocessing Pipeline (Stage 01 - Preprocess)
Enhances raw document scans/photos: deskewing, glare reduction, contrast enhancement, binarization.
"""

from typing import Tuple, Optional, Dict, Any
import cv2
import numpy as np


class DocumentPreprocessor:
    """
    Robust OpenCV-based image preprocessing for identity documents (Passports, Visas, IDs).
    Handles perspective correction, deskewing, shadow/glare reduction, and OCR contrast enhancement.
    """

    def __init__(self, target_dpi_width: int = 1200):
        self.target_dpi_width = target_dpi_width

    def resize_to_standard(self, image: np.ndarray) -> np.ndarray:
        """Resizes image to standard width while preserving aspect ratio."""
        h, w = image.shape[:2]
        if w == self.target_dpi_width:
            return image
        scale = self.target_dpi_width / float(w)
        new_h = int(h * scale)
        return cv2.resize(image, (self.target_dpi_width, new_h), interpolation=cv2.INTER_CUBIC)

    def detect_and_deskew(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detects document skew angle using Hough line transform and text orientation,
        then rotates the image to achieve horizontal alignment.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        
        # Blur and edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # Hough line transform
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=image.shape[1] // 4, maxLineGap=20)
        
        angles = []
        if lines is not None:
            for line in lines:
                coords = line.flatten()
                if len(coords) >= 4:
                    x1, y1, x2, y2 = coords[:4]
                    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    # Consider near-horizontal lines (-45 to 45 deg)
                    if -45 < angle < 45:
                        angles.append(angle)

        skew_angle = float(np.median(angles)) if angles else 0.0

        # Rotate if significant skew detected
        if abs(skew_angle) > 0.4:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            rot_mat = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
            deskewed = cv2.warpAffine(
                image, rot_mat, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            return deskewed, skew_angle
        
        return image, 0.0

    def normalize_illumination(self, image: np.ndarray) -> np.ndarray:
        """
        Removes non-uniform lighting, flash reflection, and shadows using CLAHE on the L-channel in LAB space.
        """
        if len(image.shape) == 2:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
        
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def denoise_and_sharpen(self, image: np.ndarray) -> np.ndarray:
        """
        Reduces background noise while preserving sharp character edges,
        followed by an unsharp masking kernel.
        """
        # Bilateral filter retains edges while smoothing noise
        if len(image.shape) == 3:
            filtered = cv2.bilateralFilter(image, d=9, sigmaColor=50, sigmaSpace=50)
        else:
            filtered = cv2.bilateralFilter(image, d=9, sigmaColor=50, sigmaSpace=50)

        # Unsharp mask for high-definition text boundaries
        gaussian = cv2.GaussianBlur(filtered, (0, 0), 2.0)
        sharpened = cv2.addWeighted(filtered, 1.4, gaussian, -0.4, 0)
        return sharpened

    def binarize_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Creates clean high-contrast black-and-white image optimized for OCR text segmentation.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        
        # Illumination flattening via morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        normalized = cv2.divide(gray, background, scale=255)
        
        # Otsu's thresholding
        _, binarized = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binarized

    def extract_document_contour(self, image: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Finds the 4-corner document quadrangle in a photo and performs perspective rectification.
        If no distinct 4-point quadrilateral is found, returns original image.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 30, 120)

        # Dilate edges to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edged, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        doc_contour = None
        for c in contours[:5]:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # Check for quadrilateral covering at least 25% of image area
            if len(approx) == 4 and cv2.contourArea(c) > (h * w * 0.25):
                doc_contour = approx
                break

        if doc_contour is not None:
            warped = self._four_point_transform(image, doc_contour.reshape(4, 2))
            return warped, doc_contour
        
        return image, None

    def _four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Standard 4-point perspective warp to top-down view."""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left

        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    def process(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Executes full preprocessing pipeline.
        Returns dictionary containing enhanced color, binarized OCR image, and metadata.
        """
        standard_img = self.resize_to_standard(image)
        rectified_img, contour = self.extract_document_contour(standard_img)
        deskewed_img, skew_angle = self.detect_and_deskew(rectified_img)
        enhanced_img = self.normalize_illumination(deskewed_img)
        sharpened_img = self.denoise_and_sharpen(enhanced_img)
        binarized_img = self.binarize_for_ocr(sharpened_img)

        return {
            "processed_image": sharpened_img,
            "binarized_image": binarized_img,
            "skew_angle": round(skew_angle, 2),
            "perspective_rectified": contour is not None,
            "dimensions": {"width": sharpened_img.shape[1], "height": sharpened_img.shape[0]}
        }
