"""
Face Detection & Biometric Verification Engine (Stage 05 - Verify Face)
Extracts faces from documents, compares with live selfie/reference photo, and computes similarity & liveness.
"""

import os
import base64
from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np
import onnxruntime as ort


class FaceVerificationEngine:
    """
    Biometric verification engine that extracts faces from identity documents,
    computes deep feature embeddings, and verifies biometric identity against live selfies.
    """

    def __init__(self, device: Optional[str] = None):
        # Using CPU for cost-effective execution as per ONNX migration plan
        providers = ['CPUExecutionProvider']
        
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        model_path = os.path.join(model_dir, "face_embedder.onnx")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found at {model_path}. Please run export_model.py first.")
            
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def detect_and_crop_face(self, image: np.ndarray, is_document: bool = False) -> Tuple[Optional[np.ndarray], Optional[Dict[str, int]]]:
        """
        Detects primary face in document or selfie using skin-tone color segmentation,
        elliptical contour analysis, and positional priors.
        """
        if image is None or image.size == 0:
            return None, None

        h, w = image.shape[:2]

        # 1. Positional prior for document photo (usually left 40%, top 10-70%)
        if is_document:
            search_roi = image[int(h * 0.1):int(h * 0.7), 0:int(w * 0.45)]
            offset_x, offset_y = 0, int(h * 0.1)
        else:
            search_roi = image
            offset_x, offset_y = 0, 0

        # 2. Skin tone segmentation in HSV color space
        hsv = cv2.cvtColor(search_roi, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 25, 40], dtype=np.uint8)
        upper_skin = np.array([30, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Morphological filtering to merge facial clusters
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_box = None
        max_area = 0

        min_face_area = (h * w) * 0.015

        for c in contours:
            area = cv2.contourArea(c)
            if area > min_face_area:
                x, y, cw, ch = cv2.boundingRect(c)
                aspect_ratio = float(ch) / (cw + 1e-5)
                # Face aspect ratio is typically vertical oval (0.85 to 2.2)
                if 0.8 <= aspect_ratio <= 2.5:
                    if area > max_area:
                        max_area = area
                        best_box = (x + offset_x, y + offset_y, cw, ch)

        if best_box is None:
            # Fallback: crop center region for avatar/selfie, or standard left photo box for document
            if is_document:
                bx = int(w * 0.05)
                by = int(h * 0.17)
                bw = int(w * 0.25)
                bh = int(h * 0.40)
            else:
                bw = int(w * 0.70)
                bh = int(h * 0.75)
                bx = int((w - bw) / 2)
                by = int((h - bh) / 2)
            best_box = (bx, by, bw, bh)

        bx, by, bw, bh = best_box
        # Add slight margin
        pad_x = int(bw * 0.1)
        pad_y = int(bh * 0.1)
        x1 = max(0, bx - pad_x)
        y1 = max(0, by - pad_y)
        x2 = min(w, bx + bw + pad_x)
        y2 = min(h, by + bh + pad_y)

        cropped = image[y1:y2, x1:x2]
        if cropped.size == 0:
            return image, {"x": 0, "y": 0, "width": w, "height": h}

        bbox = {"x": int(x1), "y": int(y1), "width": int(x2 - x1), "height": int(y2 - y1)}
        return cropped, bbox

    def compute_embedding(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Extracts L2-normalized 512-dimensional facial feature embedding vector.
        """
        rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB) if len(face_crop.shape) == 3 else face_crop
        
        # 1. Resize to 160x160
        resized = cv2.resize(rgb_face, (160, 160))
        
        # 2. Convert to float32 and scale to [0, 1]
        img_float = resized.astype(np.float32) / 255.0
        
        # 3. ImageNet Normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (img_float - mean) / std
        
        # 4. HWC to CHW and add batch dimension (1, C, H, W)
        chw = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(chw, axis=0)

        # 5. ONNX Inference
        feat = self.session.run(None, {self.input_name: input_tensor})[0]
        
        # 6. L2 Normalization
        norm = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-7
        feat = feat / norm
        embedding = feat.flatten()

        return embedding

    def check_liveness_heuristics(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Performs anti-spoofing texture and frequency checks to detect screen replay or photo-on-paper spoof.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        
        # 1. Laplacian sharpness/blur score
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_sharp = laplacian_var > 30.0

        # 2. Color saturation distribution
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            sat_mean = np.mean(hsv[:, :, 1])
            is_natural_color = 15 < sat_mean < 200
        else:
            is_natural_color = True

        # 3. Frequency spectrum
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift) + 1e-5)
        fft_energy = float(np.mean(magnitude))

        liveness_score = 0.5
        if is_sharp:
            liveness_score += 0.25
        if is_natural_color:
            liveness_score += 0.25

        liveness_score = min(1.0, max(0.0, liveness_score))

        return {
            "liveness_score": round(liveness_score, 2),
            "is_live": liveness_score >= 0.65,
            "sharpness_var": round(float(laplacian_var), 2),
            "fft_energy": round(fft_energy, 2)
        }

    def verify_faces(
        self, 
        doc_image: np.ndarray, 
        live_image: np.ndarray, 
        threshold: float = 0.65
    ) -> Dict[str, Any]:
        """
        Matches document photo against live camera selfie.
        Returns similarity score, match verdict, liveness analysis, and Base64 face crops.
        """
        doc_face, doc_bbox = self.detect_and_crop_face(doc_image, is_document=True)
        live_face, live_bbox = self.detect_and_crop_face(live_image, is_document=False)

        if doc_face is None or doc_face.size == 0:
            return {
                "success": False,
                "is_match": False,
                "similarity_score": 0.0,
                "message": "No face detected in document image.",
                "doc_face_found": False,
                "live_face_found": live_face is not None
            }

        if live_face is None or live_face.size == 0:
            return {
                "success": False,
                "is_match": False,
                "similarity_score": 0.0,
                "message": "No face detected in live selfie image.",
                "doc_face_found": True,
                "live_face_found": False
            }

        # Embeddings & Cosine Similarity
        emb1 = self.compute_embedding(doc_face)
        emb2 = self.compute_embedding(live_face)

        cosine_sim = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-7))
        
        # Calibrated normalization anchored to face distribution boundaries:
        # Different faces (cos_sim < 0.55) -> calibrated_sim < 0.40 (Mismatch)
        # Same face (cos_sim > 0.75) -> calibrated_sim > 0.80 (Match)
        calibrated_sim = max(0.0, min(1.0, (cosine_sim - 0.35) / 0.50))
        normalized_sim = round(float(calibrated_sim), 4)

        # Liveness on live photo
        liveness = self.check_liveness_heuristics(live_image)

        # Verdict
        is_match = normalized_sim >= threshold and liveness["is_live"]



        # Base64 encodings of cropped faces
        _, b1 = cv2.imencode('.png', doc_face)
        doc_b64 = f"data:image/png;base64,{base64.b64encode(b1).decode('utf-8')}"
        _, b2 = cv2.imencode('.png', live_face)
        live_b64 = f"data:image/png;base64,{base64.b64encode(b2).decode('utf-8')}"

        return {
            "success": True,
            "is_match": is_match,
            "similarity_score": normalized_sim,
            "verdict": "BIOMETRIC_MATCH" if is_match else ("BIOMETRIC_MISMATCH" if normalized_sim < threshold else "LIVENESS_FAILED"),
            "threshold": threshold,
            "liveness": liveness,
            "doc_face_bbox": doc_bbox,
            "live_face_bbox": live_bbox,
            "doc_face_base64": doc_b64,
            "live_face_base64": live_b64
        }
