"""
Privacy Service for AI Gallery
Handles face detection, license plate detection, and image blurring
"""

# Try to import OpenCV
try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("Warning: opencv-python not installed. Face and plate detection will be limited.")

# Try to import PIL
try:
    from PIL import Image, ImageFilter, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: Pillow not installed. Image blur functionality will be limited.")

import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class PrivacyService:
    def __init__(self):
        # Load OpenCV Haar Cascade for face detection
        self.face_cascade = None
        self.load_face_detector()

    def load_face_detector(self):
        """Load OpenCV Haar Cascade for face detection"""
        if not HAS_OPENCV:
            print("OpenCV not available, face detection disabled")
            self.face_cascade = None
            return

        try:
            # Try to load the face cascade classifier
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)

            if self.face_cascade.empty():
                print("Warning: Could not load face cascade classifier")
                self.face_cascade = None
            else:
                print("Face detector loaded successfully")
        except Exception as e:
            print(f"Error loading face detector: {e}")
            self.face_cascade = None

    def detect_faces(self, image_path: str) -> List[Dict]:
        """
        Detect faces in an image

        Args:
            image_path: Path to the image file

        Returns:
            List of face bounding boxes: [{"type": "face", "x": int, "y": int, "w": int, "h": int}, ...]
        """
        if not HAS_OPENCV or self.face_cascade is None:
            print("Face detector not available")
            return []

        try:
            # Load image with OpenCV
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not load image: {image_path}")
                return []

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            # Convert to our format
            face_zones = []
            for (x, y, w, h) in faces:
                face_zones.append({
                    "type": "face",
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h)
                })

            return face_zones

        except Exception as e:
            print(f"Error detecting faces: {e}")
            return []

    def detect_license_plates(self, image_path: str) -> List[Dict]:
        """
        Detect license plates using pattern matching and contour detection

        Args:
            image_path: Path to the image file

        Returns:
            List of plate bounding boxes
        """
        if not HAS_OPENCV:
            print("License plate detection not available (requires OpenCV)")
            return []

        try:
            # Load image with OpenCV
            img = cv2.imread(image_path)
            if img is None:
                return []

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Apply bilateral filter to reduce noise
            gray = cv2.bilateralFilter(gray, 11, 17, 17)

            # Detect edges
            edged = cv2.Canny(gray, 30, 200)

            # Find contours
            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Sort contours by area
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

            plate_zones = []

            for contour in contours:
                # Approximate the contour
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.018 * peri, True)

                # Check if it's a rectangle (4 points)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)

                    # License plate aspect ratio check (roughly 2:1 to 3:1)
                    aspect_ratio = w / float(h)

                    if 2.0 <= aspect_ratio <= 6.0 and w > 80 and h > 20:
                        plate_zones.append({
                            "type": "plate",
                            "x": int(x),
                            "y": int(y),
                            "w": int(w),
                            "h": int(h)
                        })

            return plate_zones[:3]  # Return top 3 candidates

        except Exception as e:
            print(f"Error detecting license plates: {e}")
            return []

    def analyze_image_privacy(self, image_path: str) -> Dict:
        """
        Perform full privacy analysis on an image

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary with analysis results:
            {
                "has_faces": bool,
                "has_plates": bool,
                "is_nsfw": bool,
                "privacy_zones": List[Dict]
            }
        """
        # Detect faces
        faces = self.detect_faces(image_path)

        # Detect license plates
        plates = self.detect_license_plates(image_path)

        # Combine all privacy zones
        privacy_zones = faces + plates

        return {
            "has_faces": len(faces) > 0,
            "has_plates": len(plates) > 0,
            "is_nsfw": False,  # TODO: Implement NSFW detection
            "privacy_zones": privacy_zones
        }

    def apply_blur_to_zones(self, image_path: str, zones: List[Dict],
                           blur_type: str = "gaussian", blur_strength: int = 30):
        """
        Apply blur to specified zones in an image

        Args:
            image_path: Path to the image file
            zones: List of zones to blur
            blur_type: "gaussian" or "pixelate"
            blur_strength: Blur intensity (higher = more blur)

        Returns:
            PIL Image with blurred zones (or None if PIL not available)
        """
        if not HAS_PIL:
            print("PIL not available, cannot apply blur")
            return None

        try:
            # Load image with PIL
            img = Image.open(image_path)

            if not zones:
                return img

            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Apply blur to each zone
            for zone in zones:
                x, y, w, h = zone['x'], zone['y'], zone['w'], zone['h']

                # Extract the region
                region = img.crop((x, y, x + w, y + h))

                if blur_type == "gaussian":
                    # Apply Gaussian blur
                    blurred_region = region.filter(ImageFilter.GaussianBlur(radius=blur_strength))
                elif blur_type == "pixelate":
                    # Pixelate effect
                    small_size = max(1, w // blur_strength, h // blur_strength)
                    temp = region.resize((small_size, small_size), Image.BILINEAR)
                    blurred_region = temp.resize((w, h), Image.NEAREST)
                else:
                    blurred_region = region.filter(ImageFilter.GaussianBlur(radius=blur_strength))

                # Paste back
                img.paste(blurred_region, (x, y))

            return img

        except Exception as e:
            print(f"Error applying blur: {e}")
            # Return original image if error
            try:
                return Image.open(image_path)
            except:
                return None

    def generate_privacy_thumbnail(self, image_path: str, zones: List[Dict],
                                   size: int = 500, blur_strength: int = 30):
        """
        Generate a thumbnail with privacy zones blurred

        Args:
            image_path: Path to the image file
            zones: List of zones to blur
            size: Thumbnail size
            blur_strength: Blur intensity

        Returns:
            PIL Image thumbnail with blurred zones (or None if PIL not available)
        """
        if not HAS_PIL:
            print("PIL not available, cannot generate thumbnail")
            return None

        try:
            # Load and blur the image
            img = self.apply_blur_to_zones(image_path, zones, blur_strength=blur_strength)

            if img is None:
                return None

            # Create thumbnail
            img.thumbnail((size, size), Image.Resampling.LANCZOS)

            return img

        except Exception as e:
            print(f"Error generating privacy thumbnail: {e}")
            # Fallback to regular thumbnail
            try:
                img = Image.open(image_path)
                img.thumbnail((size, size), Image.Resampling.LANCZOS)
                return img
            except:
                return None

    def detect_text_patterns(self, image_path: str) -> List[str]:
        """
        Detect text patterns that might be sensitive (email, phone, etc.)
        This is a placeholder for OCR-based detection

        Args:
            image_path: Path to the image file

        Returns:
            List of detected sensitive text patterns
        """
        # TODO: Implement OCR-based text detection
        # Could use pytesseract or easyocr
        return []


# Singleton instance
_privacy_service = None


def get_privacy_service() -> PrivacyService:
    """Get or create privacy service singleton"""
    global _privacy_service
    if _privacy_service is None:
        _privacy_service = PrivacyService()
    return _privacy_service
