#!/usr/bin/env python3
"""
Billboard Replacer
==================

Replace billboards in videos or images with your own ads.

Usage:
    # Video
    python replace_billboard.py --input video.mp4 --ad my_ad.png

    # Image
    python replace_billboard.py --input photo.jpg --ad my_ad.png

    # Debug mode (show outlines only)
    python replace_billboard.py --input photo.jpg --ad my_ad.png --debug

Requirements:
    pip install ultralytics opencv-python numpy
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


class BillboardReplacer:
    def __init__(self, model_path: str = "billboard_best.pt", confidence: float = 0.5):
        print(f"Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.confidence = confidence
        # Filter settings
        self.min_aspect_ratio = 1.2  # Width/height - billboards are usually wide
        self.max_frame_coverage = 0.5  # Max % of frame a detection can cover

    def is_valid_detection(self, bbox: tuple, frame_shape: tuple) -> bool:
        """Filter out unlikely billboard detections."""
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        frame_h, frame_w = frame_shape[:2]

        # Check if detection spans almost entire frame width (wall murals)
        width_ratio = w / frame_w
        height_ratio = h / frame_h
        if width_ratio > 0.95 and height_ratio > 0.65:
            return False

        # Check aspect ratio for small detections (likely false positives)
        # Billboards are usually wide rectangles, not squares
        aspect = w / (h + 1e-6)
        if 0.7 < aspect < 1.4:  # Nearly square
            if w < frame_w * 0.35:  # Small square = likely not a billboard
                return False

        return True

    def find_billboard_contour(self, frame: np.ndarray, bbox: tuple) -> tuple:
        """
        Find the billboard rectangle within a detected region.

        Strategy:
        1. Try to find quadrilateral via edge detection (multiple methods)
        2. If not found, use YOLO bbox

        Returns:
            (contour, corners) - 4 corner points for both contour and perspective
        """
        x1, y1, x2, y2 = map(int, bbox)
        bbox_w, bbox_h = x2 - x1, y2 - y1
        bbox_area = bbox_w * bbox_h

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            return corners.reshape(-1, 1, 2).astype(np.int32), corners

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        best_quad = None
        best_score = 0

        # Try multiple edge detection methods
        edge_methods = []

        # Method 1: Standard Canny
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges1 = cv2.Canny(blurred, 30, 100)
        edge_methods.append(edges1)

        # Method 2: Stronger blur + Canny (reduces internal noise)
        very_blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        edges2 = cv2.Canny(very_blurred, 20, 80)
        edge_methods.append(edges2)

        # Method 3: Morphological gradient (finds outer edges better)
        kernel = np.ones((5, 5), np.uint8)
        gradient = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, kernel)
        _, edges3 = cv2.threshold(gradient, 30, 255, cv2.THRESH_BINARY)
        edge_methods.append(edges3)

        # Method 4: Bilateral filter + Canny (preserves edges, smooths interior)
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        edges4 = cv2.Canny(bilateral, 30, 100)
        edge_methods.append(edges4)

        for edges in edge_methods:
            # Dilate to connect edge segments
            edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

            # Close gaps
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < bbox_area * 0.15:
                    continue

                for eps_mult in [0.02, 0.03, 0.04, 0.05, 0.06]:
                    epsilon = eps_mult * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)

                    if len(approx) == 4 and cv2.isContourConvex(approx):
                        rect = cv2.minAreaRect(approx)
                        rect_area = rect[1][0] * rect[1][1]
                        if rect_area > 0:
                            rectangularity = area / rect_area
                            score = area * rectangularity
                            if score > best_score:
                                best_score = score
                                best_quad = approx.reshape(4, 2)
                        break

        if best_quad is not None:
            best_quad[:, 0] += x1
            best_quad[:, 1] += y1
            corners = self._order_corners(best_quad)
            return corners.reshape(-1, 1, 2).astype(np.int32), corners

        # Fallback: use YOLO bbox
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        return corners.reshape(-1, 1, 2).astype(np.int32), corners

    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """Order corners as top-left, top-right, bottom-right, bottom-left."""
        corners = corners.astype(np.float32)
        by_y = corners[np.argsort(corners[:, 1])]
        top = by_y[:2][np.argsort(by_y[:2, 0])]
        bot = by_y[2:][np.argsort(by_y[2:, 0])]
        return np.array([top[0], top[1], bot[1], bot[0]], dtype=np.float32)

    def draw_debug(self, frame: np.ndarray, contour: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Draw debug visualization with rectangle outline and corner dots."""
        result = frame.copy()

        # Draw rectangle outline (bright green, thick)
        cv2.polylines(result, [corners.astype(np.int32)], True, (0, 255, 0), 3)

        # Draw corner points (red with white border)
        for i, corner in enumerate(corners):
            x, y = int(corner[0]), int(corner[1])
            cv2.circle(result, (x, y), 12, (0, 0, 255), -1)
            cv2.circle(result, (x, y), 14, (255, 255, 255), 2)
            # Label corners 1-4
            cv2.putText(result, str(i+1), (x + 15, y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return result

    def replace(self, frame: np.ndarray, replacement: np.ndarray,
                contour: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Warp replacement image onto frame using contour mask.
        """
        h, w = replacement.shape[:2]
        height, width = frame.shape[:2]

        # Source points (corners of replacement image)
        src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

        # Perspective transform to corners
        M = cv2.getPerspectiveTransform(src_pts, corners)
        warped = cv2.warpPerspective(replacement, M, (width, height))

        # Create mask from full contour (not just 4 corners)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)

        # Slight erosion to avoid edge artifacts
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)

        # Feather edges for smoother blending
        mask_blur = cv2.GaussianBlur(mask, (5, 5), 0)
        mask_3ch = cv2.cvtColor(mask_blur, cv2.COLOR_GRAY2BGR) / 255.0

        # Blend
        result = (warped * mask_3ch + frame * (1 - mask_3ch)).astype(np.uint8)
        return result

    def process_image(self, image_path: str, replacement_path: str,
                      output_path: str = None, debug: bool = False) -> str:
        image_path = Path(image_path)
        if output_path is None:
            suffix = "_debug" if debug else "_replaced"
            output_path = image_path.parent / f"{image_path.stem}{suffix}{image_path.suffix}"

        print(f"\nProcessing image: {image_path}")

        frame = cv2.imread(str(image_path))
        replacement = cv2.imread(replacement_path)

        if frame is None:
            raise FileNotFoundError(f"Could not load: {image_path}")
        if replacement is None:
            raise FileNotFoundError(f"Could not load: {replacement_path}")

        # Detect billboard
        results = self.model(frame, conf=self.confidence, verbose=False)

        if len(results[0].boxes) == 0:
            print("  No billboard detected!")
            return None

        # Process each detected billboard
        valid_count = 0
        for i, det in enumerate(results[0].boxes):
            bbox = tuple(det.xyxy[0].cpu().numpy())
            conf = float(det.conf[0].cpu().numpy())

            # Filter unlikely detections
            if not self.is_valid_detection(bbox, frame.shape):
                print(f"  Detection {i+1}: conf={conf:.2f} - SKIPPED (failed filters)")
                continue

            # Find contour and corners
            contour, corners = self.find_billboard_contour(frame, bbox)
            valid_count += 1
            print(f"  Billboard {valid_count}: conf={conf:.2f}, contour_points={len(contour)}")

            if debug:
                frame = self.draw_debug(frame, contour, corners)
            else:
                frame = self.replace(frame, replacement, contour, corners)

        if valid_count == 0:
            print("  No valid billboards after filtering!")
            return None

        cv2.imwrite(str(output_path), frame)
        print(f"  Saved: {output_path}")
        return str(output_path)

    def process_video(self, video_path: str, replacement_path: str,
                      output_path: str = None, debug: bool = False) -> str:
        video_path = Path(video_path)
        if output_path is None:
            suffix = "_debug" if debug else "_replaced"
            output_path = video_path.parent / f"{video_path.stem}{suffix}.mp4"

        print(f"\nProcessing video: {video_path}")

        replacement = cv2.imread(replacement_path)
        if replacement is None:
            raise FileNotFoundError(f"Could not load: {replacement_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Frames: {total_frames}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Detect on first frame and lock contours/corners
        ret, first_frame = cap.read()
        if not ret:
            raise RuntimeError("Could not read first frame")

        results = self.model(first_frame, conf=self.confidence, verbose=False)

        if len(results[0].boxes) == 0:
            print("  No billboard detected!")
            cap.release()
            out.release()
            return None

        # Get contours and corners for each billboard (locked for all frames)
        locked_billboards = []
        valid_count = 0
        for i, det in enumerate(results[0].boxes):
            bbox = tuple(det.xyxy[0].cpu().numpy())
            conf = float(det.conf[0].cpu().numpy())

            # Filter unlikely detections
            if not self.is_valid_detection(bbox, first_frame.shape):
                print(f"  Detection {i+1}: conf={conf:.2f} - SKIPPED (failed filters)")
                continue

            contour, corners = self.find_billboard_contour(first_frame, bbox)
            locked_billboards.append((contour, corners))
            valid_count += 1
            print(f"  Billboard {valid_count}: conf={conf:.2f}, contour_points={len(contour)}, locked")

        if not locked_billboards:
            print("  No valid billboards after filtering!")
            cap.release()
            out.release()
            return None

        # Precompute warp data
        h, w = replacement.shape[:2]
        src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

        warp_data = []
        for contour, corners in locked_billboards:
            M = cv2.getPerspectiveTransform(src_pts, corners)

            # Create mask from contour
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)
            mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
            mask_blur = cv2.GaussianBlur(mask, (5, 5), 0)
            mask_3ch = cv2.cvtColor(mask_blur, cv2.COLOR_GRAY2BGR) / 255.0

            warp_data.append((M, mask_3ch, contour, corners))

        # Process all frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_num = 0

        print(f"\n  Processing {total_frames} frames...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if debug:
                for M, mask_3ch, contour, corners in warp_data:
                    frame = self.draw_debug(frame, contour, corners)
            else:
                for M, mask_3ch, contour, corners in warp_data:
                    warped = cv2.warpPerspective(replacement, M, (width, height))
                    frame = (warped * mask_3ch + frame * (1 - mask_3ch)).astype(np.uint8)

            out.write(frame)
            frame_num += 1

            if frame_num % 100 == 0:
                pct = (frame_num / total_frames) * 100
                print(f"    {frame_num}/{total_frames} ({pct:.0f}%)")

        cap.release()
        out.release()

        print(f"\n  Done! Saved: {output_path}")
        return str(output_path)

    def process(self, input_path: str, replacement_path: str,
                output_path: str = None, debug: bool = False) -> str:
        input_path = Path(input_path)
        ext = input_path.suffix.lower()

        video_exts = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

        if ext in video_exts:
            return self.process_video(str(input_path), replacement_path, output_path, debug)
        elif ext in image_exts:
            return self.process_image(str(input_path), replacement_path, output_path, debug)
        else:
            raise ValueError(f"Unknown file type: {ext}")


def main():
    parser = argparse.ArgumentParser(
        description="Replace billboards in images or videos with your own ads",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python replace_billboard.py --input video.mp4 --ad coca_cola.png
    python replace_billboard.py --input photo.jpg --ad nike_ad.jpg --output result.jpg
    python replace_billboard.py -i photo.jpg -a my_ad.png --debug
        """
    )

    parser.add_argument("--input", "-i", required=True, help="Input image or video")
    parser.add_argument("--ad", "-a", required=True, help="Replacement ad image")
    parser.add_argument("--output", "-o", help="Output path (optional)")
    parser.add_argument("--model", "-m", default="billboard_best.pt", help="YOLO model path")
    parser.add_argument("--confidence", "-c", type=float, default=0.3, help="Detection confidence")
    parser.add_argument("--debug", "-d", action="store_true", help="Debug mode: show outlines only")

    args = parser.parse_args()

    replacer = BillboardReplacer(model_path=args.model, confidence=args.confidence)
    replacer.process(args.input, args.ad, args.output, args.debug)


if __name__ == "__main__":
    main()
