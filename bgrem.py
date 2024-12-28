import cv2
import numpy as np
import onnxruntime as ort
import json
from typing import List, Tuple, Dict

class Colors:
    def __init__(self):
        # Create a color palette
        self.palette = np.random.uniform(0, 255, size=(80, 3)).astype(np.uint8)
    
    def get(self, idx):
        # Return RGB values directly
        return [int(x) for x in self.palette[idx]]
    
    @staticmethod
    def hex_to_rgba(color: List[int], alpha: int) -> List[int]:
        # Modified to take RGB list directly instead of hex string
        return [*color, alpha]

class YOLOSegmentation:
    def __init__(
        self,
        model_path: str,
        nms_model_path: str,
        mask_model_path: str,
        labels_path: str,
        use_cuda: bool = False
    ):
        # Set providers based on CUDA availability and user preference
        providers = ['CPUExecutionProvider']
        if use_cuda and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
        
        # Initialize ONNX Runtime sessions with appropriate provider
        try:
            self.net = ort.InferenceSession(model_path, providers=providers)
            self.nms = ort.InferenceSession(nms_model_path, providers=providers)
            self.mask = ort.InferenceSession(mask_model_path, providers=providers)
            
            print(f"Models loaded successfully using: {self.net.get_providers()[0]}")
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
        
        # Load labels
        try:
            with open(labels_path, 'r') as f:
                self.labels = json.load(f)
        except Exception as e:
            print(f"Error loading labels: {e}")
            raise
        
        self.num_classes = len(self.labels)
        self.colors = Colors()

    def preprocess(
        self,
        image: np.ndarray,
        model_width: int,
        model_height: int,
        stride: int = 32
    ) -> Tuple[np.ndarray, float, float]:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        
        w, h = image.shape[1], image.shape[0]
        if w % stride != 0:
            w = ((w // stride) + (1 if w % stride >= stride/2 else 0)) * stride
        if h % stride != 0:
            h = ((h // stride) + (1 if h % stride >= stride/2 else 0)) * stride
        
        image = cv2.resize(image, (w, h))
        
        max_size = max(w, h)
        x_pad, x_ratio = max_size - w, max_size / w
        y_pad, y_ratio = max_size - h, max_size / h
        
        image = cv2.copyMakeBorder(image, 0, y_pad, 0, x_pad, cv2.BORDER_CONSTANT)
        
        blob = cv2.dnn.blobFromImage(
            image,
            1/255.0,
            (model_width, model_height),
            swapRB=True
        )
        
        return blob, x_ratio, y_ratio

    @staticmethod
    def overflow_boxes(box: List[float], max_size: int) -> List[float]:
        x, y, w, h = box
        x = max(0, x)
        y = max(0, y)
        w = min(w, max_size - x)
        h = min(h, max_size - y)
        return [x, y, w, h]

    def detect_image(
        self,
        image: np.ndarray,
        topk: int = 100,
        iou_threshold: float = 0.45,
        score_threshold: float = 0.25,
        input_shape: List[int] = [1, 3, 640, 640]
    ) -> Tuple[np.ndarray, List[Dict]]:
        try:
            model_width, model_height = input_shape[2:]
            max_size = max(model_width, model_height)
            
            input_tensor, x_ratio, y_ratio = self.preprocess(image, model_width, model_height)
            
            outputs = self.net.run(None, {'images': input_tensor.astype(np.float32)})
            output0, output1 = outputs
            
            nms_config = np.array([self.num_classes, topk, iou_threshold, score_threshold], dtype=np.float32)
            selected = self.nms.run(None, {
                'detection': output0,
                'config': nms_config
            })[0]
            
            boxes = []
            overlay = np.zeros((model_height, model_width, 4), dtype=np.uint8)
            
            for idx in range(selected.shape[1]):
                data = selected[0, idx]
                box = data[:4]
                scores = data[4:4+self.num_classes]
                score = np.max(scores)
                label = np.argmax(scores)
                color = self.colors.get(label)  # Now returns RGB list
                
                box = self.overflow_boxes([
                    box[0] - 0.5 * box[2],
                    box[1] - 0.5 * box[3],
                    box[2],
                    box[3]
                ], max_size)
                
                scaled_box = self.overflow_boxes([
                    int(box[0] * x_ratio),
                    int(box[1] * y_ratio),
                    int(box[2] * x_ratio),
                    int(box[3] * y_ratio)
                ], max_size)
                
                boxes.append({
                    'label': self.labels[label],
                    'probability': float(score),
                    'color': color,
                    'bbox': scaled_box
                })
                
                mask_input = np.concatenate([box, data[4+self.num_classes:]])
                mask_config = np.array([
                    max_size,
                    *scaled_box,
                    *Colors.hex_to_rgba(color, 120)  # Pass RGB list directly
                ], dtype=np.float32)
                
                mask_output = self.mask.run(None, {
                    'detection': mask_input.astype(np.float32),
                    'mask': output1,
                    'config': mask_config,
                    'overlay': overlay
                })[0]
                
                overlay = mask_output
            
            return overlay, boxes
            
        except Exception as e:
            print(f"Error during detection: {e}")
            raise

    def visualize(
        self,
        image: np.ndarray,
        overlay: np.ndarray,
        boxes: List[Dict],
        remove_background: bool = True
        ) -> np.ndarray:
        overlay = cv2.resize(overlay, (image.shape[1], image.shape[0]))
        
        if remove_background:
            # Create a transparent background
            result = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
            
            # Use the alpha channel of the overlay as a mask
            mask = (overlay[..., 3:] > 0).astype(np.float32)
            
            # Apply the mask to the original image and add alpha channel
            result[..., :3] = image * mask
            result[..., 3] = (mask * 255).astype(np.uint8)[..., 0]
        else:
            result = image.copy()
            mask = overlay[..., 3:] / 255.0
            result = result * (1 - mask) + overlay[..., :3] * mask
        
        return result

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv8 Segmentation with Background Removal')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--use_cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--output', type=str, default='./output_images/output.png', help='Path to output image')
    parser.add_argument('--keep_background', action='store_true', help='Keep the background (don\'t remove it)')
    args = parser.parse_args()
    
    try:
        detector = YOLOSegmentation(
            model_path='./models/yolov8n-seg.onnx',
            nms_model_path='./models/nms-yolov8.onnx',
            mask_model_path='./models/mask-yolov8-seg.onnx',
            labels_path='./labels/labels.json',
            use_cuda=args.use_cuda
        )
        
        image = cv2.imread(args.image)
        if image is None:
            raise ValueError(f"Could not load image from {args.image}")
        
        overlay, boxes = detector.detect_image(image)
        result = detector.visualize(image, overlay, boxes, remove_background=not args.keep_background)
        
        # Save the result as PNG to preserve transparency
        cv2.imwrite(args.output, result)
        print(f"Results saved to {args.output}")
        
        # For display purposes, create a version with a checkered background
        h, w = result.shape[:2]
        checkered = np.zeros((h, w, 3), dtype=np.uint8)
        checkered[::20, ::20] = 255
        checkered[10::20, 10::20] = 255
        
        # Blend the result with the checkered background
        alpha = result[:, :, 3:] / 255.0
        display_result = result[:, :, :3] * alpha + checkered * (1 - alpha)
        
        cv2.imshow('Result (with checkered background)', display_result.astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error: {e}")