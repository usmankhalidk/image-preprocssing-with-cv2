"""
Image Preprocessing API with Flask and OpenCV
Converts images with deskewing, resizing, and compression
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import io
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure maximum file size (16MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def deskew(mat):
    """
    Deskew an image by detecting lines and rotating to correct angle.
    
    Args:
        mat: OpenCV image matrix (BGR or grayscale)
    
    Returns:
        Deskewed image matrix
    """
    # Convert to grayscale if needed
    if len(mat.shape) == 3:
        gray = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
    else:
        gray = mat.copy()
    
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Edge detection with better parameters
    edges = cv2.Canny(blurred, 50, 150)
    
    # Detect lines using Probabilistic Hough Transform with better parameters
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=20, maxLineGap=10)
    
    # If no lines detected, return original image
    if lines is None or len(lines) == 0:
        return mat
    
    # Calculate angles from detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        
        # Skip vertical lines
        if dx == 0:
            continue
        
        # Calculate angle in degrees
        angle = np.arctan2(dy, dx) * 180 / np.pi
        
        # Normalize angles to [-45, 45] range
        if abs(angle) > 45:
            angle -= np.sign(angle) * 90
        
        angles.append(angle)
    
    # If no valid angles found, return original
    if len(angles) == 0:
        return mat
    
    # Get median angle for robustness
    angles.sort()
    median_angle = angles[len(angles) // 2]
    
    # Skip rotation if angle is very small
    if abs(median_angle) < 0.5:
        return mat
    
    # Calculate new image dimensions after rotation
    h, w = mat.shape[:2]
    rad = median_angle * np.pi / 180
    abs_cos = abs(np.cos(rad))
    abs_sin = abs(np.sin(rad))
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)
    
    # Get rotation matrix
    center = (w / 2, h / 2)
    rot_mat = cv2.getRotationMatrix2D(center, -median_angle, 1.0)
    
    # Adjust translation to center the rotated image
    rot_mat[0, 2] += (new_w / 2) - center[0]
    rot_mat[1, 2] += (new_h / 2) - center[1]
    
    # Apply rotation with white background
    deskewed = cv2.warpAffine(
        mat, 
        rot_mat, 
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    
    return deskewed


def auto_adjust_brightness_contrast(image):
    """
    Automatically adjust brightness and contrast based on image statistics.
    
    Args:
        image: Grayscale image
    
    Returns:
        Adjusted image
    """
    # Calculate image statistics
    mean_intensity = np.mean(image)
    std_intensity = np.std(image)
    
    # More conservative target values for better results
    target_mean = 140  # Slightly brighter
    target_std = 50    # Less aggressive contrast
    
    # Calculate adjustment factors with more conservative limits
    if mean_intensity > 0:
        brightness_factor = target_mean / mean_intensity
    else:
        brightness_factor = 1.0
    
    if std_intensity > 0:
        contrast_factor = target_std / std_intensity
    else:
        contrast_factor = 1.0
    
    # More conservative limits to prevent over-processing
    brightness_factor = np.clip(brightness_factor, 0.7, 1.5)
    contrast_factor = np.clip(contrast_factor, 0.7, 1.5)
    
    # Apply brightness and contrast adjustment
    adjusted = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=(target_mean - mean_intensity * contrast_factor))
    
    return adjusted


def enhance_contrast(image):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better local contrast.
    
    Args:
        image: Grayscale image
    
    Returns:
        Enhanced image
    """
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # Apply CLAHE
    enhanced = clahe.apply(image)
    
    return enhanced


def apply_adaptive_thresholding(image):
    """
    Apply adaptive thresholding to make text clearer.
    
    Args:
        image: Grayscale image
    
    Returns:
        Thresholded image
    """
    # Apply Gaussian adaptive thresholding with better parameters
    thresh = cv2.adaptiveThreshold(
        image, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        15,  # larger block size for better text detection
        3    # higher C constant for better contrast
    )
    
    # Apply morphological operations to clean up the result
    # Use smaller kernel to preserve text details
    kernel = np.ones((1, 1), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Optional: Apply slight dilation to connect broken characters
    kernel_dilate = np.ones((1, 1), np.uint8)
    thresh = cv2.dilate(thresh, kernel_dilate, iterations=1)
    
    return thresh


def preprocess_image(image_data):
    """
    Optimized image preprocessing pipeline for OCR: resize, grayscale, auto-adjust, 
    deskew, denoise, adaptive thresholding, and compress.
    
    Args:
        image_data: Raw image bytes
    
    Returns:
        Compressed JPEG image bytes
    """
    try:
        # Decode image from bytes
        nparr = np.frombuffer(image_data, np.uint8)
        mat = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if mat is None:
            raise ValueError("Failed to decode image")
        
        # 1. RESIZE - Resize if too large (preserve aspect ratio)
        max_dim = 1600
        h, w = mat.shape[:2]
        
        if w > max_dim or h > max_dim:
            scale = max_dim / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            mat = cv2.resize(mat, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 2. CONVERT TO GRAYSCALE
        mat = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
        
        # 3. AUTO BRIGHTNESS/CONTRAST - Adjust early for better line detection
        mat = auto_adjust_brightness_contrast(mat)
        
        # 4. DESKEW - Fix perspective/skew (needs good contrast for line detection)
        mat = deskew(mat)
        
        # 5. DENOISE - Remove noise after deskewing
        mat = cv2.fastNlMeansDenoising(mat, None, h=8, templateWindowSize=7, searchWindowSize=21)
        
        # 6. ENHANCE CONTRAST - Apply CLAHE for better local contrast
        mat = enhance_contrast(mat)
        
        # 7. ADAPTIVE THRESHOLDING - Make text clearer (final step)
        mat = apply_adaptive_thresholding(mat)
        
        # 8. COMPRESS - Compress to <400KB with quality reduction
        max_size_kb = 400
        quality = 85
        
        while quality >= 60:
            # Encode image to JPEG with current quality
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            success, compressed = cv2.imencode('.jpg', mat, encode_param)
            
            if not success:
                raise ValueError("Failed to encode image")
            
            # Check size
            size_kb = len(compressed) / 1024
            
            if size_kb <= max_size_kb:
                break
            
            # Reduce quality and try again
            quality -= 5
        
        return compressed.tobytes()
        
    except Exception as e:
        print(f"Preprocessing error: {e}")
        # Return original image data on error
        return image_data


@app.route('/api/preprocess', methods=['POST'])
def preprocess():
    """
    API endpoint to preprocess uploaded images.
    Expects multipart/form-data with 'image' file field.
    Returns processed JPEG image.
    """
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        # Check if file is actually selected
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Read file data
        image_data = file.read()
        
        # Process the image
        processed_image = preprocess_image(image_data)
        
        # Return processed image
        return send_file(
            io.BytesIO(processed_image),
            mimetype='image/jpeg',
            as_attachment=False
        )
        
    except Exception as e:
        print(f'Preprocess API error: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200


if __name__ == '__main__':
    print('Server running on http://localhost:3000')
    app.run(host='0.0.0.0', port=3000, debug=True)