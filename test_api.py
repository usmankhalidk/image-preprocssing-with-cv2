#!/usr/bin/env python3
"""
Test script for the improved preprocess API
"""

import requests
import sys
import os

def test_preprocess_api(image_path, output_path="processed_image.jpg"):
    """
    Test the preprocess API with an image file
    
    Args:
        image_path: Path to input image
        output_path: Path to save processed image
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found")
        return False
    
    try:
        # Prepare the file for upload
        with open(image_path, 'rb') as f:
            files = {'image': f}
            
            # Send POST request to the API
            print(f"Processing image: {image_path}")
            response = requests.post('http://localhost:3000/api/preprocess', files=files)
        
        if response.status_code == 200:
            # Save the processed image
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ Success! Processed image saved to: {output_path}")
            print(f"📊 Response size: {len(response.content)} bytes")
            return True
        else:
            print(f"❌ Error: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API. Make sure the server is running on localhost:3000")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <image_path> [output_path]")
        print("Example: python test_api.py test_image.jpg processed.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "processed_image.jpg"
    
    success = test_preprocess_api(image_path, output_path)
    sys.exit(0 if success else 1)
