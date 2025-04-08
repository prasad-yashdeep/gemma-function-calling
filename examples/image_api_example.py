"""
Example implementation of an image recognition API with the Gemma SDK.

This example demonstrates how to use the Gemma SDK with image-based APIs.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import base64
from PIL import Image
import io

from gemma_sdk import GemmaSDK, APIConverter
from gemma_sdk.runtime import ReActExecutor, ImageProcessor


# Define the image recognition API functions
IMAGE_API_SPEC = {
    "endpoints": [
        {
            "name": "analyze_image",
            "description": "Analyze an image and identify objects, scenes, or content",
            "image_processing": True,
            "parameters": [
                {
                    "name": "image",
                    "type": "image",
                    "description": "Image file path or URL to analyze",
                    "required": True
                },
                {
                    "name": "analysis_type",
                    "type": "string",
                    "description": "Type of analysis to perform",
                    "enum": ["objects", "scenes", "text", "general"],
                    "required": False
                }
            ]
        },
        {
            "name": "compare_images",
            "description": "Compare two images and determine similarity",
            "image_processing": True,
            "parameters": [
                {
                    "name": "image1",
                    "type": "image",
                    "description": "First image file path or URL",
                    "required": True
                },
                {
                    "name": "image2",
                    "type": "image",
                    "description": "Second image file path or URL",
                    "required": True
                }
            ]
        }
    ]
}


# Sample image analysis results
SAMPLE_ANALYSIS = {
    "objects": {
        "cat.jpg": ["cat", "sofa", "plant"],
        "dog.jpg": ["dog", "ball", "grass"],
        "city.jpg": ["buildings", "cars", "people", "traffic lights"],
        "beach.jpg": ["ocean", "sand", "palm trees", "people"],
        "food.jpg": ["plate", "pasta", "fork", "vegetables"]
    },
    "scenes": {
        "cat.jpg": "indoor living room",
        "dog.jpg": "outdoor park",
        "city.jpg": "urban cityscape",
        "beach.jpg": "tropical beach",
        "food.jpg": "restaurant dining"
    },
    "text": {
        "cat.jpg": "No text detected",
        "dog.jpg": "No text detected",
        "city.jpg": "MAIN ST, HOTEL, PARKING",
        "beach.jpg": "BEACH ACCESS, LIFEGUARD",
        "food.jpg": "MENU, SPECIAL"
    }
}


# Function implementations
def analyze_image(image: str, analysis_type: str = "general") -> Dict[str, Any]:
    """
    Analyze an image and identify objects, scenes, or content.
    
    Args:
        image: Base64-encoded image data, file path, or URL
        analysis_type: Type of analysis to perform
        
    Returns:
        Analysis results
    """
    # In a real implementation, this would call a computer vision API
    # For this example, we'll return mock data based on the image filename
    
    # Extract filename from path or use a default
    if "," in image and ";base64," in image:
        # This is a base64 image, use a default filename
        filename = "default.jpg"
    else:
        filename = os.path.basename(image).lower()
    
    # Find the closest match in our sample data
    best_match = "cat.jpg"  # Default
    for sample_file in SAMPLE_ANALYSIS["objects"].keys():
        if sample_file in filename or filename in sample_file:
            best_match = sample_file
            break
    
    # Return analysis based on type
    if analysis_type == "objects":
        return {
            "analysis_type": "objects",
            "objects_detected": SAMPLE_ANALYSIS["objects"].get(best_match, ["unknown"]),
            "confidence": 0.92
        }
    elif analysis_type == "scenes":
        return {
            "analysis_type": "scenes",
            "scene_type": SAMPLE_ANALYSIS["scenes"].get(best_match, "unknown"),
            "confidence": 0.89
        }
    elif analysis_type == "text":
        return {
            "analysis_type": "text",
            "text_detected": SAMPLE_ANALYSIS["text"].get(best_match, "No text detected"),
            "confidence": 0.78
        }
    else:  # general
        return {
            "analysis_type": "general",
            "objects": SAMPLE_ANALYSIS["objects"].get(best_match, ["unknown"]),
            "scene": SAMPLE_ANALYSIS["scenes"].get(best_match, "unknown"),
            "text": SAMPLE_ANALYSIS["text"].get(best_match, "No text detected"),
            "confidence": 0.85
        }


def compare_images(image1: str, image2: str) -> Dict[str, Any]:
    """
    Compare two images and determine similarity.
    
    Args:
        image1: Base64-encoded image data, file path, or URL for first image
        image2: Base64-encoded image data, file path, or URL for second image
        
    Returns:
        Comparison results
    """
    # In a real implementation, this would compute image similarity
    # For this example, we'll return mock data
    
    # Extract filenames from paths or use defaults
    if "," in image1 and ";base64," in image1:
        filename1 = "default1.jpg"
    else:
        filename1 = os.path.basename(image1).lower()
    
    if "," in image2 and ";base64," in image2:
        filename2 = "default2.jpg"
    else:
        filename2 = os.path.basename(image2).lower()
    
    # Determine if the images are similar based on filenames
    # In a real implementation, this would be based on image content
    if filename1 == filename2:
        similarity = 1.0
    elif filename1.split('.')[0] == filename2.split('.')[0]:
        similarity = 0.9
    else:
        # Check if they're in the same category
        category1 = None
        category2 = None
        
        for sample_file in SAMPLE_ANALYSIS["scenes"].keys():
            if sample_file in filename1 or filename1 in sample_file:
                category1 = SAMPLE_ANALYSIS["scenes"][sample_file]
            if sample_file in filename2 or filename2 in sample_file:
                category2 = SAMPLE_ANALYSIS["scenes"][sample_file]
        
        if category1 and category2 and category1 == category2:
            similarity = 0.7
        else:
            similarity = 0.2
    
    return {
        "similarity_score": similarity,
        "matching_features": int(similarity * 10),
        "is_same_scene": similarity > 0.6,
        "is_same_object": similarity > 0.8
    }


def create_sample_images():
    """Create sample images for testing."""
    # Create a directory for sample images
    sample_dir = Path("sample_images")
    sample_dir.mkdir(exist_ok=True)
    
    # Create a simple red square image
    red_img = Image.new('RGB', (100, 100), color='red')
    red_img.save(sample_dir / "red_square.jpg")
    
    # Create a simple blue square image
    blue_img = Image.new('RGB', (100, 100), color='blue')
    blue_img.save(sample_dir / "blue_square.jpg")
    
    # Create a simple cat-like shape (very basic)
    cat_img = Image.new('RGB', (200, 200), color='white')
    # Draw a simple cat face (very abstract)
    draw = ImageDraw.Draw(cat_img)
    # Head
    draw.ellipse((50, 50, 150, 150), fill='gray')
    # Ears
    draw.polygon([(50, 50), (70, 20), (90, 50)], fill='gray')
    draw.polygon([(150, 50), (130, 20), (110, 50)], fill='gray')
    # Eyes
    draw.ellipse((80, 80, 95, 95), fill='green')
    draw.ellipse((105, 80, 120, 95), fill='green')
    # Nose
    draw.polygon([(95, 100), (105, 100), (100, 110)], fill='pink')
    cat_img.save(sample_dir / "cat.jpg")
    
    return sample_dir


def main():
    # Get Hugging Face token from environment variable
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Please set the HF_TOKEN environment variable")
        return
    
    # Create sample images
    try:
        from PIL import ImageDraw
        sample_dir = create_sample_images()
        print(f"Created sample images in {sample_dir}")
    except Exception as e:
        print(f"Error creating sample images: {e}")
        print("Using existing images if available")
        sample_dir = Path("sample_images")
    
    # Save the image API spec to a file
    api_spec_path = Path("image_api_spec.json")
    with open(api_spec_path, "w") as f:
        json.dump(IMAGE_API_SPEC, f, indent=2)
    
    # Convert the API spec to Gemma function definitions
    converter = APIConverter()
    functions = converter.convert_from_image_api(api_spec_path)
    
    # Create function implementations
    implementations = {
        "analyze_image": analyze_image,
        "compare_images": compare_images
    }
    
    # Create a ReAct executor
    executor = ReActExecutor(
        model_name="google/gemma-7b",
        hf_token=hf_token,
        functions=functions
    )
    
    # Register function implementations
    for func_def in functions:
        func_name = func_def["name"]
        if func_name in implementations:
            executor.runtime.register_function(func_def, implementations[func_name])
    
    # Check if sample images exist
    cat_image_path = sample_dir / "cat.jpg"
    red_square_path = sample_dir / "red_square.jpg"
    blue_square_path = sample_dir / "blue_square.jpg"
    
    if cat_image_path.exists():
        # Execute queries
        print(f"Executing query: What objects are in this image? {cat_image_path}")
        result = executor.execute(f"What objects are in this image? {cat_image_path}")
        print("\nResult:")
        print(result)
    
    if red_square_path.exists() and blue_square_path.exists():
        print(f"\nExecuting query: Compare these two images: {red_square_path} and {blue_square_path}")
        result = executor.execute(f"Compare these two images: {red_square_path} and {blue_square_path}")
        print("\nResult:")
        print(result)


if __name__ == "__main__":
    main()
