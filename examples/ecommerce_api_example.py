"""
Example implementation of an e-commerce API with the Gemma SDK.

This example demonstrates how to use the Gemma SDK to create a product search and recommendation system.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

from gemma_sdk import GemmaSDK, APIConverter
from gemma_function_sdk.runtime import ReActExecutor


# Sample product database
PRODUCTS = [
    {
        "id": "p001",
        "name": "Smartphone X",
        "category": "Electronics",
        "price": 799.99,
        "description": "Latest smartphone with 6.5-inch display, 128GB storage, and triple camera system.",
        "in_stock": True
    },
    {
        "id": "p002",
        "name": "Wireless Headphones",
        "category": "Electronics",
        "price": 149.99,
        "description": "Noise-cancelling wireless headphones with 30-hour battery life.",
        "in_stock": True
    },
    {
        "id": "p003",
        "name": "Running Shoes",
        "category": "Sports",
        "price": 89.99,
        "description": "Lightweight running shoes with responsive cushioning.",
        "in_stock": True
    },
    {
        "id": "p004",
        "name": "Coffee Maker",
        "category": "Home",
        "price": 59.99,
        "description": "Programmable coffee maker with 12-cup capacity.",
        "in_stock": False
    },
    {
        "id": "p005",
        "name": "Laptop Pro",
        "category": "Electronics",
        "price": 1299.99,
        "description": "High-performance laptop with 16GB RAM and 512GB SSD.",
        "in_stock": True
    }
]


# Define the e-commerce API functions
ECOMMERCE_API_SPEC = {
    "endpoints": [
        {
            "name": "search_products",
            "description": "Search for products in the catalog",
            "parameters": [
                {
                    "name": "query",
                    "type": "string",
                    "description": "Search query or keywords",
                    "required": False
                },
                {
                    "name": "category",
                    "type": "string",
                    "description": "Filter by category",
                    "required": False
                },
                {
                    "name": "min_price",
                    "type": "number",
                    "description": "Minimum price",
                    "required": False
                },
                {
                    "name": "max_price",
                    "type": "number",
                    "description": "Maximum price",
                    "required": False
                },
                {
                    "name": "in_stock_only",
                    "type": "boolean",
                    "description": "Only show products that are in stock",
                    "required": False
                }
            ]
        },
        {
            "name": "get_product_details",
            "description": "Get detailed information about a specific product",
            "parameters": [
                {
                    "name": "product_id",
                    "type": "string",
                    "description": "Product ID",
                    "required": True
                }
            ]
        },
        {
            "name": "get_product_recommendations",
            "description": "Get product recommendations based on a product ID",
            "parameters": [
                {
                    "name": "product_id",
                    "type": "string",
                    "description": "Product ID to base recommendations on",
                    "required": True
                },
                {
                    "name": "limit",
                    "type": "integer",
                    "description": "Maximum number of recommendations to return",
                    "required": False
                }
            ]
        }
    ]
}


# Function implementations
def search_products(query: Optional[str] = None, category: Optional[str] = None, 
                   min_price: Optional[float] = None, max_price: Optional[float] = None,
                   in_stock_only: bool = False) -> Dict[str, Any]:
    """
    Search for products in the catalog.
    
    Args:
        query: Search query or keywords
        category: Filter by category
        min_price: Minimum price
        max_price: Maximum price
        in_stock_only: Only show products that are in stock
        
    Returns:
        List of matching products
    """
    results = PRODUCTS.copy()
    
    # Apply filters
    if query:
        query = query.lower()
        results = [p for p in results if query in p["name"].lower() or query in p["description"].lower()]
    
    if category:
        results = [p for p in results if p["category"].lower() == category.lower()]
    
    if min_price is not None:
        results = [p for p in results if p["price"] >= min_price]
    
    if max_price is not None:
        results = [p for p in results if p["price"] <= max_price]
    
    if in_stock_only:
        results = [p for p in results if p["in_stock"]]
    
    return {
        "count": len(results),
        "products": results
    }


def get_product_details(product_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific product.
    
    Args:
        product_id: Product ID
        
    Returns:
        Product details
    """
    for product in PRODUCTS:
        if product["id"] == product_id:
            return product
    
    return {"error": f"Product with ID {product_id} not found"}


def get_product_recommendations(product_id: str, limit: int = 3) -> Dict[str, Any]:
    """
    Get product recommendations based on a product ID.
    
    Args:
        product_id: Product ID to base recommendations on
        limit: Maximum number of recommendations to return
        
    Returns:
        List of recommended products
    """
    # Find the product
    product = None
    for p in PRODUCTS:
        if p["id"] == product_id:
            product = p
            break
    
    if not product:
        return {"error": f"Product with ID {product_id} not found"}
    
    # Get products in the same category
    same_category = [p for p in PRODUCTS if p["category"] == product["category"] and p["id"] != product_id]
    
    # Sort by price similarity
    same_category.sort(key=lambda p: abs(p["price"] - product["price"]))
    
    recommendations = same_category[:limit]
    
    return {
        "based_on": product["name"],
        "recommendations": recommendations
    }


def main():
    # Get Hugging Face token from environment variable
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Please set the HF_TOKEN environment variable")
        return
    
    # Save the e-commerce API spec to a file
    api_spec_path = Path("ecommerce_api_spec.json")
    with open(api_spec_path, "w") as f:
        json.dump(ECOMMERCE_API_SPEC, f, indent=2)
    
    # Convert the API spec to Gemma function definitions
    converter = APIConverter()
    functions = converter.convert_from_rest(api_spec_path)
    
    # Create function implementations
    implementations = {
        "search_products": search_products,
        "get_product_details": get_product_details,
        "get_product_recommendations": get_product_recommendations
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
    
    # Execute queries
    print("Executing query: Show me electronics under $200")
    result = executor.execute("Show me electronics under $200")
    print("\nResult:")
    print(result)
    
    print("\nExecuting query: Tell me about the Laptop Pro and recommend similar products")
    result = executor.execute("Tell me about the Laptop Pro and recommend similar products")
    print("\nResult:")
    print(result)


if __name__ == "__main__":
    main()
