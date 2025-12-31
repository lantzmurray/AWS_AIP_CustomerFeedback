#!/usr/bin/env python3
"""
Test script for enhanced theme extraction implementation.
"""

import sys
import os

# Add parent directory to path to import survey processing module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import functions we want to test
from survey_processing_lambda import extract_common_themes

def test_theme_extraction():
    """Test enhanced theme extraction with sample comments."""
    
    # Sample survey comments for testing
    test_comments = [
        "The product quality is excellent and design is very intuitive",
        "Customer service was very helpful and responsive to my issues",
        "The price is reasonable for the value provided",
        "Delivery was fast and the package arrived in good condition",
        "The user interface is easy to navigate but some features are confusing",
        "Had some technical issues with the latest update",
        "Billing was straightforward but the payment process was slow",
        "Overall satisfaction is high with the product features"
    ]
    
    print("Testing enhanced theme extraction...")
    print(f"Input comments: {len(test_comments)}")
    
    # Test theme extraction
    themes = extract_common_themes(test_comments, "test-request-123")
    
    print(f"Extracted themes: {themes}")
    print(f"Number of themes: {len(themes)}")
    
    # Verify we got themes
    if themes:
        print("✅ Theme extraction successful")
        
        # Check for expected theme categories
        expected_themes = [
            'Product Quality', 'Customer Service', 'Price & Value', 
            'Delivery & Shipping', 'User Experience', 'Technical Issues',
            'Billing & Payment', 'Product Features'
        ]
        
        found_themes = [theme for theme in themes if theme in expected_themes]
        print(f"✅ Found {len(found_themes)} expected theme categories: {found_themes}")
        
        if len(found_themes) >= 3:
            print("✅ Good theme coverage achieved")
        else:
            print("⚠️ Limited theme coverage")
    else:
        print("❌ Theme extraction failed")
    
    return themes

if __name__ == "__main__":
    test_theme_extraction()