#!/usr/bin/env python3
"""
Test script for the AWS Bedrock Image Generation script.
This script validates the functionality without actually calling AWS services.
"""

import json
import sys
from unittest.mock import patch, MagicMock
from generate_sample_images import ImageGenerator, load_config


def test_config_loading():
    """Test configuration loading."""
    print("Testing configuration loading...")
    
    # Test default config loading
    config = load_config("non_existent_config.json")
    assert config["aws_region"] == "us-east-1"
    assert config["model_id"] == "amazon.titan-image-generator-v1"
    assert config["image_count"] == 7
    print("✓ Default configuration loaded correctly")
    
    # Test actual config file loading
    config = load_config("config.json")
    assert "aws_region" in config
    assert "model_id" in config
    print("✓ Configuration file loaded correctly")
    
    return True


def test_image_generator_initialization():
    """Test ImageGenerator initialization."""
    print("\nTesting ImageGenerator initialization...")
    
    config = load_config("config.json")
    
    # Mock the boto3 client to avoid AWS credential requirement
    with patch('generate_sample_images.boto3.client') as mock_client:
        mock_bedrock = MagicMock()
        mock_client.return_value = mock_bedrock
        
        generator = ImageGenerator(config)
        
        # Verify the client was initialized with correct parameters
        mock_client.assert_called_once_with('bedrock-runtime', region_name=config['aws_region'])
        
        # Verify generator attributes
        assert generator.region == config['aws_region']
        assert generator.model_id == config['model_id']
        assert generator.output_dir.name == 'images'
        assert generator.image_count == config['image_count']
        
        print("✓ ImageGenerator initialized correctly")
        
    return True


def test_prompt_generation():
    """Test prompt generation."""
    print("\nTesting prompt generation...")
    
    config = load_config("config.json")
    
    with patch('generate_sample_images.boto3.client'):
        generator = ImageGenerator(config)
        prompts = generator.get_customer_feedback_prompts()
        
        assert len(prompts) == 10  # We have 10 predefined prompts
        
        # Check prompt structure
        for prompt_info in prompts:
            assert "prompt" in prompt_info
            assert "filename_prefix" in prompt_info
            assert "description" in prompt_info
            assert len(prompt_info["prompt"]) > 0
            assert len(prompt_info["filename_prefix"]) > 0
            assert len(prompt_info["description"]) > 0
        
        print(f"✓ Generated {len(prompts)} valid prompts")
        
    return True


def test_image_generation_mock():
    """Test image generation with mocked AWS response."""
    print("\nTesting image generation (mocked)...")
    
    config = load_config("config.json")
    
    # Mock the boto3 client and response
    with patch('generate_sample_images.boto3.client') as mock_client:
        # Setup mock response
        mock_response = {
            'body': MagicMock()
        }
        mock_response['body'].read.return_value = json.dumps({
            'images': ['iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==']  # Minimal PNG in base64
        }).encode('utf-8')
        
        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model.return_value = mock_response
        mock_client.return_value = mock_bedrock
        
        generator = ImageGenerator(config)
        
        # Test image generation
        prompt = "A test prompt for image generation"
        image_data = generator.generate_image(prompt)
        
        # Verify the image was generated
        assert image_data is not None
        assert len(image_data) > 0
        
        # Verify the correct API was called
        mock_bedrock.invoke_model.assert_called_once()
        
        # Check the request body
        call_args = mock_bedrock.invoke_model.call_args
        assert call_args[1]['modelId'] == config['model_id']
        assert call_args[1]['contentType'] == 'application/json'
        assert call_args[1]['accept'] == 'application/json'
        
        request_body = json.loads(call_args[1]['body'])
        assert request_body['textToImageParams']['text'] == prompt
        assert request_body['taskType'] == 'TEXT_IMAGE'
        
        print("✓ Image generation (mocked) works correctly")
        
    return True


def test_save_image_mock():
    """Test image saving with mocked file operations."""
    print("\nTesting image saving (mocked)...")
    
    config = load_config("config.json")
    
    with patch('generate_sample_images.boto3.client'):
        generator = ImageGenerator(config)
        
        # Mock file operations
        with patch('builtins.open', create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            # Test saving
            test_data = b"fake image data"
            filename = "test_image.png"
            
            result = generator.save_image(test_data, filename)
            
            assert result is True
            mock_open.assert_called_once_with(generator.output_dir / filename, 'wb')
            mock_file.write.assert_called_once_with(test_data)
            
            print("✓ Image saving (mocked) works correctly")
            
    return True


def main():
    """Run all tests."""
    print("Running tests for AWS Bedrock Image Generation Script\n")
    
    tests = [
        test_config_loading,
        test_image_generator_initialization,
        test_prompt_generation,
        test_image_generation_mock,
        test_save_image_mock
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"✗ {test.__name__} failed")
        except Exception as e:
            failed += 1
            print(f"✗ {test.__name__} failed with error: {str(e)}")
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("All tests passed! ✓")
        return 0
    else:
        print("Some tests failed! ✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())