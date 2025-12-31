#!/usr/bin/env python3
"""
Test script for audio review generation functionality.

This script performs basic tests to verify the audio generation script
is working correctly without actually calling AWS Polly.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add the current directory to the path to import our script
sys.path.insert(0, os.path.dirname(__file__))

from generate_audio_reviews import AudioReviewGenerator


class TestAudioReviewGenerator(unittest.TestCase):
    """Test cases for AudioReviewGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_config = {
            "aws_region": "us-east-1",
            "text_reviews_dir": "../text_reviews",
            "audio_output_dir": "../audio",
            "output_format": "mp3",
            "sample_rate": "22050",
            "voice_selection": "rotation",
            "language_preference": "en-US",
            "use_neural_voices": True,
            "max_file_size_mb": 25,
            "log_level": "ERROR",  # Reduce noise during tests
            "metadata_file": "test_audio_metadata.json",
            "voice_rotation": True,
            "speech_rate": "100%",
            "pitch": "0%",
            "volume": "0dB"
        }
        
        # Create a temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(self.test_config, self.temp_config)
        self.temp_config.close()

    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_config.name)

    @patch('boto3.client')
    def test_initialization(self, mock_boto_client):
        """Test that the generator initializes correctly."""
        mock_polly = Mock()
        mock_boto_client.return_value = mock_polly
        
        generator = AudioReviewGenerator(self.temp_config.name)
        
        self.assertEqual(generator.config['aws_region'], 'us-east-1')
        self.assertEqual(generator.config['output_format'], 'mp3')
        mock_boto_client.assert_called_once_with('polly', region_name='us-east-1')

    def test_load_config_defaults(self):
        """Test loading default configuration when no file is provided."""
        generator = AudioReviewGenerator()
        
        self.assertIn('aws_region', generator.config)
        self.assertIn('output_format', generator.config)
        self.assertEqual(generator.config['output_format'], 'mp3')

    def test_voice_selection(self):
        """Test voice selection strategies."""
        with patch('boto3.client'):
            generator = AudioReviewGenerator(self.temp_config.name)
            
            # Test rotation strategy
            generator.config['voice_selection'] = 'rotation'
            voice1 = generator._select_voice('en-US', 0)
            voice2 = generator._select_voice('en-US', 1)
            
            # Should get different voices for different indices
            self.assertNotEqual(voice1['id'], voice2['id'])
            
            # Test first strategy
            generator.config['voice_selection'] = 'first'
            voice3 = generator._select_voice('en-US', 5)
            voice4 = generator._select_voice('en-US', 10)
            
            # Should always get the first voice
            self.assertEqual(voice3['id'], voice4['id'])

    def test_get_text_reviews(self):
        """Test reading text reviews from directory."""
        with patch('boto3.client'):
            generator = AudioReviewGenerator(self.temp_config.name)
            
            # Create temporary test files
            with tempfile.TemporaryDirectory() as temp_dir:
                generator.config['text_reviews_dir'] = temp_dir
                
                # Create test review files
                test_reviews = [
                    ('review_CUST-001.txt', 'Test review 1 content'),
                    ('review_CUST-002.txt', 'Test review 2 content'),
                    ('not_a_review.txt', 'Should not be included')
                ]
                
                for filename, content in test_reviews:
                    filepath = os.path.join(temp_dir, filename)
                    with open(filepath, 'w') as f:
                        f.write(content)
                
                reviews = generator._get_text_reviews()
                
                # Should only include files starting with 'review_'
                self.assertEqual(len(reviews), 2)
                # Sort by customer ID to ensure consistent order
                reviews.sort(key=lambda x: x[0])
                self.assertEqual(reviews[0][0], 'CUST-001')
                self.assertEqual(reviews[0][1], 'Test review 1 content')

    @patch('boto3.client')
    def test_generate_speech_success(self, mock_boto_client):
        """Test successful speech generation."""
        # Mock Polly client response
        mock_response = {
            'AudioStream': Mock()
        }
        mock_response['AudioStream'].read.return_value = b'fake_audio_data'
        
        mock_polly = Mock()
        mock_polly.synthesize_speech.return_value = mock_response
        mock_boto_client.return_value = mock_polly
        
        generator = AudioReviewGenerator(self.temp_config.name)
        
        audio_data = generator._generate_speech('Test text', 'Joanna', 'en-US')
        
        self.assertIsNotNone(audio_data)
        self.assertEqual(audio_data, b'fake_audio_data')
        mock_polly.synthesize_speech.assert_called_once()

    @patch('boto3.client')
    def test_generate_speech_failure(self, mock_boto_client):
        """Test speech generation failure handling."""
        # Mock Polly client to raise an exception
        mock_polly = Mock()
        mock_polly.synthesize_speech.side_effect = Exception("AWS Error")
        mock_boto_client.return_value = mock_polly
        
        generator = AudioReviewGenerator(self.temp_config.name)
        
        audio_data = generator._generate_speech('Test text', 'Joanna', 'en-US')
        
        self.assertIsNone(audio_data)

    def test_save_audio_file(self):
        """Test saving audio files."""
        with patch('boto3.client'):
            generator = AudioReviewGenerator(self.temp_config.name)
            
            # Create temporary directory for output
            with tempfile.TemporaryDirectory() as temp_dir:
                generator.config['audio_output_dir'] = temp_dir
                
                voice_info = {'id': 'Joanna', 'gender': 'Female', 'language': 'en-US'}
                audio_data = b'fake_audio_data'
                
                filename = generator._save_audio_file('CUST-001', audio_data, voice_info)
                
                self.assertIsNotNone(filename)
                self.assertTrue(filename.startswith('audio_CUST-001_Joanna_en-US_'))
                self.assertTrue(filename.endswith('.mp3'))
                
                # Check file was actually created
                filepath = os.path.join(temp_dir, filename)
                self.assertTrue(os.path.exists(filepath))
                
                # Check file contents
                with open(filepath, 'rb') as f:
                    saved_data = f.read()
                self.assertEqual(saved_data, audio_data)

    def test_create_metadata(self):
        """Test metadata file creation."""
        with patch('boto3.client'):
            generator = AudioReviewGenerator(self.temp_config.name)
            
            # Create temporary directory for output
            with tempfile.TemporaryDirectory() as temp_dir:
                generator.config['audio_output_dir'] = temp_dir
                
                test_results = [
                    {
                        'customer_id': 'CUST-001',
                        'success': True,
                        'audio_file': 'test1.mp3',
                        'voice_used': {'id': 'Joanna'}
                    },
                    {
                        'customer_id': 'CUST-002',
                        'success': False,
                        'error': 'Test error'
                    }
                ]
                
                generator._create_metadata(test_results)
                
                # Check metadata file was created
                metadata_path = os.path.join(temp_dir, 'test_audio_metadata.json')
                self.assertTrue(os.path.exists(metadata_path))
                
                # Check metadata contents
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.assertEqual(metadata['total_reviews_processed'], 2)
                self.assertEqual(metadata['successful_generations'], 1)
                self.assertEqual(metadata['failed_generations'], 1)
                self.assertIn('results', metadata)
                self.assertEqual(len(metadata['results']), 2)


def run_integration_test():
    """Run a basic integration test with real files."""
    print("Running integration test...")
    
    # Check if text reviews directory exists and has files
    text_dir = os.path.join(os.path.dirname(__file__), '../text_reviews')
    if not os.path.exists(text_dir):
        print(f"Text reviews directory not found: {text_dir}")
        return False
    
    text_files = [f for f in os.listdir(text_dir) if f.endswith('.txt')]
    if not text_files:
        print("No text review files found")
        return False
    
    print(f"Found {len(text_files)} text review files")
    
    # Test configuration loading
    try:
        generator = AudioReviewGenerator()
        print("✓ Generator initialized successfully")
        print(f"✓ Default config loaded: {generator.config['output_format']} format")
        print(f"✓ Language preference: {generator.config['language_preference']}")
        print(f"✓ Voice selection: {generator.config['voice_selection']}")
    except Exception as e:
        print(f"✗ Failed to initialize generator: {e}")
        return False
    
    # Test voice selection
    try:
        voice = generator._select_voice('en-US', 0)
        print(f"✓ Voice selection works: {voice['id']} ({voice['gender']})")
    except Exception as e:
        print(f"✗ Voice selection failed: {e}")
        return False
    
    # Test reading text reviews
    try:
        reviews = generator._get_text_reviews()
        print(f"✓ Successfully loaded {len(reviews)} text reviews")
        if reviews:
            print(f"  First review: {reviews[0][0]} ({len(reviews[0][1])} characters)")
    except Exception as e:
        print(f"✗ Failed to load text reviews: {e}")
        return False
    
    print("✓ Integration test passed!")
    return True


if __name__ == '__main__':
    print("Audio Review Generation Test Suite")
    print("=" * 40)
    
    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n" + "=" * 40)
    
    # Run integration test
    success = run_integration_test()
    
    if success:
        print("\n🎉 All tests passed! The audio generation script is ready to use.")
        print("\nTo generate audio files, run:")
        print("python generate_audio_reviews.py")
        print("\nMake sure you have AWS credentials configured before running.")
    else:
        print("\n❌ Some tests failed. Please check the configuration and try again.")
        sys.exit(1)