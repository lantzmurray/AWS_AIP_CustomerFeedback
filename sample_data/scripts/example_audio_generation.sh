#!/bin/bash

# Example script demonstrating how to use the audio review generation script

echo "AWS Polly Audio Review Generation Examples"
echo "========================================"
echo

# Ensure we're in the scripts directory
cd "$(dirname "$0")"

# Example 1: Basic usage with default settings
echo "Example 1: Basic audio generation with default settings"
echo "Command: python3 generate_audio_reviews.py"
echo "Description: Generates audio files for all text reviews using default configuration"
echo

# Example 2: Using custom configuration
echo "Example 2: Using custom audio configuration"
echo "Command: python3 generate_audio_reviews.py --config audio_config.json"
echo "Description: Uses the audio_config.json file for settings"
echo

# Example 3: Generate with Spanish voices
echo "Example 3: Generate audio with Spanish voices"
echo "Command: python3 generate_audio_reviews.py --language es-ES --voice random"
echo "Description: Uses Spanish voices with random selection"
echo

# Example 4: Generate OGG Vorbis format
echo "Example 4: Generate OGG Vorbis format files"
echo "Command: python3 generate_audio_reviews.py --format ogg_vorbis --region us-west-2"
echo "Description: Generates OGG Vorbis files using us-west-2 region"
echo

# Example 5: Debug mode with verbose logging
echo "Example 5: Debug mode with verbose logging"
echo "Command: python3 generate_audio_reviews.py --log-level DEBUG --voice rotation"
echo "Description: Runs with debug logging and voice rotation"
echo

# Example 6: Custom output directory
echo "Example 6: Custom output directory"
echo "Command: python3 generate_audio_reviews.py --output-dir ../custom_audio"
echo "Description: Saves audio files to a custom directory"
echo

echo "Prerequisites:"
echo "1. AWS credentials configured (aws configure or environment variables)"
echo "2. Required dependencies installed: pip install -r requirements.txt"
echo "3. Text review files in ../text_reviews directory"
echo

echo "To run any example, uncomment the corresponding line below:"
echo

# Uncomment one of the following lines to run:
# python3 generate_audio_reviews.py
# python3 generate_audio_reviews.py --config audio_config.json
# python3 generate_audio_reviews.py --language es-ES --voice random
# python3 generate_audio_reviews.py --format ogg_vorbis --region us-west-2
# python3 generate_audio_reviews.py --log-level DEBUG --voice rotation
# python3 generate_audio_reviews.py --output-dir ../custom_audio

echo "Make sure you have AWS credentials configured before running!"