#!/usr/bin/env python3
"""
AWS Polly Text-to-Speech Audio Review Generator

This script converts text reviews to audio files using AWS Polly.
It reads text reviews from the text_reviews folder and generates
audio files with various voices and languages.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3
from botocore.exceptions import BotoCoreError, ClientError


class AudioReviewGenerator:
    """Generates audio reviews from text using AWS Polly."""

    # Available Polly voices with language support
    VOICE_OPTIONS = {
        "en-US": [
            {"id": "Joanna", "gender": "Female", "neural": True},
            {"id": "Matthew", "gender": "Male", "neural": True},
            {"id": "Ivy", "gender": "Female", "neural": True},
            {"id": "Justin", "gender": "Male", "neural": True},
            {"id": "Kendra", "gender": "Female", "neural": True},
            {"id": "Kimberly", "gender": "Female", "neural": True},
            {"id": "Salli", "gender": "Female", "neural": True},
            {"id": "Joey", "gender": "Male", "neural": True}
        ],
        "en-GB": [
            {"id": "Emma", "gender": "Female", "neural": True},
            {"id": "Brian", "gender": "Male", "neural": True},
            {"id": "Amy", "gender": "Female", "neural": True}
        ],
        "en-AU": [
            {"id": "Nicole", "gender": "Female", "neural": True},
            {"id": "Russell", "gender": "Male", "neural": True}
        ],
        "es-ES": [
            {"id": "Lucia", "gender": "Female", "neural": True},
            {"id": "Enrique", "gender": "Male", "neural": True}
        ],
        "fr-FR": [
            {"id": "Céline", "gender": "Female", "neural": True},
            {"id": "Mathieu", "gender": "Male", "neural": True}
        ],
        "de-DE": [
            {"id": "Marlene", "gender": "Female", "neural": True},
            {"id": "Hans", "gender": "Male", "neural": True}
        ]
    }

    # Output format options
    OUTPUT_FORMATS = ["mp3", "ogg_vorbis", "pcm", "json"]

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the audio review generator."""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # Initialize AWS Polly client
        try:
            self.polly_client = boto3.client(
                'polly',
                region_name=self.config.get('aws_region', 'us-east-1')
            )
            self.logger.info(f"Initialized Polly client for region: {self.config.get('aws_region', 'us-east-1')}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Polly client: {str(e)}")
            raise

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults."""
        default_config = {
            "aws_region": "us-east-1",
            "text_reviews_dir": "../text_reviews",
            "audio_output_dir": "../audio",
            "output_format": "mp3",
            "sample_rate": "22050",
            "voice_selection": "random",
            "language_preference": "en-US",
            "use_neural_voices": True,
            "max_file_size_mb": 25,
            "log_level": "INFO",
            "metadata_file": "audio_generation_metadata.json",
            "voice_rotation": True,
            "speech_rate": "100%",
            "pitch": "0%",
            "volume": "0dB"
        }

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                # Merge with defaults
                default_config.update(user_config)
                print(f"Loaded configuration from {config_path}")
            except Exception as e:
                print(f"Warning: Failed to load config file {config_path}: {str(e)}")
                print("Using default configuration")

        return default_config

    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('audio_review_generator')
        logger.setLevel(getattr(logging, self.config.get('log_level', 'INFO')))
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logger.level)
        
        # Create file handler
        log_file = os.path.join(
            os.path.dirname(__file__), 
            'audio_generation.log'
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logger.level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger

    def _get_text_reviews(self) -> List[Tuple[str, str]]:
        """Read text reviews from the text_reviews directory."""
        reviews_dir = os.path.join(os.path.dirname(__file__), self.config['text_reviews_dir'])
        reviews = []
        
        if not os.path.exists(reviews_dir):
            self.logger.error(f"Text reviews directory not found: {reviews_dir}")
            return reviews
        
        for filename in os.listdir(reviews_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(reviews_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    
                    # Extract customer ID from filename
                    customer_id = filename.replace('review_', '').replace('.txt', '')
                    reviews.append((customer_id, content))
                    self.logger.debug(f"Loaded review for customer: {customer_id}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to read review file {filename}: {str(e)}")
        
        self.logger.info(f"Loaded {len(reviews)} text reviews")
        return reviews

    def _select_voice(self, language: str, index: int = 0) -> Dict:
        """Select a voice based on configuration."""
        voices = self.VOICE_OPTIONS.get(language, self.VOICE_OPTIONS["en-US"])
        
        # Filter neural voices if specified
        if self.config.get('use_neural_voices', True):
            neural_voices = [v for v in voices if v.get('neural', False)]
            if neural_voices:
                voices = neural_voices
        
        # Select voice based on strategy
        voice_selection = self.config.get('voice_selection', 'random')
        
        if voice_selection == 'random':
            selected_voice = random.choice(voices)
        elif voice_selection == 'rotation':
            selected_voice = voices[index % len(voices)]
        elif voice_selection == 'first':
            selected_voice = voices[0]
        else:
            # Default to first voice
            selected_voice = voices[0]
        
        self.logger.debug(f"Selected voice: {selected_voice['id']} ({selected_voice['gender']})")
        return selected_voice

    def _generate_speech(self, text: str, voice_id: str, language: str) -> Optional[bytes]:
        """Generate speech from text using AWS Polly."""
        try:
            # Prepare speech marks request
            speech_params = {
                'Engine': 'neural' if self.config.get('use_neural_voices', True) else 'standard',
                'LanguageCode': language,
                'VoiceId': voice_id,
                'OutputFormat': self.config.get('output_format', 'mp3'),
                'SampleRate': self.config.get('sample_rate', '22050'),
                'Text': text,
                'TextType': 'text'
            }
            
            # Add speech rate and pitch if specified
            if self.config.get('speech_rate') != '100%':
                speech_params['SpeechMarkTypes'] = ['ssml']
                # Wrap text with SSML for rate control
                rate = self.config.get('speech_rate', '100%')
                pitch = self.config.get('pitch', '0%')
                volume = self.config.get('volume', '0dB')
                
                speech_params['Text'] = f"""
                <speak>
                    <prosody rate="{rate}" pitch="{pitch}" volume="{volume}">
                        {text}
                    </prosody>
                </speak>
                """
                speech_params['TextType'] = 'ssml'
            
            # Generate speech
            response = self.polly_client.synthesize_speech(**speech_params)
            
            # Read audio stream
            audio_data = response['AudioStream'].read()
            
            # Check file size
            max_size_bytes = self.config.get('max_file_size_mb', 25) * 1024 * 1024
            if len(audio_data) > max_size_bytes:
                self.logger.warning(f"Generated audio exceeds maximum size limit")
                return None
            
            return audio_data
            
        except (BotoCoreError, ClientError) as e:
            self.logger.error(f"AWS Polly error: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Error generating speech: {str(e)}")
            return None

    def _save_audio_file(self, customer_id: str, audio_data: bytes, voice_info: Dict) -> Optional[str]:
        """Save audio data to file."""
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.join(
                os.path.dirname(__file__), 
                self.config['audio_output_dir']
            )
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            voice_id = voice_info['id']
            language = voice_info.get('language', 'en-US')
            file_format = self.config.get('output_format', 'mp3')
            
            filename = f"audio_{customer_id}_{voice_id}_{language}_{timestamp}.{file_format}"
            file_path = os.path.join(output_dir, filename)
            
            # Save audio file
            with open(file_path, 'wb') as f:
                f.write(audio_data)
            
            self.logger.info(f"Saved audio file: {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Error saving audio file: {str(e)}")
            return None

    def _create_metadata(self, generation_results: List[Dict]) -> None:
        """Create metadata file with generation details."""
        try:
            metadata = {
                "generation_timestamp": datetime.now().isoformat(),
                "total_reviews_processed": len(generation_results),
                "successful_generations": sum(1 for r in generation_results if r.get('success')),
                "failed_generations": sum(1 for r in generation_results if not r.get('success')),
                "configuration": self.config,
                "results": generation_results
            }
            
            # Save metadata file
            output_dir = os.path.join(
                os.path.dirname(__file__), 
                self.config['audio_output_dir']
            )
            metadata_path = os.path.join(output_dir, self.config['metadata_file'])
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved metadata file: {metadata_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating metadata file: {str(e)}")

    def generate_audio_reviews(self) -> None:
        """Generate audio reviews from text reviews."""
        self.logger.info("Starting audio review generation")
        
        # Get text reviews
        reviews = self._get_text_reviews()
        if not reviews:
            self.logger.error("No text reviews found to process")
            return
        
        generation_results = []
        language = self.config.get('language_preference', 'en-US')
        
        # Process each review
        for index, (customer_id, text) in enumerate(reviews):
            self.logger.info(f"Processing review for customer: {customer_id}")
            
            # Select voice
            voice_info = self._select_voice(language, index)
            voice_id = voice_info['id']
            
            # Add language to voice info for metadata
            voice_info['language'] = language
            
            # Generate speech
            audio_data = self._generate_speech(text, voice_id, language)
            
            if audio_data:
                # Save audio file
                filename = self._save_audio_file(customer_id, audio_data, voice_info)
                
                if filename:
                    generation_results.append({
                        "customer_id": customer_id,
                        "success": True,
                        "audio_file": filename,
                        "voice_used": voice_info,
                        "text_length": len(text),
                        "audio_size_bytes": len(audio_data),
                        "generation_timestamp": datetime.now().isoformat()
                    })
                    self.logger.info(f"Successfully generated audio for customer: {customer_id}")
                else:
                    generation_results.append({
                        "customer_id": customer_id,
                        "success": False,
                        "error": "Failed to save audio file",
                        "voice_used": voice_info
                    })
            else:
                generation_results.append({
                    "customer_id": customer_id,
                    "success": False,
                    "error": "Failed to generate speech",
                    "voice_used": voice_info
                })
            
            # Add small delay to avoid rate limiting
            time.sleep(0.5)
        
        # Create metadata file
        self._create_metadata(generation_results)
        
        # Log summary
        successful = sum(1 for r in generation_results if r.get('success'))
        total = len(generation_results)
        self.logger.info(f"Audio generation completed: {successful}/{total} reviews processed successfully")


def main():
    """Main function to run the audio review generator."""
    parser = argparse.ArgumentParser(
        description='Generate audio reviews from text using AWS Polly'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--region',
        type=str,
        help='AWS region (overrides config)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for audio files (overrides config)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['mp3', 'ogg_vorbis', 'pcm', 'json'],
        help='Output audio format (overrides config)'
    )
    parser.add_argument(
        '--language',
        type=str,
        help='Language code (overrides config)'
    )
    parser.add_argument(
        '--voice',
        type=str,
        help='Voice selection strategy: random, rotation, first (overrides config)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (overrides config)'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize generator
        generator = AudioReviewGenerator(args.config)
        
        # Override config with command line arguments
        if args.region:
            generator.config['aws_region'] = args.region
        if args.output_dir:
            generator.config['audio_output_dir'] = args.output_dir
        if args.format:
            generator.config['output_format'] = args.format
        if args.language:
            generator.config['language_preference'] = args.language
        if args.voice:
            generator.config['voice_selection'] = args.voice
        if args.log_level:
            generator.config['log_level'] = args.log_level
            generator.logger.setLevel(getattr(logging, args.log_level))
        
        # Generate audio reviews
        generator.generate_audio_reviews()
        
    except KeyboardInterrupt:
        print("\nAudio generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()