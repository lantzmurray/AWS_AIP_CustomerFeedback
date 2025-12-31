#!/usr/bin/env python3
"""
AWS Bedrock Image Generation Script

This script generates sample images using Amazon Titan Image Generator model
through AWS Bedrock. The images are related to customer feedback and reviews
for testing the image processing pipeline.

Requirements:
- AWS credentials configured (environment variables, ~/.aws/credentials, or IAM role)
- Appropriate AWS permissions for Bedrock operations
- Python 3.8+

Usage:
    python generate_sample_images.py [--config config.json]
"""

import json
import logging
import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError, NoCredentialsError


class ImageGenerator:
    """Class to handle image generation using AWS Bedrock."""

    def __init__(self, config: Dict):
        """
        Initialize the ImageGenerator with configuration.

        Args:
            config: Configuration dictionary containing AWS settings
        """
        self.config = config
        self.region = config.get('aws_region', 'us-east-1')
        self.model_id = config.get('model_id', 'amazon.titan-image-generator-v1')
        self.output_dir = Path(config.get('output_dir', 'sample_data/images'))
        self.image_count = config.get('image_count', 7)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Bedrock client
        try:
            self.bedrock_client = boto3.client('bedrock-runtime', region_name=self.region)
            logging.info(f"Initialized Bedrock client for region: {self.region}")
        except NoCredentialsError:
            logging.error("AWS credentials not found. Please configure your credentials.")
            raise
        except Exception as e:
            logging.error(f"Failed to initialize Bedrock client: {str(e)}")
            raise

    def get_customer_feedback_prompts(self) -> List[Dict[str, str]]:
        """
        Generate prompts related to customer feedback and reviews.

        Returns:
            List of dictionaries containing prompt information
        """
        return [
            {
                "prompt": "A customer service representative helping a satisfied customer in a modern retail store, professional lighting, high quality",
                "filename_prefix": "customer_service_positive",
                "description": "Positive customer service interaction"
            },
            {
                "prompt": "A frustrated customer looking at a broken product, dramatic lighting, realistic style, customer complaint scenario",
                "filename_prefix": "customer_complaint",
                "description": "Customer complaint scenario"
            },
            {
                "prompt": "A team meeting analyzing customer feedback charts and graphs on a screen, corporate office environment, professional setting",
                "filename_prefix": "feedback_analysis",
                "description": "Team analyzing customer feedback"
            },
            {
                "prompt": "A customer writing a positive review on a smartphone, coffee shop setting, natural lighting, modern lifestyle",
                "filename_prefix": "writing_review",
                "description": "Customer writing a review"
            },
            {
                "prompt": "A product showcase with 5-star rating symbols, clean white background, commercial photography style",
                "filename_prefix": "product_rating",
                "description": "Product with high ratings"
            },
            {
                "prompt": "A customer satisfaction survey form on a tablet with checkmarks, office environment, professional documentation",
                "filename_prefix": "satisfaction_survey",
                "description": "Customer satisfaction survey"
            },
            {
                "prompt": "A dashboard with customer metrics and analytics, data visualization, modern UI design, business intelligence",
                "filename_prefix": "analytics_dashboard",
                "description": "Customer analytics dashboard"
            },
            {
                "prompt": "A diverse group of happy customers holding shopping bags, retail environment, joyful atmosphere, natural expressions",
                "filename_prefix": "happy_customers",
                "description": "Group of satisfied customers"
            },
            {
                "prompt": "A customer support agent wearing headset, call center environment, professional lighting, service industry",
                "filename_prefix": "support_agent",
                "description": "Customer support agent"
            },
            {
                "prompt": "A feedback form with handwritten notes, pen on paper, detailed comments, customer feedback documentation",
                "filename_prefix": "feedback_form",
                "description": "Written feedback form"
            }
        ]

    def generate_image(self, prompt: str) -> Optional[bytes]:
        """
        Generate an image using AWS Bedrock Titan Image Generator.

        Args:
            prompt: Text prompt for image generation

        Returns:
            Generated image as bytes, or None if generation failed
        """
        try:
            # Prepare the request body
            request_body = {
                "textToImageParams": {
                    "text": prompt
                },
                "taskType": "TEXT_IMAGE",
                "imageGenerationConfig": {
                    "numberOfImages": 1,
                    "quality": "standard",
                    "cfgScale": 8.0,
                    "seed": 42,
                    "height": 512,
                    "width": 512
                }
            }

            # Convert to JSON bytes
            request_body_json = json.dumps(request_body)

            # Invoke the model
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=request_body_json
            )

            # Parse the response
            response_body = json.loads(response.get('body').read())
            
            # Extract the base64 image data
            images = response_body.get('images', [])
            if not images:
                logging.error("No images returned in the response")
                return None

            # Decode the base64 image
            import base64
            image_data = base64.b64decode(images[0])
            
            return image_data

        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logging.error(f"AWS ClientError: {error_code} - {error_message}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error generating image: {str(e)}")
            return None

    def save_image(self, image_data: bytes, filename: str) -> bool:
        """
        Save image data to a file.

        Args:
            image_data: Image data as bytes
            filename: Output filename

        Returns:
            True if successful, False otherwise
        """
        try:
            filepath = self.output_dir / filename
            with open(filepath, 'wb') as f:
                f.write(image_data)
            logging.info(f"Image saved to: {filepath}")
            return True
        except Exception as e:
            logging.error(f"Failed to save image {filename}: {str(e)}")
            return False

    def generate_sample_images(self) -> Tuple[int, int]:
        """
        Generate multiple sample images based on customer feedback prompts.

        Returns:
            Tuple of (successful_generations, attempted_generations)
        """
        prompts = self.get_customer_feedback_prompts()
        successful = 0
        attempted = 0

        logging.info(f"Starting generation of {min(self.image_count, len(prompts))} images...")

        for i, prompt_info in enumerate(prompts[:self.image_count]):
            attempted += 1
            prompt = prompt_info['prompt']
            filename_prefix = prompt_info['filename_prefix']
            description = prompt_info['description']

            logging.info(f"Generating image {attempted}/{self.image_count}: {description}")
            logging.info(f"Prompt: {prompt}")

            # Generate the image
            image_data = self.generate_image(prompt)
            
            if image_data:
                # Create filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{filename_prefix}_{timestamp}.png"
                
                # Save the image
                if self.save_image(image_data, filename):
                    successful += 1
                    logging.info(f"Successfully generated and saved: {filename}")
                else:
                    logging.error(f"Failed to save image for: {description}")
            else:
                logging.error(f"Failed to generate image for: {description}")

            # Add a small delay to avoid rate limiting
            time.sleep(1)

        logging.info(f"Image generation complete. {successful}/{attempted} images generated successfully.")
        return successful, attempted

    def create_metadata_file(self, successful_generations: int, attempted_generations: int):
        """
        Create a metadata file with information about the generated images.

        Args:
            successful_generations: Number of successfully generated images
            attempted_generations: Number of attempted generations
        """
        metadata = {
            "generation_timestamp": datetime.now().isoformat(),
            "model_used": self.model_id,
            "aws_region": self.region,
            "attempted_generations": attempted_generations,
            "successful_generations": successful_generations,
            "success_rate": successful_generations / attempted_generations if attempted_generations > 0 else 0,
            "prompts_used": self.get_customer_feedback_prompts()[:self.image_count]
        }

        metadata_file = self.output_dir / "image_generation_metadata.json"
        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            logging.info(f"Metadata saved to: {metadata_file}")
        except Exception as e:
            logging.error(f"Failed to save metadata: {str(e)}")


def load_config(config_path: Optional[str] = None) -> Dict:
    """
    Load configuration from file or use defaults.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    default_config = {
        "aws_region": "us-east-1",
        "model_id": "amazon.titan-image-generator-v1",
        "output_dir": "sample_data/images",
        "image_count": 7,
        "log_level": "INFO"
    }

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            # Merge with defaults
            default_config.update(user_config)
            logging.info(f"Configuration loaded from: {config_path}")
        except Exception as e:
            logging.warning(f"Failed to load config from {config_path}: {str(e)}. Using defaults.")
    else:
        logging.info("Using default configuration")

    return default_config


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('image_generation.log')
        ]
    )


def main():
    """Main function to run the image generation script."""
    parser = argparse.ArgumentParser(description='Generate sample images using AWS Bedrock')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--count', type=int, help='Number of images to generate')
    parser.add_argument('--output-dir', type=str, help='Output directory for images')
    parser.add_argument('--region', type=str, help='AWS region')
    parser.add_argument('--log-level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override config with command line arguments
        if args.count:
            config['image_count'] = args.count
        if args.output_dir:
            config['output_dir'] = args.output_dir
        if args.region:
            config['aws_region'] = args.region
        if args.log_level:
            config['log_level'] = args.log_level

        logging.info("Starting AWS Bedrock Image Generation Script")
        logging.info(f"Configuration: {json.dumps(config, indent=2)}")

        # Initialize generator
        generator = ImageGenerator(config)

        # Generate images
        successful, attempted = generator.generate_sample_images()

        # Create metadata file
        generator.create_metadata_file(successful, attempted)

        # Exit with appropriate code
        if successful > 0:
            logging.info(f"Script completed successfully. Generated {successful} images.")
            sys.exit(0)
        else:
            logging.error("No images were generated successfully.")
            sys.exit(1)

    except Exception as e:
        logging.error(f"Script failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()