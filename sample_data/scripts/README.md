# AWS Data Processing Scripts

This directory contains scripts for generating sample data using AWS services. Currently includes image generation and audio review generation capabilities.

## Available Scripts

### 1. Image Generation Script
The `generate_sample_images.py` script generates sample images using Amazon Titan Image Generator model through AWS Bedrock. The images are related to customer feedback and reviews, making them suitable for testing the image processing pipeline.

### 2. Audio Review Generation Script
The `generate_audio_reviews.py` script converts text reviews to audio files using AWS Polly text-to-speech service. It supports multiple voices, languages, and output formats with comprehensive error handling and metadata generation.

## Quick Start

### Image Generation
See the detailed guide in the original documentation below or run:
```bash
python generate_sample_images.py --help
```

### Audio Generation
For detailed instructions, see `AUDIO_GENERATION_GUIDE.md` or run:
```bash
python generate_audio_reviews.py --help
```

## Dependencies

Install all required dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Each script uses its own configuration file:
- `config.json` - For image generation
- `audio_config.json` - For audio generation

## AWS Permissions

Ensure your IAM role/user has permissions for:
- **Image Generation**: `bedrock:InvokeModel`
- **Audio Generation**: `polly:SynthesizeSpeech`, `polly:DescribeVoices`, `polly:ListLexicons`

---

# AWS Bedrock Image Generation Script (Original Documentation)

## Prerequisites

1. **AWS Account**: You need an AWS account with access to AWS Bedrock
2. **AWS Credentials**: Configure your AWS credentials using one of the following methods:
   - AWS CLI: `aws configure`
   - Environment variables: `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
   - IAM role (if running on EC2 or other AWS services)
3. **Bedrock Access**: Ensure you have access to the Titan Image Generator model in your AWS region
4. **Python 3.8+**: The script requires Python 3.8 or higher

## Installation

1. Navigate to the scripts directory:
   ```bash
   cd sample_data/scripts
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

The script can be configured using:
1. Command-line arguments
2. Configuration file (`config.json`)
3. Default values (built into the script)

### Configuration File

The `config.json` file contains settings such as:
- AWS region
- Model ID
- Output directory
- Number of images to generate
- Image generation parameters
- Custom prompts

You can modify this file to change the default behavior of the script.

## Usage

### Basic Usage

Run the script with default settings:
```bash
python generate_sample_images.py
```

### With Custom Configuration

Use a custom configuration file:
```bash
python generate_sample_images.py --config custom_config.json
```

### Command-Line Options

- `--config PATH`: Path to configuration file
- `--count NUMBER`: Number of images to generate (overrides config file)
- `--output-dir PATH`: Output directory for images (overrides config file)
- `--region REGION`: AWS region (overrides config file)
- `--log-level LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

### Examples

Generate 5 images with custom output directory:
```bash
python generate_sample_images.py --count 5 --output-dir ../custom_images
```

Use a specific AWS region:
```bash
python generate_sample_images.py --region us-west-2
```

Enable debug logging:
```bash
python generate_sample_images.py --log-level DEBUG
```

## Output

The script generates:
1. **Images**: PNG files with descriptive names and timestamps
2. **Metadata**: `image_generation_metadata.json` with generation details
3. **Log File**: `image_generation.log` with execution logs

### Image Naming Convention

Images are named using the pattern: `{prompt_type}_{timestamp}.png`

For example: `customer_service_positive_20231211_153022.png`

## Prompts

The script includes 10 predefined prompts related to customer feedback and reviews:

1. Customer service positive interaction
2. Customer complaint scenario
3. Team analyzing feedback
4. Customer writing a review
5. Product with high ratings
6. Customer satisfaction survey
7. Customer analytics dashboard
8. Group of satisfied customers
9. Customer support agent
10. Written feedback form

By default, the script generates 7 images (configurable), using the first 7 prompts.

## Error Handling

The script includes comprehensive error handling for:
- AWS credential issues
- Bedrock service errors
- File system errors
- Network connectivity issues

If an image generation fails, the script logs the error and continues with the next prompt.

## Troubleshooting

### Common Issues

1. **AWS Credentials Not Found**
   - Ensure your AWS credentials are properly configured
   - Check environment variables or ~/.aws/credentials file

2. **Access Denied Error**
   - Verify you have permission to access AWS Bedrock
   - Ensure the Titan Image Generator model is available in your region

3. **Model Not Available**
   - Check if the model is available in your selected AWS region
   - You may need to request access to the model through the AWS console

4. **Rate Limiting**
   - The script includes a 1-second delay between requests to avoid rate limiting
   - If you encounter rate limits, wait longer before retrying

### Debug Mode

Enable debug logging for more detailed error information:
```bash
python generate_sample_images.py --log-level DEBUG
```

## AWS IAM Permissions

Ensure your IAM user/role has the following permissions:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel"
            ],
            "Resource": "arn:aws:bedrock:*:*:model/amazon.titan-image-generator-v1"
        }
    ]
}
```

## Customization

### Adding New Prompts

You can add new prompts by:
1. Modifying the `get_customer_feedback_prompts()` method in the script
2. Adding prompts to the `config.json` file
3. Creating a custom configuration file with additional prompts

### Changing Image Parameters

Image generation parameters can be modified in the configuration file:
- `numberOfImages`: Number of images per prompt (always 1 for this script)
- `quality`: Image quality (standard, premium)
- `cfgScale`: Guidance scale (0.0-10.0)
- `seed`: Random seed for reproducible results
- `height` and `width`: Image dimensions

## Security Considerations

1. **AWS Credentials**: Never commit AWS credentials to version control
2. **IAM Permissions**: Use least-privilege access for your IAM role/user
3. **Data Privacy**: Be aware of AWS Bedrock's data handling policies

## Support

For issues related to:
- AWS Bedrock service: Check AWS documentation and support
- Script functionality: Review the logs and configuration
- AWS account issues: Contact AWS support

## License

This script is provided as-is for educational and testing purposes.