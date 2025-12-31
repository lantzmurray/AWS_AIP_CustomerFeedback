# AWS Polly Audio Review Generation Guide

This guide explains how to use the audio review generation script that converts text reviews to audio files using AWS Polly text-to-speech service.

## Overview

The `generate_audio_reviews.py` script reads customer reviews from text files and converts them to audio files using AWS Polly. It supports multiple voices, languages, and output formats, with comprehensive error handling and metadata generation.

## Features

- **Multi-language Support**: Supports English (US, UK, AU), Spanish, French, and German
- **Voice Variety**: Multiple neural and standard voices for each language
- **Flexible Configuration**: JSON-based configuration with command-line overrides
- **Error Handling**: Comprehensive error handling with detailed logging
- **Metadata Generation**: Automatic generation of metadata for all audio files
- **Voice Selection Strategies**: Random, rotation, or first voice selection
- **Speech Customization**: Adjustable speech rate, pitch, and volume
- **Multiple Output Formats**: MP3, OGG Vorbis, PCM, and JSON with speech marks

## Prerequisites

1. **AWS Account**: You need an AWS account with access to AWS Polly
2. **AWS Credentials**: Configure your AWS credentials using one of the following methods:
   - AWS CLI: `aws configure`
   - Environment variables: `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
   - IAM role (if running on EC2 or other AWS services)
3. **Python 3.8+**: The script requires Python 3.8 or higher
4. **Required Dependencies**: Install dependencies from `requirements.txt`

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
1. Configuration file (`audio_config.json`)
2. Command-line arguments
3. Default values (built into the script)

### Configuration File Structure

The `audio_config.json` file contains settings such as:
- AWS region
- Input and output directories
- Voice selection preferences
- Output format settings
- Speech customization options
- Language and voice configurations

### Key Configuration Options

- `aws_region`: AWS region for Polly service (default: us-east-1)
- `text_reviews_dir`: Directory containing text review files (default: ../text_reviews)
- `audio_output_dir`: Directory for generated audio files (default: ../audio)
- `output_format`: Audio format - mp3, ogg_vorbis, pcm, json (default: mp3)
- `voice_selection`: Strategy for voice selection - random, rotation, first (default: rotation)
- `language_preference`: Primary language code (default: en-US)
- `use_neural_voices`: Use neural voices when available (default: true)
- `speech_rate`: Speech speed percentage (default: 100%)
- `pitch`: Voice pitch adjustment (default: 0%)
- `volume`: Volume adjustment in dB (default: 0dB)

## Usage

### Basic Usage

Run the script with default settings:
```bash
python generate_audio_reviews.py
```

### With Custom Configuration

Use a custom configuration file:
```bash
python generate_audio_reviews.py --config custom_audio_config.json
```

### Command-Line Options

- `--config PATH`: Path to configuration file
- `--region REGION`: AWS region (overrides config file)
- `--output-dir PATH`: Output directory for audio files (overrides config file)
- `--format FORMAT`: Output audio format - mp3, ogg_vorbis, pcm, json (overrides config file)
- `--language CODE`: Language code (overrides config file)
- `--voice STRATEGY`: Voice selection strategy - random, rotation, first (overrides config file)
- `--log-level LEVEL`: Logging level - DEBUG, INFO, WARNING, ERROR (overrides config file)

### Examples

Generate audio reviews with Spanish voices:
```bash
python generate_audio_reviews.py --language es-ES --voice random
```

Use a specific AWS region:
```bash
python generate_audio_reviews.py --region us-west-2
```

Generate OGG Vorbis format files:
```bash
python generate_audio_reviews.py --format ogg_vorbis
```

Enable debug logging:
```bash
python generate_audio_reviews.py --log-level DEBUG
```

## Supported Languages and Voices

### English (US)
- Joanna (Female, Neural)
- Matthew (Male, Neural)
- Ivy (Female, Neural)
- Justin (Male, Neural)
- Kendra (Female, Neural)
- Kimberly (Female, Neural)
- Salli (Female, Neural)
- Joey (Male, Neural)

### English (UK)
- Emma (Female, Neural)
- Brian (Male, Neural)
- Amy (Female, Neural)

### English (Australia)
- Nicole (Female, Neural)
- Russell (Male, Neural)

### Spanish (Spain)
- Lucia (Female, Neural)
- Enrique (Male, Neural)

### French (France)
- Céline (Female, Neural)
- Mathieu (Male, Neural)

### German (Germany)
- Marlene (Female, Neural)
- Hans (Male, Neural)

## Voice Selection Strategies

1. **Random**: Randomly selects a voice from the available pool
2. **Rotation**: Cycles through voices in order for each review
3. **First**: Always uses the first available voice

## Output Files

The script generates:

1. **Audio Files**: Named using the pattern `audio_{customer_id}_{voice_id}_{language}_{timestamp}.{format}`
2. **Metadata File**: `audio_generation_metadata.json` with comprehensive generation details
3. **Log File**: `audio_generation.log` with execution logs

### Audio File Naming Convention

Example: `audio_CUST-00001_Joanna_en-US_20231211_153022.mp3`

### Metadata Structure

The metadata file contains:
- Generation timestamp
- Processing statistics
- Configuration used
- Detailed results for each review
- Voice information
- File sizes and text lengths

## Error Handling

The script includes comprehensive error handling for:
- AWS credential issues
- Polly service errors
- File system errors
- Network connectivity issues
- Invalid configuration
- Text processing errors

If an audio generation fails, the script logs the error and continues with the next review.

## Troubleshooting

### Common Issues

1. **AWS Credentials Not Found**
   - Ensure your AWS credentials are properly configured
   - Check environment variables or ~/.aws/credentials file

2. **Access Denied Error**
   - Verify you have permission to access AWS Polly
   - Ensure Polly is available in your selected region

3. **Voice Not Available**
   - Check if the selected voice is available in your region
   - Some neural voices may not be available in all regions

4. **File Size Limit Exceeded**
   - Reduce the text length or adjust the max_file_size_mb configuration
   - Consider splitting long reviews into multiple parts

5. **Text Reviews Not Found**
   - Verify the text_reviews_dir path is correct
   - Ensure text files have .txt extension

### Debug Mode

Enable debug logging for more detailed error information:
```bash
python generate_audio_reviews.py --log-level DEBUG
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
                "polly:SynthesizeSpeech",
                "polly:DescribeVoices",
                "polly:ListLexicons"
            ],
            "Resource": "*"
        }
    ]
}
```

## Performance Considerations

1. **Rate Limiting**: The script includes a 0.5-second delay between requests to avoid rate limiting
2. **Batch Processing**: For large numbers of reviews, consider processing in batches
3. **File Size**: Monitor audio file sizes, especially for long reviews
4. **Neural Voices**: Neural voices provide better quality but may have different rate limits

## Customization

### Adding New Languages

1. Add the language code and voices to the `VOICE_OPTIONS` dictionary in the script
2. Update the configuration file with the new language settings
3. Ensure the selected AWS region supports the new language

### Custom Voice Presets

Create voice presets for different types of reviews:
```json
"voice_presets": {
    "positive_reviews": {
        "voice_ids": ["Joanna", "Matthew"],
        "speech_rate": "105%",
        "pitch": "+5%",
        "volume": "+2dB"
    }
}
```

### Speech Customization

Adjust speech parameters for different effects:
- `speech_rate`: 50-200% (default: 100%)
- `pitch`: -50% to +50% (default: 0%)
- `volume`: -20dB to +20dB (default: 0dB)

## Security Considerations

1. **AWS Credentials**: Never commit AWS credentials to version control
2. **IAM Permissions**: Use least-privilege access for your IAM role/user
3. **Data Privacy**: Be aware of AWS Polly's data handling policies
4. **File Access**: Ensure proper file permissions for input and output directories

## Integration with Other Scripts

This script can be integrated with other data processing pipelines:
1. Run after text review generation
2. Feed audio files into audio processing Lambda functions
3. Use metadata for tracking and analytics
4. Combine with image generation for multimedia content

## Support

For issues related to:
- AWS Polly service: Check AWS documentation and support
- Script functionality: Review the logs and configuration
- AWS account issues: Contact AWS support

## License

This script is provided as-is for educational and testing purposes.