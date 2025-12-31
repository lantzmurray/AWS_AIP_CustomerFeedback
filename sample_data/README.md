# Sample Data Structure

This directory contains comprehensive sample data files for the AWS AI project, organized by data type to support the multimodal data processing pipeline. The data represents realistic customer feedback across multiple modalities.

## Quick Start

1. **Understand the Data Structure**: Read the [`sample_data_guide.md`](sample_data_guide.md) for a complete overview of the data collection
2. **Try the Examples**: Follow step-by-step tutorials in [`usage_examples.md`](usage_examples.md)
3. **Check Data Quality**: Review quality validation in [`data_quality_report.md`](data_quality_report.md)
4. **Process the Data**: Use the Lambda functions in [`Code/multimodal_processing/`](../Code/multimodal_processing/)

```bash
# Quick test with survey data
python Code/multimodal_processing/survey_processing_script.py --input-path sample_data/surveys --output-path output

# Process text reviews
python Code/multimodal_processing/text_processing_lambda.py

# Process image data
python Code/multimodal_processing/image_processing_lambda.py

# Process audio data
python Code/multimodal_processing/audio_processing_lambda.py
```

## Documentation

| Document | Purpose |
|----------|---------|
| [`sample_data_guide.md`](sample_data_guide.md) | Comprehensive overview of the entire sample data collection |
| [`usage_examples.md`](usage_examples.md) | Step-by-step examples and code snippets for data processing |
| [`data_quality_report.md`](data_quality_report.md) | Quality checks and validation results |
| [`surveys/survey_metadata.md`](surveys/survey_metadata.md) | Detailed information about survey data |
| [`text_reviews/text_reviews_metadata.md`](text_reviews/text_reviews_metadata.md) | Text review specifications and details |
| [`images/image_prompts_metadata.md`](images/image_prompts_metadata.md) | Image prompt information and usage |
| [`audio/audio_transcripts_metadata.md`](audio/audio_transcripts_metadata.md) | Audio transcript specifications |

## Directory Structure

### text_reviews/
Contains customer text feedback files that will be processed by the text processing pipeline.
- **Purpose**: Store raw customer reviews, feedback, and comments
- **Processing**: Text analysis, sentiment analysis, and feature extraction
- **Related Code**: [`Code/multimodal_processing/text_processing_lambda.py`](../Code/multimodal_processing/text_processing_lambda.py)
- **Files**: 14 detailed text reviews (150-300 words each)
- **Format**: Individual text files with naming convention `review_CUST-XXXXX.txt`

### images/
Contains image generation prompts and sample images for the image processing pipeline.
- **Purpose**: Store input prompts and generated images
- **Processing**: Image analysis, classification, and generation
- **Related Code**: [`Code/multimodal_processing/image_processing_lambda.py`](../Code/multimodal_processing/image_processing_lambda.py)
- **Files**: 10 image prompt files
- **Format**: Text prompts with naming convention `prompt_CUST-XXXXX.txt`

### audio/
Contains audio transcripts and generated audio files for the audio processing pipeline.
- **Purpose**: Store audio recordings and their transcriptions
- **Processing**: Speech-to-text, audio analysis, and audio generation
- **Related Code**: [`Code/multimodal_processing/audio_processing_lambda.py`](../Code/multimodal_processing/audio_processing_lambda.py)
- **Files**: 8 transcript files with speech annotations
- **Format**: Text transcripts with naming convention `transcript_CUST-XXXXX.txt`

### surveys/
Contains CSV survey data files for the survey processing pipeline.
- **Purpose**: Store structured survey responses and questionnaires
- **Processing**: Data validation, statistical analysis, and aggregation
- **Related Code**: [`Code/multimodal_processing/survey_processing_script.py`](../Code/multimodal_processing/survey_processing_script.py)
- **Files**: [`customer_feedback_survey.csv`](surveys/customer_feedback_survey.csv) with 55 customer records
- **Format**: CSV with customer IDs, ratings, and multimedia upload flags

## Data Flow Architecture

```
Customer Feedback Collection
├── Structured Survey Data (CSV)
│   ├── Customer IDs (CUST-00001 to CUST-00055)
│   ├── Rating Scales (1-5 stars)
│   └── Multimedia Flags (image/audio indicators)
├── Unstructured Text Reviews
│   ├── Detailed Narratives
│   ├── Sentiment Indicators
│   └── Thematic Content
├── Visual Content (Images)
│   ├── Product Showcase
│   ├── Issue Documentation
│   └── Comparison Images
└── Audio Content (Voice)
    ├── Speech Transcripts
    ├── Emotional Indicators
    └── Conversation Patterns
```

## Data Processing Pipeline

The sample data in these directories follows the processing workflow described in:
- [`Architecture/multimodal_data_processing.md`](../Architecture/multimodal_data_processing.md)
- [`Implementation_Guide/part2_multimodal_data_processing.md`](../Implementation_Guide/part2_multimodal_data_processing.md)

### Processing Steps

1. **Data Validation**
   - Quality checks using [`Code/data_validation/text_validation_lambda.py`](../Code/data_validation/text_validation_lambda.py)
   - Consistency validation across data types
   - Format and structure verification

2. **Multimodal Processing**
   - Text analysis with Amazon Comprehend
   - Image analysis with Amazon Rekognition
   - Audio transcription with Amazon Transcribe
   - Survey analysis with statistical methods

3. **Data Formatting**
   - Foundation Model preparation
   - Standardized output formats
   - Cross-modal correlation

## Key Statistics

| Metric | Value |
|--------|-------|
| Total Customers | 55 |
| Survey Responses | 55 |
| Text Reviews | 14 |
| Image Prompts | 10 |
| Audio Transcripts | 8 |
| Multimodal Customers | 13 (both image and audio) |
| Date Range | July 12, 2025 - December 1, 2025 |
| Rating Distribution | 25% (5★), 25% (4★), 25% (3★), 15% (2★), 10% (1★) |

## Usage Examples

### Basic Processing

```python
# Process survey data
import pandas as pd
df = pd.read_csv('sample_data/surveys/customer_feedback_survey.csv')
print(f"Loaded {len(df)} survey responses")

# Process text reviews
import glob
for file_path in glob.glob('sample_data/text_reviews/review_*.txt'):
    with open(file_path, 'r') as f:
        content = f.read()
    # Process content with text_processing_lambda.py
```

### Advanced Integration

```python
# Complete customer profile processing
def process_customer(customer_id):
    # Process all available data types for a customer
    survey_data = get_survey_data(customer_id)
    text_review = get_text_review(customer_id)
    image_data = get_image_data(customer_id)
    audio_data = get_audio_data(customer_id)
    
    # Create consolidated profile
    return create_customer_profile(survey_data, text_review, image_data, audio_data)

# Example: process_customer('CUST-00001')
```

## Data Quality

All sample data meets the quality standards defined in:
- [`Architecture/data_quality_enhancement.md`](../Architecture/data_quality_enhancement.md)
- [`Code/data_validation/glue_data_quality_ruleset.py`](../Code/data_validation/glue_data_quality_ruleset.py)

### Quality Features
- **Consistency**: Uniform customer ID format across all data types
- **Diversity**: Balanced rating distribution and varied feedback scenarios
- **Realism**: Natural language patterns and plausible customer experiences
- **Integration**: Cross-modal data correlation and consistency

## Notes

- This directory structure is designed to support the multimodal data processing pipeline
- Each subdirectory corresponds to a specific data type and processing workflow
- The .gitkeep files ensure empty directories are tracked in version control
- All data is synthetically generated for testing purposes (no real customer information)
- Customer IDs are consistent across all data types for cross-modal analysis