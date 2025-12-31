# Sample Data Guide

## Overview

This guide provides a comprehensive overview of the sample data collection designed for testing the AWS AI project's multimodal data processing pipeline. The sample data represents realistic customer feedback across multiple modalities, including structured surveys, detailed text reviews, image prompts, and audio transcripts.

## Data Types and Relationships

### 1. Survey Data (Structured Feedback)
- **Location**: [`surveys/customer_feedback_survey.csv`](surveys/customer_feedback_survey.csv)
- **Format**: CSV with 55 customer records
- **Content**: Structured ratings, demographic indicators, and brief feedback
- **Key Fields**: Customer ID, ratings (1-5 scale), recommendation status, multimedia upload flags
- **Date Range**: July 12, 2025 - December 1, 2025

### 2. Text Reviews (Detailed Feedback)
- **Location**: [`text_reviews/review_CUST-XXXXX.txt`](text_reviews/)
- **Format**: Individual text files (150-300 words each)
- **Content**: Expanded narratives that complement survey responses
- **Coverage**: 14 customers with detailed reviews
- **Naming Convention**: `review_CUST-XXXXX.txt`

### 3. Image Prompts (Visual Feedback)
- **Location**: [`images/prompt_CUST-XXXXX.txt`](images/)
- **Format**: Text prompts for AI image generation
- **Content**: Detailed descriptions of customer-uploaded images
- **Coverage**: 10 customers who indicated image uploads
- **Naming Convention**: `prompt_CUST-XXXXX.txt`

### 4. Audio Transcripts (Voice Feedback)
- **Location**: [`audio/transcript_CUST-XXXXX.txt`](audio/)
- **Format**: Text transcripts with speech annotations
- **Content**: Conversational feedback with natural speech patterns
- **Coverage**: 8 customers who indicated audio uploads
- **Naming Convention**: `transcript_CUST-XXXXX.txt`

## Customer ID Relationships

All data types are linked through consistent customer IDs in the format `CUST-XXXXX`:

```
Survey Data (All 55 customers)
├── Text Reviews (14 customers)
├── Image Prompts (10 customers)
├── Audio Transcripts (8 customers)
└── Multimodal Customers (13 customers with both image and audio)
```

### Multimodal Data Distribution

| Data Type | Count | Percentage |
|-----------|-------|------------|
| Survey Only | 17 | 31% |
| Survey + Text | 8 | 15% |
| Survey + Images | 3 | 5% |
| Survey + Audio | 1 | 2% |
| Survey + Text + Images | 4 | 7% |
| Survey + Text + Audio | 2 | 4% |
| Survey + Images + Audio | 5 | 9% |
| All Four Types | 2 | 4% |
| Total | 55 | 100% |

## Dataset Statistics

### Rating Distribution

| Rating | Overall | Product Quality | Customer Service | Delivery Speed | Value for Money |
|--------|---------|-----------------|------------------|----------------|-----------------|
| 5 Stars | 25% | 27% | 25% | 24% | 27% |
| 4 Stars | 25% | 24% | 27% | 24% | 24% |
| 3 Stars | 25% | 24% | 24% | 27% | 24% |
| 2 Stars | 15% | 15% | 15% | 15% | 15% |
| 1 Star | 10% | 10% | 9% | 10% | 10% |

### Feedback Sentiment Analysis

| Sentiment | Text Reviews | Audio Transcripts |
|-----------|--------------|-------------------|
| Positive (4-5 stars) | 36% | 38% |
| Neutral (3 stars) | 29% | 25% |
| Negative (1-2 stars) | 35% | 37% |

### Temporal Distribution

| Month | Survey Responses | Text Reviews | Image Prompts | Audio Transcripts |
|-------|------------------|--------------|---------------|-------------------|
| July 2025 | 6 | 0 | 0 | 0 |
| August 2025 | 10 | 1 | 1 | 1 |
| September 2025 | 10 | 2 | 2 | 1 |
| October 2025 | 10 | 4 | 3 | 2 |
| November 2025 | 10 | 5 | 3 | 3 |
| December 2025 | 9 | 2 | 1 | 1 |

## Data Flow Architecture

```
Customer Feedback Collection
├── Structured Survey Data (CSV)
│   ├── Customer IDs
│   ├── Rating Scales
│   └── Multimedia Flags
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

## Integration with AWS AI Pipeline

### Processing Workflow

1. **Data Validation**
   - Quality checks using [`Code/data_validation/text_validation_lambda.py`](../Code/data_validation/text_validation_lambda.py)
   - Consistency validation across data types
   - Format and structure verification

2. **Multimodal Processing**
   - Text analysis with [`Code/multimodal_processing/text_processing_lambda.py`](../Code/multimodal_processing/text_processing_lambda.py)
   - Image processing with [`Code/multimodal_processing/image_processing_lambda.py`](../Code/multimodal_processing/image_processing_lambda.py)
   - Audio transcription with [`Code/multimodal_processing/audio_processing_lambda.py`](../Code/multimodal_processing/audio_processing_lambda.py)
   - Survey analysis with [`Code/multimodal_processing/survey_processing_script.py`](../Code/multimodal_processing/survey_processing_script.py)

3. **Data Formatting**
   - Foundation Model preparation with [`Code/fm_formatting/claude_formatting_lambda.py`](../Code/fm_formatting/claude_formatting_lambda.py)
   - Standardized output formats
   - Cross-modal correlation

## Use Cases for Sample Data

### 1. Text Processing Testing
- Sentiment analysis validation
- Entity extraction accuracy
- Key phrase identification
- Topic modeling development

### 2. Image Analysis Testing
- Object detection accuracy
- Text extraction from images
- Content moderation validation
- Visual sentiment analysis

### 3. Audio Processing Testing
- Speech-to-text accuracy
- Speaker identification
- Emotion detection from voice
- Conversation analysis

### 4. Multimodal Integration Testing
- Cross-modal data correlation
- Sentiment consistency validation
- Customer journey reconstruction
- Comprehensive feedback analysis

### 5. Quality Assurance Testing
- Data validation pipeline testing
- Error handling scenarios
- Performance benchmarking
- Scalability assessment

## Data Quality Characteristics

### Consistency Features
- Uniform customer ID format across all data types
- Consistent date formats (ISO 8601)
- Aligned sentiment indicators between modalities
- Coherent narrative threads across data types

### Diversity Features
- Balanced rating distribution
- Varied feedback length and detail
- Multiple product and service categories
- Different customer interaction patterns

### Realism Features
- Natural language patterns in text and audio
- Realistic customer scenarios
- Appropriate emotional expressions
- Plausible product experiences

## Limitations and Considerations

### Synthetic Data Constraints
- All data is artificially generated for testing purposes
- No real customer information or PII is included
- Product details are generalized rather than specific
- Geographic and demographic information is limited

### Coverage Limitations
- Limited to English language content
- Focused on product/service feedback scenarios
- No industry-specific terminology
- Simplified customer journey representations

### Usage Recommendations
- Use for pipeline testing and development only
- Supplement with real data for production deployments
- Consider expanding dataset for edge case testing
- Validate with domain-specific data when applicable

## Future Expansion Opportunities

### Data Volume Increases
- Expand customer base to 500+ records
- Add more temporal diversity
- Include seasonal variations
- Incorporate longitudinal customer data

### Modality Enhancements
- Add video content support
- Include chat conversation logs
- Incorporate social media feedback
- Add sensor/IoT data streams

### Diversity Improvements
- Multi-language support
- Cultural variation inclusion
- Demographic diversity expansion
- Industry-specific datasets

## Related Documentation

- [Usage Examples](usage_examples.md) - Step-by-step processing examples
- [Data Quality Report](data_quality_report.md) - Quality validation details
- [Survey Metadata](surveys/survey_metadata.md) - Survey-specific information
- [Text Reviews Metadata](text_reviews/text_reviews_metadata.md) - Text review details
- [Image Prompts Metadata](images/image_prompts_metadata.md) - Image prompt information
- [Audio Transcripts Metadata](audio/audio_transcripts_metadata.md) - Audio transcript details