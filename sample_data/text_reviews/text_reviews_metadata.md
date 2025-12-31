# Text Reviews Metadata

## Overview
This directory contains detailed text review files that expand on the customer feedback collected in the survey data. These files provide rich, qualitative data that complements the structured survey responses and serves as test data for the text processing Lambda function.

## File Naming Convention
The text review files follow this naming pattern:
- `review_CUST-XXXXX.txt`
- Where `XXXXX` corresponds to customer IDs from the survey data (e.g., CUST-00001, CUST-00002, etc.)

## Relationship to Survey Data
Each text review file corresponds to a specific customer's survey response and maintains consistency with:
- The overall rating given in the survey
- Specific feedback themes mentioned in the survey
- The submission date of the survey
- Any additional context provided in the survey comments

The text reviews expand upon the brief survey comments, providing more detailed narratives that include:
- Specific product features and their performance
- Detailed customer service interactions
- Delivery and shipping experiences
- Suggestions for improvement
- Emotional context and sentiment

## Types of Reviews Included

### Positive Reviews (5-star ratings)
These reviews express high satisfaction with:
- Exceptional product quality and performance
- Outstanding customer service experiences
- Fast delivery and excellent packaging
- Strong value for money propositions
- High likelihood to recommend to others

Examples: `review_CUST-00001.txt`, `review_CUST-00004.txt`, `review_CUST-00025.txt`, `review_CUST-00034.txt`, `review_CUST-00047.txt`

### Mixed Reviews (3-4 star ratings)
These reviews provide balanced feedback with:
- Specific pros and cons
- Areas for improvement alongside positive aspects
- Generally satisfactory experiences with some reservations
- Constructive criticism and suggestions
- Moderate likelihood to recommend

Examples: `review_CUST-00002.txt`, `review_CUST-00010.txt`

### Negative Reviews (1-2 star ratings)
These reviews express dissatisfaction with:
- Product quality issues and failures
- Poor customer service experiences
- Delivery problems and damaged items
- Difficult return and refund processes
- Strong warnings to other potential customers

Examples: `review_CUST-00003.txt`, `review_CUST-00007.txt`, `review_CUST-00012.txt`, `review_CUST-00015.txt`, `review_CUST-00018.txt`, `review_CUST-00029.txt`, `review_CUST-00040.txt`

## Thematic Coverage

### Product Features
Many reviews mention specific product features, including:
- Build quality and materials
- Design and aesthetics
- Functionality and performance
- Ease of use and user interface
- Durability and longevity

### Customer Service Interactions
Reviews frequently detail customer service experiences, covering:
- Response times and availability
- Knowledge and helpfulness of representatives
- Problem resolution effectiveness
- Communication quality
- Follow-up and support quality

### Delivery and Shipping
Delivery experiences are commonly discussed, including:
- Shipping speed and timeliness
- Packaging quality and security
- Condition of items upon arrival
- Tracking and communication
- Delivery personnel professionalism

## Usage with Text Processing Pipeline

These text review files are designed to work seamlessly with the text processing Lambda function by providing:

1. **Rich Test Data**: Realistic, detailed reviews that simulate actual customer feedback
2. **Varied Content**: Different writing styles, lengths, and structures to test processing robustness
3. **Sentiment Diversity**: Clear positive, negative, and mixed sentiments for sentiment analysis testing
4. **Thematic Variety**: Multiple topics and themes for topic modeling and categorization
5. **Consistent Metadata**: Customer IDs and dates that link back to the structured survey data

### Processing Considerations
- Files are plain text format for easy ingestion
- Each file contains 150-300 words of detailed content
- Reviews use natural language with varied vocabulary and expressions
- Sentiments are clearly expressed but not always explicitly stated
- Reviews include both factual descriptions and emotional reactions

## Integration with Multimodal Data
These text reviews complement the other data types in the project:
- **Survey Data**: Provides structured ratings and brief comments
- **Text Reviews**: Offer detailed narratives and expanded feedback
- **Image Data**: May include product photos uploaded by customers
- **Audio Data**: Could contain voice recordings of customer feedback

Together, these data sources create a comprehensive customer feedback ecosystem for testing multimodal processing capabilities.