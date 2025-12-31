# Customer Feedback Survey Dataset

## Overview
This dataset contains customer feedback survey responses collected over the past 6 months (July 2025 - December 2025). The dataset includes 55 survey records with varied ratings, feedback text, and multimedia upload indicators.

## Data Structure

### File Information
- **Filename**: `customer_feedback_survey.csv`
- **Format**: CSV (Comma-Separated Values)
- **Records**: 55 customer survey responses
- **Date Range**: July 12, 2025 - December 1, 2025

### Column Descriptions

| Column Name | Data Type | Description | Example Values |
|-------------|-----------|-------------|----------------|
| customer_id | String | Unique customer identifier in format CUST-XXXXX | CUST-00001 |
| submission_date | Date | Date of survey submission in YYYY-MM-DD format | 2025-12-01 |
| overall_rating | Integer | Overall satisfaction rating (1-5 stars) | 5 |
| product_quality_rating | Integer | Product quality rating (1-5 stars) | 4 |
| customer_service_rating | Integer | Customer service rating (1-5 stars) | 3 |
| delivery_speed_rating | Integer | Delivery speed rating (1-5 stars) | 4 |
| value_for_money_rating | Integer | Value for money rating (1-5 stars) | 5 |
| would_recommend | String | Would you recommend this product/service | Yes/No |
| feedback_text | String | Text feedback/comments from customer | "Excellent product quality!" |
| has_uploaded_image | String | Whether customer uploaded an image | Yes/No |
| has_uploaded_audio | String | Whether customer uploaded audio feedback | Yes/No |

## Data Assumptions and Generation Details

### Rating Scale
- **1 star**: Very dissatisfied
- **2 stars**: Dissatisfied
- **3 stars**: Neutral
- **4 stars**: Satisfied
- **5 stars**: Very satisfied

### Customer IDs
- Format: CUST-XXXXX (5-digit sequential numbering)
- Range: CUST-00001 to CUST-00055

### Submission Dates
- Span: July 12, 2025 to December 1, 2025 (approximately 5.5 months)
- Distribution: Relatively evenly distributed across the time period
- Format: ISO 8601 (YYYY-MM-DD)

### Feedback Text Characteristics
- Length: Varies from brief comments to detailed feedback
- Sentiment: Mix of positive, neutral, and negative feedback
- Topics: Product quality, customer service experience, delivery, value for money
- Realism: Includes realistic language patterns and customer concerns

### Multimedia Upload Distribution
- **Image uploads**: Approximately 40% of customers uploaded images (22 out of 55)
- **Audio uploads**: Approximately 36% of customers uploaded audio feedback (20 out of 55)
- **Both image and audio**: Approximately 24% of customers uploaded both (13 out of 55)
- **Neither**: Approximately 31% of customers uploaded neither (17 out of 55)

### Rating Distribution
- **5-star ratings**: ~25% of overall ratings
- **4-star ratings**: ~25% of overall ratings
- **3-star ratings**: ~25% of overall ratings
- **2-star ratings**: ~15% of overall ratings
- **1-star ratings**: ~10% of overall ratings

### Correlation Patterns
- Customers with higher overall ratings tend to have higher individual category ratings
- Customers who uploaded multimedia (images/audio) tend to provide more detailed feedback
- Negative feedback often includes specific complaints about product quality or customer service
- Positive feedback frequently mentions specific positive experiences

## Intended Use Cases

This dataset is designed to support testing of multimodal data processing pipelines, including:
- Text sentiment analysis
- Rating correlation analysis
- Customer feedback categorization
- Multimedia content processing workflows
- Time-based trend analysis

## Data Quality Notes

1. **Consistency**: All required fields are populated for every record
2. **Validity**: All ratings are within the 1-5 range
3. **Realism**: Feedback text reflects realistic customer experiences
4. **Diversity**: Dataset includes varied customer experiences and satisfaction levels
5. **Temporal Distribution**: Submission dates are distributed across the specified time period

## Limitations

1. **Synthetic Data**: This is artificially generated data for testing purposes
2. **Customer Demographics**: No demographic information is included
3. **Product Details**: Specific product information is not included
4. **Geographic Data**: No location information is provided
5. **Purchase History**: No transaction or purchase history data is included