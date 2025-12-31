# Image Prompts Metadata

## File Naming Convention

The image prompt files follow the naming convention: `prompt_CUST-XXXXX.txt`

- `prompt_` - Prefix indicating the file contains an image generation prompt
- `CUST-XXXXX` - Customer ID from the survey data who has uploaded an image
- `.txt` - File extension

## Relationship to Survey Data

Each image prompt is directly linked to a customer's feedback from the customer feedback survey. The prompts are created for customers who have indicated they uploaded images (`has_uploaded_image = "Yes"` in the survey data). 

Each prompt file contains:
- Customer ID matching the survey data
- Date of image upload matching the survey submission date
- A detailed image generation prompt that reflects the customer's feedback and rating

## Types of Image Prompts Included

### 1. Product Showcase Images
- **Purpose**: Display products in their best light
- **Characteristics**: Professional studio lighting, clean backgrounds, sharp focus
- **Examples**: `prompt_CUST-00001.txt`, `prompt_CUST-00014.txt`, `prompt_CUST-00016.txt`

### 2. Problem/Demonstration Images
- **Purpose**: Show issues with products or negative experiences
- **Characteristics**: Dramatic lighting, visible damage, documentary style
- **Examples**: `prompt_CUST-00003.txt`, `prompt_CUST-00007.txt`, `prompt_CUST-00018.txt`

### 3. Comparison Images
- **Purpose**: Show before/after scenarios or product comparisons
- **Characteristics**: Split composition, contrasting elements, clear division
- **Examples**: `prompt_CUST-00006.txt`

### 4. Lifestyle Images
- **Purpose**: Show how products fit into daily life
- **Characteristics**: Natural settings, human interaction, contextual elements
- **Examples**: `prompt_CUST-00012.txt`

### 5. Detail Shots
- **Purpose**: Focus on specific features or issues
- **Characteristics**: Macro photography, shallow depth of field, focused details
- **Examples**: `prompt_CUST-00010.txt`

### 6. Standard Product Images
- **Purpose**: Show average or typical product appearance
- **Characteristics**: Neutral lighting, straightforward composition, realistic but not enhanced
- **Examples**: `prompt_CUST-00020.txt`

## Integration with Image Processing Pipeline

These image prompts are designed to work with the AWS AI project's image processing pipeline:

1. **Input Generation**: The prompts provide detailed instructions for AI image generation services
2. **Quality Testing**: Different prompt types test various aspects of the image processing pipeline
3. **Feedback Correlation**: Generated images can be correlated with original customer feedback
4. **Sentiment Analysis**: Images can be analyzed for visual sentiment matching text feedback

## Usage Recommendations

When using these prompts with image generation services:

1. Use the exact prompt text for consistent results
2. Adjust resolution parameters based on your specific needs
3. Consider the customer's rating when evaluating generated images
4. Use multiple prompts to test different aspects of your image processing pipeline

## Customer Coverage

The current set of prompts covers customers with the following IDs:
- CUST-00001 (5-star rating, excellent experience)
- CUST-00003 (2-star rating, quality issues)
- CUST-00006 (4-star rating, replacement after damage)
- CUST-00007 (1-star rating, terrible experience)
- CUST-00010 (4-star rating, good product quality)
- CUST-00012 (5-star rating, perfect experience)
- CUST-00014 (4-star rating, very satisfied)
- CUST-00016 (5-star rating, exceptional quality)
- CUST-00018 (1-star rating, worst purchase)
- CUST-00020 (3-star rating, average experience)

This provides a balanced distribution across different rating levels and experience types.