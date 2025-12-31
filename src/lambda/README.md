# Code Repository

This folder contains all the Python code implementations for the AWS data validation and processing pipeline.

## Folder Structure

### data_validation/
Contains code for data validation components:
- `glue_data_quality_ruleset.py` - AWS Glue Data Quality ruleset creation
- `text_validation_lambda.py` - Lambda function for custom text validation
- `cloudwatch_dashboard.py` - CloudWatch Dashboard creation and monitoring

### multimodal_processing/
Contains code for processing different data types:
- `text_processing_lambda.py` - Lambda function for processing text reviews with Comprehend
- `image_processing_lambda.py` - Lambda function for processing product images with Textract/Rekognition
- `audio_processing_lambda.py` - Lambda function for processing audio recordings with Transcribe
- `survey_processing_script.py` - SageMaker Processing script for survey data

### fm_formatting/
Contains code for foundation model formatting:
- `model_selection_strategy.py` - Model selection strategy based on evaluation results
- `sagemaker_processing_job.py` - Python script to run SageMaker Processing job
- `claude_formatting_lambda.py` - Lambda Function for formatting data for Claude

### utils/
Contains shared utility functions:
- `common_functions.py` - Shared helper functions used across multiple components

## Dependencies

### Required Python Packages
```bash
pip install boto3 pandas numpy sagemaker
```

### AWS SDK Versions
- boto3 >= 1.26.0
- sagemaker >= 2.100.0

## Deployment Instructions

### Lambda Functions
1. Package each Lambda function with its dependencies
2. Upload to AWS Lambda
3. Configure appropriate IAM roles
4. Set up S3 event triggers

### SageMaker Processing
1. Upload processing script to S3
2. Configure processing job parameters
3. Submit job using SageMaker SDK

### Glue Scripts
1. Run scripts locally or in AWS Glue Studio
2. Ensure appropriate IAM permissions
3. Verify data catalog setup

## Environment Variables

### Required for Lambda Functions
- `BUCKET_NAME`: S3 bucket for data storage
- `REGION`: AWS region for service calls
- `BEDROCK_REGION`: Region for Bedrock model access

### Optional
- `LOG_LEVEL`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)
- `QUALITY_THRESHOLD`: Minimum quality score for processing

## Security Considerations

- Never hardcode AWS credentials in code
- Use IAM roles for service access
- Implement least privilege principle
- Validate all input data
- Use encryption for sensitive data

## Monitoring and Logging

- All Lambda functions include CloudWatch logging
- Error handling with appropriate log levels
- Performance metrics tracking
- Custom CloudWatch metrics where applicable

## Testing

### Unit Tests
Run unit tests for each component:
```bash
python -m pytest tests/
```

### Integration Tests
Test end-to-end pipeline with sample data:
```bash
python integration_tests.py
```

## Contributing

When adding new code:
1. Follow existing code style
2. Add appropriate documentation
3. Include error handling
4. Add logging statements
5. Update this README