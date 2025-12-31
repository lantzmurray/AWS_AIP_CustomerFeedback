# Getting Started Guide

This guide will help you get the AWS AI Customer Feedback System up and running quickly.

## Prerequisites

### Required AWS Services
- AWS Account with appropriate permissions
- AWS CLI installed and configured
- Python 3.8+ with boto3
- Node.js 16+ (for frontend development)
- Git for version control

### Required Permissions
- S3 (Full access)
- Lambda (Full access)
- Glue (Full access)
- CloudWatch (Full access)
- Comprehend (Full access)
- Textract (Full access)
- Rekognition (Full access)
- Transcribe (Full access)
- SageMaker (Full access)
- Bedrock (Full access)
- IAM (Full access for role management)

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/aws-ai-customer-feedback.git
cd aws-ai-customer-feedback
```

### 2. Set Configuration
```bash
export USER_ID=YOUR_UNIQUE_ID
export AWS_REGION=YOUR_AWS_REGION
export ACCOUNT_ID=YOUR_AWS_ACCOUNT_ID
```

### 3. Deploy Infrastructure
```bash
./scripts/deploy.sh
```

### 4. Upload Sample Data
```bash
aws s3 sync sample_data/ s3://customer-feedback-analysis-${USER_ID}-raw/
```

### 5. Access Your System
```bash
echo "Frontend URL: https://$(aws cloudfront list-distributions | jq -r '.DistributionList.Items[0].DomainName')"
```

## Detailed Setup Instructions

### Phase 1: Infrastructure Setup

#### Create S3 Buckets
```bash
# Create main buckets
aws s3 mb s3://customer-feedback-analysis-${USER_ID}-raw
aws s3 mb s3://customer-feedback-analysis-${USER_ID}-processed
aws s3 mb s3://customer-feedback-analysis-${USER_ID}-results
aws s3 mb s3://customer-feedback-analysis-${USER_ID}-frontend
aws s3 mb s3://customer-feedback-analysis-${USER_ID}-logs

# Enable versioning and encryption
aws s3api put-bucket-versioning --bucket customer-feedback-analysis-${USER_ID}-raw --versioning-configuration Status=Enabled
aws s3api put-bucket-encryption --bucket customer-feedback-analysis-${USER_ID}-raw --server-side-encryption-configuration '{"Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]}'
```

#### Create IAM Roles
```bash
# Create Lambda execution role
aws iam create-role \
  --role-name AWSLambdaExecutionRole-CustomerFeedback \
  --assume-role-policy-document file://deployment/lambda-trust-policy.json \
  --description "Role for Lambda functions"

# Attach policies
aws iam attach-role-policy \
  --role-name AWSLambdaExecutionRole-CustomerFeedback \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam attach-role-policy \
  --role-name AWSLambdaExecutionRole-CustomerFeedback \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

### Phase 2: Lambda Function Deployment

#### Package Functions
```bash
cd src/lambda

# Package each function
cd data_validation
zip -r text_validation_lambda.zip text_validation_lambda.py

cd ../multimodal_processing
zip -r text_processing_lambda.zip text_processing_lambda.py
zip -r image_processing_lambda.zip image_processing_lambda.py
zip -r audio_processing_lambda.zip audio_processing_lambda.py
zip -r survey_processing_lambda.zip survey_processing_lambda.py

cd ../fm_formatting
zip -r claude_formatting_lambda.zip claude_formatting_lambda.py
```

#### Deploy Functions
```bash
# Deploy text validation function
aws lambda create-function \
  --function-name TextValidationFunction \
  --runtime python3.8 \
  --role arn:aws:iam::${ACCOUNT_ID}:role/AWSLambdaExecutionRole-CustomerFeedback \
  --handler text_validation_lambda.lambda_handler \
  --zip-file fileb://text_validation_lambda.zip \
  --timeout 300 \
  --memory-size 512 \
  --environment Variables='{BUCKET_NAME=customer-feedback-analysis-'${USER_ID}'-raw,PROCESSED_BUCKET=customer-feedback-analysis-'${USER_ID}'-processed}'
```

### Phase 3: Frontend Deployment

#### Build and Deploy
```bash
cd frontend
npm install
npm run build

# Deploy to S3
aws s3 sync dist/ s3://customer-feedback-analysis-${USER_ID}-frontend/ --delete

# Configure static website
aws s3 website s3://customer-feedback-analysis-${USER_ID}-frontend/ \
  --index-document index.html \
  --error-document error.html
```

## Testing Your Deployment

### Upload Test Data
```bash
# Upload sample data
aws s3 sync sample_data/ s3://customer-feedback-analysis-${USER_ID}-raw/ \
  --exclude "*" --include="*.txt" --include="*.jpg" --include="*.mp3" --include="*.csv"
```

### Monitor Processing
```bash
# Watch Lambda logs
aws logs tail /aws/lambda/TextValidationFunction --follow

# Check CloudWatch metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/Lambda \
  --metric-name Invocations \
  --dimensions Name=FunctionName,Value=TextValidationFunction \
  --start-time $(date -u -d '-1 hour' +%Y-%m-%dT%H:%M:%SZ) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ)
```

## Troubleshooting

### Common Issues

#### Lambda Function Timeouts
**Symptoms**: Functions failing with timeout errors
**Solutions**: 
- Increase timeout settings
- Optimize code for better performance
- Increase memory allocation

```bash
aws lambda update-function-configuration \
  --function-name TextValidationFunction \
  --timeout 600
```

#### S3 Access Denied
**Symptoms**: Unable to read/write to S3 buckets
**Solutions**:
- Check IAM permissions
- Verify bucket policies
- Ensure correct AWS credentials

```bash
# Test bucket access
aws s3 ls s3://customer-feedback-analysis-${USER_ID}-raw/
```

#### Frontend Connectivity Issues
**Symptoms**: 503/504 errors
**Solutions**:
- Check CloudFront configuration
- Verify Lambda function status
- Clear CloudFront cache

```bash
# Invalidate CloudFront cache
aws cloudfront create-invalidation \
  --distribution-id $DISTRIBUTION_ID \
  --paths '/*'
```

## Next Steps

After successful deployment:

1. **Explore Features**: Try uploading different types of customer feedback
2. **Monitor Performance**: Check CloudWatch dashboards for system health
3. **Customize Configuration**: Adjust settings for your specific use case
4. **Scale Operations**: Monitor costs and optimize as needed

## Support

For additional help:
- Check the [troubleshooting guide](../troubleshooting/README.md)
- Review [API documentation](../api-reference/README.md)
- Open an issue on GitHub
- Contact the project maintainers

---

For more detailed information, please refer to the [architecture documentation](../../architecture/technical-design/README.md) and [deployment guide](../../deployment/README.md).