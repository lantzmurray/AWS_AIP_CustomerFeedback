#!/bin/bash

# AWS AI Project - Feedback Form Deployment Script
# Deploys the frontend to S3 static website hosting

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    print_error "AWS CLI is not installed. Please install it first."
    echo "Visit: https://aws.amazon.com/cli/"
    exit 1
fi

# Check if AWS CLI is configured
if ! aws sts get-caller-identity &> /dev/null; then
    print_error "AWS CLI is not configured. Please run 'aws configure' first."
    exit 1
fi

# Get bucket name from command line argument
if [ $# -eq 0 ]; then
    print_error "Please provide a bucket name as an argument."
    echo "Usage: ./deploy.sh your-bucket-name [region]"
    echo "Example: ./deploy.sh my-feedback-form us-east-1"
    exit 1
fi

BUCKET_NAME=$1
REGION=${2:-us-east-1}  # Default to us-east-1 if not specified

print_status "Starting deployment of AWS AI Project Feedback Form..."
print_status "Bucket: $BUCKET_NAME"
print_status "Region: $REGION"

# Check if bucket exists
if aws s3 ls s://$BUCKET_NAME 2>&1 | grep -q 'NoSuchBucket'; then
    print_status "Bucket does not exist. Creating bucket..."
    
    # Create bucket (with region-specific handling)
    if [ "$REGION" = "us-east-1" ]; then
        aws s3 mb s://$BUCKET_NAME --region $REGION
    else
        aws s3 mb s://$BUCKET_NAME --region $REGION --create-bucket-configuration LocationConstraint=$REGION
    fi
    
    print_status "Bucket created successfully."
else
    print_status "Bucket already exists."
fi

# Enable static website hosting
print_status "Configuring static website hosting..."
aws s3 website s://$BUCKET_NAME --index-document index.html --error-document index.html

# Create bucket policy for public access
print_status "Setting bucket policy for public access..."
cat > /tmp/bucket-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "PublicReadGetObject",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::$BUCKET_NAME/*"
        }
    ]
}
EOF

aws s3api put-bucket-policy --bucket $BUCKET_NAME --policy file:///tmp/bucket-policy.json
rm /tmp/bucket-policy.json

# Sync files to S3
print_status "Uploading files to S3..."
aws s3 sync . s://$BUCKET_NAME --delete --exclude "*.sh" --exclude "README.md" --exclude ".git/*"

# Set content types for proper serving
print_status "Setting content types..."
aws s3 cp s://$BUCKET_NAME/index.html s://$BUCKET_NAME/index.html --content-type text/html
aws s3 cp s://$BUCKET_NAME/css/style.css s://$BUCKET_NAME/css/style.css --content-type text/css
aws s3 cp s://$BUCKET_NAME/js/config.js s://$BUCKET_NAME/js/config.js --content-type application/javascript
aws s3 cp s://$BUCKET_NAME/js/app.js s://$BUCKET_NAME/js/app.js --content-type application/javascript

# Get bucket website URL
WEBSITE_URL="http://$BUCKET_NAME.s3-website-$REGION.amazonaws.com"

print_status "Deployment completed successfully!"
echo ""
print_status "Website URL: $WEBSITE_URL"
print_warning "Note: It may take a few minutes for the website to become accessible."
echo ""
print_status "Next steps:"
echo "1. Update the API endpoint in js/config.js with your actual API Gateway URL"
echo "2. Test the form at $WEBSITE_URL"
echo "3. Consider setting up a custom domain with CloudFront for production use"
echo ""
print_warning "Security Note: This bucket is configured for public read access."
print_warning "For production, consider using CloudFront with OAI for better security."

# Optional: Ask if user wants to open the website
read -p "Do you want to open the website in your browser? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v open &> /dev/null; then
        open $WEBSITE_URL
    elif command -v xdg-open &> /dev/null; then
        xdg-open $WEBSITE_URL
    elif command -v start &> /dev/null; then
        start $WEBSITE_URL
    else
        print_warning "Could not detect how to open the browser. Please manually visit: $WEBSITE_URL"
    fi
fi