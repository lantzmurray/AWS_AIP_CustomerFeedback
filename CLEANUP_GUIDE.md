# AWS AI Project Cleanup Guide

## Overview

This guide provides a comprehensive cleanup script and instructions to remove all AWS resources deployed by the customer feedback analysis project to avoid incurring charges.

## ⚠️ Important Warning

**This cleanup will permanently delete ALL AWS resources associated with this project!**
- This action cannot be undone
- All data in S3 buckets will be permanently deleted
- All Lambda functions, IAM roles, and other resources will be removed

## Resources That Will Be Cleaned Up

Based on the `rebuild_project.sh` script, the following AWS resources will be removed:

### S3 Buckets
- `customer-feedback-analysis-lm-raw` (raw data bucket)
- `customer-feedback-analysis-lm-processed` (processed data bucket)
- `customer-feedback-analysis-lm-results` (results bucket)
- `customer-feedback-analysis-lm-frontend-{environment}` (frontend bucket)

### Lambda Functions
- `TextValidationFunctionLM`
- `TextProcessingFunctionLM`
- `ImageProcessingFunctionLM`
- `AudioProcessingFunctionLM`
- `SurveyProcessingFunctionLM`
- `FoundationModelFormatter`
- `FoundationModelRealTimeFormatter`

### IAM Roles
- `AWSGlueServiceRole-CustomerFeedbackLM`
- `AWSLambdaExecutionRole-CustomerFeedbackLM`
- `FoundationModelFormattingRole`

### Other Resources
- SQS Queue: `lambda-processing-failed-dlq`
- Glue Database: `customer_feedback_lm_db`
- Glue Tables: `text_reviews`, `images`, `audio`, `surveys`
- Glue Data Quality Rulesets: `customer_reviews_lm_ruleset`, `survey_data_lm_ruleset`, `image_metadata_lm_ruleset`, `audio_metadata_lm_ruleset`
- CloudWatch Dashboards: `CustomerFeedbackQualityLM`, `CustomerFeedbackS3LM`
- S3 Event Notifications
- Lambda Permissions

## Cleanup Script

Create a file named `cleanup_aws_resources.sh` with the following content:

```bash
#!/bin/bash

# AWS AI Project Comprehensive Cleanup Script
# This script removes all AWS resources deployed by project to avoid charges
# Date: December 15, 2024
# Version: 1.0

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration variables with defaults
AWS_REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
PROJECT_NAME="customer-feedback-analysis-lm"
ENVIRONMENT="${ENVIRONMENT:-dev}"

# Resource Names (matching rebuild_project.sh)
RAW_BUCKET="${PROJECT_NAME}-raw"
PROCESSED_BUCKET="${PROJECT_NAME}-processed"
RESULTS_BUCKET="${PROJECT_NAME}-results"
FRONTEND_BUCKET="${PROJECT_NAME}-frontend-${ENVIRONMENT}"

# Function Names
TEXT_VALIDATION_FUNCTION="TextValidationFunctionLM"
TEXT_PROCESSING_FUNCTION="TextProcessingFunctionLM"
IMAGE_PROCESSING_FUNCTION="ImageProcessingFunctionLM"
AUDIO_PROCESSING_FUNCTION="AudioProcessingFunctionLM"
SURVEY_PROCESSING_FUNCTION="SurveyProcessingFunctionLM"
FOUNDATION_MODEL_FORMATTER="FoundationModelFormatter"
REAL_TIME_FORMATTER="FoundationModelRealTimeFormatter"

# IAM Role Names
GLUE_ROLE="AWSGlueServiceRole-CustomerFeedbackLM"
LAMBDA_EXECUTION_ROLE="AWSLambdaExecutionRole-CustomerFeedbackLM"
FOUNDATION_MODEL_ROLE="FoundationModelFormattingRole"

# Other Resources
DLQ_NAME="lambda-processing-failed-dlq"
GLUE_DATABASE="customer_feedback_lm_db"

# Logging
LOG_FILE="cleanup_aws_resources_$(date +%Y%m%d_%H%M%S).log"

# Initialize log file
echo "AWS AI Project Cleanup Log - $(date)" > "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# Helper functions
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1" >> "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >> "$LOG_FILE"
}

# Function to check if resource exists and delete it
safe_delete() {
    local resource_type="$1"
    local resource_name="$2"
    local delete_command="$3"
    local check_command="$4"
    
    log "Checking $resource_type: $resource_name"
    
    if eval "$check_command" >/dev/null 2>&1; then
        log "Found $resource_type: $resource_name. Deleting..."
        if eval "$delete_command" >> "$LOG_FILE" 2>&1; then
            log "✓ Successfully deleted $resource_type: $resource_name"
        else
            warn "✗ Failed to delete $resource_type: $resource_name. Check log for details."
        fi
    else
        log "$resource_type not found: $resource_name. Skipping."
    fi
}

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -r, --region REGION     AWS region (default: us-east-1)"
    echo "  -e, --environment ENV   Environment (default: dev)"
    echo "  -a, --account-id ID     AWS account ID"
    echo "  --dry-run              Show what would be deleted without actually deleting"
    echo "  --force                Skip confirmation prompts"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Interactive cleanup with confirmations"
    echo "  $0 --dry-run                          # Show what would be deleted"
    echo "  $0 --force -r us-west-2 -e prod      # Force cleanup in us-west-2 for prod"
}

# Function to display banner
display_banner() {
    echo -e "${BLUE}"
    echo "=================================================="
    echo "    AWS AI Project Comprehensive Cleanup Script"
    echo "=================================================="
    echo "Region: $AWS_REGION"
    echo "Account ID: $ACCOUNT_ID"
    echo "Project: $PROJECT_NAME"
    echo "Environment: $ENVIRONMENT"
    echo "Log File: $LOG_FILE"
    echo "=================================================="
    echo -e "${NC}"
}

# Function to check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check AWS CLI version
    if ! aws --version | grep -q "aws-cli/2"; then
        error "AWS CLI v2 is required. Please install AWS CLI v2."
        exit 1
    fi
    
    # Check if AWS credentials are configured
    if ! aws sts get-caller-identity >/dev/null 2>&1; then
        error "AWS credentials not configured. Please run 'aws configure'."
        exit 1
    fi
    
    # Verify account ID
    if [[ -z "$ACCOUNT_ID" ]]; then
        error "Could not determine AWS account ID. Please set AWS_ACCOUNT_ID environment variable."
        exit 1
    fi
    
    log "Prerequisites check completed successfully."
}

# Function to get user confirmation
get_confirmation() {
    if [[ "$FORCE" == "true" ]]; then
        return 0
    fi
    
    echo ""
    echo -e "${RED}WARNING: This will permanently delete ALL AWS resources associated with this project!${NC}"
    echo -e "${RED}This action cannot be undone and will result in data loss.${NC}"
    echo ""
    read -p "Are you sure you want to continue? Type 'DELETE' to confirm: " confirmation
    
    if [[ "$confirmation" != "DELETE" ]]; then
        log "Cleanup cancelled by user."
        exit 0
    fi
}

# Function to cleanup Lambda functions
cleanup_lambda_functions() {
    log "Cleaning up Lambda functions..."
    
    local functions=(
        "$TEXT_VALIDATION_FUNCTION"
        "$TEXT_PROCESSING_FUNCTION"
        "$IMAGE_PROCESSING_FUNCTION"
        "$AUDIO_PROCESSING_FUNCTION"
        "$SURVEY_PROCESSING_FUNCTION"
        "$FOUNDATION_MODEL_FORMATTER"
        "$REAL_TIME_FORMATTER"
    )
    
    for function in "${functions[@]}"; do
        safe_delete "Lambda Function" "$function" \
            "aws lambda delete-function --function-name \"$function\" --region \"$AWS_REGION\"" \
            "aws lambda get-function --function-name \"$function\" --region \"$AWS_REGION\""
    done
}

# Function to cleanup S3 event notifications
cleanup_s3_notifications() {
    log "Cleaning up S3 event notifications..."
    
    local buckets=("$RAW_BUCKET" "$PROCESSED_BUCKET")
    
    for bucket in "${buckets[@]}"; do
        if aws s3 ls | grep -q "$bucket"; then
            log "Removing S3 notifications from bucket: $bucket"
            aws s3api put-bucket-notification-configuration \
                --bucket "$bucket" \
                --notification-configuration '{}' \
                --region "$AWS_REGION" >> "$LOG_FILE" 2>&1 || warn "Failed to remove notifications from $bucket"
        fi
    done
}

# Function to cleanup SQS queues
cleanup_sqs_queues() {
    log "Cleaning up SQS queues..."
    
    safe_delete "SQS Queue" "$DLQ_NAME" \
        "aws sqs delete-queue --queue-url \"\$(aws sqs get-queue-url --queue-name \"$DLQ_NAME\" --region \"$AWS_REGION\" --output text)\" --region \"$AWS_REGION\"" \
        "aws sqs get-queue-url --queue-name \"$DLQ_NAME\" --region \"$AWS_REGION\""
}

# Function to cleanup S3 buckets
cleanup_s3_buckets() {
    log "Cleaning up S3 buckets..."
    
    local buckets=("$RAW_BUCKET" "$PROCESSED_BUCKET" "$RESULTS_BUCKET" "$FRONTEND_BUCKET")
    
    for bucket in "${buckets[@]}"; do
        if aws s3 ls | grep -q "$bucket"; then
            log "Emptying S3 bucket: $bucket"
            aws s3 rm "s3://$bucket" --recursive >> "$LOG_FILE" 2>&1 || warn "Failed to empty bucket $bucket"
            
            safe_delete "S3 Bucket" "$bucket" \
                "aws s3 rb \"s3://$bucket\" --force" \
                "aws s3 ls | grep -q \"$bucket\""
        else
            log "S3 bucket not found: $bucket. Skipping."
        fi
    done
}

# Function to cleanup IAM roles
cleanup_iam_roles() {
    log "Cleaning up IAM roles..."
    
    local roles=("$GLUE_ROLE" "$LAMBDA_EXECUTION_ROLE" "$FOUNDATION_MODEL_ROLE")
    
    for role in "${roles[@]}"; do
        # First detach all policies
        if aws iam get-role --role-name "$role" >/dev/null 2>&1; then
            log "Detaching policies from IAM role: $role"
            
            # Get attached policies
            attached_policies=$(aws iam list-attached-role-policies --role-name "$role" --query 'AttachedPolicies[].PolicyArn' --output text 2>/dev/null || echo "")
            
            for policy_arn in $attached_policies; do
                if [[ -n "$policy_arn" ]]; then
                    aws iam detach-role-policy --role-name "$role" --policy-arn "$policy_arn" >> "$LOG_FILE" 2>&1 || warn "Failed to detach policy $policy_arn from role $role"
                fi
            done
        fi
        
        safe_delete "IAM Role" "$role" \
            "aws iam delete-role --role-name \"$role\"" \
            "aws iam get-role --role-name \"$role\""
    done
}

# Function to cleanup Glue resources
cleanup_glue_resources() {
    log "Cleaning up Glue resources..."
    
    # Delete Glue tables
    local table_types=("text_reviews" "images" "audio" "surveys")
    
    for table_type in "${table_types[@]}"; do
        safe_delete "Glue Table" "$table_type" \
            "aws glue delete-table --database-name \"$GLUE_DATABASE\" --name \"$table_type\"" \
            "aws glue get-table --database-name \"$GLUE_DATABASE\" --name \"$table_type\""
    done
    
    # Delete Glue database
    safe_delete "Glue Database" "$GLUE_DATABASE" \
        "aws glue delete-database --name \"$GLUE_DATABASE\"" \
        "aws glue get-database --name \"$GLUE_DATABASE\""
    
    # Delete Glue Data Quality rulesets
    local rulesets=(
        "customer_reviews_lm_ruleset"
        "survey_data_lm_ruleset"
        "image_metadata_lm_ruleset"
        "audio_metadata_lm_ruleset"
    )
    
    for ruleset in "${rulesets[@]}"; do
        safe_delete "Glue Data Quality Ruleset" "$ruleset" \
            "aws glue delete-data-quality-ruleset --name \"$ruleset\"" \
            "aws glue get-data-quality-ruleset --name \"$ruleset\""
    done
}

# Function to cleanup CloudWatch dashboards
cleanup_cloudwatch_dashboards() {
    log "Cleaning up CloudWatch dashboards..."
    
    local dashboards=("CustomerFeedbackQualityLM" "CustomerFeedbackS3LM")
    
    for dashboard in "${dashboards[@]}"; do
        safe_delete "CloudWatch Dashboard" "$dashboard" \
            "aws cloudwatch delete-dashboards --dashboard-names \"$dashboard\"" \
            "aws cloudwatch get-dashboard --dashboard-name \"$dashboard\""
    done
}

# Function to cleanup Lambda permissions
cleanup_lambda_permissions() {
    log "Cleaning up Lambda permissions..."
    
    local functions=(
        "$TEXT_VALIDATION_FUNCTION"
        "$TEXT_PROCESSING_FUNCTION"
        "$IMAGE_PROCESSING_FUNCTION"
        "$AUDIO_PROCESSING_FUNCTION"
        "$FOUNDATION_MODEL_FORMATTER"
    )
    
    for function in "${functions[@]}"; do
        if aws lambda get-function --function-name "$function" --region "$AWS_REGION" >/dev/null 2>&1; then
            # Get policy and remove S3 permissions
            policy=$(aws lambda get-policy --function-name "$function" --region "$AWS_REGION" --query 'Policy' --output text 2>/dev/null || echo "")
            if [[ -n "$policy" ]]; then
                log "Removing permissions for Lambda function: $function"
                aws lambda remove-permission --function-name "$function" --region "$AWS_REGION" --statement-id "s3-invoke" >> "$LOG_FILE" 2>&1 || true
                aws lambda remove-permission --function-name "$function" --region "$AWS_REGION" --statement-id "s3-invoke-processed" >> "$LOG_FILE" 2>&1 || true
                aws lambda remove-permission --function-name "$function" --region "$AWS_REGION" --statement-id "s3-invoke-foundation" >> "$LOG_FILE" 2>&1 || true
            fi
        fi
    done
}

# Main execution
main() {
    local dry_run=false
    local force=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                exit 0
                ;;
            -r|--region)
                AWS_REGION="$2"
                shift 2
                ;;
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -a|--account-id)
                ACCOUNT_ID="$2"
                shift 2
                ;;
            --dry-run)
                dry_run=true
                shift
                ;;
            --force)
                force=true
                shift
                ;;
            *)
                error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
    
    # Set global variables
    FORCE="$force"
    
    # Update resource names based on environment
    RAW_BUCKET="${PROJECT_NAME}-raw"
    PROCESSED_BUCKET="${PROJECT_NAME}-processed"
    RESULTS_BUCKET="${PROJECT_NAME}-results"
    FRONTEND_BUCKET="${PROJECT_NAME}-frontend-${ENVIRONMENT}"
    
    display_banner
    
    if [[ "$dry_run" == "true" ]]; then
        log "DRY RUN MODE: Showing what would be deleted without actually deleting..."
        echo ""
        echo "Resources that would be deleted:"
        echo "- Lambda Functions: $TEXT_VALIDATION_FUNCTION, $TEXT_PROCESSING_FUNCTION, $IMAGE_PROCESSING_FUNCTION, $AUDIO_PROCESSING_FUNCTION, $SURVEY_PROCESSING_FUNCTION, $FOUNDATION_MODEL_FORMATTER, $REAL_TIME_FORMATTER"
        echo "- S3 Buckets: $RAW_BUCKET, $PROCESSED_BUCKET, $RESULTS_BUCKET, $FRONTEND_BUCKET"
        echo "- IAM Roles: $GLUE_ROLE, $LAMBDA_EXECUTION_ROLE, $FOUNDATION_MODEL_ROLE"
        echo "- SQS Queue: $DLQ_NAME"
        echo "- Glue Database: $GLUE_DATABASE"
        echo "- Glue Tables: text_reviews, images, audio, surveys"
        echo "- Glue Data Quality Rulesets: customer_reviews_lm_ruleset, survey_data_lm_ruleset, image_metadata_lm_ruleset, audio_metadata_lm_ruleset"
        echo "- CloudWatch Dashboards: CustomerFeedbackQualityLM, CustomerFeedbackS3LM"
        echo ""
        log "Dry run completed. No resources were deleted."
        exit 0
    fi
    
    check_prerequisites
    get_confirmation
    
    log "Starting comprehensive cleanup..."
    
    # Cleanup in reverse order of deployment to handle dependencies
    cleanup_lambda_permissions
    cleanup_s3_notifications
    cleanup_lambda_functions
    cleanup_sqs_queues
    cleanup_cloudwatch_dashboards
    cleanup_glue_resources
    cleanup_s3_buckets
    cleanup_iam_roles
    
    # Display completion message
    echo -e "${GREEN}"
    echo "=================================================="
    echo "           Cleanup Completed Successfully!"
    echo "=================================================="
    echo "Log file: $LOG_FILE"
    echo ""
    echo "Note: Some resources may take a few minutes to be fully deleted."
    echo "Check AWS Console to verify all resources have been removed."
    echo "=================================================="
    echo -e "${NC}"
}

# Trap to handle script interruption
trap 'error "Script interrupted. Check log file: $LOG_FILE"' INT TERM

# Execute main function with all arguments
main "$@"
```

## How to Run the Cleanup Script

### Prerequisites

1. **AWS CLI v2** installed and configured
2. **Appropriate IAM permissions** to delete all the resources
3. **Bash shell** (Linux, macOS, or WSL on Windows)

### Step-by-Step Instructions

1. **Create the cleanup script file:**
   ```bash
   # Copy the script content above into a new file
   nano cleanup_aws_resources.sh
   # Paste the script content, save and exit
   ```

2. **Make the script executable:**
   ```bash
   chmod +x cleanup_aws_resources.sh
   ```

3. **Optional: Do a dry run first:**
   ```bash
   ./cleanup_aws_resources.sh --dry-run
   ```

4. **Run the interactive cleanup:**
   ```bash
   ./cleanup_aws_resources.sh
   ```

5. **Or run with specific environment:**
   ```bash
   ./cleanup_aws_resources.sh -e prod -r us-west-2
   ```

6. **Or force cleanup without confirmations:**
   ```bash
   ./cleanup_aws_resources.sh --force
   ```

### Command Line Options

- `-h, --help`: Show help message
- `-r, --region REGION`: AWS region (default: us-east-1)
- `-e, --environment ENV`: Environment (default: dev)
- `-a, --account-id ID`: AWS account ID
- `--dry-run`: Show what would be deleted without actually deleting
- `--force`: Skip confirmation prompts

### Environment Variables

You can set these environment variables to override defaults:

```bash
export AWS_REGION=us-west-2
export AWS_ACCOUNT_ID=123456789012
export ENVIRONMENT=prod
./cleanup_aws_resources.sh
```

## Cleanup Order

The script follows this order to handle AWS resource dependencies correctly:

1. **Lambda Permissions** - Remove S3 invoke permissions
2. **S3 Event Notifications** - Remove bucket notifications
3. **Lambda Functions** - Delete all Lambda functions
4. **SQS Queues** - Delete dead letter queues
5. **CloudWatch Dashboards** - Delete monitoring dashboards
6. **Glue Resources** - Delete tables, database, and data quality rulesets
7. **S3 Buckets** - Empty and delete all S3 buckets
8. **IAM Roles** - Detach policies and delete IAM roles

## Verification

After running the cleanup script:

1. **Check the log file** for any errors:
   ```bash
   cat cleanup_aws_resources_YYYYMMDD_HHMMSS.log
   ```

2. **Verify in AWS Console:**
   - Navigate to each service (Lambda, S3, IAM, Glue, CloudWatch, SQS)
   - Confirm no resources with the project naming convention exist

3. **Check billing** to ensure no new charges are accumulating

## Troubleshooting

### Common Issues

1. **Permission Denied Errors**
   - Ensure your IAM user/role has full access to delete all resource types
   - Some resources may need specific permissions (e.g., IAM role deletion)

2. **Resource Not Found Warnings**
   - These are normal if some resources were already deleted
   - The script continues and attempts to delete all resources

3. **Dependency Errors**
   - If you get dependency errors, run the script again
   - Some resources take time to be fully deleted

4. **Script Interruption**
   - If the script is interrupted, check the log file
   - You can safely run the script again - it will only delete remaining resources

### Manual Cleanup

If the script fails, you can manually clean up resources:

```bash
# List and delete Lambda functions
aws lambda list-functions --region $AWS_REGION --query 'Functions[?contains(FunctionName, `LM`)].FunctionName' --output text | xargs -I {} aws lambda delete-function --function-name {} --region $AWS_REGION

# List and delete S3 buckets
aws s3 ls | grep 'customer-feedback-analysis-lm' | awk '{print $3}' | xargs -I {} aws s3 rb s3://{} --force

# List and delete IAM roles
aws iam list-roles --query 'Roles[?contains(RoleName, `CustomerFeedbackLM`)].RoleName' --output text | xargs -I {} aws iam delete-role --role-name {}
```

## Safety Features

The script includes several safety features:

1. **Confirmation Required**: Must type "DELETE" to confirm
2. **Dry Run Mode**: See what would be deleted without actually deleting
3. **Logging**: All actions are logged to a timestamped file
4. **Error Handling**: Script continues even if individual resources fail to delete
5. **Dependency Order**: Resources are deleted in the correct order
6. **Existence Checks**: Script checks if resources exist before attempting deletion

## Post-Cleanup

After successful cleanup:

1. **Verify no charges** are accumulating in your AWS account
2. **Remove local files** if you no longer need them
3. **Revoke any temporary IAM permissions** you created for the cleanup
4. **Monitor your AWS bill** for the next billing cycle

## Support

If you encounter issues:

1. Check the log file for specific error messages
2. Verify your AWS credentials and permissions
3. Ensure you're using AWS CLI v2
4. Check AWS service limits and quotas
5. Review AWS CloudTrail logs for detailed audit information