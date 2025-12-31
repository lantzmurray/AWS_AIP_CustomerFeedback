# AWS AI Project Cleanup Script

## 🚨 IMPORTANT: Run this script to avoid AWS charges!

This cleanup script will remove ALL AWS resources deployed by the customer feedback analysis project to prevent ongoing charges.

## Files Created

1. **`cleanup_aws_resources.sh`** - The executable cleanup script
2. **`CLEANUP_GUIDE.md`** - Comprehensive documentation and instructions

## Quick Start

### 1. Make Script Executable (Already Done)
```bash
chmod +x cleanup_aws_resources.sh
```

### 2. Test with Dry Run (Recommended)
```bash
./cleanup_aws_resources.sh --dry-run
```

### 3. Run Actual Cleanup
```bash
./cleanup_aws_resources.sh
```

## What Will Be Deleted?

The script will remove these AWS resources:
- **7 Lambda Functions** (TextValidationFunctionLM, TextProcessingFunctionLM, etc.)
- **4 S3 Buckets** (raw, processed, results, frontend)
- **3 IAM Roles** (Glue, Lambda execution, Foundation model)
- **1 SQS Queue** (DLQ for failed Lambda executions)
- **1 Glue Database** (customer_feedback_lm_db)
- **4 Glue Tables** (text_reviews, images, audio, surveys)
- **4 Glue Data Quality Rulesets**
- **2 CloudWatch Dashboards**
- **S3 Event Notifications**
- **Lambda Permissions**

## Safety Features

✅ **Confirmation Required** - Must type "DELETE" to confirm  
✅ **Dry Run Mode** - See what would be deleted without actually deleting  
✅ **Comprehensive Logging** - All actions logged to timestamped file  
✅ **Error Handling** - Script continues even if individual resources fail  
✅ **Dependency Order** - Resources deleted in correct order  
✅ **Existence Checks** - Only attempts deletion of existing resources  

## Command Options

```bash
./cleanup_aws_resources.sh [OPTIONS]

Options:
  -h, --help              Show help message
  -r, --region REGION     AWS region (default: us-east-1)
  -e, --environment ENV   Environment (default: dev)
  -a, --account-id ID     AWS account ID
  --dry-run              Show what would be deleted without actually deleting
  --force                Skip confirmation prompts
```

## Examples

```bash
# Interactive cleanup with confirmations
./cleanup_aws_resources.sh

# Dry run to see what would be deleted
./cleanup_aws_resources.sh --dry-run

# Force cleanup without confirmations
./cleanup_aws_resources.sh --force

# Cleanup specific environment and region
./cleanup_aws_resources.sh -e prod -r us-west-2
```

## Prerequisites

- ✅ AWS CLI v2 installed and configured
- ✅ Appropriate IAM permissions to delete all resource types
- ✅ Bash shell (Linux, macOS, or WSL on Windows)

## Verification After Cleanup

1. Check the generated log file: `cleanup_aws_resources_YYYYMMDD_HHMMSS.log`
2. Verify in AWS Console that no resources with project naming exist
3. Monitor your AWS billing to ensure no new charges

## ⚠️ Final Warning

**This will permanently delete ALL project resources and data!**
- This action cannot be undone
- All S3 bucket data will be permanently erased
- All Lambda functions, IAM roles, and infrastructure will be removed

**Only run this script if you want to completely decommission the project!**

## Support

For detailed instructions, troubleshooting, and manual cleanup commands, see [`CLEANUP_GUIDE.md`](CLEANUP_GUIDE.md).