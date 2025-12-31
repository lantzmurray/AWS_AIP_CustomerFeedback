#!/usr/bin/env python3
"""
Batch Processor

Handles large-scale batch formatting operations with support for SageMaker
processing jobs, parallel processing, and cost optimization.
"""

import json
import boto3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import logging
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class BatchProcessor:
    """
    Handles batch processing of formatted data with SageMaker integration.
    """
    
    def __init__(self):
        """Initialize batch processor."""
        self.s3_client = boto3.client('s3')
        self.sagemaker_client = boto3.client('sagemaker')
        self.sqs_client = boto3.client('sqs')
        self.cloudwatch_client = boto3.client('cloudwatch')
        self.batch_queue = queue.Queue()
        self.processing_jobs = {}
        
    def create_batch_job(self, job_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a batch processing job.
        
        Args:
            job_config: Configuration for the batch job
            
        Returns:
            Batch job information
        """
        job_id = f"batch-job-{int(time.time())}"
        
        # Validate job configuration
        validation_result = self._validate_job_config(job_config)
        if not validation_result["valid"]:
            return {
                "job_id": job_id,
                "status": "failed",
                "error": validation_result["error"],
                "created_at": datetime.now().isoformat()
            }
        
        # Create job record
        job_record = {
            "job_id": job_id,
            "config": job_config,
            "status": "created",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "items_processed": 0,
            "items_total": job_config.get("total_items", 0),
            "error_count": 0,
            "processing_time_seconds": 0,
            "cost_estimate": self._estimate_job_cost(job_config)
        }
        
        # Store job record
        self._store_job_record(job_record)
        
        # Start processing based on job type
        if job_config.get("use_sagemaker", False):
            job_record["status"] = "submitting_to_sagemaker"
            sagemaker_job = self._submit_sagemaker_job(job_id, job_config)
            job_record["sagemaker_job_name"] = sagemaker_job.get("ProcessingJobName", "")
        else:
            job_record["status"] = "queued"
            self.batch_queue.put(job_id)
        
        # Update job record
        self._update_job_record(job_id, job_record)
        
        return job_record
    
    def process_batch_queue(self, max_workers: int = 10) -> None:
        """
        Process items from the batch queue with parallel workers.
        
        Args:
            max_workers: Maximum number of parallel workers
        """
        logger.info(f"Starting batch queue processing with {max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while True:
                try:
                    # Get job from queue (with timeout)
                    job_id = self.batch_queue.get(timeout=1)
                    
                    if job_id == "SHUTDOWN":
                        logger.info("Shutdown signal received, stopping batch processing")
                        break
                    
                    # Submit job for processing
                    future = executor.submit(self._process_single_job, job_id)
                    
                    # Add completion callback
                    future.add_done_callback(
                        lambda f, jid=job_id: self._job_completion_callback(f, jid)
                    )
                    
                except queue.Empty:
                    # No jobs in queue, continue
                    continue
                except Exception as e:
                    logger.error(f"Error processing batch queue: {str(e)}")
                    time.sleep(5)
    
    def _process_single_job(self, job_id: str) -> Dict[str, Any]:
        """
        Process a single batch job.
        
        Args:
            job_id: ID of the job to process
            
        Returns:
            Job processing result
        """
        logger.info(f"Processing batch job: {job_id}")
        
        # Get job record
        job_record = self._get_job_record(job_id)
        if not job_record:
            return {"job_id": job_id, "status": "error", "error": "Job record not found"}
        
        try:
            # Update status to processing
            job_record["status"] = "processing"
            job_record["started_at"] = datetime.now().isoformat()
            self._update_job_record(job_id, job_record)
            
            # Get job configuration
            job_config = job_record["config"]
            
            # Process based on job type
            if job_config.get("use_sagemaker", False):
                result = self._monitor_sagemaker_job(job_id, job_config)
            else:
                result = self._process_local_batch(job_id, job_config)
            
            # Update job record with results
            job_record.update(result)
            job_record["status"] = "completed"
            job_record["completed_at"] = datetime.now().isoformat()
            
            # Calculate processing time
            if "started_at" in job_record:
                start_time = datetime.fromisoformat(job_record["started_at"])
                end_time = datetime.fromisoformat(job_record["completed_at"])
                job_record["processing_time_seconds"] = (end_time - start_time).total_seconds()
            
            self._update_job_record(job_id, job_record)
            
            # Send completion metrics
            self._send_job_completion_metrics(job_record)
            
            return job_record
            
        except Exception as e:
            logger.error(f"Error processing job {job_id}: {str(e)}")
            
            # Update job record with error
            job_record["status"] = "failed"
            job_record["error"] = str(e)
            job_record["failed_at"] = datetime.now().isoformat()
            self._update_job_record(job_id, job_record)
            
            # Send error metrics
            self._send_job_error_metrics(job_id, str(e))
            
            return job_record
    
    def _submit_sagemaker_job(self, job_id: str, job_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit job to SageMaker for processing.
        
        Args:
            job_id: Job identifier
            job_config: Job configuration
            
        Returns:
            SageMaker job information
        """
        # Build SageMaker processing job
        processing_job_name = f"fm-formatting-{job_id}"
        
        # Determine processing script and container
        data_type = job_config.get("data_type", "text")
        script_name = self._get_sagemaker_script(data_type)
        
        # Configure input and output paths
        input_path = job_config.get("input_path", "")
        output_path = job_config.get("output_path", "")
        
        # Create processing job definition
        job_definition = {
            "ProcessingJobName": processing_job_name,
            "ProcessingResources": {
                "InstanceType": job_config.get("instance_type", "ml.m5.xlarge"),
                "InstanceCount": job_config.get("instance_count", 1),
                "VolumeSizeInGB": job_config.get("volume_size_gb", 30)
            },
            "AppSpecification": {
                "ImageUri": job_config.get(
                    "container_image", 
                    "763104351884.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3"
                ),
                "ContainerEntrypoint": ["python3"],
                "ContainerArguments": [
                    "--input-path", input_path,
                    "--output-path", output_path,
                    "--data-type", data_type,
                    "--model", job_config.get("target_model", "claude-v2"),
                    "--format", job_config.get("output_format", "jsonl")
                ]
            },
            "RoleArn": job_config.get(
                "execution_role", 
                os.environ.get("SAGEMAKER_EXECUTION_ROLE", "")
            ),
            "ProcessingInputDataConfig": [
                {
                    "ChannelName": "input",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": input_path
                        }
                    }
                }
            ],
            "ProcessingOutputDataConfig": [
                {
                    "ChannelName": "output",
                    "S3OutputPath": output_path
                }
            ],
            "Environment": {
                "AWS_DEFAULT_REGION": os.environ.get("AWS_REGION", "us-east-1"),
                "JOB_ID": job_id,
                "DATA_TYPE": data_type
            },
            "MaxRuntimeInSeconds": job_config.get("max_runtime_seconds", 3600),
            "Tags": [
                {"Key": "Project", "Value": "CustomerFeedbackAnalysis"},
                {"Key": "Component", "Value": "FoundationModelFormatting"},
                {"Key": "JobId", "Value": job_id}
            ]
        }
        
        try:
            # Submit job to SageMaker
            response = self.sagemaker_client.create_processing_job(**job_definition)
            
            job_arn = response.get("ProcessingJobArn", "")
            logger.info(f"Submitted SageMaker job: {processing_job_name}")
            
            return {
                "job_name": processing_job_name,
                "job_arn": job_arn,
                "status": "submitted"
            }
            
        except Exception as e:
            logger.error(f"Error submitting SageMaker job: {str(e)}")
            return {
                "job_name": processing_job_name,
                "status": "failed",
                "error": str(e)
            }
    
    def _monitor_sagemaker_job(self, job_id: str, job_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Monitor SageMaker job until completion.
        
        Args:
            job_id: Job identifier
            job_config: Job configuration
            
        Returns:
            Job monitoring result
        """
        processing_job_name = f"fm-formatting-{job_id}"
        max_wait_time = job_config.get("max_runtime_seconds", 3600)
        check_interval = 30  # seconds
        
        start_time = time.time()
        
        while True:
            try:
                # Get job status
                response = self.sagemaker_client.describe_processing_job(
                    ProcessingJobName=processing_job_name
                )
                
                status = response.get("ProcessingJobStatus", "")
                
                # Update job record with status
                job_record = self._get_job_record(job_id)
                if job_record:
                    job_record["sagemaker_status"] = status
                    job_record["sagemaker_message"] = response.get("FailureReason", "")
                    self._update_job_record(job_id, job_record)
                
                # Check if job is complete
                if status in ["Completed", "Failed", "Stopped"]:
                    if status == "Completed":
                        # Get output location
                        output_config = response.get("ProcessingOutputDataConfig", [])
                        if output_config:
                            output_path = output_config[0].get("S3OutputPath", "")
                            
                            return {
                                "status": "completed",
                                "output_path": output_path,
                                "items_processed": self._count_processed_items(output_path),
                                "sagemaker_status": status
                            }
                    
                    return {
                        "status": status.lower(),
                        "sagemaker_status": status,
                        "error": response.get("FailureReason", "Unknown error")
                    }
                
                # Check timeout
                if time.time() - start_time > max_wait_time:
                    return {
                        "status": "timeout",
                        "sagemaker_status": status,
                        "error": f"Job timed out after {max_wait_time} seconds"
                    }
                
                # Wait before next check
                time.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error monitoring SageMaker job: {str(e)}")
                return {
                    "status": "error",
                    "error": str(e)
                }
    
    def _process_local_batch(self, job_id: str, job_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process batch job locally (without SageMaker).
        
        Args:
            job_id: Job identifier
            job_config: Job configuration
            
        Returns:
            Local processing result
        """
        logger.info(f"Processing batch job {job_id} locally")
        
        # Get processing parameters
        input_path = job_config.get("input_path", "")
        output_path = job_config.get("output_path", "")
        data_type = job_config.get("data_type", "text")
        target_model = job_config.get("target_model", "claude-v2")
        output_format = job_config.get("output_format", "jsonl")
        
        # List input files
        try:
            input_files = self._list_input_files(input_path)
            total_items = len(input_files)
            
            # Process each file
            processed_count = 0
            error_count = 0
            
            for input_file in input_files:
                try:
                    # Process single file
                    result = self._process_input_file(
                        input_file, data_type, target_model, output_format
                    )
                    
                    if result["success"]:
                        processed_count += 1
                        # Upload processed file
                        self._upload_processed_file(
                            result["output_data"], output_path, input_file
                        )
                    else:
                        error_count += 1
                        logger.error(f"Error processing file {input_file}: {result['error']}")
                
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error processing file {input_file}: {str(e)}")
            
            return {
                "status": "completed",
                "items_processed": processed_count,
                "items_total": total_items,
                "error_count": error_count,
                "output_path": output_path
            }
            
        except Exception as e:
            logger.error(f"Error in local batch processing: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "items_processed": 0,
                "error_count": 1
            }
    
    def _process_input_file(self, input_file: str, data_type: str, 
                          target_model: str, output_format: str) -> Dict[str, Any]:
        """
        Process a single input file.
        
        Args:
            input_file: S3 path to input file
            data_type: Type of data to process
            target_model: Target foundation model
            output_format: Output format
            
        Returns:
            Processing result
        """
        try:
            # Download file from S3
            bucket, key = self._parse_s3_path(input_file)
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            
            # Read and parse data
            file_content = response['Body'].read().decode('utf-8')
            data = json.loads(file_content)
            
            # Get appropriate formatter
            if data_type == "text":
                from text_formatter import create_text_formatter
                formatter = create_text_formatter()
            elif data_type == "image":
                from image_formatter import create_image_formatter
                formatter = create_image_formatter()
            elif data_type == "audio":
                from audio_formatter import create_audio_formatter
                formatter = create_audio_formatter()
            elif data_type == "survey":
                from survey_formatter import create_survey_formatter
                formatter = create_survey_formatter()
            else:
                return {
                    "success": False,
                    "error": f"Unsupported data type: {data_type}"
                }
            
            # Format data
            if target_model.startswith("claude"):
                formatted_data = formatter.format_for_claude(data, output_format)
            elif target_model.startswith("titan"):
                formatted_data = formatter.format_for_titan(data, output_format)
            else:
                formatted_data = formatter.format_for_training(data, output_format)
            
            return {
                "success": True,
                "output_data": formatted_data,
                "original_data": data
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _upload_processed_file(self, formatted_data: Dict[str, Any], 
                             output_path: str, input_file: str) -> None:
        """
        Upload processed file to output location.
        
        Args:
            formatted_data: Formatted data to upload
            output_path: S3 path for output
            input_file: Original input file path
        """
        try:
            # Generate output filename
            input_filename = input_file.split('/')[-1]
            base_name = input_filename.replace('.json', '').replace('_processed', '')
            
            if "jsonl_line" in formatted_data:
                output_filename = f"{base_name}_formatted.jsonl"
                output_content = formatted_data["jsonl_line"] + "\n"
            elif "parquet_data" in formatted_data:
                output_filename = f"{base_name}_formatted.parquet"
                output_content = formatted_data["parquet_data"]
            else:
                output_filename = f"{base_name}_formatted.json"
                output_content = json.dumps(formatted_data, indent=2)
            
            # Upload to S3
            bucket, key_prefix = self._parse_s3_path(output_path)
            output_key = f"{key_prefix}/{output_filename}"
            
            self.s3_client.put_object(
                Bucket=bucket,
                Key=output_key,
                Body=output_content,
                ContentType='application/json' if output_format != "parquet" else 'application/x-parquet'
            )
            
            logger.info(f"Uploaded processed file: s3://{bucket}/{output_key}")
            
        except Exception as e:
            logger.error(f"Error uploading processed file: {str(e)}")
    
    def _list_input_files(self, input_path: str) -> List[str]:
        """List input files from S3 path."""
        try:
            bucket, prefix = self._parse_s3_path(input_path)
            
            response = self.s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix,
                MaxKeys=1000
            )
            
            files = []
            for obj in response.get('KeyContents', []):
                if not obj['Key'].endswith('/'):
                    files.append(f"s3://{bucket}/{obj['Key']}")
            
            return files
            
        except Exception as e:
            logger.error(f"Error listing input files: {str(e)}")
            return []
    
    def _count_processed_items(self, output_path: str) -> int:
        """Count processed items in output path."""
        try:
            bucket, prefix = self._parse_s3_path(output_path)
            
            response = self.s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix,
                MaxKeys=1000
            )
            
            return len(response.get('KeyContents', []))
            
        except Exception as e:
            logger.error(f"Error counting processed items: {str(e)}")
            return 0
    
    def _validate_job_config(self, job_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate batch job configuration."""
        result = {"valid": True, "error": None}
        
        # Check required fields
        required_fields = ["input_path", "output_path", "data_type"]
        for field in required_fields:
            if field not in job_config:
                result["valid"] = False
                result["error"] = f"Missing required field: {field}"
                return result
        
        # Validate data type
        valid_data_types = ["text", "image", "audio", "survey"]
        if job_config.get("data_type") not in valid_data_types:
            result["valid"] = False
            result["error"] = f"Invalid data type: {job_config.get('data_type')}"
            return result
        
        # Validate output format
        valid_formats = ["json", "jsonl", "parquet"]
        output_format = job_config.get("output_format", "jsonl")
        if output_format not in valid_formats:
            result["valid"] = False
            result["error"] = f"Invalid output format: {output_format}"
            return result
        
        return result
    
    def _estimate_job_cost(self, job_config: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate cost for batch job."""
        # Instance pricing (example for us-east-1)
        instance_pricing = {
            "ml.m5.large": 0.104,  # per hour
            "ml.m5.xlarge": 0.208,
            "ml.m5.2xlarge": 0.416,
            "ml.c5.large": 0.085,
            "ml.c5.xlarge": 0.170,
            "ml.r5.large": 0.126,
            "ml.r5.xlarge": 0.252
        }
        
        instance_type = job_config.get("instance_type", "ml.m5.xlarge")
        instance_count = job_config.get("instance_count", 1)
        estimated_hours = job_config.get("estimated_hours", 1.0)
        
        hourly_rate = instance_pricing.get(instance_type, 0.208)
        hourly_cost = hourly_rate * instance_count
        total_cost = hourly_cost * estimated_hours
        
        # Add data transfer costs (simplified)
        data_size_gb = job_config.get("estimated_data_size_gb", 1.0)
        transfer_cost = data_size_gb * 0.05  # $0.05 per GB
        
        total_cost += transfer_cost
        
        return {
            "instance_type": instance_type,
            "instance_count": instance_count,
            "estimated_hours": estimated_hours,
            "hourly_rate": hourly_rate,
            "compute_cost": hourly_cost * estimated_hours,
            "transfer_cost": transfer_cost,
            "total_estimated_cost": total_cost,
            "currency": "USD"
        }
    
    def _get_sagemaker_script(self, data_type: str) -> str:
        """Get appropriate SageMaker processing script."""
        scripts = {
            "text": "format_text_data.py",
            "image": "format_image_data.py",
            "audio": "format_audio_data.py",
            "survey": "format_survey_data.py"
        }
        
        return scripts.get(data_type, "format_text_data.py")
    
    def _parse_s3_path(self, s3_path: str) -> tuple:
        """Parse S3 path into bucket and key."""
        # Remove s3:// prefix if present
        if s3_path.startswith("s3://"):
            s3_path = s3_path[5:]
        
        # Split at first slash
        if "/" in s3_path:
            bucket, key = s3_path.split("/", 1)
        else:
            bucket = s3_path
            key = ""
        
        return bucket, key
    
    def _store_job_record(self, job_record: Dict[str, Any]) -> None:
        """Store job record in DynamoDB."""
        try:
            dynamodb = boto3.resource('dynamodb')
            table = dynamodb.Table(os.environ.get('JOB_TRACKING_TABLE', 'fm-batch-jobs'))
            
            table.put_item(Item=job_record)
            
        except Exception as e:
            logger.error(f"Error storing job record: {str(e)}")
    
    def _get_job_record(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job record from DynamoDB."""
        try:
            dynamodb = boto3.resource('dynamodb')
            table = dynamodb.Table(os.environ.get('JOB_TRACKING_TABLE', 'fm-batch-jobs'))
            
            response = table.get_item(Key={"job_id": job_id})
            return response.get("Item")
            
        except Exception as e:
            logger.error(f"Error getting job record: {str(e)}")
            return None
    
    def _update_job_record(self, job_id: str, job_record: Dict[str, Any]) -> None:
        """Update job record in DynamoDB."""
        try:
            dynamodb = boto3.resource('dynamodb')
            table = dynamodb.Table(os.environ.get('JOB_TRACKING_TABLE', 'fm-batch-jobs'))
            
            table.update_item(
                Key={"job_id": job_id},
                UpdateExpression="set #record = :record",
                ExpressionAttributeValues={":record": job_record}
            )
            
        except Exception as e:
            logger.error(f"Error updating job record: {str(e)}")
    
    def _job_completion_callback(self, future, job_id: str) -> None:
        """Callback for job completion."""
        try:
            result = future.result()
            logger.info(f"Job {job_id} completed with status: {result.get('status', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Job {job_id} failed with error: {str(e)}")
    
    def _send_job_completion_metrics(self, job_record: Dict[str, Any]) -> None:
        """Send job completion metrics to CloudWatch."""
        try:
            metrics = []
            
            # Job completion metrics
            metrics.append({
                "MetricName": "BatchJobCompleted",
                "Value": 1,
                "Unit": "Count",
                "Dimensions": [
                    {"Name": "DataType", "Value": job_record.get("config", {}).get("data_type", "unknown")},
                    {"Name": "TargetModel", "Value": job_record.get("config", {}).get("target_model", "unknown")}
                ]
            })
            
            # Processing time metrics
            processing_time = job_record.get("processing_time_seconds", 0)
            if processing_time > 0:
                metrics.append({
                    "MetricName": "BatchJobProcessingTime",
                    "Value": processing_time,
                    "Unit": "Seconds",
                    "Dimensions": [
                        {"Name": "DataType", "Value": job_record.get("config", {}).get("data_type", "unknown")}
                    ]
                })
            
            # Items processed metrics
            items_processed = job_record.get("items_processed", 0)
            metrics.append({
                "MetricName": "BatchJobItemsProcessed",
                "Value": items_processed,
                "Unit": "Count",
                "Dimensions": [
                    {"Name": "DataType", "Value": job_record.get("config", {}).get("data_type", "unknown")}
                ]
            })
            
            # Error count metrics
            error_count = job_record.get("error_count", 0)
            metrics.append({
                "MetricName": "BatchJobErrors",
                "Value": error_count,
                "Unit": "Count",
                "Dimensions": [
                    {"Name": "DataType", "Value": job_record.get("config", {}).get("data_type", "unknown")}
                ]
            })
            
            # Cost metrics
            cost_estimate = job_record.get("cost_estimate", {})
            if cost_estimate:
                metrics.append({
                    "MetricName": "BatchJobEstimatedCost",
                    "Value": cost_estimate.get("total_estimated_cost", 0),
                    "Unit": "USD",
                    "Dimensions": [
                        {"Name": "DataType", "Value": job_record.get("config", {}).get("data_type", "unknown")}
                    ]
                })
            
            # Send metrics to CloudWatch
            self.cloudwatch_client.put_metric_data(
                Namespace="CustomerFeedback/BatchProcessing",
                MetricData=metrics
            )
            
            logger.info(f"Sent {len(metrics)} job completion metrics to CloudWatch")
            
        except Exception as e:
            logger.error(f"Error sending job completion metrics: {str(e)}")
    
    def _send_job_error_metrics(self, job_id: str, error_message: str) -> None:
        """Send job error metrics to CloudWatch."""
        try:
            metrics = [{
                "MetricName": "BatchJobFailed",
                "Value": 1,
                "Unit": "Count",
                "Dimensions": [
                    {"Name": "JobId", "Value": job_id},
                    {"Name": "ErrorType", "Value": "processing_error"}
                ]
            }]
            
            self.cloudwatch_client.put_metric_data(
                Namespace="CustomerFeedback/BatchProcessing",
                MetricData=metrics
            )
            
            logger.info(f"Sent job error metrics for {job_id}")
            
        except Exception as e:
            logger.error(f"Error sending job error metrics: {str(e)}")
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a batch job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job status information
        """
        return self._get_job_record(job_id)
    
    def list_active_jobs(self) -> List[Dict[str, Any]]:
        """
        List all active batch jobs.
        
        Returns:
            List of active job records
        """
        try:
            dynamodb = boto3.resource('dynamodb')
            table = dynamodb.Table(os.environ.get('JOB_TRACKING_TABLE', 'fm-batch-jobs'))
            
            response = table.scan(
                FilterExpression="attribute_exists(status) AND status IN (:statuses)",
                ExpressionAttributeValues={":statuses": ["created", "queued", "processing", "submitting_to_sagemaker"]}
            )
            
            return response.get("Items", [])
            
        except Exception as e:
            logger.error(f"Error listing active jobs: {str(e)}")
            return []
    
    def shutdown_batch_processing(self) -> None:
        """Shutdown batch processing gracefully."""
        logger.info("Shutting down batch processing")
        self.batch_queue.put("SHUTDOWN")

# Factory function
def create_batch_processor() -> BatchProcessor:
    """
    Factory function to create a batch processor instance.
    
    Returns:
        BatchProcessor instance
    """
    return BatchProcessor()

if __name__ == "__main__":
    # Example usage
    processor = create_batch_processor()
    
    # Sample job configuration
    job_config = {
        "input_path": "s3://customer-feedback-analysis-bucket/processed-data/text/",
        "output_path": "s3://customer-feedback-analysis-bucket/formatted-data/text/",
        "data_type": "text",
        "target_model": "claude-v2",
        "output_format": "jsonl",
        "use_sagemaker": False,
        "instance_type": "ml.m5.xlarge",
        "instance_count": 1,
        "estimated_hours": 2.0,
        "estimated_data_size_gb": 5.0,
        "max_runtime_seconds": 3600
    }
    
    # Create batch job
    job = processor.create_batch_job(job_config)
    print(f"Created batch job: {json.dumps(job, indent=2)}")
    
    # Start batch processing (in a real implementation, this would be a separate service)
    # processor.process_batch_queue(max_workers=5)
    
    # Get job status
    job_id = job["job_id"]
    status = processor.get_job_status(job_id)
    print(f"Job status: {json.dumps(status, indent=2)}")