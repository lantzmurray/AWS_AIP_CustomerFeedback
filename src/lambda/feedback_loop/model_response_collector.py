#!/usr/bin/env python3
"""
Model Response Collector for Feedback Loop

This module captures and analyzes foundation model responses to create
a closed-loop system for data quality enhancement.

Author: AWS AI Project
Date: 2025-12-12
"""

import json
import boto3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from decimal import Decimal
import re
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelResponse:
    """Data structure for foundation model responses."""
    response_id: str
    model_name: str
    model_version: str
    input_data_id: str
    input_data_type: str
    request_timestamp: datetime
    response_timestamp: datetime
    response_content: str
    confidence_score: Optional[float] = None
    token_count: Optional[int] = None
    processing_time_ms: Optional[int] = None
    error_occurred: bool = False
    error_message: Optional[str] = None
    quality_metrics: Optional[Dict[str, Any]] = None

@dataclass
class QualityInsight:
    """Data structure for quality insights derived from model responses."""
    insight_id: str
    response_id: str
    insight_type: str  # 'response_quality', 'data_issue', 'confidence_pattern'
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    recommendation: str
    affected_data_type: str
    created_timestamp: datetime
    confidence_score: float

class ModelResponseCollector:
    """
    Collects and analyzes foundation model responses for quality enhancement.
    
    This class implements the core feedback loop functionality by:
    1. Capturing model responses with quality metrics
    2. Analyzing response patterns and quality trends
    3. Identifying data quality issues from model performance
    4. Triggering adaptive quality improvements
    """
    
    def __init__(self, region_name: str = 'us-east-1'):
        """Initialize the Model Response Collector."""
        self.region_name = region_name
        self.s3_client = boto3.client('s3', region_name=region_name)
        self.dynamodb_client = boto3.resource('dynamodb', region_name=region_name)
        self.cloudwatch_client = boto3.client('cloudwatch', region_name=region_name)
        self.lambda_client = boto3.client('lambda', region_name=region_name)
        
        # Initialize DynamoDB tables
        self.responses_table = self.dynamodb_client.Table('model_responses')
        self.insights_table = self.dynamodb_client.Table('quality_insights')
        self.quality_trends_table = self.dynamodb_client.Table('quality_trends')
        
        # S3 buckets
        self.responses_bucket = 'customer-feedback-analysis-lm-responses'
        self.insights_bucket = 'customer-feedback-analysis-lm-insights'
        
        logger.info("ModelResponseCollector initialized")
    
    def capture_model_response(self, 
                            model_name: str,
                            model_version: str,
                            input_data_id: str,
                            input_data_type: str,
                            response_content: str,
                            confidence_score: Optional[float] = None,
                            token_count: Optional[int] = None,
                            processing_time_ms: Optional[int] = None,
                            error_occurred: bool = False,
                            error_message: Optional[str] = None) -> str:
        """
        Capture a foundation model response with associated metadata.
        
        Args:
            model_name: Name of the foundation model (e.g., 'claude-v2')
            model_version: Version of the model
            input_data_id: Unique identifier for the input data
            input_data_type: Type of input data (text, image, audio, survey)
            response_content: The model's response content
            confidence_score: Model's confidence score (if available)
            token_count: Number of tokens in response
            processing_time_ms: Processing time in milliseconds
            error_occurred: Whether an error occurred during processing
            error_message: Error message if an error occurred
            
        Returns:
            response_id: Unique identifier for the captured response
        """
        try:
            # Generate unique response ID
            response_id = self._generate_response_id(
                model_name, input_data_id, datetime.now()
            )
            
            # Create response object
            response = ModelResponse(
                response_id=response_id,
                model_name=model_name,
                model_version=model_version,
                input_data_id=input_data_id,
                input_data_type=input_data_type,
                request_timestamp=datetime.now(),
                response_timestamp=datetime.now(),
                response_content=response_content,
                confidence_score=confidence_score,
                token_count=token_count,
                processing_time_ms=processing_time_ms,
                error_occurred=error_occurred,
                error_message=error_message,
                quality_metrics=self._calculate_response_quality_metrics(response_content)
            )
            
            # Store in DynamoDB
            self._store_response_in_dynamodb(response)
            
            # Store raw response in S3 for detailed analysis
            self._store_response_in_s3(response)
            
            # Send metrics to CloudWatch
            self._send_response_metrics(response)
            
            # Trigger immediate analysis for critical issues
            if error_occurred or (confidence_score and confidence_score < 0.5):
                self._trigger_immediate_analysis(response)
            
            logger.info(f"Captured model response: {response_id}")
            return response_id
            
        except Exception as e:
            logger.error(f"Error capturing model response: {str(e)}")
            raise
    
    def analyze_response_patterns(self, 
                               time_window_hours: int = 24) -> List[QualityInsight]:
        """
        Analyze model response patterns to identify quality issues.
        
        Args:
            time_window_hours: Time window in hours for analysis
            
        Returns:
            List of quality insights identified from response patterns
        """
        try:
            # Get recent responses from DynamoDB
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            recent_responses = self._get_recent_responses(cutoff_time)
            
            insights = []
            
            # Analyze confidence score patterns
            confidence_insights = self._analyze_confidence_patterns(recent_responses)
            insights.extend(confidence_insights)
            
            # Analyze error patterns
            error_insights = self._analyze_error_patterns(recent_responses)
            insights.extend(error_insights)
            
            # Analyze response quality patterns
            quality_insights = self._analyze_response_quality_patterns(recent_responses)
            insights.extend(quality_insights)
            
            # Analyze processing time patterns
            performance_insights = self._analyze_performance_patterns(recent_responses)
            insights.extend(performance_insights)
            
            # Store insights
            for insight in insights:
                self._store_quality_insight(insight)
            
            logger.info(f"Generated {len(insights)} quality insights from pattern analysis")
            return insights
            
        except Exception as e:
            logger.error(f"Error analyzing response patterns: {str(e)}")
            return []
    
    def generate_quality_improvement_actions(self, 
                                           insights: List[QualityInsight]) -> List[Dict[str, Any]]:
        """
        Generate specific quality improvement actions based on insights.
        
        Args:
            insights: List of quality insights
            
        Returns:
            List of improvement actions to be executed
        """
        try:
            actions = []
            
            for insight in insights:
                if insight.severity in ['high', 'critical']:
                    action = self._create_improvement_action(insight)
                    if action:
                        actions.append(action)
            
            # Prioritize actions by severity and potential impact
            actions.sort(key=lambda x: (
                self._severity_priority(x['severity']),
                -x.get('impact_score', 0)
            ))
            
            logger.info(f"Generated {len(actions)} quality improvement actions")
            return actions
            
        except Exception as e:
            logger.error(f"Error generating improvement actions: {str(e)}")
            return []
    
    def execute_improvement_actions(self, actions: List[Dict[str, Any]]) -> List[str]:
        """
        Execute quality improvement actions.
        
        Args:
            actions: List of improvement actions to execute
            
        Returns:
            List of execution results
        """
        try:
            results = []
            
            for action in actions:
                result = self._execute_single_action(action)
                results.append(result)
                
                # Log execution to CloudWatch
                self._log_action_execution(action, result)
            
            logger.info(f"Executed {len(actions)} improvement actions")
            return results
            
        except Exception as e:
            logger.error(f"Error executing improvement actions: {str(e)}")
            return []
    
    def get_quality_trends(self, 
                         data_type: Optional[str] = None,
                         days: int = 7) -> Dict[str, Any]:
        """
        Get quality trends over time.
        
        Args:
            data_type: Specific data type to analyze (optional)
            days: Number of days to analyze
            
        Returns:
            Quality trends data
        """
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            # Get trend data from DynamoDB
            trend_data = self._get_quality_trend_data(start_time, end_time, data_type)
            
            # Calculate trend metrics
            trends = {
                'period': f'{days} days',
                'data_type': data_type or 'all',
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'metrics': self._calculate_trend_metrics(trend_data),
                'recommendations': self._generate_trend_recommendations(trend_data)
            }
            
            return trends
            
        except Exception as e:
            logger.error(f"Error getting quality trends: {str(e)}")
            return {}
    
    # Private helper methods
    
    def _generate_response_id(self, model_name: str, input_data_id: str, timestamp: datetime) -> str:
        """Generate unique response ID."""
        content = f"{model_name}_{input_data_id}_{timestamp.isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _calculate_response_quality_metrics(self, response_content: str) -> Dict[str, Any]:
        """Calculate quality metrics for model response."""
        if not response_content:
            return {'length': 0, 'completeness': 0.0, 'structure_score': 0.0}
        
        length = len(response_content)
        
        # Basic quality indicators
        completeness = min(1.0, length / 1000)  # Assume 1000 chars is complete
        
        # Structure score based on JSON formatting or structured content
        structure_score = 0.0
        try:
            json.loads(response_content)
            structure_score = 1.0  # Valid JSON
        except:
            # Check for other structured patterns
            if re.search(r'\{.*\}|\[.*\]', response_content):
                structure_score = 0.7
            elif re.search(r'\n.*:', response_content):
                structure_score = 0.5
        
        return {
            'length': length,
            'completeness': completeness,
            'structure_score': structure_score
        }
    
    def _store_response_in_dynamodb(self, response: ModelResponse):
        """Store response in DynamoDB table."""
        item = asdict(response)
        
        # Convert datetime to ISO string for DynamoDB
        item['request_timestamp'] = response.request_timestamp.isoformat()
        item['response_timestamp'] = response.response_timestamp.isoformat()
        
        # Convert any float values to Decimal for DynamoDB
        if response.confidence_score is not None:
            item['confidence_score'] = Decimal(str(response.confidence_score))
        
        self.responses_table.put_item(Item=item)
    
    def _store_response_in_s3(self, response: ModelResponse):
        """Store raw response in S3."""
        key = f"responses/{response.response_id}/{response.input_data_type}/{response.input_data_id}.json"
        
        content = {
            'response': asdict(response),
            'raw_content': response.response_content,
            'captured_at': datetime.now().isoformat()
        }
        
        self.s3_client.put_object(
            Bucket=self.responses_bucket,
            Key=key,
            Body=json.dumps(content, default=str),
            ContentType='application/json'
        )
    
    def _send_response_metrics(self, response: ModelResponse):
        """Send response metrics to CloudWatch."""
        metrics = [
            {
                'MetricName': 'ModelResponseCount',
                'Value': 1,
                'Unit': 'Count',
                'Dimensions': [
                    {'Name': 'ModelName', 'Value': response.model_name},
                    {'Name': 'DataType', 'Value': response.input_data_type},
                    {'Name': 'ErrorOccurred', 'Value': str(response.error_occurred)}
                ]
            }
        ]
        
        if response.confidence_score is not None:
            metrics.append({
                'MetricName': 'ModelConfidenceScore',
                'Value': response.confidence_score,
                'Unit': 'None',
                'Dimensions': [
                    {'Name': 'ModelName', 'Value': response.model_name},
                    {'Name': 'DataType', 'Value': response.input_data_type}
                ]
            })
        
        if response.processing_time_ms is not None:
            metrics.append({
                'MetricName': 'ModelProcessingTime',
                'Value': response.processing_time_ms,
                'Unit': 'Milliseconds',
                'Dimensions': [
                    {'Name': 'ModelName', 'Value': response.model_name},
                    {'Name': 'DataType', 'Value': response.input_data_type}
                ]
            })
        
        self.cloudwatch_client.put_metric_data(
            Namespace='CustomerFeedback/ModelResponses',
            MetricData=metrics
        )
    
    def _trigger_immediate_analysis(self, response: ModelResponse):
        """Trigger immediate analysis for critical issues."""
        try:
            # Invoke analysis Lambda asynchronously
            self.lambda_client.invoke(
                FunctionName='ImmediateQualityAnalysis',
                InvocationType='Event',
                Payload=json.dumps({
                    'response_id': response.response_id,
                    'issue_type': 'low_confidence' if response.confidence_score and response.confidence_score < 0.5 else 'error',
                    'data_type': response.input_data_type
                })
            )
        except Exception as e:
            logger.error(f"Error triggering immediate analysis: {str(e)}")
    
    def _get_recent_responses(self, cutoff_time: datetime) -> List[Dict[str, Any]]:
        """Get recent responses from DynamoDB."""
        try:
            response = self.responses_table.scan(
                FilterExpression='#ts >= :cutoff',
                ExpressionAttributeNames={'#ts': 'response_timestamp'},
                ExpressionAttributeValues={':cutoff': cutoff_time.isoformat()}
            )
            return response.get('Items', [])
        except Exception as e:
            logger.error(f"Error getting recent responses: {str(e)}")
            return []
    
    def _analyze_confidence_patterns(self, responses: List[Dict[str, Any]]) -> List[QualityInsight]:
        """Analyze confidence score patterns."""
        insights = []
        
        if not responses:
            return insights
        
        # Group by data type
        by_data_type = {}
        for response in responses:
            data_type = response.get('input_data_type', 'unknown')
            if data_type not in by_data_type:
                by_data_type[data_type] = []
            by_data_type[data_type].append(response)
        
        for data_type, type_responses in by_data_type.items():
            confidence_scores = [
                r.get('confidence_score', 0) for r in type_responses 
                if r.get('confidence_score') is not None
            ]
            
            if not confidence_scores:
                continue
            
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            low_confidence_count = sum(1 for score in confidence_scores if score < 0.5)
            
            # Generate insight if confidence is consistently low
            if avg_confidence < 0.6 or low_confidence_count / len(confidence_scores) > 0.3:
                insight = QualityInsight(
                    insight_id=self._generate_response_id('confidence', data_type, datetime.now()),
                    response_id='pattern_analysis',
                    insight_type='confidence_pattern',
                    severity='high' if avg_confidence < 0.5 else 'medium',
                    description=f'Low confidence scores detected for {data_type} data (avg: {avg_confidence:.2f})',
                    recommendation=f'Review data quality and preprocessing for {data_type} inputs',
                    affected_data_type=data_type,
                    created_timestamp=datetime.now(),
                    confidence_score=0.8
                )
                insights.append(insight)
        
        return insights
    
    def _analyze_error_patterns(self, responses: List[Dict[str, Any]]) -> List[QualityInsight]:
        """Analyze error patterns in responses."""
        insights = []
        
        error_responses = [r for r in responses if r.get('error_occurred', False)]
        
        if not error_responses:
            return insights
        
        # Group errors by data type
        error_by_type = {}
        for response in error_responses:
            data_type = response.get('input_data_type', 'unknown')
            if data_type not in error_by_type:
                error_by_type[data_type] = []
            error_by_type[data_type].append(response)
        
        for data_type, type_errors in error_by_type.items():
            error_rate = len(type_errors) / len([r for r in responses if r.get('input_data_type') == data_type])
            
            if error_rate > 0.1:  # More than 10% error rate
                insight = QualityInsight(
                    insight_id=self._generate_response_id('error', data_type, datetime.now()),
                    response_id='pattern_analysis',
                    insight_type='data_issue',
                    severity='critical' if error_rate > 0.3 else 'high',
                    description=f'High error rate ({error_rate:.1%}) detected for {data_type} data',
                    recommendation=f'Investigate data format and preprocessing pipeline for {data_type}',
                    affected_data_type=data_type,
                    created_timestamp=datetime.now(),
                    confidence_score=0.9
                )
                insights.append(insight)
        
        return insights
    
    def _analyze_response_quality_patterns(self, responses: List[Dict[str, Any]]) -> List[QualityInsight]:
        """Analyze response quality patterns."""
        insights = []
        
        # Analyze response completeness and structure
        quality_scores = []
        for response in responses:
            quality_metrics = response.get('quality_metrics', {})
            if quality_metrics:
                completeness = quality_metrics.get('completeness', 0)
                structure_score = quality_metrics.get('structure_score', 0)
                overall_quality = (completeness + structure_score) / 2
                quality_scores.append(overall_quality)
        
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            low_quality_count = sum(1 for score in quality_scores if score < 0.5)
            
            if avg_quality < 0.6 or low_quality_count / len(quality_scores) > 0.3:
                insight = QualityInsight(
                    insight_id=self._generate_response_id('quality', 'all', datetime.now()),
                    response_id='pattern_analysis',
                    insight_type='response_quality',
                    severity='medium' if avg_quality < 0.5 else 'low',
                    description=f'Low response quality detected (avg score: {avg_quality:.2f})',
                    recommendation='Review model prompts and input data formatting',
                    affected_data_type='all',
                    created_timestamp=datetime.now(),
                    confidence_score=0.7
                )
                insights.append(insight)
        
        return insights
    
    def _analyze_performance_patterns(self, responses: List[Dict[str, Any]]) -> List[QualityInsight]:
        """Analyze processing time patterns."""
        insights = []
        
        processing_times = [
            r.get('processing_time_ms', 0) for r in responses 
            if r.get('processing_time_ms') is not None
        ]
        
        if not processing_times:
            return insights
        
        avg_time = sum(processing_times) / len(processing_times)
        max_time = max(processing_times)
        
        # Flag if processing times are unusually high
        if avg_time > 10000:  # More than 10 seconds average
            insight = QualityInsight(
                insight_id=self._generate_response_id('performance', 'all', datetime.now()),
                response_id='pattern_analysis',
                insight_type='performance_issue',
                severity='medium' if avg_time < 20000 else 'high',
                description=f'High processing times detected (avg: {avg_time/1000:.1f}s, max: {max_time/1000:.1f}s)',
                recommendation='Optimize model parameters or consider model scaling',
                affected_data_type='all',
                created_timestamp=datetime.now(),
                confidence_score=0.8
            )
            insights.append(insight)
        
        return insights
    
    def _store_quality_insight(self, insight: QualityInsight):
        """Store quality insight in DynamoDB."""
        item = asdict(insight)
        item['created_timestamp'] = insight.created_timestamp.isoformat()
        item['confidence_score'] = Decimal(str(insight.confidence_score))
        
        self.insights_table.put_item(Item=item)
        
        # Also store in S3 for analysis
        key = f"insights/{insight.insight_type}/{insight.insight_id}.json"
        self.s3_client.put_object(
            Bucket=self.insights_bucket,
            Key=key,
            Body=json.dumps(item, default=str),
            ContentType='application/json'
        )
    
    def _create_improvement_action(self, insight: QualityInsight) -> Optional[Dict[str, Any]]:
        """Create improvement action based on insight."""
        action_map = {
            'confidence_pattern': self._create_confidence_improvement_action,
            'data_issue': self._create_data_issue_improvement_action,
            'response_quality': self._create_quality_improvement_action,
            'performance_issue': self._create_performance_improvement_action
        }
        
        action_creator = action_map.get(insight.insight_type)
        if action_creator:
            return action_creator(insight)
        
        return None
    
    def _create_confidence_improvement_action(self, insight: QualityInsight) -> Dict[str, Any]:
        """Create action for confidence-related issues."""
        return {
            'action_id': f"confidence_improvement_{insight.insight_id}",
            'action_type': 'data_quality_improvement',
            'severity': insight.severity,
            'target_data_type': insight.affected_data_type,
            'description': f"Improve data quality for {insight.affected_data_type} to boost model confidence",
            'steps': [
                'Review and clean input data format',
                'Enhance preprocessing pipeline',
                'Adjust model parameters if needed'
            ],
            'impact_score': 8.0,
            'estimated_effort': 'medium',
            'insight_id': insight.insight_id
        }
    
    def _create_data_issue_improvement_action(self, insight: QualityInsight) -> Dict[str, Any]:
        """Create action for data-related issues."""
        return {
            'action_id': f"data_issue_fix_{insight.insight_id}",
            'action_type': 'data_pipeline_fix',
            'severity': insight.severity,
            'target_data_type': insight.affected_data_type,
            'description': f"Fix data pipeline issues for {insight.affected_data_type}",
            'steps': [
                'Identify root cause of data format issues',
                'Update validation rules',
                'Implement better error handling',
                'Retest with sample data'
            ],
            'impact_score': 9.0,
            'estimated_effort': 'high',
            'insight_id': insight.insight_id
        }
    
    def _create_quality_improvement_action(self, insight: QualityInsight) -> Dict[str, Any]:
        """Create action for response quality issues."""
        return {
            'action_id': f"response_quality_{insight.insight_id}",
            'action_type': 'prompt_optimization',
            'severity': insight.severity,
            'target_data_type': insight.affected_data_type,
            'description': f"Optimize prompts and formatting for better response quality",
            'steps': [
                'Review current prompt templates',
                'Enhance prompt structure',
                'Add better formatting instructions',
                'Test with various inputs'
            ],
            'impact_score': 6.0,
            'estimated_effort': 'low',
            'insight_id': insight.insight_id
        }
    
    def _create_performance_improvement_action(self, insight: QualityInsight) -> Dict[str, Any]:
        """Create action for performance issues."""
        return {
            'action_id': f"performance_opt_{insight.insight_id}",
            'action_type': 'performance_optimization',
            'severity': insight.severity,
            'target_data_type': insight.affected_data_type,
            'description': f"Optimize processing performance for {insight.affected_data_type}",
            'steps': [
                'Analyze bottlenecks in processing pipeline',
                'Optimize Lambda memory allocation',
                'Consider model scaling or caching',
                'Implement batch processing where possible'
            ],
            'impact_score': 7.0,
            'estimated_effort': 'medium',
            'insight_id': insight.insight_id
        }
    
    def _severity_priority(self, severity: str) -> int:
        """Get numeric priority for severity level."""
        priority_map = {
            'critical': 4,
            'high': 3,
            'medium': 2,
            'low': 1
        }
        return priority_map.get(severity, 0)
    
    def _execute_single_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single improvement action."""
        try:
            action_type = action.get('action_type')
            
            if action_type == 'data_quality_improvement':
                return self._execute_data_quality_improvement(action)
            elif action_type == 'data_pipeline_fix':
                return self._execute_data_pipeline_fix(action)
            elif action_type == 'prompt_optimization':
                return self._execute_prompt_optimization(action)
            elif action_type == 'performance_optimization':
                return self._execute_performance_optimization(action)
            else:
                return {
                    'action_id': action.get('action_id'),
                    'status': 'failed',
                    'error': f'Unknown action type: {action_type}'
                }
                
        except Exception as e:
            return {
                'action_id': action.get('action_id'),
                'status': 'failed',
                'error': str(e)
            }
    
    def _execute_data_quality_improvement(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data quality improvement action."""
        # Trigger data reprocessing with enhanced quality checks
        target_data_type = action.get('target_data_type')
        
        # Invoke data quality enhancement Lambda
        response = self.lambda_client.invoke(
            FunctionName='DataQualityEnhancement',
            Payload=json.dumps({
                'data_type': target_data_type,
                'enhancement_type': 'quality_improvement',
                'action_id': action.get('action_id')
            })
        )
        
        return {
            'action_id': action.get('action_id'),
            'status': 'initiated',
            'message': f'Data quality improvement initiated for {target_data_type}',
            'lambda_response': response.get('Payload', {}).read().decode()
        }
    
    def _execute_data_pipeline_fix(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data pipeline fix action."""
        target_data_type = action.get('target_data_type')
        
        # Trigger pipeline fix Lambda
        response = self.lambda_client.invoke(
            FunctionName='DataPipelineFix',
            Payload=json.dumps({
                'data_type': target_data_type,
                'fix_type': 'pipeline_repair',
                'action_id': action.get('action_id')
            })
        )
        
        return {
            'action_id': action.get('action_id'),
            'status': 'initiated',
            'message': f'Pipeline fix initiated for {target_data_type}',
            'lambda_response': response.get('Payload', {}).read().decode()
        }
    
    def _execute_prompt_optimization(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute prompt optimization action."""
        target_data_type = action.get('target_data_type')
        
        # Trigger prompt optimization Lambda
        response = self.lambda_client.invoke(
            FunctionName='PromptOptimization',
            Payload=json.dumps({
                'data_type': target_data_type,
                'optimization_type': 'quality_enhancement',
                'action_id': action.get('action_id')
            })
        )
        
        return {
            'action_id': action.get('action_id'),
            'status': 'initiated',
            'message': f'Prompt optimization initiated for {target_data_type}',
            'lambda_response': response.get('Payload', {}).read().decode()
        }
    
    def _execute_performance_optimization(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute performance optimization action."""
        target_data_type = action.get('target_data_type')
        
        # Trigger performance optimization Lambda
        response = self.lambda_client.invoke(
            FunctionName='PerformanceOptimization',
            Payload=json.dumps({
                'data_type': target_data_type,
                'optimization_type': 'performance_boost',
                'action_id': action.get('action_id')
            })
        )
        
        return {
            'action_id': action.get('action_id'),
            'status': 'initiated',
            'message': f'Performance optimization initiated for {target_data_type}',
            'lambda_response': response.get('Payload', {}).read().decode()
        }
    
    def _log_action_execution(self, action: Dict[str, Any], result: Dict[str, Any]):
        """Log action execution to CloudWatch."""
        self.cloudwatch_client.put_metric_data(
            Namespace='CustomerFeedback/QualityImprovement',
            MetricData=[
                {
                    'MetricName': 'ImprovementActionExecuted',
                    'Value': 1,
                    'Unit': 'Count',
                    'Dimensions': [
                        {'Name': 'ActionType', 'Value': action.get('action_type', 'unknown')},
                        {'Name': 'Severity', 'Value': action.get('severity', 'unknown')},
                        {'Name': 'Status', 'Value': result.get('status', 'unknown')}
                    ]
                }
            ]
        )
    
    def _get_quality_trend_data(self, start_time: datetime, end_time: datetime, data_type: Optional[str]) -> List[Dict[str, Any]]:
        """Get quality trend data from DynamoDB."""
        try:
            # Query quality trends table
            filter_expr = '#ts BETWEEN :start AND :end'
            expr_attrs = {'#ts': 'timestamp'}
            expr_values = {
                ':start': start_time.isoformat(),
                ':end': end_time.isoformat()
            }
            
            if data_type:
                filter_expr += ' AND data_type = :dt'
                expr_values[':dt'] = data_type
            
            response = self.quality_trends_table.scan(
                FilterExpression=filter_expr,
                ExpressionAttributeNames=expr_attrs,
                ExpressionAttributeValues=expr_values
            )
            
            return response.get('Items', [])
            
        except Exception as e:
            logger.error(f"Error getting quality trend data: {str(e)}")
            return []
    
    def _calculate_trend_metrics(self, trend_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate trend metrics from data."""
        if not trend_data:
            return {}
        
        # Calculate various trend metrics
        confidence_scores = [d.get('avg_confidence', 0) for d in trend_data if d.get('avg_confidence')]
        error_rates = [d.get('error_rate', 0) for d in trend_data if d.get('error_rate')]
        processing_times = [d.get('avg_processing_time', 0) for d in trend_data if d.get('avg_processing_time')]
        
        metrics = {}
        
        if confidence_scores:
            metrics['confidence_trend'] = {
                'average': sum(confidence_scores) / len(confidence_scores),
                'min': min(confidence_scores),
                'max': max(confidence_scores),
                'trend_direction': 'improving' if confidence_scores[-1] > confidence_scores[0] else 'declining'
            }
        
        if error_rates:
            metrics['error_rate_trend'] = {
                'average': sum(error_rates) / len(error_rates),
                'min': min(error_rates),
                'max': max(error_rates),
                'trend_direction': 'improving' if error_rates[-1] < error_rates[0] else 'declining'
            }
        
        if processing_times:
            metrics['performance_trend'] = {
                'average': sum(processing_times) / len(processing_times),
                'min': min(processing_times),
                'max': max(processing_times),
                'trend_direction': 'improving' if processing_times[-1] < processing_times[0] else 'declining'
            }
        
        return metrics
    
    def _generate_trend_recommendations(self, trend_data: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on trend data."""
        recommendations = []
        
        # Analyze trends and generate recommendations
        for data_point in trend_data[-5:]:  # Last 5 data points
            if data_point.get('error_rate', 0) > 0.1:
                recommendations.append(f"High error rate detected on {data_point.get('timestamp')}: investigate data pipeline")
            
            if data_point.get('avg_confidence', 0) < 0.6:
                recommendations.append(f"Low confidence scores on {data_point.get('timestamp')}: review data quality")
            
            if data_point.get('avg_processing_time', 0) > 10000:
                recommendations.append(f"Slow processing on {data_point.get('timestamp')}: consider optimization")
        
        # Remove duplicates and return
        return list(set(recommendations))

# Example usage and testing
if __name__ == "__main__":
    collector = ModelResponseCollector()
    
    # Test capturing a model response
    response_id = collector.capture_model_response(
        model_name="claude-v2",
        model_version="2.1",
        input_data_id="test-001",
        input_data_type="text",
        response_content='{"sentiment": "positive", "confidence": 0.85}',
        confidence_score=0.85,
        token_count=150,
        processing_time_ms=2500
    )
    
    print(f"Captured response: {response_id}")
    
    # Test pattern analysis
    insights = collector.analyze_response_patterns(time_window_hours=1)
    print(f"Generated {len(insights)} insights")
    
    # Test improvement actions
    actions = collector.generate_quality_improvement_actions(insights)
    print(f"Generated {len(actions)} improvement actions")
    
    # Test quality trends
    trends = collector.get_quality_trends(days=7)
    print(f"Quality trends: {trends}")