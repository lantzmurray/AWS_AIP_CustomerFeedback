#!/usr/bin/env python3
"""
Feedback Loop Orchestrator Lambda

This Lambda function orchestrates the feedback loop system by:
1. Collecting model responses
2. Analyzing patterns and generating insights
3. Triggering quality improvement actions
4. Monitoring the effectiveness of improvements

Author: AWS AI Project
Date: 2025-12-12
"""

import json
import boto3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import os

# Import our feedback loop components
from model_response_collector import ModelResponseCollector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
REGION_NAME = os.environ.get('AWS_REGION', 'us-east-1')
ANALYSIS_INTERVAL_HOURS = int(os.environ.get('ANALYSIS_INTERVAL_HOURS', '6'))
ENABLE_AUTO_IMPROVEMENTS = os.environ.get('ENABLE_AUTO_IMPROVEMENTS', 'true').lower() == 'true'

@dataclass
class FeedbackLoopConfig:
    """Configuration for feedback loop execution."""
    analysis_interval_hours: int
    enable_auto_improvements: bool
    min_confidence_threshold: float = 0.6
    max_error_rate_threshold: float = 0.1
    max_processing_time_ms: int = 10000

class FeedbackLoopOrchestrator:
    """
    Orchestrates the feedback loop system for continuous data quality improvement.
    
    This class implements the core feedback loop logic by:
    1. Coordinating model response collection
    2. Running periodic pattern analysis
    3. Executing improvement actions
    4. Monitoring and reporting on effectiveness
    """
    
    def __init__(self, config: Optional[FeedbackLoopConfig] = None):
        """Initialize the Feedback Loop Orchestrator."""
        self.config = config or FeedbackLoopConfig(
            analysis_interval_hours=ANALYSIS_INTERVAL_HOURS,
            enable_auto_improvements=ENABLE_AUTO_IMPROVEMENTS
        )
        
        # Initialize AWS clients
        self.lambda_client = boto3.client('lambda', region_name=REGION_NAME)
        self.cloudwatch_client = boto3.client('cloudwatch', region_name=REGION_NAME)
        self.sns_client = boto3.client('sns', region_name=REGION_NAME)
        
        # Initialize model response collector
        self.response_collector = ModelResponseCollector(region_name=REGION_NAME)
        
        # SNS topic for notifications
        self.notification_topic = os.environ.get('NOTIFICATION_TOPIC_ARN')
        
        logger.info("FeedbackLoopOrchestrator initialized")
    
    def run_feedback_loop_cycle(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a complete feedback loop cycle.
        
        Args:
            event: Lambda trigger event
            
        Returns:
            Results of the feedback loop cycle
        """
        try:
            logger.info("Starting feedback loop cycle")
            start_time = datetime.now()
            
            # Step 1: Analyze response patterns
            insights = self.response_collector.analyze_response_patterns(
                time_window_hours=self.config.analysis_interval_hours
            )
            
            # Step 2: Generate improvement actions
            actions = self.response_collector.generate_quality_improvement_actions(insights)
            
            # Step 3: Execute improvement actions (if auto-improvement is enabled)
            execution_results = []
            if self.config.enable_auto_improvements:
                execution_results = self.response_collector.execute_improvement_actions(actions)
            else:
                logger.info("Auto-improvements disabled, actions queued for manual review")
            
            # Step 4: Generate quality trends report
            trends = self.response_collector.get_quality_trends(days=7)
            
            # Step 5: Send notifications for critical issues
            critical_insights = [i for i in insights if i.severity in ['high', 'critical']]
            if critical_insights:
                self._send_critical_notifications(critical_insights, actions)
            
            # Step 6: Log cycle metrics
            cycle_duration = (datetime.now() - start_time).total_seconds() * 1000
            self._log_cycle_metrics(insights, actions, execution_results, cycle_duration)
            
            # Prepare results
            results = {
                'cycle_start_time': start_time.isoformat(),
                'cycle_duration_ms': cycle_duration,
                'insights_generated': len(insights),
                'actions_generated': len(actions),
                'actions_executed': len(execution_results),
                'critical_insights': len(critical_insights),
                'trends_summary': trends.get('metrics', {}),
                'auto_improvements_enabled': self.config.enable_auto_improvements
            }
            
            logger.info(f"Feedback loop cycle completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error in feedback loop cycle: {str(e)}")
            self._send_error_notification(str(e))
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def handle_model_response(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle incoming model response for immediate processing.
        
        Args:
            event: Event containing model response data
            
        Returns:
            Processing result
        """
        try:
            logger.info("Processing model response")
            
            # Extract response data from event
            response_data = event.get('response', {})
            model_metadata = event.get('model_metadata', {})
            
            # Capture the response
            response_id = self.response_collector.capture_model_response(
                model_name=model_metadata.get('model_name', 'unknown'),
                model_version=model_metadata.get('model_version', 'unknown'),
                input_data_id=response_data.get('input_data_id', 'unknown'),
                input_data_type=response_data.get('input_data_type', 'unknown'),
                response_content=response_data.get('response_content', ''),
                confidence_score=response_data.get('confidence_score'),
                token_count=response_data.get('token_count'),
                processing_time_ms=response_data.get('processing_time_ms'),
                error_occurred=response_data.get('error_occurred', False),
                error_message=response_data.get('error_message')
            )
            
            # Trigger immediate analysis if this is a critical response
            if (response_data.get('error_occurred', False) or 
                (response_data.get('confidence_score') and response_data.get('confidence_score') < 0.5)):
                
                self._trigger_immediate_analysis(response_id, response_data)
            
            return {
                'status': 'success',
                'response_id': response_id,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error handling model response: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_feedback_loop_status(self) -> Dict[str, Any]:
        """
        Get current status of the feedback loop system.
        
        Returns:
            Status information
        """
        try:
            # Get recent metrics
            trends = self.response_collector.get_quality_trends(days=1)
            
            # Get recent insights
            recent_insights = self.response_collector.analyze_response_patterns(time_window_hours=24)
            
            # Calculate system health
            health_score = self._calculate_system_health(trends, recent_insights)
            
            status = {
                'timestamp': datetime.now().isoformat(),
                'system_health_score': health_score,
                'config': {
                    'analysis_interval_hours': self.config.analysis_interval_hours,
                    'enable_auto_improvements': self.config.enable_auto_improvements,
                    'min_confidence_threshold': self.config.min_confidence_threshold,
                    'max_error_rate_threshold': self.config.max_error_rate_threshold
                },
                'recent_metrics': trends.get('metrics', {}),
                'recent_insights_count': len(recent_insights),
                'critical_insights_count': len([i for i in recent_insights if i.severity in ['high', 'critical']]),
                'health_status': self._get_health_status(health_score)
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting feedback loop status: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def update_feedback_loop_config(self, config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update feedback loop configuration.
        
        Args:
            config_updates: Configuration updates
            
        Returns:
            Update result
        """
        try:
            # Update configuration
            if 'analysis_interval_hours' in config_updates:
                self.config.analysis_interval_hours = config_updates['analysis_interval_hours']
            
            if 'enable_auto_improvements' in config_updates:
                self.config.enable_auto_improvements = config_updates['enable_auto_improvements']
            
            if 'min_confidence_threshold' in config_updates:
                self.config.min_confidence_threshold = config_updates['min_confidence_threshold']
            
            if 'max_error_rate_threshold' in config_updates:
                self.config.max_error_rate_threshold = config_updates['max_error_rate_threshold']
            
            if 'max_processing_time_ms' in config_updates:
                self.config.max_processing_time_ms = config_updates['max_processing_time_ms']
            
            # Log configuration change
            self._log_config_change(config_updates)
            
            return {
                'status': 'success',
                'updated_config': {
                    'analysis_interval_hours': self.config.analysis_interval_hours,
                    'enable_auto_improvements': self.config.enable_auto_improvements,
                    'min_confidence_threshold': self.config.min_confidence_threshold,
                    'max_error_rate_threshold': self.config.max_error_rate_threshold,
                    'max_processing_time_ms': self.config.max_processing_time_ms
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error updating feedback loop config: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    # Private helper methods
    
    def _trigger_immediate_analysis(self, response_id: str, response_data: Dict[str, Any]):
        """Trigger immediate analysis for critical responses."""
        try:
            self.lambda_client.invoke(
                FunctionName='ImmediateQualityAnalysis',
                InvocationType='Event',
                Payload=json.dumps({
                    'response_id': response_id,
                    'response_data': response_data,
                    'trigger_reason': 'critical_response',
                    'timestamp': datetime.now().isoformat()
                })
            )
            logger.info(f"Triggered immediate analysis for response: {response_id}")
            
        except Exception as e:
            logger.error(f"Error triggering immediate analysis: {str(e)}")
    
    def _send_critical_notifications(self, insights: List[Any], actions: List[Dict[str, Any]]):
        """Send notifications for critical insights."""
        if not self.notification_topic:
            logger.warning("No notification topic configured")
            return
        
        try:
            message = {
                'alert_type': 'critical_quality_issues',
                'timestamp': datetime.now().isoformat(),
                'critical_insights': [
                    {
                        'insight_id': insight.insight_id,
                        'insight_type': insight.insight_type,
                        'severity': insight.severity,
                        'description': insight.description,
                        'affected_data_type': insight.affected_data_type
                    }
                    for insight in insights
                ],
                'recommended_actions': [
                    {
                        'action_id': action.get('action_id'),
                        'action_type': action.get('action_type'),
                        'severity': action.get('severity'),
                        'description': action.get('description')
                    }
                    for action in actions
                ],
                'requires_immediate_attention': True
            }
            
            self.sns_client.publish(
                TopicArn=self.notification_topic,
                Subject=f"Critical Quality Issues Detected - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                Message=json.dumps(message, default=str),
                MessageAttributes={
                    'alert_type': {
                        'DataType': 'String',
                        'StringValue': 'critical_quality_issues'
                    }
                }
            )
            
            logger.info(f"Sent critical notifications for {len(insights)} insights")
            
        except Exception as e:
            logger.error(f"Error sending critical notifications: {str(e)}")
    
    def _send_error_notification(self, error_message: str):
        """Send error notification."""
        if not self.notification_topic:
            return
        
        try:
            message = {
                'alert_type': 'feedback_loop_error',
                'timestamp': datetime.now().isoformat(),
                'error_message': error_message,
                'requires_immediate_attention': True
            }
            
            self.sns_client.publish(
                TopicArn=self.notification_topic,
                Subject=f"Feedback Loop Error - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                Message=json.dumps(message, default=str)
            )
            
        except Exception as e:
            logger.error(f"Error sending error notification: {str(e)}")
    
    def _log_cycle_metrics(self, insights: List[Any], actions: List[Dict[str, Any]], 
                         execution_results: List[Dict[str, Any]], cycle_duration: float):
        """Log feedback loop cycle metrics to CloudWatch."""
        try:
            metrics = [
                {
                    'MetricName': 'FeedbackLoopCycleDuration',
                    'Value': cycle_duration,
                    'Unit': 'Milliseconds'
                },
                {
                    'MetricName': 'InsightsGenerated',
                    'Value': len(insights),
                    'Unit': 'Count'
                },
                {
                    'MetricName': 'ActionsGenerated',
                    'Value': len(actions),
                    'Unit': 'Count'
                },
                {
                    'MetricName': 'ActionsExecuted',
                    'Value': len(execution_results),
                    'Unit': 'Count'
                }
            ]
            
            # Count insights by severity
            severity_counts = {}
            for insight in insights:
                severity = insight.severity
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            for severity, count in severity_counts.items():
                metrics.append({
                    'MetricName': 'InsightsBySeverity',
                    'Value': count,
                    'Unit': 'Count',
                    'Dimensions': [
                        {'Name': 'Severity', 'Value': severity}
                    ]
                })
            
            self.cloudwatch_client.put_metric_data(
                Namespace='CustomerFeedback/FeedbackLoop',
                MetricData=metrics
            )
            
        except Exception as e:
            logger.error(f"Error logging cycle metrics: {str(e)}")
    
    def _log_config_change(self, config_updates: Dict[str, Any]):
        """Log configuration changes to CloudWatch."""
        try:
            self.cloudwatch_client.put_metric_data(
                Namespace='CustomerFeedback/FeedbackLoop',
                MetricData=[
                    {
                        'MetricName': 'ConfigurationChange',
                        'Value': 1,
                        'Unit': 'Count',
                        'Dimensions': [
                            {'Name': 'ChangeType', 'Value': 'config_update'}
                        ]
                    }
                ]
            )
            
            logger.info(f"Configuration updated: {config_updates}")
            
        except Exception as e:
            logger.error(f"Error logging config change: {str(e)}")
    
    def _calculate_system_health(self, trends: Dict[str, Any], insights: List[Any]) -> float:
        """Calculate overall system health score."""
        try:
            health_score = 100.0  # Start with perfect health
            
            # Deduct points for critical insights
            critical_count = len([i for i in insights if i.severity == 'critical'])
            high_count = len([i for i in insights if i.severity == 'high'])
            
            health_score -= (critical_count * 20)  # 20 points per critical issue
            health_score -= (high_count * 10)       # 10 points per high issue
            
            # Deduct points for poor metrics
            metrics = trends.get('metrics', {})
            
            confidence_trend = metrics.get('confidence_trend', {})
            if confidence_trend.get('trend_direction') == 'declining':
                health_score -= 15
            
            error_trend = metrics.get('error_rate_trend', {})
            if error_trend.get('trend_direction') == 'declining':
                health_score -= 15
            
            performance_trend = metrics.get('performance_trend', {})
            if performance_trend.get('trend_direction') == 'declining':
                health_score -= 10
            
            # Ensure score doesn't go below 0
            return max(0.0, health_score)
            
        except Exception as e:
            logger.error(f"Error calculating system health: {str(e)}")
            return 50.0  # Return neutral score on error
    
    def _get_health_status(self, health_score: float) -> str:
        """Get health status based on score."""
        if health_score >= 90:
            return 'excellent'
        elif health_score >= 75:
            return 'good'
        elif health_score >= 60:
            return 'fair'
        elif health_score >= 40:
            return 'poor'
        else:
            return 'critical'

# Lambda handler function
def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for feedback loop orchestrator.
    
    Expected event structure:
    {
        "action": "run_cycle" | "handle_response" | "get_status" | "update_config",
        "data": { ... }
    }
    """
    try:
        logger.info(f"Received event: {json.dumps(event, default=str)}")
        
        # Initialize orchestrator
        orchestrator = FeedbackLoopOrchestrator()
        
        action = event.get('action', 'run_cycle')
        
        if action == 'run_cycle':
            return orchestrator.run_feedback_loop_cycle(event.get('data', {}))
        
        elif action == 'handle_response':
            return orchestrator.handle_model_response(event.get('data', {}))
        
        elif action == 'get_status':
            return orchestrator.get_feedback_loop_status()
        
        elif action == 'update_config':
            return orchestrator.update_feedback_loop_config(event.get('data', {}))
        
        else:
            return {
                'status': 'error',
                'error': f'Unknown action: {action}',
                'timestamp': datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Lambda handler error: {str(e)}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

# For local testing
if __name__ == "__main__":
    # Test the orchestrator
    orchestrator = FeedbackLoopOrchestrator()
    
    # Test running a cycle
    test_event = {"action": "run_cycle", "data": {}}
    result = lambda_handler(test_event, None)
    print(f"Cycle result: {json.dumps(result, default=str, indent=2)}")
    
    # Test getting status
    status_event = {"action": "get_status", "data": {}}
    status = lambda_handler(status_event, None)
    print(f"Status: {json.dumps(status, default=str, indent=2)}")