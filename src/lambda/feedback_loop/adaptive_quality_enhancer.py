#!/usr/bin/env python3
"""
Adaptive Quality Enhancer

This module implements adaptive quality enhancement by learning from
model responses and continuously improving data processing pipelines.

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
import statistics
from collections import defaultdict, Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QualityPattern:
    """Represents a quality pattern identified from model responses."""
    pattern_id: str
    pattern_type: str  # 'confidence_decline', 'error_spike', 'performance_degradation'
    data_type: str
    severity: str
    confidence: float
    description: str
    detected_at: datetime
    indicators: Dict[str, Any]
    recommendations: List[str]

@dataclass
class EnhancementAction:
    """Represents an adaptive enhancement action."""
    action_id: str
    action_type: str  # 'preprocessing_adjustment', 'parameter_tuning', 'pipeline_optimization'
    target_component: str
    priority: int
    description: str
    parameters: Dict[str, Any]
    expected_improvement: float
    created_at: datetime

class AdaptiveQualityEnhancer:
    """
    Implements adaptive quality enhancement through machine learning and pattern recognition.
    
    This class provides:
    1. Pattern detection from model response data
    2. Adaptive parameter tuning based on performance
    3. Dynamic pipeline optimization
    4. Continuous learning from feedback
    """
    
    def __init__(self, region_name: str = 'us-east-1'):
        """Initialize Adaptive Quality Enhancer."""
        self.region_name = region_name
        self.dynamodb_client = boto3.resource('dynamodb', region_name=region_name)
        self.s3_client = boto3.client('s3', region_name=region_name)
        self.lambda_client = boto3.client('lambda', region_name=region_name)
        self.cloudwatch_client = boto3.client('cloudwatch', region_name=region_name)
        
        # Initialize DynamoDB tables
        self.patterns_table = self.dynamodb_client.Table('quality_patterns')
        self.enhancements_table = self.dynamodb_client.Table('enhancement_actions')
        self.learning_history_table = self.dynamodb_client.Table('learning_history')
        
        # S3 buckets
        self.patterns_bucket = 'customer-feedback-analysis-lm-patterns'
        self.enhancements_bucket = 'customer-feedback-analysis-lm-enhancements'
        
        # Enhancement thresholds
        self.min_confidence_threshold = 0.6
        self.max_error_rate_threshold = 0.1
        self.max_processing_time_ms = 10000
        
        logger.info("AdaptiveQualityEnhancer initialized")
    
    def detect_quality_patterns(self, time_window_hours: int = 24) -> List[QualityPattern]:
        """
        Detect quality patterns from recent model responses.
        
        Args:
            time_window_hours: Time window for pattern detection
            
        Returns:
            List of detected quality patterns
        """
        try:
            logger.info(f"Detecting quality patterns in last {time_window_hours} hours")
            
            # Get recent response data
            response_data = self._get_recent_response_data(time_window_hours)
            
            patterns = []
            
            # Detect confidence decline patterns
            confidence_patterns = self._detect_confidence_decline_patterns(response_data)
            patterns.extend(confidence_patterns)
            
            # Detect error spike patterns
            error_patterns = self._detect_error_spike_patterns(response_data)
            patterns.extend(error_patterns)
            
            # Detect performance degradation patterns
            performance_patterns = self._detect_performance_degradation_patterns(response_data)
            patterns.extend(performance_patterns)
            
            # Detect data quality patterns
            quality_patterns = self._detect_data_quality_patterns(response_data)
            patterns.extend(quality_patterns)
            
            # Store detected patterns
            for pattern in patterns:
                self._store_quality_pattern(pattern)
            
            logger.info(f"Detected {len(patterns)} quality patterns")
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting quality patterns: {str(e)}")
            return []
    
    def generate_adaptive_enhancements(self, patterns: List[QualityPattern]) -> List[EnhancementAction]:
        """
        Generate adaptive enhancement actions based on detected patterns.
        
        Args:
            patterns: List of quality patterns
            
        Returns:
            List of enhancement actions
        """
        try:
            logger.info(f"Generating adaptive enhancements for {len(patterns)} patterns")
            
            enhancements = []
            
            # Group patterns by data type and severity
            pattern_groups = self._group_patterns_by_context(patterns)
            
            for context, context_patterns in pattern_groups.items():
                # Generate enhancements for each pattern group
                context_enhancements = self._generate_context_enhancements(context, context_patterns)
                enhancements.extend(context_enhancements)
            
            # Prioritize enhancements by impact and feasibility
            enhancements = self._prioritize_enhancements(enhancements)
            
            # Store enhancement actions
            for enhancement in enhancements:
                self._store_enhancement_action(enhancement)
            
            logger.info(f"Generated {len(enhancements)} adaptive enhancements")
            return enhancements
            
        except Exception as e:
            logger.error(f"Error generating adaptive enhancements: {str(e)}")
            return []
    
    def execute_adaptive_enhancements(self, enhancements: List[EnhancementAction]) -> List[Dict[str, Any]]:
        """
        Execute adaptive enhancement actions.
        
        Args:
            enhancements: List of enhancement actions
            
        Returns:
            Execution results
        """
        try:
            logger.info(f"Executing {len(enhancements)} adaptive enhancements")
            
            results = []
            
            for enhancement in enhancements:
                result = self._execute_single_enhancement(enhancement)
                results.append(result)
                
                # Track learning from this enhancement
                self._track_enhancement_learning(enhancement, result)
            
            # Update enhancement effectiveness metrics
            self._update_enhancement_metrics(results)
            
            logger.info(f"Completed execution of adaptive enhancements")
            return results
            
        except Exception as e:
            logger.error(f"Error executing adaptive enhancements: {str(e)}")
            return []
    
    def learn_from_enhancement_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Learn from enhancement results to improve future recommendations.
        
        Args:
            results: Enhancement execution results
            
        Returns:
            Learning insights and updated parameters
        """
        try:
            logger.info("Learning from enhancement results")
            
            learning_insights = {
                'successful_enhancements': [],
                'failed_enhancements': [],
                'parameter_adjustments': {},
                'pattern_detection_improvements': {},
                'overall_effectiveness': 0.0
            }
            
            # Analyze successful enhancements
            successful_results = [r for r in results if r.get('status') == 'success']
            failed_results = [r for r in results if r.get('status') == 'failed']
            
            learning_insights['successful_enhancements'] = [
                {
                    'enhancement_id': r.get('enhancement_id'),
                    'action_type': r.get('action_type'),
                    'improvement_magnitude': r.get('improvement_magnitude', 0)
                }
                for r in successful_results
            ]
            
            learning_insights['failed_enhancements'] = [
                {
                    'enhancement_id': r.get('enhancement_id'),
                    'action_type': r.get('action_type'),
                    'error': r.get('error')
                }
                for r in failed_results
            ]
            
            # Learn parameter adjustments
            parameter_adjustments = self._learn_parameter_adjustments(successful_results)
            learning_insights['parameter_adjustments'] = parameter_adjustments
            
            # Learn pattern detection improvements
            pattern_improvements = self._learn_pattern_detection_improvements(successful_results)
            learning_insights['pattern_detection_improvements'] = pattern_improvements
            
            # Calculate overall effectiveness
            if results:
                success_rate = len(successful_results) / len(results)
                avg_improvement = statistics.mean([
                    r.get('improvement_magnitude', 0) for r in successful_results
                ]) if successful_results else 0
                learning_insights['overall_effectiveness'] = success_rate * avg_improvement
            
            # Store learning insights
            self._store_learning_insights(learning_insights)
            
            # Update adaptive thresholds based on learning
            self._update_adaptive_thresholds(learning_insights)
            
            logger.info(f"Learning completed. Overall effectiveness: {learning_insights['overall_effectiveness']:.2f}")
            return learning_insights
            
        except Exception as e:
            logger.error(f"Error learning from enhancement results: {str(e)}")
            return {}
    
    def get_adaptive_recommendations(self, data_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current adaptive recommendations based on learned patterns.
        
        Args:
            data_type: Specific data type for recommendations
            
        Returns:
            Adaptive recommendations
        """
        try:
            # Get recent patterns
            recent_patterns = self._get_recent_patterns(data_type)
            
            # Get successful enhancement history
            enhancement_history = self._get_enhancement_history(data_type)
            
            # Generate recommendations
            recommendations = {
                'timestamp': datetime.now().isoformat(),
                'data_type': data_type or 'all',
                'current_patterns': [asdict(p) for p in recent_patterns],
                'recommended_enhancements': [],
                'parameter_adjustments': {},
                'confidence_scores': {},
                'expected_improvements': {}
            }
            
            # Analyze patterns and generate specific recommendations
            for pattern in recent_patterns:
                pattern_recommendations = self._generate_pattern_recommendations(pattern)
                recommendations['recommended_enhancements'].extend(pattern_recommendations)
            
            # Generate parameter adjustment recommendations
            parameter_recs = self._generate_parameter_recommendations(enhancement_history)
            recommendations['parameter_adjustments'] = parameter_recs
            
            # Calculate confidence scores for recommendations
            confidence_scores = self._calculate_recommendation_confidence(
                recent_patterns, enhancement_history
            )
            recommendations['confidence_scores'] = confidence_scores
            
            # Estimate expected improvements
            expected_improvements = self._estimate_expected_improvements(
                recommendations['recommended_enhancements']
            )
            recommendations['expected_improvements'] = expected_improvements
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting adaptive recommendations: {str(e)}")
            return {}
    
    # Private helper methods
    
    def _get_recent_response_data(self, time_window_hours: int) -> List[Dict[str, Any]]:
        """Get recent model response data for pattern detection."""
        try:
            # This would typically query DynamoDB or S3 for recent responses
            # For now, return mock data structure
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            
            # In real implementation, this would query the model_responses table
            # For demonstration, we'll return the expected structure
            return []
            
        except Exception as e:
            logger.error(f"Error getting recent response data: {str(e)}")
            return []
    
    def _detect_confidence_decline_patterns(self, response_data: List[Dict[str, Any]]) -> List[QualityPattern]:
        """Detect patterns of declining confidence scores."""
        patterns = []
        
        # Group by data type and analyze confidence trends
        by_data_type = defaultdict(list)
        for response in response_data:
            data_type = response.get('input_data_type', 'unknown')
            confidence = response.get('confidence_score')
            if confidence is not None:
                by_data_type[data_type].append({
                    'timestamp': response.get('response_timestamp'),
                    'confidence': confidence
                })
        
        for data_type, confidence_data in by_data_type.items():
            if len(confidence_data) < 5:  # Need sufficient data points
                continue
            
            # Sort by timestamp
            confidence_data.sort(key=lambda x: x['timestamp'])
            
            # Calculate trend
            confidences = [d['confidence'] for d in confidence_data]
            trend = self._calculate_trend(confidences)
            
            # Detect significant decline
            if trend < -0.1:  # Declining trend
                avg_confidence = statistics.mean(confidences)
                min_confidence = min(confidences)
                
                pattern = QualityPattern(
                    pattern_id=f"confidence_decline_{data_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    pattern_type='confidence_decline',
                    data_type=data_type,
                    severity='high' if avg_confidence < 0.5 else 'medium',
                    confidence=abs(trend),
                    description=f"Confidence scores declining for {data_type} data (trend: {trend:.3f})",
                    detected_at=datetime.now(),
                    indicators={
                        'trend': trend,
                        'avg_confidence': avg_confidence,
                        'min_confidence': min_confidence,
                        'data_points': len(confidences)
                    },
                    recommendations=[
                        'Review data preprocessing pipeline',
                        'Enhance data quality validation',
                        'Adjust model parameters',
                        'Consider additional training data'
                    ]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_error_spike_patterns(self, response_data: List[Dict[str, Any]]) -> List[QualityPattern]:
        """Detect patterns of error rate spikes."""
        patterns = []
        
        # Group by data type and analyze error rates over time
        by_data_type = defaultdict(list)
        for response in response_data:
            data_type = response.get('input_data_type', 'unknown')
            timestamp = response.get('response_timestamp')
            error_occurred = response.get('error_occurred', False)
            
            # Group by hour for trend analysis
            hour_key = timestamp[:13] if timestamp else datetime.now().strftime('%Y-%m-%dT%H')
            by_data_type[data_type].append({
                'hour': hour_key,
                'error': error_occurred
            })
        
        for data_type, error_data in by_data_type.items():
            # Calculate error rates by hour
            hourly_errors = defaultdict(lambda: {'total': 0, 'errors': 0})
            for item in error_data:
                hourly_errors[item['hour']]['total'] += 1
                if item['error']:
                    hourly_errors[item['hour']]['errors'] += 1
            
            # Calculate error rates
            error_rates = []
            for hour, counts in hourly_errors.items():
                if counts['total'] > 0:
                    error_rate = counts['errors'] / counts['total']
                    error_rates.append(error_rate)
            
            if len(error_rates) < 3:  # Need sufficient data points
                continue
            
            # Detect spikes
            avg_error_rate = statistics.mean(error_rates)
            max_error_rate = max(error_rates)
            
            if max_error_rate > self.max_error_rate_threshold * 2:  # Significant spike
                pattern = QualityPattern(
                    pattern_id=f"error_spike_{data_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    pattern_type='error_spike',
                    data_type=data_type,
                    severity='critical' if max_error_rate > 0.3 else 'high',
                    confidence=min(1.0, max_error_rate / self.max_error_rate_threshold),
                    description=f"Error rate spike detected for {data_type} data (max: {max_error_rate:.1%})",
                    detected_at=datetime.now(),
                    indicators={
                        'avg_error_rate': avg_error_rate,
                        'max_error_rate': max_error_rate,
                        'threshold': self.max_error_rate_threshold,
                        'data_points': len(error_rates)
                    },
                    recommendations=[
                        'Investigate data format issues',
                        'Check input validation rules',
                        'Review error handling logic',
                        'Implement better data sanitization'
                    ]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_performance_degradation_patterns(self, response_data: List[Dict[str, Any]]) -> List[QualityPattern]:
        """Detect patterns of performance degradation."""
        patterns = []
        
        # Group by data type and analyze processing times
        by_data_type = defaultdict(list)
        for response in response_data:
            data_type = response.get('input_data_type', 'unknown')
            processing_time = response.get('processing_time_ms')
            if processing_time is not None:
                by_data_type[data_type].append(processing_time)
        
        for data_type, times in by_data_type.items():
            if len(times) < 5:  # Need sufficient data points
                continue
            
            # Calculate performance metrics
            avg_time = statistics.mean(times)
            max_time = max(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0
            
            # Detect performance degradation
            if avg_time > self.max_processing_time_ms:
                severity = 'critical' if avg_time > self.max_processing_time_ms * 2 else 'high'
                
                pattern = QualityPattern(
                    pattern_id=f"performance_degradation_{data_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    pattern_type='performance_degradation',
                    data_type=data_type,
                    severity=severity,
                    confidence=min(1.0, avg_time / self.max_processing_time_ms),
                    description=f"Performance degradation detected for {data_type} data (avg: {avg_time/1000:.1f}s)",
                    detected_at=datetime.now(),
                    indicators={
                        'avg_processing_time': avg_time,
                        'max_processing_time': max_time,
                        'std_processing_time': std_time,
                        'threshold': self.max_processing_time_ms,
                        'data_points': len(times)
                    },
                    recommendations=[
                        'Optimize processing algorithms',
                        'Increase Lambda memory allocation',
                        'Implement caching mechanisms',
                        'Consider parallel processing'
                    ]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_data_quality_patterns(self, response_data: List[Dict[str, Any]]) -> List[QualityPattern]:
        """Detect patterns related to data quality issues."""
        patterns = []
        
        # Analyze response quality metrics
        quality_scores = []
        for response in response_data:
            quality_metrics = response.get('quality_metrics', {})
            if quality_metrics:
                completeness = quality_metrics.get('completeness', 0)
                structure_score = quality_metrics.get('structure_score', 0)
                overall_quality = (completeness + structure_score) / 2
                quality_scores.append(overall_quality)
        
        if len(quality_scores) < 10:  # Need sufficient data points
            return patterns
        
        # Calculate quality trends
        avg_quality = statistics.mean(quality_scores)
        min_quality = min(quality_scores)
        
        # Detect quality issues
        if avg_quality < 0.6 or min_quality < 0.3:
            pattern = QualityPattern(
                pattern_id=f"data_quality_issue_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                pattern_type='data_quality_issue',
                data_type='all',
                severity='high' if avg_quality < 0.4 else 'medium',
                confidence=1.0 - avg_quality,
                description=f"Data quality issues detected (avg quality: {avg_quality:.2f})",
                detected_at=datetime.now(),
                indicators={
                    'avg_quality_score': avg_quality,
                    'min_quality_score': min_quality,
                    'data_points': len(quality_scores)
                },
                recommendations=[
                    'Enhance data validation rules',
                    'Improve data preprocessing',
                    'Add quality checks at ingestion',
                    'Implement data cleansing pipelines'
                ]
            )
            patterns.append(pattern)
        
        return patterns
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend for a series of values."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear regression
        n = len(values)
        x = list(range(n))
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        # Calculate slope (trend)
        denominator = n * sum_x2 - sum_x ** 2
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
    
    def _group_patterns_by_context(self, patterns: List[QualityPattern]) -> Dict[Tuple[str, str], List[QualityPattern]]:
        """Group patterns by data type and severity for contextual analysis."""
        groups = defaultdict(list)
        
        for pattern in patterns:
            context = (pattern.data_type, pattern.severity)
            groups[context].append(pattern)
        
        return dict(groups)
    
    def _generate_context_enhancements(self, context: Tuple[str, str], patterns: List[QualityPattern]) -> List[EnhancementAction]:
        """Generate enhancements for a specific context."""
        data_type, severity = context
        enhancements = []
        
        # Analyze patterns to determine best enhancement strategy
        pattern_types = [p.pattern_type for p in patterns]
        
        # Generate enhancements based on pattern types
        if 'confidence_decline' in pattern_types:
            enhancement = self._create_confidence_enhancement(data_type, severity, patterns)
            if enhancement:
                enhancements.append(enhancement)
        
        if 'error_spike' in pattern_types:
            enhancement = self._create_error_reduction_enhancement(data_type, severity, patterns)
            if enhancement:
                enhancements.append(enhancement)
        
        if 'performance_degradation' in pattern_types:
            enhancement = self._create_performance_enhancement(data_type, severity, patterns)
            if enhancement:
                enhancements.append(enhancement)
        
        if 'data_quality_issue' in pattern_types:
            enhancement = self._create_quality_enhancement(data_type, severity, patterns)
            if enhancement:
                enhancements.append(enhancement)
        
        return enhancements
    
    def _create_confidence_enhancement(self, data_type: str, severity: str, patterns: List[QualityPattern]) -> Optional[EnhancementAction]:
        """Create enhancement for confidence-related issues."""
        confidence_patterns = [p for p in patterns if p.pattern_type == 'confidence_decline']
        
        if not confidence_patterns:
            return None
        
        # Calculate average confidence decline
        avg_decline = statistics.mean([p.indicators.get('trend', 0) for p in confidence_patterns])
        
        enhancement = EnhancementAction(
            action_id=f"confidence_enhancement_{data_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            action_type='preprocessing_adjustment',
            target_component=f'{data_type}_processing',
            priority=3 if severity == 'critical' else 2,
            description=f"Enhance preprocessing to improve confidence scores for {data_type} data",
            parameters={
                'enhance_data_cleaning': True,
                'improve_format_validation': True,
                'adjust_preprocessing_steps': True,
                'confidence_threshold_adjustment': abs(avg_decline) * 0.5
            },
            expected_improvement=min(0.3, abs(avg_decline)),
            created_at=datetime.now()
        )
        
        return enhancement
    
    def _create_error_reduction_enhancement(self, data_type: str, severity: str, patterns: List[QualityPattern]) -> Optional[EnhancementAction]:
        """Create enhancement for error reduction."""
        error_patterns = [p for p in patterns if p.pattern_type == 'error_spike']
        
        if not error_patterns:
            return None
        
        max_error_rate = max([p.indicators.get('max_error_rate', 0) for p in error_patterns])
        
        enhancement = EnhancementAction(
            action_id=f"error_reduction_{data_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            action_type='pipeline_optimization',
            target_component=f'{data_type}_pipeline',
            priority=4 if severity == 'critical' else 3,
            description=f"Optimize pipeline to reduce error rates for {data_type} data",
            parameters={
                'enhance_input_validation': True,
                'improve_error_handling': True,
                'add_data_sanitization': True,
                'implement_retry_logic': True,
                'error_threshold_adjustment': max_error_rate * 0.8
            },
            expected_improvement=min(0.5, max_error_rate),
            created_at=datetime.now()
        )
        
        return enhancement
    
    def _create_performance_enhancement(self, data_type: str, severity: str, patterns: List[QualityPattern]) -> Optional[EnhancementAction]:
        """Create enhancement for performance improvement."""
        performance_patterns = [p for p in patterns if p.pattern_type == 'performance_degradation']
        
        if not performance_patterns:
            return None
        
        avg_processing_time = statistics.mean([p.indicators.get('avg_processing_time', 0) for p in performance_patterns])
        
        enhancement = EnhancementAction(
            action_id=f"performance_enhancement_{data_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            action_type='parameter_tuning',
            target_component=f'{data_type}_lambda',
            priority=2 if severity == 'critical' else 1,
            description=f"Tune parameters to improve performance for {data_type} data",
            parameters={
                'increase_memory_allocation': True,
                'optimize_timeout_settings': True,
                'enable_batch_processing': True,
                'implement_caching': True,
                'memory_multiplier': min(2.0, avg_processing_time / self.max_processing_time_ms)
            },
            expected_improvement=min(0.4, (avg_processing_time - self.max_processing_time_ms) / avg_processing_time),
            created_at=datetime.now()
        )
        
        return enhancement
    
    def _create_quality_enhancement(self, data_type: str, severity: str, patterns: List[QualityPattern]) -> Optional[EnhancementAction]:
        """Create enhancement for data quality improvement."""
        quality_patterns = [p for p in patterns if p.pattern_type == 'data_quality_issue']
        
        if not quality_patterns:
            return None
        
        avg_quality = statistics.mean([p.indicators.get('avg_quality_score', 0) for p in quality_patterns])
        
        enhancement = EnhancementAction(
            action_id=f"quality_enhancement_{data_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            action_type='preprocessing_adjustment',
            target_component=f'{data_type}_validation',
            priority=3 if severity == 'critical' else 2,
            description=f"Enhance data quality validation for {data_type} data",
            parameters={
                'strengthen_validation_rules': True,
                'add_quality_checks': True,
                'implement_data_cleaning': True,
                'enhance_preprocessing': True,
                'quality_threshold_adjustment': (1.0 - avg_quality) * 0.7
            },
            expected_improvement=min(0.5, 1.0 - avg_quality),
            created_at=datetime.now()
        )
        
        return enhancement
    
    def _prioritize_enhancements(self, enhancements: List[EnhancementAction]) -> List[EnhancementAction]:
        """Prioritize enhancements by impact and feasibility."""
        # Sort by priority (higher first) and expected improvement
        return sorted(
            enhancements,
            key=lambda x: (x.priority, x.expected_improvement),
            reverse=True
        )
    
    def _store_quality_pattern(self, pattern: QualityPattern):
        """Store quality pattern in DynamoDB."""
        item = asdict(pattern)
        item['detected_at'] = pattern.detected_at.isoformat()
        item['confidence'] = Decimal(str(pattern.confidence))
        
        self.patterns_table.put_item(Item=item)
        
        # Also store in S3 for analysis
        key = f"patterns/{pattern.pattern_type}/{pattern.pattern_id}.json"
        self.s3_client.put_object(
            Bucket=self.patterns_bucket,
            Key=key,
            Body=json.dumps(item, default=str),
            ContentType='application/json'
        )
    
    def _store_enhancement_action(self, enhancement: EnhancementAction):
        """Store enhancement action in DynamoDB."""
        item = asdict(enhancement)
        item['created_at'] = enhancement.created_at.isoformat()
        item['expected_improvement'] = Decimal(str(enhancement.expected_improvement))
        
        self.enhancements_table.put_item(Item=item)
        
        # Also store in S3 for tracking
        key = f"enhancements/{enhancement.action_type}/{enhancement.action_id}.json"
        self.s3_client.put_object(
            Bucket=self.enhancements_bucket,
            Key=key,
            Body=json.dumps(item, default=str),
            ContentType='application/json'
        )
    
    def _execute_single_enhancement(self, enhancement: EnhancementAction) -> Dict[str, Any]:
        """Execute a single enhancement action."""
        try:
            action_type = enhancement.action_type
            target_component = enhancement.target_component
            
            # Execute based on action type
            if action_type == 'preprocessing_adjustment':
                return self._execute_preprocessing_adjustment(enhancement)
            elif action_type == 'parameter_tuning':
                return self._execute_parameter_tuning(enhancement)
            elif action_type == 'pipeline_optimization':
                return self._execute_pipeline_optimization(enhancement)
            else:
                return {
                    'enhancement_id': enhancement.action_id,
                    'status': 'failed',
                    'error': f'Unknown action type: {action_type}'
                }
                
        except Exception as e:
            return {
                'enhancement_id': enhancement.action_id,
                'status': 'failed',
                'error': str(e)
            }
    
    def _execute_preprocessing_adjustment(self, enhancement: EnhancementAction) -> Dict[str, Any]:
        """Execute preprocessing adjustment enhancement."""
        try:
            # Invoke preprocessing adjustment Lambda
            response = self.lambda_client.invoke(
                FunctionName='PreprocessingAdjustment',
                Payload=json.dumps({
                    'target_component': enhancement.target_component,
                    'parameters': enhancement.parameters,
                    'enhancement_id': enhancement.action_id
                })
            )
            
            return {
                'enhancement_id': enhancement.action_id,
                'status': 'success',
                'action_type': 'preprocessing_adjustment',
                'improvement_magnitude': enhancement.expected_improvement,
                'lambda_response': response.get('Payload', {}).read().decode()
            }
            
        except Exception as e:
            return {
                'enhancement_id': enhancement.action_id,
                'status': 'failed',
                'error': str(e)
            }
    
    def _execute_parameter_tuning(self, enhancement: EnhancementAction) -> Dict[str, Any]:
        """Execute parameter tuning enhancement."""
        try:
            # Invoke parameter tuning Lambda
            response = self.lambda_client.invoke(
                FunctionName='ParameterTuning',
                Payload=json.dumps({
                    'target_component': enhancement.target_component,
                    'parameters': enhancement.parameters,
                    'enhancement_id': enhancement.action_id
                })
            )
            
            return {
                'enhancement_id': enhancement.action_id,
                'status': 'success',
                'action_type': 'parameter_tuning',
                'improvement_magnitude': enhancement.expected_improvement,
                'lambda_response': response.get('Payload', {}).read().decode()
            }
            
        except Exception as e:
            return {
                'enhancement_id': enhancement.action_id,
                'status': 'failed',
                'error': str(e)
            }
    
    def _execute_pipeline_optimization(self, enhancement: EnhancementAction) -> Dict[str, Any]:
        """Execute pipeline optimization enhancement."""
        try:
            # Invoke pipeline optimization Lambda
            response = self.lambda_client.invoke(
                FunctionName='PipelineOptimization',
                Payload=json.dumps({
                    'target_component': enhancement.target_component,
                    'parameters': enhancement.parameters,
                    'enhancement_id': enhancement.action_id
                })
            )
            
            return {
                'enhancement_id': enhancement.action_id,
                'status': 'success',
                'action_type': 'pipeline_optimization',
                'improvement_magnitude': enhancement.expected_improvement,
                'lambda_response': response.get('Payload', {}).read().decode()
            }
            
        except Exception as e:
            return {
                'enhancement_id': enhancement.action_id,
                'status': 'failed',
                'error': str(e)
            }
    
    def _track_enhancement_learning(self, enhancement: EnhancementAction, result: Dict[str, Any]):
        """Track learning from enhancement execution."""
        try:
            learning_record = {
                'enhancement_id': enhancement.action_id,
                'action_type': enhancement.action_type,
                'target_component': enhancement.target_component,
                'parameters': enhancement.parameters,
                'expected_improvement': enhancement.expected_improvement,
                'actual_result': result,
                'timestamp': datetime.now().isoformat()
            }
            
            self.learning_history_table.put_item(Item=learning_record)
            
        except Exception as e:
            logger.error(f"Error tracking enhancement learning: {str(e)}")
    
    def _update_enhancement_metrics(self, results: List[Dict[str, Any]]):
        """Update enhancement effectiveness metrics in CloudWatch."""
        try:
            successful_count = len([r for r in results if r.get('status') == 'success'])
            total_count = len(results)
            
            if total_count > 0:
                success_rate = successful_count / total_count
                avg_improvement = statistics.mean([
                    r.get('improvement_magnitude', 0) for r in results 
                    if r.get('status') == 'success'
                ]) if successful_count > 0 else 0
                
                metrics = [
                    {
                        'MetricName': 'EnhancementSuccessRate',
                        'Value': success_rate,
                        'Unit': 'Percent'
                    },
                    {
                        'MetricName': 'AverageEnhancementImprovement',
                        'Value': avg_improvement,
                        'Unit': 'None'
                    }
                ]
                
                self.cloudwatch_client.put_metric_data(
                    Namespace='CustomerFeedback/AdaptiveEnhancement',
                    MetricData=metrics
                )
                
        except Exception as e:
            logger.error(f"Error updating enhancement metrics: {str(e)}")
    
    def _store_learning_insights(self, insights: Dict[str, Any]):
        """Store learning insights for future reference."""
        try:
            # Store in S3 for analysis
            key = f"learning/insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.s3_client.put_object(
                Bucket=self.enhancements_bucket,
                Key=key,
                Body=json.dumps(insights, default=str),
                ContentType='application/json'
            )
                
        except Exception as e:
            logger.error(f"Error storing learning insights: {str(e)}")
    
    def _update_adaptive_thresholds(self, insights: Dict[str, Any]):
        """Update adaptive thresholds based on learning insights."""
        try:
            # Adjust thresholds based on learning
            overall_effectiveness = insights.get('overall_effectiveness', 0)
            
            if overall_effectiveness > 0.7:  # High effectiveness - can be more aggressive
                self.min_confidence_threshold *= 0.95  # Lower threshold slightly
                self.max_error_rate_threshold *= 1.05  # Allow slightly more errors
            elif overall_effectiveness < 0.3:  # Low effectiveness - be more conservative
                self.min_confidence_threshold *= 1.05  # Raise threshold
                self.max_error_rate_threshold *= 0.95  # Be stricter on errors
            
            # Ensure thresholds stay within reasonable bounds
            self.min_confidence_threshold = max(0.3, min(0.9, self.min_confidence_threshold))
            self.max_error_rate_threshold = max(0.05, min(0.3, self.max_error_rate_threshold))
            
            logger.info(f"Updated adaptive thresholds: confidence={self.min_confidence_threshold:.2f}, error_rate={self.max_error_rate_threshold:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating adaptive thresholds: {str(e)}")
    
    def _learn_parameter_adjustments(self, successful_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Learn optimal parameter adjustments from successful enhancements."""
        adjustments = {}
        
        # Group by action type
        by_action_type = defaultdict(list)
        for result in successful_results:
            action_type = result.get('action_type')
            improvement = result.get('improvement_magnitude', 0)
            if action_type and improvement > 0:
                by_action_type[action_type].append(improvement)
        
        # Calculate optimal adjustments
        for action_type, improvements in by_action_type.items():
            if improvements:
                avg_improvement = statistics.mean(improvements)
                adjustments[action_type] = {
                    'avg_improvement': avg_improvement,
                    'success_count': len(improvements),
                    'recommended_multiplier': min(2.0, 1.0 + avg_improvement)
                }
        
        return adjustments
    
    def _learn_pattern_detection_improvements(self, successful_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Learn improvements for pattern detection."""
        improvements = {}
        
        # Analyze which patterns led to successful enhancements
        pattern_success_rate = defaultdict(lambda: {'success': 0, 'total': 0})
        
        for result in successful_results:
            # This would ideally link back to the original patterns
            # For now, we'll use action type as a proxy
            action_type = result.get('action_type')
            if action_type:
                pattern_success_rate[action_type]['success'] += 1
                pattern_success_rate[action_type]['total'] += 1
        
        # Calculate success rates
        for pattern_type, counts in pattern_success_rate.items():
            if counts['total'] > 0:
                success_rate = counts['success'] / counts['total']
                improvements[pattern_type] = {
                    'success_rate': success_rate,
                    'total_attempts': counts['total'],
                    'confidence_boost': min(0.2, success_rate * 0.3)
                }
        
        return improvements
    
    def _get_recent_patterns(self, data_type: Optional[str] = None) -> List[QualityPattern]:
        """Get recent quality patterns."""
        try:
            # Query patterns table
            filter_expr = ''
            expr_attrs = {}
            expr_values = {}
            
            if data_type:
                filter_expr = 'data_type = :dt'
                expr_attrs['#dt'] = 'data_type'
                expr_values[':dt'] = data_type
            
            # In real implementation, this would query DynamoDB
            # For now, return empty list
            return []
            
        except Exception as e:
            logger.error(f"Error getting recent patterns: {str(e)}")
            return []
    
    def _get_enhancement_history(self, data_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get enhancement history for learning."""
        try:
            # Query enhancement history
            # In real implementation, this would query DynamoDB
            return []
            
        except Exception as e:
            logger.error(f"Error getting enhancement history: {str(e)}")
            return []
    
    def _generate_pattern_recommendations(self, pattern: QualityPattern) -> List[Dict[str, Any]]:
        """Generate specific recommendations for a pattern."""
        recommendations = []
        
        for rec in pattern.recommendations:
            recommendations.append({
                'pattern_id': pattern.pattern_id,
                'recommendation': rec,
                'confidence': pattern.confidence,
                'severity': pattern.severity,
                'data_type': pattern.data_type
            })
        
        return recommendations
    
    def _generate_parameter_recommendations(self, enhancement_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate parameter adjustment recommendations."""
        recommendations = {}
        
        # Analyze successful parameter adjustments
        successful_adjustments = [h for h in enhancement_history if h.get('status') == 'success']
        
        if successful_adjustments:
            # Group by parameter type
            param_effectiveness = defaultdict(list)
            
            for adjustment in successful_adjustments:
                parameters = adjustment.get('parameters', {})
                improvement = adjustment.get('improvement_magnitude', 0)
                
                for param_name, param_value in parameters.items():
                    if improvement > 0:
                        param_effectiveness[param_name].append({
                            'value': param_value,
                            'improvement': improvement
                        })
            
            # Calculate optimal values
            for param_name, values in param_effectiveness.items():
                if values:
                    avg_improvement = statistics.mean([v['improvement'] for v in values])
                    recommendations[param_name] = {
                        'recommended_value': values[0]['value'],  # Use most successful
                        'expected_improvement': avg_improvement,
                        'confidence': min(1.0, len(values) / 10)  # More data = higher confidence
                    }
        
        return recommendations
    
    def _calculate_recommendation_confidence(self, patterns: List[QualityPattern], 
                                         history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate confidence scores for recommendations."""
        confidence_scores = {}
        
        # Base confidence on pattern confidence and historical success
        pattern_confidence = statistics.mean([p.confidence for p in patterns]) if patterns else 0.5
        
        # Historical success rate
        if history:
            successful_count = len([h for h in history if h.get('status') == 'success'])
            historical_confidence = successful_count / len(history)
        else:
            historical_confidence = 0.5
        
        # Combine confidences
        overall_confidence = (pattern_confidence + historical_confidence) / 2
        
        confidence_scores['overall'] = overall_confidence
        confidence_scores['pattern_based'] = pattern_confidence
        confidence_scores['historical_based'] = historical_confidence
        
        return confidence_scores
    
    def _estimate_expected_improvements(self, enhancements: List[Dict[str, Any]]) -> Dict[str, float]:
        """Estimate expected improvements from enhancements."""
        improvements = {}
        
        # Group by enhancement type
        by_type = defaultdict(list)
        for enhancement in enhancements:
            enhancement_type = enhancement.get('action_type', 'unknown')
            expected_improvement = enhancement.get('expected_improvement', 0)
            by_type[enhancement_type].append(expected_improvement)
        
        # Calculate expected improvements by type
        for enhancement_type, improvement_values in by_type.items():
            if improvement_values:
                improvements[enhancement_type] = statistics.mean(improvement_values)
        
        # Overall expected improvement
        if improvements:
            improvements['overall'] = statistics.mean(improvements.values())
        
        return improvements

# Example usage and testing
if __name__ == "__main__":
    enhancer = AdaptiveQualityEnhancer()
    
    # Test pattern detection
    patterns = enhancer.detect_quality_patterns(time_window_hours=24)
    print(f"Detected {len(patterns)} patterns")
    
    # Test enhancement generation
    enhancements = enhancer.generate_adaptive_enhancements(patterns)
    print(f"Generated {len(enhancements)} enhancements")
    
    # Test getting recommendations
    recommendations = enhancer.get_adaptive_recommendations()
    print(f"Recommendations: {json.dumps(recommendations, default=str, indent=2)}")