#!/usr/bin/env python3
"""
Unified Quality Score Calculator for Phase 2 Multimodal Data Processing.

This module provides standardized quality score calculation across all components
following the 70% validation + 30% processing weighting pattern.
"""

import logging

# Configure logging
logger = logging.getLogger()

def calculate_unified_quality_score(validation_results, processing_results):
    """
    Calculate unified combined quality score from validation and processing results.
    
    This function standardizes quality score calculation across all Phase 2 components
    using consistent weighting: 70% validation + 30% processing.
    
    Args:
        validation_results (dict): Validation results from Phase 1
        processing_results (dict): Processing results from Phase 2
        
    Returns:
        float: Unified combined quality score (0-1)
    """
    
    # Extract validation score
    validation_score = validation_results.get('quality_score', 0.0)
    
    # Calculate processing quality score using component-specific logic
    processing_score = calculate_component_processing_score(processing_results)
    
    # Apply standardized weighting (70% validation, 30% processing)
    combined_score = (validation_score * 0.7) + (processing_score * 0.3)
    
    # Ensure score is within valid range
    final_score = round(min(max(combined_score, 0.0), 1.0), 3)
    
    logger.debug(f"Quality score calculation: validation={validation_score}, processing={processing_score}, combined={final_score}")
    
    return final_score

def calculate_component_processing_score(processing_results):
    """
    Calculate processing quality score based on component-specific results.
    
    This function provides component-agnostic processing score calculation
    by examining common patterns across different processing types.
    
    Args:
        processing_results (dict): Processing results from Phase 2
        
    Returns:
        float: Processing quality score (0-1)
    """
    
    processing_score = 0.0
    
    # Base score for successful processing (40% of processing score)
    if processing_results.get('processing_status') == 'COMPLETED':
        processing_score += 0.4
    elif processing_results.get('processing_status') != 'FAILED':
        # Assume successful if status not explicitly failed
        processing_score += 0.4
    
    # Text/Transcript processing bonus (20% of processing score)
    if processing_results.get('transcript') or processing_results.get('extracted_text'):
        text_length = len(processing_results.get('transcript', '') or processing_results.get('extracted_text', ''))
        if text_length > 0:
            processing_score += 0.2
    
    # Entity/Feature extraction bonus (20% of processing score)
    entities = processing_results.get('entities', [])
    labels = processing_results.get('labels', [])
    key_phrases = processing_results.get('key_phrases', [])
    
    if entities or labels or key_phrases:
        processing_score += 0.2
    
    # Additional analysis bonus (20% of processing score)
    insights = processing_results.get('insights', {})
    if insights:
        # Check for various insight types
        if any(key in insights for key in ['sentiment_analysis', 'speaker_analysis', 'dominant_labels', 'priority_analysis']):
            processing_score += 0.1
        
        # Check for quality indicators
        if any(key in insights for key in ['confidence', 'content_safe', 'transcription_quality']):
            processing_score += 0.1
    
    # Cap processing score at 1.0
    return min(processing_score, 1.0)

def validate_quality_score_components(validation_results, processing_results):
    """
    Validate that quality score components are present and valid.
    
    Args:
        validation_results (dict): Validation results from Phase 1
        processing_results (dict): Processing results from Phase 2
        
    Returns:
        dict: Validation results with any issues found
    """
    
    validation_issues = []
    processing_issues = []
    
    # Validate validation results
    if not isinstance(validation_results, dict):
        validation_issues.append("Validation results must be a dictionary")
    elif 'quality_score' not in validation_results:
        validation_issues.append("Missing quality_score in validation results")
    elif not isinstance(validation_results.get('quality_score'), (int, float)):
        validation_issues.append("quality_score must be numeric")
    elif not (0.0 <= validation_results.get('quality_score', 0.0) <= 1.0):
        validation_issues.append("quality_score must be between 0.0 and 1.0")
    
    # Validate processing results
    if not isinstance(processing_results, dict):
        processing_issues.append("Processing results must be a dictionary")
    elif 'processing_status' not in processing_results:
        processing_issues.append("Missing processing_status in processing results")
    
    return {
        'validation_issues': validation_issues,
        'processing_issues': processing_issues,
        'is_valid': len(validation_issues) == 0 and len(processing_issues) == 0
    }

def get_quality_score_breakdown(validation_results, processing_results):
    """
    Get a detailed breakdown of the quality score calculation.
    
    Args:
        validation_results (dict): Validation results from Phase 1
        processing_results (dict): Processing results from Phase 2
        
    Returns:
        dict: Detailed breakdown of quality score components
    """
    
    validation_score = validation_results.get('quality_score', 0.0)
    processing_score = calculate_component_processing_score(processing_results)
    combined_score = (validation_score * 0.7) + (processing_score * 0.3)
    
    return {
        'validation_score': validation_score,
        'validation_weight': 0.7,
        'validation_contribution': validation_score * 0.7,
        'processing_score': processing_score,
        'processing_weight': 0.3,
        'processing_contribution': processing_score * 0.3,
        'combined_score': round(min(max(combined_score, 0.0), 1.0), 3),
        'weighting_formula': '70% validation + 30% processing'
    }

def log_quality_metrics(validation_results, processing_results, combined_score, component_type, request_id=None):
    """
    Log quality metrics for monitoring and debugging.
    
    Args:
        validation_results (dict): Validation results from Phase 1
        processing_results (dict): Processing results from Phase 2
        combined_score (float): Final combined quality score
        component_type (str): Type of component (Text, Image, Audio, Survey)
        request_id (str): Request ID for tracking
    """
    
    breakdown = get_quality_score_breakdown(validation_results, processing_results)
    
    logger.info(f"Quality Score Metrics for {component_type} - Request ID: {request_id}")
    logger.info(f"  Validation Score: {breakdown['validation_score']:.3f} (weight: {breakdown['validation_weight']})")
    logger.info(f"  Processing Score: {breakdown['processing_score']:.3f} (weight: {breakdown['processing_weight']})")
    logger.info(f"  Combined Score: {breakdown['combined_score']:.3f}")
    logger.info(f"  Formula: {breakdown['weighting_formula']}")
    
    # Log processing status if available
    processing_status = processing_results.get('processing_status', 'UNKNOWN')
    logger.info(f"  Processing Status: {processing_status}")
    
    return breakdown