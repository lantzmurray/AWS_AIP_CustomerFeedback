#!/usr/bin/env python3
"""
Model Selection Strategy for Foundation Models

This script creates a model selection strategy based on evaluation results
to optimize performance and cost for different use cases.

Enhanced for Phase 3 with support for additional foundation models and
integration with the new formatting components.
"""

import json
import pandas as pd
import numpy as np
import os
import sys
import boto3
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add the current directory to the Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the new Phase 3 formatters
from foundation_model_formatter import FoundationModelFormatter
from batch_processor import BatchProcessor

def create_model_selection_strategy(results_df):
    """
    Create a model selection strategy based on evaluation results.
    
    Args:
        results_df (pd.DataFrame): DataFrame with model evaluation results
        
    Returns:
        dict: Model selection strategy
    """
    
    # Calculate overall scores
    model_scores = results_df.groupby("model_id").agg({
        "latency": "mean",
        "similarity_score": "mean",
        "cost_per_1k_tokens": "mean",
        "accuracy": "mean"
    }).reset_index()
    
    # Normalize scores (lower latency and cost is better, higher similarity and accuracy is better)
    max_latency = model_scores["latency"].max()
    max_cost = model_scores["cost_per_1k_tokens"].max()
    
    model_scores["latency_score"] = 1 - (model_scores["latency"] / max_latency)
    model_scores["cost_score"] = 1 - (model_scores["cost_per_1k_tokens"] / max_cost)
    
    # Calculate weighted score (adjust weights based on priorities)
    model_scores["overall_score"] = (
        0.4 * model_scores["similarity_score"] + 
        0.2 * model_scores["accuracy"] +
        0.2 * model_scores["latency_score"] +
        0.2 * model_scores["cost_score"]
    )
    
    # Sort by overall score
    model_scores = model_scores.sort_values("overall_score", ascending=False)
    
    # Create strategy
    strategy = {
        "primary_model": model_scores.iloc[0]["model_id"],
        "fallback_models": model_scores.iloc[1:3]["model_id"].tolist(),
        "model_scores": model_scores.to_dict(orient="records"),
        "use_case_recommendations": generate_use_case_recommendations(model_scores),
        "created_timestamp": datetime.now().isoformat()
    }
    
    return strategy

def generate_use_case_recommendations(model_scores):
    """
    Generate recommendations for different use cases.
    
    Args:
        model_scores (pd.DataFrame): Model scores DataFrame
        
    Returns:
        dict: Use case recommendations
    """
    
    recommendations = {}
    
    # Best for accuracy-critical tasks
    accuracy_best = model_scores.loc[model_scores["accuracy"].idxmax()]
    recommendations["accuracy_critical"] = {
        "model": accuracy_best["model_id"],
        "reason": f"Highest accuracy score: {accuracy_best['accuracy']:.3f}"
    }
    
    # Best for real-time applications
    latency_best = model_scores.loc[model_scores["latency"].idxmin()]
    recommendations["real_time"] = {
        "model": latency_best["model_id"],
        "reason": f"Lowest latency: {latency_best['latency']:.2f}ms"
    }
    
    # Best for cost-sensitive applications
    cost_best = model_scores.loc[model_scores["cost_per_1k_tokens"].idxmin()]
    recommendations["cost_sensitive"] = {
        "model": cost_best["model_id"],
        "reason": f"Lowest cost: ${cost_best['cost_per_1k_tokens']:.4f}/1K tokens"
    }
    
    # Best for balanced performance
    balanced_best = model_scores.loc[model_scores["overall_score"].idxmax()]
    recommendations["balanced"] = {
        "model": balanced_best["model_id"],
        "reason": f"Best overall score: {balanced_best['overall_score']:.3f}"
    }
    
    return recommendations

def evaluate_model_performance(test_results, model_config):
    """
    Evaluate model performance against test cases.
    
    Args:
        test_results (list): List of test results
        model_config (dict): Model configuration
        
    Returns:
        dict: Performance evaluation
    """
    
    evaluation = {
        "model_id": model_config["model_id"],
        "test_cases": len(test_results),
        "passed_tests": 0,
        "failed_tests": 0,
        "average_latency": 0,
        "average_similarity": 0,
        "total_cost": 0
    }
    
    latencies = []
    similarities = []
    
    for result in test_results:
        if result["status"] == "success":
            evaluation["passed_tests"] += 1
            latencies.append(result.get("latency", 0))
            similarities.append(result.get("similarity_score", 0))
        else:
            evaluation["failed_tests"] += 1
        
        evaluation["total_cost"] += result.get("cost", 0)
    
    if latencies:
        evaluation["average_latency"] = np.mean(latencies)
        evaluation["average_similarity"] = np.mean(similarities)
        evaluation["success_rate"] = evaluation["passed_tests"] / evaluation["test_cases"]
    else:
        evaluation["success_rate"] = 0
    
    return evaluation

def create_model_benchmark_config():
    """
    Create enhanced configuration for model benchmarking.
    
    Returns:
        dict: Benchmark configuration
    """
    
    config = {
        "models": [
            {
                "model_id": "anthropic.claude-v2",
                "name": "Claude v2",
                "max_tokens": 100000,
                "cost_per_1k_tokens": 0.008,
                "expected_latency": 2000,  # ms
                "supports_multimodal": True,
                "supports_conversation": True,
                "strengths": ["analysis", "reasoning", "structured_output"],
                "data_types": ["text", "image", "audio", "survey"]
            },
            {
                "model_id": "anthropic.claude-instant-v1",
                "name": "Claude Instant",
                "max_tokens": 100000,
                "cost_per_1k_tokens": 0.0008,
                "expected_latency": 500,  # ms
                "supports_multimodal": False,
                "supports_conversation": True,
                "strengths": ["speed", "cost_efficiency"],
                "data_types": ["text", "survey"]
            },
            {
                "model_id": "anthropic.claude-3-sonnet-v1",
                "name": "Claude 3 Sonnet",
                "max_tokens": 200000,
                "cost_per_1k_tokens": 0.015,
                "expected_latency": 1500,  # ms
                "supports_multimodal": True,
                "supports_conversation": True,
                "strengths": ["analysis", "reasoning", "multimodal", "structured_output"],
                "data_types": ["text", "image", "audio", "survey"]
            },
            {
                "model_id": "anthropic.claude-3-haiku-v1",
                "name": "Claude 3 Haiku",
                "max_tokens": 200000,
                "cost_per_1k_tokens": 0.00025,
                "expected_latency": 300,  # ms
                "supports_multimodal": True,
                "supports_conversation": True,
                "strengths": ["speed", "cost_efficiency", "multimodal"],
                "data_types": ["text", "image", "survey"]
            },
            {
                "model_id": "amazon.titan-text-express-v1",
                "name": "Titan Text Express",
                "max_tokens": 8000,
                "cost_per_1k_tokens": 0.0008,
                "expected_latency": 800,  # ms
                "supports_multimodal": False,
                "supports_conversation": False,
                "strengths": ["cost_efficiency", "reliability"],
                "data_types": ["text", "survey"]
            },
            {
                "model_id": "amazon.titan-text-lite-v1",
                "name": "Titan Text Lite",
                "max_tokens": 4000,
                "cost_per_1k_tokens": 0.0003,
                "expected_latency": 400,  # ms
                "supports_multimodal": False,
                "supports_conversation": False,
                "strengths": ["speed", "cost_efficiency"],
                "data_types": ["text", "survey"]
            },
            {
                "model_id": "ai21.j2-ultra-v1",
                "name": "Jurassic-2 Ultra",
                "max_tokens": 8192,
                "cost_per_1k_tokens": 0.0188,
                "expected_latency": 1200,  # ms
                "supports_multimodal": False,
                "supports_conversation": False,
                "strengths": ["analysis", "structured_output"],
                "data_types": ["text", "survey"]
            },
            {
                "model_id": "cohere.command-text-v14",
                "name": "Command Text",
                "max_tokens": 4096,
                "cost_per_1k_tokens": 0.0015,
                "expected_latency": 600,  # ms
                "supports_multimodal": False,
                "supports_conversation": True,
                "strengths": ["generation", "summarization"],
                "data_types": ["text", "survey"]
            }
        ],
        "test_cases": [
            {
                "type": "text_analysis",
                "prompt": "Analyze this customer review and extract key insights.",
                "expected_response_type": "structured_analysis",
                "data_type": "text",
                "complexity": "medium"
            },
            {
                "type": "multimodal_analysis",
                "prompt": "Analyze this product image and extract text feedback.",
                "expected_response_type": "multimodal_analysis",
                "data_type": "image",
                "complexity": "high",
                "requires_multimodal": True
            },
            {
                "type": "conversation_analysis",
                "prompt": "Analyze this customer service call transcript.",
                "expected_response_type": "conversation_analysis",
                "data_type": "audio",
                "complexity": "high",
                "requires_conversation": True
            },
            {
                "type": "survey_analysis",
                "prompt": "Analyze this survey response and provide insights.",
                "expected_response_type": "survey_analysis",
                "data_type": "survey",
                "complexity": "medium"
            },
            {
                "type": "summarization",
                "prompt": "Summarize this customer feedback in 3 key points.",
                "expected_response_type": "summary",
                "data_type": "text",
                "complexity": "low"
            },
            {
                "type": "sentiment_analysis",
                "prompt": "What is the sentiment of this customer feedback?",
                "expected_response_type": "sentiment",
                "data_type": "text",
                "complexity": "low"
            },
            {
                "type": "recommendation",
                "prompt": "Based on this feedback, what improvements would you recommend?",
                "expected_response_type": "recommendations",
                "data_type": "text",
                "complexity": "medium"
            },
            {
                "type": "batch_processing",
                "prompt": "Process this batch of customer reviews efficiently.",
                "expected_response_type": "batch_analysis",
                "data_type": "text",
                "complexity": "high",
                "batch_size": 100
            }
        ],
        "evaluation_criteria": {
            "latency_weight": 0.2,
            "accuracy_weight": 0.3,
            "cost_weight": 0.2,
            "similarity_weight": 0.2,
            "multimodal_capability_weight": 0.05,
            "conversation_capability_weight": 0.05
        },
        "data_type_weights": {
            "text": 1.0,
            "image": 1.2,
            "audio": 1.1,
            "survey": 0.9
        },
        "complexity_multipliers": {
            "low": 0.8,
            "medium": 1.0,
            "high": 1.3
        }
    }
    
    return config

def optimize_model_selection_for_use_case(strategy, use_case, constraints=None):
    """
    Optimize model selection for a specific use case.
    
    Args:
        strategy (dict): Model selection strategy
        use_case (str): Specific use case
        constraints (dict): Optional constraints
        
    Returns:
        dict: Optimized model recommendation
    """
    
    use_case_recommendations = strategy.get("use_case_recommendations", {})
    
    if use_case in use_case_recommendations:
        recommended_model = use_case_recommendations[use_case]["model"]
    else:
        # Default to primary model
        recommended_model = strategy["primary_model"]
    
    # Apply constraints if provided
    if constraints:
        recommended_model = apply_constraints(recommended_model, strategy, constraints)
    
    # Find model details
    model_details = None
    for model in strategy["model_scores"]:
        if model["model_id"] == recommended_model:
            model_details = model
            break
    
    recommendation = {
        "use_case": use_case,
        "recommended_model": recommended_model,
        "model_details": model_details,
        "reasoning": use_case_recommendations.get(use_case, {}).get("reason", "Default primary model"),
        "fallback_models": strategy["fallback_models"],
        "constraints_applied": constraints is not None
    }
    
    return recommendation

def apply_constraints(primary_model, strategy, constraints):
    """
    Apply constraints to model selection.
    
    Args:
        primary_model (str): Primary recommended model
        strategy (dict): Model selection strategy
        constraints (dict): Constraints to apply
        
    Returns:
        str: Constrained model recommendation
    """
    
    # Check cost constraint
    if "max_cost_per_1k_tokens" in constraints:
        max_cost = constraints["max_cost_per_1k_tokens"]
        for model in strategy["model_scores"]:
            if (model["model_id"] == primary_model and 
                model["cost_per_1k_tokens"] > max_cost):
                # Find best model within cost constraint
                affordable_models = [
                    m for m in strategy["model_scores"] 
                    if m["cost_per_1k_tokens"] <= max_cost
                ]
                if affordable_models:
                    return affordable_models[0]["model_id"]
    
    # Check latency constraint
    if "max_latency_ms" in constraints:
        max_latency = constraints["max_latency_ms"]
        for model in strategy["model_scores"]:
            if (model["model_id"] == primary_model and 
                model["latency"] > max_latency):
                # Find fastest model within constraint
                fast_models = [
                    m for m in strategy["model_scores"] 
                    if m["latency"] <= max_latency
                ]
                if fast_models:
                    return fast_models[0]["model_id"]
    
    # Check accuracy constraint
    if "min_accuracy" in constraints:
        min_accuracy = constraints["min_accuracy"]
        for model in strategy["model_scores"]:
            if (model["model_id"] == primary_model and 
                model["accuracy"] < min_accuracy):
                # Find most accurate model meeting constraint
                accurate_models = [
                    m for m in strategy["model_scores"] 
                    if m["accuracy"] >= min_accuracy
                ]
                if accurate_models:
                    return accurate_models[0]["model_id"]
    
    return primary_model

def select_model_for_data_type(data_type: str, complexity: str = "medium",
                             requirements: Optional[Dict] = None) -> str:
    """
    Select best model for a specific data type and complexity.
    
    Args:
        data_type (str): Type of data (text, image, audio, survey)
        complexity (str): Complexity level (low, medium, high)
        requirements (dict): Additional requirements
        
    Returns:
        str: Recommended model ID
    """
    
    config = create_model_benchmark_config()
    models = config["models"]
    
    # Filter models that support the data type
    compatible_models = [
        model for model in models
        if data_type in model["data_types"]
    ]
    
    # Apply additional requirements
    if requirements:
        if requirements.get("requires_multimodal"):
            compatible_models = [
                model for model in compatible_models
                if model["supports_multimodal"]
            ]
        
        if requirements.get("requires_conversation"):
            compatible_models = [
                model for model in compatible_models
                if model["supports_conversation"]
            ]
        
        if requirements.get("max_cost_per_1k_tokens"):
            max_cost = requirements["max_cost_per_1k_tokens"]
            compatible_models = [
                model for model in compatible_models
                if model["cost_per_1k_tokens"] <= max_cost
            ]
    
    # Score models based on complexity and data type weights
    data_type_weights = config["data_type_weights"]
    complexity_multipliers = config["complexity_multipliers"]
    
    best_model = None
    best_score = -1
    
    for model in compatible_models:
        # Base score from strengths
        score = 0
        
        # Check if model strengths match requirements
        if complexity == "high" and "analysis" in model["strengths"]:
            score += 3
        if complexity == "low" and "speed" in model["strengths"]:
            score += 3
        if "cost_efficiency" in model["strengths"]:
            score += 2
        
        # Apply data type weight
        score *= data_type_weights.get(data_type, 1.0)
        
        # Apply complexity multiplier
        score *= complexity_multipliers.get(complexity, 1.0)
        
        # Penalty for high cost
        if model["cost_per_1k_tokens"] > 0.01:
            score *= 0.8
        
        # Penalty for high latency
        if model["expected_latency"] > 1500:
            score *= 0.9
        
        if score > best_score:
            best_score = score
            best_model = model["model_id"]
    
    return best_model or compatible_models[0]["model_id"] if compatible_models else "anthropic.claude-v2"

def create_batch_processing_strategy(batch_size: int, data_type: str,
                                  time_constraint: Optional[int] = None) -> Dict:
    """
    Create a strategy for batch processing with model selection.
    
    Args:
        batch_size (int): Size of batch
        data_type (str): Type of data in batch
        time_constraint (int): Optional time constraint in seconds
        
    Returns:
        dict: Batch processing strategy
    """
    
    config = create_model_benchmark_config()
    models = config["models"]
    
    # Filter models suitable for batch processing
    batch_models = [
        model for model in models
        if ("cost_efficiency" in model["strengths"] or
            "speed" in model["strengths"]) and
           data_type in model["data_types"]
    ]
    
    # Calculate estimated costs and times
    strategies = []
    
    for model in batch_models:
        estimated_time = (model["expected_latency"] / 1000) * batch_size
        estimated_cost = model["cost_per_1k_tokens"] * batch_size * 0.1  # Assume 100 tokens per item
        
        strategy_viable = True
        if time_constraint and estimated_time > time_constraint:
            strategy_viable = False
        
        strategies.append({
            "model_id": model["model_id"],
            "model_name": model["name"],
            "estimated_time_seconds": estimated_time,
            "estimated_cost_usd": estimated_cost,
            "viable": strategy_viable,
            "cost_per_item": estimated_cost / batch_size,
            "time_per_item": estimated_time / batch_size
        })
    
    # Sort by cost (for viable strategies)
    viable_strategies = [s for s in strategies if s["viable"]]
    viable_strategies.sort(key=lambda x: x["estimated_cost_usd"])
    
    if not viable_strategies:
        # If none are viable, sort by time
        strategies.sort(key=lambda x: x["estimated_time_seconds"])
        recommended = strategies[0]
    else:
        recommended = viable_strategies[0]
    
    return {
        "batch_size": batch_size,
        "data_type": data_type,
        "time_constraint": time_constraint,
        "recommended_strategy": recommended,
        "all_strategies": strategies,
        "processing_recommendation": f"Use {recommended['model_name']} for optimal cost-performance ratio"
    }

def optimize_for_cost_performance_tradeoff(use_case: str, budget_constraint: Optional[float] = None,
                                          performance_requirement: Optional[float] = None) -> Dict:
    """
    Optimize model selection for cost-performance tradeoff.
    
    Args:
        use_case (str): Specific use case
        budget_constraint (float): Maximum cost per 1K tokens
        performance_requirement (float): Minimum accuracy requirement
        
    Returns:
        dict: Optimization recommendation
    """
    
    config = create_model_benchmark_config()
    models = config["models"]
    
    # Filter models based on constraints
    viable_models = models.copy()
    
    if budget_constraint:
        viable_models = [
            model for model in viable_models
            if model["cost_per_1k_tokens"] <= budget_constraint
        ]
    
    if performance_requirement:
        # For this example, we'll use expected performance based on model characteristics
        # In practice, this would use actual benchmark data
        viable_models = [
            model for model in viable_models
            if estimate_model_performance(model["model_id"]) >= performance_requirement
        ]
    
    if not viable_models:
        return {
            "use_case": use_case,
            "status": "no_viable_models",
            "message": "No models meet the specified constraints",
            "recommendations": [
                "Consider relaxing budget constraints",
                "Consider lowering performance requirements",
                "Consider hybrid approach with multiple models"
            ]
        }
    
    # Calculate cost-performance ratio
    for model in viable_models:
        performance_score = estimate_model_performance(model["model_id"])
        cost_performance_ratio = performance_score / model["cost_per_1k_tokens"]
        model["cost_performance_ratio"] = cost_performance_ratio
    
    # Sort by cost-performance ratio
    viable_models.sort(key=lambda x: x["cost_performance_ratio"], reverse=True)
    
    best_model = viable_models[0]
    
    return {
        "use_case": use_case,
        "recommended_model": best_model["model_id"],
        "model_name": best_model["name"],
        "estimated_performance": estimate_model_performance(best_model["model_id"]),
        "cost_per_1k_tokens": best_model["cost_per_1k_tokens"],
        "cost_performance_ratio": best_model["cost_performance_ratio"],
        "alternatives": [
            {
                "model_id": model["model_id"],
                "model_name": model["name"],
                "reason": "Better performance" if "analysis" in model["strengths"] else "Lower cost"
            }
            for model in viable_models[1:3]
        ],
        "status": "success"
    }

def estimate_model_performance(model_id: str) -> float:
    """
    Estimate model performance based on model characteristics.
    
    Args:
        model_id (str): Model identifier
        
    Returns:
        float: Estimated performance score (0-1)
    """
    
    performance_estimates = {
        "anthropic.claude-v2": 0.92,
        "anthropic.claude-instant-v1": 0.85,
        "anthropic.claude-3-sonnet-v1": 0.95,
        "anthropic.claude-3-haiku-v1": 0.88,
        "amazon.titan-text-express-v1": 0.83,
        "amazon.titan-text-lite-v1": 0.78,
        "ai21.j2-ultra-v1": 0.90,
        "cohere.command-text-v14": 0.86
    }
    
    return performance_estimates.get(model_id, 0.80)

def create_model_rotation_strategy(models: List[str], rotation_interval: str = "daily") -> Dict:
    """
    Create a model rotation strategy for load balancing and cost optimization.
    
    Args:
        models (list): List of model IDs to rotate
        rotation_interval (str): Rotation interval (hourly, daily, weekly)
        
    Returns:
        dict: Model rotation strategy
    """
    
    config = create_model_benchmark_config()
    model_details = {model["model_id"]: model for model in config["models"]}
    
    # Create rotation schedule
    rotation_schedule = []
    current_time = datetime.now()
    
    for i, model_id in enumerate(models):
        if model_id in model_details:
            schedule_entry = {
                "model_id": model_id,
                "model_name": model_details[model_id]["name"],
                "position": i,
                "active_percentage": 100 / len(models),
                "estimated_cost_per_hour": calculate_hourly_cost(model_id, rotation_interval),
                "strengths": model_details[model_id]["strengths"]
            }
            rotation_schedule.append(schedule_entry)
    
    return {
        "rotation_interval": rotation_interval,
        "models": models,
        "rotation_schedule": rotation_schedule,
        "total_models": len(models),
        "rotation_strategy": "round_robin",
        "fallback_model": models[0] if models else "anthropic.claude-v2",
        "created_timestamp": datetime.now().isoformat()
    }

def calculate_hourly_cost(model_id: str, rotation_interval: str) -> float:
    """
    Calculate estimated hourly cost for a model based on rotation interval.
    
    Args:
        model_id (str): Model identifier
        rotation_interval (str): Rotation interval
        
    Returns:
        float: Estimated hourly cost
    """
    
    config = create_model_benchmark_config()
    model_details = {model["model_id"]: model for model in config["models"]}
    
    if model_id not in model_details:
        return 0.0
    
    cost_per_1k = model_details[model_id]["cost_per_1k_tokens"]
    
    # Estimate tokens per hour based on rotation interval
    if rotation_interval == "hourly":
        tokens_per_hour = 10000  # Assume 10K tokens per hour when active
    elif rotation_interval == "daily":
        tokens_per_hour = 417  # 10K tokens per day / 24 hours
    elif rotation_interval == "weekly":
        tokens_per_hour = 60  # 10K tokens per week / 168 hours
    else:
        tokens_per_hour = 1000
    
    return (tokens_per_hour / 1000) * cost_per_1k

def save_strategy_to_appconfig(strategy, bucket_name, object_key):
    """
    Save model selection strategy to AWS AppConfig.
    
    Args:
        strategy (dict): Model selection strategy
        bucket_name (str): S3 bucket name
        object_key (str): S3 object key
    """
    
    import boto3
    
    s3_client = boto3.client('s3')
    
    # Save strategy to S3
    s3_client.put_object(
        Bucket=bucket_name,
        Key=object_key,
        Body=json.dumps(strategy, indent=2),
        ContentType='application/json'
    )
    
    print(f"Saved model selection strategy to s3://{bucket_name}/{object_key}")

def load_strategy_from_appconfig(bucket_name, object_key):
    """
    Load model selection strategy from AWS AppConfig.
    
    Args:
        bucket_name (str): S3 bucket name
        object_key (str): S3 object key
        
    Returns:
        dict: Model selection strategy
    """
    
    import boto3
    
    s3_client = boto3.client('s3')
    
    # Load strategy from S3
    response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
    strategy = json.loads(response['Body'].read().decode('utf-8'))
    
    print(f"Loaded model selection strategy from s3://{bucket_name}/{object_key}")
    return strategy

if __name__ == "__main__":
    # Example usage with sample data
    sample_data = {
        "model_id": ["anthropic.claude-v2", "anthropic.claude-instant-v1", "amazon.titan-text-express-v1"],
        "latency": [1500, 400, 700],
        "similarity_score": [0.95, 0.88, 0.82],
        "cost_per_1k_tokens": [0.008, 0.0008, 0.0008],
        "accuracy": [0.92, 0.85, 0.83]
    }
    
    results_df = pd.DataFrame(sample_data)
    
    # Generate strategy
    strategy = create_model_selection_strategy(results_df)
    print(json.dumps(strategy, indent=2))
    
    # Save strategy
    save_strategy_to_appconfig(
        strategy, 
        "customer-feedback-analysis-bucket", 
        "config/model-selection-strategy.json"
    )