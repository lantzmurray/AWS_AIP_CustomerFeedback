#!/usr/bin/env python3
"""
Survey Data Formatter

Specialized formatter for survey response data that supports structured data formats
for foundation models and includes statistical summaries and insights integration 
with Phase 2 survey processing output.
"""

import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
import boto3
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import statistics
import re

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class SurveyFormatter:
    """
    Specialized formatter for survey data with analytics capabilities.
    """
    
    def __init__(self):
        """Initialize survey formatter."""
        self.s3_client = boto3.client('s3')
        self.templates = self._load_templates()
        
    def _load_templates(self) -> Dict[str, str]:
        """Load survey formatting templates."""
        return {
            "claude_analytics": """You are analyzing customer survey responses to generate business insights.

SURVEY SUMMARY:
{summary_text}

RATINGS:
{ratings}

COMMENTS:
{comments}

IMPROVEMENT AREAS:
{improvement_areas}

STATISTICAL ANALYSIS:
{statistical_analysis}

TREND ANALYSIS:
{trend_analysis}

PRIORITY SCORE: {priority_score}

METADATA:
- Customer ID: {customer_id}
- Survey Date: {survey_date}
- Survey Type: {survey_type}
- Quality Score: {quality_score}

Please provide:
1. Overall customer satisfaction assessment
2. Key themes and patterns
3. Specific improvement areas to address
4. Priority level for follow-up
5. Recommended action plan

Format your response as structured JSON with the following keys:
satisfaction_assessment, key_themes, improvement_areas, priority_level, action_plan""",
            
            "titan_analysis": """Analyze this customer survey response:

Summary: {summary_text}
Ratings: {ratings}
Priority Score: {priority_score}

Provide analysis in JSON format with: satisfaction_level, main_themes, improvement_priorities, action_items.""",
            
            "training_prompt": """Customer survey analysis:
Summary: {summary_text}
Ratings: {ratings}
Comments: {comments}
Priority: {priority_score}""",
            
            "training_completion": """{satisfaction_assessment}. Key themes: {key_themes}. Priority: {priority_level}. Actions: {action_plan}."""
        }
    
    def format_for_claude(self, processed_survey: Dict[str, Any], 
                          format_type: str = "analytics") -> Dict[str, Any]:
        """
        Format processed survey data for Claude models.
        
        Args:
            processed_survey: Processed survey data from Phase 2
            format_type: Output format (analytics, jsonl, parquet)
            
        Returns:
            Formatted data for Claude
        """
        if format_type == "analytics":
            return self._format_claude_analytics(processed_survey)
        elif format_type == "jsonl":
            return self._format_jsonl(processed_survey, "claude")
        elif format_type == "parquet":
            return self._format_parquet(processed_survey, "claude")
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def format_for_titan(self, processed_survey: Dict[str, Any],
                         format_type: str = "json") -> Dict[str, Any]:
        """
        Format processed survey data for Titan models.
        
        Args:
            processed_survey: Processed survey data from Phase 2
            format_type: Output format (json, jsonl, parquet)
            
        Returns:
            Formatted data for Titan
        """
        if format_type == "json":
            return self._format_titan_json(processed_survey)
        elif format_type == "jsonl":
            return self._format_jsonl(processed_survey, "titan")
        elif format_type == "parquet":
            return self._format_parquet(processed_survey, "titan")
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def format_for_training(self, processed_survey: Dict[str, Any],
                            output_format: str = "jsonl") -> Dict[str, Any]:
        """
        Format processed survey data for model training.
        
        Args:
            processed_survey: Processed survey data from Phase 2
            output_format: Output format (jsonl, parquet)
            
        Returns:
            Training data format
        """
        if output_format == "jsonl":
            return self._format_training_jsonl(processed_survey)
        elif output_format == "parquet":
            return self._format_training_parquet(processed_survey)
        else:
            raise ValueError(f"Unsupported training format: {output_format}")
    
    def format_analytics_request(self, processed_survey: Dict[str, Any]) -> Dict[str, Any]:
        """
        Structure survey data for statistical analysis.
        
        Args:
            processed_survey: Processed survey data from Phase 2
            
        Returns:
            Analytics-ready format
        """
        # Extract ratings
        ratings = processed_survey.get("ratings", {})
        
        # Calculate statistical summaries
        statistical_analysis = self._calculate_statistics(ratings)
        
        # Extract comments and themes
        comments = processed_survey.get("comments", [])
        improvement_areas = processed_survey.get("improvement_areas", [])
        
        # Analyze themes
        theme_analysis = self._analyze_themes(comments, improvement_areas)
        
        return {
            "survey_id": processed_survey.get("metadata", {}).get("survey_id", "unknown"),
            "customer_id": processed_survey.get("customer_id", "unknown"),
            "ratings": ratings,
            "statistical_summary": statistical_analysis,
            "theme_analysis": theme_analysis,
            "priority_score": processed_survey.get("priority_score", 0.0),
            "survey_metadata": processed_survey.get("metadata", {}),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def create_trend_context(self, processed_survey: Dict[str, Any], 
                           historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Add historical trend data to survey analysis.
        
        Args:
            processed_survey: Current survey data
            historical_data: Historical survey responses
            
        Returns:
            Survey data with trend context
        """
        current_ratings = processed_survey.get("ratings", {})
        
        # Calculate trend metrics
        trend_analysis = self._calculate_trends(current_ratings, historical_data)
        
        # Identify patterns
        patterns = self._identify_patterns(historical_data)
        
        return {
            "current_survey": processed_survey,
            "historical_data": historical_data,
            "trend_analysis": trend_analysis,
            "identified_patterns": patterns,
            "trend_context": {
                "data_points": len(historical_data) + 1,
                "time_period": self._calculate_time_period(historical_data),
                "trend_direction": trend_analysis.get("overall_trend", "stable")
            }
        }
    
    def generate_business_prompts(self, processed_survey: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create business-focused prompts for survey analysis.
        
        Args:
            processed_survey: Processed survey data
            
        Returns:
            Business-focused prompts
        """
        ratings = processed_survey.get("ratings", {})
        priority_score = processed_survey.get("priority_score", 0.0)
        
        # Generate different business perspectives
        prompts = {
            "executive_summary": f"""Based on this customer survey with an overall satisfaction rating of {ratings.get('overall_satisfaction', 0)}/5 and priority score of {priority_score}, provide a concise executive summary for leadership.""",
            
            "operational_insights": f"""Analyze the operational implications of this survey response, focusing on process improvements and customer service enhancements.""",
            
            "product_development": f"""Extract product-related insights from this survey that could inform product development and improvement initiatives.""",
            
            "customer_experience": f"""Evaluate the customer experience journey based on this survey feedback and identify key touchpoints for improvement.""",
            
            "competitive_analysis": f"""Assess how this survey response might reflect our competitive position and market perception."""
        }
        
        return prompts
    
    def aggregate_survey_data(self, survey_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine multiple survey responses for batch analysis.
        
        Args:
            survey_responses: List of survey responses
            
        Returns:
            Aggregated survey data
        """
        if not survey_responses:
            return {"error": "No survey responses provided"}
        
        # Aggregate ratings
        all_ratings = [response.get("ratings", {}) for response in survey_responses]
        aggregated_ratings = self._aggregate_ratings(all_ratings)
        
        # Aggregate comments
        all_comments = []
        for response in survey_responses:
            all_comments.extend(response.get("comments", []))
        
        # Aggregate improvement areas
        all_improvements = []
        for response in survey_responses:
            all_improvements.extend(response.get("improvement_areas", []))
        
        # Calculate summary statistics
        summary_stats = {
            "total_responses": len(survey_responses),
            "response_date_range": self._get_date_range(survey_responses),
            "average_priority_score": sum(r.get("priority_score", 0) for r in survey_responses) / len(survey_responses),
            "high_priority_count": sum(1 for r in survey_responses if r.get("priority_score", 0) > 7.0),
            "low_satisfaction_count": sum(1 for r in survey_responses if r.get("ratings", {}).get("overall_satisfaction", 5) < 3.0)
        }
        
        return {
            "aggregated_ratings": aggregated_ratings,
            "all_comments": all_comments,
            "all_improvement_areas": all_improvements,
            "summary_statistics": summary_stats,
            "aggregation_timestamp": datetime.now().isoformat()
        }
    
    def create_benchmark_context(self, processed_survey: Dict[str, Any], 
                              industry_benchmarks: Dict[str, float]) -> Dict[str, Any]:
        """
        Add industry benchmarks to survey analysis.
        
        Args:
            processed_survey: Current survey data
            industry_benchmarks: Industry benchmark data
            
        Returns:
            Survey data with benchmark context
        """
        current_ratings = processed_survey.get("ratings", {})
        
        # Compare with benchmarks
        benchmark_comparison = {}
        for metric, benchmark_value in industry_benchmarks.items():
            current_value = current_ratings.get(metric, 0.0)
            
            benchmark_comparison[metric] = {
                "current_value": current_value,
                "benchmark_value": benchmark_value,
                "difference": current_value - benchmark_value,
                "percent_difference": ((current_value - benchmark_value) / benchmark_value * 100) if benchmark_value > 0 else 0,
                "performance_level": self._assess_performance_level(current_value, benchmark_value)
            }
        
        return {
            "survey_data": processed_survey,
            "industry_benchmarks": industry_benchmarks,
            "benchmark_comparison": benchmark_comparison,
            "overall_benchmark_score": self._calculate_benchmark_score(benchmark_comparison),
            "benchmark_analysis_timestamp": datetime.now().isoformat()
        }
    
    def _format_claude_analytics(self, processed_survey: Dict[str, Any]) -> Dict[str, Any]:
        """Format survey data for Claude analytics."""
        
        # Extract data
        summary_text = processed_survey.get("summary_text", "")
        ratings = processed_survey.get("ratings", {})
        comments = processed_survey.get("comments", [])
        improvement_areas = processed_survey.get("improvement_areas", [])
        metadata = processed_survey.get("metadata", {})
        
        # Format ratings
        ratings_text = json.dumps(ratings, indent=2)
        
        # Format comments
        comments_text = "\n".join(f"- {comment}" for comment in comments)
        
        # Format improvement areas
        improvement_text = "\n".join(f"- {area}" for area in improvement_areas)
        
        # Calculate statistical analysis
        statistical_analysis = self._calculate_statistics(ratings)
        statistical_text = json.dumps(statistical_analysis, indent=2)
        
        # Calculate trend analysis (simplified)
        trend_analysis = self._generate_trend_summary(ratings)
        trend_text = json.dumps(trend_analysis, indent=2)
        
        # Extract metadata
        customer_id = processed_survey.get("customer_id", "N/A")
        survey_date = metadata.get("survey_date", "N/A")
        survey_type = metadata.get("survey_type", "N/A")
        quality_score = metadata.get("quality_score", 0.0)
        priority_score = processed_survey.get("priority_score", 0.0)
        
        # Create analytics prompt
        prompt = self.templates["claude_analytics"].format(
            summary_text=summary_text,
            ratings=ratings_text,
            comments=comments_text,
            improvement_areas=improvement_text,
            statistical_analysis=statistical_text,
            trend_analysis=trend_text,
            priority_score=priority_score,
            customer_id=customer_id,
            survey_date=survey_date,
            survey_type=survey_type,
            quality_score=quality_score
        )
        
        # Create Claude analytics request
        analytics_request = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2000,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "top_p": 0.999,
            "top_k": 250
        }
        
        # Add metadata
        analytics_request["metadata"] = {
            "data_type": "survey",
            "model": "claude-v2",
            "format_timestamp": datetime.now().isoformat(),
            "source_data_id": metadata.get("id", customer_id),
            "survey_type": survey_type,
            "priority_score": priority_score,
            "quality_score": quality_score
        }
        
        return analytics_request
    
    def _format_titan_json(self, processed_survey: Dict[str, Any]) -> Dict[str, Any]:
        """Format survey data for Titan Text model."""
        
        summary_text = processed_survey.get("summary_text", "")
        ratings = processed_survey.get("ratings", {})
        priority_score = processed_survey.get("priority_score", 0.0)
        metadata = processed_survey.get("metadata", {})
        
        # Format for Titan
        ratings_text = json.dumps(ratings)
        
        # Create Titan prompt
        prompt = self.templates["titan_analysis"].format(
            summary_text=summary_text,
            ratings=ratings_text,
            priority_score=priority_score
        )
        
        titan_request = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 2000,
                "temperature": 0.7,
                "topP": 0.999,
                "stopSequences": []
            }
        }
        
        # Add metadata
        titan_request["metadata"] = {
            "data_type": "survey",
            "model": "titan-text-express-v1",
            "format_timestamp": datetime.now().isoformat(),
            "source_data_id": metadata.get("id", metadata.get("customer_id", "unknown")),
            "survey_type": metadata.get("survey_type", "unknown"),
            "priority_score": priority_score,
            "quality_score": metadata.get("quality_score", 0.0)
        }
        
        return titan_request
    
    def _format_jsonl(self, processed_survey: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Format survey data as JSONL for training."""
        
        summary_text = processed_survey.get("summary_text", "")
        ratings = processed_survey.get("ratings", {})
        comments = processed_survey.get("comments", [])
        priority_score = processed_survey.get("priority_score", 0.0)
        metadata = processed_survey.get("metadata", {})
        
        # Create training prompt
        comments_text = ", ".join(comments)
        
        prompt = self.templates["training_prompt"].format(
            summary_text=summary_text,
            ratings=json.dumps(ratings),
            comments=comments_text,
            priority_score=priority_score
        )
        
        # Generate completion
        completion = self.templates["training_completion"].format(
            satisfaction_assessment=f"Overall satisfaction: {ratings.get('overall_satisfaction', 0)}/5",
            key_themes="Customer service and product quality",
            priority_level="Medium" if priority_score < 7 else "High",
            action_plan="Improve customer response time and product quality"
        )
        
        jsonl_record = {
            "prompt": prompt,
            "completion": completion,
            "data_type": "survey",
            "model": model,
            "quality_score": metadata.get("quality_score", 0.0),
            "customer_id": processed_survey.get("customer_id", ""),
            "survey_id": metadata.get("survey_id", ""),
            "survey_type": metadata.get("survey_type", ""),
            "priority_score": priority_score,
            "timestamp": datetime.now().isoformat()
        }
        
        return {"jsonl_line": json.dumps(jsonl_record)}
    
    def _format_parquet(self, processed_survey: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Format survey data as Parquet for analytics."""
        
        summary_text = processed_survey.get("summary_text", "")
        ratings = processed_survey.get("ratings", {})
        comments = processed_survey.get("comments", [])
        improvement_areas = processed_survey.get("improvement_areas", [])
        metadata = processed_survey.get("metadata", {})
        
        # Create DataFrame
        df_data = {
            "summary_text": [summary_text],
            "overall_satisfaction": [ratings.get("overall_satisfaction", 0.0)],
            "product_quality": [ratings.get("product_quality", 0.0)],
            "customer_service": [ratings.get("customer_service", 0.0)],
            "comments": [", ".join(comments)],
            "improvement_areas": [", ".join(improvement_areas)],
            "data_type": ["survey"],
            "model": [model],
            "quality_score": [metadata.get("quality_score", 0.0)],
            "customer_id": [processed_survey.get("customer_id", "")],
            "survey_id": [metadata.get("survey_id", "")],
            "survey_type": [metadata.get("survey_type", "")],
            "priority_score": [processed_survey.get("priority_score", 0.0)],
            "timestamp": [datetime.now()]
        }
        
        df = pd.DataFrame(df_data)
        
        # Convert to Parquet
        table = pa.Table.from_pandas(df)
        parquet_buffer = pa.BufferOutputStream()
        pq.write_table(table, parquet_buffer)
        
        return {
            "parquet_data": parquet_buffer.getvalue().to_pybytes(),
            "schema": str(table.schema),
            "row_count": len(df)
        }
    
    def _format_training_jsonl(self, processed_survey: Dict[str, Any]) -> Dict[str, Any]:
        """Format survey data specifically for training."""
        
        summary_text = processed_survey.get("summary_text", "")
        ratings = processed_survey.get("ratings", {})
        metadata = processed_survey.get("metadata", {})
        
        # Create fine-tuning format
        training_record = {
            "instruction": "Analyze this customer survey response and provide business insights.",
            "input": f"""Summary: {summary_text}
Ratings: {json.dumps(ratings)}
Priority Score: {processed_survey.get('priority_score', 0.0)}""",
            "output": f"""Satisfaction Assessment: {ratings.get('overall_satisfaction', 0)}/5
Key Themes: Customer service and product quality
Priority Level: {'High' if processed_survey.get('priority_score', 0) > 7 else 'Medium'}
Action Plan: Improve customer experience and product features""",
            "data_type": "survey_analysis",
            "quality_score": metadata.get("quality_score", 0.0)
        }
        
        return {"jsonl_line": json.dumps(training_record)}
    
    def _format_training_parquet(self, processed_survey: Dict[str, Any]) -> Dict[str, Any]:
        """Format survey data as Parquet for training."""
        
        summary_text = processed_survey.get("summary_text", "")
        ratings = processed_survey.get("ratings", {})
        metadata = processed_survey.get("metadata", {})
        
        # Create training DataFrame
        df_data = {
            "instruction": ["Analyze this customer survey response and provide business insights."],
            "input": [f"""Summary: {summary_text}
Ratings: {json.dumps(ratings)}
Priority Score: {processed_survey.get('priority_score', 0.0)}"""],
            "output": [f"""Satisfaction Assessment: {ratings.get('overall_satisfaction', 0)}/5
Key Themes: Customer service and product quality
Priority Level: {'High' if processed_survey.get('priority_score', 0) > 7 else 'Medium'}
Action Plan: Improve customer experience and product features"""],
            "data_type": ["survey_analysis"],
            "quality_score": [metadata.get("quality_score", 0.0)],
            "customer_id": [processed_survey.get("customer_id", "")],
            "survey_id": [metadata.get("survey_id", "")],
            "timestamp": [datetime.now()]
        }
        
        df = pd.DataFrame(df_data)
        
        # Convert to Parquet
        table = pa.Table.from_pandas(df)
        parquet_buffer = pa.BufferOutputStream()
        pq.write_table(table, parquet_buffer)
        
        return {
            "parquet_data": parquet_buffer.getvalue().to_pybytes(),
            "schema": str(table.schema),
            "row_count": len(df)
        }
    
    def _calculate_statistics(self, ratings: Dict[str, float]) -> Dict[str, Any]:
        """Calculate statistical summary of ratings."""
        if not ratings:
            return {"error": "No ratings provided"}
        
        rating_values = list(ratings.values())
        
        stats = {
            "mean": statistics.mean(rating_values),
            "median": statistics.median(rating_values),
            "mode": statistics.mode(rating_values) if len(set(rating_values)) < len(rating_values) else rating_values[0],
            "min": min(rating_values),
            "max": max(rating_values),
            "std_dev": statistics.stdev(rating_values) if len(rating_values) > 1 else 0.0,
            "variance": statistics.variance(rating_values) if len(rating_values) > 1 else 0.0,
            "count": len(rating_values)
        }
        
        # Add rating-specific insights
        stats["insights"] = {
            "highest_rated_category": max(ratings.items(), key=lambda x: x[1])[0] if ratings else None,
            "lowest_rated_category": min(ratings.items(), key=lambda x: x[1])[0] if ratings else None,
            "rating_range": max(rating_values) - min(rating_values) if rating_values else 0,
            "consistency_score": 1.0 - (stats["std_dev"] / 5.0)  # Normalize by max rating
        }
        
        return stats
    
    def _analyze_themes(self, comments: List[str], improvement_areas: List[str]) -> Dict[str, Any]:
        """Analyze themes from comments and improvement areas."""
        # Combine all text
        all_text = " ".join(comments + improvement_areas).lower()
        
        # Define theme keywords
        theme_keywords = {
            "Customer Service": ["service", "support", "staff", "help", "representative", "agent"],
            "Product Quality": ["quality", "product", "item", "manufacturing", "design", "durability"],
            "Price/Value": ["price", "cost", "value", "expensive", "cheap", "worth"],
            "Delivery/Shipping": ["delivery", "shipping", "arrival", "package", "logistics"],
            "Communication": ["communication", "information", "updates", "notification", "clarity"],
            "User Experience": ["experience", "easy", "difficult", "interface", "usability", "navigation"]
        }
        
        # Count theme occurrences
        theme_counts = {}
        for theme, keywords in theme_keywords.items():
            count = sum(all_text.count(keyword) for keyword in keywords)
            if count > 0:
                theme_counts[theme] = count
        
        # Sort by count
        sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "identified_themes": dict(sorted_themes),
            "dominant_theme": sorted_themes[0][0] if sorted_themes else None,
            "theme_diversity": len(sorted_themes),
            "total_themes_found": len(theme_keywords)
        }
    
    def _calculate_trends(self, current_ratings: Dict[str, float], 
                         historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate trend analysis from historical data."""
        if not historical_data:
            return {"trend": "insufficient_data"}
        
        # Extract historical ratings for comparison
        historical_ratings = [item.get("ratings", {}) for item in historical_data]
        
        trends = {}
        for category in current_ratings:
            if category in historical_ratings[0]:  # Check if category exists in historical data
                historical_values = [rating.get(category, 0) for rating in historical_ratings]
                current_value = current_ratings[category]
                
                # Calculate trend
                if len(historical_values) >= 2:
                    recent_avg = statistics.mean(historical_values[-3:])  # Last 3 data points
                    older_avg = statistics.mean(historical_values[:-3]) if len(historical_values) > 3 else statistics.mean(historical_values[:-1])
                    
                    if recent_avg > older_avg:
                        trend_direction = "improving"
                    elif recent_avg < older_avg:
                        trend_direction = "declining"
                    else:
                        trend_direction = "stable"
                    
                    trends[category] = {
                        "direction": trend_direction,
                        "current_value": current_value,
                        "recent_average": recent_avg,
                        "historical_average": older_avg,
                        "change_percentage": ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0
                    }
        
        # Overall trend
        trend_directions = [trend.get("direction", "stable") for trend in trends.values()]
        improving_count = trend_directions.count("improving")
        declining_count = trend_directions.count("declining")
        
        if improving_count > declining_count:
            overall_trend = "improving"
        elif declining_count > improving_count:
            overall_trend = "declining"
        else:
            overall_trend = "stable"
        
        return {
            "category_trends": trends,
            "overall_trend": overall_trend,
            "data_points": len(historical_data)
        }
    
    def _identify_patterns(self, historical_data: List[Dict[str, Any]]) -> List[str]:
        """Identify patterns in historical survey data."""
        patterns = []
        
        if len(historical_data) < 3:
            return ["Insufficient data for pattern analysis"]
        
        # Analyze rating patterns
        all_ratings = [item.get("ratings", {}) for item in historical_data]
        
        # Check for consistent issues
        low_categories = set()
        for ratings in all_ratings:
            for category, score in ratings.items():
                if score < 3.0:  # Below average
                    low_categories.add(category)
        
        if low_categories:
            patterns.append(f"Consistently low ratings in: {', '.join(low_categories)}")
        
        # Check for seasonal patterns (simplified)
        dates = [item.get("metadata", {}).get("survey_date", "") for item in historical_data]
        # In a real implementation, this would analyze actual dates for seasonality
        
        # Check for improvement trends
        overall_scores = [self._calculate_overall_score(ratings) for ratings in all_ratings]
        if len(overall_scores) >= 3:
            recent_scores = overall_scores[-3:]
            older_scores = overall_scores[:-3]
            
            if statistics.mean(recent_scores) > statistics.mean(older_scores):
                patterns.append("Recent improvement in overall satisfaction")
            elif statistics.mean(recent_scores) < statistics.mean(older_scores):
                patterns.append("Recent decline in overall satisfaction")
        
        return patterns
    
    def _calculate_time_period(self, historical_data: List[Dict[str, Any]]) -> str:
        """Calculate the time period covered by historical data."""
        if not historical_data:
            return "Unknown"
        
        dates = []
        for item in historical_data:
            date_str = item.get("metadata", {}).get("survey_date", "")
            if date_str:
                try:
                    dates.append(datetime.fromisoformat(date_str.replace('Z', '+00:00')))
                except:
                    continue
        
        if len(dates) < 2:
            return "Insufficient date data"
        
        time_diff = max(dates) - min(dates)
        days = time_diff.days
        
        if days < 30:
            return f"{days} days"
        elif days < 365:
            months = days // 30
            return f"{months} months"
        else:
            years = days // 365
            return f"{years} years"
    
    def _get_date_range(self, survey_responses: List[Dict[str, Any]]) -> Dict[str, str]:
        """Get the date range of survey responses."""
        dates = []
        for response in survey_responses:
            date_str = response.get("metadata", {}).get("survey_date", "")
            if date_str:
                try:
                    dates.append(datetime.fromisoformat(date_str.replace('Z', '+00:00')))
                except:
                    continue
        
        if not dates:
            return {"start": "Unknown", "end": "Unknown"}
        
        return {
            "start": min(dates).isoformat(),
            "end": max(dates).isoformat()
        }
    
    def _aggregate_ratings(self, all_ratings: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Aggregate ratings across multiple responses."""
        if not all_ratings:
            return {}
        
        # Get all unique categories
        all_categories = set()
        for ratings in all_ratings:
            all_categories.update(ratings.keys())
        
        aggregated = {}
        for category in all_categories:
            values = [ratings.get(category, 0) for ratings in all_ratings if category in ratings]
            if values:
                aggregated[category] = {
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                    "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0
                }
        
        return aggregated
    
    def _assess_performance_level(self, current_value: float, benchmark_value: float) -> str:
        """Assess performance level compared to benchmark."""
        if benchmark_value == 0:
            return "unknown"
        
        percent_difference = ((current_value - benchmark_value) / benchmark_value) * 100
        
        if percent_difference >= 10:
            return "excellent"
        elif percent_difference >= 5:
            return "good"
        elif percent_difference >= -5:
            return "average"
        elif percent_difference >= -10:
            return "below_average"
        else:
            return "poor"
    
    def _calculate_benchmark_score(self, benchmark_comparison: Dict[str, Dict[str, Any]]) -> float:
        """Calculate overall benchmark score."""
        if not benchmark_comparison:
            return 0.0
        
        scores = []
        for comparison in benchmark_comparison.values():
            percent_diff = comparison.get("percent_difference", 0)
            # Convert percent difference to score (0-100)
            score = max(0, min(100, 50 + percent_diff))
            scores.append(score)
        
        return statistics.mean(scores) if scores else 0.0
    
    def _generate_trend_summary(self, ratings: Dict[str, float]) -> Dict[str, Any]:
        """Generate a simplified trend summary."""
        if not ratings:
            return {"summary": "No ratings available"}
        
        overall_score = self._calculate_overall_score(ratings)
        
        return {
            "overall_score": overall_score,
            "score_category": "High" if overall_score >= 4.0 else "Medium" if overall_score >= 3.0 else "Low",
            "rating_count": len(ratings),
            "summary": f"Overall satisfaction score of {overall_score:.1f}/5 based on {len(ratings)} categories"
        }
    
    def _calculate_overall_score(self, ratings: Dict[str, float]) -> float:
        """Calculate overall score from ratings."""
        if not ratings:
            return 0.0
        
        return sum(ratings.values()) / len(ratings)
    
    def validate_survey_data(self, processed_survey: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate processed survey data for formatting requirements.
        
        Args:
            processed_survey: Processed survey data to validate
            
        Returns:
            Validation result
        """
        validation_result = {
            "valid": True,
            "issues": [],
            "warnings": [],
            "quality_score": 0.0
        }
        
        # Check required fields
        required_fields = ["summary_text", "ratings", "metadata"]
        for field in required_fields:
            if field not in processed_survey:
                validation_result["valid"] = False
                validation_result["issues"].append(f"Missing required field: {field}")
        
        # Check ratings
        ratings = processed_survey.get("ratings", {})
        if not ratings:
            validation_result["issues"].append("No ratings data available")
        else:
            # Check rating values
            for category, score in ratings.items():
                if not isinstance(score, (int, float)) or score < 0 or score > 5:
                    validation_result["warnings"].append(f"Invalid rating value for {category}: {score}")
        
        # Check comments
        comments = processed_survey.get("comments", [])
        if not comments:
            validation_result["warnings"].append("No comments provided")
        
        # Check priority score
        priority_score = processed_survey.get("priority_score", 0.0)
        if not isinstance(priority_score, (int, float)) or priority_score < 0 or priority_score > 10:
            validation_result["warnings"].append(f"Invalid priority score: {priority_score}")
        
        # Calculate overall quality score
        metadata = processed_survey.get("metadata", {})
        base_quality = metadata.get("quality_score", 0.0)
        
        # Adjust quality based on validation results
        quality_adjustment = 0.0
        if validation_result["issues"]:
            quality_adjustment -= len(validation_result["issues"]) * 0.2
        if validation_result["warnings"]:
            quality_adjustment -= len(validation_result["warnings"]) * 0.1
        
        validation_result["quality_score"] = max(0.0, min(1.0, base_quality + quality_adjustment))
        
        return validation_result

# Factory function
def create_survey_formatter() -> SurveyFormatter:
    """
    Factory function to create a survey formatter instance.
    
    Returns:
        SurveyFormatter instance
    """
    return SurveyFormatter()

if __name__ == "__main__":
    # Example usage
    formatter = create_survey_formatter()
    
    # Sample processed survey data
    sample_survey = {
        "summary_text": "Customer provided mixed feedback on product quality and customer service. Overall satisfaction is moderate with specific areas for improvement identified.",
        "ratings": {
            "overall_satisfaction": 3.5,
            "product_quality": 3.8,
            "customer_service": 3.2,
            "value_for_money": 3.7,
            "delivery_speed": 4.1
        },
        "comments": [
            "Product quality is good but customer service needs improvement",
            "Delivery was faster than expected",
            "Price is reasonable for the quality received"
        ],
        "improvement_areas": [
            "customer service response time",
            "product packaging",
            "website user experience"
        ],
        "priority_score": 6.5,
        "metadata": {
            "customer_id": "CUST-00003",
            "survey_id": "SURV-001",
            "survey_type": "customer_satisfaction",
            "survey_date": "2023-12-01",
            "quality_score": 0.87
        }
    }
    
    # Validate data
    validation = formatter.validate_survey_data(sample_survey)
    print("Validation result:")
    print(json.dumps(validation, indent=2))
    
    # Format analytics request
    analytics = formatter.format_analytics_request(sample_survey)
    print("\nAnalytics request:")
    print(json.dumps(analytics, indent=2))
    
    # Format for Claude
    claude_formatted = formatter.format_for_claude(sample_survey, "analytics")
    print("\nClaude formatted (analytics):")
    print(json.dumps(claude_formatted, indent=2))