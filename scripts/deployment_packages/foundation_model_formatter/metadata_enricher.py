#!/usr/bin/env python3
"""
Metadata Enrichment Component

Enriches processed data with foundation model-specific metadata including
customer demographics, product information, business context, and temporal data.
"""

import json
import boto3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import logging
import requests
import hashlib

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class MetadataEnricher:
    """
    Enriches data with additional metadata for foundation model processing.
    """
    
    def __init__(self):
        """Initialize the metadata enricher."""
        self.s3_client = boto3.client('s3')
        self.dynamodb_client = boto3.client('dynamodb')
        self.context_cache = {}
        
    def add_business_context(self, data: Dict[str, Any], 
                           context_type: str = "customer_feedback") -> Dict[str, Any]:
        """
        Add business context to data.
        
        Args:
            data: Original data to enrich
            context_type: Type of business context to add
            
        Returns:
            Enriched data with business context
        """
        enriched_data = data.copy()
        
        # Add industry context
        industry_context = self._get_industry_context()
        enriched_data["industry_context"] = industry_context
        
        # Add company context
        company_context = self._get_company_context()
        enriched_data["company_context"] = company_context
        
        # Add business objectives
        business_objectives = self._get_business_objectives(context_type)
        enriched_data["business_objectives"] = business_objectives
        
        # Add competitive landscape
        competitive_context = self._get_competitive_context()
        enriched_data["competitive_context"] = competitive_context
        
        # Add market trends
        market_trends = self._get_market_trends()
        enriched_data["market_trends"] = market_trends
        
        return enriched_data
    
    def enrich_customer_profile(self, data: Dict[str, Any], 
                             customer_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Enhance data with customer history and profile information.
        
        Args:
            data: Original data to enrich
            customer_id: Customer ID for profile lookup
            
        Returns:
            Enriched data with customer profile
        """
        enriched_data = data.copy()
        
        if not customer_id:
            customer_id = data.get("metadata", {}).get("customer_id")
        
        if not customer_id:
            logger.warning("No customer ID provided for profile enrichment")
            return enriched_data
        
        # Get customer profile
        customer_profile = self._get_customer_profile(customer_id)
        enriched_data["customer_profile"] = customer_profile
        
        # Get customer history
        customer_history = self._get_customer_history(customer_id)
        enriched_data["customer_history"] = customer_history
        
        # Get customer segmentation
        customer_segment = self._get_customer_segmentation(customer_id)
        enriched_data["customer_segment"] = customer_segment
        
        # Get customer preferences
        customer_preferences = self._get_customer_preferences(customer_id)
        enriched_data["customer_preferences"] = customer_preferences
        
        # Get customer lifetime value
        clv_data = self._calculate_customer_lifetime_value(customer_id)
        enriched_data["customer_lifetime_value"] = clv_data
        
        return enriched_data
    
    def add_product_context(self, data: Dict[str, Any], 
                          product_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Include product details and specifications.
        
        Args:
            data: Original data to enrich
            product_id: Product ID for product lookup
            
        Returns:
            Enriched data with product context
        """
        enriched_data = data.copy()
        
        if not product_id:
            product_id = data.get("metadata", {}).get("product_id")
        
        if not product_id:
            logger.warning("No product ID provided for product context")
            return enriched_data
        
        # Get product details
        product_details = self._get_product_details(product_id)
        enriched_data["product_details"] = product_details
        
        # Get product category information
        category_info = self._get_product_category_info(product_id)
        enriched_data["product_category"] = category_info
        
        # Get product performance metrics
        product_performance = self._get_product_performance(product_id)
        enriched_data["product_performance"] = product_performance
        
        # Get product reviews summary
        reviews_summary = self._get_product_reviews_summary(product_id)
        enriched_data["product_reviews_summary"] = reviews_summary
        
        # Get product quality metrics
        quality_metrics = self._get_product_quality_metrics(product_id)
        enriched_data["product_quality_metrics"] = quality_metrics
        
        return enriched_data
    
    def create_temporal_context(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add time-based context to data.
        
        Args:
            data: Original data to enrich
            
        Returns:
            Enriched data with temporal context
        """
        enriched_data = data.copy()
        
        # Get current time
        current_time = datetime.now()
        
        # Extract data timestamp
        data_timestamp = self._extract_data_timestamp(data)
        
        if data_timestamp:
            # Calculate time differences
            time_diff = current_time - data_timestamp
            
            temporal_context = {
                "data_timestamp": data_timestamp.isoformat(),
                "current_timestamp": current_time.isoformat(),
                "time_difference_hours": time_diff.total_seconds() / 3600,
                "time_difference_days": time_diff.days,
                "is_recent": time_diff.days <= 30,
                "season": self._get_season(data_timestamp),
                "day_of_week": data_timestamp.strftime("%A"),
                "month": data_timestamp.strftime("%B"),
                "quarter": f"Q{(data_timestamp.month - 1) // 3 + 1}",
                "is_weekend": data_timestamp.weekday() >= 5,
                "is_business_hours": self._is_business_hours(data_timestamp)
            }
        else:
            temporal_context = {
                "current_timestamp": current_time.isoformat(),
                "data_timestamp": None,
                "time_context": "unknown"
            }
        
        enriched_data["temporal_context"] = temporal_context
        
        # Add temporal trends
        temporal_trends = self._get_temporal_trends(data)
        enriched_data["temporal_trends"] = temporal_trends
        
        return enriched_data
    
    def generate_quality_metrics(self, data: Dict[str, Any], 
                              data_type: str) -> Dict[str, Any]:
        """
        Create data quality indicators.
        
        Args:
            data: Original data to analyze
            data_type: Type of data (text, image, audio, survey)
            
        Returns:
            Quality metrics for the data
        """
        quality_metrics = {
            "data_type": data_type,
            "assessment_timestamp": datetime.now().isoformat(),
            "overall_score": 0.0,
            "completeness": 0.0,
            "accuracy": 0.0,
            "consistency": 0.0,
            "timeliness": 0.0,
            "validity": 0.0
        }
        
        # Calculate completeness
        completeness_score = self._calculate_completeness(data, data_type)
        quality_metrics["completeness"] = completeness_score
        
        # Calculate accuracy (based on existing quality scores)
        accuracy_score = self._calculate_accuracy(data, data_type)
        quality_metrics["accuracy"] = accuracy_score
        
        # Calculate consistency
        consistency_score = self._calculate_consistency(data, data_type)
        quality_metrics["consistency"] = consistency_score
        
        # Calculate timeliness
        timeliness_score = self._calculate_timeliness(data)
        quality_metrics["timeliness"] = timeliness_score
        
        # Calculate validity
        validity_score = self._calculate_validity(data, data_type)
        quality_metrics["validity"] = validity_score
        
        # Calculate overall score (weighted average)
        weights = {
            "completeness": 0.25,
            "accuracy": 0.30,
            "consistency": 0.20,
            "timeliness": 0.15,
            "validity": 0.10
        }
        
        overall_score = sum(
            quality_metrics[metric] * weights[metric] 
            for metric in weights.keys()
        )
        quality_metrics["overall_score"] = overall_score
        
        # Add quality recommendations
        quality_metrics["recommendations"] = self._generate_quality_recommendations(
            quality_metrics, data_type
        )
        
        return quality_metrics
    
    def add_sentiment_context(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add sentiment analysis context and trends.
        
        Args:
            data: Original data to enrich
            
        Returns:
            Enriched data with sentiment context
        """
        enriched_data = data.copy()
        
        # Extract current sentiment
        current_sentiment = self._extract_sentiment(data)
        
        # Get historical sentiment trends
        historical_sentiment = self._get_historical_sentiment(data)
        
        # Calculate sentiment trajectory
        sentiment_trajectory = self._calculate_sentiment_trajectory(
            current_sentiment, historical_sentiment
        )
        
        # Add sentiment context
        sentiment_context = {
            "current_sentiment": current_sentiment,
            "historical_trend": historical_sentiment,
            "trajectory": sentiment_trajectory,
            "sentiment_score": current_sentiment.get("Score", 0.0),
            "sentiment_label": current_sentiment.get("Sentiment", "UNKNOWN"),
            "confidence_level": self._get_confidence_level(current_sentiment.get("Score", 0.0))
        }
        
        enriched_data["sentiment_context"] = sentiment_context
        
        return enriched_data
    
    def add_geographic_context(self, data: Dict[str, Any], 
                           location_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Add geographic and regional context.
        
        Args:
            data: Original data to enrich
            location_id: Location identifier
            
        Returns:
            Enriched data with geographic context
        """
        enriched_data = data.copy()
        
        if not location_id:
            location_id = data.get("metadata", {}).get("location_id")
        
        if not location_id:
            logger.warning("No location ID provided for geographic context")
            return enriched_data
        
        # Get location details
        location_details = self._get_location_details(location_id)
        enriched_data["location_details"] = location_details
        
        # Get regional trends
        regional_trends = self._get_regional_trends(location_id)
        enriched_data["regional_trends"] = regional_trends
        
        # Get cultural context
        cultural_context = self._get_cultural_context(location_id)
        enriched_data["cultural_context"] = cultural_context
        
        return enriched_data
    
    def _get_industry_context(self) -> Dict[str, Any]:
        """Get industry-specific context."""
        # In a real implementation, this would fetch from a database or API
        return {
            "industry": "Retail/E-commerce",
            "sector": "Consumer Goods",
            "market_size": "Large",
            "growth_rate": "5.2%",
            "key_trends": [
                "Digital transformation",
                "Customer experience focus",
                "Sustainability initiatives",
                "Personalization"
            ],
            "regulations": [
                "Consumer protection laws",
                "Data privacy regulations",
                "E-commerce compliance"
            ]
        }
    
    def _get_company_context(self) -> Dict[str, Any]:
        """Get company-specific context."""
        return {
            "company_name": "Customer Feedback Corp",
            "company_size": "Medium",
            "business_model": "B2C E-commerce",
            "target_market": "Consumer",
            "value_proposition": "Quality products at competitive prices",
            "strategic_priorities": [
                "Customer satisfaction",
                "Product innovation",
                "Operational efficiency",
                "Market expansion"
            ]
        }
    
    def _get_business_objectives(self, context_type: str) -> Dict[str, Any]:
        """Get business objectives based on context type."""
        objectives = {
            "customer_feedback": {
                "primary_objective": "Improve customer satisfaction",
                "secondary_objectives": [
                    "Reduce customer churn",
                    "Increase product quality",
                    "Enhance service delivery",
                    "Identify improvement opportunities"
                ],
                "success_metrics": [
                    "Customer satisfaction score",
                    "Net Promoter Score (NPS)",
                    "Customer retention rate",
                    "Product quality ratings"
                ]
            },
            "product_analysis": {
                "primary_objective": "Optimize product portfolio",
                "secondary_objectives": [
                    "Identify product gaps",
                    "Improve product features",
                    "Enhance user experience",
                    "Increase product adoption"
                ],
                "success_metrics": [
                    "Product usage metrics",
                    "Feature adoption rate",
                    "User satisfaction scores",
                    "Product return rates"
                ]
            }
        }
        
        return objectives.get(context_type, objectives["customer_feedback"])
    
    def _get_competitive_context(self) -> Dict[str, Any]:
        """Get competitive landscape context."""
        return {
            "market_position": "Top 3",
            "main_competitors": [
                "Competitor A",
                "Competitor B",
                "Competitor C"
            ],
            "competitive_advantages": [
                "Product quality",
                "Customer service",
                "Brand reputation"
            ],
            "competitive_challenges": [
                "Price competition",
                "Market saturation",
                "Technology disruption"
            ]
        }
    
    def _get_market_trends(self) -> List[str]:
        """Get current market trends."""
        return [
            "Increased focus on customer experience",
            "Growing demand for sustainable products",
            "Rise of mobile commerce",
            "Personalization at scale",
            "AI-driven customer insights"
        ]
    
    def _get_customer_profile(self, customer_id: str) -> Dict[str, Any]:
        """Get customer profile from database."""
        # In a real implementation, this would query a customer database
        # For now, return a mock profile based on customer ID
        profile_hash = hashlib.md5(customer_id.encode()).hexdigest()
        
        return {
            "customer_id": customer_id,
            "customer_segment": self._determine_segment(profile_hash),
            "loyalty_tier": self._determine_loyalty_tier(profile_hash),
            "registration_date": "2022-01-15",
            "total_purchases": 15,
            "avg_order_value": 125.50,
            "preferred_channels": ["email", "mobile_app"],
            "communication_preferences": {
                "frequency": "weekly",
                "type": "promotional",
                "time": "morning"
            }
        }
    
    def _get_customer_history(self, customer_id: str) -> Dict[str, Any]:
        """Get customer interaction history."""
        # Mock history data
        return {
            "recent_purchases": [
                {"date": "2023-11-15", "product_id": "PROD-123", "amount": 89.99},
                {"date": "2023-10-20", "product_id": "PROD-456", "amount": 124.50},
                {"date": "2023-09-10", "product_id": "PROD-789", "amount": 67.25}
            ],
            "recent_interactions": [
                {"date": "2023-11-20", "type": "support_call", "duration": 300},
                {"date": "2023-11-10", "type": "website_visit", "pages_viewed": 8},
                {"date": "2023-10-25", "type": "email_open", "campaign": "holiday_sale"}
            ],
            "feedback_history": [
                {"date": "2023-10-15", "rating": 4.2, "sentiment": "POSITIVE"},
                {"date": "2023-09-05", "rating": 3.8, "sentiment": "NEUTRAL"},
                {"date": "2023-08-12", "rating": 4.5, "sentiment": "POSITIVE"}
            ]
        }
    
    def _get_customer_segmentation(self, customer_id: str) -> Dict[str, Any]:
        """Get customer segmentation data."""
        # Mock segmentation
        return {
            "demographic_segment": "Millennial Professional",
            "behavioral_segment": "Frequent Shopper",
            "value_segment": "High Value",
            "lifecycle_stage": "Mature",
            "risk_segment": "Low Risk",
            "engagement_level": "High"
        }
    
    def _get_customer_preferences(self, customer_id: str) -> Dict[str, Any]:
        """Get customer preferences."""
        return {
            "product_preferences": ["electronics", "home_goods", "clothing"],
            "brand_preferences": ["quality_brands", "eco_friendly"],
            "price_sensitivity": "medium",
            "quality_importance": "high",
            "innovation_adoption": "early_adopter",
            "communication_channel": "email",
            "purchase_frequency": "monthly"
        }
    
    def _calculate_customer_lifetime_value(self, customer_id: str) -> Dict[str, Any]:
        """Calculate customer lifetime value."""
        # Mock CLV calculation
        return {
            "current_clv": 2450.75,
            "projected_clv": 3200.00,
            "clv_tier": "high",
            "time_to_payback": 8,  # months
            "retention_probability": 0.85,
            "churn_risk": "low"
        }
    
    def _get_product_details(self, product_id: str) -> Dict[str, Any]:
        """Get product details from catalog."""
        # Mock product details
        return {
            "product_id": product_id,
            "product_name": f"Premium Product {product_id[-3:]}",
            "category": "Electronics",
            "subcategory": "Audio Equipment",
            "brand": "QualityBrand",
            "price": 149.99,
            "launch_date": "2023-01-01",
            "product_lifecycle": "growth",
            "features": [
                "High quality sound",
                "Wireless connectivity",
                "Long battery life",
                "Noise cancellation"
            ]
        }
    
    def _get_product_category_info(self, product_id: str) -> Dict[str, Any]:
        """Get product category information."""
        return {
            "category": "Electronics",
            "subcategory": "Audio Equipment",
            "category_growth_rate": "8.5%",
            "market_share": "12.3%",
            "competition_level": "high",
            "seasonality": "moderate",
            "price_elasticity": "medium"
        }
    
    def _get_product_performance(self, product_id: str) -> Dict[str, Any]:
        """Get product performance metrics."""
        return {
            "sales_volume": 15420,
            "revenue": 2312850.50,
            "profit_margin": "22.5%",
            "return_rate": "3.2%",
            "customer_rating": 4.2,
            "market_position": "#2",
            "inventory_turnover": 8.5,
            "growth_rate": "12.3%"
        }
    
    def _get_product_reviews_summary(self, product_id: str) -> Dict[str, Any]:
        """Get summary of product reviews."""
        return {
            "total_reviews": 342,
            "average_rating": 4.2,
            "rating_distribution": {
                "5_star": 45,
                "4_star": 128,
                "3_star": 89,
                "2_star": 52,
                "1_star": 28
            },
            "common_themes": [
                "Great sound quality",
                "Easy to use",
                "Good battery life",
                "Price concerns",
                "Connectivity issues"
            ],
            "sentiment_breakdown": {
                "positive": 0.72,
                "neutral": 0.18,
                "negative": 0.10
            }
        }
    
    def _get_product_quality_metrics(self, product_id: str) -> Dict[str, Any]:
        """Get product quality metrics."""
        return {
            "defect_rate": "1.2%",
            "quality_score": 4.1,
            "warranty_claims": "2.8%",
            "customer_complaints": 15,
            "quality_trend": "improving",
            "manufacturing_quality": 4.3,
            "reliability_score": 4.0
        }
    
    def _extract_data_timestamp(self, data: Dict[str, Any]) -> Optional[datetime]:
        """Extract timestamp from data."""
        # Check various possible timestamp fields
        timestamp_fields = [
            "timestamp",
            "created_at",
            "date",
            "review_date",
            "survey_date",
            "call_date"
        ]
        
        # Check metadata first
        metadata = data.get("metadata", {})
        for field in timestamp_fields:
            if field in metadata:
                try:
                    return datetime.fromisoformat(metadata[field].replace('Z', '+00:00'))
                except:
                    continue
        
        # Check top-level fields
        for field in timestamp_fields:
            if field in data:
                try:
                    return datetime.fromisoformat(data[field].replace('Z', '+00:00'))
                except:
                    continue
        
        return None
    
    def _get_season(self, date: datetime) -> str:
        """Get season for date."""
        month = date.month
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Fall"
    
    def _is_business_hours(self, date: datetime) -> bool:
        """Check if date is during business hours."""
        # Monday-Friday, 9 AM - 6 PM
        if date.weekday() >= 5:  # Weekend
            return False
        
        return 9 <= date.hour < 18
    
    def _get_temporal_trends(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get temporal trends related to data."""
        # Mock temporal trends
        return {
            "seasonal_pattern": "higher_activity_in_holidays",
            "day_of_week_pattern": "peak_on_mondays",
            "time_of_day_pattern": "peak_afternoon",
            "monthly_trend": "steady_growth",
            "yearly_trend": "cyclical_with_peaks"
        }
    
    def _calculate_completeness(self, data: Dict[str, Any], data_type: str) -> float:
        """Calculate data completeness score."""
        # Define required fields for each data type
        required_fields = {
            "text": ["original_text", "sentiment", "metadata"],
            "image": ["extracted_text", "labels", "metadata"],
            "audio": ["transcript", "sentiment", "metadata"],
            "survey": ["summary_text", "ratings", "metadata"]
        }
        
        if data_type not in required_fields:
            return 0.0
        
        required = required_fields[data_type]
        present = sum(1 for field in required if field in data and data[field])
        
        return present / len(required)
    
    def _calculate_accuracy(self, data: Dict[str, Any], data_type: str) -> float:
        """Calculate data accuracy score."""
        # Use existing quality scores if available
        metadata = data.get("metadata", {})
        base_quality = metadata.get("quality_score", 0.0)
        
        # Adjust based on data type specific indicators
        if data_type == "text":
            sentiment_score = data.get("sentiment", {}).get("Score", 0.0)
            return (base_quality + sentiment_score) / 2
        elif data_type == "image":
            label_confidence = sum(label.get("Confidence", 0) for label in data.get("labels", []))
            avg_confidence = label_confidence / len(data.get("labels", [1])) if data.get("labels") else 0
            return (base_quality + avg_confidence) / 2
        else:
            return base_quality
    
    def _calculate_consistency(self, data: Dict[str, Any], data_type: str) -> float:
        """Calculate data consistency score."""
        # Simplified consistency check
        consistency_indicators = 0
        total_checks = 0
        
        # Check metadata consistency
        metadata = data.get("metadata", {})
        if metadata.get("customer_id") and metadata.get("product_id"):
            consistency_indicators += 1
        total_checks += 1
        
        # Check data type specific consistency
        if data_type == "text":
            if data.get("sentiment") and data.get("key_phrases"):
                consistency_indicators += 1
            total_checks += 1
        elif data_type == "image":
            if data.get("labels") and data.get("extracted_text"):
                consistency_indicators += 1
            total_checks += 1
        
        return consistency_indicators / total_checks if total_checks > 0 else 0.0
    
    def _calculate_timeliness(self, data: Dict[str, Any]) -> float:
        """Calculate data timeliness score."""
        # Check how recent the data is
        timestamp = self._extract_data_timestamp(data)
        if not timestamp:
            return 0.5  # Neutral score
        
        days_old = (datetime.now() - timestamp).days
        
        if days_old <= 1:
            return 1.0  # Very recent
        elif days_old <= 7:
            return 0.8  # Recent
        elif days_old <= 30:
            return 0.6  # Moderately recent
        elif days_old <= 90:
            return 0.4  # Old
        else:
            return 0.2  # Very old
    
    def _calculate_validity(self, data: Dict[str, Any], data_type: str) -> float:
        """Calculate data validity score."""
        validity_score = 1.0
        
        # Check for invalid data indicators
        if data_type == "text":
            text = data.get("original_text", "")
            if len(text) < 10:
                validity_score -= 0.3
            elif len(text) > 50000:
                validity_score -= 0.2
        
        elif data_type == "image":
            if not data.get("labels"):
                validity_score -= 0.4
        
        elif data_type == "audio":
            transcript = data.get("transcript", "")
            if len(transcript) < 50:
                validity_score -= 0.3
        
        elif data_type == "survey":
            ratings = data.get("ratings", {})
            if not ratings:
                validity_score -= 0.5
        
        return max(0.0, validity_score)
    
    def _generate_quality_recommendations(self, quality_metrics: Dict[str, Any], 
                                     data_type: str) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        if quality_metrics["completeness"] < 0.8:
            recommendations.append(f"Improve data completeness for {data_type} - missing required fields")
        
        if quality_metrics["accuracy"] < 0.7:
            recommendations.append("Enhance data accuracy through validation and verification")
        
        if quality_metrics["consistency"] < 0.7:
            recommendations.append("Improve data consistency across different fields")
        
        if quality_metrics["timeliness"] < 0.6:
            recommendations.append("Reduce data processing latency for more timely insights")
        
        if quality_metrics["validity"] < 0.7:
            recommendations.append(f"Address data validity issues specific to {data_type}")
        
        return recommendations
    
    def _extract_sentiment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract sentiment information from data."""
        # Check various sentiment fields
        sentiment_fields = ["sentiment", "overall_sentiment", "sentiment_analysis"]
        
        for field in sentiment_fields:
            if field in data:
                return data[field]
        
        # Check in metadata
        metadata = data.get("metadata", {})
        for field in sentiment_fields:
            if field in metadata:
                return metadata[field]
        
        return {"Sentiment": "UNKNOWN", "Score": 0.0}
    
    def _get_historical_sentiment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get historical sentiment trends."""
        # Mock historical sentiment
        return {
            "last_7_days": {
                "average": 0.75,
                "trend": "improving"
            },
            "last_30_days": {
                "average": 0.68,
                "trend": "stable"
            },
            "last_90_days": {
                "average": 0.72,
                "trend": "improving"
            }
        }
    
    def _calculate_sentiment_trajectory(self, current: Dict[str, Any], 
                                   historical: Dict[str, Any]) -> str:
        """Calculate sentiment trajectory."""
        current_score = current.get("Score", 0.0)
        historical_avg = historical.get("last_30_days", {}).get("average", 0.0)
        
        if current_score > historical_avg + 0.1:
            return "improving"
        elif current_score < historical_avg - 0.1:
            return "declining"
        else:
            return "stable"
    
    def _get_confidence_level(self, score: float) -> str:
        """Get confidence level from score."""
        if score >= 0.8:
            return "high"
        elif score >= 0.6:
            return "medium"
        elif score >= 0.4:
            return "low"
        else:
            return "very_low"
    
    def _determine_segment(self, profile_hash: str) -> str:
        """Determine customer segment from hash."""
        segments = ["Value Conscious", "Quality Focused", "Brand Loyal", "Price Sensitive"]
        index = int(profile_hash[0], 16) % len(segments)
        return segments[index]
    
    def _determine_loyalty_tier(self, profile_hash: str) -> str:
        """Determine loyalty tier from hash."""
        tiers = ["Bronze", "Silver", "Gold", "Platinum"]
        index = int(profile_hash[1], 16) % len(tiers)
        return tiers[index]
    
    def _get_location_details(self, location_id: str) -> Dict[str, Any]:
        """Get location details."""
        # Mock location data
        return {
            "location_id": location_id,
            "country": "United States",
            "region": "North America",
            "state": "California",
            "city": "San Francisco",
            "timezone": "PST",
            "cultural_context": "western_individualistic",
            "economic_profile": "high_income"
        }
    
    def _get_regional_trends(self, location_id: str) -> Dict[str, Any]:
        """Get regional trends."""
        return {
            "market_trends": [
                "Increased online shopping",
                "Focus on sustainability",
                "Demand for fast delivery"
            ],
            "consumer_preferences": [
                "Quality over price",
                "Digital experiences",
                "Personalized products"
            ],
            "competitive_landscape": "high_competition"
        }
    
    def _get_cultural_context(self, location_id: str) -> Dict[str, Any]:
        """Get cultural context."""
        return {
            "communication_style": "direct",
            "decision_factors": ["quality", "reviews", "brand_reputation"],
            "service_expectations": "high",
            "price_sensitivity": "medium",
            "cultural_values": ["innovation", "efficiency", "sustainability"]
        }

# Factory function
def create_metadata_enricher() -> MetadataEnricher:
    """
    Factory function to create a metadata enricher instance.
    
    Returns:
        MetadataEnricher instance
    """
    return MetadataEnricher()

if __name__ == "__main__":
    # Example usage
    enricher = create_metadata_enricher()
    
    # Sample data
    sample_data = {
        "original_text": "This product is amazing! Great quality and fast shipping.",
        "sentiment": {"Sentiment": "POSITIVE", "Score": 0.92},
        "metadata": {
            "customer_id": "CUST-00001",
            "product_id": "PROD-12345",
            "review_date": "2023-12-01",
            "quality_score": 0.91
        }
    }
    
    # Add business context
    enriched_data = enricher.add_business_context(sample_data)
    print("Business context added:")
    print(json.dumps(enriched_data.get("business_context", {}), indent=2))
    
    # Enrich customer profile
    customer_enriched = enricher.enrich_customer_profile(sample_data)
    print("\nCustomer profile enriched:")
    print(json.dumps(customer_enriched.get("customer_profile", {}), indent=2))
    
    # Generate quality metrics
    quality_metrics = enricher.generate_quality_metrics(sample_data, "text")
    print("\nQuality metrics:")
    print(json.dumps(quality_metrics, indent=2))