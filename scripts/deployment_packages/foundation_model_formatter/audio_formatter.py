#!/usr/bin/env python3
"""
Audio Data Formatter

Specialized formatter for audio transcript data that supports conversational formats 
for audio data and includes speaker diarization metadata integration with 
Phase 2 audio processing output.
"""

import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
import boto3
from typing import Dict, List, Any, Optional, Union
import logging
import re

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class AudioFormatter:
    """
    Specialized formatter for audio data with conversation analysis capabilities.
    """
    
    def __init__(self):
        """Initialize audio formatter."""
        self.s3_client = boto3.client('s3')
        self.templates = self._load_templates()
        
    def _load_templates(self) -> Dict[str, str]:
        """Load audio formatting templates."""
        return {
            "claude_conversation": """You are analyzing customer service call transcripts to provide business insights.

CALL TRANSCRIPT:
{transcript}

SPEAKER INFORMATION:
{speaker_info}

ANALYSIS DATA:
- Overall Sentiment: {sentiment} (confidence: {sentiment_score})
- Key Phrases: {key_phrases}
- Entities: {entities}
- Call Duration: {duration} minutes
- Language: {language}

METADATA:
- Call ID: {call_id}
- Customer ID: {customer_id}
- Call Date: {call_date}
- Quality Score: {quality_score}

Please provide:
1. Summary of the call conversation
2. Customer satisfaction assessment
3. Key issues or concerns raised
4. Service quality evaluation
5. Improvement recommendations
6. Follow-up actions needed

Format your response as structured JSON with the following keys:
call_summary, satisfaction_assessment, key_issues, service_quality, improvements, follow_up_actions""",
            
            "titan_analysis": """Analyze this customer service call transcript:

Transcript: {transcript}
Speakers: {speaker_info}
Sentiment: {sentiment}
Duration: {duration} minutes

Provide analysis in JSON format with: summary, satisfaction_level, issues, service_rating, action_items.""",
            
            "training_prompt": """Customer service call analysis:
Transcript: {transcript}
Speakers: {speakers}
Sentiment: {sentiment}
Duration: {duration} minutes""",
            
            "training_completion": """{call_summary}. Customer satisfaction: {satisfaction_level}. Key issues: {issues}. Service quality: {service_quality}."""
        }
    
    def format_for_claude(self, processed_audio: Dict[str, Any], 
                         format_type: str = "conversation") -> Dict[str, Any]:
        """
        Format processed audio data for Claude models.
        
        Args:
            processed_audio: Processed audio data from Phase 2
            format_type: Output format (conversation, jsonl, parquet)
            
        Returns:
            Formatted data for Claude
        """
        if format_type == "conversation":
            return self._format_claude_conversation(processed_audio)
        elif format_type == "jsonl":
            return self._format_jsonl(processed_audio, "claude")
        elif format_type == "parquet":
            return self._format_parquet(processed_audio, "claude")
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def format_for_titan(self, processed_audio: Dict[str, Any],
                        format_type: str = "json") -> Dict[str, Any]:
        """
        Format processed audio data for Titan models.
        
        Args:
            processed_audio: Processed audio data from Phase 2
            format_type: Output format (json, jsonl, parquet)
            
        Returns:
            Formatted data for Titan
        """
        if format_type == "json":
            return self._format_titan_json(processed_audio)
        elif format_type == "jsonl":
            return self._format_jsonl(processed_audio, "titan")
        elif format_type == "parquet":
            return self._format_parquet(processed_audio, "titan")
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def format_for_training(self, processed_audio: Dict[str, Any],
                           output_format: str = "jsonl") -> Dict[str, Any]:
        """
        Format processed audio data for model training.
        
        Args:
            processed_audio: Processed audio data from Phase 2
            output_format: Output format (jsonl, parquet)
            
        Returns:
            Training data format
        """
        if output_format == "jsonl":
            return self._format_training_jsonl(processed_audio)
        elif output_format == "parquet":
            return self._format_training_parquet(processed_audio)
        else:
            raise ValueError(f"Unsupported training format: {output_format}")
    
    def format_conversation(self, processed_audio: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format transcript as structured conversation.
        
        Args:
            processed_audio: Processed audio data from Phase 2
            
        Returns:
            Structured conversation format
        """
        transcript = processed_audio.get("transcript", "")
        speakers = processed_audio.get("speakers", [])
        
        # Create conversation segments
        conversation_segments = self._segment_conversation(transcript, speakers)
        
        # Identify speakers
        speaker_info = self._analyze_speakers(speakers)
        
        return {
            "conversation_segments": conversation_segments,
            "speaker_info": speaker_info,
            "total_segments": len(conversation_segments),
            "conversation_summary": self._summarize_conversation(conversation_segments)
        }
    
    def add_speaker_context(self, processed_audio: Dict[str, Any], 
                           speaker_profiles: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add speaker identification and context to audio data.
        
        Args:
            processed_audio: Processed audio data
            speaker_profiles: Speaker profile information
            
        Returns:
            Enhanced audio data with speaker context
        """
        enhanced_data = processed_audio.copy()
        
        speakers = processed_audio.get("speakers", [])
        
        # Enhance each speaker with profile information
        enhanced_speakers = []
        for speaker in speakers:
            speaker_id = speaker.get("Speaker", "unknown")
            profile = speaker_profiles.get(speaker_id, {})
            
            enhanced_speaker = speaker.copy()
            enhanced_speaker.update({
                "role": profile.get("role", "unknown"),
                "department": profile.get("department", ""),
                "experience_level": profile.get("experience_level", ""),
                "language_preference": profile.get("language_preference", ""),
                "customer_interaction_style": profile.get("customer_interaction_style", "")
            })
            
            enhanced_speakers.append(enhanced_speaker)
        
        enhanced_data["speakers"] = enhanced_speakers
        enhanced_data["speaker_profiles"] = speaker_profiles
        
        return enhanced_data
    
    def segment_dialog(self, processed_audio: Dict[str, Any], 
                     max_segment_length: int = 300) -> List[Dict[str, Any]]:
        """
        Break long conversations into manageable segments.
        
        Args:
            processed_audio: Processed audio data
            max_segment_length: Maximum segment length in seconds
            
        Returns:
            List of conversation segments
        """
        transcript = processed_audio.get("transcript", "")
        duration = processed_audio.get("metadata", {}).get("duration", 0)
        
        if duration <= max_segment_length:
            return [processed_audio]
        
        # Calculate number of segments needed
        num_segments = int(duration / max_segment_length) + 1
        
        # Split transcript into segments
        words = transcript.split()
        words_per_segment = len(words) // num_segments
        
        segments = []
        for i in range(num_segments):
            start_idx = i * words_per_segment
            end_idx = start_idx + words_per_segment if i < num_segments - 1 else len(words)
            
            segment_words = words[start_idx:end_idx]
            segment_transcript = " ".join(segment_words)
            
            segment_start_time = i * max_segment_length
            segment_end_time = min((i + 1) * max_segment_length, duration)
            
            segment = {
                "segment_id": i + 1,
                "transcript": segment_transcript,
                "start_time": segment_start_time,
                "end_time": segment_end_time,
                "duration": segment_end_time - segment_start_time,
                "metadata": processed_audio.get("metadata", {})
            }
            
            segments.append(segment)
        
        return segments
    
    def generate_dialog_summary(self, processed_audio: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create conversation context and summary.
        
        Args:
            processed_audio: Processed audio data
            
        Returns:
            Dialog summary and context
        """
        transcript = processed_audio.get("transcript", "")
        sentiment = processed_audio.get("sentiment", {})
        key_phrases = processed_audio.get("key_phrases", [])
        entities = processed_audio.get("entities", [])
        metadata = processed_audio.get("metadata", {})
        
        # Extract key themes
        themes = self._extract_conversation_themes(key_phrases, entities)
        
        # Analyze conversation flow
        conversation_flow = self._analyze_conversation_flow(transcript)
        
        # Identify action items
        action_items = self._extract_action_items(transcript, entities)
        
        return {
            "themes": themes,
            "conversation_flow": conversation_flow,
            "action_items": action_items,
            "sentiment_trajectory": self._analyze_sentiment_trajectory(transcript),
            "key_moments": self._identify_key_moments(transcript, sentiment),
            "summary_metadata": {
                "duration_minutes": metadata.get("duration", 0) / 60,
                "speaker_count": len(processed_audio.get("speakers", [])),
                "language": metadata.get("language_code", "unknown"),
                "overall_sentiment": sentiment.get("Sentiment", "unknown")
            }
        }
    
    def _format_claude_conversation(self, processed_audio: Dict[str, Any]) -> Dict[str, Any]:
        """Format audio data as Claude conversation."""
        
        # Extract data
        transcript = processed_audio.get("transcript", "")
        speakers = processed_audio.get("speakers", [])
        sentiment = processed_audio.get("sentiment", {})
        key_phrases = processed_audio.get("key_phrases", [])
        entities = processed_audio.get("entities", [])
        metadata = processed_audio.get("metadata", {})
        
        # Format speaker information
        speaker_info = json.dumps(speakers, indent=2)
        
        # Format sentiment
        sentiment_text = sentiment.get("Sentiment", "Unknown")
        sentiment_score = sentiment.get("Score", 0.0)
        
        # Format key phrases
        key_phrases_text = ", ".join([phrase.get("Text", "") for phrase in key_phrases])
        
        # Format entities
        entities_text = ", ".join([entity.get("Text", "") for entity in entities])
        
        # Extract metadata
        call_id = metadata.get("call_id", "N/A")
        customer_id = metadata.get("customer_id", "N/A")
        call_date = metadata.get("call_date", "N/A")
        duration = metadata.get("duration", 0)
        language = metadata.get("language_code", "N/A")
        quality_score = metadata.get("quality_score", 0.0)
        
        # Create conversation prompt
        prompt = self.templates["claude_conversation"].format(
            transcript=transcript,
            speaker_info=speaker_info,
            sentiment=sentiment_text,
            sentiment_score=sentiment_score,
            key_phrases=key_phrases_text,
            entities=entities_text,
            duration=duration,
            language=language,
            call_id=call_id,
            customer_id=customer_id,
            call_date=call_date,
            quality_score=quality_score
        )
        
        # Create Claude conversation format
        conversation = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 3000,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "top_p": 0.999,
            "top_k": 250
        }
        
        # Add metadata
        conversation["metadata"] = {
            "data_type": "audio",
            "model": "claude-v2",
            "format_timestamp": datetime.now().isoformat(),
            "source_data_id": metadata.get("id", call_id),
            "call_duration": duration,
            "language": language,
            "quality_score": quality_score
        }
        
        return conversation
    
    def _format_titan_json(self, processed_audio: Dict[str, Any]) -> Dict[str, Any]:
        """Format audio data for Titan Text model."""
        
        transcript = processed_audio.get("transcript", "")
        speakers = processed_audio.get("speakers", [])
        sentiment = processed_audio.get("sentiment", {})
        metadata = processed_audio.get("metadata", {})
        
        # Format for Titan
        speaker_info = ", ".join([f"{sp.get('Speaker', 'Unknown')}: {sp.get('Timestamp', '')}" 
                                for sp in speakers])
        sentiment_text = sentiment.get("Sentiment", "Unknown")
        duration = metadata.get("duration", 0)
        
        # Create Titan prompt
        prompt = self.templates["titan_analysis"].format(
            transcript=transcript,
            speaker_info=speaker_info,
            sentiment=sentiment_text,
            duration=duration
        )
        
        titan_request = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 3000,
                "temperature": 0.7,
                "topP": 0.999,
                "stopSequences": []
            }
        }
        
        # Add metadata
        titan_request["metadata"] = {
            "data_type": "audio",
            "model": "titan-text-express-v1",
            "format_timestamp": datetime.now().isoformat(),
            "source_data_id": metadata.get("id", metadata.get("call_id", "unknown")),
            "call_duration": duration,
            "language": metadata.get("language_code", "unknown"),
            "quality_score": metadata.get("quality_score", 0.0)
        }
        
        return titan_request
    
    def _format_jsonl(self, processed_audio: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Format audio data as JSONL for training."""
        
        transcript = processed_audio.get("transcript", "")
        speakers = processed_audio.get("speakers", [])
        sentiment = processed_audio.get("sentiment", {})
        metadata = processed_audio.get("metadata", {})
        
        # Create training prompt
        speakers_text = ", ".join([sp.get("Speaker", "Unknown") for sp in speakers])
        sentiment_text = sentiment.get("Sentiment", "Unknown")
        duration = metadata.get("duration", 0)
        
        prompt = self.templates["training_prompt"].format(
            transcript=transcript,
            speakers=speakers_text,
            sentiment=sentiment_text,
            duration=duration
        )
        
        # Generate completion
        completion = self.templates["training_completion"].format(
            call_summary="Customer service call discussed product issues",
            satisfaction_level="Medium",
            issues="Product quality concerns",
            service_quality="Agent was helpful but needs more training"
        )
        
        jsonl_record = {
            "prompt": prompt,
            "completion": completion,
            "data_type": "audio",
            "model": model,
            "quality_score": metadata.get("quality_score", 0.0),
            "customer_id": metadata.get("customer_id", ""),
            "call_id": metadata.get("call_id", ""),
            "duration": duration,
            "language": metadata.get("language_code", ""),
            "sentiment": sentiment_text,
            "timestamp": datetime.now().isoformat()
        }
        
        return {"jsonl_line": json.dumps(jsonl_record)}
    
    def _format_parquet(self, processed_audio: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Format audio data as Parquet for analytics."""
        
        transcript = processed_audio.get("transcript", "")
        speakers = processed_audio.get("speakers", [])
        sentiment = processed_audio.get("sentiment", {})
        key_phrases = processed_audio.get("key_phrases", [])
        entities = processed_audio.get("entities", [])
        metadata = processed_audio.get("metadata", {})
        
        # Extract key information
        key_phrases_text = ", ".join([phrase.get("Text", "") for phrase in key_phrases])
        entities_text = ", ".join([entity.get("Text", "") for entity in entities])
        speakers_text = ", ".join([sp.get("Speaker", "Unknown") for sp in speakers])
        
        # Create DataFrame
        df_data = {
            "transcript": [transcript],
            "speakers": [speakers_text],
            "sentiment": [sentiment.get("Sentiment", "")],
            "sentiment_score": [sentiment.get("Score", 0.0)],
            "key_phrases": [key_phrases_text],
            "entities": [entities_text],
            "data_type": ["audio"],
            "model": [model],
            "quality_score": [metadata.get("quality_score", 0.0)],
            "customer_id": [metadata.get("customer_id", "")],
            "call_id": [metadata.get("call_id", "")],
            "duration": [metadata.get("duration", 0)],
            "language": [metadata.get("language_code", "")],
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
    
    def _format_training_jsonl(self, processed_audio: Dict[str, Any]) -> Dict[str, Any]:
        """Format audio data specifically for training."""
        
        transcript = processed_audio.get("transcript", "")
        speakers = processed_audio.get("speakers", [])
        sentiment = processed_audio.get("sentiment", {})
        metadata = processed_audio.get("metadata", {})
        
        # Create fine-tuning format
        training_record = {
            "instruction": "Analyze this customer service call transcript and provide business insights.",
            "input": f"""Transcript: {transcript}
Speakers: {', '.join([sp.get('Speaker', 'Unknown') for sp in speakers])}
Sentiment: {sentiment.get('Sentiment', 'Unknown')}
Duration: {metadata.get('duration', 0)} minutes""",
            "output": f"""Call Summary: Customer service interaction
Satisfaction Level: {sentiment.get('Sentiment', 'Unknown')}
Key Issues: Product/service concerns
Service Quality: {metadata.get('quality_score', 0.0)}""",
            "data_type": "audio_transcript",
            "quality_score": metadata.get("quality_score", 0.0)
        }
        
        return {"jsonl_line": json.dumps(training_record)}
    
    def _format_training_parquet(self, processed_audio: Dict[str, Any]) -> Dict[str, Any]:
        """Format audio data as Parquet for training."""
        
        transcript = processed_audio.get("transcript", "")
        speakers = processed_audio.get("speakers", [])
        sentiment = processed_audio.get("sentiment", {})
        metadata = processed_audio.get("metadata", {})
        
        # Create training DataFrame
        df_data = {
            "instruction": ["Analyze this customer service call transcript and provide business insights."],
            "input": [f"""Transcript: {transcript}
Speakers: {', '.join([sp.get('Speaker', 'Unknown') for sp in speakers])}
Sentiment: {sentiment.get('Sentiment', 'Unknown')}
Duration: {metadata.get('duration', 0)} minutes"""],
            "output": [f"""Call Summary: Customer service interaction
Satisfaction Level: {sentiment.get('Sentiment', 'Unknown')}
Key Issues: Product/service concerns
Service Quality: {metadata.get('quality_score', 0.0)}"""],
            "data_type": ["audio_transcript"],
            "quality_score": [metadata.get("quality_score", 0.0)],
            "customer_id": [metadata.get("customer_id", "")],
            "call_id": [metadata.get("call_id", "")],
            "duration": [metadata.get("duration", 0)],
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
    
    def _segment_conversation(self, transcript: str, speakers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create conversation segments from transcript and speaker info."""
        segments = []
        
        # Simple segmentation based on speaker changes
        # In a real implementation, this would use timestamps
        current_speaker = None
        current_text = []
        
        # Split transcript into sentences for better segmentation
        sentences = re.split(r'[.!?]+', transcript)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Determine speaker (simplified - in reality would use speaker diarization)
            speaker = speakers[0].get("Speaker", "Unknown") if speakers else "Unknown"
            
            if current_speaker != speaker:
                # Save previous segment
                if current_text:
                    segments.append({
                        "speaker": current_speaker,
                        "text": " ".join(current_text),
                        "timestamp": None  # Would be actual timestamp
                    })
                
                # Start new segment
                current_speaker = speaker
                current_text = [sentence]
            else:
                current_text.append(sentence)
        
        # Add final segment
        if current_text:
            segments.append({
                "speaker": current_speaker,
                "text": " ".join(current_text),
                "timestamp": None
            })
        
        return segments
    
    def _analyze_speakers(self, speakers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze speaker information."""
        speaker_count = len(speakers)
        speaker_ids = [sp.get("Speaker", "Unknown") for sp in speakers]
        
        return {
            "speaker_count": speaker_count,
            "speaker_ids": speaker_ids,
            "dominant_speaker": speaker_ids[0] if speaker_ids else "Unknown",
            "speaker_distribution": {sid: speaker_ids.count(sid) for sid in set(speaker_ids)}
        }
    
    def _summarize_conversation(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize conversation segments."""
        if not segments:
            return {"total_segments": 0, "summary": "No conversation segments found"}
        
        speaker_contributions = {}
        for segment in segments:
            speaker = segment.get("speaker", "Unknown")
            text_length = len(segment.get("text", ""))
            
            if speaker not in speaker_contributions:
                speaker_contributions[speaker] = 0
            speaker_contributions[speaker] += text_length
        
        return {
            "total_segments": len(segments),
            "speaker_contributions": speaker_contributions,
            "average_segment_length": sum(len(s.get("text", "")) for s in segments) / len(segments)
        }
    
    def _extract_conversation_themes(self, key_phrases: List[Dict[str, Any]], 
                                   entities: List[Dict[str, Any]]) -> List[str]:
        """Extract key themes from conversation."""
        themes = []
        
        # Extract from key phrases
        for phrase in key_phrases:
            text = phrase.get("Text", "").lower()
            if "customer service" in text or "support" in text:
                themes.append("Customer Service")
            elif "product" in text or "quality" in text:
                themes.append("Product Quality")
            elif "billing" in text or "price" in text:
                themes.append("Billing/Pricing")
            elif "delivery" in text or "shipping" in text:
                themes.append("Delivery/Shipping")
        
        # Extract from entities
        for entity in entities:
            entity_type = entity.get("Type", "").upper()
            if entity_type == "PRODUCT":
                themes.append("Product")
            elif entity_type == "ORGANIZATION":
                themes.append("Company")
            elif entity_type == "LOCATION":
                themes.append("Location")
        
        return list(set(themes))
    
    def _analyze_conversation_flow(self, transcript: str) -> Dict[str, Any]:
        """Analyze the flow of conversation."""
        # Simplified analysis - in reality would use more sophisticated NLP
        words = transcript.lower().split()
        
        # Identify transitions
        transitions = 0
        transition_words = ["but", "however", "although", "meanwhile", "also", "additionally"]
        for word in transition_words:
            transitions += words.count(word)
        
        # Identify questions
        questions = transcript.count("?")
        
        return {
            "total_words": len(words),
            "transition_count": transitions,
            "question_count": questions,
            "conversation_complexity": "High" if transitions > 5 else "Medium" if transitions > 2 else "Low"
        }
    
    def _extract_action_items(self, transcript: str, entities: List[Dict[str, Any]]) -> List[str]:
        """Extract action items from conversation."""
        action_items = []
        
        # Look for action-oriented phrases
        action_patterns = [
            r"will\s+(\w+)",
            r"going\s+to\s+(\w+)",
            r"need\s+to\s+(\w+)",
            r"should\s+(\w+)",
            r"follow\s+up\s+on"
        ]
        
        for pattern in action_patterns:
            matches = re.findall(pattern, transcript.lower())
            for match in matches:
                action_items.append(f"Action: {match}")
        
        return action_items
    
    def _analyze_sentiment_trajectory(self, transcript: str) -> Dict[str, Any]:
        """Analyze sentiment changes throughout conversation."""
        # Simplified sentiment trajectory analysis
        # In reality, would use sentiment analysis on segments
        
        # Look for sentiment indicators
        positive_words = ["good", "great", "excellent", "happy", "satisfied", "thank"]
        negative_words = ["bad", "terrible", "awful", "unhappy", "dissatisfied", "problem", "issue"]
        
        transcript_lower = transcript.lower()
        positive_count = sum(transcript_lower.count(word) for word in positive_words)
        negative_count = sum(transcript_lower.count(word) for word in negative_words)
        
        if positive_count > negative_count:
            trajectory = "Improving"
        elif negative_count > positive_count:
            trajectory = "Declining"
        else:
            trajectory = "Stable"
        
        return {
            "trajectory": trajectory,
            "positive_indicators": positive_count,
            "negative_indicators": negative_count,
            "sentiment_balance": positive_count - negative_count
        }
    
    def _identify_key_moments(self, transcript: str, sentiment: Dict[str, Any]) -> List[str]:
        """Identify key moments in conversation."""
        key_moments = []
        
        # Look for emphasis indicators
        emphasis_patterns = [
            r"important\s+to\s+(\w+)",
            r"crucial\s+(\w+)",
            r"critical\s+(\w+)",
            r"main\s+(\w+)"
        ]
        
        for pattern in emphasis_patterns:
            matches = re.findall(pattern, transcript.lower())
            for match in matches:
                key_moments.append(f"Key moment: {match}")
        
        # Add sentiment-based moments
        if sentiment.get("Sentiment") == "NEGATIVE":
            key_moments.append("Negative sentiment expressed")
        elif sentiment.get("Sentiment") == "POSITIVE":
            key_moments.append("Positive resolution achieved")
        
        return key_moments
    
    def validate_audio_data(self, processed_audio: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate processed audio data for formatting requirements.
        
        Args:
            processed_audio: Processed audio data to validate
            
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
        required_fields = ["transcript", "sentiment", "speakers", "metadata"]
        for field in required_fields:
            if field not in processed_audio:
                validation_result["valid"] = False
                validation_result["issues"].append(f"Missing required field: {field}")
        
        # Check transcript quality
        transcript = processed_audio.get("transcript", "")
        if len(transcript) < 50:
            validation_result["warnings"].append("Transcript is very short (< 50 characters)")
        elif len(transcript) > 50000:
            validation_result["warnings"].append("Transcript is very long (> 50,000 characters)")
        
        # Check speaker information
        speakers = processed_audio.get("speakers", [])
        if not speakers:
            validation_result["warnings"].append("No speaker information available")
        
        # Check duration
        metadata = processed_audio.get("metadata", {})
        duration = metadata.get("duration", 0)
        if duration < 10:
            validation_result["warnings"].append("Very short audio duration (< 10 seconds)")
        elif duration > 3600:
            validation_result["warnings"].append("Very long audio duration (> 1 hour)")
        
        # Calculate overall quality score
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
def create_audio_formatter() -> AudioFormatter:
    """
    Factory function to create an audio formatter instance.
    
    Returns:
        AudioFormatter instance
    """
    return AudioFormatter()

if __name__ == "__main__":
    # Example usage
    formatter = create_audio_formatter()
    
    # Sample processed audio data
    sample_audio = {
        "transcript": "Customer: Hi, I'm having issues with my recent order. Agent: I'm sorry to hear that. Can you provide your order number? Customer: Yes, it's ORD-12345. The product arrived damaged. Agent: I apologize for the inconvenience. I'll process a replacement for you right away.",
        "speakers": [
            {"Speaker": "spk_0", "Timestamp": "00:00:05"},
            {"Speaker": "spk_1", "Timestamp": "00:00:12"},
            {"Speaker": "spk_0", "Timestamp": "00:00:18"},
            {"Speaker": "spk_1", "Timestamp": "00:00:25"}
        ],
        "sentiment": {"Sentiment": "NEGATIVE", "Score": 0.65},
        "key_phrases": [
            {"Text": "order issues", "Score": 0.88},
            {"Text": "damaged product", "Score": 0.92}
        ],
        "entities": [
            {"Text": "ORD-12345", "Type": "ORDER_ID", "Score": 0.95},
            {"Text": "replacement", "Type": "ACTION", "Score": 0.87}
        ],
        "metadata": {
            "call_id": "CALL-001",
            "customer_id": "CUST-00004",
            "duration": 300,
            "language_code": "en-US",
            "quality_score": 0.91
        }
    }
    
    # Validate data
    validation = formatter.validate_audio_data(sample_audio)
    print("Validation result:")
    print(json.dumps(validation, indent=2))
    
    # Format conversation
    conversation = formatter.format_conversation(sample_audio)
    print("\nFormatted conversation:")
    print(json.dumps(conversation, indent=2))
    
    # Format for Claude
    claude_formatted = formatter.format_for_claude(sample_audio, "conversation")
    print("\nClaude formatted (conversation):")
    print(json.dumps(claude_formatted, indent=2))