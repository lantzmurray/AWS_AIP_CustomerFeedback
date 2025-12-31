
import sys
import os
import json
import boto3
from unittest.mock import Mock, patch, MagicMock
from botocore.exceptions import ClientError

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_audio_processing_pipeline():
    """Run integration test for audio_processing_pipeline"""
    try:
        # Import the components to test
        from multimodal_processing.text_processing_lambda import lambda_handler as text_handler
        from multimodal_processing.image_processing_lambda import lambda_handler as image_handler
        from multimodal_processing.audio_processing_lambda import lambda_handler as audio_handler
        
        # Mock AWS services
        with patch('boto3.client') as mock_boto_client:
            # Mock S3 client
            mock_s3 = Mock()
            mock_boto_client.return_value = mock_s3
            
            # Mock Comprehend client
            mock_comprehend = Mock()
            mock_comprehend.detect_sentiment.return_value = {
                'Sentiment': 'POSITIVE',
                'SentimentScore': {'Positive': 0.9, 'Negative': 0.1, 'Neutral': 0.0, 'Mixed': 0.0}
            }
            mock_comprehend.detect_key_phrases.return_value = {
                'KeyPhrases': [{'Text': 'excellent service', 'Score': 0.95}]
            }
            mock_comprehend.detect_entities.return_value = {
                'Entities': [{'Text': 'customer service', 'Type': 'ORGANIZATION', 'Score': 0.9}]
            }
            
            # Mock Textract and Rekognition for image processing
            mock_textract = Mock()
            mock_textract.detect_document_text.return_value = {
                'Blocks': [
                    {'BlockType': 'LINE', 'Text': 'Test text', 'Confidence': 95.0}
                ]
            }
            
            mock_rekognition = Mock()
            mock_rekognition.detect_labels.return_value = {
                'Labels': [{'Name': 'Product', 'Confidence': 95.0}]
            }
            mock_rekognition.detect_text.return_value = {
                'TextDetections': [{'DetectedText': 'Test text', 'Type': 'LINE', 'Confidence': 95.0}]
            }
            mock_rekognition.detect_moderation_labels.return_value = {'ModerationLabels': []}
            mock_rekognition.detect_faces.return_value = {'FaceDetails': []}
            
            # Mock Transcribe for audio processing
            mock_transcribe = Mock()
            mock_transcribe.start_transcription_job.return_value = {}
            mock_transcribe.get_transcription_job.return_value = {
                'TranscriptionJob': {
                    'TranscriptionJobStatus': 'COMPLETED',
                    'MediaFormat': 'mp3'
                }
            }
            
            # Configure boto3 to return different clients
            def mock_boto_client_factory(service_name):
                if service_name == 's3':
                    return mock_s3
                elif service_name == 'comprehend':
                    return mock_comprehend
                elif service_name == 'textract':
                    return mock_textract
                elif service_name == 'rekognition':
                    return mock_rekognition
                elif service_name == 'transcribe':
                    return mock_transcribe
                else:
                    return Mock()
            
            # Mock S3 get_object response
            event_data = {"Records":[{"eventVersion":"2.1","eventSource":"aws:s3","awsRegion":"us-east-1","eventTime":"2023-12-09T14:00:00.000Z","eventName":"ObjectCreated:Put","s3":{"bucket":{"name":"lm-ai-feedback-raw"},"object":{"key":"audio/call_recording_CUST-00001.mp3","size":2048000,"eTag":"d41d8cd98f00b204e9800998ecf8427e"}}}]}
            
            if 'audio_processing_pipeline' in ['text_processing_pipeline', 'text_api_processing']:
                # Mock S3 response for text processing
                mock_s3.get_object.return_value = {
                    'Body': Mock()
                }
                mock_s3.get_object.return_value['Body'].read.return_value.decode.return_value = json.dumps({
                    'customer_id': 'CUST-00001',
                    'text_content': 'Excellent service and very helpful staff!',
                    'validation_results': {
                        'quality_score': 0.85,
                        'checks': {
                            'content_valid': True,
                            'length_valid': True,
                            'customer_id_valid': True,
                            'no_sensitive_data': True
                        }
                    }
                })
                
                # Mock S3 put_object
                mock_s3.put_object.return_value = {}
                
                # Test text processing
                if 'audio_processing_pipeline' == 'text_processing_pipeline':
                    context = Mock()
                    context.aws_request_id = 'test-request-id'
                    result = text_handler(event_data, context)
                    
                    if result['statusCode'] == 200:
                        print("✅ Text processing pipeline test passed")
                        return True
                    else:
                        print(f"❌ Text processing pipeline test failed: {result}")
                        return False
                        
                elif 'audio_processing_pipeline' == 'text_api_processing':
                    context = Mock()
                    context.aws_request_id = 'test-request-id'
                    result = text_handler(event_data, context)
                    
                    if result['statusCode'] == 200:
                        print("✅ API text processing test passed")
                        return True
                    else:
                        print(f"❌ API text processing test failed: {result}")
                        return False
            
            elif 'audio_processing_pipeline' in ['image_processing_pipeline', 'image_raw_processing']:
                # Mock S3 response for image processing
                mock_s3.get_object.return_value = {
                    'Body': Mock()
                }
                mock_s3.get_object.return_value['Body'].read.return_value.decode.return_value = json.dumps({
                    'customer_id': 'CUST-00001',
                    'image_key': 'images/product_image_CUST-00001.jpg',
                    'image_metadata': {
                        'file_size': 1024000,
                        'width': 1920,
                        'height': 1080,
                        'format': 'JPEG'
                    },
                    'validation_results': {
                        'quality_score': 0.90,
                        'checks': {
                            'format_valid': True,
                            'size_valid': True,
                            'resolution_valid': True,
                            'content_safe': True
                        }
                    }
                })
                
                # Mock S3 put_object
                mock_s3.put_object.return_value = {}
                
                context = Mock()
                context.aws_request_id = 'test-request-id'
                result = image_handler(event_data, context)
                
                if result['statusCode'] == 200:
                    print("✅ Image processing pipeline test passed")
                    return True
                else:
                    print(f"❌ Image processing pipeline test failed: {result}")
                    return False
            
            elif 'audio_processing_pipeline' == 'audio_processing_pipeline':
                # Mock S3 response for audio processing
                mock_s3.get_object.return_value = {
                    'Body': Mock()
                }
                mock_s3.get_object.return_value['Body'].read.return_value.decode.return_value = json.dumps({
                    'results': {
                        'transcripts': [
                            {'transcript': 'Test audio transcript for customer service call.'}
                        ],
                        'speaker_labels': {
                            'segments': [
                                {'speaker_label': 'spk_0', 'start_time': 0.0, 'end_time': 5.0}
                            ]
                        }
                    }
                })
                
                # Mock S3 put_object
                mock_s3.put_object.return_value = {}
                
                context = Mock()
                context.aws_request_id = 'test-request-id'
                result = audio_handler(event_data, context)
                
                if result['statusCode'] == 200:
                    print("✅ Audio processing pipeline test passed")
                    return True
                else:
                    print(f"❌ Audio processing pipeline test failed: {result}")
                    return False
            
            elif 'audio_processing_pipeline' in ['metadata_preservation', 'quality_score_consistency', 'error_handling', 'dlq_functionality']:
                # Generic test for cross-component functionality
                print(f"✅ audio_processing_pipeline test passed")
                return True
            
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {str(e)}")
        return False

if __name__ == '__main__':
    success = run_audio_processing_pipeline()
    sys.exit(0 if success else 1)
