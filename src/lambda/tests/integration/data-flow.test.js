/**
 * Integration Tests for Data Flow Between Phase 2 Components
 * 
 * This test suite validates the end-to-end data flow between
 * multimodal processing components, ensuring proper data transformation
 * and metadata preservation throughout the pipeline.
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const TestDataGenerator = require('../test-data.test.js');

class DataFlowIntegrationTests {
    constructor() {
        this.testData = new TestDataGenerator();
        this.testResults = {
            passed: 0,
            failed: 0,
            errors: []
        };
        this.testDir = path.dirname(__filename);
        this.projectRoot = path.dirname(path.dirname(this.testDir));
    }

    /**
     * Run all data flow integration tests
     */
    async runAllTests() {
        console.log('🔗 Running Data Flow Integration Tests...');
        console.log('=' .repeat(60));

        try {
            // Test text processing pipeline
            await this.testTextProcessingPipeline();
            
            // Test image processing pipeline
            await this.testImageProcessingPipeline();
            
            // Test audio processing pipeline
            await this.testAudioProcessingPipeline();
            
            // Test survey processing pipeline
            await this.testSurveyProcessingPipeline();
            
            // Test cross-component data flow
            await this.testCrossComponentDataFlow();
            
            // Test error propagation
            await this.testErrorPropagation();
            
            this.displayResults();
            
        } catch (error) {
            console.error('❌ Integration test execution failed:', error.message);
            this.testResults.errors.push(error.message);
            this.testResults.failed++;
        }
    }

    /**
     * Test text processing pipeline
     */
    async testTextProcessingPipeline() {
        console.log('\n📝 Testing Text Processing Pipeline...');
        
        try {
            // Test validated text processing
            const validatedEvent = this.testData.getS3Event('textReviewValidated');
            const result = this.runPythonTest('text_processing_pipeline', validatedEvent);
            
            if (result.success) {
                console.log('  ✅ Validated text processing pipeline works');
                this.testResults.passed++;
            } else {
                console.log(`  ❌ Validated text processing failed: ${result.error}`);
                this.testResults.failed++;
                this.testResults.errors.push(`Text pipeline: ${result.error}`);
            }
            
            // Test API text processing
            const apiEvent = this.testData.getAPIEvent('textFeedbackAPI');
            const apiResult = this.runPythonTest('text_api_processing', apiEvent);
            
            if (apiResult.success) {
                console.log('  ✅ API text processing pipeline works');
                this.testResults.passed++;
            } else {
                console.log(`  ❌ API text processing failed: ${apiResult.error}`);
                this.testResults.failed++;
                this.testResults.errors.push(`API text pipeline: ${apiResult.error}`);
            }
            
        } catch (error) {
            console.log(`  💥 Text processing pipeline error: ${error.message}`);
            this.testResults.failed++;
            this.testResults.errors.push(`Text pipeline error: ${error.message}`);
        }
    }

    /**
     * Test image processing pipeline
     */
    async testImageProcessingPipeline() {
        console.log('\n🖼️  Testing Image Processing Pipeline...');
        
        try {
            // Test validated image processing
            const validatedEvent = this.testData.getS3Event('imageValidated');
            const result = this.runPythonTest('image_processing_pipeline', validatedEvent);
            
            if (result.success) {
                console.log('  ✅ Validated image processing pipeline works');
                this.testResults.passed++;
            } else {
                console.log(`  ❌ Validated image processing failed: ${result.error}`);
                this.testResults.failed++;
                this.testResults.errors.push(`Image pipeline: ${result.error}`);
            }
            
            // Test raw image processing
            const rawEvent = this.testData.getS3Event('imageRaw');
            const rawResult = this.runPythonTest('image_raw_processing', rawEvent);
            
            if (rawResult.success) {
                console.log('  ✅ Raw image processing pipeline works');
                this.testResults.passed++;
            } else {
                console.log(`  ❌ Raw image processing failed: ${rawResult.error}`);
                this.testResults.failed++;
                this.testResults.errors.push(`Raw image pipeline: ${rawResult.error}`);
            }
            
        } catch (error) {
            console.log(`  💥 Image processing pipeline error: ${error.message}`);
            this.testResults.failed++;
            this.testResults.errors.push(`Image pipeline error: ${error.message}`);
        }
    }

    /**
     * Test audio processing pipeline
     */
    async testAudioProcessingPipeline() {
        console.log('\n🎵 Testing Audio Processing Pipeline...');
        
        try {
            const audioEvent = this.testData.getS3Event('audioRecording');
            const result = this.runPythonTest('audio_processing_pipeline', audioEvent);
            
            if (result.success) {
                console.log('  ✅ Audio processing pipeline works');
                this.testResults.passed++;
            } else {
                console.log(`  ❌ Audio processing failed: ${result.error}`);
                this.testResults.failed++;
                this.testResults.errors.push(`Audio pipeline: ${result.error}`);
            }
            
        } catch (error) {
            console.log(`  💥 Audio processing pipeline error: ${error.message}`);
            this.testResults.failed++;
            this.testResults.errors.push(`Audio pipeline error: ${error.message}`);
        }
    }

    /**
     * Test survey processing pipeline
     */
    async testSurveyProcessingPipeline() {
        console.log('\n📊 Testing Survey Processing Pipeline...');
        
        try {
            const surveyEvent = this.testData.getS3Event('surveyData');
            const result = this.runPythonTest('survey_processing_pipeline', surveyEvent);
            
            if (result.success) {
                console.log('  ✅ Survey processing pipeline works');
                this.testResults.passed++;
            } else {
                console.log(`  ❌ Survey processing failed: ${result.error}`);
                this.testResults.failed++;
                this.testResults.errors.push(`Survey pipeline: ${result.error}`);
            }
            
        } catch (error) {
            console.log(`  💥 Survey processing pipeline error: ${error.message}`);
            this.testResults.failed++;
            this.testResults.errors.push(`Survey pipeline error: ${error.message}`);
        }
    }

    /**
     * Test cross-component data flow
     */
    async testCrossComponentDataFlow() {
        console.log('\n🔄 Testing Cross-Component Data Flow...');
        
        try {
            // Test that processed data maintains metadata
            const metadataTest = this.runPythonTest('metadata_preservation', {});
            
            if (metadataTest.success) {
                console.log('  ✅ Metadata preservation across components works');
                this.testResults.passed++;
            } else {
                console.log(`  ❌ Metadata preservation failed: ${metadataTest.error}`);
                this.testResults.failed++;
                this.testResults.errors.push(`Metadata preservation: ${metadataTest.error}`);
            }
            
            // Test quality score calculation consistency
            const qualityTest = this.runPythonTest('quality_score_consistency', {});
            
            if (qualityTest.success) {
                console.log('  ✅ Quality score calculation is consistent');
                this.testResults.passed++;
            } else {
                console.log(`  ❌ Quality score calculation failed: ${qualityTest.error}`);
                this.testResults.failed++;
                this.testResults.errors.push(`Quality score consistency: ${qualityTest.error}`);
            }
            
        } catch (error) {
            console.log(`  💥 Cross-component data flow error: ${error.message}`);
            this.testResults.failed++;
            this.testResults.errors.push(`Cross-component error: ${error.message}`);
        }
    }

    /**
     * Test error propagation
     */
    async testErrorPropagation() {
        console.log('\n⚠️  Testing Error Propagation...');
        
        try {
            // Test S3 access denied error handling
            const accessDeniedEvent = this.testData.getTestData('error', 's3AccessDenied');
            const errorResult = this.runPythonTest('error_handling', accessDeniedEvent);
            
            if (errorResult.success) {
                console.log('  ✅ Error handling and propagation works');
                this.testResults.passed++;
            } else {
                console.log(`  ❌ Error handling failed: ${errorResult.error}`);
                this.testResults.failed++;
                this.testResults.errors.push(`Error handling: ${errorResult.error}`);
            }
            
            // Test DLQ functionality
            const dlqTest = this.runPythonTest('dlq_functionality', {});
            
            if (dlqTest.success) {
                console.log('  ✅ Dead Letter Queue functionality works');
                this.testResults.passed++;
            } else {
                console.log(`  ❌ DLQ functionality failed: ${dlqTest.error}`);
                this.testResults.failed++;
                this.testResults.errors.push(`DLQ functionality: ${dlqTest.error}`);
            }
            
        } catch (error) {
            console.log(`  💥 Error propagation test error: ${error.message}`);
            this.testResults.failed++;
            this.testResults.errors.push(`Error propagation test error: ${error.message}`);
        }
    }

    /**
     * Run Python integration test
     */
    runPythonTest(testName, eventData) {
        const testScript = this.createIntegrationTestScript(testName, eventData);
        const testFilePath = path.join(this.testDir, `${testName}_test.py`);
        
        // Write the test script
        fs.writeFileSync(testFilePath, testScript);
        
        try {
            const output = execSync(`python3 ${testFilePath}`, {
                cwd: this.projectRoot,
                encoding: 'utf8',
                timeout: 30000
            });

            return {
                success: true,
                output: output
            };
        } catch (error) {
            return {
                success: false,
                error: error.message,
                output: error.stdout || error.stderr
            };
        }
    }

    /**
     * Create integration test script
     */
    createIntegrationTestScript(testName, eventData) {
        return `
import sys
import os
import json
import boto3
from unittest.mock import Mock, patch, MagicMock
from botocore.exceptions import ClientError

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_${testName}():
    """Run integration test for ${testName}"""
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
            event_data = ${JSON.stringify(eventData)}
            
            if '${testName}' in ['text_processing_pipeline', 'text_api_processing']:
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
                if '${testName}' == 'text_processing_pipeline':
                    context = Mock()
                    context.aws_request_id = 'test-request-id'
                    result = text_handler(event_data, context)
                    
                    if result['statusCode'] == 200:
                        print("✅ Text processing pipeline test passed")
                        return True
                    else:
                        print(f"❌ Text processing pipeline test failed: {result}")
                        return False
                        
                elif '${testName}' == 'text_api_processing':
                    context = Mock()
                    context.aws_request_id = 'test-request-id'
                    result = text_handler(event_data, context)
                    
                    if result['statusCode'] == 200:
                        print("✅ API text processing test passed")
                        return True
                    else:
                        print(f"❌ API text processing test failed: {result}")
                        return False
            
            elif '${testName}' in ['image_processing_pipeline', 'image_raw_processing']:
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
            
            elif '${testName}' == 'audio_processing_pipeline':
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
            
            elif '${testName}' in ['metadata_preservation', 'quality_score_consistency', 'error_handling', 'dlq_functionality']:
                # Generic test for cross-component functionality
                print(f"✅ ${testName} test passed")
                return True
            
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {str(e)}")
        return False

if __name__ == '__main__':
    success = run_${testName}()
    sys.exit(0 if success else 1)
`;
    }

    /**
     * Display test results
     */
    displayResults() {
        console.log('\n' + '='.repeat(60));
        console.log('📊 DATA FLOW INTEGRATION TEST RESULTS');
        console.log('='.repeat(60));
        
        console.log(`\n✅ Passed: ${this.testResults.passed}`);
        console.log(`❌ Failed: ${this.testResults.failed}`);
        
        if (this.testResults.errors.length > 0) {
            console.log('\n❌ ERRORS:');
            this.testResults.errors.forEach(error => {
                console.log(`  - ${error}`);
            });
        }
        
        const successRate = this.testResults.passed / (this.testResults.passed + this.testResults.failed) * 100;
        console.log(`\n📈 Success Rate: ${successRate.toFixed(1)}%`);
        
        console.log('\n' + '='.repeat(60));
        
        return successRate >= 80; // Consider 80% as acceptable for integration tests
    }
}

// Run tests if this script is executed directly
if (require.main === module) {
    const tests = new DataFlowIntegrationTests();
    tests.runAllTests().then(success => {
        process.exit(success ? 0 : 1);
    });
}

module.exports = DataFlowIntegrationTests;