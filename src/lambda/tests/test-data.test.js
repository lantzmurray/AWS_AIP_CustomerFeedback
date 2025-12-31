/**
 * Test Data and Mock S3 Events for Phase 2 Testing
 * 
 * This file contains test data, mock S3 events, and test fixtures
 * for comprehensive testing of multimodal data processing components.
 */

const fs = require('fs');
const path = require('path');

class TestDataGenerator {
    constructor() {
        this.testDir = path.dirname(__filename);
        this.dataDir = path.join(this.testDir, 'data');
        this.fixturesDir = path.join(this.testDir, 'fixtures');
    }

    /**
     * Generate all test data and fixtures
     */
    generateAllTestData() {
        console.log('📦 Generating test data and fixtures...');
        
        this.ensureDirectoryExists(this.dataDir);
        this.ensureDirectoryExists(this.fixturesDir);
        
        // Generate mock S3 events
        this.generateMockS3Events();
        
        // Generate test data files
        this.generateTextTestData();
        this.generateImageTestData();
        this.generateAudioTestData();
        this.generateSurveyTestData();
        
        // Generate error scenarios
        this.generateErrorScenarios();
        
        console.log('✅ Test data generation complete');
    }

    /**
     * Generate mock S3 events for testing
     */
    generateMockS3Events() {
        const s3Events = {
            // Validated text review event
            textReviewValidated: {
                "Records": [
                    {
                        "eventVersion": "2.1",
                        "eventSource": "aws:s3",
                        "awsRegion": "us-east-1",
                        "eventTime": "2023-12-09T14:00:00.000Z",
                        "eventName": "ObjectCreated:Put",
                        "s3": {
                            "bucket": {
                                "name": "lm-ai-feedback-dev"
                            },
                            "object": {
                                "key": "processed/text_reviews/review_CUST-00001_validated.json",
                                "size": 1024,
                                "eTag": "d41d8cd98f00b204e9800998ecf8427e"
                            }
                        }
                    }
                ]
            },
            
            // Raw text review event
            textReviewRaw: {
                "Records": [
                    {
                        "eventVersion": "2.1",
                        "eventSource": "aws:s3",
                        "awsRegion": "us-east-1",
                        "eventTime": "2023-12-09T14:00:00.000Z",
                        "eventName": "ObjectCreated:Put",
                        "s3": {
                            "bucket": {
                                "name": "lm-ai-feedback-raw"
                            },
                            "object": {
                                "key": "text_reviews/review_CUST-00002.txt",
                                "size": 512,
                                "eTag": "d41d8cd98f00b204e9800998ecf8427e"
                            }
                        }
                    }
                ]
            },
            
            // Validated image event
            imageValidated: {
                "Records": [
                    {
                        "eventVersion": "2.1",
                        "eventSource": "aws:s3",
                        "awsRegion": "us-east-1",
                        "eventTime": "2023-12-09T14:00:00.000Z",
                        "eventName": "ObjectCreated:Put",
                        "s3": {
                            "bucket": {
                                "name": "lm-ai-feedback-dev"
                            },
                            "object": {
                                "key": "processed/images/prompt_CUST-00001_validated.json",
                                "size": 2048,
                                "eTag": "d41d8cd98f00b204e9800998ecf8427e"
                            }
                        }
                    }
                ]
            },
            
            // Raw image event
            imageRaw: {
                "Records": [
                    {
                        "eventVersion": "2.1",
                        "eventSource": "aws:s3",
                        "awsRegion": "us-east-1",
                        "eventTime": "2023-12-09T14:00:00.000Z",
                        "eventName": "ObjectCreated:Put",
                        "s3": {
                            "bucket": {
                                "name": "lm-ai-feedback-raw"
                            },
                            "object": {
                                "key": "images/product_image_CUST-00001.jpg",
                                "size": 1024000,
                                "eTag": "d41d8cd98f00b204e9800998ecf8427e"
                            }
                        }
                    }
                ]
            },
            
            // Audio recording event
            audioRecording: {
                "Records": [
                    {
                        "eventVersion": "2.1",
                        "eventSource": "aws:s3",
                        "awsRegion": "us-east-1",
                        "eventTime": "2023-12-09T14:00:00.000Z",
                        "eventName": "ObjectCreated:Put",
                        "s3": {
                            "bucket": {
                                "name": "lm-ai-feedback-raw"
                            },
                            "object": {
                                "key": "audio/call_recording_CUST-00001.mp3",
                                "size": 2048000,
                                "eTag": "d41d8cd98f00b204e9800998ecf8427e"
                            }
                        }
                    }
                ]
            },
            
            // Survey data event
            surveyData: {
                "Records": [
                    {
                        "eventVersion": "2.1",
                        "eventSource": "aws:s3",
                        "awsRegion": "us-east-1",
                        "eventTime": "2023-12-09T14:00:00.000Z",
                        "eventName": "ObjectCreated:Put",
                        "s3": {
                            "bucket": {
                                "name": "lm-ai-feedback-raw"
                            },
                            "object": {
                                "key": "surveys/customer_feedback_survey.csv",
                                "size": 5120,
                                "eTag": "d41d8cd98f00b204e9800998ecf8427e"
                            }
                        }
                    }
                ]
            }
        };

        // Write S3 events to fixtures
        Object.entries(s3Events).forEach(([name, event]) => {
            const filePath = path.join(this.fixturesDir, `s3-event-${name}.json`);
            fs.writeFileSync(filePath, JSON.stringify(event, null, 2));
        });

        // Write API Gateway events
        const apiEvents = {
            textFeedbackAPI: {
                "body": JSON.stringify({
                    "customerId": "CUST-00001",
                    "rating": 5,
                    "feedback": "Excellent service and very helpful staff! The product quality exceeded my expectations.",
                    "timestamp": "2023-12-09T14:00:00.000Z"
                }),
                "httpMethod": "POST",
                "headers": {
                    "Content-Type": "application/json"
                }
            },
            
            invalidAPI: {
                "body": JSON.stringify({
                    "customerId": "",  // Invalid empty customer ID
                    "rating": 10,      // Invalid rating
                    "feedback": "Short" // Too short feedback
                }),
                "httpMethod": "POST",
                "headers": {
                    "Content-Type": "application/json"
                }
            }
        };

        Object.entries(apiEvents).forEach(([name, event]) => {
            const filePath = path.join(this.fixturesDir, `api-event-${name}.json`);
            fs.writeFileSync(filePath, JSON.stringify(event, null, 2));
        });
    }

    /**
     * Generate test data for text processing
     */
    generateTextTestData() {
        const textData = {
            // Validated text data (Phase 1 output)
            validatedText: {
                "customer_id": "CUST-00001",
                "text_content": "Excellent service and very helpful staff! The product quality exceeded my expectations. I would definitely recommend this to others.",
                "validation_results": {
                    "quality_score": 0.85,
                    "checks": {
                        "content_valid": true,
                        "length_valid": true,
                        "customer_id_valid": true,
                        "no_sensitive_data": true
                    },
                    "validation_timestamp": "2023-12-09T14:00:00.000Z"
                },
                "metadata": {
                    "source": "text_review",
                    "file_name": "review_CUST-00001.txt",
                    "validation_version": "1.0"
                }
            },
            
            // Low quality validated text
            lowQualityText: {
                "customer_id": "CUST-00002",
                "text_content": "bad product",
                "validation_results": {
                    "quality_score": 0.45,
                    "checks": {
                        "content_valid": true,
                        "length_valid": false,
                        "customer_id_valid": true,
                        "no_sensitive_data": true
                    },
                    "validation_timestamp": "2023-12-09T14:00:00.000Z"
                },
                "metadata": {
                    "source": "text_review",
                    "file_name": "review_CUST-00002.txt",
                    "validation_version": "1.0"
                }
            },
            
            // Raw text content
            rawTextContent: "The customer service was terrible. I waited for 30 minutes and no one helped me. The product broke after one week.",
            
            // Mixed sentiment text
            mixedSentimentText: "The product quality is excellent, but the delivery was delayed by 3 days. Customer service was helpful though.",
            
            // Empty/invalid text
            emptyText: "",
            invalidText: "   \n\t   ",
            longText: "This is a very long text that exceeds the normal processing limits. ".repeat(50)
        };

        // Write text test data
        Object.entries(textData).forEach(([name, data]) => {
            const filePath = path.join(this.dataDir, `text-${name}.json`);
            fs.writeFileSync(filePath, JSON.stringify(data, null, 2));
        });
    }

    /**
     * Generate test data for image processing
     */
    generateImageTestData() {
        const imageData = {
            // Validated image metadata
            validatedImage: {
                "customer_id": "CUST-00001",
                "image_key": "images/product_image_CUST-00001.jpg",
                "image_metadata": {
                    "file_size": 1024000,
                    "width": 1920,
                    "height": 1080,
                    "format": "JPEG",
                    "color_space": "RGB"
                },
                "validation_results": {
                    "quality_score": 0.90,
                    "checks": {
                        "format_valid": true,
                        "size_valid": true,
                        "resolution_valid": true,
                        "content_safe": true
                    },
                    "validation_timestamp": "2023-12-09T14:00:00.000Z"
                },
                "metadata": {
                    "source": "image_upload",
                    "file_name": "product_image_CUST-00001.jpg",
                    "validation_version": "1.0"
                }
            },
            
            // Low quality image
            lowQualityImage: {
                "customer_id": "CUST-00002",
                "image_key": "images/blurry_image_CUST-00002.jpg",
                "image_metadata": {
                    "file_size": 512000,
                    "width": 640,
                    "height": 480,
                    "format": "JPEG",
                    "color_space": "RGB"
                },
                "validation_results": {
                    "quality_score": 0.55,
                    "checks": {
                        "format_valid": true,
                        "size_valid": true,
                        "resolution_valid": false,
                        "content_safe": true
                    },
                    "validation_timestamp": "2023-12-09T14:00:00.000Z"
                },
                "metadata": {
                    "source": "image_upload",
                    "file_name": "blurry_image_CUST-00002.jpg",
                    "validation_version": "1.0"
                }
            },
            
            // Mock Textract response
            textractResponse: {
                "Blocks": [
                    {
                        "BlockType": "LINE",
                        "Text": "Product Quality: Excellent",
                        "Confidence": 95.5,
                        "Geometry": {
                            "BoundingBox": {
                                "Width": 0.8,
                                "Height": 0.1,
                                "Left": 0.1,
                                "Top": 0.2
                            }
                        }
                    },
                    {
                        "BlockType": "LINE",
                        "Text": "Customer Rating: 5/5",
                        "Confidence": 92.3,
                        "Geometry": {
                            "BoundingBox": {
                                "Width": 0.6,
                                "Height": 0.08,
                                "Left": 0.15,
                                "Top": 0.35
                            }
                        }
                    }
                ]
            },
            
            // Mock Rekognition labels response
            rekognitionLabels: {
                "Labels": [
                    {
                        "Name": "Product",
                        "Confidence": 98.5,
                        "Instances": [],
                        "Parents": []
                    },
                    {
                        "Name": "Electronics",
                        "Confidence": 95.2,
                        "Instances": [],
                        "Parents": [
                            {"Name": "Technology"}
                        ]
                    },
                    {
                        "Name": "Customer Review",
                        "Confidence": 87.3,
                        "Instances": [],
                        "Parents": [
                            {"Name": "Text"},
                            {"Name": "Document"}
                        ]
                    }
                ]
            },
            
            // Mock moderation labels
            moderationLabels: [],
            
            // Mock face detection
            faceDetection: {
                "FaceDetails": [
                    {
                        "BoundingBox": {
                            "Width": 0.15,
                            "Height": 0.2,
                            "Left": 0.4,
                            "Top": 0.3
                        },
                        "Confidence": 99.1,
                        "Gender": {
                            "Value": "Female",
                            "Confidence": 95.5
                        },
                        "AgeRange": {
                            "Low": 25,
                            "High": 35
                        },
                        "Emotions": [
                            {"Type": "HAPPY", "Confidence": 85.2},
                            {"Type": "CALM", "Confidence": 78.9}
                        ],
                        "Quality": {
                            "Brightness": 85.3,
                            "Sharpness": 92.1
                        }
                    }
                ]
            }
        };

        // Write image test data
        Object.entries(imageData).forEach(([name, data]) => {
            const filePath = path.join(this.dataDir, `image-${name}.json`);
            fs.writeFileSync(filePath, JSON.stringify(data, null, 2));
        });
    }

    /**
     * Generate test data for audio processing
     */
    generateAudioTestData() {
        const audioData = {
            // Mock transcription response
            transcriptionResponse: {
                "results": {
                    "transcripts": [
                        {
                            "transcript": "Hello, I'm calling about my recent order. The product arrived damaged and I'd like to request a refund. The customer service representative was very helpful and processed my request quickly."
                        }
                    ],
                    "speaker_labels": {
                        "segments": [
                            {
                                "speaker_label": "spk_0",
                                "start_time": 0.0,
                                "end_time": 5.2
                            },
                            {
                                "speaker_label": "spk_1",
                                "start_time": 5.3,
                                "end_time": 12.8
                            },
                            {
                                "speaker_label": "spk_0",
                                "start_time": 12.9,
                                "end_time": 18.5
                            }
                        ]
                    }
                },
                "status": "COMPLETED"
            },
            
            // Mock sentiment analysis
            sentimentAnalysis: {
                "Sentiment": "MIXED",
                "SentimentScore": {
                    "Positive": 0.45,
                    "Negative": 0.35,
                    "Neutral": 0.15,
                    "Mixed": 0.05
                }
            },
            
            // Mock key phrases
            keyPhrases: {
                "KeyPhrases": [
                    {
                        "Text": "customer service",
                        "Score": 0.95,
                        "BeginOffset": 85,
                        "EndOffset": 100
                    },
                    {
                        "Text": "product arrived damaged",
                        "Score": 0.88,
                        "BeginOffset": 42,
                        "EndOffset": 62
                    },
                    {
                        "Text": "request a refund",
                        "Score": 0.82,
                        "BeginOffset": 68,
                        "EndOffset": 85
                    }
                ]
            },
            
            // Mock entities
            entities: {
                "Entities": [
                    {
                        "Text": "customer service",
                        "Score": 0.95,
                        "Type": "ORGANIZATION",
                        "BeginOffset": 85,
                        "EndOffset": 100
                    },
                    {
                        "Text": "order",
                        "Score": 0.88,
                        "Type": "COMMERCIAL_ITEM",
                        "BeginOffset": 35,
                        "EndOffset": 40
                    }
                ]
            }
        };

        // Write audio test data
        Object.entries(audioData).forEach(([name, data]) => {
            const filePath = path.join(this.dataDir, `audio-${name}.json`);
            fs.writeFileSync(filePath, JSON.stringify(data, null, 2));
        });
    }

    /**
     * Generate test data for survey processing
     */
    generateSurveyTestData() {
        const surveyData = {
            // Sample survey CSV data
            surveyCSV: `customer_id,survey_date,overall_satisfaction,product_rating,service_rating,improvement_area,comments
CUST-00001,2023-12-01,5,5,5,Product Quality,"Excellent product, exceeded expectations!"
CUST-00002,2023-12-02,2,3,1,Customer Service,"Waited too long for support, very disappointed."
CUST-00003,2023-12-03,4,4,4,Delivery,"Fast delivery but packaging could be better."
CUST-00004,2023-12-04,3,3,3,Price,"Good value for money, average quality."
CUST-00005,2023-12-05,1,2,1,Product Quality,"Product broke after one week, terrible quality."`,
            
            // Expected processed survey summaries
            processedSummaries: [
                {
                    "customer_id": "CUST-00001",
                    "survey_date": "2023-12-01",
                    "summary_text": "Customer CUST-00001 reported being very satisfied, rated product 5/5, rated customer service 5/5, suggested improvements in Product Quality, noted: 'Excellent product, exceeded expectations!'.",
                    "ratings": {
                        "overall_satisfaction": 5,
                        "product_rating": 5,
                        "service_rating": 5
                    },
                    "comments": "Excellent product, exceeded expectations!",
                    "improvement_areas": ["Product Quality"],
                    "sentiment_indicators": {
                        "positive_words": 3,
                        "negative_words": 0,
                        "sentiment_balance": 3
                    },
                    "priority_score": 0.0
                },
                {
                    "customer_id": "CUST-00002",
                    "survey_date": "2023-12-02",
                    "summary_text": "Customer CUST-00002 reported being dissatisfied, rated product 3/5, rated customer service 1/5, suggested improvements in Customer Service, noted: 'Waited too long for support, very disappointed.'.",
                    "ratings": {
                        "overall_satisfaction": 2,
                        "product_rating": 3,
                        "service_rating": 1
                    },
                    "comments": "Waited too long for support, very disappointed.",
                    "improvement_areas": ["Customer Service"],
                    "sentiment_indicators": {
                        "positive_words": 0,
                        "negative_words": 3,
                        "sentiment_balance": -3
                    },
                    "priority_score": 6.5
                }
            ],
            
            // Expected statistics
            surveyStatistics: {
                "total_surveys": 5,
                "response_rate": {
                    "completion_rate": 0.95,
                    "avg_fields_completed": 5.7
                },
                "avg_satisfaction": 3.0,
                "satisfaction_distribution": {
                    "5": 1,
                    "4": 1,
                    "3": 1,
                    "2": 1,
                    "1": 1
                },
                "top_issues": {
                    "Product Quality": 2,
                    "Customer Service": 1,
                    "Delivery": 1,
                    "Price": 1
                }
            }
        };

        // Write survey test data
        Object.entries(surveyData).forEach(([name, data]) => {
            const filePath = path.join(this.dataDir, `survey-${name}.json`);
            fs.writeFileSync(filePath, JSON.stringify(data, null, 2));
        });
    }

    /**
     * Generate error scenarios for testing
     */
    generateErrorScenarios() {
        const errorScenarios = {
            // S3 access errors
            s3AccessDenied: {
                "Records": [
                    {
                        "eventVersion": "2.1",
                        "eventSource": "aws:s3",
                        "s3": {
                            "bucket": {
                                "name": "forbidden-bucket"
                            },
                            "object": {
                                "key": "restricted/file.txt"
                            }
                        }
                    }
                ]
            },
            
            // Invalid file formats
            invalidFileFormat: {
                "Records": [
                    {
                        "eventVersion": "2.1",
                        "eventSource": "aws:s3",
                        "s3": {
                            "bucket": {
                                "name": "test-bucket"
                            },
                            "object": {
                                "key": "data/invalid_file.xyz"
                            }
                        }
                    }
                ]
            },
            
            // Malformed JSON
            malformedJSON: {
                "customer_id": "CUST-00001",
                "text_content": "Test content",
                "validation_results": {
                    "quality_score": "not-a-number",  // Invalid type
                    "checks": "not-an-object"        // Invalid type
                }
            },
            
            // Missing required fields
            missingFields: {
                "customer_id": "CUST-00001"
                // Missing text_content and validation_results
            },
            
            // Empty S3 object
            emptyS3Object: {
                "Records": [
                    {
                        "eventVersion": "2.1",
                        "eventSource": "aws:s3",
                        "s3": {
                            "bucket": {
                                "name": "test-bucket"
                            },
                            "object": {
                                "key": "data/empty_file.txt",
                                "size": 0
                            }
                        }
                    }
                ]
            },
            
            // Large file scenario
            largeFile: {
                "Records": [
                    {
                        "eventVersion": "2.1",
                        "eventSource": "aws:s3",
                        "s3": {
                            "bucket": {
                                "name": "test-bucket"
                            },
                            "object": {
                                "key": "data/large_file.txt",
                                "size": 104857600  // 100MB
                            }
                        }
                    }
                ]
            }
        };

        // Write error scenarios
        Object.entries(errorScenarios).forEach(([name, scenario]) => {
            const filePath = path.join(this.dataDir, `error-${name}.json`);
            fs.writeFileSync(filePath, JSON.stringify(scenario, null, 2));
        });
    }

    /**
     * Ensure directory exists
     */
    ensureDirectoryExists(dirPath) {
        if (!fs.existsSync(dirPath)) {
            fs.mkdirSync(dirPath, { recursive: true });
        }
    }

    /**
     * Get test data by name
     */
    getTestData(category, name) {
        const filePath = path.join(this.dataDir, `${category}-${name}.json`);
        if (fs.existsSync(filePath)) {
            return JSON.parse(fs.readFileSync(filePath, 'utf8'));
        }
        return null;
    }

    /**
     * Get S3 event fixture by name
     */
    getS3Event(name) {
        const filePath = path.join(this.fixturesDir, `s3-event-${name}.json`);
        if (fs.existsSync(filePath)) {
            return JSON.parse(fs.readFileSync(filePath, 'utf8'));
        }
        return null;
    }

    /**
     * Get API event fixture by name
     */
    getAPIEvent(name) {
        const filePath = path.join(this.fixturesDir, `api-event-${name}.json`);
        if (fs.existsSync(filePath)) {
            return JSON.parse(fs.readFileSync(filePath, 'utf8'));
        }
        return null;
    }
}

// Generate test data if this script is executed directly
if (require.main === module) {
    const generator = new TestDataGenerator();
    generator.generateAllTestData();
}

module.exports = TestDataGenerator;