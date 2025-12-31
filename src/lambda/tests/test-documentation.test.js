/**
 * Test Documentation for Phase 2 Multimodal Data Processing
 * 
 * This file contains comprehensive documentation for all tests,
 * testing strategies, and validation criteria for Phase 2 components.
 */

const fs = require('fs');
const path = require('path');

class TestDocumentation {
    constructor() {
        this.testDir = path.dirname(__filename);
        this.reportsDir = path.join(this.testDir, 'reports');
    }

    /**
     * Generate comprehensive test documentation
     */
    generateTestDocumentation() {
        console.log('📚 Generating Test Documentation...');
        
        const documentation = {
            title: 'Phase 2 Multimodal Data Processing Test Documentation',
            version: '2.0',
            lastUpdated: new Date().toISOString(),
            sections: {
                overview: this.generateOverview(),
                testStrategy: this.generateTestStrategy(),
                unitTests: this.generateUnitTestDocumentation(),
                integrationTests: this.generateIntegrationTestDocumentation(),
                testData: this.generateTestDataDocumentation(),
                validationCriteria: this.generateValidationCriteria(),
                executionGuide: this.generateExecutionGuide(),
                troubleshooting: this.generateTroubleshootingGuide()
            }
        };

        // Save documentation
        const docPath = path.join(this.reportsDir, 'test-documentation.json');
        fs.writeFileSync(docPath, JSON.stringify(documentation, null, 2));
        
        // Generate markdown version
        this.generateMarkdownDocumentation(documentation, docPath.replace('.json', '.md'));
        
        console.log(`✅ Test documentation generated: ${docPath}`);
        return documentation;
    }

    /**
     * Generate overview section
     */
    generateOverview() {
        return {
            purpose: 'Comprehensive testing of Phase 2 multimodal data processing components',
            scope: [
                'Text Processing Lambda function',
                'Image Processing Lambda function', 
                'Audio Processing Lambda function',
                'Survey Processing Script'
            ],
            objectives: [
                'Validate functional correctness of all components',
                'Ensure proper error handling and edge case coverage',
                'Verify integration between Phase 1 validation and Phase 2 processing',
                'Test data flow and metadata preservation',
                'Validate retry logic and failure scenarios',
                'Ensure code coverage meets quality standards'
            ],
            testingApproach: 'Hybrid approach combining JavaScript test orchestration with Python unit and integration tests'
        };
    }

    /**
     * Generate test strategy section
     */
    generateTestStrategy() {
        return {
            methodology: 'Test-Driven Development with comprehensive coverage',
            testLevels: [
                {
                    level: 'Unit Tests',
                    purpose: 'Test individual functions and methods in isolation',
                    coverage: 'Function-level testing with mocked dependencies',
                    tools: 'Python unittest framework with JavaScript orchestration'
                },
                {
                    level: 'Integration Tests',
                    purpose: 'Test component interactions and data flow',
                    coverage: 'End-to-end pipeline testing',
                    tools: 'Custom integration test framework with AWS service mocking'
                },
                {
                    level: 'System Tests',
                    purpose: 'Validate complete system behavior',
                    coverage: 'Full workflow testing with real data scenarios',
                    tools: 'JavaScript-based test execution with comprehensive reporting'
                }
            ],
            testingTypes: [
                {
                    type: 'Functional Testing',
                    description: 'Verify that components perform their intended functions correctly',
                    examples: ['Text sentiment analysis', 'Image label detection', 'Audio transcription']
                },
                {
                    type: 'Error Handling Testing',
                    description: 'Ensure graceful handling of errors and edge cases',
                    examples: ['S3 access failures', 'Invalid data formats', 'Service timeouts']
                },
                {
                    type: 'Performance Testing',
                    description: 'Validate performance characteristics and resource usage',
                    examples: ['Processing time limits', 'Memory usage', 'Retry behavior']
                },
                {
                    type: 'Security Testing',
                    description: 'Verify security measures and data protection',
                    examples: ['Content moderation', 'Sensitive data handling', 'Access controls']
                }
            ]
        };
    }

    /**
     * Generate unit test documentation
     */
    generateUnitTestDocumentation() {
        return {
            textProcessing: {
                component: 'text_processing_lambda.py',
                testFile: 'unit/test_text_processing_lambda.py',
                testCases: [
                    {
                        name: 'test_lambda_handler_initialization',
                        purpose: 'Test Lambda handler initialization and environment setup',
                        inputs: ['Mock event', 'Mock context'],
                        expectedOutputs: ['Proper response structure', 'Correct status codes'],
                        assertions: ['Handler returns expected response format', 'Environment variables loaded correctly']
                    },
                    {
                        name: 'test_process_validated_s3_event',
                        purpose: 'Test processing of validated text data from Phase 1',
                        inputs: ['S3 event with validated text file'],
                        expectedOutputs: ['Processed text with sentiment analysis', 'Combined quality score'],
                        assertions: ['Text content processed correctly', 'Sentiment analysis performed', 'Quality score calculated']
                    },
                    {
                        name: 'test_process_api_request',
                        purpose: 'Test direct API Gateway requests',
                        inputs: ['API Gateway event with feedback data'],
                        expectedOutputs: ['Processed feedback response', 'Stored results in S3'],
                        assertions: ['Input validation works', 'Response format correct', 'S3 storage successful']
                    },
                    {
                        name: 'test_error_handling',
                        purpose: 'Test error handling for various failure scenarios',
                        inputs: ['Invalid events', 'S3 access errors', 'Service failures'],
                        expectedOutputs: ['Proper error responses', 'DLQ messages'],
                        assertions: ['Errors caught and logged', 'Appropriate status codes', 'DLQ functionality']
                    },
                    {
                        name: 'test_retry_logic',
                        purpose: 'Test retry mechanism for transient failures',
                        inputs: ['Mock service failures', 'Network timeouts'],
                        expectedOutputs: ['Exponential backoff', 'Maximum retry enforcement'],
                        assertions: ['Retry attempts logged', 'Backoff delays correct', 'Max retries respected']
                    }
                ],
                mockServices: ['boto3.client', 'comprehend', 's3', 'cloudwatch', 'sqs'],
                coverageTarget: '85%'
            },
            imageProcessing: {
                component: 'image_processing_lambda.py',
                testFile: 'unit/test_image_processing_lambda.py',
                testCases: [
                    {
                        name: 'test_lambda_handler_initialization',
                        purpose: 'Test Lambda handler initialization',
                        inputs: ['Mock event', 'Mock context'],
                        expectedOutputs: ['Proper response structure', 'Client initialization'],
                        assertions: ['AWS clients initialized', 'Environment variables loaded']
                    },
                    {
                        name: 'test_process_validated_image_metadata',
                        purpose: 'Test processing of validated image metadata',
                        inputs: ['S3 event with validated image metadata'],
                        expectedOutputs: ['Processed image analysis', 'Combined results'],
                        assertions: ['Textract integration works', 'Rekognition analysis performed', 'Results combined correctly']
                    },
                    {
                        name: 'test_process_raw_image',
                        purpose: 'Test processing of raw image files',
                        inputs: ['S3 event with raw image'],
                        expectedOutputs: ['Image analysis results', 'Metadata extraction'],
                        assertions: ['Image content analyzed', 'Labels detected', 'Text extracted', 'Faces detected']
                    },
                    {
                        name: 'test_content_moderation',
                        purpose: 'Test content moderation and safety checks',
                        inputs: ['Images with various content types'],
                        expectedOutputs: ['Moderation labels', 'Safety flags'],
                        assertions: ['Inappropriate content flagged', 'Safe content passes', 'Moderation thresholds applied']
                    }
                ],
                mockServices: ['boto3.client', 'textract', 'rekognition', 's3', 'cloudwatch'],
                coverageTarget: '80%'
            },
            audioProcessing: {
                component: 'audio_processing_lambda.py',
                testFile: 'unit/test_audio_processing_lambda.py',
                testCases: [
                    {
                        name: 'test_lambda_handler_initialization',
                        purpose: 'Test Lambda handler initialization',
                        inputs: ['Mock event', 'Mock context'],
                        expectedOutputs: ['Proper response structure', 'Transcribe client setup'],
                        assertions: ['Transcribe client initialized', 'Environment configuration loaded']
                    },
                    {
                        name: 'test_audio_transcription',
                        purpose: 'Test audio transcription process',
                        inputs: ['S3 event with audio file'],
                        expectedOutputs: ['Transcription job started', 'Job completion handling'],
                        assertions: ['Transcribe job created', 'Job status monitored', 'Results retrieved']
                    },
                    {
                        name: 'test_sentiment_analysis',
                        purpose: 'Test sentiment analysis of transcribed audio',
                        inputs: ['Transcribed text content'],
                        expectedOutputs: ['Sentiment scores', 'Key phrases', 'Entities'],
                        assertions: ['Comprehend integration works', 'Sentiment detected correctly', 'Key phrases extracted']
                    },
                    {
                        name: 'test_speaker_analysis',
                        purpose: 'Test speaker identification and analysis',
                        inputs: ['Multi-speaker audio'],
                        expectedOutputs: ['Speaker segments', 'Speaker attribution'],
                        assertions: ['Multiple speakers detected', 'Speaker segments accurate', 'Timing correct']
                    }
                ],
                mockServices: ['boto3.client', 'transcribe', 'comprehend', 's3'],
                coverageTarget: '75%'
            },
            surveyProcessing: {
                component: 'survey_processing_script.py',
                testFile: 'unit/test_survey_processing_script.py',
                testCases: [
                    {
                        name: 'test_process_survey_data',
                        purpose: 'Test main survey processing function',
                        inputs: ['Survey CSV data'],
                        expectedOutputs: ['Processed summaries', 'Statistics', 'Trends'],
                        assertions: ['Data loaded correctly', 'Summaries generated', 'Statistics calculated']
                    },
                    {
                        name: 'test_calculate_summary_statistics',
                        purpose: 'Test statistics calculation',
                        inputs: ['Survey response data'],
                        expectedOutputs: ['Summary statistics', 'Demographics', 'Ratings breakdown'],
                        assertions: ['Satisfaction averages correct', 'Response rates calculated', 'Distributions accurate']
                    },
                    {
                        name: 'test_generate_survey_summaries',
                        purpose: 'Test natural language summary generation',
                        inputs: ['Individual survey responses'],
                        expectedOutputs: ['Formatted summaries', 'Priority scores'],
                        assertions: ['Summaries coherent', 'Priority scores logical', 'Key information captured']
                    },
                    {
                        name: 'test_trend_analysis',
                        purpose: 'Test trend analysis functionality',
                        inputs: ['Time-series survey data'],
                        expectedOutputs: ['Trend patterns', 'Correlations'],
                        assertions: ['Temporal trends identified', 'Correlations calculated', 'Insights meaningful']
                    }
                ],
                mockServices: ['pandas', 'numpy', 'boto3.client'],
                coverageTarget: '90%'
            }
        };
    }

    /**
     * Generate integration test documentation
     */
    generateIntegrationTestDocumentation() {
        return {
            dataFlow: {
                purpose: 'Test end-to-end data flow between components',
                testFile: 'integration/data-flow.test.js',
                scenarios: [
                    {
                        name: 'text_processing_pipeline',
                        description: 'Test complete text processing workflow from Phase 1 validation to Phase 2 processing',
                        steps: [
                            'S3 event triggers text processing Lambda',
                            'Validated text data retrieved from S3',
                            'Amazon Comprehend performs sentiment analysis',
                            'Key phrases and entities extracted',
                            'Results combined with validation metadata',
                            'Processed data saved to S3',
                            'Metrics sent to CloudWatch'
                        ],
                        validationPoints: [
                            'Data flows correctly between Phase 1 and Phase 2',
                            'Metadata preserved throughout pipeline',
                            'Quality scores calculated correctly',
                            'Error handling works at each step'
                        ]
                    },
                    {
                        name: 'image_processing_pipeline',
                        description: 'Test complete image processing workflow',
                        steps: [
                            'S3 event triggers image processing Lambda',
                            'Image metadata validated and retrieved',
                            'Amazon Textract extracts text from image',
                            'Amazon Rekognition detects labels and content',
                            'Content moderation performed',
                            'Face detection and analysis completed',
                            'Results combined and saved to S3'
                        ],
                        validationPoints: [
                            'Image content analyzed comprehensively',
                            'Multiple AWS services integrated correctly',
                            'Safety checks performed',
                            'Processing results accurate'
                        ]
                    },
                    {
                        name: 'audio_processing_pipeline',
                        description: 'Test complete audio processing workflow',
                        steps: [
                            'S3 event triggers audio processing Lambda',
                            'Amazon Transcribe starts transcription job',
                            'Job completion monitored and handled',
                            'Transcribed text analyzed with Comprehend',
                            'Speaker analysis performed',
                            'Results saved to S3 with metadata'
                        ],
                        validationPoints: [
                            'Async transcription handled correctly',
                            'Speaker identification works',
                            'Sentiment analysis accurate',
                            'Error handling robust'
                        ]
                    },
                    {
                        name: 'survey_processing_pipeline',
                        description: 'Test complete survey processing workflow',
                        steps: [
                            'SageMaker processing job started',
                            'Survey CSV data loaded and cleaned',
                            'Statistical analysis performed',
                            'Natural language summaries generated',
                            'Trend analysis completed',
                            'Results saved to S3'
                        ],
                        validationPoints: [
                            'Data processing comprehensive',
                            'Statistical calculations accurate',
                            'Summaries meaningful and coherent',
                            'Trends identified correctly'
                        ]
                    }
                ]
            },
            errorPropagation: {
                purpose: 'Test error handling and propagation across components',
                scenarios: [
                    {
                        name: 's3_access_denied',
                        description: 'Test handling of S3 access permission errors',
                        expectedBehavior: 'Error caught, logged, and sent to DLQ',
                        validation: 'Proper error response and DLQ message'
                    },
                    {
                        name: 'service_timeout',
                        description: 'Test handling of AWS service timeouts',
                        expectedBehavior: 'Retry logic engaged with exponential backoff',
                        validation: 'Retry attempts logged and max retries enforced'
                    },
                    {
                        name: 'invalid_data_format',
                        description: 'Test handling of malformed input data',
                        expectedBehavior: 'Validation error returned with clear message',
                        validation: 'Appropriate error code and descriptive message'
                    }
                ]
            },
            metadataPreservation: {
                purpose: 'Test metadata preservation throughout processing pipeline',
                validationPoints: [
                    'Customer ID maintained across all processing steps',
                    'Validation scores combined with processing results',
                    'Timestamps preserved and updated appropriately',
                    'Environment and version information tracked',
                    'Processing lineage documented'
                ]
            }
        };
    }

    /**
     * Generate test data documentation
     */
    generateTestDataDocumentation() {
        return {
            categories: [
                {
                    name: 'Validated Data',
                    purpose: 'Simulate Phase 1 validated output for Phase 2 processing',
                    files: [
                        'text-validatedText.json',
                        'image-validatedImage.json',
                        'audio-transcriptionResponse.json',
                        'survey-processedSummaries.json'
                    ],
                    characteristics: [
                        'Contains validation metadata',
                        'Includes quality scores',
                        'Has proper customer identification',
                        'Follows expected schema'
                    ]
                },
                {
                    name: 'Raw Data',
                    purpose: 'Simulate unprocessed input data',
                    files: [
                        'text-rawTextContent.json',
                        'image-rekognitionLabels.json',
                        'survey-surveyCSV.json'
                    ],
                    characteristics: [
                        'Represents real-world input',
                        'Various data formats',
                        'Edge cases included',
                        'Error scenarios covered'
                    ]
                },
                {
                    name: 'Mock AWS Responses',
                    purpose: 'Simulate AWS service responses for testing',
                    files: [
                        'image-textractResponse.json',
                        'image-rekognitionLabels.json',
                        'audio-sentimentAnalysis.json'
                    ],
                    characteristics: [
                        'Realistic response format',
                        'Various confidence levels',
                        'Edge cases included',
                        'Error responses covered'
                    ]
                },
                {
                    name: 'Error Scenarios',
                    purpose: 'Test error handling and edge cases',
                    files: [
                        'error-s3AccessDenied.json',
                        'error-invalidFileFormat.json',
                        'error-malformedJSON.json',
                        'error-missingFields.json'
                    ],
                    characteristics: [
                        'Covers common failure modes',
                        'Tests error propagation',
                        'Validates recovery mechanisms',
                        'Includes edge cases'
                    ]
                }
            ],
            s3Events: [
                {
                    name: 'textReviewValidated',
                    purpose: 'Trigger text processing for validated data',
                    bucket: 'lm-ai-feedback-dev',
                    key: 'processed/text_reviews/review_CUST-00001_validated.json'
                },
                {
                    name: 'imageValidated',
                    purpose: 'Trigger image processing for validated data',
                    bucket: 'lm-ai-feedback-dev',
                    key: 'processed/images/prompt_CUST-00001_validated.json'
                },
                {
                    name: 'audioRecording',
                    purpose: 'Trigger audio processing',
                    bucket: 'lm-ai-feedback-raw',
                    key: 'audio/call_recording_CUST-00001.mp3'
                }
            ],
            apiEvents: [
                {
                    name: 'textFeedbackAPI',
                    purpose: 'Test direct API Gateway text processing',
                    method: 'POST',
                    contentType: 'application/json',
                    body: {
                        customerId: 'CUST-00001',
                        rating: 5,
                        feedback: 'Excellent service and very helpful staff!',
                        timestamp: '2023-12-09T14:00:00.000Z'
                    }
                }
            ]
        };
    }

    /**
     * Generate validation criteria documentation
     */
    generateValidationCriteria() {
        return {
            successCriteria: [
                {
                    criterion: 'Unit Test Pass Rate',
                    threshold: '≥ 90%',
                    description: 'At least 90% of unit tests must pass',
                    measurement: 'Number of passed tests / total tests'
                },
                {
                    criterion: 'Integration Test Pass Rate',
                    threshold: '≥ 80%',
                    description: 'At least 80% of integration tests must pass',
                    measurement: 'Number of passed tests / total tests'
                },
                {
                    criterion: 'Code Coverage',
                    threshold: '≥ 75%',
                    description: 'Code coverage must be at least 75%',
                    measurement: 'Lines covered / total lines'
                },
                {
                    criterion: 'No Critical Errors',
                    threshold: '0',
                    description: 'No critical errors in test execution',
                    measurement: 'Count of critical errors'
                }
            ],
            qualityMetrics: [
                {
                    metric: 'Test Execution Time',
                    target: '< 5 minutes',
                    description: 'Complete test suite should run within 5 minutes',
                    measurement: 'Total execution time'
                },
                {
                    metric: 'Error Handling Coverage',
                    target: '100%',
                    description: 'All error paths must be tested',
                    measurement: 'Error paths tested / total error paths'
                },
                {
                    metric: 'Edge Case Coverage',
                    target: '≥ 80%',
                    description: 'At least 80% of edge cases covered',
                    measurement: 'Edge cases tested / total edge cases'
                }
            ],
            deploymentReadiness: [
                {
                    check: 'All Unit Tests Pass',
                    required: true,
                    description: 'All unit tests must pass before deployment'
                },
                {
                    check: 'Integration Tests Stable',
                    required: true,
                    description: 'Integration tests must show consistent results'
                },
                {
                    check: 'No Blocking Issues',
                    required: true,
                    description: 'No issues that would block deployment'
                },
                {
                    check: 'Documentation Complete',
                    required: false,
                    description: 'Test documentation should be complete and up to date'
                }
            ]
        };
    }

    /**
     * Generate execution guide
     */
    generateExecutionGuide() {
        return {
            prerequisites: [
                'Node.js 14+ installed',
                'Python 3.8+ installed',
                'Test dependencies installed (pytest, unittest, mock)',
                'AWS credentials configured (for integration tests)',
                'S3 buckets created for testing'
            ],
            setup: [
                {
                    step: 1,
                    action: 'Install Node.js dependencies',
                    command: 'npm install'
                },
                {
                    step: 2,
                    action: 'Install Python test dependencies',
                    command: 'pip install -r requirements-test.txt'
                },
                {
                    step: 3,
                    action: 'Set environment variables',
                    variables: [
                        'AWS_REGION=us-east-1',
                        'ENVIRONMENT=test',
                        'PROCESSED_BUCKET=test-bucket'
                    ]
                },
                {
                    step: 4,
                    action: 'Generate test data',
                    command: 'node Code/tests/test-data.test.js'
                }
            ],
            execution: [
                {
                    command: 'Run complete test suite',
                    instruction: 'node Code/tests/run-tests.test.js',
                    description: 'Executes all unit and integration tests with comprehensive reporting'
                },
                {
                    command: 'Run unit tests only',
                    instruction: 'node Code/tests/test-runner.test.js --unit-only',
                    description: 'Executes only unit tests for all components'
                },
                {
                    command: 'Run integration tests only',
                    instruction: 'node Code/tests/integration/data-flow.test.js',
                    description: 'Executes only integration tests'
                },
                {
                    command: 'Generate test data',
                    instruction: 'node Code/tests/test-data.test.js',
                    description: 'Generates test data and fixtures'
                }
            ],
            troubleshooting: [
                {
                    issue: 'Test execution fails',
                    solution: 'Check Python and Node.js installations, verify dependencies',
                    commands: ['python3 --version', 'node --version', 'pip list']
                },
                {
                    issue: 'AWS service errors',
                    solution: 'Verify credentials and permissions, check service availability',
                    commands: ['aws sts get-caller-identity', 'aws s3 ls']
                },
                {
                    issue: 'Test data missing',
                    solution: 'Regenerate test data using test-data generator',
                    commands: ['node Code/tests/test-data.test.js']
                }
            ]
        };
    }

    /**
     * Generate troubleshooting guide
     */
    generateTroubleshootingGuide() {
        return {
            commonIssues: [
                {
                    issue: 'Module Import Errors',
                    symptoms: ['ImportError: No module named', 'ModuleNotFoundError'],
                    causes: ['Missing dependencies', 'Incorrect Python path', 'Virtual environment issues'],
                    solutions: [
                        'Install missing dependencies: pip install -r requirements.txt',
                        'Check Python path and virtual environment',
                        'Verify all required modules are installed'
                    ],
                    prevention: 'Maintain up-to-date requirements.txt and use virtual environments'
                },
                {
                    issue: 'AWS Service Connection Errors',
                    symptoms: ['ClientError', 'ConnectionTimeout', 'AccessDenied'],
                    causes: ['Invalid credentials', 'Insufficient permissions', 'Network issues'],
                    solutions: [
                        'Verify AWS credentials: aws configure',
                        'Check IAM permissions for required services',
                        'Test network connectivity to AWS endpoints'
                    ],
                    prevention: 'Use IAM roles for Lambda functions and test credentials locally'
                },
                {
                    issue: 'Test Data Issues',
                    symptoms: ['FileNotFoundError', 'JSON decode errors', 'Missing test fixtures'],
                    causes: ['Test data not generated', 'Corrupted test files', 'Path issues'],
                    solutions: [
                        'Regenerate test data: node test-data.test.js',
                        'Verify test data file integrity',
                        'Check file paths and permissions'
                    ],
                    prevention: 'Generate test data before running tests and version control test fixtures'
                },
                {
                    issue: 'Mock Service Failures',
                    symptoms: ['Mock object not configured', 'Unexpected mock responses'],
                    causes: ['Incorrect mock setup', 'Mock configuration conflicts'],
                    solutions: [
                        'Review mock configuration in test files',
                        'Ensure mock objects properly initialized',
                        'Check for mock conflicts between tests'
                    ],
                    prevention: 'Use consistent mock patterns and isolate test setups'
                }
            ],
            debuggingTips: [
                {
                    tip: 'Enable Verbose Logging',
                    instruction: 'Set logging level to DEBUG for detailed output',
                    example: 'logging.basicConfig(level=logging.DEBUG)'
                },
                {
                    tip: 'Use Breakpoints',
                    instruction: 'Add debug breakpoints to inspect test execution',
                    example: 'import pdb; pdb.set_trace()'
                },
                {
                    tip: 'Check AWS Service Responses',
                    instruction: 'Log actual AWS service responses for debugging',
                    example: 'print(f"AWS Response: {response}")'
                },
                {
                    tip: 'Validate Test Data',
                    instruction: 'Verify test data structure and content before tests',
                    example: 'assert isinstance(test_data, dict)'
                }
            ],
            performanceOptimization: [
                {
                    technique: 'Mock External Services',
                    description: 'Mock AWS services to avoid real API calls during testing',
                    benefit: 'Faster test execution and no dependency on service availability'
                },
                {
                    technique: 'Use Test Fixtures',
                    description: 'Pre-generate and reuse test data fixtures',
                    benefit: 'Consistent test data and faster test setup'
                },
                {
                    technique: 'Parallel Test Execution',
                    description: 'Run tests in parallel where possible',
                    benefit: 'Reduced total test execution time'
                }
            ]
        };
    }

    /**
     * Generate markdown documentation
     */
    generateMarkdownDocumentation(documentation, mdPath) {
        const markdown = this.convertToMarkdown(documentation);
        fs.writeFileSync(mdPath, markdown);
        console.log(`📄 Markdown documentation generated: ${mdPath}`);
    }

    /**
     * Convert documentation to markdown format
     */
    convertToMarkdown(doc) {
        return `# ${doc.title}

**Version:** ${doc.version}  
**Last Updated:** ${doc.lastUpdated}

## Overview

${doc.sections.overview.purpose}

### Scope
${doc.sections.overview.scope.map(item => `- ${item}`).join('\n')}

### Objectives
${doc.sections.overview.objectives.map(item => `- ${item}`).join('\n')}

### Testing Approach
${doc.sections.overview.testingApproach}

## Test Strategy

### Methodology
${doc.sections.testStrategy.methodology}

### Test Levels
${doc.sections.testStrategy.testLevels.map(level => `
#### ${level.level}
- **Purpose:** ${level.purpose}
- **Coverage:** ${level.coverage}
- **Tools:** ${level.tools}
`).join('\n')}

### Testing Types
${doc.sections.testStrategy.testingTypes.map(type => `
#### ${type.type}
- **Description:** ${type.description}
- **Examples:** ${type.examples.join(', ')}
`).join('\n')}

## Unit Tests

### Text Processing Lambda
**Component:** ${doc.sections.unitTests.textProcessing.component}  
**Test File:** ${doc.sections.unitTests.textProcessing.testFile}  
**Coverage Target:** ${doc.sections.unitTests.textProcessing.coverageTarget}

#### Test Cases
${doc.sections.unitTests.textProcessing.testCases.map(test => `
##### ${test.name}
- **Purpose:** ${test.purpose}
- **Inputs:** ${test.inputs.join(', ')}
- **Expected Outputs:** ${test.expectedOutputs.join(', ')}
- **Assertions:** ${test.assertions.join(', ')}
`).join('\n')}

**Mock Services:** ${doc.sections.unitTests.textProcessing.mockServices.join(', ')}

### Image Processing Lambda
**Component:** ${doc.sections.unitTests.imageProcessing.component}  
**Test File:** ${doc.sections.unitTests.imageProcessing.testFile}  
**Coverage Target:** ${doc.sections.unitTests.imageProcessing.coverageTarget}

#### Test Cases
${doc.sections.unitTests.imageProcessing.testCases.map(test => `
##### ${test.name}
- **Purpose:** ${test.purpose}
- **Inputs:** ${test.inputs.join(', ')}
- **Expected Outputs:** ${test.expectedOutputs.join(', ')}
- **Assertions:** ${test.assertions.join(', ')}
`).join('\n')}

**Mock Services:** ${doc.sections.unitTests.imageProcessing.mockServices.join(', ')}

### Audio Processing Lambda
**Component:** ${doc.sections.unitTests.audioProcessing.component}  
**Test File:** ${doc.sections.unitTests.audioProcessing.testFile}  
**Coverage Target:** ${doc.sections.unitTests.audioProcessing.coverageTarget}

#### Test Cases
${doc.sections.unitTests.audioProcessing.testCases.map(test => `
##### ${test.name}
- **Purpose:** ${test.purpose}
- **Inputs:** ${test.inputs.join(', ')}
- **Expected Outputs:** ${test.expectedOutputs.join(', ')}
- **Assertions:** ${test.assertions.join(', ')}
`).join('\n')}

**Mock Services:** ${doc.sections.unitTests.audioProcessing.mockServices.join(', ')}

### Survey Processing Script
**Component:** ${doc.sections.unitTests.surveyProcessing.component}  
**Test File:** ${doc.sections.unitTests.surveyProcessing.testFile}  
**Coverage Target:** ${doc.sections.unitTests.surveyProcessing.coverageTarget}

#### Test Cases
${doc.sections.unitTests.surveyProcessing.testCases.map(test => `
##### ${test.name}
- **Purpose:** ${test.purpose}
- **Inputs:** ${test.inputs.join(', ')}
- **Expected Outputs:** ${test.expectedOutputs.join(', ')}
- **Assertions:** ${test.assertions.join(', ')}
`).join('\n')}

**Mock Services:** ${doc.sections.unitTests.surveyProcessing.mockServices.join(', ')}

## Integration Tests

### Data Flow Testing
**Purpose:** ${doc.sections.integrationTests.dataFlow.purpose}  
**Test File:** ${doc.sections.integrationTests.dataFlow.testFile}

#### Scenarios
${doc.sections.integrationTests.dataFlow.scenarios.map(scenario => `
##### ${scenario.name}
**Description:** ${scenario.description}

**Steps:**
${scenario.steps.map(step => `- ${step}`).join('\n')}

**Validation Points:**
${scenario.validationPoints.map(point => `- ${point}`).join('\n')}
`).join('\n')}

### Error Propagation Testing
**Purpose:** ${doc.sections.integrationTests.errorPropagation.purpose}

#### Scenarios
${doc.sections.integrationTests.errorPropagation.scenarios.map(scenario => `
##### ${scenario.name}
**Description:** ${scenario.description}
**Expected Behavior:** ${scenario.expectedBehavior}
**Validation:** ${scenario.validation}
`).join('\n')}

### Metadata Preservation Testing
**Purpose:** ${doc.sections.integrationTests.metadataPreservation.purpose}

**Validation Points:**
${doc.sections.integrationTests.metadataPreservation.validationPoints.map(point => `- ${point}`).join('\n')}

## Test Data

### Data Categories
${doc.sections.testData.categories.map(category => `
#### ${category.name}
**Purpose:** ${category.purpose}

**Files:**
${category.files.map(file => `- ${file}`).join('\n')}

**Characteristics:**
${category.characteristics.map(char => `- ${char}`).join('\n')}
`).join('\n')}

### S3 Events
${doc.sections.testData.s3Events.map(event => `
#### ${event.name}
- **Purpose:** ${event.purpose}
- **Bucket:** ${event.bucket}
- **Key:** ${event.key}
`).join('\n')}

### API Events
${doc.sections.testData.apiEvents.map(event => `
#### ${event.name}
- **Purpose:** ${event.purpose}
- **Method:** ${event.method}
- **Content-Type:** ${event.contentType}
- **Body:** ${JSON.stringify(event.body, null, 2)}
`).join('\n')}

## Validation Criteria

### Success Criteria
${doc.sections.validationCriteria.successCriteria.map(criterion => `
#### ${criterion.criterion}
- **Threshold:** ${criterion.threshold}
- **Description:** ${criterion.description}
- **Measurement:** ${criterion.measurement}
`).join('\n')}

### Quality Metrics
${doc.sections.validationCriteria.qualityMetrics.map(metric => `
#### ${metric.metric}
- **Target:** ${metric.target}
- **Description:** ${metric.description}
- **Measurement:** ${metric.measurement}
`).join('\n')}

### Deployment Readiness
${doc.sections.validationCriteria.deploymentReadiness.map(check => `
#### ${check.check}
- **Required:** ${check.required ? 'Yes' : 'No'}
- **Description:** ${check.description}
`).join('\n')}

## Execution Guide

### Prerequisites
${doc.sections.executionGuide.prerequisites.map(prereq => `- ${prereq}`).join('\n')}

### Setup Steps
${doc.sections.executionGuide.setup.map(step => `
#### Step ${step.step}: ${step.action}
**Command:** \`${step.command}\`
`).join('\n')}

### Execution Commands
${doc.sections.executionGuide.execution.map(cmd => `
#### ${cmd.command}
**Instruction:** ${cmd.instruction}
**Description:** ${cmd.description}
`).join('\n')}

### Troubleshooting
${doc.sections.executionGuide.troubleshooting.map(issue => `
#### ${issue.issue}
**Symptoms:** ${issue.symptoms ? issue.symptoms.join(', ') : 'N/A'}
**Causes:** ${issue.causes ? issue.causes.join(', ') : 'N/A'}
**Solutions:** ${issue.solutions ? issue.solutions.map(sol => `- ${sol}`).join('\n') : 'N/A'}
**Prevention:** ${issue.prevention || 'N/A'}
`).join('\n')}

## Troubleshooting Guide

### Common Issues
${doc.sections.troubleshooting.commonIssues.map(issue => `
#### ${issue.issue}
**Symptoms:** ${issue.symptoms.join(', ')}
**Causes:** ${issue.causes.join(', ')}
**Solutions:** ${issue.solutions.map(sol => `- ${sol}`).join('\n')}
**Prevention:** ${issue.prevention}
`).join('\n')}

### Debugging Tips
${doc.sections.troubleshooting.debuggingTips.map(tip => `
#### ${tip.tip}
**Instruction:** ${tip.instruction || 'N/A'}
**Example:** \`${tip.example || 'N/A'}\`
`).join('\n')}

### Performance Optimization
${doc.sections.troubleshooting.performanceOptimization.map(technique => `
#### ${technique.technique}
**Description:** ${technique.description || 'N/A'}
**Benefit:** ${technique.benefit || 'N/A'}
`).join('\n')}

---

*This documentation is automatically generated as part of the Phase 2 testing process.*
`;
    }
}

// Generate documentation if this script is executed directly
if (require.main === module) {
    const docGenerator = new TestDocumentation();
    docGenerator.generateTestDocumentation();
}

module.exports = TestDocumentation;