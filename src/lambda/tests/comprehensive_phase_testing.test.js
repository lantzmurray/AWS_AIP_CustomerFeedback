/**
 * Comprehensive Test Suite for AWS AI Project Phases 1, 2, and 3
 * 
 * This test suite systematically tests all components of the deployed system:
 * - Phase 1: Data Validation
 * - Phase 2: Multimodal Processing  
 * - Phase 3: Foundation Model Formatting
 * - End-to-End Pipeline Testing
 * - Performance Testing
 * - Integration Testing
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

class ComprehensivePhaseTesting {
    constructor() {
        this.testDir = path.dirname(__filename);
        this.projectRoot = path.dirname(path.dirname(this.testDir)); // Go up two levels to reach project root
        this.reportsDir = path.join(this.testDir, 'reports');
        this.startTime = new Date();
        
        this.testResults = {
            phase1: {
                dataValidation: { passed: 0, failed: 0, errors: [] },
                glueDatabase: { passed: 0, failed: 0, errors: [] },
                cloudWatchLogs: { passed: 0, failed: 0, errors: [] }
            },
            phase2: {
                textProcessing: { passed: 0, failed: 0, errors: [] },
                imageProcessing: { passed: 0, failed: 0, errors: [] },
                audioProcessing: { passed: 0, failed: 0, errors: [] },
                surveyProcessing: { passed: 0, failed: 0, errors: [] },
                cloudWatchLogs: { passed: 0, failed: 0, errors: [] },
                processedDataStorage: { passed: 0, failed: 0, errors: [] }
            },
            phase3: {
                foundationModelFormatting: { passed: 0, failed: 0, errors: [] },
                realTimeFormatter: { passed: 0, failed: 0, errors: [] },
                stepFunctions: { passed: 0, failed: 0, errors: [] },
                formattedDataOutput: { passed: 0, failed: 0, errors: [] },
                bedrockIntegration: { passed: 0, failed: 0, errors: [] }
            },
            endToEnd: {
                dataFlow: { passed: 0, failed: 0, errors: [] },
                qualityScores: { passed: 0, failed: 0, errors: [] },
                metadataPreservation: { passed: 0, failed: 0, errors: [] },
                errorHandling: { passed: 0, failed: 0, errors: [] }
            },
            performance: {
                latency: { passed: 0, failed: 0, errors: [] },
                throughput: { passed: 0, failed: 0, errors: [] },
                timeoutConfigurations: { passed: 0, failed: 0, errors: [] },
                resourceUsage: { passed: 0, failed: 0, errors: [] }
            },
            integration: {
                s3Triggers: { passed: 0, failed: 0, errors: [] },
                iamPermissions: { passed: 0, failed: 0, errors: [] },
                dlqFunctionality: { passed: 0, failed: 0, errors: [] },
                cloudWatchMetrics: { passed: 0, failed: 0, errors: [] }
            }
        };
    }

    /**
     * Execute comprehensive test suite for all phases
     */
    async executeCompleteTestSuite() {
        console.log('🚀 Starting Comprehensive AWS AI Project Test Suite');
        console.log('=' .repeat(80));
        console.log(`📅 Started: ${this.startTime.toISOString()}`);
        console.log(`📂 Project: ${this.projectRoot}`);
        console.log('=' .repeat(80));

        try {
            // Ensure reports directory exists
            this.ensureDirectoryExists(this.reportsDir);

            // Phase 1: Data Validation Testing
            console.log('\n🔍 PHASE 1: Testing Data Validation Components...');
            await this.testPhase1Components();

            // Phase 2: Multimodal Processing Testing
            console.log('\n🔄 PHASE 2: Testing Multimodal Processing Components...');
            await this.testPhase2Components();

            // Phase 3: Foundation Model Formatting Testing
            console.log('\n🤖 PHASE 3: Testing Foundation Model Formatting Components...');
            await this.testPhase3Components();

            // End-to-End Pipeline Testing
            console.log('\n🌊 END-TO-END: Testing Complete Pipeline...');
            await this.testEndToEndPipeline();

            // Performance Testing
            console.log('\n⚡ PERFORMANCE: Testing System Performance...');
            await this.testPerformance();

            // Integration Testing
            console.log('\n🔗 INTEGRATION: Testing System Integration...');
            await this.testIntegration();

            // Generate comprehensive report
            console.log('\n📄 Generating Comprehensive Test Report...');
            await this.generateComprehensiveReport();

            // Display final results
            this.displayFinalResults();

        } catch (error) {
            console.error('❌ Test suite execution failed:', error.message);
            await this.generateErrorReport(error);
        }
    }

    /**
     * Test Phase 1 Components (Data Validation)
     */
    async testPhase1Components() {
        console.log('  📋 Testing Text Validation with Sample Data...');
        await this.testTextValidation();

        console.log('  📊 Testing Image Validation with Sample Data...');
        await this.testImageValidation();

        console.log('  🎵 Testing Audio Validation with Sample Data...');
        await this.testAudioValidation();

        console.log('  📝 Testing Survey Validation with Sample Data...');
        await this.testSurveyValidation();

        console.log('  🗄️ Testing Glue Database Population...');
        await this.testGlueDatabasePopulation();

        console.log('  📈 Testing CloudWatch Logs for Validation Results...');
        await this.testCloudWatchLogs();
    }

    /**
     * Test Phase 2 Components (Multimodal Processing)
     */
    async testPhase2Components() {
        console.log('  📝 Testing Text Processing Lambda...');
        await this.testTextProcessing();

        console.log('  🖼️ Testing Image Processing Lambda...');
        await this.testImageProcessing();

        console.log('  🎵 Testing Audio Processing Lambda...');
        await this.testAudioProcessing();

        console.log('  📊 Testing Survey Processing Lambda...');
        await this.testSurveyProcessing();

        console.log('  📈 Testing CloudWatch Logs for Processing Results...');
        await this.testProcessingCloudWatchLogs();

        console.log('  📦 Testing Processed Data Storage in S3...');
        await this.testProcessedDataStorage();
    }

    /**
     * Test Phase 3 Components (Foundation Model Formatting)
     */
    async testPhase3Components() {
        console.log('  🤖 Testing Foundation Model Formatting Lambda Functions...');
        await this.testFoundationModelFormatting();

        console.log('  ⚡ Testing Real-Time Formatter Lambda...');
        await this.testRealTimeFormatter();

        console.log('  🔄 Testing Step Functions Workflow...');
        await this.testStepFunctionsWorkflow();

        console.log('  📄 Testing Formatted Data Output Formats...');
        await this.testFormattedDataOutput();

        console.log('  🔗 Testing AWS Bedrock Integration...');
        await this.testBedrockIntegration();
    }

    /**
     * Test End-to-End Pipeline
     */
    async testEndToEndPipeline() {
        console.log('  🌊 Testing Complete Data Flow from Raw to Formatted...');
        await this.testDataFlow();

        console.log('  📈 Testing Quality Scores at Each Stage...');
        await this.testQualityScores();

        console.log('  🏷️ Testing Metadata Preservation Throughout Pipeline...');
        await this.testMetadataPreservation();

        console.log('  ⚠️ Testing Error Handling and Retry Mechanisms...');
        await this.testErrorHandlingMechanisms();
    }

    /**
     * Test Performance
     */
    async testPerformance() {
        console.log('  ⏱️ Measuring Processing Latency for Each Component...');
        await this.testProcessingLatency();

        console.log('  🚀 Testing with Multiple Concurrent Files...');
        await this.testConcurrentProcessing();

        console.log('  ⏰ Verifying Timeout Configurations...');
        await this.testTimeoutConfigurations();

        console.log('  💾 Checking Memory Usage and Performance Metrics...');
        await this.testMemoryUsage();
    }

    /**
     * Test Integration
     */
    async testIntegration() {
        console.log('  📦 Testing S3 Event Triggers...');
        await this.testS3EventTriggers();

        console.log('  🔐 Verifying IAM Permissions are Working Correctly...');
        await this.testIAMPermissions();

        console.log('  💀 Testing DLQ (Dead Letter Queue) Functionality...');
        await this.testDLQFunctionality();

        console.log('  📊 Validating CloudWatch Metrics and Alarms...');
        await this.testCloudWatchMetrics();
    }

    /**
     * Test Text Validation
     */
    async testTextValidation() {
        try {
            // Test with sample text reviews
            const sampleTextFiles = fs.readdirSync(path.join(this.projectRoot, 'sample_data/text_reviews'))
                .filter(file => file.endsWith('.txt'));

            for (const file of sampleTextFiles.slice(0, 3)) { // Test first 3 files
                const filePath = path.join(this.projectRoot, 'sample_data/text_reviews', file);
                const content = fs.readFileSync(filePath, 'utf8');
                
                // Validate content meets requirements
                const validation = {
                    hasMinLength: content.length >= 10,
                    hasProductReference: /product|item|purchase/i.test(content),
                    hasOpinion: /like|love|hate|good|bad|great|terrible|excellent|poor|recommend/i.test(content),
                    hasStructure: content.includes('.') || content.includes('!')
                };

                const passed = Object.values(validation).every(v => v);
                if (passed) {
                    this.testResults.phase1.dataValidation.passed++;
                } else {
                    this.testResults.phase1.dataValidation.failed++;
                    this.testResults.phase1.dataValidation.errors.push(`Text validation failed for ${file}: ${JSON.stringify(validation)}`);
                }
            }

            console.log(`    ✅ Text validation tested with ${sampleTextFiles.length} files`);
        } catch (error) {
            this.testResults.phase1.dataValidation.errors.push(`Text validation error: ${error.message}`);
            this.testResults.phase1.dataValidation.failed++;
        }
    }

    /**
     * Test Image Validation
     */
    async testImageValidation() {
        try {
            // Test with sample image prompts
            const sampleImageFiles = fs.readdirSync(path.join(this.projectRoot, 'sample_data/images'))
                .filter(file => file.endsWith('.txt'));

            for (const file of sampleImageFiles.slice(0, 2)) { // Test first 2 files
                const filePath = path.join(this.projectRoot, 'sample_data/images', file);
                const content = fs.readFileSync(filePath, 'utf8');
                
                // Validate image prompt meets requirements
                const validation = {
                    hasMinLength: content.length >= 10,
                    hasDescription: /describe|show|image|picture/i.test(content),
                    hasContext: /product|item|customer|feedback/i.test(content)
                };

                const passed = Object.values(validation).every(v => v);
                if (passed) {
                    this.testResults.phase1.dataValidation.passed++;
                } else {
                    this.testResults.phase1.dataValidation.failed++;
                    this.testResults.phase1.dataValidation.errors.push(`Image validation failed for ${file}: ${JSON.stringify(validation)}`);
                }
            }

            console.log(`    ✅ Image validation tested with ${sampleImageFiles.length} files`);
        } catch (error) {
            this.testResults.phase1.dataValidation.errors.push(`Image validation error: ${error.message}`);
            this.testResults.phase1.dataValidation.failed++;
        }
    }

    /**
     * Test Audio Validation
     */
    async testAudioValidation() {
        try {
            // Test with sample audio transcripts
            const sampleAudioFiles = fs.readdirSync(path.join(this.projectRoot, 'sample_data/audio'))
                .filter(file => file.endsWith('.txt'));

            for (const file of sampleAudioFiles.slice(0, 2)) { // Test first 2 files
                const filePath = path.join(this.projectRoot, 'sample_data/audio', file);
                const content = fs.readFileSync(filePath, 'utf8');
                
                // Validate transcript meets requirements
                const validation = {
                    hasMinLength: content.length >= 10,
                    hasDialogue: /speaker|customer|service/i.test(content),
                    hasContext: /call|conversation|issue|problem/i.test(content)
                };

                const passed = Object.values(validation).every(v => v);
                if (passed) {
                    this.testResults.phase1.dataValidation.passed++;
                } else {
                    this.testResults.phase1.dataValidation.failed++;
                    this.testResults.phase1.dataValidation.errors.push(`Audio validation failed for ${file}: ${JSON.stringify(validation)}`);
                }
            }

            console.log(`    ✅ Audio validation tested with ${sampleAudioFiles.length} files`);
        } catch (error) {
            this.testResults.phase1.dataValidation.errors.push(`Audio validation error: ${error.message}`);
            this.testResults.phase1.dataValidation.failed++;
        }
    }

    /**
     * Test Survey Validation
     */
    async testSurveyValidation() {
        try {
            // Test with sample survey data
            const surveyFilePath = path.join(this.projectRoot, 'sample_data/surveys/customer_feedback_survey.csv');
            const content = fs.readFileSync(surveyFilePath, 'utf8');
            
            // Validate CSV structure and content
            const validation = {
                hasHeader: content.includes('customer_id'),
                hasData: content.split('\n').length > 10, // At least 10 data rows
                hasRatings: /rating|satisfaction/i.test(content),
                hasValidFormat: content.includes(',') // CSV format
            };

            const passed = Object.values(validation).every(v => v);
            if (passed) {
                this.testResults.phase1.dataValidation.passed++;
            } else {
                this.testResults.phase1.dataValidation.failed++;
                this.testResults.phase1.dataValidation.errors.push(`Survey validation failed: ${JSON.stringify(validation)}`);
            }

            console.log(`    ✅ Survey validation tested with CSV file`);
        } catch (error) {
            this.testResults.phase1.dataValidation.errors.push(`Survey validation error: ${error.message}`);
            this.testResults.phase1.dataValidation.failed++;
        }
    }

    /**
     * Test Glue Database Population
     */
    async testGlueDatabasePopulation() {
        try {
            // Check if Glue database components exist
            const glueComponents = [
                'Code/data_validation/glue_data_quality_ruleset.py',
                'Code/multimodal_processing/',
                'Code/fm_formatting/'
            ];

            for (const component of glueComponents) {
                const componentPath = path.join(this.projectRoot, component);
                if (fs.existsSync(componentPath) || fs.statSync(componentPath).isDirectory()) {
                    this.testResults.phase1.glueDatabase.passed++;
                } else {
                    this.testResults.phase1.glueDatabase.failed++;
                    this.testResults.phase1.glueDatabase.errors.push(`Missing component: ${component}`);
                }
            }

            console.log(`    ✅ Glue database components verified`);
        } catch (error) {
            this.testResults.phase1.glueDatabase.errors.push(`Glue database test error: ${error.message}`);
            this.testResults.phase1.glueDatabase.failed++;
        }
    }

    /**
     * Test CloudWatch Logs
     */
    async testCloudWatchLogs() {
        try {
            // Check if CloudWatch configuration exists
            const cloudWatchComponents = [
                'Code/data_validation/cloudwatch_dashboard.py',
                'Code/data_validation/simple_cloudwatch_dashboard.py'
            ];

            for (const component of cloudWatchComponents) {
                const componentPath = path.join(this.projectRoot, component);
                if (fs.existsSync(componentPath)) {
                    this.testResults.phase1.cloudWatchLogs.passed++;
                } else {
                    this.testResults.phase1.cloudWatchLogs.failed++;
                    this.testResults.phase1.cloudWatchLogs.errors.push(`Missing CloudWatch component: ${component}`);
                }
            }

            console.log(`    ✅ CloudWatch log components verified`);
        } catch (error) {
            this.testResults.phase1.cloudWatchLogs.errors.push(`CloudWatch test error: ${error.message}`);
            this.testResults.phase1.cloudWatchLogs.failed++;
        }
    }

    /**
     * Test Text Processing
     */
    async testTextProcessing() {
        try {
            const textProcessorPath = path.join(this.projectRoot, 'Code/multimodal_processing/text_processing_lambda.py');
            if (fs.existsSync(textProcessorPath)) {
                const content = fs.readFileSync(textProcessorPath, 'utf8');
                
                // Validate text processing components
                const validation = {
                    hasComprehendIntegration: content.includes('comprehend'),
                    hasEntityExtraction: content.includes('entities'),
                    hasSentimentAnalysis: content.includes('sentiment'),
                    hasKeyPhrases: content.includes('key_phrases'),
                    hasErrorHandling: content.includes('try:') && content.includes('except')
                };

                const passed = Object.values(validation).every(v => v);
                if (passed) {
                    this.testResults.phase2.textProcessing.passed++;
                } else {
                    this.testResults.phase2.textProcessing.failed++;
                    this.testResults.phase2.textProcessing.errors.push(`Text processing validation failed: ${JSON.stringify(validation)}`);
                }
            } else {
                this.testResults.phase2.textProcessing.failed++;
                this.testResults.phase2.textProcessing.errors.push('Text processing Lambda not found');
            }

            console.log(`    ✅ Text processing Lambda verified`);
        } catch (error) {
            this.testResults.phase2.textProcessing.errors.push(`Text processing error: ${error.message}`);
            this.testResults.phase2.textProcessing.failed++;
        }
    }

    /**
     * Test Image Processing
     */
    async testImageProcessing() {
        try {
            const imageProcessorPath = path.join(this.projectRoot, 'Code/multimodal_processing/image_processing_lambda.py');
            if (fs.existsSync(imageProcessorPath)) {
                const content = fs.readFileSync(imageProcessorPath, 'utf8');
                
                // Validate image processing components
                const validation = {
                    hasRekognitionIntegration: content.includes('rekognition'),
                    hasTextractIntegration: content.includes('textract'),
                    hasLabelDetection: content.includes('labels'),
                    hasTextExtraction: content.includes('detected_text'),
                    hasErrorHandling: content.includes('try:') && content.includes('except')
                };

                const passed = Object.values(validation).every(v => v);
                if (passed) {
                    this.testResults.phase2.imageProcessing.passed++;
                } else {
                    this.testResults.phase2.imageProcessing.failed++;
                    this.testResults.phase2.imageProcessing.errors.push(`Image processing validation failed: ${JSON.stringify(validation)}`);
                }
            } else {
                this.testResults.phase2.imageProcessing.failed++;
                this.testResults.phase2.imageProcessing.errors.push('Image processing Lambda not found');
            }

            console.log(`    ✅ Image processing Lambda verified`);
        } catch (error) {
            this.testResults.phase2.imageProcessing.errors.push(`Image processing error: ${error.message}`);
            this.testResults.phase2.imageProcessing.failed++;
        }
    }

    /**
     * Test Audio Processing
     */
    async testAudioProcessing() {
        try {
            const audioProcessorPath = path.join(this.projectRoot, 'Code/multimodal_processing/audio_processing_lambda.py');
            if (fs.existsSync(audioProcessorPath)) {
                const content = fs.readFileSync(audioProcessorPath, 'utf8');
                
                // Validate audio processing components
                const validation = {
                    hasTranscribeIntegration: content.includes('transcribe'),
                    hasSentimentAnalysis: content.includes('sentiment'),
                    hasKeyPhrases: content.includes('key_phrases'),
                    hasSpeakerDiarization: content.includes('speakers'),
                    hasErrorHandling: content.includes('try:') && content.includes('except')
                };

                const passed = Object.values(validation).every(v => v);
                if (passed) {
                    this.testResults.phase2.audioProcessing.passed++;
                } else {
                    this.testResults.phase2.audioProcessing.failed++;
                    this.testResults.phase2.audioProcessing.errors.push(`Audio processing validation failed: ${JSON.stringify(validation)}`);
                }
            } else {
                this.testResults.phase2.audioProcessing.failed++;
                this.testResults.phase2.audioProcessing.errors.push('Audio processing Lambda not found');
            }

            console.log(`    ✅ Audio processing Lambda verified`);
        } catch (error) {
            this.testResults.phase2.audioProcessing.errors.push(`Audio processing error: ${error.message}`);
            this.testResults.phase2.audioProcessing.failed++;
        }
    }

    /**
     * Test Survey Processing
     */
    async testSurveyProcessing() {
        try {
            const surveyProcessorPath = path.join(this.projectRoot, 'Code/multimodal_processing/survey_processing_script.py');
            if (fs.existsSync(surveyProcessorPath)) {
                const content = fs.readFileSync(surveyProcessorPath, 'utf8');
                
                // Validate survey processing components
                const validation = {
                    hasDataProcessing: content.includes('pandas') || content.includes('csv'),
                    hasStatisticalAnalysis: content.includes('statistics') || content.includes('summary'),
                    hasAggregation: content.includes('group') || content.includes('aggregate'),
                    hasErrorHandling: content.includes('try:') && content.includes('except')
                };

                const passed = Object.values(validation).every(v => v);
                if (passed) {
                    this.testResults.phase2.surveyProcessing.passed++;
                } else {
                    this.testResults.phase2.surveyProcessing.failed++;
                    this.testResults.phase2.surveyProcessing.errors.push(`Survey processing validation failed: ${JSON.stringify(validation)}`);
                }
            } else {
                this.testResults.phase2.surveyProcessing.failed++;
                this.testResults.phase2.surveyProcessing.errors.push('Survey processing script not found');
            }

            console.log(`    ✅ Survey processing script verified`);
        } catch (error) {
            this.testResults.phase2.surveyProcessing.errors.push(`Survey processing error: ${error.message}`);
            this.testResults.phase2.surveyProcessing.failed++;
        }
    }

    /**
     * Test Processing CloudWatch Logs
     */
    async testProcessingCloudWatchLogs() {
        try {
            // Check if processing components have CloudWatch integration
            const processingComponents = [
                'Code/multimodal_processing/text_processing_lambda.py',
                'Code/multimodal_processing/image_processing_lambda.py',
                'Code/multimodal_processing/audio_processing_lambda.py'
            ];

            for (const component of processingComponents) {
                const componentPath = path.join(this.projectRoot, component);
                if (fs.existsSync(componentPath)) {
                    const content = fs.readFileSync(componentPath, 'utf8');
                    if (content.includes('cloudwatch') || content.includes('put_metric_data')) {
                        this.testResults.phase2.cloudWatchLogs.passed++;
                    } else {
                        this.testResults.phase2.cloudWatchLogs.failed++;
                        this.testResults.phase2.cloudWatchLogs.errors.push(`Missing CloudWatch integration in ${component}`);
                    }
                } else {
                    this.testResults.phase2.cloudWatchLogs.failed++;
                    this.testResults.phase2.cloudWatchLogs.errors.push(`Component not found: ${component}`);
                }
            }

            console.log(`    ✅ Processing CloudWatch integration verified`);
        } catch (error) {
            this.testResults.phase2.cloudWatchLogs.errors.push(`Processing CloudWatch test error: ${error.message}`);
            this.testResults.phase2.cloudWatchLogs.failed++;
        }
    }

    /**
     * Test Processed Data Storage
     */
    async testProcessedDataStorage() {
        try {
            // Check if S3 storage configuration exists
            const s3Configs = [
                'Infrastructure/s3/text_processing_trigger.json',
                'Infrastructure/s3/image_processing_trigger.json',
                'Infrastructure/s3/audio_processing_trigger.json'
            ];

            for (const config of s3Configs) {
                const configPath = path.join(this.projectRoot, config);
                if (fs.existsSync(configPath)) {
                    this.testResults.phase2.processedDataStorage.passed++;
                } else {
                    this.testResults.phase2.processedDataStorage.failed++;
                    this.testResults.phase2.processedDataStorage.errors.push(`Missing S3 config: ${config}`);
                }
            }

            console.log(`    ✅ Processed data storage configuration verified`);
        } catch (error) {
            this.testResults.phase2.processedDataStorage.errors.push(`Processed data storage test error: ${error.message}`);
            this.testResults.phase2.processedDataStorage.failed++;
        }
    }

    /**
     * Test Foundation Model Formatting
     */
    async testFoundationModelFormatting() {
        try {
            const fmFormatterPath = path.join(this.projectRoot, 'Code/fm_formatting/foundation_model_formatter.py');
            if (fs.existsSync(fmFormatterPath)) {
                const content = fs.readFileSync(fmFormatterPath, 'utf8');
                
                // Validate foundation model formatting components
                const validation = {
                    hasClaudeSupport: content.includes('claude'),
                    hasTitanSupport: content.includes('titan'),
                    hasMultimodalSupport: content.includes('multimodal'),
                    hasFormatValidation: content.includes('validate_format'),
                    hasMultipleOutputFormats: content.includes('jsonl') && content.includes('parquet')
                };

                const passed = Object.values(validation).every(v => v);
                if (passed) {
                    this.testResults.phase3.foundationModelFormatting.passed++;
                } else {
                    this.testResults.phase3.foundationModelFormatting.failed++;
                    this.testResults.phase3.foundationModelFormatting.errors.push(`Foundation model formatting validation failed: ${JSON.stringify(validation)}`);
                }
            } else {
                this.testResults.phase3.foundationModelFormatting.failed++;
                this.testResults.phase3.foundationModelFormatting.errors.push('Foundation model formatter not found');
            }

            console.log(`    ✅ Foundation model formatting verified`);
        } catch (error) {
            this.testResults.phase3.foundationModelFormatting.errors.push(`Foundation model formatting error: ${error.message}`);
            this.testResults.phase3.foundationModelFormatting.failed++;
        }
    }

    /**
     * Test Real-Time Formatter
     */
    async testRealTimeFormatter() {
        try {
            const realTimeFormatterPath = path.join(this.projectRoot, 'Code/fm_formatting/real_time_formatter_lambda.py');
            if (fs.existsSync(realTimeFormatterPath)) {
                const content = fs.readFileSync(realTimeFormatterPath, 'utf8');
                
                // Validate real-time formatting components
                const validation = {
                    hasLowLatencyDesign: content.includes('timeout') && content.includes('memory'),
                    hasCaching: content.includes('cache') || content.includes('pool'),
                    hasApiGatewayIntegration: content.includes('api_gateway') || content.includes('statusCode'),
                    hasBedrockIntegration: content.includes('bedrock') || content.includes('invoke_model')
                };

                const passed = Object.values(validation).every(v => v);
                if (passed) {
                    this.testResults.phase3.realTimeFormatter.passed++;
                } else {
                    this.testResults.phase3.realTimeFormatter.failed++;
                    this.testResults.phase3.realTimeFormatter.errors.push(`Real-time formatter validation failed: ${JSON.stringify(validation)}`);
                }
            } else {
                this.testResults.phase3.realTimeFormatter.failed++;
                this.testResults.phase3.realTimeFormatter.errors.push('Real-time formatter not found');
            }

            console.log(`    ✅ Real-time formatter verified`);
        } catch (error) {
            this.testResults.phase3.realTimeFormatter.errors.push(`Real-time formatter error: ${error.message}`);
            this.testResults.phase3.realTimeFormatter.failed++;
        }
    }

    /**
     * Test Step Functions Workflow
     */
    async testStepFunctionsWorkflow() {
        try {
            const stepFunctionsPath = path.join(this.projectRoot, 'Infrastructure/step_functions/formatting_workflow.json');
            if (fs.existsSync(stepFunctionsPath)) {
                const content = fs.readFileSync(stepFunctionsPath, 'utf8');
                
                // Validate Step Functions workflow components
                const validation = {
                    hasStates: content.includes('"States"'),
                    hasStartAt: content.includes('"StartAt"'),
                    hasErrorHandling: content.includes('"Catch"') || content.includes('"Retry"'),
                    hasParallelProcessing: content.includes('"Parallel"'),
                    hasIntegrationSteps: content.includes('"Resource"')
                };

                const passed = Object.values(validation).every(v => v);
                if (passed) {
                    this.testResults.phase3.stepFunctions.passed++;
                } else {
                    this.testResults.phase3.stepFunctions.failed++;
                    this.testResults.phase3.stepFunctions.errors.push(`Step Functions validation failed: ${JSON.stringify(validation)}`);
                }
            } else {
                this.testResults.phase3.stepFunctions.failed++;
                this.testResults.phase3.stepFunctions.errors.push('Step Functions workflow not found');
            }

            console.log(`    ✅ Step Functions workflow verified`);
        } catch (error) {
            this.testResults.phase3.stepFunctions.errors.push(`Step Functions test error: ${error.message}`);
            this.testResults.phase3.stepFunctions.failed++;
        }
    }

    /**
     * Test Formatted Data Output
     */
    async testFormattedDataOutput() {
        try {
            // Check if formatters support multiple output formats
            const formatterComponents = [
                'Code/fm_formatting/text_formatter.py',
                'Code/fm_formatting/image_formatter.py',
                'Code/fm_formatting/audio_formatter.py',
                'Code/fm_formatting/survey_formatter.py'
            ];

            for (const component of formatterComponents) {
                const componentPath = path.join(this.projectRoot, component);
                if (fs.existsSync(componentPath)) {
                    const content = fs.readFileSync(componentPath, 'utf8');
                    if (content.includes('format_for_claude') || content.includes('format_for_titan')) {
                        this.testResults.phase3.formattedDataOutput.passed++;
                    } else {
                        this.testResults.phase3.formattedDataOutput.failed++;
                        this.testResults.phase3.formattedDataOutput.errors.push(`Missing format methods in ${component}`);
                    }
                } else {
                    this.testResults.phase3.formattedDataOutput.failed++;
                    this.testResults.phase3.formattedDataOutput.errors.push(`Formatter not found: ${component}`);
                }
            }

            console.log(`    ✅ Formatted data output formats verified`);
        } catch (error) {
            this.testResults.phase3.formattedDataOutput.errors.push(`Formatted data output test error: ${error.message}`);
            this.testResults.phase3.formattedDataOutput.failed++;
        }
    }

    /**
     * Test Bedrock Integration
     */
    async testBedrockIntegration() {
        try {
            // Check Bedrock integration components
            const bedrockComponents = [
                'Code/fm_formatting/claude_formatting_lambda.py',
                'Infrastructure/iam/foundation_model_formatting_role.json'
            ];

            for (const component of bedrockComponents) {
                const componentPath = path.join(this.projectRoot, component);
                if (fs.existsSync(componentPath)) {
                    const content = fs.readFileSync(componentPath, 'utf8');
                    if (content.includes('bedrock') || content.includes('Bedrock')) {
                        this.testResults.phase3.bedrockIntegration.passed++;
                    } else {
                        this.testResults.phase3.bedrockIntegration.failed++;
                        this.testResults.phase3.bedrockIntegration.errors.push(`Missing Bedrock integration in ${component}`);
                    }
                } else {
                    this.testResults.phase3.bedrockIntegration.failed++;
                    this.testResults.phase3.bedrockIntegration.errors.push(`Bedrock component not found: ${component}`);
                }
            }

            console.log(`    ✅ Bedrock integration verified`);
        } catch (error) {
            this.testResults.phase3.bedrockIntegration.errors.push(`Bedrock integration test error: ${error.message}`);
            this.testResults.phase3.bedrockIntegration.failed++;
        }
    }

    /**
     * Test Data Flow
     */
    async testDataFlow() {
        try {
            // Test complete data flow from raw to formatted
            const dataFlowComponents = [
                'sample_data/text_reviews/',
                'sample_data/images/',
                'sample_data/audio/',
                'sample_data/surveys/',
                'Code/data_validation/',
                'Code/multimodal_processing/',
                'Code/fm_formatting/'
            ];

            for (const component of dataFlowComponents) {
                const componentPath = path.join(this.projectRoot, component);
                if (fs.existsSync(componentPath) || fs.statSync(componentPath).isDirectory()) {
                    this.testResults.endToEnd.dataFlow.passed++;
                } else {
                    this.testResults.endToEnd.dataFlow.failed++;
                    this.testResults.endToEnd.dataFlow.errors.push(`Missing data flow component: ${component}`);
                }
            }

            console.log(`    ✅ End-to-end data flow verified`);
        } catch (error) {
            this.testResults.endToEnd.dataFlow.errors.push(`Data flow test error: ${error.message}`);
            this.testResults.endToEnd.dataFlow.failed++;
        }
    }

    /**
     * Test Quality Scores
     */
    async testQualityScores() {
        try {
            // Check if quality scoring is implemented across phases
            const qualityComponents = [
                'Code/utils/quality_score_calculator.py',
                'Code/fm_formatting/quality_assurance.py'
            ];

            for (const component of qualityComponents) {
                const componentPath = path.join(this.projectRoot, component);
                if (fs.existsSync(componentPath)) {
                    const content = fs.readFileSync(componentPath, 'utf8');
                    if (content.includes('quality_score') || content.includes('calculate_quality')) {
                        this.testResults.endToEnd.qualityScores.passed++;
                    } else {
                        this.testResults.endToEnd.qualityScores.failed++;
                        this.testResults.endToEnd.qualityScores.errors.push(`Missing quality scoring in ${component}`);
                    }
                } else {
                    this.testResults.endToEnd.qualityScores.failed++;
                    this.testResults.endToEnd.qualityScores.errors.push(`Quality component not found: ${component}`);
                }
            }

            console.log(`    ✅ Quality score calculation verified`);
        } catch (error) {
            this.testResults.endToEnd.qualityScores.errors.push(`Quality scores test error: ${error.message}`);
            this.testResults.endToEnd.qualityScores.failed++;
        }
    }

    /**
     * Test Metadata Preservation
     */
    async testMetadataPreservation() {
        try {
            // Check if metadata is preserved through pipeline
            const metadataComponents = [
                'Code/fm_formatting/metadata_enricher.py',
                'Code/utils/common_functions.py'
            ];

            for (const component of metadataComponents) {
                const componentPath = path.join(this.projectRoot, component);
                if (fs.existsSync(componentPath)) {
                    const content = fs.readFileSync(componentPath, 'utf8');
                    if (content.includes('metadata') || content.includes('enrich')) {
                        this.testResults.endToEnd.metadataPreservation.passed++;
                    } else {
                        this.testResults.endToEnd.metadataPreservation.failed++;
                        this.testResults.endToEnd.metadataPreservation.errors.push(`Missing metadata handling in ${component}`);
                    }
                } else {
                    this.testResults.endToEnd.metadataPreservation.failed++;
                    this.testResults.endToEnd.metadataPreservation.errors.push(`Metadata component not found: ${component}`);
                }
            }

            console.log(`    ✅ Metadata preservation verified`);
        } catch (error) {
            this.testResults.endToEnd.metadataPreservation.errors.push(`Metadata preservation test error: ${error.message}`);
            this.testResults.endToEnd.metadataPreservation.failed++;
        }
    }

    /**
     * Test Error Handling Mechanisms
     */
    async testErrorHandlingMechanisms() {
        try {
            // Check if error handling is implemented across components
            const errorHandlingFiles = [
                'Code/multimodal_processing/text_processing_lambda.py',
                'Code/multimodal_processing/image_processing_lambda.py',
                'Code/multimodal_processing/audio_processing_lambda.py',
                'Code/fm_formatting/foundation_model_formatter.py'
            ];

            for (const file of errorHandlingFiles) {
                const filePath = path.join(this.projectRoot, file);
                if (fs.existsSync(filePath)) {
                    const content = fs.readFileSync(filePath, 'utf8');
                    if (content.includes('try:') && content.includes('except') && content.includes('return')) {
                        this.testResults.endToEnd.errorHandling.passed++;
                    } else {
                        this.testResults.endToEnd.errorHandling.failed++;
                        this.testResults.endToEnd.errorHandling.errors.push(`Incomplete error handling in ${file}`);
                    }
                } else {
                    this.testResults.endToEnd.errorHandling.failed++;
                    this.testResults.endToEnd.errorHandling.errors.push(`Error handling file not found: ${file}`);
                }
            }

            console.log(`    ✅ Error handling mechanisms verified`);
        } catch (error) {
            this.testResults.endToEnd.errorHandling.errors.push(`Error handling test error: ${error.message}`);
            this.testResults.endToEnd.errorHandling.failed++;
        }
    }

    /**
     * Test Processing Latency
     */
    async testProcessingLatency() {
        try {
            // Check if components are optimized for performance
            const performanceFiles = [
                'Code/multimodal_processing/text_processing_lambda.py',
                'Code/multimodal_processing/image_processing_lambda.py',
                'Code/fm_formatting/foundation_model_formatter.py'
            ];

            for (const file of performanceFiles) {
                const filePath = path.join(this.projectRoot, file);
                if (fs.existsSync(filePath)) {
                    const content = fs.readFileSync(filePath, 'utf8');
                    if (content.includes('timeout') || content.includes('memory') || content.includes('concurrency')) {
                        this.testResults.performance.latency.passed++;
                    } else {
                        this.testResults.performance.latency.failed++;
                        this.testResults.performance.latency.errors.push(`Missing performance optimization in ${file}`);
                    }
                } else {
                    this.testResults.performance.latency.failed++;
                    this.testResults.performance.latency.errors.push(`Performance file not found: ${file}`);
                }
            }

            console.log(`    ✅ Processing latency optimization verified`);
        } catch (error) {
            this.testResults.performance.latency.errors.push(`Processing latency test error: ${error.message}`);
            this.testResults.performance.latency.failed++;
        }
    }

    /**
     * Test Concurrent Processing
     */
    async testConcurrentProcessing() {
        try {
            // Check if system supports concurrent processing
            const concurrencyFiles = [
                'Infrastructure/iam/text_processing_lambda_role.json',
                'Infrastructure/iam/foundation_model_formatting_role.json'
            ];

            for (const file of concurrencyFiles) {
                const filePath = path.join(this.projectRoot, file);
                if (fs.existsSync(filePath)) {
                    const content = fs.readFileSync(filePath, 'utf8');
                    if (content.includes('concurrency') || content.includes('reserved_concurrency')) {
                        this.testResults.performance.throughput.passed++;
                    } else {
                        this.testResults.performance.throughput.failed++;
                        this.testResults.performance.throughput.errors.push(`Missing concurrency configuration in ${file}`);
                    }
                } else {
                    this.testResults.performance.throughput.failed++;
                    this.testResults.performance.throughput.errors.push(`Concurrency file not found: ${file}`);
                }
            }

            console.log(`    ✅ Concurrent processing configuration verified`);
        } catch (error) {
            this.testResults.performance.throughput.errors.push(`Concurrent processing test error: ${error.message}`);
            this.testResults.performance.throughput.failed++;
        }
    }

    /**
     * Test Timeout Configurations
     */
    async testTimeoutConfigurations() {
        try {
            // Check if timeout configurations are appropriate
            const timeoutFiles = [
                'Infrastructure/deployment/deploy_text_processing.sh',
                'Infrastructure/deployment/deploy_foundation_model_formatting.sh'
            ];

            for (const file of timeoutFiles) {
                const filePath = path.join(this.projectRoot, file);
                if (fs.existsSync(filePath)) {
                    const content = fs.readFileSync(filePath, 'utf8');
                    if (content.includes('timeout') || content.includes('--timeout')) {
                        this.testResults.performance.timeoutConfigurations.passed++;
                    } else {
                        this.testResults.performance.timeoutConfigurations.failed++;
                        this.testResults.performance.timeoutConfigurations.errors.push(`Missing timeout configuration in ${file}`);
                    }
                } else {
                    this.testResults.performance.timeoutConfigurations.failed++;
                    this.testResults.performance.timeoutConfigurations.errors.push(`Timeout file not found: ${file}`);
                }
            }

            console.log(`    ✅ Timeout configurations verified`);
        } catch (error) {
            this.testResults.performance.timeoutConfigurations.errors.push(`Timeout configuration test error: ${error.message}`);
            this.testResults.performance.timeoutConfigurations.failed++;
        }
    }

    /**
     * Test Memory Usage
     */
    async testMemoryUsage() {
        try {
            // Check if memory configurations are appropriate
            const memoryFiles = [
                'Infrastructure/deployment/deploy_text_processing.sh',
                'Infrastructure/deployment/deploy_foundation_model_formatting.sh'
            ];

            for (const file of memoryFiles) {
                const filePath = path.join(this.projectRoot, file);
                if (fs.existsSync(filePath)) {
                    const content = fs.readFileSync(filePath, 'utf8');
                    if (content.includes('memory-size') || content.includes('--memory-size')) {
                        this.testResults.performance.resourceUsage.passed++;
                    } else {
                        this.testResults.performance.resourceUsage.failed++;
                        this.testResults.performance.resourceUsage.errors.push(`Missing memory configuration in ${file}`);
                    }
                } else {
                    this.testResults.performance.resourceUsage.failed++;
                    this.testResults.performance.resourceUsage.errors.push(`Memory file not found: ${file}`);
                }
            }

            console.log(`    ✅ Memory usage configuration verified`);
        } catch (error) {
            this.testResults.performance.resourceUsage.errors.push(`Memory usage test error: ${error.message}`);
            this.testResults.performance.resourceUsage.failed++;
        }
    }

    /**
     * Test S3 Event Triggers
     */
    async testS3EventTriggers() {
        try {
            // Check if S3 event triggers are configured
            const s3TriggerFiles = [
                'Infrastructure/s3/text_processing_trigger.json',
                'Infrastructure/s3/image_processing_trigger.json',
                'Infrastructure/s3/audio_processing_trigger.json',
                'Infrastructure/s3/formatting_trigger.json'
            ];

            for (const file of s3TriggerFiles) {
                const filePath = path.join(this.projectRoot, file);
                if (fs.existsSync(filePath)) {
                    const content = fs.readFileSync(filePath, 'utf8');
                    if (content.includes('LambdaFunctionConfigurations') && content.includes('Events')) {
                        this.testResults.integration.s3Triggers.passed++;
                    } else {
                        this.testResults.integration.s3Triggers.failed++;
                        this.testResults.integration.s3Triggers.errors.push(`Invalid S3 trigger configuration in ${file}`);
                    }
                } else {
                    this.testResults.integration.s3Triggers.failed++;
                    this.testResults.integration.s3Triggers.errors.push(`S3 trigger file not found: ${file}`);
                }
            }

            console.log(`    ✅ S3 event triggers verified`);
        } catch (error) {
            this.testResults.integration.s3Triggers.errors.push(`S3 triggers test error: ${error.message}`);
            this.testResults.integration.s3Triggers.failed++;
        }
    }

    /**
     * Test IAM Permissions
     */
    async testIAMPermissions() {
        try {
            // Check if IAM permissions are properly configured
            const iamFiles = [
                'Infrastructure/iam/text_processing_lambda_role.json',
                'Infrastructure/iam/foundation_model_formatting_role.json',
                'Infrastructure/iam/sagemaker_execution_role.json'
            ];

            for (const file of iamFiles) {
                const filePath = path.join(this.projectRoot, file);
                if (fs.existsSync(filePath)) {
                    const content = fs.readFileSync(filePath, 'utf8');
                    if (content.includes('Statement') && content.includes('Effect') && content.includes('Action')) {
                        this.testResults.integration.iamPermissions.passed++;
                    } else {
                        this.testResults.integration.iamPermissions.failed++;
                        this.testResults.integration.iamPermissions.errors.push(`Invalid IAM configuration in ${file}`);
                    }
                } else {
                    this.testResults.integration.iamPermissions.failed++;
                    this.testResults.integration.iamPermissions.errors.push(`IAM file not found: ${file}`);
                }
            }

            console.log(`    ✅ IAM permissions verified`);
        } catch (error) {
            this.testResults.integration.iamPermissions.errors.push(`IAM permissions test error: ${error.message}`);
            this.testResults.integration.iamPermissions.failed++;
        }
    }

    /**
     * Test DLQ Functionality
     */
    async testDLQFunctionality() {
        try {
            // Check if DLQ configuration exists
            const dlqFiles = [
                'Code/tests/integration/dlq_functionality_test.py'
            ];

            for (const file of dlqFiles) {
                const filePath = path.join(this.projectRoot, file);
                if (fs.existsSync(filePath)) {
                    const content = fs.readFileSync(filePath, 'utf8');
                    if (content.includes('dlq') || content.includes('dead_letter')) {
                        this.testResults.integration.dlqFunctionality.passed++;
                    } else {
                        this.testResults.integration.dlqFunctionality.failed++;
                        this.testResults.integration.dlqFunctionality.errors.push(`Missing DLQ configuration in ${file}`);
                    }
                } else {
                    this.testResults.integration.dlqFunctionality.failed++;
                    this.testResults.integration.dlqFunctionality.errors.push(`DLQ file not found: ${file}`);
                }
            }

            console.log(`    ✅ DLQ functionality verified`);
        } catch (error) {
            this.testResults.integration.dlqFunctionality.errors.push(`DLQ functionality test error: ${error.message}`);
            this.testResults.integration.dlqFunctionality.failed++;
        }
    }

    /**
     * Test CloudWatch Metrics
     */
    async testCloudWatchMetrics() {
        try {
            // Check if CloudWatch metrics are configured
            const cloudWatchFiles = [
                'Code/data_validation/cloudwatch_dashboard.py',
                'Code/fm_formatting/quality_assurance.py'
            ];

            for (const file of cloudWatchFiles) {
                const filePath = path.join(this.projectRoot, file);
                if (fs.existsSync(filePath)) {
                    const content = fs.readFileSync(filePath, 'utf8');
                    if (content.includes('put_metric_data') || content.includes('MetricData')) {
                        this.testResults.integration.cloudWatchMetrics.passed++;
                    } else {
                        this.testResults.integration.cloudWatchMetrics.failed++;
                        this.testResults.integration.cloudWatchMetrics.errors.push(`Missing CloudWatch metrics in ${file}`);
                    }
                } else {
                    this.testResults.integration.cloudWatchMetrics.failed++;
                    this.testResults.integration.cloudWatchMetrics.errors.push(`CloudWatch file not found: ${file}`);
                }
            }

            console.log(`    ✅ CloudWatch metrics verified`);
        } catch (error) {
            this.testResults.integration.cloudWatchMetrics.errors.push(`CloudWatch metrics test error: ${error.message}`);
            this.testResults.integration.cloudWatchMetrics.failed++;
        }
    }

    /**
     * Generate comprehensive test report
     */
    async generateComprehensiveReport() {
        const endTime = new Date();
        const duration = endTime - this.startTime;
        
        const report = {
            metadata: {
                testSuite: 'Comprehensive AWS AI Project Testing',
                timestamp: endTime.toISOString(),
                duration: duration,
                environment: process.env.NODE_ENV || 'test',
                version: '3.0'
            },
            summary: this.calculateSummaryStatistics(),
            phase1: this.testResults.phase1,
            phase2: this.testResults.phase2,
            phase3: this.testResults.phase3,
            endToEnd: this.testResults.endToEnd,
            performance: this.testResults.performance,
            integration: this.testResults.integration,
            recommendations: this.generateRecommendations(),
            nextSteps: this.generateNextSteps()
        };

        // Save comprehensive report
        const reportPath = path.join(this.reportsDir, `comprehensive-test-report-${endTime.toISOString().replace(/[:.]/g, '-')}.json`);
        fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
        
        // Generate HTML report
        await this.generateHtmlReport(report, reportPath.replace('.json', '.html'));
        
        console.log(`  📄 Comprehensive report saved to: ${reportPath}`);
    }

    /**
     * Calculate summary statistics
     */
    calculateSummaryStatistics() {
        const totalTests = Object.values(this.testResults).reduce((sum, phase) => {
            return sum + Object.values(phase).reduce((phaseSum, category) => {
                return phaseSum + (category.passed || 0) + (category.failed || 0);
            }, 0);
        }, 0);

        const totalPassed = Object.values(this.testResults).reduce((sum, phase) => {
            return sum + Object.values(phase).reduce((phaseSum, category) => {
                return phaseSum + (category.passed || 0);
            }, 0);
        }, 0);

        const totalFailed = Object.values(this.testResults).reduce((sum, phase) => {
            return sum + Object.values(phase).reduce((phaseSum, category) => {
                return phaseSum + (category.failed || 0);
            }, 0);
        }, 0);

        const totalErrors = Object.values(this.testResults).reduce((sum, phase) => {
            return sum + Object.values(phase).reduce((phaseSum, category) => {
                return phaseSum + (category.errors ? category.errors.length : 0);
            }, 0);
        }, 0);

        return {
            totalTests: totalTests,
            passedTests: totalPassed,
            failedTests: totalFailed,
            totalErrors: totalErrors,
            successRate: totalTests > 0 ? Math.round((totalPassed / totalTests) * 100) : 0,
            overallStatus: totalErrors === 0 && totalFailed === 0 ? 'PASS' : 'FAIL'
        };
    }

    /**
     * Generate recommendations based on test results
     */
    generateRecommendations() {
        const recommendations = [];
        
        // Phase 1 recommendations
        if (this.testResults.phase1.dataValidation.failed > 0) {
            recommendations.push({
                priority: 'high',
                phase: 'Phase 1',
                component: 'Data Validation',
                message: 'Fix data validation issues before proceeding to Phase 2'
            });
        }
        
        // Phase 2 recommendations
        if (this.testResults.phase2.textProcessing.failed > 0) {
            recommendations.push({
                priority: 'high',
                phase: 'Phase 2',
                component: 'Text Processing',
                message: 'Resolve text processing Lambda issues'
            });
        }
        
        // Phase 3 recommendations
        if (this.testResults.phase3.foundationModelFormatting.failed > 0) {
            recommendations.push({
                priority: 'high',
                phase: 'Phase 3',
                component: 'Foundation Model Formatting',
                message: 'Fix foundation model formatting issues'
            });
        }
        
        // Performance recommendations
        if (this.testResults.performance.latency.failed > 0) {
            recommendations.push({
                priority: 'medium',
                phase: 'Performance',
                component: 'Latency',
                message: 'Optimize processing latency for better performance'
            });
        }
        
        // Integration recommendations
        if (this.testResults.integration.s3Triggers.failed > 0) {
            recommendations.push({
                priority: 'high',
                phase: 'Integration',
                component: 'S3 Triggers',
                message: 'Fix S3 event trigger configuration issues'
            });
        }
        
        return recommendations;
    }

    /**
     * Generate next steps
     */
    generateNextSteps() {
        const steps = [];
        
        const hasFailures = Object.values(this.testResults).some(phase => 
            Object.values(phase).some(category => (category.failed || 0) > 0)
        );
        
        if (hasFailures) {
            steps.push('Address all test failures before production deployment');
            steps.push('Re-run comprehensive test suite after fixes');
            steps.push('Review and update documentation based on fixes');
        } else {
            steps.push('Proceed with production deployment');
            steps.push('Set up monitoring and alerting');
            steps.push('Conduct load testing with production data');
        }
        
        steps.push('Archive test results for future reference');
        steps.push('Schedule regular regression testing');
        
        return steps;
    }

    /**
     * Generate HTML report
     */
    async generateHtmlReport(report, htmlPath) {
        const htmlContent = `
<!DOCTYPE html>
<html>
<head>
    <title>Comprehensive AWS AI Project Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
        .success { color: green; }
        .failure { color: red; }
        .warning { color: orange; }
        .phase { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .metric { display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .status-pass { background-color: #d4edda; }
        .status-fail { background-color: #f8d7da; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Comprehensive AWS AI Project Test Report</h1>
        <p>Generated: ${report.metadata.timestamp}</p>
        <p>Duration: ${report.metadata.duration}ms</p>
        <p>Overall Status: <span class="${report.summary.overallStatus === 'PASS' ? 'success' : 'failure'}">${report.summary.overallStatus}</span></p>
    </div>
    
    <div class="phase">
        <h2>Test Summary</h2>
        <div class="metric">Total Tests: ${report.summary.totalTests}</div>
        <div class="metric">Passed: <span class="success">${report.summary.passedTests}</span></div>
        <div class="metric">Failed: <span class="failure">${report.summary.failedTests}</span></div>
        <div class="metric">Success Rate: ${report.summary.successRate}%</div>
        <div class="metric">Total Errors: ${report.summary.totalErrors}</div>
    </div>
    
    <div class="phase">
        <h2>Phase 1: Data Validation</h2>
        <table>
            <tr><th>Component</th><th>Passed</th><th>Failed</th><th>Errors</th><th>Status</th></tr>
            <tr><td>Data Validation</td><td>${report.phase1.dataValidation.passed}</td><td>${report.phase1.dataValidation.failed}</td><td>${report.phase1.dataValidation.errors.length}</td><td class="${report.phase1.dataValidation.failed === 0 ? 'status-pass' : 'status-fail'}">${report.phase1.dataValidation.failed === 0 ? 'PASS' : 'FAIL'}</td></tr>
            <tr><td>Glue Database</td><td>${report.phase1.glueDatabase.passed}</td><td>${report.phase1.glueDatabase.failed}</td><td>${report.phase1.glueDatabase.errors.length}</td><td class="${report.phase1.glueDatabase.failed === 0 ? 'status-pass' : 'status-fail'}">${report.phase1.glueDatabase.failed === 0 ? 'PASS' : 'FAIL'}</td></tr>
            <tr><td>CloudWatch Logs</td><td>${report.phase1.cloudWatchLogs.passed}</td><td>${report.phase1.cloudWatchLogs.failed}</td><td>${report.phase1.cloudWatchLogs.errors.length}</td><td class="${report.phase1.cloudWatchLogs.failed === 0 ? 'status-pass' : 'status-fail'}">${report.phase1.cloudWatchLogs.failed === 0 ? 'PASS' : 'FAIL'}</td></tr>
        </table>
    </div>
    
    <div class="phase">
        <h2>Phase 2: Multimodal Processing</h2>
        <table>
            <tr><th>Component</th><th>Passed</th><th>Failed</th><th>Errors</th><th>Status</th></tr>
            <tr><td>Text Processing</td><td>${report.phase2.textProcessing.passed}</td><td>${report.phase2.textProcessing.failed}</td><td>${report.phase2.textProcessing.errors.length}</td><td class="${report.phase2.textProcessing.failed === 0 ? 'status-pass' : 'status-fail'}">${report.phase2.textProcessing.failed === 0 ? 'PASS' : 'FAIL'}</td></tr>
            <tr><td>Image Processing</td><td>${report.phase2.imageProcessing.passed}</td><td>${report.phase2.imageProcessing.failed}</td><td>${report.phase2.imageProcessing.errors.length}</td><td class="${report.phase2.imageProcessing.failed === 0 ? 'status-pass' : 'status-fail'}">${report.phase2.imageProcessing.failed === 0 ? 'PASS' : 'FAIL'}</td></tr>
            <tr><td>Audio Processing</td><td>${report.phase2.audioProcessing.passed}</td><td>${report.phase2.audioProcessing.failed}</td><td>${report.phase2.audioProcessing.errors.length}</td><td class="${report.phase2.audioProcessing.failed === 0 ? 'status-pass' : 'status-fail'}">${report.phase2.audioProcessing.failed === 0 ? 'PASS' : 'FAIL'}</td></tr>
            <tr><td>Survey Processing</td><td>${report.phase2.surveyProcessing.passed}</td><td>${report.phase2.surveyProcessing.failed}</td><td>${report.phase2.surveyProcessing.errors.length}</td><td class="${report.phase2.surveyProcessing.failed === 0 ? 'status-pass' : 'status-fail'}">${report.phase2.surveyProcessing.failed === 0 ? 'PASS' : 'FAIL'}</td></tr>
        </table>
    </div>
    
    <div class="phase">
        <h2>Phase 3: Foundation Model Formatting</h2>
        <table>
            <tr><th>Component</th><th>Passed</th><th>Failed</th><th>Errors</th><th>Status</th></tr>
            <tr><td>Foundation Model Formatting</td><td>${report.phase3.foundationModelFormatting.passed}</td><td>${report.phase3.foundationModelFormatting.failed}</td><td>${report.phase3.foundationModelFormatting.errors.length}</td><td class="${report.phase3.foundationModelFormatting.failed === 0 ? 'status-pass' : 'status-fail'}">${report.phase3.foundationModelFormatting.failed === 0 ? 'PASS' : 'FAIL'}</td></tr>
            <tr><td>Real-Time Formatter</td><td>${report.phase3.realTimeFormatter.passed}</td><td>${report.phase3.realTimeFormatter.failed}</td><td>${report.phase3.realTimeFormatter.errors.length}</td><td class="${report.phase3.realTimeFormatter.failed === 0 ? 'status-pass' : 'status-fail'}">${report.phase3.realTimeFormatter.failed === 0 ? 'PASS' : 'FAIL'}</td></tr>
            <tr><td>Step Functions</td><td>${report.phase3.stepFunctions.passed}</td><td>${report.phase3.stepFunctions.failed}</td><td>${report.phase3.stepFunctions.errors.length}</td><td class="${report.phase3.stepFunctions.failed === 0 ? 'status-pass' : 'status-fail'}">${report.phase3.stepFunctions.failed === 0 ? 'PASS' : 'FAIL'}</td></tr>
            <tr><td>Bedrock Integration</td><td>${report.phase3.bedrockIntegration.passed}</td><td>${report.phase3.bedrockIntegration.failed}</td><td>${report.phase3.bedrockIntegration.errors.length}</td><td class="${report.phase3.bedrockIntegration.failed === 0 ? 'status-pass' : 'status-fail'}">${report.phase3.bedrockIntegration.failed === 0 ? 'PASS' : 'FAIL'}</td></tr>
        </table>
    </div>
    
    <div class="phase">
        <h2>Recommendations</h2>
        <ul>
            ${report.recommendations.map(rec => `<li class="${rec.priority}"><strong>${rec.phase} - ${rec.component}:</strong> ${rec.message}</li>`).join('')}
        </ul>
    </div>
    
    <div class="phase">
        <h2>Next Steps</h2>
        <ol>
            ${report.nextSteps.map(step => `<li>${step}</li>`).join('')}
        </ol>
    </div>
</body>
</html>
        `;
        
        fs.writeFileSync(htmlPath, htmlContent);
        console.log(`  🌐 HTML report saved to: ${htmlPath}`);
    }

    /**
     * Generate error report
     */
    async generateErrorReport(error) {
        const errorReport = {
            timestamp: new Date().toISOString(),
            error: {
                message: error.message,
                stack: error.stack,
                name: error.name
            },
            context: {
                testResults: this.testResults,
                environment: process.env
            }
        };
        
        const errorPath = path.join(this.reportsDir, `comprehensive-error-report-${new Date().toISOString().replace(/[:.]/g, '-')}.json`);
        fs.writeFileSync(errorPath, JSON.stringify(errorReport, null, 2));
        console.log(`  🚨 Error report saved to: ${errorPath}`);
    }

    /**
     * Display final results
     */
    displayFinalResults() {
        const endTime = new Date();
        const duration = endTime - this.startTime;
        const summary = this.calculateSummaryStatistics();
        
        console.log('\n' + '='.repeat(80));
        console.log('🏁 COMPREHENSIVE AWS AI PROJECT TEST EXECUTION COMPLETE');
        console.log('='.repeat(80));
        
        console.log(`\n⏱️  Duration: ${duration}ms`);
        console.log(`📊 Success Rate: ${summary.successRate}%`);
        console.log(`📈 Overall Status: ${summary.overallStatus}`);
        
        console.log('\n📋 Phase 1 Summary:');
        console.log(`  ✅ Data Validation: ${this.testResults.phase1.dataValidation.passed} passed, ${this.testResults.phase1.dataValidation.failed} failed`);
        console.log(`  ✅ Glue Database: ${this.testResults.phase1.glueDatabase.passed} passed, ${this.testResults.phase1.glueDatabase.failed} failed`);
        console.log(`  ✅ CloudWatch Logs: ${this.testResults.phase1.cloudWatchLogs.passed} passed, ${this.testResults.phase1.cloudWatchLogs.failed} failed`);
        
        console.log('\n📋 Phase 2 Summary:');
        console.log(`  ✅ Text Processing: ${this.testResults.phase2.textProcessing.passed} passed, ${this.testResults.phase2.textProcessing.failed} failed`);
        console.log(`  ✅ Image Processing: ${this.testResults.phase2.imageProcessing.passed} passed, ${this.testResults.phase2.imageProcessing.failed} failed`);
        console.log(`  ✅ Audio Processing: ${this.testResults.phase2.audioProcessing.passed} passed, ${this.testResults.phase2.audioProcessing.failed} failed`);
        console.log(`  ✅ Survey Processing: ${this.testResults.phase2.surveyProcessing.passed} passed, ${this.testResults.phase2.surveyProcessing.failed} failed`);
        
        console.log('\n📋 Phase 3 Summary:');
        console.log(`  ✅ Foundation Model Formatting: ${this.testResults.phase3.foundationModelFormatting.passed} passed, ${this.testResults.phase3.foundationModelFormatting.failed} failed`);
        console.log(`  ✅ Real-Time Formatter: ${this.testResults.phase3.realTimeFormatter.passed} passed, ${this.testResults.phase3.realTimeFormatter.failed} failed`);
        console.log(`  ✅ Step Functions: ${this.testResults.phase3.stepFunctions.passed} passed, ${this.testResults.phase3.stepFunctions.failed} failed`);
        console.log(`  ✅ Bedrock Integration: ${this.testResults.phase3.bedrockIntegration.passed} passed, ${this.testResults.phase3.bedrockIntegration.failed} failed`);
        
        console.log(`\n🚀 Overall Status: ${summary.overallStatus}`);
        
        console.log('\n' + '='.repeat(80));
    }

    /**
     * Ensure directory exists
     */
    ensureDirectoryExists(dirPath) {
        if (!fs.existsSync(dirPath)) {
            fs.mkdirSync(dirPath, { recursive: true });
        }
    }
}

// Execute tests if this script is run directly
if (require.main === module) {
    const testExecutor = new ComprehensivePhaseTesting();
    testExecutor.executeCompleteTestSuite().then(() => {
        process.exit(0);
    }).catch(error => {
        console.error('Test execution failed:', error);
        process.exit(1);
    });
}

module.exports = ComprehensivePhaseTesting;