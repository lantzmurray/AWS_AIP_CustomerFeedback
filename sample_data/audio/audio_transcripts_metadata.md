# Audio Transcripts Metadata

## Overview
This directory contains audio transcript files designed for text-to-speech generation in the AWS AI project. These transcripts represent realistic customer feedback audio recordings that correspond to survey responses from customers who have uploaded audio files.

## File Naming Convention
The transcript files follow the naming convention:
```
transcript_CUST-XXXXX.txt
```
where `CUST-XXXXX` matches the customer ID from the survey data. Each transcript is created for a customer who has `has_uploaded_audio = "Yes"` in the customer feedback survey data.

## Relationship to Survey Data
Each transcript file is directly linked to a customer's survey response and maintains consistency with:
- Customer ID from the survey
- Submission date (used as the audio recording date)
- Overall rating and specific category ratings
- Written feedback text content
- Customer sentiment and emotional tone

The transcripts expand upon the written feedback by providing more detailed, conversational expressions of the customer's experience, including natural speech patterns, emotional indicators, and additional context that would typically be found in audio feedback.

## Transcript Types Included

### 1. Short, Concise Feedback (30-60 seconds when spoken)
- Example: `transcript_CUST-00007.txt`, `transcript_CUST-00028.txt`
- Characteristics: Brief, to-the-point, limited emotional expression
- Use Case: Quick feedback from busy customers or those with less detailed experiences

### 2. Detailed, Emotional Feedback (2-3 minutes when spoken)
- Example: `transcript_CUST-00006.txt`, `transcript_CUST-00012.txt`, `transcript_CUST-00021.txt`
- Characteristics: Rich emotional content, storytelling elements, detailed experiences
- Use Case: Customers with strong feelings about their experience who want to share comprehensive feedback

### 3. Technical Feedback with Specific Details
- Example: `transcript_CUST-00015.txt`
- Characteristics: Technical terminology, specific product details, performance metrics
- Use Case: Technically-inclined customers providing detailed product analysis

### 4. Customer Service Call Transcripts
- Example: `transcript_CUST-00018.txt`
- Characteristics: Dialogue format, includes both customer and service representative speech
- Use Case: Documenting specific customer service interactions and experiences

### 5. Product Experience Narratives
- Example: `transcript_CUST-00004.txt`, `transcript_CUST-00021.txt`
- Characteristics: Storytelling format, chronological experience description
- Use Case: Customers sharing their complete journey from purchase to product use

## Natural Speech Elements
All transcripts include natural speech elements to ensure realistic text-to-speech generation:

### Pauses and Timing
- `[pause]` - Short pauses for breath or thought
- `[short pause]` - Brief pauses between phrases
- `[long pause]` - Extended pauses for emphasis
- `[thoughtful pause]` - Deliberate thinking pauses

### Emotional Indicators
- `[light laugh]` - Light laughter or amusement
- `[sigh]` - Expressions of frustration or disappointment
- `[angry sigh]` - Angry or frustrated exhalation
- `[frustrated sigh]` - Frustration with the experience

### Emphasis and Tone
- `[emphasizes]` - Words or phrases with strong emphasis
- `[enthusiastic tone]` - Excited, positive emotional state
- `[angry tone]` - Angry or frustrated emotional state
- `[neutral tone]` - Unemotional, factual delivery
- `[disappointed tone]` - Sad or disappointed emotional state

### Speech Patterns
- Filler words and natural conversation patterns
- False starts and self-corrections
- Natural rhythm and flow of speech
- Conversational language rather than formal writing

## Integration with AWS Polly
These transcripts are optimized for use with AWS Polly text-to-speech service:

### Recommended SSML Tags
When processing with Polly, consider using these SSML tags:
- `<break time="1s"/>` for pauses
- `<emphasis level="strong">` for emphasized words
- `<amazon:effect name="whispered">` for whispered speech
- `<prosody rate="slow">` for slower speech

### Voice Selection
Different customer personas may be better represented by different Polly voices:
- Male voices for certain customer demographics
- Female voices for others
- Neural voices for more natural speech patterns

## Audio Processing Pipeline Integration

### Lambda Function Processing
The audio processing Lambda function can:
1. Read transcript files from S3
2. Generate audio using AWS Polly
3. Store generated audio files in the appropriate S3 bucket
4. Update the database with audio file locations

### Data Relationships
Each generated audio file should maintain a relationship to:
- Original customer ID
- Survey response data
- Transcript file
- Audio metadata (duration, voice used, generation timestamp)

### Quality Assurance
When generating audio from these transcripts:
- Verify speech naturalness
- Check appropriate pause timing
- Ensure emotional tone matches content
- Validate audio quality metrics

## Usage Guidelines

### For Text-to-Speech Generation
1. Select appropriate voice based on customer persona
2. Apply appropriate SSML tags for speech elements
3. Generate audio at consistent quality settings
4. Store with appropriate metadata

### For Testing and Development
1. Use various transcript types to test different scenarios
2. Verify audio processing pipeline handles all speech elements
3. Test with different voice options and settings
4. Validate integration with survey data

### For Analysis and Machine Learning
1. Transcripts can be used for sentiment analysis training
2. Speech patterns can inform customer experience models
3. Emotional indicators can enhance customer satisfaction predictions
4. Transcript diversity ensures robust model training

## Future Enhancements
Potential improvements to the transcript collection:
- Add more diverse customer demographics
- Include additional languages and accents
- Expand transcript types and scenarios
- Incorporate industry-specific terminology
- Add more complex emotional expressions