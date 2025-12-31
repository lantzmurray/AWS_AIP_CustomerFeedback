# Audio Generation Guide for AWS Polly

## Overview
This guide provides recommendations for generating natural-sounding audio from the transcript files using AWS Polly text-to-speech service. It includes voice recommendations, synthesis settings, and integration guidance with the audio processing pipeline.

## Recommended AWS Polly Voices for Different Customer Personas

### Male Voices

#### Matthew (US English, Neural)
- **Best for**: Professional, detailed feedback
- **Characteristics**: Clear, articulate, professional tone
- **Recommended for**: Technical feedback (transcript_CUST-00015.txt), customer service calls (transcript_CUST-00018.txt)
- **Sample Use Case**: "The technical performance is particularly disappointing. The product claims to support 4K resolution at 60Hz..."

#### Joey (US English, Neural)
- **Best for**: Casual, conversational feedback
- **Characteristics**: Friendly, approachable, natural-sounding
- **Recommended for**: Product experience narratives (transcript_CUST-00004.txt, transcript_CUST-00021.txt)
- **Sample Use Case**: "Hi there! I wanted to share the story of my experience with this product because it's been such a game-changer for me."

#### Brian (US English, Neural)
- **Best for**: Neutral, factual feedback
- **Characteristics**: Balanced tone, clear enunciation
- **Recommended for**: Neutral experiences (transcript_CUST-00028.txt)
- **Sample Use Case**: "Hi, just wanted to share some feedback on my recent purchase. Overall, it's a decent product that does what it's supposed to do."

### Female Voices

#### Joanna (US English, Neural)
- **Best for**: Enthusiastic, positive feedback
- **Characteristics**: Warm, expressive, engaging
- **Recommended for**: Detailed positive feedback (transcript_CUST-00012.txt)
- **Sample Use Case**: "Hello! I just had to share my experience because it was absolutely PERFECT from start to finish."

#### Kendra (US English, Neural)
- **Best for**: Professional, articulate feedback
- **Characteristics**: Clear pronunciation, professional tone
- **Recommended for**: Detailed experiences (transcript_CUST-00006.txt)
- **Sample Use Case**: "Hello, I wanted to share my experience with your product because it's been quite a journey..."

#### Kimberly (US English, Neural)
- **Best for**: Emotional, expressive feedback
- **Characteristics**: Expressive, emotional range
- **Recommended for**: Negative experiences (transcript_CUST-00007.txt)
- **Sample Use Case**: "I am absolutely FURIOUS with my experience. The product arrived broken - completely unusable."

## Speech Synthesis Settings

### Speech Rate (Words per Minute)
- **Slow (120-140 WPM)**: For technical content or when emphasizing important points
  ```xml
  <prosody rate="slow">The technical performance is particularly disappointing.</prosody>
  ```
- **Normal (140-160 WPM)**: Default for most conversational content
  ```xml
  <prosody rate="medium">Hi there! I wanted to share my experience.</prosody>
  ```
- **Fast (160-180 WPM)**: For excited or enthusiastic speech
  ```xml
  <prosody rate="fast">It's been absolutely fantastic!</prosody>
  ```

### Pitch Adjustments
- **Lower Pitch**: For serious or disappointed content
  ```xml
  <prosody pitch="-20%">I am absolutely FURIOUS with my experience.</prosody>
  ```
- **Normal Pitch**: Default for neutral content
- **Higher Pitch**: For excited or enthusiastic content
  ```xml
  <prosody pitch="+20%">It's been absolutely fantastic!</prosody>
  ```

### Volume Adjustments
- **Louder**: For emphasis or strong emotions
  ```xml
  <prosody volume="loud">The product arrived broken!</prosody>
  ```
- **Normal**: Default volume level
- **Softer**: For thoughtful or reflective moments
  ```xml
  <prosody volume="soft">I'm not really sure what to think...</prosody>
  ```

## SSML Implementation for Natural Speech Elements

### Pauses
```xml
<!-- Short pause -->
<break time="500ms"/>

<!-- Medium pause -->
<break time="1s"/>

<!-- Long pause -->
<break time="2s"/>

<!-- Very long pause -->
<break time="3s"/>
```

### Emphasis
```xml
<!-- Moderate emphasis -->
<emphasis level="moderate">The product quality is good.</emphasis>

<!-- Strong emphasis -->
<emphasis level="strong">The product arrived BROKEN!</emphasis>

<!-- Reduced emphasis -->
<emphasis level="reduced">It was just okay.</emphasis>
```

### Emotional Effects
```xml
<!-- Whispered speech -->
<amazon:effect name="whispered">I can't believe this happened.</amazon:effect>

<!-- Excited speech -->
<prosody pitch="+10%" rate="fast">It's absolutely fantastic!</prosody>

<!-- Disappointed speech -->
<prosody pitch="-15%" rate="slow">I'm really disappointed.</prosody>
```

## Integration with Audio Processing Lambda Function

### Sample Lambda Function Code
```python
import boto3
import os
from datetime import datetime

def generate_audio_from_transcript(transcript_path, customer_id, voice_id="Joanna"):
    """
    Generate audio from transcript using AWS Polly
    """
    # Read transcript content
    s3_client = boto3.client('s3')
    polly_client = boto3.client('polly')
    
    # Get transcript from S3
    bucket_name = os.environ['S3_BUCKET']
    transcript_key = f"transcripts/{transcript_path}"
    
    response = s3_client.get_object(Bucket=bucket_name, Key=transcript_key)
    transcript_content = response['Body'].read().decode('utf-8')
    
    # Extract just the transcript content (skip metadata)
    lines = transcript_content.split('\n')
    transcript_text = '\n'.join([line for line in lines if not line.startswith('Customer ID:') and not line.startswith('Date of Audio Recording:')])
    
    # Convert transcript indicators to SSML
    ssml_text = convert_to_ssml(transcript_text)
    
    # Generate speech
    response = polly_client.synthesize_speech(
        Engine='neural',
        LanguageCode='en-US',
        VoiceId=voice_id,
        OutputFormat='mp3',
        TextType='ssml',
        Text=ssml_text
    )
    
    # Save audio to S3
    audio_key = f"audio/{customer_id}.mp3"
    s3_client.put_object(
        Bucket=bucket_name,
        Key=audio_key,
        Body=response['AudioStream'].read(),
        ContentType='audio/mpeg'
    )
    
    return audio_key

def convert_to_ssml(transcript_text):
    """
    Convert transcript indicators to SSML format
    """
    # Replace pause indicators with SSML breaks
    ssml_text = transcript_text.replace('[pause]', '<break time="1s"/>')
    ssml_text = ssml_text.replace('[short pause]', '<break time="500ms"/>')
    ssml_text = ssml_text.replace('[long pause]', '<break time="2s"/>')
    ssml_text = ssml_text.replace('[thoughtful pause]', '<break time="1.5s"/>')
    
    # Replace emphasis indicators with SSML emphasis
    ssml_text = ssml_text.replace('*emphasis*', '<emphasis level="strong">')
    ssml_text = ssml_text.replace('*emphasis*', '</emphasis>')
    
    # Replace emotional indicators with appropriate prosody
    ssml_text = ssml_text.replace('[light laugh]', '<prosody pitch="+10%" rate="fast">[light laugh]</prosody>')
    ssml_text = ssml_text.replace('[sigh]', '<prosody pitch="-10%" rate="slow">[sigh]</prosody>')
    ssml_text = ssml_text.replace('[angry sigh]', '<prosody pitch="-20%" volume="loud">[angry sigh]</prosody>')
    
    # Wrap in SSML root element
    ssml_text = f"<speak>{ssml_text}</speak>"
    
    return ssml_text
```

## Voice Selection Strategy

### Based on Customer Rating
- **5-star ratings**: Use enthusiastic voices (Joanna, Joey)
- **4-star ratings**: Use positive but balanced voices (Matthew, Kendra)
- **3-star ratings**: Use neutral voices (Brian, Kimberly)
- **1-2 star ratings**: Use disappointed or frustrated voices (lower pitch, slower rate)

### Based on Feedback Type
- **Technical feedback**: Use precise, articulate voices (Matthew, Kendra)
- **Emotional feedback**: Use expressive voices (Joanna, Kimberly)
- **Customer service calls**: Use conversational voices (Joey, Brian)
- **Product narratives**: Use storytelling voices (Joanna, Joey)

## Quality Assurance Checklist

### Audio Quality
- [ ] Clear audio without background noise
- [ ] Consistent volume levels throughout
- [ ] Natural speech rhythm and timing
- [ ] Appropriate emotional tone for content

### Content Accuracy
- [ ] All transcript text converted to speech
- [ ] Proper pronunciation of technical terms
- [ ] Appropriate emphasis on key points
- [ ] Natural pause placement

### Technical Specifications
- [ ] Audio format: MP3
- [ ] Sample rate: 22050 Hz
- [ ] Bitrate: 64 kbps (mono) or 128 kbps (stereo)
- [ ] File size optimized for streaming

## Performance Optimization

### Batch Processing
```python
def batch_generate_audio(transcript_list):
    """
    Generate audio for multiple transcripts in batch
    """
    for transcript in transcript_list:
        try:
            # Select voice based on customer profile
            voice_id = select_voice_for_customer(transcript['customer_id'])
            
            # Generate audio
            audio_key = generate_audio_from_transcript(
                transcript['path'],
                transcript['customer_id'],
                voice_id
            )
            
            # Update database
            update_audio_record(transcript['customer_id'], audio_key)
            
        except Exception as e:
            log_error(f"Failed to generate audio for {transcript['customer_id']}: {str(e)}")
```

### Caching Strategy
- Cache frequently used voice profiles
- Store generated audio with appropriate cache headers
- Implement CDN distribution for audio files

## Monitoring and Metrics

### Key Performance Indicators
- Audio generation success rate
- Average processing time per transcript
- Audio file quality metrics
- User engagement with generated audio

### Error Handling
- Retry logic for failed Polly requests
- Fallback voice options
- Graceful degradation for unsupported SSML features

## Future Enhancements

### Advanced SSML Features
- Custom pronunciation dictionaries
- Dynamic voice selection based on content analysis
- Real-time voice parameter adjustment

### Multilingual Support
- Additional language voices
- Language detection in transcripts
- Cross-language voice consistency

### Personalization
- Voice preference learning
- Adaptive speech synthesis
- Context-aware voice selection