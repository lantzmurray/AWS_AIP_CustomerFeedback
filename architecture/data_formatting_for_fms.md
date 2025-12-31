# Data Formatting for Foundation Models Architecture

## 1. Purpose

This module defines how validated and enriched customer feedback is transformed into **foundation model–ready requests**, with a focus on Claude in Amazon Bedrock. It also covers how responses are post-processed into structured insights.

---

## 2. Foundation Model Integration

### 2.1 Claude in Amazon Bedrock

- **Primary Role**
  - Generate actionable insights, summaries, recommendations, and categorization from customer feedback.

- **Expected Input Format**
  - Structured, conversation-style payloads including:
    - System messages (instructions and policies)
    - User messages (customer feedback context)
    - Optional multimodal content (e.g., images)

- **Capabilities Used**
  - Text analysis
  - Multimodal image + text understanding
  - Dialogue-style reasoning

---

### 2.2 Conversation Templates

To ensure consistent, high-quality prompts, the system uses **conversation templates**:

- **Structure**
  - `system` messages for task definition, constraints, and safety.
  - `user` messages containing the feedback and context.
  - Optional `assistant` examples for few-shot guidance.

- **Components**
  - Domain-specific instructions (e.g., retail products vs. services).
  - Required output schema (JSON where relevant).
  - Context fields (customer segment, product category, channel).

- **Customization**
  - Different templates for:
    - Sentiment and theme analysis
    - Root-cause analysis
    - Summarization
    - Recommendation generation

---

## 3. Formatting Strategies by Modality

### 3.1 Text Data Formatting

- **Input**
  - Processed text reviews, including entities, sentiment, and key phrases.

- **Output**
  - Conversation objects that include:
    - Brief context summary
    - Raw or lightly cleaned review text
    - Explicit analysis request (e.g., “Identify top 3 issues and recommendations.”)

- **Key Fields**
  - `customer_feedback_text`
  - `derived_entities`
  - `sentiment_summary`
  - `requested_outputs` (e.g., issues, themes, recommendations)

---

### 3.2 Image Data Formatting

- **Input**
  - Image-derived artifacts from the multimodal pipeline:
    - Extracted text (Textract)
    - Labels and objects (Rekognition)

- **Output**
  - Multimodal model requests that include:
    - Image reference (e.g., S3 URI or base64-encoded image)
    - Extracted text/snippets
    - Instructions such as:
      - “Identify visible product defects.”
      - “Explain how this image aligns or conflicts with the textual review.”

- **Components**
  - `image_reference`
  - `image_labels`
  - `extracted_text`
  - `analysis_instructions`

---

### 3.3 Audio Data Formatting

- **Input**
  - Transcribed call data with sentiment and entities.

- **Output**
  - Conversation-based payloads that include:
    - Transcript (full or summarized)
    - Speaker roles (if available)
    - Sentiment over time or per segment
    - Request for:
      - Call summary
      - Customer emotion trajectory
      - Recommended actions

- **Components**
  - `call_transcript`
  - `speaker_segments`
  - `sentiment_overview`
  - `requested_insights`

---

### 3.4 Survey Data Formatting

- **Input**
  - Aggregated survey metrics and natural language summaries.

- **Output**
  - Analysis requests focused on:
    - Trend identification
    - Driver analysis for high/low scores
    - Cross-segment comparisons

- **Components**
  - `survey_metrics` (e.g., CSAT, NPS)
  - `survey_summary_text`
  - `comparison_dimensions` (e.g., region, product line)
  - `requested_analyses`

---

## 4. Architecture Components

### 4.1 Request Formatting Engine

- Implemented with AWS Lambda (or equivalent).
- Responsibilities:
  - Load appropriate conversation template.
  - Map enriched feedback into template variables.
  - Assemble a complete request payload for Amazon Bedrock.

### 4.2 Template Repository

- Stores predefined templates for:
  - Text-only analysis.
  - Multimodal (text + image) analysis.
  - Audio-derived transcript analysis.
  - Survey analytics.

- Supports:
  - Versioning and gradual rollout of updated prompts.
  - Domain- or client-specific variants.

### 4.3 Multimodal Handler

- Combines:
  - Text fields
  - Image references
  - Metadata
- Handles:
  - Base64 encoding for images when required.
  - Inclusion of multiple images and/or snippets in a single request.

### 4.4 Response Processing

- Post-processes model outputs into:
  - Structured JSON where possible.
  - Normalized insight records stored in S3/Glue tables.

- Includes:
  - Output validation and schema checks.
  - Error handling and retry logic (e.g., for malformed responses).
  - Mapping of insights back to original feedback items.

---

## 5. Quality Assurance

To keep model interactions reliable:

- **Input Validation**
  - Ensure all required fields are present before sending a request.

- **Format Verification**
  - Validate that payloads comply with Bedrock and model-specific APIs.

- **Response Quality Checks**
  - Sanity checks on returned JSON.
  - Optional secondary validation (e.g., using smaller models to check for missing fields).

- **Error Handling**
  - Retry with backoff for transient failures.
  - Fallback templates or flows for repeated failures.
