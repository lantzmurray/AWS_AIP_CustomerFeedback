# Implementation Guide – AI-First Version

> This folder contains **implementation runbooks** for both humans and AI agents.
> Follow the parts **in order** unless you have a specific reason not to.

---

## 0. What You Are Building

You are implementing a **customer feedback processing pipeline** that:

1. Validates incoming data.
2. Processes multimodal inputs (text, images, audio, surveys).
3. Formats results for foundation models (Claude on Amazon Bedrock).

---

## 1. Files in This Folder

- `prerequisites.md`  
  AWS account requirements, IAM permissions, environment setup, and cost notes.

- `part1_data_validation_workflow.md`  
  Implementation steps for **data validation** using AWS Glue Data Quality + Lambda.

- `part2_multimodal_data_processing.md`  
  Implementation steps for **multimodal processing** (text, image, audio, survey).

- `part3_data_formatting_for_fms.md`  
  Implementation steps for **formatting data and calling Claude** in Amazon Bedrock.

---

## 2. Execution Order (Do This in Sequence)

1. **Read and satisfy `prerequisites.md`.**  
   - Confirm services, IAM roles, Python env, and sample data are ready.

2. **Implement Part 1 – Validation.**  
   - Create S3 buckets, Glue catalog, data quality rules, and validation Lambdas.

3. **Implement Part 2 – Multimodal processing.**  
   - Create processing Lambdas and SageMaker Processing jobs for each data type.

4. **Implement Part 3 – Foundation model formatting.**  
   - Create the Claude formatting Lambda and integrate with Amazon Bedrock.

---

## 3. Conventions for AI Agents

When this document (and the Part 1–3 docs) say:

- **[ACTION]** – Generate the corresponding code, CLI command, or config.
- **[VERIFY]** – Run a check and confirm the expected result exists (bucket, Lambda, log, etc.).
- **[DECISION]** – Choose a value (e.g., region, bucket name) and keep it consistent.

Assumptions:

- Default region: `us-east-1` (override if needed).
- Naming pattern:  
  `customer-feedback-analysis-<your-initials>` for S3 buckets.  
  You MUST replace `<your-initials>` with a real value and use the same value everywhere.

---

## 4. Dependencies

- The **Code** folder contains Python scripts referenced in these guides.
- The **Architecture** folder explains the high-level design.
- This **Implementation** folder tells you exactly how to deploy and wire components.

When in doubt:

- Use **Architecture** to understand *why*.
- Use **Implementation** to do *how*.
