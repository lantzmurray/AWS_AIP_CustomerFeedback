# Architecture Diagrams

This directory contains technical architecture diagrams for the AWS AI Customer Feedback System.

## System Architecture Overview

```mermaid
graph TB
    subgraph "User Interface Layer"
        A[Customer Feedback Web App] --> B[Mobile App Interface]
        A --> C[API Gateway]
        B --> C
    end
    
    subgraph "Data Ingestion Layer"
        C --> D[S3 Raw Data Bucket]
        D --> E[Data Validation Service]
        E --> F[Quality Scoring Engine]
    end
    
    subgraph "Processing Layer"
        F --> G[Text Processing Pipeline]
        F --> H[Image Processing Pipeline]
        F --> I[Audio Processing Pipeline]
        F --> J[Survey Processing Pipeline]
        
        G --> K[Amazon Comprehend]
        H --> L[Amazon Textract + Rekognition]
        I --> M[Amazon Transcribe + Comprehend]
        J --> N[Amazon SageMaker Processing]
    end
    
    subgraph "AI Integration Layer"
        K --> O[Foundation Model Formatter]
        L --> O
        M --> O
        N --> O
        O --> P[Amazon Bedrock - Claude AI]
        O --> Q[Response Processor]
    end
    
    subgraph "Storage & Output Layer"
        Q --> R[S3 Processed Data Bucket]
        Q --> S[S3 Results Bucket]
        R --> T[Data Catalog - Glue]
        S --> U[Analytics Dashboard]
    end
    
    subgraph "Monitoring & Security Layer"
        V[CloudWatch Monitoring] --> W[Alerting System]
        X[IAM Security] --> Y[Access Control]
        Z[CloudTrail Audit] --> AA[Compliance Reporting]
    end
```

## Data Flow Architecture

```mermaid
flowchart TD
    A[Customer Input] --> B{Input Type?}
    
    B -->|Text| C[Text Validation]
    B -->|Image| D[Image Validation]
    B -->|Audio| E[Audio Validation]
    B -->|Survey| F[Survey Validation]
    
    C --> G[Quality Score]
    D --> G
    E --> G
    F --> G
    
    G --> H{Score > 70%?}
    H -->|Yes| I[Processing Queue]
    H -->|No| J[Quarantine]
    
    I --> K[Text Processing]
    I --> L[Image Processing]
    I --> M[Audio Processing]
    I --> N[Survey Processing]
    
    K --> O[Entity Extraction]
    L --> P[Text + Label Extraction]
    M --> Q[Transcription + Sentiment]
    N --> R[Statistical Analysis]
    
    O --> S[Foundation Model Formatting]
    P --> S
    Q --> S
    R --> S
    
    S --> T[Claude AI Analysis]
    T --> U[Insight Generation]
    U --> V[Result Storage]
    
    V --> W[Dashboard Update]
    V --> X[Alert Generation]
    
    J --> Y[Error Notification]
    Y --> Z[Manual Review]
```

## Component Interaction Diagram

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant API_Gateway
    participant S3_Raw
    participant Lambda_Validate
    participant Lambda_Process
    participant AWS_Services
    participant Bedrock
    participant S3_Results
    participant Dashboard
    
    User->>Frontend: Submit feedback
    Frontend->>API_Gateway: Process request
    API_Gateway->>S3_Raw: Store raw data
    S3_Raw->>Lambda_Validate: Trigger validation
    
    Lambda_Validate->>Lambda_Validate: Validate format
    Lambda_Validate->>Lambda_Validate: Calculate quality score
    Lambda_Validate->>S3_Raw: Store validation results
    
    alt Quality Score >= 70%
        Lambda_Validate->>Lambda_Process: Trigger processing
        Lambda_Process->>AWS_Services: Extract insights
        AWS_Services->>Lambda_Process: Return analysis
        Lambda_Process->>Bedrock: Format for Claude
        Bedrock->>Lambda_Process: Generate insights
        Lambda_Process->>S3_Results: Store results
        S3_Results->>Dashboard: Update display
    else Quality Score < 70%
        Lambda_Validate->>S3_Raw: Move to quarantine
        Lambda_Validate->>User: Send error notification
    end
    
    Dashboard->>User: Display processed results
```

## Service Integration Architecture

```mermaid
graph LR
    subgraph "Core AWS Services"
        A[S3 Storage] --> B[Lambda Computing]
        B --> C[Glue Data Catalog]
        C --> D[CloudWatch Monitoring]
    end
    
    subgraph "AI Services"
        E[Amazon Comprehend] --> F[Text Analysis]
        G[Amazon Textract] --> H[Image Text Extraction]
        I[Amazon Rekognition] --> J[Image Analysis]
        K[Amazon Transcribe] --> L[Audio Transcription]
        M[Amazon SageMaker] --> N[Advanced Processing]
    end
    
    subgraph "Foundation Models"
        O[Amazon Bedrock] --> P[Claude AI]
        P --> Q[Insight Generation]
    end
    
    subgraph "Frontend Services"
        R[CloudFront CDN] --> S[Static Website]
        S --> T[Interactive Dashboard]
    end
    
    B --> E
    B --> G
    B --> I
    B --> K
    B --> M
    
    F --> O
    H --> O
    J --> O
    L --> O
    N --> O
    
    Q --> A
    Q --> R
```

## Security Architecture

```mermaid
graph TB
    subgraph "Identity & Access Management"
        A[IAM Roles] --> B[Service Principals]
        A --> C[User Principals]
        A --> D[Resource Policies]
    end
    
    subgraph "Data Protection"
        E[S3 Encryption] --> F[AES-256 at Rest]
        G[TLS Encryption] --> H[Data in Transit]
        I[KMS Management] --> J[Key Rotation]
    end
    
    subgraph "Network Security"
        K[VPC Endpoints] --> L[Private Connectivity]
        M[Security Groups] --> N[Traffic Filtering]
        O[WAF Rules] --> P[Web Protection]
    end
    
    subgraph "Monitoring & Audit"
        Q[CloudTrail] --> R[API Audit Log]
        S[CloudWatch] --> T[Security Metrics]
        U[Config Rules] --> V[Compliance Monitoring]
    end
    
    B --> E
    C --> G
    D --> K
    
    F --> Q
    H --> S
    J --> U
```

## Performance Architecture

```mermaid
graph TB
    subgraph "Scalability Components"
        A[Auto Scaling Lambda] --> B[Concurrent Execution]
        C[S3 Multi-Part Upload] --> D[Large File Handling]
        E[Glue Dynamic Workers] --> F[Variable Data Volume]
    end
    
    subgraph "Performance Optimization"
        G[Memory Tuning] --> H[Optimal Lambda Performance]
        I[Batch Processing] --> J[Reduced API Calls]
        K[Caching Layer] --> L[Faster Response Times]
    end
    
    subgraph "Cost Optimization"
        M[Spot Instances] --> N[Reduced Compute Cost]
        O[S3 Lifecycle Policies] --> P[Storage Tier Optimization]
        Q[Request Batching] --> R[Lower API Costs]
    end
    
    subgraph "Monitoring & Alerting"
        S[Performance Metrics] --> T[Real-time Dashboards]
        U[Cost Tracking] --> V[Budget Alerts]
        W[Error Monitoring] --> X[Proactive Notifications]
    end
    
    B --> G
    D --> I
    F --> K
    
    H --> S
    J --> U
    L --> W
```

## Deployment Architecture

```mermaid
graph TB
    subgraph "Development Environment"
        A[Dev S3 Buckets] --> B[Dev Lambda Functions]
        B --> C[Dev Glue Database]
        C --> D[Dev Monitoring]
    end
    
    subgraph "Staging Environment"
        E[Staging S3 Buckets] --> F[Staging Lambda Functions]
        F --> G[Staging Glue Database]
        G --> H[Staging Monitoring]
    end
    
    subgraph "Production Environment"
        I[Prod S3 Buckets] --> J[Prod Lambda Functions]
        J --> K[Prod Glue Database]
        K --> L[Prod Monitoring]
    end
    
    subgraph "CI/CD Pipeline"
        M[Code Repository] --> N[Build Pipeline]
        N --> O[Test Automation]
        O --> P[Deployment Pipeline]
        P --> I
    end
    
    subgraph "Shared Services"
        Q[IAM Roles] --> A
        Q --> E
        Q --> I
        R[CloudWatch] --> D
        R --> H
        R --> L
    end
```

## Data Model Architecture

```mermaid
erDiagram
    CUSTOMER {
        string customer_id PK
        string name
        string email
        string phone
        datetime created_at
        datetime updated_at
    }
    
    FEEDBACK {
        string feedback_id PK
        string customer_id FK
        string feedback_type
        text content
        int rating
        datetime submitted_at
        string status
    }
    
    TEXT_REVIEW {
        string review_id PK
        string feedback_id FK
        text review_text
        json entities
        string sentiment
        json key_phrases
        float quality_score
    }
    
    IMAGE_ANALYSIS {
        string image_id PK
        string feedback_id FK
        string image_url
        json extracted_text
        json labels
        json faces
        float quality_score
    }
    
    AUDIO_TRANSCRIPTION {
        string audio_id PK
        string feedback_id FK
        string audio_url
        text transcript
        string sentiment
        json speaker_analysis
        float confidence_score
    }
    
    SURVEY_RESPONSE {
        string survey_id PK
        string feedback_id FK
        json responses
        text summary
        json statistics
        float satisfaction_score
    }
    
    AI_INSIGHT {
        string insight_id PK
        string feedback_id FK
        text insight_text
        string insight_type
        float confidence_score
        datetime generated_at
    }
    
    CUSTOMER ||--o{ FEEDBACK : submits
    FEEDBACK ||--|| TEXT_REVIEW : contains
    FEEDBACK ||--|| IMAGE_ANALYSIS : contains
    FEEDBACK ||--|| AUDIO_TRANSCRIPTION : contains
    FEEDBACK ||--|| SURVEY_RESPONSE : contains
    FEEDBACK ||--o{ AI_INSIGHT : generates
```

## Integration Patterns

### Event-Driven Pattern
```mermaid
graph TB
    A[S3 Events] --> B[Lambda Triggers]
    B --> C[Asynchronous Processing]
    C --> D[Event Bridge]
    D --> E[Error Handling]
```

### Request-Response Pattern
```mermaid
graph TB
    A[API Gateway] --> B[Lambda Functions]
    B --> C[Synchronous Processing]
    C --> D[Immediate Response]
    D --> E[Response Validation]
```

### Queue-Based Pattern
```mermaid
graph TB
    A[SQS Queues] --> B[Message Buffering]
    B --> C[Batch Processing]
    C --> D[Decoupled Services]
    D --> E[Error Recovery]
```

## Technology Stack

### Backend Technologies
- **Python 3.8+**: Lambda functions and data processing
- **AWS Services**: Comprehensive cloud integration
- **Serverless Architecture**: Lambda, API Gateway, S3
- **Event-Driven Processing**: S3 triggers, Step Functions

### AI/ML Services
- **Amazon Comprehend**: Natural language processing
- **Amazon Textract**: Text extraction from images
- **Amazon Rekognition**: Image and video analysis
- **Amazon Transcribe**: Speech-to-text conversion
- **Amazon Bedrock**: Foundation model integration

### Frontend Technologies
- **HTML5/CSS3**: Modern web standards
- **JavaScript ES6+**: Interactive features
- **Chart.js**: Data visualization
- **Responsive Design**: Mobile-first approach

### Infrastructure Technologies
- **AWS CloudFormation**: Infrastructure as code
- **Terraform**: Multi-cloud support
- **Docker**: Container support
- **AWS SAM**: Serverless application model

## Monitoring and Observability

### Key Metrics
- **Processing Latency**: < 30 seconds for 95th percentile
- **System Availability**: 99.9% uptime SLA
- **Error Rate**: < 2% for all operations
- **Scalability**: Handle 10x load increase automatically

### Monitoring Tools
- **CloudWatch**: Metrics and logs
- **X-Ray**: Distributed tracing
- **CloudTrail**: Audit logging
- **AWS Config**: Configuration monitoring

## Security and Compliance

### Data Protection
- **Encryption at Rest**: AES-256 for all S3 storage
- **Encryption in Transit**: TLS 1.2+ for all communications
- **Access Control**: IAM with principle of least privilege
- **Audit Logging**: CloudTrail for all API calls

### Compliance Features
- **Data Retention**: Configurable lifecycle policies
- **Privacy Protection**: PII detection and handling
- **Regional Deployment**: Data residency compliance
- **Monitoring**: Real-time security and compliance dashboards

---

For more detailed technical information, please refer to the [technical design documentation](../technical-design/README.md).