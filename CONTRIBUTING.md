# Contributing to AWS AI Customer Feedback System

Thank you for your interest in contributing to the AWS AI Customer Feedback System! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Issues

We use GitHub Issues to track public bugs and feature requests. Please follow these guidelines:

1. **Search existing issues** before creating a new one
2. **Use descriptive titles** that clearly explain the issue
3. **Provide detailed information** including:
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (AWS region, Python version, etc.)
   - Screenshots if applicable

### Submitting Pull Requests

We welcome pull requests! Please follow these steps:

1. **Fork the repository** to your GitHub account
2. **Create a feature branch** from the main branch
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** following our coding standards
4. **Test your changes** thoroughly
5. **Submit a pull request** with a clear description

## 🛠️ Development Setup

### Prerequisites

- AWS Account with appropriate permissions
- AWS CLI installed and configured
- Python 3.8+ with pip
- Node.js 16+ with npm
- Git
- Docker (for local testing)

### Local Development Setup

1. **Clone your forked repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/aws-ai-customer-feedback.git
   cd aws-ai-customer-feedback
   ```

2. **Set up Python virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Install Node.js dependencies**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

4. **Configure local environment**
   ```bash
   cp .env.example .env
   # Edit .env with your local configuration
   ```

5. **Run local tests**
   ```bash
   npm test
   ```

### Development Workflow

1. **Create a new branch** for your feature or bugfix
2. **Make your changes** following our coding standards
3. **Add tests** for new functionality
4. **Run the test suite** to ensure everything passes
5. **Update documentation** if needed
6. **Commit your changes** with descriptive messages
7. **Push to your fork** and create a pull request

## 📝 Coding Standards

### Python Code

We follow PEP 8 with some additional conventions:

```python
# Good example
def process_customer_feedback(feedback_data: dict) -> dict:
    """
    Process customer feedback data and return insights.
    
    Args:
        feedback_data: Dictionary containing feedback information
        
    Returns:
        Dictionary containing processed insights
        
    Raises:
        ValueError: If feedback data is invalid
    """
    if not feedback_data:
        raise ValueError("Feedback data cannot be empty")
    
    # Process feedback
    processed_data = _validate_and_clean_data(feedback_data)
    insights = _generate_insights(processed_data)
    
    return insights


def _validate_and_clean_data(data: dict) -> dict:
    """Internal helper for data validation."""
    # Implementation details
    pass
```

#### Python Guidelines

- **Type hints** required for all function signatures
- **Docstrings** required for all public functions and classes
- **Line length** maximum 88 characters
- **Imports** grouped: standard library, third-party, local
- **Variable names** should be descriptive and snake_case
- **Constants** should be UPPER_CASE

### JavaScript Code

We use modern ES6+ features with ESLint configuration:

```javascript
// Good example
/**
 * Process customer feedback form submission
 * @param {Object} formData - Form data from customer
 * @returns {Promise<Object>} Processed feedback data
 */
async function processFeedback(formData) {
  try {
    const validatedData = validateFormData(formData);
    const processedData = await submitToAPI(validatedData);
    return processedData;
  } catch (error) {
    console.error('Error processing feedback:', error);
    throw new Error('Failed to process feedback');
  }
}

// Helper function
function validateFormData(data) {
  if (!data.customerId || !data.feedback) {
    throw new Error('Missing required fields');
  }
  return data;
}
```

#### JavaScript Guidelines

- **Use const/let** instead of var
- **Arrow functions** for callbacks and short functions
- **Template literals** for string interpolation
- **Async/await** for asynchronous operations
- **JSDoc comments** for all public functions
- **Descriptive variable names** in camelCase

### File Organization

```
src/
├── lambda/
│   ├── data_validation/
│   ├── multimodal_processing/
│   └── fm_formatting/
├── frontend/
│   ├── src/
│   ├── public/
│   └── tests/
├── utils/
└── tests/
```

- **Group related files** in appropriate directories
- **Use descriptive filenames** that clearly indicate purpose
- **Keep files focused** on a single responsibility
- **Avoid deeply nested** directory structures

## 🧪 Testing Guidelines

### Test Structure

We use a comprehensive testing approach:

```
tests/
├── unit/                    # Unit tests for individual components
├── integration/             # Integration tests for workflows
├── e2e/                   # End-to-end tests
├── fixtures/               # Test data and fixtures
└── reports/               # Test reports and coverage
```

### Writing Tests

#### Unit Tests (Python)

```python
import pytest
from unittest.mock import Mock, patch
from src.lambda.text_processing import process_text_feedback

class TestTextProcessing:
    def test_process_valid_feedback(self):
        """Test processing of valid feedback data."""
        feedback_data = {
            "customer_id": "cust_123",
            "feedback": "Great service!",
            "rating": 5
        }
        
        result = process_text_feedback(feedback_data)
        
        assert result["status"] == "success"
        assert "sentiment" in result
        assert result["customer_id"] == "cust_123"
    
    def test_process_invalid_feedback_raises_error(self):
        """Test that invalid feedback raises appropriate error."""
        invalid_data = {"invalid": "data"}
        
        with pytest.raises(ValueError, match="Invalid feedback data"):
            process_text_feedback(invalid_data)
    
    @patch('src.lambda.text_processing.comprehend_client')
    def test_aws_service_error_handling(self, mock_comprehend):
        """Test handling of AWS service errors."""
        mock_comprehend.detect_sentiment.side_effect = Exception("AWS Error")
        
        feedback_data = {"customer_id": "cust_123", "feedback": "test"}
        
        result = process_text_feedback(feedback_data)
        
        assert result["status"] == "error"
        assert "AWS Error" in result["error_message"]
```

#### Integration Tests

```python
import boto3
from tests.integration.aws_helpers import create_test_s3_bucket, cleanup_test_bucket

class TestTextProcessingIntegration:
    @pytest.fixture(scope="class")
    def test_environment(self):
        """Set up test AWS resources."""
        bucket_name = create_test_s3_bucket()
        yield {"bucket": bucket_name}
        cleanup_test_bucket(bucket_name)
    
    def test_end_to_end_text_processing(self, test_environment):
        """Test complete text processing workflow."""
        # Upload test data
        s3 = boto3.client("s3")
        test_file = "test-feedback.txt"
        s3.put_object(
            Bucket=test_environment["bucket"],
            Key=test_file,
            Body="Great product quality!"
        )
        
        # Trigger Lambda function (via S3 event simulation)
        # ... test implementation
        
        # Verify results
        response = s3.get_object(
            Bucket=test_environment["bucket"],
            Key=f"processed/{test_file}"
        )
        result = json.loads(response["Body"].read())
        
        assert result["status"] == "success"
        assert "sentiment" in result
```

### Test Data Management

- **Use fixtures** for consistent test data
- **Mock external services** to avoid dependencies
- **Clean up resources** after tests
- **Use descriptive test names** that explain what is being tested

### Coverage Requirements

- **Unit tests**: 95%+ code coverage
- **Integration tests**: Cover all major workflows
- **E2E tests**: Cover critical user journeys

## 📋 Pull Request Process

### Before Submitting

1. **Run all tests** and ensure they pass
2. **Check code coverage** meets requirements
3. **Update documentation** for any API changes
4. **Add changelog entry** for new features
5. **Rebase** your branch if needed

### Pull Request Template

```markdown
## Description
Brief description of changes and their purpose.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Added new tests for new functionality

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] No merge conflicts
```

### Review Process

1. **Automated checks** must pass (tests, linting, coverage)
2. **Code review** by at least one maintainer
3. **Discussion** of any significant changes
4. **Approval** required before merging
5. **Squash merge** to maintain clean history

## 🔧 Development Tools

### Required Tools

- **Python**: 3.8+
- **Node.js**: 16+
- **AWS CLI**: Latest version
- **Docker**: Latest version

### Recommended Tools

- **IDE**: VS Code with Python and JavaScript extensions
- **Git Client**: SourceTree, GitKraken, or command line
- **API Testing**: Postman or Insomnia
- **AWS Console**: For resource management

### VS Code Extensions

- Python
- Pylance
- ESLint
- Prettier
- GitLens
- AWS Toolkit
- Docker

## 📚 Documentation

### Types of Documentation

1. **API Documentation**: Auto-generated from code
2. **User Guides**: Step-by-step instructions
3. **Architecture Docs**: System design and decisions
4. **Troubleshooting**: Common issues and solutions

### Documentation Standards

- **Clear and concise** language
- **Code examples** for all major features
- **Diagrams** for complex workflows
- **Regular updates** as features change

## 🚀 Release Process

### Version Management

We use [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

1. **All tests pass** in all environments
2. **Documentation updated** for new features
3. **Changelog updated** with all changes
4. **Security review** completed for sensitive changes
5. **Performance testing** for significant changes
6. **Version bumped** appropriately
7. **Tag created** in Git
8. **Release published** to GitHub

## 🏷️ Labels and Milestones

### Issue Labels

- `bug`: Bug reports and issues
- `enhancement`: Feature requests
- `documentation`: Documentation improvements
- `good first issue`: Good for newcomers
- `help wanted`: Community contributions welcome
- `security`: Security-related issues
- `performance`: Performance improvements

### Milestones

- `v1.0.x`: Current stable release
- `v1.1.x`: Next minor release
- `v2.0.0`: Next major release

## 🤖 Automated Processes

### CI/CD Pipeline

1. **Code quality checks** on every push
2. **Automated testing** on multiple Python versions
3. **Security scanning** for vulnerabilities
4. **Dependency updates** for security patches
5. **Automated deployment** for main branch

### Code Quality Tools

- **ESLint**: JavaScript linting
- **Pylint**: Python linting
- **Black**: Python code formatting
- **Prettier**: JavaScript code formatting
- **SonarQube**: Code quality analysis

## 🆘 Getting Help

### Community Support

- **GitHub Issues**: For bug reports and feature requests
- **Discussions**: For general questions and ideas
- **Wiki**: For detailed documentation

### Developer Resources

- **Architecture Documentation**: `docs/architecture/`
- **API Reference**: `docs/api-reference/`
- **Troubleshooting Guide**: `docs/troubleshooting/`

### Contact Information

- **Maintainer**: Lantz Murray
- **Email**: [maintainer-email@example.com]
- **Discord**: [Community server link]

## 📄 Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:

- **Be respectful** and professional
- **Welcome newcomers** and help them learn
- **Focus on constructive** feedback
- **Avoid personal attacks** or harassment
- **Follow GitHub's** Community Guidelines

## 🎉 Recognition

Contributors are recognized through:

- **Contributors section** in README
- **Release notes** mentioning significant contributions
- **Annual recognition** for top contributors
- **Swag and stickers** for active contributors

Thank you for contributing to the AWS AI Customer Feedback System! Your contributions help make this project better for everyone.