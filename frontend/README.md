# AWS AI Project - Feedback Form

A cost-optimized, responsive frontend for submitting customer feedback to the AWS AI Project.

## Overview

This frontend application provides a simple, accessible form for customers to submit feedback including:
- Customer ID
- Star rating (1-5)
- Text feedback

The application is built with vanilla HTML, CSS, and JavaScript to minimize costs and dependencies.

## Features

- **Responsive Design**: Works on desktop and mobile devices
- **Form Validation**: Client-side validation for all fields
- **Interactive Star Rating**: Visual 1-5 star rating component
- **Loading States**: Visual feedback during form submission
- **Error Handling**: Clear error messages for submission issues
- **Accessibility**: ARIA labels, semantic HTML, keyboard navigation
- **Cost Optimized**: Static files suitable for S3 hosting

## File Structure

```
frontend/
├── index.html          # Main feedback form page
├── css/
│   └── style.css       # Styling for the feedback form
├── js/
│   ├── config.js        # Configuration file for API endpoint
│   └── app.js           # JavaScript for form handling and API integration
├── README.md            # This file
└── deploy.sh            # Deployment script for S3
```

## Setup Instructions

### 1. Configure API Endpoint

Edit `js/config.js` and update the `FEEDBACK_ENDPOINT` value with your actual API Gateway endpoint:

```javascript
// Replace with your actual API Gateway endpoint URL
// Format: https://your-api-id.execute-api.region.amazonaws.com/stage/feedback
FEEDBACK_ENDPOINT: 'https://your-api-id.execute-api.region.amazonaws.com/prod/feedback',
```

### 2. Local Testing

To test the application locally:

1. Open `index.html` in a web browser
2. The form will simulate submissions until a real API endpoint is configured
3. Check the browser console for submission data during testing

### 3. Deployment to S3

#### Option A: Using the Deployment Script

1. Make sure you have AWS CLI installed and configured
2. Run the deployment script:

```bash
chmod +x deploy.sh
./deploy.sh your-bucket-name
```

#### Option B: Manual Deployment

1. Create an S3 bucket:
```bash
aws s3 mb s3://your-bucket-name
```

2. Enable static website hosting:
```bash
aws s3 website s3://your-bucket-name --index-document index.html
```

3. Upload the files:
```bash
aws s3 sync . s3://your-bucket-name --delete
```

4. Set bucket policy for public access:
```bash
aws s3api put-bucket-policy --bucket your-bucket-name --policy '{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PublicReadGetObject",
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::your-bucket-name/*"
    }
  ]
}'
```

## API Integration

The form expects the following API endpoint format:

- **URL**: Your API Gateway endpoint
- **Method**: POST
- **Headers**: Content-Type: application/json
- **Body**:

```json
{
  "customerId": "CUSTOMER-123",
  "rating": 4,
  "feedback": "Great service! Very satisfied with the experience.",
  "timestamp": "2023-12-07T20:15:00.000Z"
}
```

- **Expected Response**:

```json
{
  "success": true,
  "message": "Feedback submitted successfully"
}
```

## Cost Considerations

### Hosting Costs

- **S3 Static Website Hosting**: 
  - Storage: ~$0.023 per GB/month
  - Data Transfer: First 100 GB/month free, then ~$0.09 per GB
  - Estimated monthly cost: <$1 for low traffic

### Data Transfer Costs

- **API Gateway**: 
  - 1 million requests/month free
  - $3.50 per million requests thereafter
  - Data transfer: First 1 GB/month free

### Optimization Tips

1. **Enable Gzip Compression**: Configure CloudFront in front of S3
2. **Cache Static Assets**: Set appropriate cache headers
3. **Minimize Requests**: Keep CSS and JS files small
4. **Use CloudFront**: Consider CloudFront distribution for better performance and potentially lower data transfer costs

## Browser Compatibility

This application is tested and compatible with:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Security Considerations

1. **CORS Configuration**: Ensure your API Gateway has proper CORS settings
2. **Input Validation**: Client-side validation is implemented, but server-side validation is still recommended
3. **HTTPS**: Always use HTTPS in production
4. **Rate Limiting**: Consider implementing rate limiting on your API endpoint

## Testing

### Manual Testing

1. Fill out all form fields correctly and submit
2. Try submitting with empty fields (should show validation errors)
3. Try submitting with invalid customer ID format
4. Try submitting with feedback text that's too short or too long
5. Test star rating interaction
6. Test responsive design on different screen sizes

### Automated Testing

For automated testing, you can use tools like:
- Selenium WebDriver
- Cypress
- Playwright

## Troubleshooting

### Form Not Submitting

1. Check browser console for JavaScript errors
2. Verify API endpoint is correctly configured in `config.js`
3. Check network tab in browser developer tools for failed requests

### CORS Errors

1. Ensure your API Gateway is configured with proper CORS headers
2. Verify the API endpoint URL is correct

### Styling Issues

1. Check that CSS file is properly linked in HTML
2. Verify file paths are correct after deployment

## Support

For issues related to:
- **Frontend Application**: Check this README and browser console
- **API Gateway**: Refer to AWS documentation
- **S3 Hosting**: Refer to AWS S3 documentation

## License

This project is part of the AWS AI Project and follows the project's licensing terms.