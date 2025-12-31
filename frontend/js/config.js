/**
 * Configuration for AWS AI Project Feedback Form
 */

// Environment detection
const ENVIRONMENT = (() => {
    const hostname = window.location.hostname;
    if (hostname === 'localhost' || hostname === '127.0.0.1') {
        return 'dev';
    } else if (hostname.includes('dev') || hostname.includes('staging')) {
        return 'dev';
    } else {
        return 'prod';
    }
})();

// API Configuration
const API_CONFIG = {
    // Environment-specific endpoints
    ENDPOINTS: {
        dev: {
            // Local development fallback
            FEEDBACK_ENDPOINT: window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
                ? 'http://localhost:3000/process'
                : 'https://your-api-id.execute-api.region.amazonaws.com/dev/process',
            API_KEY: null // Optional for dev
        },
        prod: {
            // Production endpoint - will be configured during deployment
            FEEDBACK_ENDPOINT: 'https://your-api-id.execute-api.region.amazonaws.com/prod/process',
            API_KEY: null // Will be configured if API key is enabled
        }
    },
    
    // Get current environment configuration
    getCurrent: () => API_CONFIG.ENDPOINTS[ENVIRONMENT],
    
    // Request timeout in milliseconds
    TIMEOUT: 15000,
    
    // Headers for API requests
    HEADERS: {
        'Content-Type': 'application/json'
    },
    
    // Retry configuration
    RETRY: {
        maxAttempts: 3,
        backoffMultiplier: 2,
        initialDelay: 1000
    },
    
    // Error handling configuration
    ERROR_HANDLING: {
        // Show detailed error messages in development
        showDetailedErrors: ENVIRONMENT === 'dev',
        // Log errors to console in development
        logErrors: ENVIRONMENT === 'dev',
        // Fallback error message for production
        fallbackErrorMessage: 'An error occurred while submitting your feedback. Please try again later.'
    }
};

// Form Configuration
const FORM_CONFIG = {
    // Maximum character count for feedback text
    MAX_FEEDBACK_LENGTH: 500,
    
    // Minimum rating (1-5 stars)
    MIN_RATING: 1,
    
    // Maximum rating (1-5 stars)
    MAX_RATING: 5,
    
    // Field validation
    VALIDATION: {
        CUSTOMER_ID: {
            required: true,
            minLength: 3,
            maxLength: 50,
            pattern: /^[A-Za-z0-9\-_]+$/
        },
        RATING: {
            required: true,
            min: 1,
            max: 5
        },
        FEEDBACK: {
            required: true,
            minLength: 10,
            maxLength: 500
        },
        PHOTOS: {
            required: false,
            maxFiles: 5,
            maxSize: 10 * 1024 * 1024, // 10MB in bytes
            allowedTypes: ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp']
        },
        AUDIO: {
            required: false,
            maxSize: 5 * 1024 * 1024, // 5MB in bytes
            allowedTypes: ['audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/ogg'],
            maxDuration: 300 // 5 minutes in seconds
        }
    },
    // Star rating configuration
    STAR_RATING: {
        emptyStar: '☆',
        filledStar: '★',
        hoverClass: 'hover',
        selectedClass: 'selected'
    }
};

// UI Configuration
const UI_CONFIG = {
    // Animation duration in milliseconds
    ANIMATION_DURATION: 300,
    
    // Auto-hide success message after this many milliseconds
    SUCCESS_MESSAGE_TIMEOUT: 5000
};

// Analytics Configuration (minimal, cost-effective)
const ANALYTICS_CONFIG = {
    // Enable basic analytics tracking
    ENABLED: false,
    
    // Simple event tracking endpoint
    ENDPOINT: null,
    
    // Events to track
    TRACKED_EVENTS: [
        'form_view',
        'form_submit_attempt',
        'form_submit_success',
        'form_submit_error'
    ]
};

// Export configuration objects
window.Config = {
    API: API_CONFIG,
    FORM: FORM_CONFIG,
    UI: UI_CONFIG,
    ANALYTICS: ANALYTICS_CONFIG
};