/**
 * AWS AI Project Feedback Form Application
 */

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize the application
    const feedbackForm = new FeedbackForm();
    feedbackForm.init();
});

/**
 * FeedbackForm class handles all form functionality
 */
class FeedbackForm {
    constructor() {
        // DOM elements
        this.form = document.getElementById('feedback-form');
        this.customerIdInput = document.getElementById('customer-id');
        this.feedbackTextarea = document.getElementById('feedback-text');
        this.charCount = document.getElementById('char-count');
        this.submitButton = document.getElementById('submit-btn');
        this.resetButton = document.getElementById('reset-btn');
        this.statusMessage = document.getElementById('status-message');
        this.loadingIndicator = document.getElementById('loading');
        
        // Star rating elements
        this.starRating = document.getElementById('star-rating');
        this.ratingInput = document.getElementById('rating');
        this.stars = [];
        
        // Photo upload elements
        this.photoInput = document.getElementById('photo-input');
        this.photoUploadArea = document.getElementById('photo-upload-area');
        this.photoPreviewContainer = document.getElementById('photo-preview-container');
        this.uploadedPhotos = [];
        
        // Audio recording elements
        this.audioButton = document.getElementById('audio-button');
        this.audioTimer = document.getElementById('audio-timer');
        this.audioPlayer = document.getElementById('audio-player');
        this.audioWaveform = document.getElementById('audio-waveform');
        this.recordedAudio = null;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.recordingStartTime = null;
        this.recordingTimer = null;
        
        // Form state
        this.currentRating = 0;
        this.isSubmitting = false;
        
        // Configuration
        this.config = window.Config;
    }
    
    /**
     * Initialize the form
     */
    init() {
        this.createStarRating();
        this.initializeMediaUploads();
        this.attachEventListeners();
        this.updateCharCount();
        this.trackAnalytics('form_view');
    }
    
    /**
     * Create the star rating component
     */
    createStarRating() {
        const { MAX_RATING, STAR_RATING } = this.config.FORM;
        
        for (let i = 1; i <= MAX_RATING; i++) {
            const star = document.createElement('span');
            star.className = `star rating-${i}`;
            star.textContent = STAR_RATING.emptyStar;
            star.dataset.rating = i;
            
            star.addEventListener('click', () => this.setRating(i));
            star.addEventListener('mouseenter', () => this.hoverRating(i));
            
            this.starRating.appendChild(star);
            this.stars.push(star);
        }
        
        // Reset rating when mouse leaves the star rating container
        this.starRating.addEventListener('mouseleave', () => this.resetHoverRating());
    }
    
    /**
     * Set the rating
     */
    setRating(rating) {
        this.currentRating = rating;
        this.ratingInput.value = rating;
        
        const { STAR_RATING } = this.config.FORM;
        
        this.stars.forEach((star, index) => {
            const starRating = parseInt(star.dataset.rating);
            star.classList.remove('selected');
            
            if (index < rating) {
                star.textContent = STAR_RATING.filledStar;
                star.classList.add(STAR_RATING.selectedClass);
                // Add rating-specific class for rainbow colors
                star.classList.add(`rating-${starRating}`);
            } else {
                star.textContent = STAR_RATING.emptyStar;
                // Keep rating-specific class for hover effects
                star.classList.add(`rating-${starRating}`);
            }
        });
    }
    
    /**
     * Handle hover effect on stars
     */
    hoverRating(rating) {
        const { STAR_RATING } = this.config.FORM;
        
        this.stars.forEach((star, index) => {
            const starRating = parseInt(star.dataset.rating);
            star.classList.remove('hover');
            
            if (index < rating) {
                star.classList.add(STAR_RATING.hoverClass);
                // Add rating-specific class for rainbow colors
                star.classList.add(`rating-${starRating}`);
            } else {
                // Keep rating-specific class for hover effects
                star.classList.add(`rating-${starRating}`);
            }
        });
    }
    
    /**
     * Reset hover effect
     */
    resetHoverRating() {
        const { STAR_RATING } = this.config.FORM;
        
        this.stars.forEach(star => {
            star.classList.remove(STAR_RATING.hoverClass);
        });
    }
    
    /**
     * Initialize media upload functionality
     */
    initializeMediaUploads() {
        // This will be expanded when we add the HTML elements
        // For now, we'll just check if the elements exist
        if (this.photoInput) {
            this.setupPhotoUpload();
        }
        
        if (this.audioButton) {
            this.setupAudioRecording();
        }
    }
    
    /**
     * Setup photo upload functionality
     */
    setupPhotoUpload() {
        this.photoInput.addEventListener('change', (e) => this.handlePhotoUpload(e));
        
        // Handle drag and drop
        if (this.photoUploadArea) {
            this.photoUploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                this.photoUploadArea.classList.add('drag-over');
            });
            
            this.photoUploadArea.addEventListener('dragleave', () => {
                this.photoUploadArea.classList.remove('drag-over');
            });
            
            this.photoUploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                this.photoUploadArea.classList.remove('drag-over');
                this.handlePhotoUpload(e);
            });
        }
    }
    
    /**
     * Handle photo upload
     */
    handlePhotoUpload(event) {
        const files = event.target.files || event.dataTransfer.files;
        const { VALIDATION } = this.config.FORM;
        
        Array.from(files).forEach(file => {
            // Check if file is an image
            if (!file.type.startsWith('image/')) {
                this.showStatusMessage('error', `${file.name} is not a valid image file.`);
                return;
            }
            
            // Check if file type is allowed
            if (!VALIDATION.PHOTOS.allowedTypes.includes(file.type)) {
                this.showStatusMessage('error', `${file.name} is not a supported image format.`);
                return;
            }
            
            // Check file size
            if (file.size > VALIDATION.PHOTOS.maxSize) {
                this.showStatusMessage('error', `${file.name} exceeds the maximum file size of ${Math.round(VALIDATION.PHOTOS.maxSize / (1024 * 1024))}MB.`);
                return;
            }
            
            // Check if we've reached the maximum number of photos
            if (this.uploadedPhotos.length >= VALIDATION.PHOTOS.maxFiles) {
                this.showStatusMessage('error', `Maximum ${VALIDATION.PHOTOS.maxFiles} photos allowed.`);
                return;
            }
            
            const reader = new FileReader();
            
            reader.onload = (e) => {
                const photoData = {
                    id: Date.now() + Math.random(),
                    url: e.target.result,
                    name: file.name,
                    size: file.size,
                    type: file.type
                };
                
                this.uploadedPhotos.push(photoData);
                this.renderPhotoPreview(photoData);
                this.updatePhotoUploadArea();
            };
            
            reader.readAsDataURL(file);
        });
        
        // Reset the file input to allow selecting the same file again
        if (this.photoInput) {
            this.photoInput.value = '';
        }
    }
    
    /**
     * Render photo preview
     */
    renderPhotoPreview(photoData) {
        const previewContainer = document.createElement('div');
        previewContainer.className = 'photo-preview';
        previewContainer.dataset.photoId = photoData.id;
        
        const img = document.createElement('img');
        img.src = photoData.url;
        img.alt = photoData.name;
        
        const removeButton = document.createElement('button');
        removeButton.className = 'remove-photo';
        removeButton.innerHTML = '×';
        removeButton.addEventListener('click', () => this.removePhoto(photoData.id));
        
        previewContainer.appendChild(img);
        previewContainer.appendChild(removeButton);
        
        this.photoPreviewContainer.appendChild(previewContainer);
    }
    
    /**
     * Remove photo
     */
    removePhoto(photoId) {
        this.uploadedPhotos = this.uploadedPhotos.filter(photo => photo.id !== photoId);
        
        const previewElement = this.photoPreviewContainer.querySelector(`[data-photo-id="${photoId}"]`);
        if (previewElement) {
            previewElement.remove();
        }
        
        this.updatePhotoUploadArea();
    }
    
    /**
     * Update photo upload area appearance
     */
    updatePhotoUploadArea() {
        if (this.uploadedPhotos.length > 0) {
            this.photoUploadArea.classList.add('has-file');
        } else {
            this.photoUploadArea.classList.remove('has-file');
        }
    }
    
    /**
     * Setup audio recording functionality
     */
    setupAudioRecording() {
        this.audioButton.addEventListener('click', () => this.toggleAudioRecording());
    }
    
    /**
     * Toggle audio recording
     */
    async toggleAudioRecording() {
        if (this.isRecording) {
            this.stopAudioRecording();
        } else {
            await this.startAudioRecording();
        }
    }
    
    /**
     * Start audio recording
     */
    async startAudioRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.mediaRecorder = new MediaRecorder(stream);
            this.audioChunks = [];
            
            this.mediaRecorder.ondataavailable = (event) => {
                this.audioChunks.push(event.data);
            };
            
            this.mediaRecorder.onstop = () => {
                const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
                this.recordedAudio = URL.createObjectURL(audioBlob);
                this.audioPlayer.src = this.recordedAudio;
                this.audioPlayer.style.display = 'block';
                
                // Stop all tracks on the stream
                stream.getTracks().forEach(track => track.stop());
            };
            
            this.mediaRecorder.start();
            this.isRecording = true;
            this.recordingStartTime = Date.now();
            
            // Update UI
            this.audioButton.classList.add('recording');
            this.audioButton.innerHTML = '⏹';
            this.startRecordingTimer();
            
        } catch (error) {
            console.error('Error accessing microphone:', error);
            this.showStatusMessage('error', 'Unable to access microphone. Please check your permissions.');
        }
    }
    
    /**
     * Stop audio recording
     */
    stopAudioRecording() {
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            this.mediaRecorder.stop();
            this.isRecording = false;
            
            // Update UI
            this.audioButton.classList.remove('recording');
            this.audioButton.innerHTML = '🎤';
            this.stopRecordingTimer();
        }
    }
    
    /**
     * Start recording timer
     */
    startRecordingTimer() {
        const { VALIDATION } = this.config.FORM;
        
        this.recordingTimer = setInterval(() => {
            const elapsed = Math.floor((Date.now() - this.recordingStartTime) / 1000);
            const minutes = Math.floor(elapsed / 60).toString().padStart(2, '0');
            const seconds = (elapsed % 60).toString().padStart(2, '0');
            this.audioTimer.textContent = `${minutes}:${seconds}`;
            
            // Check if we've exceeded maximum duration
            if (elapsed > VALIDATION.AUDIO.maxDuration) {
                this.showStatusMessage('error', `Maximum recording duration of ${Math.round(VALIDATION.AUDIO.maxDuration / 60)} minutes exceeded.`);
                this.stopAudioRecording();
            }
        }, 1000);
    }
    
    /**
     * Stop recording timer
     */
    stopRecordingTimer() {
        if (this.recordingTimer) {
            clearInterval(this.recordingTimer);
            this.recordingTimer = null;
        }
    }
    
    /**
     * Attach event listeners
     */
    attachEventListeners() {
        // Form submission
        this.form.addEventListener('submit', (e) => {
            e.preventDefault();
            this.submitForm();
        });
        
        // Reset button
        this.resetButton.addEventListener('click', () => {
            this.resetForm();
        });
        
        // Character count update
        this.feedbackTextarea.addEventListener('input', () => {
            this.updateCharCount();
        });
    }
    
    /**
     * Update character count
     */
    updateCharCount() {
        const currentLength = this.feedbackTextarea.value.length;
        const { MAX_FEEDBACK_LENGTH } = this.config.FORM;
        
        this.charCount.textContent = `${currentLength}/${MAX_FEEDBACK_LENGTH}`;
        
        if (currentLength > MAX_FEEDBACK_LENGTH) {
            this.charCount.style.color = '#d32f2f';
        } else if (currentLength > MAX_FEEDBACK_LENGTH * 0.9) {
            this.charCount.style.color = '#f57c00';
        } else {
            this.charCount.style.color = '#666';
        }
    }
    
    /**
     * Validate the form
     */
    validateForm() {
        const { VALIDATION } = this.config.FORM;
        let isValid = true;
        let errorMessage = '';
        
        // Customer ID validation
        if (!this.customerIdInput.value.trim()) {
            errorMessage = 'Customer email or ID is required.';
            isValid = false;
        } else if (
            this.customerIdInput.value.length < VALIDATION.CUSTOMER_ID.minLength ||
            this.customerIdInput.value.length > VALIDATION.CUSTOMER_ID.maxLength
        ) {
            errorMessage = `Customer email or ID must be between ${VALIDATION.CUSTOMER_ID.minLength} and ${VALIDATION.CUSTOMER_ID.maxLength} characters.`;
            isValid = false;
        } else if (!VALIDATION.CUSTOMER_ID.pattern.test(this.customerIdInput.value)) {
            errorMessage = 'Customer email or ID can only contain letters, numbers, hyphens, and underscores.';
            isValid = false;
        }
        
        // Rating validation
        if (this.currentRating < VALIDATION.RATING.min || this.currentRating > VALIDATION.RATING.max) {
            errorMessage = 'Please select a rating from 1 to 5 stars.';
            isValid = false;
        }
        
        // Feedback validation
        if (!this.feedbackTextarea.value.trim()) {
            errorMessage = 'Feedback text is required.';
            isValid = false;
        } else if (
            this.feedbackTextarea.value.length < VALIDATION.FEEDBACK.minLength ||
            this.feedbackTextarea.value.length > VALIDATION.FEEDBACK.maxLength
        ) {
            errorMessage = `Feedback must be between ${VALIDATION.FEEDBACK.minLength} and ${VALIDATION.FEEDBACK.maxLength} characters.`;
            isValid = false;
        }
        
        // Photos validation
        if (this.uploadedPhotos.length > VALIDATION.PHOTOS.maxFiles) {
            errorMessage = `Maximum ${VALIDATION.PHOTOS.maxFiles} photos allowed.`;
            isValid = false;
        } else {
            const oversizedPhotos = this.uploadedPhotos.filter(photo => photo.size > VALIDATION.PHOTOS.maxSize);
            if (oversizedPhotos.length > 0) {
                errorMessage = `One or more photos exceed the maximum size of ${Math.round(VALIDATION.PHOTOS.maxSize / (1024 * 1024))}MB.`;
                isValid = false;
            }
        }
        
        // Audio validation
        if (this.recordedAudio && this.audioTimer.textContent !== '0:00') {
            const [minutes, seconds] = this.audioTimer.textContent.split(':').map(Number);
            const totalSeconds = minutes * 60 + seconds;
            
            if (totalSeconds > VALIDATION.AUDIO.maxDuration) {
                errorMessage = `Audio recording cannot exceed ${Math.round(VALIDATION.AUDIO.maxDuration / 60)} minutes.`;
                isValid = false;
            }
        }
        
        return {
            isValid,
            errorMessage
        };
    }
    
    /**
     * Submit the form
     */
    async submitForm() {
        if (this.isSubmitting) return;
        
        const validation = this.validateForm();
        
        if (!validation.isValid) {
            this.showStatusMessage('error', validation.errorMessage);
            return;
        }
        
        this.trackAnalytics('form_submit_attempt');
        
        // Set loading state
        this.setLoadingState(true);
        
        try {
            const formData = {
                customerId: this.customerIdInput.value.trim(),
                rating: this.currentRating,
                feedback: this.feedbackTextarea.value.trim(),
                timestamp: new Date().toISOString(),
                photos: this.uploadedPhotos.map(photo => ({
                    id: photo.id,
                    name: photo.name,
                    size: photo.size,
                    type: photo.type,
                    // In a real implementation, you would upload the image to a server
                    // and store the URL here instead of the base64 data
                    data: photo.url
                })),
                audio: this.recordedAudio ? {
                    // In a real implementation, you would upload the audio to a server
                    // and store the URL here instead of the blob URL
                    url: this.recordedAudio,
                    duration: this.audioTimer.textContent
                } : null
            };
            
            const response = await this.submitFeedback(formData);
            
            if (response.success) {
                this.showStatusMessage('success', 'Thank you for your feedback! It has been submitted successfully.');
                this.trackAnalytics('form_submit_success');
                
                // Reset form after successful submission
                setTimeout(() => {
                    this.resetForm();
                }, this.config.UI.SUCCESS_MESSAGE_TIMEOUT);
            } else {
                throw new Error(response.message || 'Failed to submit feedback');
            }
        } catch (error) {
            console.error('Error submitting feedback:', error);
            this.showStatusMessage('error', `Error: ${error.message || 'Failed to submit feedback. Please try again later.'}`);
            this.trackAnalytics('form_submit_error');
        } finally {
            this.setLoadingState(false);
        }
    }
    
    /**
     * Submit feedback to the API with retry logic
     */
    async submitFeedback(data) {
        const { FEEDBACK_ENDPOINT, TIMEOUT, HEADERS, RETRY } = this.config.API;
        
        // If the endpoint is not configured, simulate a successful submission
        if (FEEDBACK_ENDPOINT.includes('your-api-id')) {
            console.log('API endpoint not configured. Simulating successful submission.');
            console.log('Data that would be submitted:', data);
            return { success: true, message: 'Simulated submission successful' };
        }
        
        let lastError;
        
        // Implement retry logic
        for (let attempt = 1; attempt <= RETRY.maxAttempts; attempt++) {
            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), TIMEOUT);
                
                const response = await fetch(FEEDBACK_ENDPOINT, {
                    method: 'POST',
                    headers: HEADERS,
                    body: JSON.stringify(data),
                    signal: controller.signal
                });
                
                clearTimeout(timeoutId);
                
                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({}));
                    throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
                }
                
                const result = await response.json();
                
                // Log successful submission
                console.log(`Feedback submitted successfully on attempt ${attempt}:`, result);
                
                return result;
                
            } catch (error) {
                lastError = error;
                console.warn(`Attempt ${attempt} failed:`, error.message);
                
                // Don't retry on client errors (4xx)
                if (error.message.includes('HTTP error! Status: 4')) {
                    throw error;
                }
                
                // If not the last attempt, wait before retrying
                if (attempt < RETRY.maxAttempts) {
                    const delay = RETRY.initialDelay * Math.pow(RETRY.backoffMultiplier, attempt - 1);
                    console.log(`Retrying in ${delay}ms...`);
                    await new Promise(resolve => setTimeout(resolve, delay));
                }
            }
        }
        
        // All attempts failed
        throw lastError;
    }
    
    /**
     * Set loading state
     */
    setLoadingState(isLoading) {
        this.isSubmitting = isLoading;
        this.submitButton.disabled = isLoading;
        
        if (isLoading) {
            this.loadingIndicator.style.display = 'inline-block';
            this.submitButton.textContent = 'Submitting';
        } else {
            this.loadingIndicator.style.display = 'none';
            this.submitButton.textContent = 'Submit Feedback';
        }
    }
    
    /**
     * Show status message
     */
    showStatusMessage(type, message) {
        this.statusMessage.textContent = message;
        this.statusMessage.className = `status-message ${type}`;
        this.statusMessage.style.display = 'block';
        
        // Scroll to the status message
        this.statusMessage.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        
        // Auto-hide success messages
        if (type === 'success') {
            setTimeout(() => {
                this.statusMessage.style.display = 'none';
            }, this.config.UI.SUCCESS_MESSAGE_TIMEOUT);
        }
    }
    
    /**
     * Reset the form
     */
    resetForm() {
        this.form.reset();
        this.setRating(0);
        this.currentRating = 0;
        this.updateCharCount();
        this.statusMessage.style.display = 'none';
        
        // Reset photo uploads
        this.uploadedPhotos = [];
        this.photoPreviewContainer.innerHTML = '';
        this.updatePhotoUploadArea();
        
        // Reset audio recording
        if (this.isRecording) {
            this.stopAudioRecording();
        }
        this.recordedAudio = null;
        this.audioPlayer.src = '';
        this.audioPlayer.style.display = 'none';
        this.audioTimer.textContent = '0:00';
    }
    
    /**
     * Track analytics events (minimal implementation)
     */
    trackAnalytics(event) {
        if (!this.config.ANALYTICS.ENABLED || !this.config.ANALYTICS.ENDPOINT) {
            return;
        }
        
        try {
            // Simple analytics tracking without external libraries
            const data = {
                event,
                url: window.location.href,
                userAgent: navigator.userAgent,
                timestamp: new Date().toISOString()
            };
            
            // Use sendBeacon for non-blocking analytics
            if (navigator.sendBeacon) {
                navigator.sendBeacon(this.config.ANALYTICS.ENDPOINT, JSON.stringify(data));
            } else {
                // Fallback for older browsers
                fetch(this.config.ANALYTICS.ENDPOINT, {
                    method: 'POST',
                    body: JSON.stringify(data),
                    keepalive: true
                }).catch(() => {
                    // Silently fail analytics to not disrupt user experience
                });
            }
        } catch (error) {
            // Silently fail analytics to not disrupt user experience
            console.debug('Analytics tracking failed:', error);
        }
    }
}