// Common functionality for Real Estate Price Predictor

// Format currency
function formatCurrency(amount) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(amount);
}

// Show alert message
function showAlert(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.role = 'alert';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    const container = document.querySelector('.container');
    container.insertBefore(alertDiv, container.firstChild);
    
    // Auto dismiss after 5 seconds
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}

// Handle API errors
function handleApiError(error) {
    console.error('API Error:', error);
    showAlert('An error occurred while fetching data. Please try again later.', 'danger');
}

// Initialize tooltips
function initTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Initialize popovers
function initPopovers() {
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
}

// Show loading spinner
function showLoading(element) {
    const originalContent = element.innerHTML;
    element.innerHTML = `
        <span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
        Loading...
    `;
    element.disabled = true;
    return originalContent;
}

// Hide loading spinner
function hideLoading(element, originalContent) {
    element.innerHTML = originalContent;
    element.disabled = false;
}

// Handle form submission
function handleFormSubmit(formId, callback) {
    const form = document.getElementById(formId);
    if (!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const submitButton = form.querySelector('button[type="submit"]');
        const originalContent = showLoading(submitButton);
        
        try {
            const formData = new FormData(form);
            const response = await fetch(form.action, {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                if (callback) callback(data);
            } else {
                showAlert(data.error || 'An error occurred', 'danger');
            }
        } catch (error) {
            handleApiError(error);
        } finally {
            hideLoading(submitButton, originalContent);
        }
    });
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Add input validation for numerical fields
    const numberInputs = document.querySelectorAll('input[type="number"]');
    numberInputs.forEach(input => {
        input.addEventListener('input', function() {
            if (this.value < 0) {
                this.value = 0;
            }
            // Add validation for specific fields
            if (this.id === 'INT_SQFT' && this.value > 10000) {
                this.value = 10000;
            }
            if (this.id === 'N_BEDROOM' && this.value > 10) {
                this.value = 10;
            }
            if (this.id === 'N_BATHROOM' && this.value > 5) {
                this.value = 5;
            }
        });
    });

    // Add form validation
    const form = document.querySelector('form');
    if (form) {
        form.addEventListener('submit', function(e) {
            const inputs = form.querySelectorAll('input[required]');
            let isValid = true;

            inputs.forEach(input => {
                if (!input.value.trim()) {
                    isValid = false;
                    input.classList.add('is-invalid');
                } else {
                    input.classList.remove('is-invalid');
                }
            });

            if (!isValid) {
                e.preventDefault();
                alert('Please fill in all required fields');
            }
        });
    }

    // Add tooltips for quality score fields
    const qualityScoreInputs = document.querySelectorAll('input[id^="QS_"]');
    qualityScoreInputs.forEach(input => {
        input.setAttribute('title', 'Enter a value between 1 and 5');
        input.setAttribute('min', '1');
        input.setAttribute('max', '5');
        input.setAttribute('step', '0.1');
    });

    initTooltips();
    initPopovers();
}); 