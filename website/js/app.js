// Configuration
const API_URL = 'https://15zjlknmve.execute-api.us-east-1.amazonaws.com/Prod/';
const API_ENDPOINTS = {
    upload: API_URL + 'upload',
    analyze: API_URL + 'analyze',
    results: (analysisId) => `${API_URL}results/${analysisId}`
};

// Debug function to test API endpoints
async function testApiEndpoints() {
    console.log('Testing API endpoints...');
    
    // Test each endpoint - skip base URL since it has CORS issues
    const endpoints = [
        { name: 'Upload', url: API_ENDPOINTS.upload, method: 'OPTIONS' },
        { name: 'Analyze', url: API_ENDPOINTS.analyze, method: 'OPTIONS' },
        { name: 'Results', url: API_ENDPOINTS.results('test'), method: 'OPTIONS' }
    ];
    
    for (const endpoint of endpoints) {
        try {
            console.log(`Testing ${endpoint.name}: ${endpoint.url}`);
            const response = await fetch(endpoint.url, {
                method: endpoint.method || 'GET',
                mode: 'cors',
                headers: {
                    'Accept': 'application/json'
                }
            });
            
            console.log(`${endpoint.name} status:`, response.status);
            if (response.ok) {
                try {
                    const data = await response.json();
                    console.log(`${endpoint.name} response:`, data);
                } catch (e) {
                    console.log(`${endpoint.name} response is not JSON:`, await response.text());
                }
            } else {
                console.log(`${endpoint.name} error:`, await response.text());
            }
        } catch (error) {
            console.error(`${endpoint.name} fetch error:`, error);
        }
    }
}

// Initialize the form
document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    uploadForm.addEventListener('submit', handleFormSubmit);
    
    console.log('BattingAI app initialized');
    console.log('API URL:', API_URL);
    
    // Run API tests
    testApiEndpoints();
    
    // Add debug button
    const debugButton = document.createElement('button');
    debugButton.textContent = 'Debug API Connection';
    debugButton.className = 'btn btn-secondary';
    debugButton.style.marginLeft = '10px';
    debugButton.addEventListener('click', function(e) {
        e.preventDefault();
        testApiEndpoints();
    });
    
    const submitButton = document.querySelector('button[type="submit"]');
    submitButton.parentNode.insertBefore(debugButton, submitButton.nextSibling);
});

// Handle form submission
async function handleFormSubmit(event) {
    event.preventDefault();
    
    const videoFile = document.getElementById('videoFile').files[0];
    const playerSelect = document.getElementById('playerSelect');
    const playerId = playerSelect.value;
    
    if (!videoFile) {
        alert('Please select a video file');
        return;
    }
    
    // Show progress
    document.getElementById('uploadProgress').style.display = 'block';
    document.querySelector('.progress-bar').style.width = '0%';
    
    try {
        // Show processing spinner
        document.getElementById('loadingSpinner').style.display = 'block';
        document.getElementById('statusMessage').textContent = 'Getting upload URL...';
        document.querySelector('.progress-bar').style.width = '10%';
        
        // Step 1: Get presigned URL from the API
        console.log('Requesting presigned URL from:', API_ENDPOINTS.upload);
        const urlResponse = await fetch(API_ENDPOINTS.upload, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            mode: 'cors',
            body: JSON.stringify({})  // Empty body since we're just requesting a URL
        });
        
        console.log('URL response status:', urlResponse.status);
        
        if (!urlResponse.ok) {
            let errorText = '';
            try {
                errorText = await urlResponse.text();
            } catch (e) {
                errorText = 'Could not read error response';
            }
            console.error('URL request error response:', errorText);
            throw new Error(`URL request failed: ${urlResponse.status} ${urlResponse.statusText}`);
        }
        
        let urlData;
        try {
            urlData = await urlResponse.json();
            console.log('URL response data:', urlData);
        } catch (e) {
            console.error('Error parsing URL response:', e);
            throw new Error('Invalid response from upload endpoint');
        }
        
        if (!urlData || !urlData.upload_url || !urlData.analysis_id) {
            throw new Error('Missing upload_url or analysis_id in response');
        }
        
        const { upload_url, analysis_id } = urlData;
        
        // Step 2: Upload the video directly to S3 using the presigned URL
        document.getElementById('statusMessage').textContent = 'Uploading video to S3...';
        document.querySelector('.progress-bar').style.width = '30%';
        
        console.log('Uploading video directly to S3 using presigned URL');
        const s3UploadResponse = await fetch(upload_url, {
            method: 'PUT',
            headers: {
                'Content-Type': 'video/mp4'
            },
            body: videoFile  // Send the raw file, not base64
        });
        
        console.log('S3 upload response status:', s3UploadResponse.status);
        
        if (!s3UploadResponse.ok) {
            let errorText = '';
            try {
                errorText = await s3UploadResponse.text();
            } catch (e) {
                errorText = 'Could not read error response';
            }
            console.error('S3 upload error response:', errorText);
            throw new Error(`S3 upload failed: ${s3UploadResponse.status} ${s3UploadResponse.statusText}`);
        }
        
        document.querySelector('.progress-bar').style.width = '60%';
        document.getElementById('statusMessage').textContent = 'Analyzing your swing...';
        
        // Step 3: Start analysis
        const analyzeResponse = await fetch(API_ENDPOINTS.analyze, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            mode: 'cors',
            body: JSON.stringify({
                analysis_id: analysis_id,
                player_id: playerId
            })
        });
        
        console.log('Analyze response status:', analyzeResponse.status);
        
        if (!analyzeResponse.ok) {
            let errorText = '';
            try {
                errorText = await analyzeResponse.text();
            } catch (e) {
                errorText = 'Could not read error response';
            }
            console.error('Analyze error response:', errorText);
            throw new Error(`Analysis failed: ${analyzeResponse.status} ${analyzeResponse.statusText}`);
        }
        
        // Poll for results
        document.querySelector('.progress-bar').style.width = '80%';
        document.getElementById('statusMessage').textContent = 'Getting results...';
        
        let results = null;
        let attempts = 0;
        const maxAttempts = 20;
        
        while (!results && attempts < maxAttempts) {
            attempts++;
            console.log(`Polling for results (attempt ${attempts}/${maxAttempts})...`);
            
            try {
                const resultsResponse = await fetch(API_ENDPOINTS.results(analysis_id), {
                    method: 'GET',
                    headers: {
                        'Accept': 'application/json'
                    },
                    mode: 'cors'
                });
                
                console.log('Results response status:', resultsResponse.status);
                
                if (resultsResponse.ok) {
                    let data;
                    try {
                        data = await resultsResponse.json();
                        console.log('Results data:', data);
                    } catch (e) {
                        console.error('Error parsing results response:', e);
                        continue;
                    }
                    
                    if (data.status === 'feedback_generated') {
                        results = data;
                        break;
                    } else {
                        console.log('Results not ready yet, status:', data.status);
                    }
                } else {
                    console.warn('Results request failed:', resultsResponse.status);
                }
                
                // Wait before trying again
                await new Promise(resolve => setTimeout(resolve, 3000));
            } catch (e) {
                console.error('Error polling for results:', e);
            }
        }
        
        document.querySelector('.progress-bar').style.width = '100%';
        
        if (results) {
            // Display the results
            displayResults(results, playerId);
            return;
        }
        
        throw new Error('Timed out waiting for results');
    } catch (error) {
        console.error('Error:', error);
        
        // Show a more detailed error message
        let errorMessage = error.message;
        if (error.message === 'Failed to fetch') {
            errorMessage = 'Failed to connect to the API. Please check your internet connection or try again later.';
        }
        
        alert('An error occurred: ' + errorMessage);
        document.getElementById('uploadProgress').style.display = 'none';
        document.getElementById('loadingSpinner').style.display = 'none';
    }
}

// Helper function to convert file to base64 (kept for compatibility)
function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => {
            // Remove the data URL prefix (e.g., "data:image/png;base64,")
            const base64 = reader.result.split(',')[1];
            resolve(base64);
        };
        reader.onerror = error => reject(error);
    });
}