// Configuration
const API_URL = 'https://15zjlknmve.execute-api.us-east-1.amazonaws.com/Prod/';
const API_ENDPOINTS = {
    upload: `${API_URL}upload`,
    analyze: `${API_URL}analyze`,
    results: (analysisId) => `${API_URL}results/${encodeURIComponent(analysisId)}`
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
    
    // Add help button and modal for upload specifications
    const helpButton = document.createElement('button');
    helpButton.textContent = 'What should I upload?';
    helpButton.className = 'btn btn-info';
    helpButton.style.marginLeft = '10px';
    helpButton.addEventListener('click', function(e) {
        e.preventDefault();
        showUploadSpecsModal();
    });

    // Create modal container
    const modal = document.createElement('div');
    modal.id = 'uploadSpecsModal';
    modal.className = 'modal fade';
    modal.innerHTML = `
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Video Upload Requirements</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <h6>For best results, please ensure your video meets these requirements:</h6>
                    <ul>
                        <li><strong>Full Body View:</strong> The entire batter must be visible in all frames, from head to feet</li>
                        <li><strong>Camera Position:</strong> Record from a side view, perpendicular to the batter's stance</li>
                        <li><strong>Framing:</strong> Keep the batter centered in the frame</li>
                        <li><strong>Background:</strong> A clean, uncluttered background helps with analysis</li>
                        <li><strong>Lighting:</strong> Good, consistent lighting throughout the swing</li>
                    </ul>
                    <div class="alert alert-info">
                        <strong>Tip:</strong> Position the camera at about waist height, 10-15 feet away from the batter
                    </div>
                </div>
            </div>
        </div>
    `;
    document.body.appendChild(modal);

    // Create frame detail modal container
    const frameDetailModal = document.createElement('div');
    frameDetailModal.id = 'frameDetailModal';
    frameDetailModal.className = 'modal fade';
    frameDetailModal.innerHTML = `
        <div class="modal-dialog modal-xl">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title"></h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <div class="container-fluid">
                        <div class="row">
                            <div class="col-md-6">
                                <h4 class="text-center mb-3">Your Swing</h4>
                                <img class="img-fluid user-frame mb-3" alt="Your Frame">
                            </div>
                            <div class="col-md-6">
                                <h4 class="text-center mb-3">Reference</h4>
                                <img class="img-fluid reference-frame mb-3" alt="Reference Frame">
                            </div>
                        </div>
                        <div class="row mt-3">
                            <div class="col-12">
                                <h5 class="similarity-score"></h5>
                                <div class="progress mb-3">
                                    <div class="progress-bar" role="progressbar"></div>
                                </div>
                                <div class="issues-container"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    document.body.appendChild(frameDetailModal);

    const submitButton = document.querySelector('button[type="submit"]');
    submitButton.parentNode.insertBefore(helpButton, submitButton.nextSibling);
});

// Function to show upload specifications modal
function showUploadSpecsModal() {
    const modal = new bootstrap.Modal(document.getElementById('uploadSpecsModal'));
    modal.show();
}

// Function to show frame detail modal
function showFrameDetail(frameIndex, similarityScore, phaseName, userFrameUrl, referenceFrameUrl, issues) {
    // Remove any existing modal
    const existingModal = document.getElementById('frameDetailModal');
    if (existingModal) {
        existingModal.remove();
    }
    
    // Create new modal HTML
    const modalHTML = `
        <div class="modal fade" id="frameDetailModal" tabindex="-1">
            <div class="modal-dialog modal-xl">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">${phaseName} - Frame ${frameIndex + 1}</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <div class="container-fluid">
                            <div class="row">
                                <div class="col-md-6">
                                    <h4 class="text-center mb-3">Your Swing</h4>
                                    <img src="${userFrameUrl}" class="img-fluid" alt="Your Frame">
                                </div>
                                <div class="col-md-6">
                                    <h4 class="text-center mb-3">Reference</h4>
                                    <img src="${referenceFrameUrl}" class="img-fluid" alt="Reference Frame">
                                </div>
                            </div>
                            <div class="row mt-4">
                                <div class="col-12">
                                    <h5>Similarity Score: ${similarityScore}%</h5>
                                    <div class="progress mb-3">
                                        <div class="progress-bar ${similarityScore < 70 ? 'bg-warning' : ''}" 
                                             role="progressbar" 
                                             style="width: ${similarityScore}%"
                                             aria-valuenow="${similarityScore}" 
                                             aria-valuemin="0" 
                                             aria-valuemax="100">
                                        </div>
                                    </div>
                                    <div class="issues-container mt-3">
                                        ${issues}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Add modal to document
    document.body.insertAdjacentHTML('beforeend', modalHTML);
    
    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('frameDetailModal'));
    modal.show();
    
    // Clean up modal when hidden
    document.getElementById('frameDetailModal').addEventListener('hidden.bs.modal', function() {
        this.remove();
    });
}

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
                video_key: urlData.video_key,  // Include the video_key from the upload response
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
        const maxAttempts = 40; // Increased from 20 to 40 to allow more time for processing
        
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
                      if (data.status === 'comparison_complete' || data.status === 'feedback_generated') {
                        results = data;
                        break;
                    } else if (data.error) {
                        throw new Error(data.error);
                    } else {
                        console.log('Results not ready yet, status:', data.status);
                    }
                } else {
                    let errorText;
                    try {
                        errorText = await resultsResponse.text();
                        console.error('Results request failed:', resultsResponse.status, errorText);
                    } catch (e) {
                        console.error('Results request failed:', resultsResponse.status, 'Could not read error response');
                    }
                    // If we get a 404, the analysis doesn't exist, so we should break
                    if (resultsResponse.status === 404) {
                        throw new Error('Analysis not found - please try uploading again');
                    }
                    // For 400 errors, check if it's a known error condition
                    if (resultsResponse.status === 400 && errorText) {
                        try {
                            const errorData = JSON.parse(errorText);
                            if (errorData.error) {
                                throw new Error(errorData.error);
                            }
                        } catch (e) {
                            // If we can't parse the error, just continue polling
                            console.warn('Could not parse error response:', e);
                        }
                    }
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

// S3 bucket name for accessing frame images
const bucket_name = 'battingai-videobucket-ayk9m1uehbg2';

// Helper function to get feedback based on similarity score
function getFeedbackForScore(score, issues) {
    if (score >= 0.9) {
        return {
            text: 'Perfect!',
            class: 'text-success',
            icon: 'bi-check-circle-fill',
            details: 'Your form closely matches the reference in this frame.'
        };
    } else if (score >= 0.7) {
        // Create a more detailed feedback message based on the score range
        let details = '';
        if (issues && issues.length > 0) {
            details = 'Focus on fixing: ' + issues.map(issue => issue.description).join(', ').toLowerCase();
        } else {
            // If no specific issues are detected, provide general feedback based on score
            if (score >= 0.8) {
                details = 'Minor adjustments needed - pay attention to your body positioning and timing.';
            } else {
                details = 'Several aspects need work - focus on matching the reference pose more closely.';
            }
        }
        return {
            text: 'Good, but needs improvement',
            class: 'text-warning',
            icon: 'bi-exclamation-circle-fill',
            details: details
        };
    } else {
        return {
            text: 'Issues need attention',
            class: 'text-danger',
            icon: 'bi-x-circle-fill',
            details: 'Significant differences from reference form. Review the issues listed below.'
        };
    }
}

// Function to display analysis results
function displayResults(results, playerId) {
    console.log("Displaying results:", results);
    
    // Hide progress indicators
    document.getElementById('uploadProgress').style.display = 'none';
    document.getElementById('loadingSpinner').style.display = 'none';
    
    // Show results container
    let container = document.getElementById('resultsContainer');
    if (!container) {
        // Create results container if it doesn't exist
        container = document.createElement('div');
        container.id = 'resultsContainer';
        container.className = 'mt-4 p-4 border rounded bg-light';
        document.querySelector('.container').appendChild(container);
    }
    
    container.innerHTML = ''; // Clear previous results
    
    // Create header
    const header = document.createElement('h3');
    header.textContent = 'Swing Analysis Results';
    container.appendChild(header);
    
    // Check if results has the expected structure
    if (!results.results) {
        const errorMsg = document.createElement('p');
        errorMsg.textContent = 'Results data is not in the expected format. Please try again.';
        container.appendChild(errorMsg);
        console.error('Invalid results format:', results);
        return;
    }
    
    // Add player info
    const playerInfo = document.createElement('p');
    playerInfo.innerHTML = `<strong>Compared to:</strong> ${results.results.player_name || playerId}`;
    container.appendChild(playerInfo);
    
    // Add overall score if available
    if (results.results.overall_score) {
        const scoreDiv = document.createElement('div');
        scoreDiv.className = 'mb-3';
        scoreDiv.innerHTML = `
            <h4>Overall Score: ${results.results.overall_score}/100</h4>
            <div class="progress">
                <div class="progress-bar" role="progressbar" style="width: ${results.results.overall_score}%" 
                     aria-valuenow="${results.results.overall_score}" aria-valuemin="0" aria-valuemax="100"></div>
            </div>
        `;
        container.appendChild(scoreDiv);
    }
    
    // Add strengths if available
    if (results.results.strengths && results.results.strengths.length > 0) {
        const strengthsDiv = document.createElement('div');
        strengthsDiv.className = 'mb-3';
        strengthsDiv.innerHTML = '<h4>Strengths</h4><ul>';
        
        results.results.strengths.forEach(strength => {
            strengthsDiv.innerHTML += `<li>${strength}</li>`;
        });
        
        strengthsDiv.innerHTML += '</ul>';
        container.appendChild(strengthsDiv);
    }
    
    // Add areas to improve if available
    if (results.results.areas_to_improve && results.results.areas_to_improve.length > 0) {
        const improvementsDiv = document.createElement('div');
        improvementsDiv.className = 'mb-3';
        improvementsDiv.innerHTML = '<h4>Areas to Improve</h4><ul>';
        
        results.results.areas_to_improve.forEach(area => {
            improvementsDiv.innerHTML += `<li>${area}</li>`;
        });
        
        improvementsDiv.innerHTML += '</ul>';
        container.appendChild(improvementsDiv);
    }
    
    // Add detailed feedback if available
    if (results.results.detailed_feedback) {
        const feedbackDiv = document.createElement('div');
        feedbackDiv.className = 'mb-3';
        feedbackDiv.innerHTML = `
            <h4>Detailed Feedback</h4>
            <p>${results.results.detailed_feedback}</p>
        `;
        container.appendChild(feedbackDiv);
    }
    
    // Add comparison results if available
    if (results.results.comparison_results && results.results.comparison_results.length > 0) {
        const comparisonDiv = document.createElement('div');
        comparisonDiv.className = 'mb-3';
        comparisonDiv.innerHTML = '<h4>Frame-by-Frame Analysis</h4>';
        
        // Create a card deck for frames
        const cardDeck = document.createElement('div');
        cardDeck.className = 'row row-cols-1 row-cols-md-3 g-4';
        
        // Define swing phase names
        const phaseNames = ['Setup', 'Load', 'Swing', 'Contact', 'Follow-through'];
        
        results.results.comparison_results.forEach((frame, index) => {            // Create a card for each frame
            const card = document.createElement('div');
            card.className = 'col';
            
            // Get annotated frames from base64 data
            const userAnnotatedUrl = frame.user_annotated ? 
                `data:image/jpeg;base64,${frame.user_annotated}` : '';
            const refAnnotatedUrl = frame.ref_annotated ? 
                `data:image/jpeg;base64,${frame.ref_annotated}` : '';

            // Determine the phase name based on index
            const phaseName = phaseNames[index % phaseNames.length] || `Phase ${index + 1}`;
            
            // Create issues list HTML
            let issuesHtml = '';
            const stanceIssues = frame.annotations.filter(a => a.type.startsWith('stance_'));
            const otherIssues = frame.issues.filter(issue => !issue.type || !issue.type.startsWith('stance_'));
            
            if (stanceIssues.length > 0 || frame.issues.length > 0) {
                issuesHtml = '<ul class="list-group list-group-flush">';
                
                // Add stance-specific feedback first
                stanceIssues.forEach(stance => {
                    issuesHtml += `
                        <li class="list-group-item">
                            <div class="d-flex justify-content-between align-items-start">
                                <div>
                                    <strong>${stance.description}</strong>
                                    <div class="text-muted small">${stance.details}</div>
                                </div>
                                <span class="badge bg-warning">${(stance.magnitude * 100).toFixed(0)}% diff</span>
                            </div>
                        </li>`;
                });
                
                // Add other issues
                frame.issues.forEach(issue => {
                    issuesHtml += `<li class="list-group-item">${issue.description}</li>`;
                });
                
                issuesHtml += '</ul>';
            } else {
                issuesHtml = '<p class="card-text text-success">Perfect form!</p>';
            }
            
            // Escape the issues HTML for the onclick handler
            const escapedIssuesHtml = issuesHtml.replace(/`/g, '\\`').replace(/'/g, "\\'").replace(/"/g, '\\"');
            
            // Create the card content
            const cardContent = document.createElement('div');
            cardContent.className = 'card h-100';
            cardContent.style.cursor = 'pointer';
            cardContent.addEventListener('click', () => {
                showFrameDetail(
                    frame.frame_index,
                    Math.round(frame.similarity_score * 100),
                    phaseName,
                    userAnnotatedUrl,
                    refAnnotatedUrl,
                    escapedIssuesHtml
                );
            });

            // Get feedback for the similarity score
            const feedback = getFeedbackForScore(frame.similarity_score, frame.issues);
            
            // Set the card's inner HTML
            cardContent.innerHTML = `
                <div class="card-header bg-primary text-white">
                    ${phaseName} - Frame ${frame.frame_index + 1}
                </div>
                <div class="row g-0">
                    <div class="col-6">
                        <div class="p-2">
                            <h6 class="text-center mb-2">Your Swing</h6>
                            ${userAnnotatedUrl ? 
                                `<img src="${userAnnotatedUrl}" class="img-fluid" alt="Your Frame ${frame.frame_index + 1}" style="max-height: 200px; object-fit: contain;">` : 
                                `<div class="text-center p-3 bg-light">
                                    <div class="swing-phase-icon">
                                        <i class="bi bi-camera-video"></i>
                                        <div class="mt-2">Frame ${frame.frame_index + 1}</div>
                                    </div>
                                </div>`
                            }
                        </div>
                    </div>
                    <div class="col-6">
                        <div class="p-2">
                            <h6 class="text-center mb-2">Reference</h6>
                            ${refAnnotatedUrl ? 
                                `<img src="${refAnnotatedUrl}" class="img-fluid" alt="Reference Frame ${frame.frame_index + 1}" style="max-height: 200px; object-fit: contain;">` : 
                                `<div class="text-center p-3 bg-light">
                                    <div class="swing-phase-icon">
                                        <i class="bi bi-camera-video"></i>
                                        <div class="mt-2">Reference ${frame.frame_index + 1}</div>
                                    </div>
                                </div>`
                            }
                        </div>
                    </div>
                </div>
                <div class="card-body">
                    <h5 class="card-title">Similarity: ${Math.round(frame.similarity_score * 100)}%</h5>
                    <div class="progress mb-3">
                        <div class="progress-bar ${frame.similarity_score < 0.7 ? 'bg-danger' : (frame.similarity_score < 0.9 ? 'bg-warning' : 'bg-success')}" role="progressbar" 
                            style="width: ${Math.round(frame.similarity_score * 100)}%" 
                            aria-valuenow="${Math.round(frame.similarity_score * 100)}" aria-valuemin="0" aria-valuemax="100"></div>
                    </div>
                    <h6 class="card-subtitle mb-2 ${feedback.class}">
                        <i class="bi ${feedback.icon} me-1"></i>
                        ${feedback.text}
                    </h6>
                    <p class="card-text small ${feedback.class} mb-3">${feedback.details}</p>
                    ${issuesHtml}
                </div>
            `;

            card.appendChild(cardContent);
            cardDeck.appendChild(card);
        });

        comparisonDiv.appendChild(cardDeck);
        container.appendChild(comparisonDiv);
        
        // Also add a traditional table view for comparison
        const tableDiv = document.createElement('div');
        tableDiv.className = 'mt-4';
        tableDiv.innerHTML = '<h5>Detailed Metrics</h5>';
        
        const table = document.createElement('table');
        table.className = 'table table-striped table-sm';
        table.innerHTML = `
            <thead>
                <tr>
                    <th>Frame</th>
                    <th>Phase</th>
                    <th>Similarity Score</th>
                    <th>Issues</th>
                </tr>
            </thead>
            <tbody>
        `;
        
        results.results.comparison_results.forEach((frame, index) => {
            const phaseName = phaseNames[index % phaseNames.length] || `Phase ${index + 1}`;
            
            let issuesHtml = '';
            if (frame.issues && frame.issues.length > 0) {
                issuesHtml = '<ul class="mb-0">';
                frame.issues.forEach(issue => {
                    issuesHtml += `<li>${issue.description}</li>`;
                });
                issuesHtml += '</ul>';
            } else {
                issuesHtml = 'No issues detected';
            }
              table.innerHTML += `
                <tr>
                    <td>${frame.frame_index + 1}</td>
                    <td>${phaseName}</td>
                    <td>
                        <div class="d-flex align-items-center">
                            <div class="progress flex-grow-1" style="height: 20px;">
                                <div class="progress-bar ${frame.similarity_score < 0.7 ? 'bg-warning' : ''}" role="progressbar" 
                                    style="width: ${Math.round(frame.similarity_score * 100)}%" 
                                    aria-valuenow="${Math.round(frame.similarity_score * 100)}" 
                                    aria-valuemin="0" aria-valuemax="100">
                                </div>
                            </div>
                            <span class="ms-2">${Math.round(frame.similarity_score * 100)}%</span>
                        </div>
                    </td>
                    <td>${issuesHtml}</td>
                </tr>
            `;
        });
        
        table.innerHTML += '</tbody>';
        tableDiv.appendChild(table);
        comparisonDiv.appendChild(tableDiv);
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

// Function to show frame detail in a modal
function showFrameDetail(frameIndex, similarityScore, phaseName, userAnnotatedUrl, refAnnotatedUrl, issuesHtml) {
    // Remove any existing modal
    const existingModal = document.getElementById('frameDetailModal');
    if (existingModal) {
        existingModal.remove();
    }

    // Parse the issuesHtml to extract issues array
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = issuesHtml;
    const issues = Array.from(tempDiv.querySelectorAll('li')).map(li => ({
        description: li.textContent
    }));

    const feedback = getFeedbackForScore(similarityScore / 100, issues); // Convert percentage to decimal

    // Create new modal container
    const modalContainer = document.createElement('div');
    modalContainer.id = 'frameDetailModal';
    modalContainer.className = 'modal fade';
    modalContainer.setAttribute('tabindex', '-1');

    // Set modal content with a cleaner background for feedback section
    modalContainer.innerHTML = `
        <div class="modal-dialog modal-xl">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">${phaseName} - Frame ${frameIndex + 1}</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <div class="container-fluid">
                        <div class="row">
                            <div class="col-md-6">
                                <h4 class="text-center mb-3">Your Swing</h4>
                                <div class="text-center">
                                    ${userAnnotatedUrl ? 
                                        `<img src="${userAnnotatedUrl}" class="img-fluid" alt="Your Frame ${frameIndex + 1}" style="max-height: 70vh; object-fit: contain;">` : 
                                        '<div class="p-3 bg-light"><i class="bi bi-camera-video" style="font-size: 3rem;"></i><p class="mt-2">Frame not available</p></div>'
                                    }
                                </div>
                            </div>
                            <div class="col-md-6">
                                <h4 class="text-center mb-3">Reference</h4>
                                <div class="text-center">
                                    ${refAnnotatedUrl ? 
                                        `<img src="${refAnnotatedUrl}" class="img-fluid" alt="Reference Frame ${frameIndex + 1}" style="max-height: 70vh; object-fit: contain;">` : 
                                        '<div class="p-3 bg-light"><i class="bi bi-camera-video" style="font-size: 3rem;"></i><p class="mt-2">Frame not available</p></div>'
                                    }
                                </div>
                            </div>
                        </div>
                        <div class="row mt-4">
                            <div class="col-12">
                                <div class="d-flex align-items-center mb-2">
                                    <h5 class="mb-0 me-2">Similarity Score: ${similarityScore}%</h5>
                                    <i class="bi ${feedback.icon} ${feedback.class}" style="font-size: 1.2rem;"></i>
                                </div>
                                <div class="progress mb-3" style="height: 20px;">
                                    <div class="progress-bar ${similarityScore < 70 ? 'bg-danger' : (similarityScore < 90 ? 'bg-warning' : 'bg-success')}" 
                                         role="progressbar" 
                                         style="width: ${similarityScore}%" 
                                         aria-valuenow="${similarityScore}" 
                                         aria-valuemin="0" 
                                         aria-valuemax="100">
                                    </div>
                                </div>
                                <div class="${feedback.class} p-3 rounded mb-4" style="background-color: ${feedback.class === 'text-success' ? 'rgba(40, 167, 69, 0.1)' : (feedback.class === 'text-warning' ? 'rgba(255, 193, 7, 0.1)' : 'rgba(220, 53, 69, 0.1)')}">
                                    <h6 class="mb-2">
                                        <i class="bi ${feedback.icon} me-1"></i>
                                        ${feedback.text}
                                    </h6>
                                    <p class="mb-0">${feedback.details}</p>
                                </div>
                                <div class="issues-section mt-4">
                                    <h5>Analysis Details</h5>
                                    ${issuesHtml}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;

    // Add modal to document
    document.body.appendChild(modalContainer);

    // Initialize and show the modal
    const modal = new bootstrap.Modal(modalContainer);
    modal.show();

    // Clean up on close
    modalContainer.addEventListener('hidden.bs.modal', function() {
        this.remove();
    });
}