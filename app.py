from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import base64
import time
import json
from pathlib import Path
from datetime import datetime
import numpy as np

app = FastAPI(title="Deepfake Detection System", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
class Config:
    IMG_SIZE = 224
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_DIR = Path(".")
    EFFICIENTNET_PATH = MODEL_DIR / "efficientnet_b0_deepfake.pth"
    MOBILENET_PATH = MODEL_DIR / "mobilenet_v3_small_deepfake.pth"

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Model loading functions
def load_efficientnet():
    """Load EfficientNet-B0 model"""
    model = models.efficientnet_b0(pretrained=False)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 1)
    
    checkpoint = torch.load(Config.EFFICIENTNET_PATH, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(Config.DEVICE)
    model.eval()
    
    return model, checkpoint

def load_mobilenet():
    """Load MobileNet-V3 Small model"""
    model = models.mobilenet_v3_small(pretrained=False)
    num_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_features, 1)
    
    checkpoint = torch.load(Config.MOBILENET_PATH, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(Config.DEVICE)
    model.eval()
    
    return model, checkpoint

# Load models at startup
print("Loading models...")
efficientnet_model, efficientnet_checkpoint = load_efficientnet()
mobilenet_model, mobilenet_checkpoint = load_mobilenet()
print(f"Models loaded successfully on {Config.DEVICE}")

# Model information
MODEL_INFO = {
    "EfficientNet-B0": {
        "parameters": sum(p.numel() for p in efficientnet_model.parameters()),
        "test_accuracy": efficientnet_checkpoint.get('test_accuracy', 'N/A'),
        "description": "Efficient and accurate, optimized for edge devices"
    },
    "MobileNet-V3 Small": {
        "parameters": sum(p.numel() for p in mobilenet_model.parameters()),
        "test_accuracy": mobilenet_checkpoint.get('test_accuracy', 'N/A'),
        "description": "Ultra-lightweight, designed for low-power IoT devices"
    }
}

def predict_image(image: Image.Image, model, model_name: str):
    """Run inference on a single image"""
    start_time = time.time()
    
    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(Config.DEVICE)
    
    # Inference
    with torch.no_grad():
        output = model(img_tensor)
        probability = torch.sigmoid(output).item()
    
    inference_time = (time.time() - start_time) * 1000  # Convert to ms
    
    # Determine prediction
    prediction = "REAL" if probability > 0.5 else "FAKE"
    confidence = probability if probability > 0.5 else (1 - probability)
    
    return {
        "model": model_name,
        "prediction": prediction,
        "confidence": round(confidence * 100, 2),
        "probability_real": round(probability * 100, 2),
        "probability_fake": round((1 - probability) * 100, 2),
        "inference_time_ms": round(inference_time, 2),
        "raw_output": round(probability, 4)
    }

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main HTML page"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deepfake Detection System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            color: #ffffff;
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        header {
            text-align: center;
            padding: 40px 0;
            border-bottom: 2px solid #ffffff;
            margin-bottom: 40px;
        }

        h1 {
            font-size: 3em;
            font-weight: 700;
            letter-spacing: 2px;
            margin-bottom: 10px;
        }

        .subtitle {
            font-size: 1.2em;
            color: #cccccc;
            margin-top: 10px;
        }

        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 40px;
        }

        .card {
            background: #2d2d2d;
            border: 2px solid #ffffff;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        }

        .card-title {
            font-size: 1.8em;
            font-weight: 600;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #ffffff;
        }

        .upload-section {
            margin-bottom: 30px;
        }

        .button-group {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
        }

        button, .file-input-wrapper {
            flex: 1;
            padding: 15px 30px;
            font-size: 1em;
            font-weight: 600;
            border: 2px solid #ffffff;
            background: #1a1a1a;
            color: #ffffff;
            cursor: pointer;
            border-radius: 8px;
            transition: all 0.3s ease;
            text-align: center;
        }

        button:hover, .file-input-wrapper:hover {
            background: #ffffff;
            color: #1a1a1a;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(255, 255, 255, 0.2);
        }

        button:active {
            transform: translateY(0);
        }

        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .file-input-wrapper {
            position: relative;
            overflow: hidden;
            display: inline-block;
        }

        .file-input-wrapper input[type=file] {
            position: absolute;
            left: -9999px;
        }

        #webcam-container {
            display: none;
            margin-top: 20px;
        }

        video, canvas {
            width: 100%;
            max-width: 500px;
            border: 2px solid #ffffff;
            border-radius: 8px;
            background: #000000;
            display: block;
            margin: 0 auto;
        }

        .image-preview {
            margin-top: 20px;
            text-align: center;
        }

        .image-preview img {
            max-width: 100%;
            max-height: 400px;
            border: 2px solid #ffffff;
            border-radius: 8px;
            margin-top: 10px;
        }

        .results-section {
            display: none;
        }

        .model-result {
            background: #1a1a1a;
            border: 2px solid #ffffff;
            border-radius: 8px;
            padding: 25px;
            margin-bottom: 20px;
        }

        .model-name {
            font-size: 1.5em;
            font-weight: 700;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #ffffff;
        }

        .prediction-badge {
            display: inline-block;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 1.3em;
            font-weight: 700;
            margin: 15px 0;
            border: 2px solid #ffffff;
        }

        .prediction-real {
            background: #ffffff;
            color: #1a1a1a;
        }

        .prediction-fake {
            background: #1a1a1a;
            color: #ffffff;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-top: 20px;
        }

        .metric-item {
            background: #2d2d2d;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #ffffff;
        }

        .metric-label {
            font-size: 0.9em;
            color: #cccccc;
            margin-bottom: 5px;
        }

        .metric-value {
            font-size: 1.3em;
            font-weight: 600;
        }

        .comparison-section {
            margin-top: 30px;
            padding-top: 30px;
            border-top: 2px solid #ffffff;
        }

        .comparison-title {
            font-size: 1.5em;
            font-weight: 600;
            margin-bottom: 20px;
        }

        .agreement-badge {
            display: inline-block;
            padding: 10px 20px;
            border-radius: 6px;
            font-weight: 600;
            margin: 10px 0;
            border: 2px solid #ffffff;
        }

        .agreement-yes {
            background: #ffffff;
            color: #1a1a1a;
        }

        .agreement-no {
            background: #1a1a1a;
            color: #ffffff;
        }

        .download-section {
            margin-top: 20px;
            text-align: center;
        }

        .loading {
            display: none;
            text-align: center;
            padding: 30px;
            font-size: 1.2em;
        }

        .spinner {
            border: 4px solid #2d2d2d;
            border-top: 4px solid #ffffff;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .info-section {
            background: #2d2d2d;
            border: 2px solid #ffffff;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
        }

        .info-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-top: 15px;
        }

        .info-item {
            text-align: center;
            padding: 15px;
            background: #1a1a1a;
            border-radius: 6px;
            border: 1px solid #ffffff;
        }

        .info-value {
            font-size: 1.5em;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .info-label {
            font-size: 0.9em;
            color: #cccccc;
        }

        @media (max-width: 968px) {
            .main-content {
                grid-template-columns: 1fr;
            }

            .metrics-grid {
                grid-template-columns: 1fr;
            }

            .info-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>DEEPFAKE DETECTION SYSTEM</h1>
            <p class="subtitle">Dual-Model AI-Powered Image Authentication</p>
            <p class="subtitle">EfficientNet-B0 & MobileNet-V3 Small</p>
        </header>

        <div class="main-content">
            <!-- Left Column: Input -->
            <div class="card">
                <h2 class="card-title">Image Input</h2>
                
                <div class="upload-section">
                    <div class="button-group">
                        <button id="webcam-btn" onclick="startWebcam()">Capture from Webcam</button>
                        <label class="file-input-wrapper">
                            Upload Image
                            <input type="file" id="file-input" accept="image/*" onchange="handleFileUpload(event)">
                        </label>
                    </div>

                    <div id="webcam-container">
                        <video id="webcam" autoplay playsinline muted></video>
                        <div style="text-align: center; margin-top: 15px;">
                            <button onclick="capturePhoto()">Capture Photo</button>
                            <button onclick="stopWebcam()">Cancel</button>
                        </div>
                    </div>

                    <canvas id="canvas" style="display: none;"></canvas>

                    <div class="image-preview" id="image-preview"></div>
                </div>

                <button id="analyze-btn" onclick="analyzeImage()" style="width: 100%; margin-top: 20px;" disabled>
                    ANALYZE IMAGE
                </button>

                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>Analyzing image with both models...</p>
                </div>
            </div>

            <!-- Right Column: Results -->
            <div class="card">
                <h2 class="card-title">Analysis Results</h2>
                
                <div id="results-placeholder" style="text-align: center; padding: 50px 0; color: #cccccc;">
                    <p style="font-size: 1.2em;">Upload or capture an image to begin analysis</p>
                    <p style="margin-top: 10px;">Both models will process your image simultaneously</p>
                </div>

                <div class="results-section" id="results-section">
                    <div id="results-container"></div>

                    <div class="comparison-section" id="comparison-section">
                        <h3 class="comparison-title">Model Comparison</h3>
                        <div id="comparison-content"></div>
                    </div>

                    <div class="download-section">
                        <button onclick="downloadResults()">Download Results (JSON)</button>
                    </div>
                </div>
            </div>
        </div>

        <!-- System Information -->
        <div class="info-section">
            <h3 style="margin-bottom: 15px;">System Information</h3>
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-value" id="device-info">Loading...</div>
                    <div class="info-label">Device</div>
                </div>
                <div class="info-item">
                    <div class="info-value">2 Models</div>
                    <div class="info-label">Active Models</div>
                </div>
                <div class="info-item">
                    <div class="info-value">224x224</div>
                    <div class="info-label">Input Size</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentImageData = null;
        let currentResults = null;
        let stream = null;

        // Load system info
        fetch('/api/system-info')
            .then(r => r.json())
            .then(data => {
                document.getElementById('device-info').textContent = data.device;
            });

        // Check webcam API availability on page load
        window.addEventListener('load', function() {
            const isSecure = window.isSecureContext !== false && (
                window.location.protocol === 'https:' || 
                window.location.hostname === 'localhost' || 
                window.location.hostname === '127.0.0.1' ||
                window.location.hostname === '[::1]' ||
                window.location.hostname.match(/^127\./)
            );
            
            if (!isSecure && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
                console.warn('⚠️ Webcam may not work: Page is not in a secure context.');
                console.warn('Please access via: http://localhost:8000 or https://');
                console.warn('Current URL:', window.location.href);
            }
            
            if (!navigator.mediaDevices) {
                console.error('❌ navigator.mediaDevices is not available');
                console.error('This usually means the page is not in a secure context.');
            } else if (!navigator.mediaDevices.getUserMedia) {
                console.error('❌ getUserMedia is not available on navigator.mediaDevices');
            } else {
                console.log('✅ Webcam API is available');
            }
        });

        async function startWebcam() {
            try {
                // Check if we're in a secure context (required for getUserMedia)
                // Chrome requires secure context: HTTPS, localhost, or 127.0.0.1
                const isSecureContext = window.isSecureContext !== false && (
                    window.location.protocol === 'https:' || 
                    window.location.hostname === 'localhost' || 
                    window.location.hostname === '127.0.0.1' ||
                    window.location.hostname === '[::1]' ||
                    window.location.hostname.match(/^127\./)
                );
                
                if (!isSecureContext) {
                    const currentUrl = window.location.href;
                    throw new Error(`Camera access requires a secure context. Please access this page via:\n- http://localhost:8000\n- http://127.0.0.1:8000\n- https://your-domain.com\n\nCurrent URL: ${currentUrl}\n\nNote: Accessing via IP address (e.g., http://192.168.x.x:8000) is not considered secure by Chrome.`);
                }
                
                // Debug information
                console.log('Secure context:', isSecureContext);
                console.log('navigator.mediaDevices:', navigator.mediaDevices);
                console.log('getUserMedia available:', navigator.mediaDevices && navigator.mediaDevices.getUserMedia);

                // Polyfill for older browsers or edge cases
                let getUserMedia = null;
                
                if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                    getUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
                } else if (navigator.getUserMedia) {
                    // Fallback for older API
                    getUserMedia = function(constraints) {
                        return new Promise(function(resolve, reject) {
                            navigator.getUserMedia(constraints, resolve, reject);
                        });
                    };
                } else if (navigator.webkitGetUserMedia) {
                    // Fallback for webkit prefix
                    getUserMedia = function(constraints) {
                        return new Promise(function(resolve, reject) {
                            navigator.webkitGetUserMedia(constraints, resolve, reject);
                        });
                    };
                } else if (navigator.mozGetUserMedia) {
                    // Fallback for moz prefix
                    getUserMedia = function(constraints) {
                        return new Promise(function(resolve, reject) {
                            navigator.mozGetUserMedia(constraints, resolve, reject);
                        });
                    };
                }
                
                if (!getUserMedia) {
                    const userAgent = navigator.userAgent || '';
                    const isChrome = /Chrome/.test(userAgent) && !/Edge/.test(userAgent);
                    const isFirefox = /Firefox/.test(userAgent);
                    const isSafari = /Safari/.test(userAgent) && !/Chrome/.test(userAgent);
                    
                    console.error('getUserMedia not found. Debug info:', {
                        mediaDevices: navigator.mediaDevices,
                        getUserMedia: navigator.getUserMedia,
                        webkitGetUserMedia: navigator.webkitGetUserMedia,
                        userAgent: userAgent,
                        isSecureContext: window.isSecureContext,
                        protocol: window.location.protocol,
                        hostname: window.location.hostname
                    });
                    
                    let errorMsg = 'Webcam API is not available. ';
                    if (isChrome) {
                        errorMsg += 'Please ensure you are accessing this page via http://localhost:8000 or https://. Chrome requires a secure context for camera access. If you are using an IP address, try using localhost instead.';
                    } else if (isFirefox || isSafari) {
                        errorMsg += 'Please ensure you are accessing this page via localhost or https://.';
                    } else {
                        errorMsg += 'Please use a modern browser like Chrome, Firefox, or Safari.';
                    }
                    throw new Error(errorMsg);
                }

                // Request webcam with better constraints for browser compatibility
                const constraints = {
                    video: {
                        width: { ideal: 1280 },
                        height: { ideal: 720 },
                        facingMode: 'user'
                    }
                };

                stream = await getUserMedia(constraints);
                const video = document.getElementById('webcam');
                
                // Set up video event handlers before setting srcObject
                video.onloadedmetadata = function() {
                    video.play().catch(err => {
                        console.error('Error playing video:', err);
                        alert('Error starting video stream. Please check your camera permissions.');
                    });
                };
                
                // Set the video source
                video.srcObject = stream;
                
                // Try to play immediately (some browsers don't need to wait for metadata)
                video.play().catch(err => {
                    // If autoplay fails, wait for metadata
                    console.log('Autoplay prevented, waiting for metadata...');
                });
                
                document.getElementById('webcam-container').style.display = 'block';
                document.getElementById('webcam-btn').disabled = true;
            } catch (err) {
                let errorMessage = 'Error accessing webcam: ';
                if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
                    errorMessage += 'Camera permission denied. Please allow camera access in your browser settings.';
                } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
                    errorMessage += 'No camera found. Please connect a camera and try again.';
                } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
                    errorMessage += 'Camera is already in use by another application.';
                } else if (err.name === 'OverconstrainedError' || err.name === 'ConstraintNotSatisfiedError') {
                    errorMessage += 'Camera does not support the required constraints.';
                } else {
                    errorMessage += err.message;
                }
                alert(errorMessage);
                console.error('Webcam error:', err);
            }
        }

        function stopWebcam() {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
                
                const video = document.getElementById('webcam');
                if (video && video.srcObject) {
                    video.srcObject = null;
                }
                
                document.getElementById('webcam-container').style.display = 'none';
                document.getElementById('webcam-btn').disabled = false;
            }
        }

        function capturePhoto() {
            const video = document.getElementById('webcam');
            const canvas = document.getElementById('canvas');
            
            // Check if video is ready and has valid dimensions
            if (!video || !video.videoWidth || !video.videoHeight) {
                alert('Video is not ready yet. Please wait a moment and try again.');
                return;
            }
            
            // Set canvas dimensions to match video
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            // Draw video frame to canvas
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            // Convert to image data URL
            try {
                currentImageData = canvas.toDataURL('image/jpeg', 0.95);
                displayImage(currentImageData);
                stopWebcam();
                document.getElementById('analyze-btn').disabled = false;
            } catch (err) {
                alert('Error capturing photo: ' + err.message);
                console.error('Capture error:', err);
            }
        }

        function handleFileUpload(event) {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    currentImageData = e.target.result;
                    displayImage(currentImageData);
                    document.getElementById('analyze-btn').disabled = false;
                };
                reader.readAsDataURL(file);
            }
        }

        function displayImage(dataUrl) {
            const preview = document.getElementById('image-preview');
            preview.innerHTML = `
                <h3 style="margin-bottom: 10px;">Selected Image</h3>
                <img src="${dataUrl}" alt="Selected image">
            `;
        }

        async function analyzeImage() {
            if (!currentImageData) return;

            document.getElementById('loading').style.display = 'block';
            document.getElementById('analyze-btn').disabled = true;
            document.getElementById('results-placeholder').style.display = 'none';

            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        image: currentImageData.split(',')[1]
                    })
                });

                if (!response.ok) {
                    throw new Error('Analysis failed');
                }

                currentResults = await response.json();
                displayResults(currentResults);
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('analyze-btn').disabled = false;
            }
        }

        function displayResults(data) {
            const container = document.getElementById('results-container');
            const comparisonContent = document.getElementById('comparison-content');
            
            let html = '';
            
            // Display results for each model
            data.predictions.forEach(pred => {
                html += `
                    <div class="model-result">
                        <div class="model-name">${pred.model}</div>
                        <div class="prediction-badge prediction-${pred.prediction.toLowerCase()}">
                            ${pred.prediction}
                        </div>
                        <div class="metrics-grid">
                            <div class="metric-item">
                                <div class="metric-label">Inference Time</div>
                                <div class="metric-value">${pred.inference_time_ms}ms</div>
                            </div>
                            <div class="metric-item">
                                <div class="metric-label">Model Parameters</div>
                                <div class="metric-value">${(pred.model_info.parameters / 1e6).toFixed(2)}M</div>
                            </div>
                            <div class="metric-item">
                                <div class="metric-label">Test Accuracy</div>
                                <div class="metric-value">${pred.model_info.test_accuracy}%</div>
                            </div>
                        </div>
                        <div style="margin-top: 15px; padding: 10px; background: #2d2d2d; border-radius: 6px; border: 1px solid #ffffff;">
                            <div class="metric-label">Model Description</div>
                            <div style="margin-top: 5px;">${pred.model_info.description}</div>
                        </div>
                    </div>
                `;
            });

            container.innerHTML = html;

            // Comparison section
            const agreement = data.predictions[0].prediction === data.predictions[1].prediction;
            const avgConfidence = ((data.predictions[0].confidence + data.predictions[1].confidence) / 2).toFixed(2);
            
            comparisonContent.innerHTML = `
                <div class="metrics-grid">
                    <div class="metric-item">
                        <div class="metric-label">Models Agreement</div>
                        <div class="agreement-badge agreement-${agreement ? 'yes' : 'no'}">
                            ${agreement ? 'AGREE' : 'DISAGREE'}
                        </div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Average Confidence</div>
                        <div class="metric-value">${avgConfidence}%</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Total Processing Time</div>
                        <div class="metric-value">${data.total_time_ms}ms</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Consensus</div>
                        <div class="metric-value">${agreement ? data.predictions[0].prediction : 'MIXED'}</div>
                    </div>
                </div>
            `;

            document.getElementById('results-section').style.display = 'block';
        }

        function downloadResults() {
            if (!currentResults) return;

            const dataStr = JSON.stringify(currentResults, null, 2);
            const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
            
            const exportFileDefaultName = `deepfake_analysis_${new Date().getTime()}.json`;
            
            const linkElement = document.createElement('a');
            linkElement.setAttribute('href', dataUri);
            linkElement.setAttribute('download', exportFileDefaultName);
            linkElement.click();
        }
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

@app.get("/api/system-info")
async def system_info():
    """Get system information"""
    return {
        "device": str(Config.DEVICE),
        "models_loaded": True,
        "models": MODEL_INFO
    }

@app.post("/api/predict")
async def predict(data: dict):
    """Predict if image is real or fake using both models"""
    try:
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        start_time = time.time()
        
        # Run both models
        efficientnet_result = predict_image(image, efficientnet_model, "EfficientNet-B0")
        mobilenet_result = predict_image(image, mobilenet_model, "MobileNet-V3 Small")
        
        total_time = (time.time() - start_time) * 1000
        
        # Add model info to results
        efficientnet_result['model_info'] = MODEL_INFO["EfficientNet-B0"]
        mobilenet_result['model_info'] = MODEL_INFO["MobileNet-V3 Small"]
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "predictions": [efficientnet_result, mobilenet_result],
            "total_time_ms": round(total_time, 2),
            "device": str(Config.DEVICE),
            "image_size": f"{image.size[0]}x{image.size[1]}"
        }
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": str(Config.DEVICE),
        "models_loaded": True
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("DEEPFAKE DETECTION SYSTEM")
    print("="*60)
    print(f"Device: {Config.DEVICE}")
    print(f"Models: EfficientNet-B0, MobileNet-V3 Small")
    print("="*60)
    print("\nStarting server at http://localhost:8000")
    print("Open your browser and navigate to http://localhost:8000")
    print("\nPress CTRL+C to stop the server")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")