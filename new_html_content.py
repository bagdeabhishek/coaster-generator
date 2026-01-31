html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D Coaster Generator</title>
    <!-- Three.js -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <!-- STLLoader -->
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/STLLoader.js"></script>
    <!-- OrbitControls -->
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            opacity: 0.9;
            font-size: 1.1em;
        }
        
        .content {
            padding: 30px;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            font-weight: 600;
            margin-bottom: 8px;
            color: #333;
        }
        
        input[type="file"],
        input[type="number"],
        select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        
        input[type="file"]:focus,
        input[type="number"]:focus,
        select:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .checkbox-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        input[type="checkbox"] {
            width: 20px;
            height: 20px;
            cursor: pointer;
        }
        
        .checkbox-group label {
            margin: 0;
            cursor: pointer;
        }
        
        .parameters-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        
        @media (max-width: 600px) {
            .parameters-grid {
                grid-template-columns: 1fr;
            }
        }
        
        .parameter-note {
            font-size: 0.85em;
            color: #666;
            margin-top: 5px;
        }
        
        .generate-btn {
            width: 100%;
            padding: 16px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .generate-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        }
        
        .generate-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        /* Progress Section */
        .progress-section {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            display: none;
        }
        
        .progress-section.active {
            display: block;
        }
        
        .progress-bar-container {
            background: #e0e0e0;
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
            margin-bottom: 10px;
        }
        
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            width: 0%;
            transition: width 0.3s ease;
        }
        
        .progress-text {
            text-align: center;
            font-weight: 600;
            color: #333;
        }
        
        .status-message {
            text-align: center;
            margin-top: 10px;
            color: #666;
        }
        
        /* Review Section */
        .review-section {
            margin-top: 30px;
            padding: 25px;
            background: #fff3e0;
            border-radius: 8px;
            display: none;
            border: 2px solid #ff9800;
        }
        
        .review-section.active {
            display: block;
        }
        
        .review-section h3 {
            margin-bottom: 15px;
            color: #e65100;
            text-align: center;
            font-size: 1.5em;
        }
        
        .review-image-container {
            text-align: center;
            margin-bottom: 20px;
        }
        
        .review-image {
            max-width: 100%;
            max-height: 400px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        .review-actions {
            display: flex;
            gap: 15px;
            justify-content: center;
            flex-wrap: wrap;
        }
        
        .approve-btn {
            padding: 14px 32px;
            background: #4caf50;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s, transform 0.2s;
        }
        
        .approve-btn:hover {
            background: #45a049;
            transform: translateY(-2px);
        }
        
        .approve-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        
        .retry-btn {
            padding: 14px 32px;
            background: #ff9800;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s, transform 0.2s;
        }
        
        .retry-btn:hover {
            background: #f57c00;
            transform: translateY(-2px);
        }
        
        .retry-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        
        .retry-form {
            display: none;
            margin-top: 20px;
            padding: 20px;
            background: white;
            border-radius: 8px;
            border: 2px dashed #ff9800;
        }
        
        .retry-form.active {
            display: block;
        }
        
        .retry-form h4 {
            margin-bottom: 15px;
            color: #e65100;
        }
        
        .retry-file-input {
            margin-bottom: 15px;
        }
        
        .retry-submit-btn {
            padding: 12px 24px;
            background: #2196f3;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
        }
        
        .retry-submit-btn:hover {
            background: #1976d2;
        }
        
        .retry-submit-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        
        /* 3D Viewer Section */
        .viewer-section {
            margin-top: 30px;
            padding: 25px;
            background: #e3f2fd;
            border-radius: 8px;
            display: none;
            border: 2px solid #2196f3;
        }
        
        .viewer-section.active {
            display: block;
        }
        
        .viewer-section h3 {
            margin-bottom: 15px;
            color: #1565c0;
            text-align: center;
            font-size: 1.5em;
        }
        
        .viewer-container {
            width: 100%;
            height: 400px;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 20px;
            position: relative;
        }
        
        .viewer-controls {
            display: flex;
            gap: 10px;
            justify-content: center;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }
        
        .viewer-btn {
            padding: 10px 20px;
            background: #2196f3;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
        }
        
        .viewer-btn:hover {
            background: #1976d2;
        }
        
        .viewer-btn.active {
            background: #1565c0;
        }
        
        .viewer-info {
            text-align: center;
            color: #666;
            font-size: 0.9em;
            margin-top: 10px;
        }
        
        /* Downloads Section */
        .downloads-section {
            margin-top: 30px;
            padding: 20px;
            background: #e8f5e9;
            border-radius: 8px;
            display: none;
        }
        
        .downloads-section.active {
            display: block;
        }
        
        .downloads-section h3 {
            margin-bottom: 15px;
            color: #2e7d32;
        }
        
        .download-buttons {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        
        .download-btn {
            padding: 12px 24px;
            background: #4caf50;
            color: white;
            text-decoration: none;
            border-radius: 6px;
            font-weight: 600;
            transition: background 0.2s;
            border: none;
            cursor: pointer;
            font-size: 16px;
        }
        
        .download-btn:hover {
            background: #45a049;
        }
        
        /* Error Section */
        .error-section {
            margin-top: 30px;
            padding: 20px;
            background: #ffebee;
            border-radius: 8px;
            display: none;
        }
        
        .error-section.active {
            display: block;
        }
        
        .error-section h3 {
            color: #c62828;
            margin-bottom: 10px;
        }
        
        .note {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin-top: 20px;
            border-radius: 4px;
        }
        
        .note p {
            color: #856404;
            font-size: 0.95em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>3D Coaster Generator</h1>
            <p>Transform images into 3D printable coasters</p>
        </div>
        
        <div class="content">
            <form id="coasterForm">
                <div class="form-group">
                    <label for="image">Upload Image</label>
                    <input type="file" id="image" name="image" accept="image/*" required>
                    <p class="parameter-note">Recommended: High contrast images with clear subjects</p>
                </div>
                
                <div class="parameters-grid">
                    <div class="form-group">
                        <label for="diameter">Diameter (mm)</label>
                        <input type="number" id="diameter" name="diameter" value="100.0" min="50" max="200" step="1">
                    </div>
                    
                    <div class="form-group">
                        <label for="thickness">Thickness (mm)</label>
                        <input type="number" id="thickness" name="thickness" value="5.0" min="2" max="20" step="0.5">
                    </div>
                    
                    <div class="form-group">
                        <label for="logo_depth">Logo Depth (mm)</label>
                        <input type="number" id="logo_depth" name="logo_depth" value="0.6" min="0.1" max="5" step="0.1">
                    </div>
                    
                    <div class="form-group">
                        <label for="scale">Logo Scale</label>
                        <input type="number" id="scale" name="scale" value="0.85" min="0.1" max="1.5" step="0.05">
                        <p class="parameter-note">Relative to coaster diameter</p>
                    </div>
                    
                    <div class="form-group">
                        <label for="top_rotate">Top Rotation (degrees)</label>
                        <input type="number" id="top_rotate" name="top_rotate" value="0" min="0" max="360" step="1">
                    </div>
                    
                    <div class="form-group">
                        <label for="bottom_rotate">Bottom Rotation (degrees)</label>
                        <input type="number" id="bottom_rotate" name="bottom_rotate" value="0" min="0" max="360" step="1">
                    </div>
                </div>
                
                <div class="form-group checkbox-group">
                    <input type="checkbox" id="flip_horizontal" name="flip_horizontal" checked>
                    <label for="flip_horizontal">Flip Horizontal (Mirror)</label>
                </div>
                
                <button type="submit" class="generate-btn" id="generateBtn">Generate 3D Coaster</button>
            </form>
            
            <!-- Progress Section -->
            <div class="progress-section" id="progressSection">
                <div class="progress-bar-container">
                    <div class="progress-bar" id="progressBar"></div>
                </div>
                <div class="progress-text" id="progressText">0%</div>
                <div class="status-message" id="statusMessage">Initializing...</div>
            </div>
            
            <!-- Review Section -->
            <div class="review-section" id="reviewSection">
                <h3>Review Generated Image</h3>
                <div class="review-image-container">
                    <img id="reviewImage" class="review-image" src="" alt="Generated Preview">
                </div>
                <div class="review-actions">
                    <button class="approve-btn" id="approveBtn">Approve & Continue</button>
                    <button class="retry-btn" id="retryBtn">Try Again</button>
                </div>
                <div class="retry-form" id="retryForm">
                    <h4>Upload a Different Image</h4>
                    <input type="file" id="retryImageInput" class="retry-file-input" accept="image/*">
                    <button class="retry-submit-btn" id="retrySubmitBtn">Submit New Image</button>
                </div>
            </div>
            
            <!-- 3D Viewer Section -->
            <div class="viewer-section" id="viewerSection">
                <h3>3D Preview</h3>
                <div class="viewer-controls">
                    <button class="viewer-btn active" id="viewBodyBtn" data-model="body">View Body</button>
                    <button class="viewer-btn" id="viewLogosBtn" data-model="logos">View Logos</button>
                    <button class="viewer-btn" id="viewBothBtn" data-model="both">View Both</button>
                </div>
                <div class="viewer-container" id="viewerContainer"></div>
                <p class="viewer-info">Click and drag to rotate • Scroll to zoom • Right-click to pan</p>
            </div>
            
            <!-- Downloads Section -->
            <div class="downloads-section" id="downloadsSection">
                <h3>Download Your Files</h3>
                <div class="download-buttons" id="downloadButtons">
                    <a href="#" class="download-btn" id="downloadBody">Download Body STL</a>
                    <a href="#" class="download-btn" id="downloadLogos">Download Logos STL</a>
                    <a href="#" class="download-btn" id="downloadPreview">Download Preview</a>
                </div>
            </div>
            
            <!-- Error Section -->
            <div class="error-section" id="errorSection">
                <h3>Error</h3>
                <p id="errorMessage"></p>
            </div>
            
            <div class="note">
                <p><strong>How it works:</strong> Upload an image and we'll generate a 3D coaster using AI to create a clean black & white design. You'll have a chance to review before the 3D model is created. The workflow involves: Image Generation (AI processing) → Review (your approval) → Vectorization → 3D Modeling → Completion.</p>
            </div>
        </div>
    </div>

    <script>
        let pollingInterval = null;
        let currentJobId = null;
        let scene = null;
        let camera = null;
        let renderer = null;
        let controls = null;
        let stlLoader = null;
        let currentMeshes = [];
        let downloadUrls = null;
        
        // Initialize Three.js scene
        function init3DViewer() {
            const container = document.getElementById('viewerContainer');
            
            // Scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x000000);
            
            // Camera
            camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
            camera.position.set(0, 0, 150);
            
            // Renderer
            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(container.clientWidth, container.clientHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            container.appendChild(renderer.domElement);
            
            // Controls
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            
            // Lights
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(50, 100, 50);
            scene.add(directionalLight);
            
            const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
            directionalLight2.position.set(-50, -50, 50);
            scene.add(directionalLight2);
            
            // STL Loader
            stlLoader = new THREE.STLLoader();
            
            // Handle window resize
            window.addEventListener('resize', onWindowResize, false);
            
            // Start animation loop
            animate();
        }
        
        function onWindowResize() {
            const container = document.getElementById('viewerContainer');
            if (camera && renderer) {
                camera.aspect = container.clientWidth / container.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(container.clientWidth, container.clientHeight);
            }
        }
        
        function animate() {
            requestAnimationFrame(animate);
            if (controls) {
                controls.update();
            }
            if (renderer && scene && camera) {
                renderer.render(scene, camera);
            }
        }
        
        function clearMeshes() {
            currentMeshes.forEach(mesh => {
                scene.remove(mesh);
                if (mesh.geometry) mesh.geometry.dispose();
                if (mesh.material) mesh.material.dispose();
            });
            currentMeshes = [];
        }
        
        function loadSTL(url, material) {
            return new Promise((resolve, reject) => {
                stlLoader.load(url,
                    function(geometry) {
                        geometry.computeVertexNormals();
                        geometry.center();
                        
                        const mesh = new THREE.Mesh(geometry, material);
                        
                        // Scale to fit view
                        const box = new THREE.Box3().setFromObject(mesh);
                        const size = box.getSize(new THREE.Vector3());
                        const maxDim = Math.max(size.x, size.y, size.z);
                        const scale = 80 / maxDim;
                        mesh.scale.set(scale, scale, scale);
                        
                        scene.add(mesh);
                        currentMeshes.push(mesh);
                        resolve(mesh);
                    },
                    function(xhr) {
                        console.log((xhr.loaded / xhr.total * 100) + '% loaded');
                    },
                    function(error) {
                        console.error('Error loading STL:', error);
                        reject(error);
                    }
                );
            });
        }
        
        async function show3DViewer(urls) {
            downloadUrls = urls;
            document.getElementById('progressSection').classList.remove('active');
            document.getElementById('viewerSection').classList.add('active');
            document.getElementById('downloadsSection').classList.add('active');
            
            // Update download links
            document.getElementById('downloadBody').href = urls.body;
            document.getElementById('downloadLogos').href = urls.logos;
            document.getElementById('downloadPreview').href = urls.preview;
            
            // Initialize viewer if not already done
            if (!renderer) {
                init3DViewer();
            }
            
            // Load both models by default
            await loadBothModels();
        }
        
        async function loadBodyModel() {
            clearMeshes();
            const material = new THREE.MeshPhongMaterial({ 
                color: 0x3498db,
                specular: 0x444444,
                shininess: 60
            });
            try {
                await loadSTL(downloadUrls.body, material);
            } catch (error) {
                console.error('Failed to load body model:', error);
            }
        }
        
        async function loadLogosModel() {
            clearMeshes();
            const material = new THREE.MeshPhongMaterial({ 
                color: 0xe74c3c,
                specular: 0x444444,
                shininess: 60
            });
            try {
                await loadSTL(downloadUrls.logos, material);
            } catch (error) {
                console.error('Failed to load logos model:', error);
            }
        }
        
        async function loadBothModels() {
            clearMeshes();
            
            const bodyMaterial = new THREE.MeshPhongMaterial({ 
                color: 0x3498db,
                specular: 0x444444,
                shininess: 60,
                transparent: true,
                opacity: 0.9
            });
            
            const logosMaterial = new THREE.MeshPhongMaterial({ 
                color: 0xe74c3c,
                specular: 0x444444,
                shininess: 60
            });
            
            try {
                await Promise.all([
                    loadSTL(downloadUrls.body, bodyMaterial),
                    loadSTL(downloadUrls.logos, logosMaterial)
                ]);
            } catch (error) {
                console.error('Failed to load models:', error);
            }
        }
        
        // Viewer button handlers
        document.getElementById('viewBodyBtn').addEventListener('click', async function() {
            setActiveViewerButton(this);
            await loadBodyModel();
        });
        
        document.getElementById('viewLogosBtn').addEventListener('click', async function() {
            setActiveViewerButton(this);
            await loadLogosModel();
        });
        
        document.getElementById('viewBothBtn').addEventListener('click', async function() {
            setActiveViewerButton(this);
            await loadBothModels();
        });
        
        function setActiveViewerButton(btn) {
            document.querySelectorAll('.viewer-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        }
        
        // Form submission
        document.getElementById('coasterForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            // Reset UI
            document.getElementById('errorSection').classList.remove('active');
            document.getElementById('downloadsSection').classList.remove('active');
            document.getElementById('reviewSection').classList.remove('active');
            document.getElementById('viewerSection').classList.remove('active');
            document.getElementById('generateBtn').disabled = true;
            document.getElementById('generateBtn').textContent = 'Processing...';
            
            // Clear any existing meshes
            if (scene) {
                clearMeshes();
            }
            
            // Get form data
            const formData = new FormData(e.target);
            
            // Convert checkbox values to boolean strings
            formData.set('flip_horizontal', document.getElementById('flip_horizontal').checked);
            
            try {
                // Start processing
                const response = await fetch('/api/process', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Failed to start processing');
                }
                
                const data = await response.json();
                currentJobId = data.job_id;
                
                // Show progress section
                document.getElementById('progressSection').classList.add('active');
                
                // Start polling
                startPolling(currentJobId);
                
            } catch (error) {
                showError(error.message);
                resetForm();
            }
        });
        
        function startPolling(jobId) {
            // Clear any existing interval
            if (pollingInterval) {
                clearInterval(pollingInterval);
            }
            
            // Poll immediately
            pollStatus(jobId);
            
            // Then poll every second
            pollingInterval = setInterval(() => pollStatus(jobId), 1000);
        }
        
        async function pollStatus(jobId) {
            try {
                const response = await fetch(`/api/status/${jobId}`);
                
                if (!response.ok) {
                    throw new Error('Failed to get status');
                }
                
                const data = await response.json();
                
                // Update progress
                document.getElementById('progressBar').style.width = data.progress + '%';
                document.getElementById('progressText').textContent = data.progress + '%';
                document.getElementById('statusMessage').textContent = data.message;
                
                // Handle different statuses
                if (data.status === 'review') {
                    // Show review section
                    clearInterval(pollingInterval);
                    showReviewSection(jobId);
                } else if (data.status === 'completed') {
                    clearInterval(pollingInterval);
                    show3DViewer(data.download_urls);
                    resetForm();
                } else if (data.status === 'failed') {
                    clearInterval(pollingInterval);
                    showError(data.error || 'Processing failed');
                    resetForm();
                }
                
            } catch (error) {
                console.error('Polling error:', error);
            }
        }
        
        function showReviewSection(jobId) {
            document.getElementById('progressSection').classList.remove('active');
            document.getElementById('reviewSection').classList.add('active');
            
            // Set the review image
            document.getElementById('reviewImage').src = `/api/preview-image/${jobId}?t=${Date.now()}`;
            
            // Enable buttons
            document.getElementById('approveBtn').disabled = false;
            document.getElementById('retryBtn').disabled = false;
            
            // Reset retry form
            document.getElementById('retryForm').classList.remove('active');
            document.getElementById('retryImageInput').value = '';
        }
        
        // Approve button handler
        document.getElementById('approveBtn').addEventListener('click', async function() {
            if (!currentJobId) return;
            
            this.disabled = true;
            document.getElementById('retryBtn').disabled = true;
            this.textContent = 'Processing...';
            
            try {
                const response = await fetch(`/api/confirm/${currentJobId}`, {
                    method: 'POST'
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Failed to confirm');
                }
                
                // Hide review section and show progress
                document.getElementById('reviewSection').classList.remove('active');
                document.getElementById('progressSection').classList.add('active');
                
                // Continue polling
                startPolling(currentJobId);
                
            } catch (error) {
                showError(error.message);
                this.disabled = false;
                document.getElementById('retryBtn').disabled = false;
                this.textContent = 'Approve & Continue';
            }
        });
        
        // Try Again button handler
        document.getElementById('retryBtn').addEventListener('click', function() {
            document.getElementById('retryForm').classList.add('active');
            document.getElementById('retryImageInput').focus();
        });
        
        // Retry submit button handler
        document.getElementById('retrySubmitBtn').addEventListener('click', async function() {
            if (!currentJobId) return;
            
            const fileInput = document.getElementById('retryImageInput');
            if (!fileInput.files || fileInput.files.length === 0) {
                alert('Please select an image file');
                return;
            }
            
            this.disabled = true;
            this.textContent = 'Uploading...';
            
            const formData = new FormData();
            formData.append('image', fileInput.files[0]);
            
            try {
                const response = await fetch(`/api/retry/${currentJobId}`, {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Failed to retry');
                }
                
                // Hide review section and show progress
                document.getElementById('reviewSection').classList.remove('active');
                document.getElementById('progressSection').classList.add('active');
                
                // Continue polling
                startPolling(currentJobId);
                
            } catch (error) {
                showError(error.message);
                this.disabled = false;
                this.textContent = 'Submit New Image';
            }
        });
        
        function showError(message) {
            document.getElementById('progressSection').classList.remove('active');
            document.getElementById('reviewSection').classList.remove('active');
            document.getElementById('errorSection').classList.add('active');
            document.getElementById('errorMessage').textContent = message;
        }
        
        function resetForm() {
            document.getElementById('generateBtn').disabled = false;
            document.getElementById('generateBtn').textContent = 'Generate 3D Coaster';
            document.getElementById('approveBtn').textContent = 'Approve & Continue';
            document.getElementById('retrySubmitBtn').disabled = false;
            document.getElementById('retrySubmitBtn').textContent = 'Submit New Image';
        }
    </script>
</body>
</html>
'''
