/**
 * 3D Coaster Generator - Frontend Application
 * Handles form submission, Three.js viewer, and API communication
 */

// Generate device fingerprint for rate limiting
function generateDeviceFingerprint() {
    const components = [
        navigator.userAgent,
        screen.width + 'x' + screen.height,
        screen.colorDepth,
        navigator.language,
        new Date().getTimezoneOffset(),
        !!window.sessionStorage,
        !!window.localStorage
    ];
    
    // Simple hash function
    let hash = 0;
    const str = components.join('|');
    for (let i = 0; i < str.length; i++) {
        const char = str.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash;
    }
    return Math.abs(hash).toString(16);
}

// Get or create device fingerprint
function getDeviceFingerprint() {
    let fp = localStorage.getItem('deviceFingerprint');
    if (!fp) {
        fp = generateDeviceFingerprint();
        localStorage.setItem('deviceFingerprint', fp);
    }
    return fp;
}

// Global state
let pollingInterval = null;
let currentJobId = null;
let scene = null;
let camera = null;
let renderer = null;
let controls = null;
let stlLoader = null;
let currentMeshes = [];
let downloadUrls = null;

// DOM Elements
const elements = {
    // Form elements
    imageInput: document.getElementById('imageInput'),
    dropZone: document.getElementById('dropZone'),
    imagePreview: document.getElementById('imagePreview'),
    stampText: document.getElementById('stampText'),
    apiKey: document.getElementById('apiKey'),
    diameter: document.getElementById('diameter'),
    thickness: document.getElementById('thickness'),
    logoDepth: document.getElementById('logoDepth'),
    scale: document.getElementById('scale'),
    topRotate: document.getElementById('topRotate'),
    bottomRotate: document.getElementById('bottomRotate'),
    flipHorizontal: document.getElementById('flipHorizontal'),
    generateBtn: document.getElementById('generateBtn'),
    generateBtnText: document.getElementById('generateBtnText'),
    
    // Dark mode
    darkModeToggle: document.getElementById('darkModeToggle'),
    sunIcon: document.getElementById('sunIcon'),
    moonIcon: document.getElementById('moonIcon'),
    
    // Sections
    progressSection: document.getElementById('progressSection'),
    progressBar: document.getElementById('progressBar'),
    progressPercent: document.getElementById('progressPercent'),
    progressStatus: document.getElementById('progressStatus'),
    reviewSection: document.getElementById('reviewSection'),
    reviewImage: document.getElementById('reviewImage'),
    viewerSection: document.getElementById('viewerSection'),
    viewerContainer: document.getElementById('viewerContainer'),
    downloadsSection: document.getElementById('downloadsSection'),
    errorSection: document.getElementById('errorSection'),
    errorMessage: document.getElementById('errorMessage'),
    
    // Buttons
    approveBtn: document.getElementById('approveBtn'),
    retryBtn: document.getElementById('retryBtn'),
    retryForm: document.getElementById('retryForm'),
    retryImageInput: document.getElementById('retryImageInput'),
    retrySubmitBtn: document.getElementById('retrySubmitBtn'),
    dismissErrorBtn: document.getElementById('dismissErrorBtn'),
    fullscreenBtn: document.getElementById('fullscreenBtn'),
    
    // Viewer buttons
    viewBodyBtn: document.getElementById('viewBodyBtn'),
    viewLogosBtn: document.getElementById('viewLogosBtn'),
    viewBothBtn: document.getElementById('viewBothBtn'),
    
    // Downloads
    downloadBody: document.getElementById('downloadBody'),
    downloadLogos: document.getElementById('downloadLogos'),
};

// ============================================
// Dark Mode
// ============================================

function initDarkMode() {
    updateDarkModeIcons();
    
    elements.darkModeToggle.addEventListener('click', () => {
        if (document.documentElement.classList.contains('dark')) {
            document.documentElement.classList.remove('dark');
            localStorage.theme = 'light';
        } else {
            document.documentElement.classList.add('dark');
            localStorage.theme = 'dark';
        }
        updateDarkModeIcons();
    });
}

function updateDarkModeIcons() {
    if (document.documentElement.classList.contains('dark')) {
        elements.sunIcon.classList.remove('hidden');
        elements.moonIcon.classList.add('hidden');
    } else {
        elements.sunIcon.classList.add('hidden');
        elements.moonIcon.classList.remove('hidden');
    }
}

// ============================================
// File Upload
// ============================================

function initFileUpload() {
    // Click to upload
    elements.dropZone.addEventListener('click', () => elements.imageInput.click());
    
    // File selected
    elements.imageInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            showImagePreview(e.target.files[0]);
        }
    });
    
    // Drag and drop
    elements.dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.dropZone.classList.add('drop-zone-active');
    });
    
    elements.dropZone.addEventListener('dragleave', () => {
        elements.dropZone.classList.remove('drop-zone-active');
    });
    
    elements.dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.dropZone.classList.remove('drop-zone-active');
        
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            const file = e.dataTransfer.files[0];
            if (file.type.startsWith('image/')) {
                elements.imageInput.files = e.dataTransfer.files;
                showImagePreview(file);
            } else {
                showError('Please upload an image file');
            }
        }
    });
}

function showImagePreview(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        elements.imagePreview.src = e.target.result;
        elements.imagePreview.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
}

// ============================================
// Form Submission
// ============================================

function initForm() {
    elements.generateBtn.addEventListener('click', handleSubmit);
    
    elements.dismissErrorBtn.addEventListener('click', () => {
        elements.errorSection.classList.add('hidden');
    });
}

async function handleSubmit() {
    // Validate
    if (!elements.imageInput.files || !elements.imageInput.files[0]) {
        showError('Please select an image file');
        return;
    }
    
    // Reset UI
    resetUI();
    
    // Prepare form data
    const formData = new FormData();
    formData.append('image', elements.imageInput.files[0]);
    formData.append('stamp_text', elements.stampText.value || 'Abhishek Does Stuff');
    formData.append('api_key', elements.apiKey.value);
    formData.append('diameter', elements.diameter.value);
    formData.append('thickness', elements.thickness.value);
    formData.append('logo_depth', elements.logoDepth.value);
    formData.append('scale', elements.scale.value);
    formData.append('top_rotate', elements.topRotate.value);
    formData.append('bottom_rotate', elements.bottomRotate.value);
    formData.append('flip_horizontal', elements.flipHorizontal.checked);
    
    // Show loading state
    elements.generateBtn.disabled = true;
    elements.generateBtnText.textContent = 'Processing...';
    elements.progressSection.classList.remove('hidden');
    
    try {
        const response = await fetch('/api/process', {
            method: 'POST',
            headers: {
                'X-Device-Fingerprint': getDeviceFingerprint()
            },
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            // Handle rate limit errors specially
            if (response.status === 429) {
                const hours = Math.ceil(error.retry_after / 3600);
                throw new Error(`${error.message}\n\nYou can bypass this limit by using your own BFL API key.`);
            }
            throw new Error(error.detail || error.message || 'Failed to start processing');
        }
        
        const data = await response.json();
        currentJobId = data.job_id;
        
        // Start polling
        startPolling(currentJobId);
        
    } catch (error) {
        showError(error.message);
        resetForm();
    }
}

function resetUI() {
    // Hide all sections
    elements.errorSection.classList.add('hidden');
    elements.progressSection.classList.add('hidden');
    elements.reviewSection.classList.add('hidden');
    elements.viewerSection.classList.add('hidden');
    elements.downloadsSection.classList.add('hidden');
    
    // Reset progress
    updateProgress(0, 'Initializing...');
    
    // Clear viewer
    if (scene) {
        clearMeshes();
    }
}

function resetForm() {
    elements.generateBtn.disabled = false;
    elements.generateBtnText.textContent = 'Generate 3D Coaster';
    elements.approveBtn.disabled = false;
    elements.approveBtn.textContent = 'Approve & Continue';
    elements.retrySubmitBtn.disabled = false;
    elements.retrySubmitBtn.textContent = 'Submit New Image';
}

// ============================================
// Job Polling
// ============================================

function startPolling(jobId) {
    // Clear existing interval
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
        updateProgress(data.progress, data.message);
        
        // Handle different statuses
        if (data.status === 'review') {
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

function updateProgress(percent, message) {
    elements.progressBar.style.width = `${percent}%`;
    elements.progressPercent.textContent = `${percent}%`;
    elements.progressStatus.textContent = message;
}

// ============================================
// Review Section
// ============================================

function showReviewSection(jobId) {
    elements.progressSection.classList.add('hidden');
    elements.reviewSection.classList.remove('hidden');
    elements.reviewSection.classList.add('animate-fade-in');
    
    // Set the review image with cache-busting
    elements.reviewImage.src = `/api/preview-image/${jobId}?t=${Date.now()}`;
    
    // Reset retry form
    elements.retryForm.classList.add('hidden');
    elements.retryImageInput.value = '';
}

// Approve button
elements.approveBtn.addEventListener('click', async function() {
    if (!currentJobId) return;
    
    this.disabled = true;
    elements.retryBtn.disabled = true;
    this.innerHTML = '<span class="animate-spin inline-block mr-2">⟳</span> Processing...';
    
    try {
        const response = await fetch(`/api/confirm/${currentJobId}`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to confirm');
        }
        
        // Hide review and show progress
        elements.reviewSection.classList.add('hidden');
        elements.progressSection.classList.remove('hidden');
        
        // Continue polling
        startPolling(currentJobId);
        
    } catch (error) {
        showError(error.message);
        this.disabled = false;
        elements.retryBtn.disabled = false;
        this.textContent = 'Approve & Continue';
    }
});

// Retry button
elements.retryBtn.addEventListener('click', () => {
    elements.retryForm.classList.remove('hidden');
    elements.retryForm.classList.add('animate-fade-in');
    elements.retryImageInput.focus();
});

// Retry submit
elements.retrySubmitBtn.addEventListener('click', async function() {
    if (!currentJobId) return;
    
    if (!elements.retryImageInput.files || elements.retryImageInput.files.length === 0) {
        showError('Please select an image file');
        return;
    }
    
    this.disabled = true;
    this.textContent = 'Uploading...';
    
    const formData = new FormData();
    formData.append('image', elements.retryImageInput.files[0]);
    
    try {
        const response = await fetch(`/api/retry/${currentJobId}`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to retry');
        }
        
        // Hide review and show progress
        elements.reviewSection.classList.add('hidden');
        elements.progressSection.classList.remove('hidden');
        
        // Continue polling
        startPolling(currentJobId);
        
    } catch (error) {
        showError(error.message);
        this.disabled = false;
        this.textContent = 'Submit New Image';
    }
});

// ============================================
// 3D Viewer (Three.js)
// ============================================

function init3DViewer() {
    // Scene setup
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    
    // Camera
    const aspect = elements.viewerContainer.clientWidth / elements.viewerContainer.clientHeight;
    camera = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);
    camera.position.set(0, 0, 150);
    
    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(elements.viewerContainer.clientWidth, elements.viewerContainer.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    elements.viewerContainer.appendChild(renderer.domElement);
    
    // Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.enableZoom = true;
    controls.enablePan = true;
    
    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    
    const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight1.position.set(50, 100, 50);
    scene.add(directionalLight1);
    
    const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
    directionalLight2.position.set(-50, -50, 50);
    scene.add(directionalLight2);
    
    // STL Loader
    stlLoader = new THREE.STLLoader();
    
    // Handle resize
    window.addEventListener('resize', onWindowResize, false);
    
    // Fullscreen button
    elements.fullscreenBtn.addEventListener('click', toggleFullscreen);
    
    // Start animation loop
    animate();
}

function toggleFullscreen() {
    if (!document.fullscreenElement) {
        elements.viewerContainer.requestFullscreen().catch(err => {
            console.log('Fullscreen not supported');
        });
    } else {
        document.exitFullscreen();
    }
}

function onWindowResize() {
    if (!camera || !renderer) return;
    
    const aspect = elements.viewerContainer.clientWidth / elements.viewerContainer.clientHeight;
    camera.aspect = aspect;
    camera.updateProjectionMatrix();
    renderer.setSize(elements.viewerContainer.clientWidth, elements.viewerContainer.clientHeight);
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
    elements.progressSection.classList.add('hidden');
    elements.viewerSection.classList.remove('hidden');
    elements.viewerSection.classList.add('animate-fade-in');
    elements.downloadsSection.classList.remove('hidden');
    elements.downloadsSection.classList.add('animate-fade-in');
    
    // Update download links
    elements.downloadBody.href = urls.body;
    elements.downloadLogos.href = urls.logos;
    
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
elements.viewBodyBtn.addEventListener('click', async function() {
    setActiveViewerButton(this);
    await loadBodyModel();
});

elements.viewLogosBtn.addEventListener('click', async function() {
    setActiveViewerButton(this);
    await loadLogosModel();
});

elements.viewBothBtn.addEventListener('click', async function() {
    setActiveViewerButton(this);
    await loadBothModels();
});

function setActiveViewerButton(btn) {
    document.querySelectorAll('.viewer-btn').forEach(b => {
        b.classList.remove('active');
        b.classList.remove('bg-blue-800');
        b.classList.add('bg-blue-500');
    });
    btn.classList.add('active');
    btn.classList.remove('bg-blue-500');
    btn.classList.add('bg-blue-800');
}

// ============================================
// Error Handling
// ============================================

function showError(message) {
    elements.progressSection.classList.add('hidden');
    elements.reviewSection.classList.add('hidden');
    elements.errorSection.classList.remove('hidden');
    elements.errorSection.classList.add('animate-fade-in');
    elements.errorMessage.textContent = message;
}

// ============================================
// Initialization
// ============================================

function init() {
    initDarkMode();
    initFileUpload();
    initForm();
    
    console.log('3D Coaster Generator initialized');
}

// Start when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
