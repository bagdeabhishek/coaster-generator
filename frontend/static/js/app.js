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
let authState = { authenticated: false, user: null };
let enabledProviders = [];
let usageState = null;

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

    // Auth / usage
    loginGoogleBtn: document.getElementById('loginGoogleBtn'),
    authUserWrap: document.getElementById('authUserWrap'),
    authAvatar: document.getElementById('authAvatar'),
    authUserName: document.getElementById('authUserName'),
    logoutBtn: document.getElementById('logoutBtn'),
    usageSummary: document.getElementById('usageSummary'),
    usageDetails: document.getElementById('usageDetails'),
    upgradeBtn: document.getElementById('upgradeBtn'),
    
    // Sections
    emptyState: document.getElementById('emptyState'),
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
    downloadCoaster: document.getElementById('downloadCoaster'),
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

// ============================================
// Auth + Usage
// ============================================

async function initAuthAndUsage() {
    if (elements.loginGoogleBtn) {
        elements.loginGoogleBtn.addEventListener('click', () => {
            window.location.href = '/auth/login/google';
        });
    }

    if (elements.logoutBtn) {
        elements.logoutBtn.addEventListener('click', async () => {
            try {
                await fetch('/api/auth/logout', { method: 'POST' });
            } catch (_) {
                // ignore logout network errors
            }
            authState = { authenticated: false, user: null };
            renderAuth();
            await refreshUsage();
        });
    }

    if (elements.upgradeBtn) {
        elements.upgradeBtn.addEventListener('click', async () => {
            try {
                const response = await fetch('/api/billing/checkout', { method: 'POST' });
                const data = await response.json();
                if (!response.ok) {
                    throw new Error(data.detail || data.message || 'Unable to start checkout');
                }
                const checkoutUrl = data.checkout_url || data.url;
                if (!checkoutUrl) {
                    throw new Error('Checkout URL missing in response');
                }
                window.location.href = checkoutUrl;
            } catch (error) {
                showError(error.message || 'Unable to start checkout');
            }
        });
    }

    const devClearBtn = document.getElementById('devClearQuotaBtn');
    if (devClearBtn) {
        devClearBtn.addEventListener('click', async () => {
            try {
                const response = await fetch('/api/dev/clear-quotas', { method: 'POST' });
                if (response.ok) {
                    alert('Quotas cleared successfully!');
                    await refreshUsage();
                } else {
                    const data = await response.json();
                    alert(`Failed to clear: ${data.detail || 'Unknown error'}`);
                }
            } catch (e) {
                alert('Error connecting to server.');
            }
        });
    }

    await Promise.all([loadAuthProviders(), loadAuthState()]);
    renderAuth();
    await refreshUsage();
}

async function loadAuthProviders() {
    try {
        const response = await fetch('/api/auth/providers');
        const data = await response.json();
        enabledProviders = Array.isArray(data.providers) ? data.providers : [];
    } catch (_) {
        enabledProviders = [];
    }
}

async function loadAuthState() {
    try {
        const response = await fetch('/api/auth/me');
        const data = await response.json();
        authState = {
            authenticated: Boolean(data.authenticated),
            user: data.user || null,
        };
    } catch (_) {
        authState = { authenticated: false, user: null };
    }
}

function renderAuth() {
    const googleAvailable = enabledProviders.includes('google');

    if (elements.loginGoogleBtn) {
        if (!authState.authenticated && googleAvailable) {
            elements.loginGoogleBtn.classList.remove('hidden');
        } else {
            elements.loginGoogleBtn.classList.add('hidden');
        }
    }

    if (elements.authUserWrap) {
        if (authState.authenticated && authState.user) {
            elements.authUserWrap.classList.remove('hidden');
            elements.authUserWrap.classList.add('flex');
            if (elements.authUserName) {
                elements.authUserName.textContent = authState.user.name || authState.user.email || 'Signed in';
            }
            if (elements.authAvatar) {
                const avatar = authState.user.avatar_url;
                if (avatar) {
                    elements.authAvatar.src = avatar;
                    elements.authAvatar.classList.remove('hidden');
                } else {
                    elements.authAvatar.classList.add('hidden');
                }
            }
        } else {
            elements.authUserWrap.classList.add('hidden');
            elements.authUserWrap.classList.remove('flex');
        }
    }
}

async function refreshUsage() {
    try {
        const response = await fetch('/api/usage', {
            headers: {
                'X-Device-Fingerprint': getDeviceFingerprint(),
            },
        });
        const data = await response.json();
        usageState = data;
        renderUsage(data);
        
        // Show dev reset button if in dev mode
        if (data.debug_mode) {
            const devBtn = document.getElementById('devClearQuotaBtn');
            if (devBtn) devBtn.classList.remove('hidden');
        }
    } catch (_) {
        if (elements.usageSummary) {
            elements.usageSummary.textContent = 'Unable to load usage right now.';
        }
        if (elements.usageDetails) {
            elements.usageDetails.textContent = 'Try refreshing the page.';
        }
    }
}

function renderUsage(data) {
    if (!elements.usageSummary || !elements.usageDetails || !elements.upgradeBtn) return;

    const authenticated = Boolean(data.authenticated);
    const tier = data.tier || 'free';
    const nextAction = data.next_action || null;

    if (!authenticated) {
        const remaining = Number(data.remaining_anon || 0);
        elements.usageSummary.textContent = `Anonymous plan: ${remaining} free generation${remaining === 1 ? '' : 's'} left`;
    } else if (tier === 'paid') {
        const remaining = Number(data.paid_remaining || 0);
        elements.usageSummary.textContent = `Paid plan: ${remaining} generation${remaining === 1 ? '' : 's'} left this cycle`;
    } else {
        const remaining = Number(data.remaining_login_bonus || 0);
        elements.usageSummary.textContent = `Signed-in free bonus: ${remaining} generation${remaining === 1 ? '' : 's'} left`;
    }

    const detailParts = [];
    if (typeof data.remaining_anon === 'number') {
        detailParts.push(`Anon left: ${Math.max(0, data.remaining_anon)}`);
    }
    if (typeof data.remaining_login_bonus === 'number') {
        detailParts.push(`Login bonus left: ${Math.max(0, data.remaining_login_bonus)}`);
    }
    if (tier === 'paid') {
        detailParts.push(`Used this cycle: ${data.paid_used || 0}/${data.paid_limit || 0}`);
    }
    if (data.message) {
        detailParts.push(data.message);
    }

    elements.usageDetails.textContent = detailParts.join(' • ');

    const showUpgrade = authenticated && nextAction === 'upgrade';
    if (showUpgrade) {
        elements.upgradeBtn.classList.remove('hidden');
    } else {
        elements.upgradeBtn.classList.add('hidden');
    }
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
        if(elements.emptyState) elements.emptyState.classList.remove('hidden');
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
    
    // Hide ALL right pane content
    if(elements.emptyState) elements.emptyState.classList.add('hidden');
    elements.reviewSection.classList.add('hidden');
    elements.viewerSection.classList.add('hidden');
    elements.downloadsSection.classList.add('hidden');
    elements.errorSection.classList.add('hidden');
    
    // Show just progress
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
            const errorData = await response.json().catch(() => ({}));
            // Handle rate limit errors specially
            if (response.status === 429) {
                const error = errorData.detail || errorData;
                if (error.error === 'quota_exceeded') {
                    await refreshUsage();
                    throw new Error(error.message || 'You have reached your generation limit.');
                }
                const hours = Math.ceil((error.retry_after || 0) / 3600);
                const message = error.message || 'Rate limit exceeded';
                throw new Error(`${message}\n\nYou can bypass this limit by using your own BFL API key.`);
            }
            throw new Error(errorData.detail || errorData.message || 'Failed to start processing');
        }
        
        const data = await response.json();
        currentJobId = data.job_id;
        await refreshUsage();
        
        // Start polling
        startPolling(currentJobId);
        
    } catch (error) {
        showError(error.message);
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
        elements.viewerSection.classList.add('hidden');
        elements.downloadsSection.classList.add('hidden');
        if(elements.emptyState) elements.emptyState.classList.add('hidden');
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
            headers: {
                'X-Device-Fingerprint': getDeviceFingerprint(),
            },
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            if (response.status === 429) {
                const error = errorData.detail || errorData;
                await refreshUsage();
                throw new Error(error.message || 'You have reached your generation limit.');
            }
            throw new Error(errorData.detail || 'Failed to retry');
        }

        await refreshUsage();
        
        // Hide review and show progress
        elements.reviewSection.classList.add('hidden');
        elements.viewerSection.classList.add('hidden');
        elements.downloadsSection.classList.add('hidden');
        if(elements.emptyState) elements.emptyState.classList.add('hidden');
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
    // Clear any existing canvases to prevent stacking
    while (elements.viewerContainer.firstChild) {
        elements.viewerContainer.removeChild(elements.viewerContainer.firstChild);
    }

    // Scene setup
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    
    // Get container dimensions (fallback to default if not available)
    let width = elements.viewerContainer.clientWidth;
    let height = elements.viewerContainer.clientHeight;
    
    // If dimensions are 0, set default size
    if (width === 0 || height === 0) {
        width = 800;
        height = 600;
        // Fallback dimensions when container isn't ready
    }
    
    // Camera - position further back to see the models
    const aspect = width / height;
    camera = new THREE.PerspectiveCamera(50, aspect, 0.1, 2000);
    camera.position.set(0, 0, 300);
    camera.lookAt(0, 0, 0);
    
    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    
    // Ensure canvas doesn't expand beyond container
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';
    renderer.domElement.style.display = 'block';
    renderer.domElement.style.position = 'absolute';
    renderer.domElement.style.top = '0';
    renderer.domElement.style.left = '0';
    renderer.domElement.style.zIndex = '1';
    
    elements.viewerContainer.appendChild(renderer.domElement);
    
    // Controls - check if OrbitControls loaded
    if (typeof THREE.OrbitControls === 'function') {
        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.enableZoom = true;
        controls.enablePan = true;
    } else {
        // OrbitControls not loaded; continue without mouse controls
        controls = null;
    }
    
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
            // Fullscreen not supported; ignore
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

function frameCameraToMeshes() {
    if (!camera || currentMeshes.length === 0) return;

    const box = new THREE.Box3();
    currentMeshes.forEach(mesh => box.expandByObject(mesh));

    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);

    if (maxDim === 0) return;

    const fov = camera.fov * (Math.PI / 180);
    let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
    cameraZ *= 1.5;

    camera.position.set(center.x, center.y, center.z + cameraZ);
    camera.near = Math.max(0.1, cameraZ / 100);
    camera.far = cameraZ * 100;
    camera.updateProjectionMatrix();
    camera.lookAt(center);

    if (controls) {
        controls.target.copy(center);
        controls.update();
    }
}

function loadSTL(url, material) {
    return new Promise((resolve, reject) => {
        stlLoader.load(url,
            function(geometry) {
                geometry.computeVertexNormals();
                geometry.center();

                const mesh = new THREE.Mesh(geometry, material);

                // Scale to fit view - compute bounding box from geometry directly
                if (!geometry.boundingBox) {
                    geometry.computeBoundingBox();
                }
                const box = geometry.boundingBox;

                if (box) {
                    const size = new THREE.Vector3();
                    box.getSize(size);
                    const maxDim = Math.max(size.x, size.y, size.z);

                    if (maxDim > 0) {
                        let scale = 80 / maxDim;
                        if (scale > 10) scale = 10;
                        if (scale < 0.1) scale = 0.1;
                        mesh.scale.set(scale, scale, scale);
                    }
                }

                scene.add(mesh);
                currentMeshes.push(mesh);
                resolve(mesh);
            },
            function(progress) {
                // Loading progress
                // Loading progress
            },
            function(error) {
                console.error('Error loading STL:', url, error);
                reject(error);
            }
        );
    });
}

async function show3DViewer(urls) {
    downloadUrls = urls;
    if(elements.emptyState) elements.emptyState.classList.add('hidden');
    elements.progressSection.classList.add('hidden');
    elements.viewerSection.classList.remove('hidden');
    elements.viewerSection.classList.add('animate-fade-in');
    elements.downloadsSection.classList.remove('hidden');
    elements.downloadsSection.classList.add('animate-fade-in');

    // Update download link (3MF file)
    elements.downloadCoaster.href = urls.combined;

    // Initialize viewer if not already done
    // Wait for the DOM to update so the container has proper dimensions
    if (!renderer) {
        await new Promise(resolve => setTimeout(resolve, 100));
        init3DViewer();
    }

    // Load both models by default (using internal STL URLs for viewer)
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
        frameCameraToMeshes();
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
        frameCameraToMeshes();
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
        frameCameraToMeshes();
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
    if(elements.emptyState) elements.emptyState.classList.add('hidden');
    elements.progressSection.classList.add('hidden');
    elements.reviewSection.classList.add('hidden');
    elements.viewerSection.classList.add('hidden');
    elements.downloadsSection.classList.add('hidden');
    elements.errorSection.classList.remove('hidden');
    elements.errorMessage.textContent = message;
    
    elements.generateBtn.disabled = false;
    elements.generateBtnText.textContent = 'Generate 3D Coaster';
}

// ============================================
// Initialization
// ============================================

function init() {
    initDarkMode();
    initFileUpload();
    initForm();
    initAuthAndUsage();
    
    // Initialized
}

// Start when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

// Advanced Settings Toggle
const advancedToggleBtn = document.getElementById('advancedToggleBtn');
const advancedContent = document.getElementById('advancedContent');
const advancedChevron = document.getElementById('advancedChevron');

if (advancedToggleBtn) {
    advancedToggleBtn.addEventListener('click', () => {
        advancedContent.classList.toggle('hidden');
        advancedChevron.classList.toggle('rotate-180');
    });
}
