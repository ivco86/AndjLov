/**
 * Enhanced Lightbox
 * Zoom, pan, keyboard navigation, and gestures
 */

const lightboxState = {
    zoom: 1,
    offsetX: 0,
    offsetY: 0,
    isDragging: false,
    startX: 0,
    startY: 0,
    currentImageElement: null
};

// ============ Zoom Controls ============

function zoomIn() {
    lightboxState.zoom = Math.min(lightboxState.zoom * 1.3, 5);
    applyZoom();
}

function zoomOut() {
    lightboxState.zoom = Math.max(lightboxState.zoom / 1.3, 0.5);
    applyZoom();
}

function resetZoom() {
    lightboxState.zoom = 1;
    lightboxState.offsetX = 0;
    lightboxState.offsetY = 0;
    applyZoom();
}

function applyZoom() {
    const img = lightboxState.currentImageElement;
    if (!img) return;

    img.style.transform = `scale(${lightboxState.zoom}) translate(${lightboxState.offsetX}px, ${lightboxState.offsetY}px)`;
    img.style.cursor = lightboxState.zoom > 1 ? 'move' : 'default';
}

// ============ Pan (Drag) ============

function initImagePan(imgElement) {
    lightboxState.currentImageElement = imgElement;

    imgElement.addEventListener('mousedown', (e) => {
        if (lightboxState.zoom <= 1) return;

        lightboxState.isDragging = true;
        lightboxState.startX = e.clientX - lightboxState.offsetX;
        lightboxState.startY = e.clientY - lightboxState.offsetY;
        imgElement.style.cursor = 'grabbing';
        e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
        if (!lightboxState.isDragging) return;

        lightboxState.offsetX = e.clientX - lightboxState.startX;
        lightboxState.offsetY = e.clientY - lightboxState.startY;
        applyZoom();
    });

    document.addEventListener('mouseup', () => {
        if (lightboxState.isDragging) {
            lightboxState.isDragging = false;
            if (lightboxState.currentImageElement) {
                lightboxState.currentImageElement.style.cursor = lightboxState.zoom > 1 ? 'move' : 'default';
            }
        }
    });

    // Mouse wheel zoom
    imgElement.addEventListener('wheel', (e) => {
        e.preventDefault();

        if (e.deltaY < 0) {
            zoomIn();
        } else {
            zoomOut();
        }
    });
}

// ============ Keyboard Navigation ============

function setupLightboxKeyboard() {
    document.addEventListener('keydown', (e) => {
        const modal = document.getElementById('imageModal');
        if (modal.style.display !== 'block') return;

        switch(e.key) {
            case 'Escape':
                closeImageModal();
                break;
            case 'ArrowLeft':
                navigateImage(-1);
                e.preventDefault();
                break;
            case 'ArrowRight':
                navigateImage(1);
                e.preventDefault();
                break;
            case '+':
            case '=':
                zoomIn();
                e.preventDefault();
                break;
            case '-':
            case '_':
                zoomOut();
                e.preventDefault();
                break;
            case '0':
                resetZoom();
                e.preventDefault();
                break;
            case 'f':
                toggleFullscreen();
                e.preventDefault();
                break;
        }
    });
}

// ============ Navigation ============

function navigateImage(direction) {
    const currentIndex = state.images.findIndex(img => img.id === state.currentImage.id);
    if (currentIndex === -1) return;

    const nextIndex = currentIndex + direction;
    if (nextIndex < 0 || nextIndex >= state.images.length) return;

    const nextImage = state.images[nextIndex];
    openImageModal(nextImage);
    resetZoom();
}

// ============ Fullscreen ============

function toggleFullscreen() {
    const modal = document.getElementById('imageModal');

    if (!document.fullscreenElement) {
        modal.requestFullscreen().catch(err => {
            console.log('Fullscreen error:', err);
        });
    } else {
        document.exitFullscreen();
    }
}

// ============ Touch Gestures (Mobile) ============

let touchStartDistance = 0;
let initialZoom = 1;

function setupTouchGestures(imgElement) {
    imgElement.addEventListener('touchstart', (e) => {
        if (e.touches.length === 2) {
            // Pinch zoom
            touchStartDistance = getDistance(e.touches[0], e.touches[1]);
            initialZoom = lightboxState.zoom;
            e.preventDefault();
        }
    });

    imgElement.addEventListener('touchmove', (e) => {
        if (e.touches.length === 2) {
            e.preventDefault();
            const currentDistance = getDistance(e.touches[0], e.touches[1]);
            const scale = currentDistance / touchStartDistance;
            lightboxState.zoom = Math.max(0.5, Math.min(5, initialZoom * scale));
            applyZoom();
        }
    });

    // Swipe navigation
    let touchStartX = 0;
    let touchStartY = 0;

    imgElement.addEventListener('touchstart', (e) => {
        if (e.touches.length === 1) {
            touchStartX = e.touches[0].clientX;
            touchStartY = e.touches[0].clientY;
        }
    }, { passive: true });

    imgElement.addEventListener('touchend', (e) => {
        if (lightboxState.zoom > 1) return; // Don't swipe when zoomed

        const touchEndX = e.changedTouches[0].clientX;
        const touchEndY = e.changedTouches[0].clientY;

        const diffX = touchStartX - touchEndX;
        const diffY = touchStartY - touchEndY;

        // Horizontal swipe is dominant
        if (Math.abs(diffX) > Math.abs(diffY) && Math.abs(diffX) > 50) {
            if (diffX > 0) {
                // Swipe left = next
                navigateImage(1);
            } else {
                // Swipe right = previous
                navigateImage(-1);
            }
        }
    });
}

function getDistance(touch1, touch2) {
    const dx = touch1.clientX - touch2.clientX;
    const dy = touch1.clientY - touch2.clientY;
    return Math.sqrt(dx * dx + dy * dy);
}

// ============ Enhanced Modal Observer ============

function enhanceLightbox() {
    // Observe modal for image changes
    const modal = document.getElementById('imageModal');
    if (!modal) return;

    const observer = new MutationObserver(() => {
        const img = modal.querySelector('.image-main-view img');
        if (img && img !== lightboxState.currentImageElement) {
            resetZoom();
            initImagePan(img);
            setupTouchGestures(img);
        }
    });

    observer.observe(modal, { childList: true, subtree: true });

    // Add zoom controls to modal
    addZoomControls();
}

function addZoomControls() {
    const modal = document.getElementById('imageModal');
    if (!modal) return;

    // Check if controls already exist
    if (document.getElementById('lightboxZoomControls')) return;

    const controls = document.createElement('div');
    controls.id = 'lightboxZoomControls';
    controls.innerHTML = `
        <button onclick="zoomOut()" title="Zoom Out (-)">🔍−</button>
        <button onclick="resetZoom()" title="Reset Zoom (0)">⊙</button>
        <button onclick="zoomIn()" title="Zoom In (+)">🔍+</button>
        <button onclick="toggleFullscreen()" title="Fullscreen (F)">⛶</button>
    `;

    modal.appendChild(controls);
}

// ============ Initialize ============

// Setup keyboard navigation on page load
setupLightboxKeyboard();

// Enhance lightbox when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', enhanceLightbox);
} else {
    enhanceLightbox();
}
