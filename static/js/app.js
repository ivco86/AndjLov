// ============ AI Gallery Frontend Application ============

// Global State
const state = {
    images: [],
    boards: [],
    tags: [],
    currentView: 'all',
    currentBoard: null,
    currentImage: null,
    searchQuery: '',
    stats: {},
    aiStyles: {},
    externalApps: {},
    selectedStyle: 'classic',
    pendingAnalyzeImageId: null,
    uploadFiles: [],
    similarImagesCache: new Map(),

    // Selection mode
    selectionMode: false,
    selectedImages: new Set(),

    // Operation locks
    isScanning: false,
    isAnalyzing: false,
    isUploading: false,

    // Sorting
    imageSort: 'date-desc'
};

// Constants
const CONFIG = {
    MIN_IMAGE_HEIGHT: 180,
    MAX_IMAGE_HEIGHT: 450,
    BASE_HEIGHT: 250,
    SEARCH_DEBOUNCE_MS: 300,
    TOAST_DURATION_MS: 5000,
    SIMILAR_IMAGES_LIMIT: 6
};

// ============ Initialization ============

document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    attachEventListeners();
});

async function initializeApp() {
    showLoading();

    // Check system health
    await checkHealth();

    // Load initial data
    await Promise.all([
        loadImages(),
        loadBoards(),
        loadTags(),
        loadExternalApps(),
        updateStats()
    ]);

    hideLoading();

    // Update UI
    renderImages();
    renderBoards();
    renderTagCloud();
    updateCounts();
}

// ============ API Calls ============

async function apiCall(endpoint, options = {}) {
    try {
        const response = await fetch(`/api${endpoint}`, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            let errorMessage = 'API request failed';
            
            try {
                const error = await response.json();
                errorMessage = error.error || error.message || errorMessage;
            } catch (e) {
                errorMessage = `HTTP ${response.status}: ${response.statusText}`;
            }
            
            throw new Error(errorMessage);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        showToast(error.message, 'error');
        throw error;
    }
}

async function checkHealth() {
    try {
        const data = await apiCall('/health');
        
        // Update AI status indicator
        const statusEl = document.getElementById('aiStatus');
        if (data.ai_connected) {
            statusEl.textContent = '🟢 AI Connected';
            statusEl.classList.add('connected');
        } else {
            statusEl.textContent = '🔴 AI Offline';
            statusEl.classList.remove('connected');
        }
        
        state.stats = data.stats;
    } catch (error) {
        console.error('Health check failed:', error);
    }
}

async function loadImages(filters = {}) {
    try {
        const params = new URLSearchParams(filters);
        const data = await apiCall(`/images?${params}`);
        state.images = data.images;
        return data.images;
    } catch (error) {
        console.error('Failed to load images:', error);
        return [];
    }
}

async function loadBoards() {
    try {
        const data = await apiCall('/boards');
        state.boards = data.boards;
        return data.boards;
    } catch (error) {
        console.error('Failed to load boards:', error);
        return [];
    }
}

async function loadTags() {
    try {
        const data = await apiCall('/tags');
        state.tags = data.tags;
        return data.tags;
    } catch (error) {
        console.error('Failed to load tags:', error);
        return [];
    }
}

async function updateStats() {
    try {
        const data = await apiCall('/health');
        state.stats = data.stats;
        updateCounts();
    } catch (error) {
        console.error('Failed to update stats:', error);
    }
}

async function loadAIStyles() {
    try {
        const data = await apiCall('/ai/styles');
        state.aiStyles = data.styles;
        return data.styles;
    } catch (error) {
        console.error('Failed to load AI styles:', error);
        return {};
    }
}

async function loadExternalApps() {
    try {
        const data = await apiCall('/external-apps');
        state.externalApps = data.apps;
        return data.apps;
    } catch (error) {
        console.error('Failed to load external apps:', error);
        return {};
    }
}

async function scanDirectory() {
    if (state.isScanning) {
        showToast('Scan already in progress', 'warning');
        return;
    }
    
    state.isScanning = true;
    const scanBtn = document.getElementById('scanBtn');
    const scanBtnEmpty = document.getElementById('scanBtnEmpty');
    const originalText = scanBtn ? scanBtn.textContent : '';

    if (scanBtn) {
        scanBtn.disabled = true;
        scanBtn.textContent = '⏳ Scanning...';
    }
    if (scanBtnEmpty) {
        scanBtnEmpty.disabled = true;
        scanBtnEmpty.textContent = '⏳ Scanning...';
    }

    try {
        const data = await apiCall('/scan', { method: 'POST' });
        showToast(`Found ${data.found} images, ${data.new} new`, 'success');
        
        await loadImages();
        await updateStats();
        renderImages();
    } catch (error) {
        showToast('Scan failed: ' + error.message, 'error');
    } finally {
        state.isScanning = false;
        if (scanBtn) {
            scanBtn.disabled = false;
            scanBtn.textContent = originalText;
        }
        if (scanBtnEmpty) {
            scanBtnEmpty.disabled = false;
            scanBtnEmpty.textContent = '🔍 Scan Directory';
        }
    }
}

function openAIStyleModal(imageId, isBatchMode = false) {
    state.pendingAnalyzeImageId = imageId;
    state.isBatchAnalyze = isBatchMode;

    if (Object.keys(state.aiStyles).length === 0) {
        loadAIStyles().then(() => {
            renderAIStylesModal();
            document.getElementById('aiStyleModal').style.display = 'block';
        });
    } else {
        renderAIStylesModal();
        document.getElementById('aiStyleModal').style.display = 'block';
    }
}

function renderAIStylesModal() {
    const container = document.getElementById('styleSelection');
    if (!container) return;

    const stylesHTML = Object.entries(state.aiStyles).map(([key, style]) => `
        <label class="style-option" for="style-${key}">
            <input
                type="radio"
                id="style-${key}"
                name="aiStyle"
                value="${key}"
                ${state.selectedStyle === key ? 'checked' : ''}
            >
            <div class="style-info">
                <div class="style-name">${style.name}</div>
                <div class="style-description">${style.description}</div>
            </div>
        </label>
    `).join('');

    container.innerHTML = stylesHTML;

    container.querySelectorAll('input[name="aiStyle"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            state.selectedStyle = e.target.value;

            const customSection = document.getElementById('customPromptSection');
            if (e.target.value === 'custom') {
                customSection.style.display = 'block';
            } else {
                customSection.style.display = 'none';
            }

            container.querySelectorAll('.style-option').forEach(opt => {
                opt.classList.remove('selected');
            });
            e.target.closest('.style-option').classList.add('selected');
        });
    });
}

async function analyzeImage(imageId, style = null, customPrompt = null) {
    try {
        showToast('Analyzing image...', 'warning');

        const analyzeStyle = style || state.selectedStyle || 'classic';
        const requestBody = { style: analyzeStyle };

        if (analyzeStyle === 'custom' && customPrompt) {
            requestBody.custom_prompt = customPrompt;
        }

        const data = await apiCall(`/images/${imageId}/analyze`, {
            method: 'POST',
            body: JSON.stringify(requestBody)
        });
        
        // Update image in state
        const imageIndex = state.images.findIndex(img => img.id === imageId);
        if (imageIndex !== -1) {
            state.images[imageIndex].description = data.description;
            state.images[imageIndex].tags = data.tags;
            state.images[imageIndex].analyzed_at = new Date().toISOString();
            
            if (data.renamed && data.new_filename) {
                state.images[imageIndex].filename = data.new_filename;
            }
        }
        
        if (data.renamed) {
            showToast(`Analyzed & renamed to: ${data.new_filename}`, 'success');
        } else {
            showToast('Image analyzed successfully!', 'success');
        }
        
        // Update UI if modal is open
        if (state.currentImage && state.currentImage.id === imageId) {
            state.currentImage.description = data.description;
            state.currentImage.tags = data.tags;
            if (data.renamed && data.new_filename) {
                state.currentImage.filename = data.new_filename;
            }
            updateModal();
        }
        
        await updateStats();
        renderImages();
        
    } catch (error) {
        showToast('Analysis failed: ' + error.message, 'error');
        throw error;
    }
}

async function batchAnalyze(limit = 10) {
    if (state.isAnalyzing) {
        showToast('Analysis already in progress', 'warning');
        return;
    }
    
    state.isAnalyzing = true;
    const analyzeBtn = document.getElementById('analyzeBtn');
    const originalText = analyzeBtn.textContent;
    
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = '⏳ Analyzing...';
    
    try {
        showToast(`Analyzing up to ${limit} images...`, 'warning');
        
        const data = await apiCall(`/analyze-batch?limit=${limit}`, { method: 'POST' });
        
        if (data.renamed > 0) {
            showToast(`Analyzed ${data.analyzed} images, renamed ${data.renamed} files${data.failed ? `, ${data.failed} failed` : ''}`, 'success');
        } else {
            showToast(`Analyzed ${data.analyzed} images${data.failed ? `, ${data.failed} failed` : ''}`, 'success');
        }
        
        await loadImages();
        await updateStats();
        renderImages();
        
    } catch (error) {
        showToast('Batch analysis failed: ' + error.message, 'error');
    } finally {
        state.isAnalyzing = false;
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = originalText;
    }
}

async function toggleFavorite(imageId) {
    try {
        const data = await apiCall(`/images/${imageId}/favorite`, { method: 'POST' });
        
        const imageIndex = state.images.findIndex(img => img.id === imageId);
        if (imageIndex !== -1) {
            state.images[imageIndex].is_favorite = data.is_favorite;
        }
        
        if (state.currentImage && state.currentImage.id === imageId) {
            state.currentImage.is_favorite = data.is_favorite;
            updateModal();
        }
        
        await updateStats();
        renderImages();
        
    } catch (error) {
        showToast('Failed to toggle favorite: ' + error.message, 'error');
    }
}

async function renameImage(imageId, newFilename) {
    try {
        const data = await apiCall(`/images/${imageId}/rename`, {
            method: 'POST',
            body: JSON.stringify({ new_filename: newFilename })
        });

        showToast('Image renamed successfully!', 'success');

        const imageIndex = state.images.findIndex(img => img.id === imageId);
        if (imageIndex !== -1) {
            state.images[imageIndex].filename = data.new_filename;
            state.images[imageIndex].filepath = data.new_filepath;
        }

        if (state.currentImage && state.currentImage.id === imageId) {
            state.currentImage.filename = data.new_filename;
            state.currentImage.filepath = data.new_filepath;
            updateModal();
        }

        renderImages();
    } catch (error) {
        showToast('Rename failed: ' + error.message, 'error');
    }
}

async function openWithApp(imageId, appId) {
    try {
        const data = await apiCall(`/images/${imageId}/open-with`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ app_id: appId })
        });

        showToast(`Opening ${data.file} with ${data.app}`, 'success');
    } catch (error) {
        showToast('Failed to open with external app: ' + error.message, 'error');
    }
}

function showOpenWithMenu(event, imageId, mediaType) {
    const apps = state.externalApps[mediaType] || [];

    if (apps.length === 0) {
        showToast('No external applications configured', 'warning');
        return;
    }

    // Create dropdown menu
    const menu = document.createElement('div');
    menu.className = 'open-with-menu';
    menu.innerHTML = apps.map(app => `
        <div class="open-with-item" data-app-id="${app.id}">
            <span>${escapeHtml(app.name)}</span>
        </div>
    `).join('');

    // Position and show menu
    const button = event.target.closest('button');
    const rect = button.getBoundingClientRect();
    menu.style.position = 'fixed';
    menu.style.top = `${rect.bottom + 5}px`;
    menu.style.left = `${rect.left}px`;
    menu.style.zIndex = '10001';

    // Add click handlers
    menu.addEventListener('click', async (e) => {
        const item = e.target.closest('.open-with-item');
        if (item) {
            const appId = item.dataset.appId;
            await openWithApp(imageId, appId);
            document.body.removeChild(menu);
        }
    });

    // Close menu on outside click
    const closeMenu = (e) => {
        if (!menu.contains(e.target) && !button.contains(e.target)) {
            if (document.body.contains(menu)) {
                document.body.removeChild(menu);
            }
            document.removeEventListener('click', closeMenu);
        }
    };

    setTimeout(() => {
        document.addEventListener('click', closeMenu);
    }, 100);

    document.body.appendChild(menu);
}

function editImage(imageId) {
    if (!state.currentImage) return;
    openEditImageModal();
}

function openEditImageModal() {
    if (!state.currentImage) return;

    const modal = document.getElementById('editImageModal');
    const filenameInput = document.getElementById('editImageFilename');
    const descriptionInput = document.getElementById('editImageDescription');
    const tagsContainer = document.getElementById('editTagsContainer');

    // Populate current values
    filenameInput.value = state.currentImage.filename;
    descriptionInput.value = state.currentImage.description || '';

    // Store current tags in a temporary state
    state.editingTags = [...(state.currentImage.tags || [])];

    // Render tags
    renderEditTags();

    modal.style.display = 'block';
}

function closeEditImageModal() {
    closeModal('editImageModal', () => {
        state.editingTags = null;
    });
}

function renderEditTags() {
    const tagsContainer = document.getElementById('editTagsContainer');

    if (!state.editingTags || state.editingTags.length === 0) {
        tagsContainer.innerHTML = '<span style="color: var(--text-muted); font-size: 13px;">No tags yet</span>';
        return;
    }

    tagsContainer.innerHTML = state.editingTags.map(tag => `
        <span class="tag" style="display: flex; align-items: center; gap: 4px;">
            ${escapeHtml(tag)}
            <span onclick="removeEditTag('${escapeHtml(tag)}')" style="cursor: pointer; font-weight: bold;">×</span>
        </span>
    `).join('');
}

function addEditTag(tagName) {
    if (!tagName || !tagName.trim()) return;

    const trimmedTag = tagName.trim();

    // Check if tag already exists
    if (state.editingTags.includes(trimmedTag)) {
        showToast('Tag already exists', 'warning');
        return;
    }

    state.editingTags.push(trimmedTag);
    renderEditTags();

    // Clear input
    document.getElementById('editImageNewTag').value = '';
}

function removeEditTag(tagName) {
    state.editingTags = state.editingTags.filter(t => t !== tagName);
    renderEditTags();
}

async function saveImageEdit() {
    if (!state.currentImage) return;

    const imageId = state.currentImage.id;
    const newFilename = document.getElementById('editImageFilename').value.trim();
    const newDescription = document.getElementById('editImageDescription').value.trim();
    const newTags = state.editingTags || [];

    if (!newFilename) {
        showToast('Filename is required', 'error');
        return;
    }

    try {
        // Update filename if changed
        if (newFilename !== state.currentImage.filename) {
            await renameImage(imageId, newFilename);
        }

        // Update description and tags
        await apiCall(`/images/${imageId}`, {
            method: 'PATCH',
            body: JSON.stringify({
                description: newDescription,
                tags: newTags
            })
        });

        showToast('Image updated successfully!', 'success');

        // Refresh current image and reload
        await loadImages();
        
        // Reopen the image modal with updated data
        const updatedImage = state.images.find(img => img.id === imageId);
        if (updatedImage) {
            await openImageModal(updatedImage);
        }

        closeEditImageModal();
    } catch (error) {
        showToast('Failed to update image: ' + error.message, 'error');
    }
}

async function searchImages(query) {
    if (!query.trim()) {
        await loadImages();
        renderImages();
        updateBreadcrumb('All Images');
        return;
    }
    
    try {
        const data = await apiCall(`/images/search?q=${encodeURIComponent(query)}`);
        state.images = data.results;
        state.searchQuery = query;
        renderImages();
        updateBreadcrumb(`Search: "${query}"`);
    } catch (error) {
        showToast('Search failed: ' + error.message, 'error');
    }
}

async function searchByTag(tag) {
    await searchImages(tag);

    const searchInput = document.getElementById('searchInput');
    if (searchInput) {
        searchInput.value = tag;
    }
}

async function createBoard(name, description, parentId) {
    try {
        const data = await apiCall('/boards', {
            method: 'POST',
            body: JSON.stringify({
                name: name,
                description: description,
                parent_id: parentId || null
            })
        });
        
        showToast('Board created successfully!', 'success');
        
        await loadBoards();
        renderBoards();
        await updateStats();
        
        return data.board_id;
    } catch (error) {
        showToast('Failed to create board: ' + error.message, 'error');
        throw error;
    }
}

async function loadBoard(boardId) {
    try {
        const data = await apiCall(`/boards/${boardId}`);
        state.currentBoard = data;
        state.images = data.images;
        renderImages();
        updateBreadcrumb(data.name);
    } catch (error) {
        showToast('Failed to load board: ' + error.message, 'error');
    }
}

async function addImageToBoard(boardId, imageId) {
    try {
        await apiCall(`/boards/${boardId}/images`, {
            method: 'POST',
            body: JSON.stringify({ image_id: imageId })
        });
        return true;
    } catch (error) {
        console.error('Failed to add image to board:', error);
        return false;
    }
}

async function removeImageFromBoard(boardId, imageId) {
    try {
        await apiCall(`/boards/${boardId}/images`, {
            method: 'DELETE',
            body: JSON.stringify({ image_id: imageId })
        });
        return true;
    } catch (error) {
        console.error('Failed to remove image from board:', error);
        return false;
    }
}

async function renameBoard(boardId, newName, newDescription) {
    try {
        const data = await apiCall(`/boards/${boardId}`, {
            method: 'PUT',
            body: JSON.stringify({
                name: newName,
                description: newDescription
            })
        });

        showToast('Board renamed successfully!', 'success');

        await loadBoards();
        renderBoards();

        return true;
    } catch (error) {
        showToast('Failed to rename board: ' + error.message, 'error');
        return false;
    }
}

async function deleteBoard(boardId, deleteSubBoards = false) {
    try {
        const params = new URLSearchParams({ delete_sub_boards: deleteSubBoards });
        await apiCall(`/boards/${boardId}?${params}`, {
            method: 'DELETE'
        });

        showToast('Board deleted successfully!', 'success');

        await loadBoards();
        renderBoards();
        await updateStats();

        // If currently viewing the deleted board, switch to all images
        if (state.currentBoard && state.currentBoard.id === boardId) {
            switchView('all');
        }

        return true;
    } catch (error) {
        showToast('Failed to delete board: ' + error.message, 'error');
        return false;
    }
}

async function mergeBoards(sourceBoardId, targetBoardId, deleteSource = true) {
    try {
        const data = await apiCall(`/boards/${sourceBoardId}/merge`, {
            method: 'POST',
            body: JSON.stringify({
                target_board_id: targetBoardId,
                delete_source: deleteSource
            })
        });

        showToast(`Board merged successfully! Moved ${data.images_moved} images.`, 'success');

        await loadBoards();
        renderBoards();
        await updateStats();

        // If currently viewing the source board, switch to target board
        if (state.currentBoard && state.currentBoard.id === sourceBoardId) {
            switchView('board', targetBoardId);
        }

        return true;
    } catch (error) {
        showToast('Failed to merge boards: ' + error.message, 'error');
        return false;
    }
}

async function getImageDetails(imageId) {
    try {
        const data = await apiCall(`/images/${imageId}`);
        return data;
    } catch (error) {
        console.error('Failed to get image details:', error);
        return null;
    }
}

async function loadSimilarImages(imageId) {
    const container = document.getElementById('similarImages');
    if (!container) return;

    // Check cache first
    if (state.similarImagesCache.has(imageId)) {
        const cached = state.similarImagesCache.get(imageId);
        renderSimilarImagesInContainer(container, cached, imageId);
        return;
    }

    container.innerHTML = '<span class="tags-placeholder">Loading...</span>';

    try {
        const data = await apiCall(`/images/${imageId}/similar?limit=${CONFIG.SIMILAR_IMAGES_LIMIT}`);
        
        const similarImages = data.similar || [];
        state.similarImagesCache.set(imageId, similarImages);
        
        renderSimilarImagesInContainer(container, similarImages, imageId);
    } catch (error) {
        console.error('Failed to load similar images:', error);
        container.innerHTML = '<span class="tags-placeholder">Failed to load similar images</span>';
    }
}

function renderSimilarImagesInContainer(container, similarImages) {
    if (similarImages.length > 0) {
        container.innerHTML = similarImages.map(img => `
            <div class="similar-image-thumb" data-image-id="${img.id}">
                <img src="/api/images/${img.id}/thumbnail?size=400" alt="${escapeHtml(img.filename)}" loading="lazy">
            </div>
        `).join('');

        // ✅ Event delegation for similar image clicks
        container.onclick = async (e) => {
            const thumb = e.target.closest('.similar-image-thumb');
            if (thumb) {
                e.stopPropagation();
                const imageId = parseInt(thumb.dataset.imageId);
                const imageDetails = await getImageDetails(imageId);
                if (imageDetails) {
                    openImageModal(imageDetails);
                }
            }
        };
    } else {
        container.innerHTML = '<span class="tags-placeholder">No similar images found</span>';
    }
}

// ============ UI Rendering ============

function getConsistentHeight(imageId, width, height) {
    if (!width || !height) {
        const heights = [200, 250, 280, 320, 350, 400];
        return heights[imageId % heights.length];
    }
    
    const aspectRatio = height / width;
    let imageHeight = Math.floor(CONFIG.BASE_HEIGHT * aspectRatio);
    
    return Math.max(CONFIG.MIN_IMAGE_HEIGHT, Math.min(CONFIG.MAX_IMAGE_HEIGHT, imageHeight));
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function renderImages() {
    const grid = document.getElementById('imageGrid');
    const emptyState = document.getElementById('emptyState');

    if (state.images.length === 0) {
        grid.style.display = 'none';
        emptyState.style.display = 'flex';
        return;
    }

    grid.style.display = 'block';
    emptyState.style.display = 'none';

    // Sort images before rendering
    const sortedImages = sortImages([...state.images], state.imageSort);

    grid.innerHTML = sortedImages.map(image => createImageCard(image)).join('');
}

function sortImages(images, sortType) {
    const sorted = [...images];

    switch (sortType) {
        case 'date-desc':
            sorted.sort((a, b) => {
                const dateA = new Date(a.created_at || 0);
                const dateB = new Date(b.created_at || 0);
                return dateB - dateA; // Newest first
            });
            break;
        case 'date-asc':
            sorted.sort((a, b) => {
                const dateA = new Date(a.created_at || 0);
                const dateB = new Date(b.created_at || 0);
                return dateA - dateB; // Oldest first
            });
            break;
        case 'name-asc':
            sorted.sort((a, b) => a.filename.localeCompare(b.filename));
            break;
        case 'name-desc':
            sorted.sort((a, b) => b.filename.localeCompare(a.filename));
            break;
        case 'size-desc':
            sorted.sort((a, b) => (b.file_size || 0) - (a.file_size || 0));
            break;
        case 'size-asc':
            sorted.sort((a, b) => (a.file_size || 0) - (b.file_size || 0));
            break;
    }

    return sorted;
}

function createImageCard(image) {
    const favoriteClass = image.is_favorite ? 'active' : '';
    const isVideo = image.media_type === 'video';

    // Check if image has been analyzed (has tags or description)
    const isAnalyzed = image.tags && image.tags.length > 0;
    const statusIconClass = isAnalyzed ? 'status-icon-analyzed' : 'status-icon-pending';
    const statusIcon = isAnalyzed ? '✓' : '✗';

    const description = image.description || 'No description yet';
    const tags = image.tags.slice(0, 3);

    // Check if image is selected
    const isSelected = state.selectedImages.has(image.id);
    const checkboxClass = isSelected ? 'checked' : '';
    const selectedClass = isSelected ? 'selected' : '';

    return `
        <div class="image-card ${selectedClass}" data-id="${image.id}" data-media-type="${image.media_type || 'image'}">
            <div class="image-card-checkbox ${checkboxClass}" data-id="${image.id}"></div>
            <div class="image-card-status-icon ${statusIconClass}">${statusIcon}</div>
            ${isVideo ?
                `<div class="image-card-video-wrapper">
                    <img
                        class="image-card-image"
                        src="/api/images/${image.id}/thumbnail?size=500"
                        alt="${escapeHtml(image.filename)}"
                        loading="lazy"
                    >
                    <div class="video-play-overlay">
                        <div class="video-play-icon">▶</div>
                        <div class="video-icon-label">VIDEO</div>
                    </div>
                </div>` :
                `<img
                    class="image-card-image"
                    src="/api/images/${image.id}/thumbnail?size=500"
                    alt="${escapeHtml(image.filename)}"
                    loading="lazy"
                >`
            }
            <div class="image-card-content">
                <div class="image-card-header">
                    <div class="image-card-filename">${escapeHtml(image.filename)}</div>
                    <div class="image-card-favorite ${favoriteClass}">⭐</div>
                </div>
                <div class="image-card-description">${escapeHtml(description)}</div>
                <div class="image-card-tags">
                    ${tags.map(tag => `<span class="tag" data-tag="${escapeHtml(tag)}">${escapeHtml(tag)}</span>`).join('')}
                    ${image.tags.length > 3 ? `<span class="tag">+${image.tags.length - 3}</span>` : ''}
                </div>
            </div>
        </div>
    `;
}

function renderBoards() {
    const boardsList = document.getElementById('boardsList');
    const boardParentSelect = document.getElementById('boardParent');

    if (state.boards.length === 0) {
        boardsList.innerHTML = '<li style="color: var(--text-muted); padding: var(--spacing-sm);">No boards yet</li>';
        boardParentSelect.innerHTML = '<option value="">-- Top Level --</option>';
        return;
    }

    boardsList.innerHTML = state.boards.map(board => createBoardItem(board)).join('');

    boardParentSelect.innerHTML = '<option value="">-- Top Level --</option>' +
        state.boards.map(board => createBoardOption(board)).join('');
}

function renderTagCloud() {
    const tagCloud = document.getElementById('tagCloud');

    if (!state.tags || state.tags.length === 0) {
        tagCloud.innerHTML = '<span style="color: var(--text-muted); font-size: 0.75rem;">No tags yet</span>';
        return;
    }

    // Take top 20 tags
    const topTags = state.tags.slice(0, 20);

    // Find min/max counts for sizing
    const counts = topTags.map(t => t.count);
    const minCount = Math.min(...counts);
    const maxCount = Math.max(...counts);

    // Assign size class based on count
    const getTagSize = (count) => {
        if (maxCount === minCount) return 'tag-size-md';

        const ratio = (count - minCount) / (maxCount - minCount);
        if (ratio >= 0.8) return 'tag-size-xl';
        if (ratio >= 0.6) return 'tag-size-lg';
        if (ratio >= 0.4) return 'tag-size-md';
        if (ratio >= 0.2) return 'tag-size-sm';
        return 'tag-size-xs';
    };

    tagCloud.innerHTML = topTags.map(tagData => {
        const sizeClass = getTagSize(tagData.count);
        return `
            <span class="tag-cloud-item ${sizeClass}" data-tag="${escapeHtml(tagData.tag)}">
                ${escapeHtml(tagData.tag)}
                <span class="tag-cloud-count">${tagData.count}</span>
            </span>
        `;
    }).join('');

    // Add event delegation for tag cloud clicks
    // Clone to remove old listeners
    const tagCloudClone = tagCloud.cloneNode(true);
    tagCloud.parentNode.replaceChild(tagCloudClone, tagCloud);

    tagCloudClone.addEventListener('click', (e) => {
        const item = e.target.closest('.tag-cloud-item[data-tag]');
        if (item) {
            const tagValue = item.dataset.tag;
            searchByTag(tagValue);
        }
    });
}

function createBoardItem(board, isSubBoard = false) {
    const imageCount = board.image_count || 0;
    const hasSubBoards = board.sub_boards && board.sub_boards.length > 0;

    // Get gradient colors based on board name hash
    const gradients = [
        ['#ff006e', '#7b2cbf'],
        ['#00f5d4', '#7b2cbf'],
        ['#ff006e', '#00f5d4'],
        ['#7b2cbf', '#ff006e'],
        ['#c1121f', '#ff006e'],
        ['#00f5d4', '#c1121f']
    ];
    const hash = board.name.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
    const [color1, color2] = gradients[hash % gradients.length];

    let html = `
        <div class="board-pill ${hasSubBoards ? 'has-children' : ''}"
             data-board-id="${board.id}"
             data-has-children="${hasSubBoards}"
             style="background: linear-gradient(135deg, ${color1}, ${color2});">
            <span class="board-pill-name">${escapeHtml(board.name)}</span>
            <span class="board-pill-count">${imageCount}</span>
            ${hasSubBoards ? '<span class="board-pill-expand">▼</span>' : ''}
        </div>
    `;

    if (hasSubBoards) {
        html += `<div class="board-sub-pills" data-parent-id="${board.id}" style="display: none;">`;
        html += board.sub_boards.map(sub => createBoardItem(sub, true)).join('');
        html += `</div>`;
    }

    return html;
}

function createBoardOption(board, prefix = '') {
    let html = `<option value="${board.id}">${prefix}${escapeHtml(board.name)}</option>`;
    
    if (board.sub_boards && board.sub_boards.length > 0) {
        html += board.sub_boards.map(sub => createBoardOption(sub, prefix + '  ')).join('');
    }
    
    return html;
}

function updateCounts() {
    document.getElementById('allCount').textContent = state.stats.total_images || 0;
    document.getElementById('favCount').textContent = state.stats.favorite_images || 0;
    document.getElementById('unanalyzedCount').textContent = state.stats.unanalyzed_images || 0;
    document.getElementById('videosCount').textContent = state.stats.video_count || 0;

    document.getElementById('statTotal').textContent = state.stats.total_images || 0;
    document.getElementById('statAnalyzed').textContent = state.stats.analyzed_images || 0;
    document.getElementById('statBoards').textContent = state.stats.total_boards || 0;
}

function updateBreadcrumb(text) {
    const breadcrumb = document.getElementById('breadcrumb');
    breadcrumb.innerHTML = `<span class="breadcrumb-item">${escapeHtml(text)}</span>`;
}

// ============ View Switching ============

async function switchView(view, param = null) {
    state.currentView = view;
    state.currentBoard = null;
    state.searchQuery = '';
    
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('active');
    });
    
    showLoading();
    
    try {
        switch (view) {
            case 'all':
                await loadImages();
                updateBreadcrumb('All Images');
                const allView = document.querySelector('[data-view="all"]');
                if (allView) allView.classList.add('active');
                break;
                
            case 'favorites':
                await loadImages({ favorites: 'true' });
                updateBreadcrumb('Favorites');
                const favView = document.querySelector('[data-view="favorites"]');
                if (favView) favView.classList.add('active');
                break;
                
            case 'unanalyzed':
                await loadImages({ analyzed: 'false' });
                updateBreadcrumb('Unanalyzed');
                const unanalyzedView = document.querySelector('[data-view="unanalyzed"]');
                if (unanalyzedView) unanalyzedView.classList.add('active');
                break;

            case 'videos':
                await loadImages({ media_type: 'video' });
                updateBreadcrumb('Videos');
                const videosView = document.querySelector('[data-view="videos"]');
                if (videosView) videosView.classList.add('active');
                break;

            case 'board':
                await loadBoard(param);
                const boardItem = document.querySelector(`[data-board-id="${param}"]`);
                if (boardItem) boardItem.classList.add('active');
                break;
        }
        
        renderImages();
    } finally {
        hideLoading();
    }
}

// ============ Modal Management ============

async function openImageModal(image) {
    state.currentImage = image;
    
    const fullDetails = await getImageDetails(image.id);
    if (fullDetails) {
        state.currentImage = fullDetails;
    }
    
    const modal = document.getElementById('imageModal');
    modal.style.display = 'block';
    
    updateModal();
}

function updateModal() {
    const image = state.currentImage;
    const modal = document.getElementById('imageModal');
    const modalBody = modal.querySelector('.modal-body');

    const statusText = image.analyzed_at ? 'Analyzed' : 'Pending Analysis';
    const statusIcon = image.analyzed_at ? '✅' : '⏳';
    const isVideo = image.media_type === 'video';

    // Media viewer (image or video)
    const mediaViewer = isVideo
        ? `<video controls style="width: 100%; max-height: 86vh; object-fit: contain;">
               <source src="/api/images/${image.id}/file" type="video/mp4">
               Your browser does not support video playback.
           </video>`
        : `<img src="/api/images/${image.id}/file" alt="${escapeHtml(image.filename)}">`;

    const mediaTypeIcon = isVideo ? '🎬' : '📄';
    const mediaTypeText = isVideo ? 'Video' : 'Image';

    modalBody.innerHTML = `
        <div class="image-detail-container">
            <div class="image-main-view">
                ${mediaViewer}
            </div>

            <div class="image-info-panel">
                <div class="detail-section title-section">
                    <h2>${escapeHtml(image.title || image.filename)}</h2>
                    <span class="status-badge">
                        <span class="status-icon">${statusIcon}</span>
                        ${statusText}
                    </span>
                </div>

                <div class="detail-section description-section">
                    <h3>Description</h3>
                    <p class="${!image.description ? 'description-placeholder' : ''}">
                        ${escapeHtml(image.description || 'No description yet. Click analyze to generate.')}
                    </p>
                </div>

                <div class="detail-section tags-section">
                    <h3>Tags</h3>
                    <div class="tags-container">
                        ${image.tags && image.tags.length > 0
                            ? image.tags.map(tag => `<span class="tag" data-tag="${escapeHtml(tag)}">${escapeHtml(tag)}</span>`).join('')
                            : '<span class="tags-placeholder">No tags yet</span>'
                        }
                    </div>
                </div>

                <div class="detail-section boards-section">
                    <h3>Boards</h3>
                    <div class="boards-container">
                        ${image.boards && image.boards.length > 0
                            ? image.boards.map(board => `<span class="tag">${escapeHtml(board.name)}</span>`).join('')
                            : '<div class="boards-placeholder">Not in any boards</div>'
                        }
                    </div>
                </div>

                <div class="detail-section metadata-section">
                    <h3>Metadata</h3>
                    <div class="metadata-grid">
                        <div class="metadata-item">
                            <span class="metadata-label">Type</span>
                            <span class="metadata-value">${mediaTypeIcon} ${mediaTypeText}</span>
                        </div>
                        <div class="metadata-item">
                            <span class="metadata-label">Dimensions</span>
                            <span class="metadata-value">${image.width && image.height ? `${image.width} × ${image.height}` : 'N/A'}</span>
                        </div>
                        <div class="metadata-item">
                            <span class="metadata-label">File Size</span>
                            <span class="metadata-value">${formatFileSize(image.file_size)}</span>
                        </div>
                        <div class="metadata-item">
                            <span class="metadata-label">Status</span>
                            <span class="metadata-value highlight">${statusText}</span>
                        </div>
                    </div>
                </div>

                <div class="image-actions">
                    <button class="action-btn primary" onclick="openAIStyleModal(${image.id})">
                        ⚡ Analyze
                    </button>
                    <button class="action-btn secondary" onclick="editImage(${image.id})">
                        ✏️ Edit
                    </button>
                    <button class="action-btn secondary" onclick="openAddToBoardModal()">
                        📋 Boards
                    </button>
                    <button class="action-btn secondary" onclick="sendImageToTelegram(${image.id})">
                        📱 Send to Telegram
                    </button>
                    <button class="action-btn secondary" onclick="showOpenWithMenu(event, ${image.id}, '${image.media_type || 'image'}')">
                        🚀 Open With
                    </button>
                </div>
            </div>

            <div class="similar-panel">
                <div class="detail-section similar-section">
                    <h3>More Like This</h3>
                    <div class="similar-images-grid" id="similarImages">
                        <span class="tags-placeholder">Loading...</span>
                    </div>
                </div>
            </div>
        </div>
    `;

    // Add event delegation for tags in modal
    // Remove old listener by cloning and replacing the node
    const tagsContainer = modalBody.querySelector('.tags-container');
    if (tagsContainer) {
        const newTagsContainer = tagsContainer.cloneNode(true);
        tagsContainer.parentNode.replaceChild(newTagsContainer, tagsContainer);

        // Add new event listener
        newTagsContainer.addEventListener('click', (e) => {
            const tag = e.target.closest('.tag[data-tag]');
            if (tag) {
                e.stopPropagation();
                e.preventDefault();
                const tagValue = tag.dataset.tag;

                // Close modal first
                closeImageModal();

                // Then search by tag
                searchByTag(tagValue);
            }
        });
    }

    loadSimilarImages(image.id);
}

function closeModal(modalId, resetCallback = null) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.style.display = 'none';
    }
    
    if (resetCallback) {
        resetCallback();
    }
}

function closeImageModal() {
    closeModal('imageModal', () => {
        state.currentImage = null;
    });
}

function closeAIStyleModal() {
    closeModal('aiStyleModal', () => {
        state.pendingAnalyzeImageId = null;
    });
}

function openCreateBoardModal() {
    const modal = document.getElementById('createBoardModal');
    modal.style.display = 'block';
    
    document.getElementById('createBoardForm').reset();
}

function closeCreateBoardModal() {
    closeModal('createBoardModal');
}

async function openAddToBoardModal(imageId = null, isBatchMode = false) {
    state.isBatchBoardAdd = isBatchMode;

    if (!isBatchMode && !state.currentImage) return;

    const modal = document.getElementById('addToBoardModal');
    const selection = document.getElementById('boardSelection');

    let imageBoardIds = [];

    if (!isBatchMode) {
        const imageBoards = state.currentImage.boards || [];
        imageBoardIds = imageBoards.map(b => b.id);
    }

    selection.innerHTML = flattenBoards(state.boards).map(board => {
        const checked = !isBatchMode && imageBoardIds.includes(board.id) ? 'checked' : '';
        return `
            <div class="board-checkbox">
                <input
                    type="checkbox"
                    id="board-${board.id}"
                    value="${board.id}"
                    ${checked}
                >
                <label for="board-${board.id}">${escapeHtml((board.prefix || '') + board.name)}</label>
            </div>
        `;
    }).join('');

    modal.style.display = 'block';
}

function closeAddToBoardModal() {
    closeModal('addToBoardModal');
}

function flattenBoards(boards, prefix = '') {
    let result = [];

    for (const board of boards) {
        result.push({ ...board, prefix });

        if (board.sub_boards && board.sub_boards.length > 0) {
            result = result.concat(flattenBoards(board.sub_boards, prefix + '  '));
        }
    }

    return result;
}

// Board Management Modals
let currentBoardAction = { boardId: null, boardName: '' };

function openRenameBoardModal(boardId) {
    const board = findBoardById(boardId, state.boards);
    if (!board) return;

    currentBoardAction.boardId = boardId;
    currentBoardAction.boardName = board.name;

    document.getElementById('renameBoardName').value = board.name;
    document.getElementById('renameBoardDescription').value = board.description || '';

    const modal = document.getElementById('renameBoardModal');
    modal.style.display = 'block';
}

function closeRenameBoardModal() {
    closeModal('renameBoardModal', () => {
        currentBoardAction = { boardId: null, boardName: '' };
    });
}

function openMergeBoardModal(boardId) {
    const board = findBoardById(boardId, state.boards);
    if (!board) return;

    currentBoardAction.boardId = boardId;
    currentBoardAction.boardName = board.name;

    document.getElementById('mergeSourceName').textContent = board.name;

    // Populate target board dropdown (exclude the source board and its sub-boards)
    const targetSelect = document.getElementById('mergeTargetBoard');
    const excludedIds = new Set([boardId, ...getAllSubBoardIds(boardId, state.boards)]);

    const availableBoards = flattenBoards(state.boards).filter(b => !excludedIds.has(b.id));

    targetSelect.innerHTML = '<option value="">-- Select Board --</option>' +
        availableBoards.map(board =>
            `<option value="${board.id}">${escapeHtml((board.prefix || '') + board.name)}</option>`
        ).join('');

    const modal = document.getElementById('mergeBoardModal');
    modal.style.display = 'block';
}

function closeMergeBoardModal() {
    closeModal('mergeBoardModal', () => {
        currentBoardAction = { boardId: null, boardName: '' };
    });
}

function openDeleteBoardModal(boardId) {
    const board = findBoardById(boardId, state.boards);
    if (!board) return;

    currentBoardAction.boardId = boardId;
    currentBoardAction.boardName = board.name;

    document.getElementById('deleteSourceName').textContent = board.name;
    document.getElementById('deleteSubBoards').checked = false;

    const modal = document.getElementById('deleteBoardModal');
    modal.style.display = 'block';
}

function closeDeleteBoardModal() {
    closeModal('deleteBoardModal', () => {
        currentBoardAction = { boardId: null, boardName: '' };
    });
}

// Helper functions for board management
function findBoardById(boardId, boards) {
    for (const board of boards) {
        if (board.id === boardId) return board;
        if (board.sub_boards && board.sub_boards.length > 0) {
            const found = findBoardById(boardId, board.sub_boards);
            if (found) return found;
        }
    }
    return null;
}

function getAllSubBoardIds(boardId, boards) {
    const board = findBoardById(boardId, boards);
    if (!board || !board.sub_boards) return [];

    let ids = [];
    for (const subBoard of board.sub_boards) {
        ids.push(subBoard.id);
        ids = ids.concat(getAllSubBoardIds(subBoard.id, boards));
    }
    return ids;
}

// Board Context Menu
function showBoardContextMenu(boardId, x, y) {
    const contextMenu = document.getElementById('boardContextMenu');
    if (!contextMenu) return;

    currentBoardAction.boardId = boardId;

    contextMenu.style.left = x + 'px';
    contextMenu.style.top = y + 'px';
    contextMenu.style.display = 'block';

    // Close context menu when clicking anywhere else
    const closeContextMenu = (e) => {
        if (!contextMenu.contains(e.target)) {
            hideBoardContextMenu();
            document.removeEventListener('click', closeContextMenu);
        }
    };

    // Delay adding the listener to avoid immediate trigger
    setTimeout(() => {
        document.addEventListener('click', closeContextMenu);
    }, 10);
}

function hideBoardContextMenu() {
    const contextMenu = document.getElementById('boardContextMenu');
    if (contextMenu) {
        contextMenu.style.display = 'none';
    }
}

// ============ Event Listeners ============

function attachEventListeners() {
    // Header buttons
    const scanBtn = document.getElementById('scanBtn');
    const scanBtnEmpty = document.getElementById('scanBtnEmpty');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const uploadBtn = document.getElementById('uploadBtn');
    const selectBtn = document.getElementById('selectBtn');

    if (scanBtn) scanBtn.addEventListener('click', scanDirectory);
    if (scanBtnEmpty) scanBtnEmpty.addEventListener('click', scanDirectory);
    if (analyzeBtn) analyzeBtn.addEventListener('click', () => batchAnalyze(10));
    if (uploadBtn) uploadBtn.addEventListener('click', openUploadModal);
    if (selectBtn) selectBtn.addEventListener('click', toggleSelectionMode);

    // Batch operations bar buttons
    const selectAllBtn = document.getElementById('selectAllBtn');
    const deselectAllBtn = document.getElementById('deselectAllBtn');
    const batchAnalyzeBtn = document.getElementById('batchAnalyzeBtn');
    const batchTagBtn = document.getElementById('batchTagBtn');
    const batchNameBtn = document.getElementById('batchNameBtn');
    const batchAddToBoardBtn = document.getElementById('batchAddToBoardBtn');
    const closeBatchBtn = document.getElementById('closeBatchBtn');

    if (selectAllBtn) selectAllBtn.addEventListener('click', selectAllImages);
    if (deselectAllBtn) deselectAllBtn.addEventListener('click', deselectAllImages);
    if (batchAnalyzeBtn) batchAnalyzeBtn.addEventListener('click', batchAnalyzeImages);
    if (batchTagBtn) batchTagBtn.addEventListener('click', batchTagImages);
    if (batchNameBtn) batchNameBtn.addEventListener('click', batchNameImages);
    if (batchAddToBoardBtn) batchAddToBoardBtn.addEventListener('click', batchAddImagesToBoard);
    if (closeBatchBtn) closeBatchBtn.addEventListener('click', () => {
        if (state.selectionMode) {
            toggleSelectionMode();
        }
    });

    // Search with debouncing
    const searchInput = document.getElementById('searchInput');
    const searchBtn = document.getElementById('searchBtn');
    let searchTimeout = null;
    
    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                searchImages(e.target.value);
            }, CONFIG.SEARCH_DEBOUNCE_MS);
        });
    }
    
    if (searchBtn) {
        searchBtn.addEventListener('click', () => {
            searchImages(searchInput.value);
        });
    }
    
    // View navigation
    document.querySelectorAll('[data-view]').forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const view = item.dataset.view;
            switchView(view);
        });
    });
    
    // ✅ Image Grid - Event delegation for image cards
    const imageGrid = document.getElementById('imageGrid');
    if (imageGrid) {
        imageGrid.addEventListener('click', (e) => {
            // Check for checkbox clicks FIRST
            const checkbox = e.target.closest('.image-card-checkbox');
            if (checkbox) {
                e.stopPropagation();
                e.preventDefault();
                const imageId = parseInt(checkbox.dataset.id);
                toggleImageSelection(imageId);
                return;
            }

            // Check for tag clicks (before card clicks)
            // This prevents opening modal when clicking on tags
            const tag = e.target.closest('.tag[data-tag]');
            if (tag) {
                e.stopPropagation();
                e.preventDefault();
                const tagValue = tag.dataset.tag;
                searchByTag(tagValue);
                return; // Stop here - don't check for card
            }

            // Check for card clicks (open modal)
            // Don't open modal if in selection mode
            if (!state.selectionMode) {
                const card = e.target.closest('.image-card');
                if (card) {
                    const imageId = parseInt(card.dataset.id);
                    const image = state.images.find(img => img.id === imageId);
                    if (image) {
                        openImageModal(image);
                    }
                }
            }
        });
    }
    
    // ✅ Boards List - Event delegation with single/double click
    const boardsList = document.getElementById('boardsList');
    if (boardsList) {
        let clickTimer = null;
        let clickPrevent = false;

        boardsList.addEventListener('click', (e) => {
            const boardPill = e.target.closest('.board-pill[data-board-id]');
            if (!boardPill) return;

            e.preventDefault();
            const boardId = parseInt(boardPill.dataset.boardId);
            const hasChildren = boardPill.dataset.hasChildren === 'true';

            // Clear existing timer
            clearTimeout(clickTimer);

            // Prevent single click if double click is detected
            if (clickPrevent) {
                clickPrevent = false;
                return;
            }

            // Set timer for single click
            clickTimer = setTimeout(() => {
                // Single click - expand/collapse if has children
                if (hasChildren) {
                    const subPills = boardsList.querySelector(`.board-sub-pills[data-parent-id="${boardId}"]`);
                    const expandIcon = boardPill.querySelector('.board-pill-expand');

                    if (subPills) {
                        const isExpanded = subPills.style.display !== 'none';
                        subPills.style.display = isExpanded ? 'none' : 'flex';

                        if (expandIcon) {
                            expandIcon.textContent = isExpanded ? '▼' : '▲';
                        }
                    }
                } else {
                    // No children - open gallery
                    switchView('board', boardId);
                }
            }, 250);
        });

        boardsList.addEventListener('dblclick', (e) => {
            const boardPill = e.target.closest('.board-pill[data-board-id]');
            if (!boardPill) return;

            e.preventDefault();
            clearTimeout(clickTimer);
            clickPrevent = true;

            const boardId = parseInt(boardPill.dataset.boardId);

            // Double click - always open gallery
            switchView('board', boardId);

            // Reset prevent flag
            setTimeout(() => { clickPrevent = false; }, 300);
        });

        // Right-click context menu on boards
        boardsList.addEventListener('contextmenu', (e) => {
            const boardPill = e.target.closest('.board-pill[data-board-id]');
            if (boardPill) {
                e.preventDefault();
                const boardId = parseInt(boardPill.dataset.boardId);
                showBoardContextMenu(boardId, e.pageX, e.pageY);
            }
        });
    }

    // Board Context Menu
    const boardContextMenu = document.getElementById('boardContextMenu');
    if (boardContextMenu) {
        boardContextMenu.addEventListener('click', (e) => {
            const menuItem = e.target.closest('.context-menu-item');
            if (menuItem) {
                const action = menuItem.dataset.action;
                const boardId = currentBoardAction.boardId;

                hideBoardContextMenu();

                if (action === 'rename') {
                    openRenameBoardModal(boardId);
                } else if (action === 'merge') {
                    openMergeBoardModal(boardId);
                } else if (action === 'delete') {
                    openDeleteBoardModal(boardId);
                }
            }
        });
    }

    // Rename Board Modal
    const renameBoardClose = document.getElementById('renameBoardClose');
    const cancelRenameBtn = document.getElementById('cancelRenameBtn');
    const renameBoardForm = document.getElementById('renameBoardForm');

    if (renameBoardClose) renameBoardClose.addEventListener('click', closeRenameBoardModal);
    if (cancelRenameBtn) cancelRenameBtn.addEventListener('click', closeRenameBoardModal);

    if (renameBoardForm) {
        renameBoardForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            const newName = document.getElementById('renameBoardName').value.trim();
            const newDescription = document.getElementById('renameBoardDescription').value.trim();

            if (newName && currentBoardAction.boardId) {
                const success = await renameBoard(currentBoardAction.boardId, newName, newDescription);
                if (success) {
                    closeRenameBoardModal();
                }
            }
        });
    }

    // Merge Board Modal
    const mergeBoardClose = document.getElementById('mergeBoardClose');
    const cancelMergeBtn = document.getElementById('cancelMergeBtn');
    const mergeBoardForm = document.getElementById('mergeBoardForm');

    if (mergeBoardClose) mergeBoardClose.addEventListener('click', closeMergeBoardModal);
    if (cancelMergeBtn) cancelMergeBtn.addEventListener('click', closeMergeBoardModal);

    if (mergeBoardForm) {
        mergeBoardForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            const targetBoardId = parseInt(document.getElementById('mergeTargetBoard').value);
            const deleteSource = document.getElementById('mergeDeleteSource').checked;

            if (targetBoardId && currentBoardAction.boardId) {
                const success = await mergeBoards(currentBoardAction.boardId, targetBoardId, deleteSource);
                if (success) {
                    closeMergeBoardModal();
                }
            } else {
                showToast('Please select a target board', 'error');
            }
        });
    }

    // Delete Board Modal
    const deleteBoardClose = document.getElementById('deleteBoardClose');
    const cancelDeleteBtn = document.getElementById('cancelDeleteBtn');
    const deleteBoardForm = document.getElementById('deleteBoardForm');

    if (deleteBoardClose) deleteBoardClose.addEventListener('click', closeDeleteBoardModal);
    if (cancelDeleteBtn) cancelDeleteBtn.addEventListener('click', closeDeleteBoardModal);

    if (deleteBoardForm) {
        deleteBoardForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            const deleteSubBoards = document.getElementById('deleteSubBoards').checked;

            if (currentBoardAction.boardId) {
                const success = await deleteBoard(currentBoardAction.boardId, deleteSubBoards);
                if (success) {
                    closeDeleteBoardModal();
                }
            }
        });
    }
    
    // Image Modal
    const modalClose = document.getElementById('modalClose');
    const modalOverlay = document.getElementById('modalOverlay');
    if (modalClose) modalClose.addEventListener('click', closeImageModal);
    if (modalOverlay) modalOverlay.addEventListener('click', closeImageModal);

    // Edit Image Modal
    const editImageClose = document.getElementById('editImageClose');
    const cancelEditImageBtn = document.getElementById('cancelEditImageBtn');
    const editImageForm = document.getElementById('editImageForm');
    const addTagBtn = document.getElementById('addTagBtn');
    const editImageNewTag = document.getElementById('editImageNewTag');

    if (editImageClose) editImageClose.addEventListener('click', closeEditImageModal);
    if (cancelEditImageBtn) cancelEditImageBtn.addEventListener('click', closeEditImageModal);

    if (editImageForm) {
        editImageForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            await saveImageEdit();
        });
    }

    if (addTagBtn) {
        addTagBtn.addEventListener('click', () => {
            const tagInput = document.getElementById('editImageNewTag');
            addEditTag(tagInput.value);
        });
    }

    if (editImageNewTag) {
        editImageNewTag.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                addEditTag(editImageNewTag.value);
            }
        });
    }

    // Create Board Modal
    const createBoardBtn = document.getElementById('createBoardBtn');
    const createBoardClose = document.getElementById('createBoardClose');
    const cancelBoardBtn = document.getElementById('cancelBoardBtn');
    const createBoardForm = document.getElementById('createBoardForm');
    
    if (createBoardBtn) createBoardBtn.addEventListener('click', openCreateBoardModal);
    if (createBoardClose) createBoardClose.addEventListener('click', closeCreateBoardModal);
    if (cancelBoardBtn) cancelBoardBtn.addEventListener('click', closeCreateBoardModal);
    
    if (createBoardForm) {
        createBoardForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const name = document.getElementById('boardName').value.trim();
            const description = document.getElementById('boardDescription').value.trim();
            const parentId = document.getElementById('boardParent').value;
            
            if (name) {
                try {
                    await createBoard(name, description, parentId || null);
                    closeCreateBoardModal();
                } catch (error) {
                    // Error already shown
                }
            }
        });
    }
    
    // Add to Board Modal
    const addToBoardClose = document.getElementById('addToBoardClose');
    const cancelAddToBoardBtn = document.getElementById('cancelAddToBoardBtn');
    const saveAddToBoardBtn = document.getElementById('saveAddToBoardBtn');
    
    if (addToBoardClose) addToBoardClose.addEventListener('click', closeAddToBoardModal);
    if (cancelAddToBoardBtn) cancelAddToBoardBtn.addEventListener('click', closeAddToBoardModal);
    
    if (saveAddToBoardBtn) {
        saveAddToBoardBtn.addEventListener('click', async () => {
            const checkboxes = document.querySelectorAll('#boardSelection input[type="checkbox"]');
            const selectedBoards = Array.from(checkboxes)
                .filter(cb => cb.checked)
                .map(cb => parseInt(cb.value));

            if (selectedBoards.length === 0) {
                showToast('Please select at least one board', 'error');
                return;
            }

            // Batch mode: add selected images to selected boards
            if (state.isBatchBoardAdd && state.pendingBatchBoardImageIds) {
                const imageIds = state.pendingBatchBoardImageIds;
                const total = imageIds.length;
                const boardsCount = selectedBoards.length;

                showToast(`Adding ${total} images to ${boardsCount} boards...`, 'info');

                for (const imageId of imageIds) {
                    for (const boardId of selectedBoards) {
                        await addImageToBoard(boardId, imageId);
                    }
                }

                showToast(`Successfully added ${total} images to ${boardsCount} boards! 📌`, 'success');

                // Clear batch state
                state.pendingBatchBoardImageIds = null;
                deselectAllImages();
                if (state.selectionMode) {
                    toggleSelectionMode();
                }
            }
            // Single image mode
            else if (state.currentImage) {
                const currentBoards = (state.currentImage.boards || []).map(b => b.id);

                const toAdd = selectedBoards.filter(id => !currentBoards.includes(id));
                const toRemove = currentBoards.filter(id => !selectedBoards.includes(id));

                for (const boardId of toAdd) {
                    await addImageToBoard(boardId, state.currentImage.id);
                }

                for (const boardId of toRemove) {
                    await removeImageFromBoard(boardId, state.currentImage.id);
                }

                showToast('Board assignments updated!', 'success');

                const updated = await getImageDetails(state.currentImage.id);
                if (updated) {
                    state.currentImage = updated;
                    updateModal();
                }
            }

            closeAddToBoardModal();
        });
    }

    // AI Style Modal
    const aiStyleClose = document.getElementById('aiStyleClose');
    const aiStyleOverlay = document.getElementById('aiStyleOverlay');
    const cancelAIStyleBtn = document.getElementById('cancelAIStyleBtn');
    const analyzeWithStyleBtn = document.getElementById('analyzeWithStyleBtn');
    
    if (aiStyleClose) aiStyleClose.addEventListener('click', closeAIStyleModal);
    if (aiStyleOverlay) aiStyleOverlay.addEventListener('click', closeAIStyleModal);
    if (cancelAIStyleBtn) cancelAIStyleBtn.addEventListener('click', closeAIStyleModal);

    if (analyzeWithStyleBtn) {
        analyzeWithStyleBtn.addEventListener('click', async () => {
            const selectedStyle = state.selectedStyle || 'classic';
            let customPrompt = null;

            if (selectedStyle === 'custom') {
                customPrompt = document.getElementById('customPrompt').value.trim();
                if (!customPrompt) {
                    showToast('Please enter a custom prompt', 'error');
                    return;
                }
            }

            // ✅ CRITICAL FIX: Capture imageId BEFORE closing modal
            const imageId = state.pendingAnalyzeImageId;
            const isBatch = state.isBatchAnalyze;

            closeAIStyleModal();

            // Check if this is batch mode
            if (isBatch) {
                await performBatchAnalyze(selectedStyle, customPrompt);
            } else {
                if (!imageId) return;
                await analyzeImage(imageId, selectedStyle, customPrompt);
            }
        });
    }

    // Upload Modal
    const uploadModalClose = document.getElementById('uploadModalClose');
    const selectFilesBtn = document.getElementById('selectFilesBtn');
    const fileInput = document.getElementById('fileInput');
    const cancelUploadBtn = document.getElementById('cancelUploadBtn');
    const startUploadBtn = document.getElementById('startUploadBtn');
    
    if (uploadModalClose) uploadModalClose.addEventListener('click', closeUploadModal);
    if (selectFilesBtn) {
        selectFilesBtn.addEventListener('click', () => {
            if (fileInput) fileInput.click();
        });
    }
    
    if (fileInput) fileInput.addEventListener('change', handleFileSelect);
    if (cancelUploadBtn) cancelUploadBtn.addEventListener('click', closeUploadModal);
    if (startUploadBtn) startUploadBtn.addEventListener('click', uploadFiles);
    
    // Drag and drop for upload area
    const uploadArea = document.getElementById('uploadArea');
    
    if (uploadArea) {
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');

            const files = Array.from(e.dataTransfer.files).filter(file =>
                file.type.startsWith('image/') || file.type.startsWith('video/')
            );

            if (files.length > 0) {
                state.uploadFiles = files;
                showUploadPreview();
            }
        });
        
        uploadArea.addEventListener('click', () => {
            if (fileInput) fileInput.click();
        });
    }
    
    // Settings Modal
    const settingsBtn = document.getElementById('settingsBtn');
    const settingsClose = document.getElementById('settingsClose');
    const settingsOverlay = document.getElementById('settingsOverlay');
    const botConfigForm = document.getElementById('botConfigForm');
    const startBotBtn = document.getElementById('startBotBtn');
    const stopBotBtn = document.getElementById('stopBotBtn');
    const testBotBtn = document.getElementById('testBotBtn');
    const viewLogsBtn = document.getElementById('viewLogsBtn');
    const refreshLogsBtn = document.getElementById('refreshLogsBtn');

    if (settingsBtn) settingsBtn.addEventListener('click', openSettingsModal);
    if (settingsClose) settingsClose.addEventListener('click', closeSettingsModal);
    if (settingsOverlay) settingsOverlay.addEventListener('click', closeSettingsModal);

    if (startBotBtn) startBotBtn.addEventListener('click', startBot);
    if (stopBotBtn) stopBotBtn.addEventListener('click', stopBot);
    if (testBotBtn) testBotBtn.addEventListener('click', testBot);
    if (viewLogsBtn) viewLogsBtn.addEventListener('click', viewBotLogs);
    if (refreshLogsBtn) refreshLogsBtn.addEventListener('click', loadBotLogs);

    if (botConfigForm) {
        botConfigForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            await saveBotConfig();
        });
    }

    // Email configuration event listeners
    const emailConfigForm = document.getElementById('emailConfigForm');
    const emailEnabledCheckbox = document.getElementById('emailEnabled');
    const testEmailBtn = document.getElementById('testEmailBtn');

    if (emailEnabledCheckbox) {
        emailEnabledCheckbox.addEventListener('change', toggleEmailConfigDetails);
    }

    if (emailConfigForm) {
        emailConfigForm.addEventListener('submit', saveEmailConfig);
    }

    if (testEmailBtn) {
        testEmailBtn.addEventListener('click', testEmailConnection);
    }

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            // Close context menu first
            if (document.getElementById('boardContextMenu').style.display === 'block') {
                hideBoardContextMenu();
            }
            else if (document.getElementById('settingsModal').style.display === 'block') {
                closeSettingsModal();
            }
            else if (document.getElementById('renameBoardModal').style.display === 'block') {
                closeRenameBoardModal();
            }
            else if (document.getElementById('mergeBoardModal').style.display === 'block') {
                closeMergeBoardModal();
            }
            else if (document.getElementById('deleteBoardModal').style.display === 'block') {
                closeDeleteBoardModal();
            }
            else if (document.getElementById('aiStyleModal').style.display === 'block') {
                closeAIStyleModal();
            }
            else if (document.getElementById('imageModal').style.display === 'block') {
                closeImageModal();
            }
            else if (document.getElementById('createBoardModal').style.display === 'block') {
                closeCreateBoardModal();
            }
            else if (document.getElementById('addToBoardModal').style.display === 'block') {
                closeAddToBoardModal();
            }
            else if (document.getElementById('uploadModal').style.display === 'block') {
                closeUploadModal();
            }
        }

        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            if (searchInput) searchInput.focus();
        }
    });

    // Sorting Controls
    const sortImagesBtn = document.getElementById('sortImagesBtn');
    const imageSortMenu = document.getElementById('imageSortMenu');

    // Image sorting
    if (sortImagesBtn && imageSortMenu) {
        sortImagesBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            imageSortMenu.style.display = imageSortMenu.style.display === 'none' ? 'block' : 'none';
        });

        imageSortMenu.addEventListener('click', (e) => {
            const option = e.target.closest('.sort-option');
            if (option) {
                state.imageSort = option.dataset.sort;

                // Update active state
                imageSortMenu.querySelectorAll('.sort-option').forEach(opt => opt.classList.remove('active'));
                option.classList.add('active');

                renderImages();
                imageSortMenu.style.display = 'none';
                showToast('Images sorted!', 'success');
            }
        });
    }

    // Close sort menus on outside click
    document.addEventListener('click', () => {
        if (imageSortMenu) imageSortMenu.style.display = 'none';
    });

    // Duplicate Finder
    const findDuplicatesBtn = document.getElementById('findDuplicatesBtn');
    const duplicatesModal = document.getElementById('duplicatesModal');
    const duplicatesClose = document.getElementById('duplicatesClose');
    const duplicatesOverlay = document.getElementById('duplicatesOverlay');

    if (findDuplicatesBtn) {
        findDuplicatesBtn.addEventListener('click', findDuplicates);
    }

    if (duplicatesClose) {
        duplicatesClose.addEventListener('click', closeDuplicatesModal);
    }

    if (duplicatesOverlay) {
        duplicatesOverlay.addEventListener('click', closeDuplicatesModal);
    }
}

// ============ Helper Functions ============

function showLoading() {
    const loadingState = document.getElementById('loadingState');
    const imageGrid = document.getElementById('imageGrid');
    const emptyState = document.getElementById('emptyState');
    
    if (loadingState) loadingState.style.display = 'flex';
    if (imageGrid) imageGrid.style.display = 'none';
    if (emptyState) emptyState.style.display = 'none';
}

function hideLoading() {
    const loadingState = document.getElementById('loadingState');
    if (loadingState) loadingState.style.display = 'none';
}

function showToast(message, type = 'success') {
    const container = document.getElementById('toastContainer');
    if (!container) return;
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => {
            if (toast.parentNode) {
                container.removeChild(toast);
            }
        }, 300);
    }, CONFIG.TOAST_DURATION_MS);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// ============ Upload Functions ============

function openUploadModal() {
    const modal = document.getElementById('uploadModal');
    const uploadPreview = document.getElementById('uploadPreview');
    const uploadArea = document.getElementById('uploadArea');
    
    if (modal) modal.style.display = 'block';
    if (uploadPreview) uploadPreview.style.display = 'none';
    if (uploadArea) uploadArea.style.display = 'block';
    state.uploadFiles = [];
}

function closeUploadModal() {
    closeModal('uploadModal', () => {
        state.uploadFiles = [];
    });
}

function handleFileSelect(e) {
    const files = Array.from(e.target.files).filter(file =>
        file.type.startsWith('image/') || file.type.startsWith('video/')
    );

    if (files.length > 0) {
        state.uploadFiles = files;
        showUploadPreview();
    }
}

function showUploadPreview() {
    const preview = document.getElementById('uploadPreview');
    const fileList = document.getElementById('uploadFileList');
    const uploadArea = document.getElementById('uploadArea');
    
    if (uploadArea) uploadArea.style.display = 'none';
    if (preview) preview.style.display = 'block';
    
    if (fileList) {
        fileList.innerHTML = state.uploadFiles.map((file, index) => `
            <div class="upload-file-item">
                <span class="upload-file-name">${escapeHtml(file.name)} (${formatFileSize(file.size)})</span>
                <button class="upload-file-remove" onclick="removeUploadFile(${index})">×</button>
            </div>
        `).join('');
    }
}

function removeUploadFile(index) {
    state.uploadFiles.splice(index, 1);
    
    const uploadArea = document.getElementById('uploadArea');
    const uploadPreview = document.getElementById('uploadPreview');
    
    if (state.uploadFiles.length === 0) {
        if (uploadArea) uploadArea.style.display = 'block';
        if (uploadPreview) uploadPreview.style.display = 'none';
    } else {
        showUploadPreview();
    }
}

async function uploadFiles() {
    if (state.uploadFiles.length === 0) return;
    
    if (state.isUploading) {
        showToast('Upload already in progress', 'warning');
        return;
    }

    state.isUploading = true;
    const uploadBtn = document.getElementById('startUploadBtn');
    const originalText = uploadBtn ? uploadBtn.textContent : '';
    
    if (uploadBtn) {
        uploadBtn.disabled = true;
        uploadBtn.textContent = 'Uploading...';
    }

    let uploaded = 0;
    let failed = 0;
    const uploadedImageIds = [];

    for (const file of state.uploadFiles) {
        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const data = await response.json();
                if (data.image_id) {
                    uploaded++;
                    uploadedImageIds.push(data.image_id);
                } else {
                    console.error('No image_id in response:', data);
                    failed++;
                }
            } else {
                failed++;
            }
        } catch (error) {
            console.error('Upload error:', error);
            failed++;
        }
    }

    const uploadMessage = `Uploaded ${uploaded} image${uploaded !== 1 ? 's' : ''}${failed > 0 ? `, ${failed} failed` : ''}`;
    showToast(uploadMessage, uploaded > 0 ? 'success' : 'error');

    closeUploadModal();

    await loadImages();
    await updateStats();
    renderImages();

    if (uploadedImageIds.length > 0) {
        showToast(`Starting AI analysis for ${uploadedImageIds.length} image${uploadedImageIds.length !== 1 ? 's' : ''}...`, 'warning');

        let analyzed = 0;
        let analyzeFailed = 0;

        for (let i = 0; i < uploadedImageIds.length; i++) {
            const imageId = uploadedImageIds[i];
            
            try {
                console.log(`Analyzing image ${i + 1}/${uploadedImageIds.length}...`);
                await analyzeImage(imageId, 'classic', null);
                analyzed++;
            } catch (error) {
                console.error(`Failed to analyze image ${imageId}:`, error);
                analyzeFailed++;
            }
        }

        await loadImages();
        await updateStats();
        renderImages();

        const messages = [];
        if (analyzed > 0) {
            messages.push(`✨ Analyzed ${analyzed} image${analyzed !== 1 ? 's' : ''}`);
        }
        if (analyzeFailed > 0) {
            messages.push(`⚠️ ${analyzeFailed} failed`);
        }
        
        if (messages.length > 0) {
            showToast(messages.join(', '), analyzed > 0 ? 'success' : 'error');
        }
    }
    
    state.isUploading = false;
    if (uploadBtn) {
        uploadBtn.disabled = false;
        uploadBtn.textContent = originalText;
    }
}

// ============ Telegram Bot Management ============

async function loadBotStatus() {
    try {
        const data = await apiCall('/telegram/status');
        updateBotStatusUI(data);
        return data;
    } catch (error) {
        console.error('Failed to load bot status:', error);
        return null;
    }
}

function updateBotStatusUI(status) {
    const statusDot = document.getElementById('botStatusDot');
    const statusText = document.getElementById('botStatusText');
    const startBtn = document.getElementById('startBotBtn');
    const stopBtn = document.getElementById('stopBotBtn');

    if (status.running) {
        statusDot.textContent = '🟢';
        statusText.textContent = `Bot Running (PID: ${status.pid})`;
        startBtn.disabled = true;
        stopBtn.disabled = false;
    } else if (status.configured) {
        statusDot.textContent = '🟡';
        statusText.textContent = 'Bot Configured (Offline)';
        startBtn.disabled = false;
        stopBtn.disabled = true;
    } else {
        statusDot.textContent = '🔴';
        statusText.textContent = 'Bot Not Configured';
        startBtn.disabled = true;
        stopBtn.disabled = true;
    }
}

async function startBot() {
    const startBtn = document.getElementById('startBotBtn');
    const originalText = startBtn.textContent;

    startBtn.disabled = true;
    startBtn.textContent = '⏳ Starting...';

    try {
        const data = await apiCall('/telegram/start', { method: 'POST' });

        if (data.success) {
            showToast(data.message, 'success');
            await loadBotStatus();
        } else {
            showToast(data.message, 'error');
        }
    } catch (error) {
        showToast('Failed to start bot: ' + error.message, 'error');
    } finally {
        startBtn.disabled = false;
        startBtn.textContent = originalText;
    }
}

async function stopBot() {
    const stopBtn = document.getElementById('stopBotBtn');
    const originalText = stopBtn.textContent;

    stopBtn.disabled = true;
    stopBtn.textContent = '⏳ Stopping...';

    try {
        const data = await apiCall('/telegram/stop', { method: 'POST' });

        if (data.success) {
            showToast(data.message, 'success');
            await loadBotStatus();
        } else {
            showToast(data.message, 'error');
        }
    } catch (error) {
        showToast('Failed to stop bot: ' + error.message, 'error');
    } finally {
        stopBtn.disabled = false;
        stopBtn.textContent = originalText;
    }
}

async function testBot() {
    const status = await loadBotStatus();

    if (status && status.running) {
        showToast('Bot is running! Try sending a photo to your bot on Telegram.', 'success');
    } else if (status && status.configured) {
        showToast('Bot is configured but not running. Click Start to begin.', 'warning');
    } else {
        showToast('Bot is not configured. Please add your bot token and save.', 'error');
    }
}

async function viewBotLogs() {
    const logsSection = document.getElementById('botLogsSection');
    const viewLogsBtn = document.getElementById('viewLogsBtn');

    // Toggle logs section visibility
    if (logsSection.style.display === 'none') {
        logsSection.style.display = 'block';
        viewLogsBtn.textContent = '📄 Hide Logs';
        await loadBotLogs();
    } else {
        logsSection.style.display = 'none';
        viewLogsBtn.textContent = '📄 View Logs';
    }
}

async function loadBotLogs() {
    const logsEl = document.getElementById('botLogs');

    try {
        logsEl.textContent = 'Loading logs...';

        const data = await apiCall('/telegram/logs?lines=200');

        if (data.logs) {
            logsEl.textContent = data.logs || 'No logs available';

            // Auto-scroll to bottom
            logsEl.scrollTop = logsEl.scrollHeight;
        } else {
            logsEl.textContent = data.message || 'No logs available';
        }
    } catch (error) {
        logsEl.textContent = 'Error loading logs: ' + error.message;
        console.error('Failed to load bot logs:', error);
    }
}

async function loadBotConfig() {
    try {
        const data = await apiCall('/telegram/config');
        const config = data.config;

        if (config.TELEGRAM_BOT_TOKEN) {
            document.getElementById('botToken').value = config.TELEGRAM_BOT_TOKEN;
        }

        if (config.AUTO_ANALYZE !== undefined) {
            document.getElementById('autoAnalyze').checked = config.AUTO_ANALYZE === 'true';
        }

        if (config.AI_STYLE) {
            document.getElementById('aiStyle').value = config.AI_STYLE;
        }
    } catch (error) {
        console.error('Failed to load bot config:', error);
    }
}

async function saveBotConfig() {
    const botToken = document.getElementById('botToken').value.trim();
    const autoAnalyze = document.getElementById('autoAnalyze').checked ? 'true' : 'false';
    const aiStyle = document.getElementById('aiStyle').value;

    if (!botToken) {
        showToast('Please enter a bot token', 'error');
        return;
    }

    try {
        const data = await apiCall('/telegram/config', {
            method: 'POST',
            body: JSON.stringify({
                bot_token: botToken,
                auto_analyze: autoAnalyze,
                ai_style: aiStyle
            })
        });

        if (data.success) {
            showToast('Configuration saved successfully!', 'success');
            await loadBotStatus();
        } else {
            showToast('Failed to save configuration', 'error');
        }
    } catch (error) {
        showToast('Failed to save configuration: ' + error.message, 'error');
    }
}

function openSettingsModal() {
    const modal = document.getElementById('settingsModal');
    modal.style.display = 'block';

    loadBotStatus();
    loadBotConfig();
    loadEmailConfig();
}

function closeSettingsModal() {
    closeModal('settingsModal');
}

// ============ Email Configuration ============

async function loadEmailConfig() {
    try {
        const response = await fetch('/api/settings/email');
        const data = await response.json();

        if (data.success && data.config) {
            const config = data.config;

            document.getElementById('emailEnabled').checked = config.enabled || false;
            document.getElementById('emailSmtpHost').value = config.smtp_host || 'smtp.gmail.com';
            document.getElementById('emailSmtpPort').value = config.smtp_port || 587;
            document.getElementById('emailUseTls').checked = config.use_tls !== false;
            document.getElementById('emailUsername').value = config.username || '';
            document.getElementById('emailPassword').value = config.password || '';
            document.getElementById('emailFromEmail').value = config.from_email || '';
            document.getElementById('emailFromName').value = config.from_name || 'AI Gallery';

            toggleEmailConfigDetails();
        }
    } catch (error) {
        console.error('Error loading email config:', error);
    }
}

function toggleEmailConfigDetails() {
    const enabled = document.getElementById('emailEnabled').checked;
    const details = document.getElementById('emailConfigDetails');
    details.style.display = enabled ? 'block' : 'none';
}

async function saveEmailConfig(e) {
    e.preventDefault();

    const config = {
        enabled: document.getElementById('emailEnabled').checked,
        smtp_host: document.getElementById('emailSmtpHost').value,
        smtp_port: parseInt(document.getElementById('emailSmtpPort').value),
        use_tls: document.getElementById('emailUseTls').checked,
        username: document.getElementById('emailUsername').value,
        password: document.getElementById('emailPassword').value,
        from_email: document.getElementById('emailFromEmail').value,
        from_name: document.getElementById('emailFromName').value
    };

    try {
        const response = await fetch('/api/settings/email', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(config)
        });

        const data = await response.json();

        if (data.success) {
            showMessage('Email settings saved successfully!', 'success');
        } else {
            showMessage('Error saving email settings: ' + (data.error || 'Unknown error'), 'error');
        }
    } catch (error) {
        console.error('Error saving email config:', error);
        showMessage('Failed to save email settings: ' + error.message, 'error');
    }
}

async function testEmailConnection() {
    try {
        showMessage('Testing email connection...', 'info');

        const response = await fetch('/api/settings/email/test', {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            showMessage('✅ Email connection successful!', 'success');
        } else {
            showMessage('❌ Connection failed: ' + (data.error || 'Unknown error'), 'error');
        }
    } catch (error) {
        console.error('Error testing email:', error);
        showMessage('Failed to test connection: ' + error.message, 'error');
    }
}

// ============ Batch Selection Mode ============

function toggleSelectionMode() {
    state.selectionMode = !state.selectionMode;

    const imageGrid = document.getElementById('imageGrid');
    const selectBtn = document.getElementById('selectBtn');

    if (state.selectionMode) {
        imageGrid.classList.add('selection-mode');
        selectBtn.textContent = '✕ Cancel';
        selectBtn.classList.remove('btn-secondary');
        selectBtn.classList.add('btn-danger');
    } else {
        imageGrid.classList.remove('selection-mode');
        selectBtn.textContent = '✓ Select';
        selectBtn.classList.remove('btn-danger');
        selectBtn.classList.add('btn-secondary');

        // Clear selection when exiting selection mode
        deselectAllImages();
    }
}

function toggleImageSelection(imageId) {
    if (state.selectedImages.has(imageId)) {
        state.selectedImages.delete(imageId);
    } else {
        state.selectedImages.add(imageId);
    }

    updateSelectionUI();
}

function selectAllImages() {
    // Select all currently visible images
    state.images.forEach(img => {
        state.selectedImages.add(img.id);
    });

    updateSelectionUI();
}

function deselectAllImages() {
    state.selectedImages.clear();
    updateSelectionUI();
}

function updateSelectionUI() {
    const count = state.selectedImages.size;
    const batchBar = document.getElementById('batchOperationsBar');
    const selectedCountEl = document.getElementById('selectedCount');

    // Update count
    if (selectedCountEl) {
        selectedCountEl.textContent = count;
    }

    // Show/hide batch operations bar
    if (count > 0) {
        batchBar.style.display = 'flex';
    } else {
        batchBar.style.display = 'none';
    }

    // Update checkboxes in the grid
    state.images.forEach(img => {
        const card = document.querySelector(`.image-card[data-id="${img.id}"]`);
        const checkbox = card?.querySelector('.image-card-checkbox');

        if (card && checkbox) {
            if (state.selectedImages.has(img.id)) {
                card.classList.add('selected');
                checkbox.classList.add('checked');
            } else {
                card.classList.remove('selected');
                checkbox.classList.remove('checked');
            }
        }
    });
}

// Batch Operations
async function batchAnalyzeImages() {
    if (state.selectedImages.size === 0) {
        showToast('No images selected', 'error');
        return;
    }

    if (state.isAnalyzing) {
        showToast('Analysis already in progress...', 'warning');
        return;
    }

    // Open AI style selector
    state.pendingBatchOperation = 'analyze';
    openAIStyleModal(true); // true indicates batch mode
}

async function performBatchAnalyze(style, customPrompt = null) {
    const selectedIds = Array.from(state.selectedImages);
    const total = selectedIds.length;

    state.isAnalyzing = true;
    showToast(`Analyzing ${total} images...`, 'info');

    let completed = 0;
    let failed = 0;

    for (const imageId of selectedIds) {
        try {
            const requestBody = { style };
            if (customPrompt) {
                requestBody.custom_prompt = customPrompt;
            }

            await apiCall(`/images/${imageId}/analyze`, {
                method: 'POST',
                body: JSON.stringify(requestBody)
            });

            completed++;

            // Update toast progress
            showToast(`Analyzed ${completed}/${total} images...`, 'info');
        } catch (error) {
            console.error(`Failed to analyze image ${imageId}:`, error);
            failed++;
        }
    }

    state.isAnalyzing = false;

    // Reload images to get updated data
    await loadImages();
    renderImages();

    if (failed > 0) {
        showToast(`Analysis complete: ${completed} succeeded, ${failed} failed`, 'warning');
    } else {
        showToast(`Successfully analyzed ${completed} images! ✨`, 'success');
    }

    // Clear selection and exit selection mode
    deselectAllImages();
    if (state.selectionMode) {
        toggleSelectionMode();
    }
}

async function batchTagImages() {
    if (state.selectedImages.size === 0) {
        showToast('No images selected', 'error');
        return;
    }

    if (state.isAnalyzing) {
        showToast('Analysis already in progress...', 'warning');
        return;
    }

    const selectedIds = Array.from(state.selectedImages);
    const total = selectedIds.length;

    state.isAnalyzing = true;
    showToast(`Generating AI tags for ${total} images...`, 'info');

    let completed = 0;
    let failed = 0;

    for (const imageId of selectedIds) {
        try {
            await apiCall(`/images/${imageId}/analyze`, {
                method: 'POST',
                body: JSON.stringify({ style: 'tags' })
            });

            completed++;
            showToast(`Tagged ${completed}/${total} images...`, 'info');
        } catch (error) {
            console.error(`Failed to tag image ${imageId}:`, error);
            failed++;
        }
    }

    state.isAnalyzing = false;

    await loadImages();
    await loadTags();
    renderImages();
    renderTagCloud();

    if (failed > 0) {
        showToast(`Tagging complete: ${completed} succeeded, ${failed} failed`, 'warning');
    } else {
        showToast(`Successfully tagged ${completed} images! 🏷️`, 'success');
    }

    deselectAllImages();
    if (state.selectionMode) {
        toggleSelectionMode();
    }
}

async function batchNameImages() {
    if (state.selectedImages.size === 0) {
        showToast('No images selected', 'error');
        return;
    }

    if (state.isAnalyzing) {
        showToast('Analysis already in progress...', 'warning');
        return;
    }

    const selectedIds = Array.from(state.selectedImages);
    const total = selectedIds.length;

    state.isAnalyzing = true;
    showToast(`Generating AI names for ${total} images...`, 'info');

    let completed = 0;
    let failed = 0;

    for (const imageId of selectedIds) {
        try {
            // Use a custom prompt focused on generating descriptive names
            await apiCall(`/images/${imageId}/analyze`, {
                method: 'POST',
                body: JSON.stringify({
                    style: 'custom',
                    custom_prompt: 'Generate a short, descriptive name for this image (max 5 words). Only return the name, nothing else.'
                })
            });

            completed++;
            showToast(`Named ${completed}/${total} images...`, 'info');
        } catch (error) {
            console.error(`Failed to name image ${imageId}:`, error);
            failed++;
        }
    }

    state.isAnalyzing = false;

    await loadImages();
    renderImages();

    if (failed > 0) {
        showToast(`Naming complete: ${completed} succeeded, ${failed} failed`, 'warning');
    } else {
        showToast(`Successfully named ${completed} images! ✏️`, 'success');
    }

    deselectAllImages();
    if (state.selectionMode) {
        toggleSelectionMode();
    }
}

async function batchAddImagesToBoard() {
    if (state.selectedImages.size === 0) {
        showToast('No images selected', 'error');
        return;
    }

    // Open the add to board modal with selected images
    const selectedIds = Array.from(state.selectedImages);

    // Store selected IDs for batch operation
    state.pendingBatchBoardImageIds = selectedIds;

    // Open modal (reuse existing add to board modal)
    openAddToBoardModal(selectedIds[0], true); // true indicates batch mode
}

// ============ Duplicate Finder ============

async function findDuplicates() {
    const modal = document.getElementById('duplicatesModal');
    const loading = document.getElementById('duplicatesLoading');
    const content = document.getElementById('duplicatesContent');
    const empty = document.getElementById('duplicatesEmpty');

    // Show modal
    if (modal) modal.style.display = 'block';
    if (loading) loading.style.display = 'block';
    if (content) content.innerHTML = '';
    if (empty) empty.style.display = 'none';

    try {
        // Get all images from current view
        const images = state.images;

        if (images.length === 0) {
            showToast('No images to analyze', 'warning');
            closeDuplicatesModal();
            return;
        }

        // Group images by file size (quick first pass)
        const sizeGroups = new Map();
        images.forEach(img => {
            const size = img.file_size || 0;
            if (!sizeGroups.has(size)) {
                sizeGroups.set(size, []);
            }
            sizeGroups.get(size).push(img);
        });

        // Find potential duplicates (same size)
        const duplicateGroups = [];
        sizeGroups.forEach((group, size) => {
            if (group.length > 1 && size > 0) {
                // Further group by filename similarity
                const nameGroups = new Map();
                group.forEach(img => {
                    // Extract base filename without extension and numbers
                    const baseName = img.filename
                        .replace(/\.[^.]+$/, '') // remove extension
                        .replace(/[_\-\s]*\d+[_\-\s]*$/, '') // remove trailing numbers
                        .toLowerCase();

                    if (!nameGroups.has(baseName)) {
                        nameGroups.set(baseName, []);
                    }
                    nameGroups.get(baseName).push(img);
                });

                nameGroups.forEach(nameGroup => {
                    if (nameGroup.length > 1) {
                        duplicateGroups.push(nameGroup);
                    }
                });
            }
        });

        if (loading) loading.style.display = 'none';

        if (duplicateGroups.length === 0) {
            if (empty) empty.style.display = 'block';
        } else {
            if (content) {
                content.innerHTML = `
                    <p style="margin-bottom: var(--spacing-lg); color: var(--text-secondary);">
                        Found ${duplicateGroups.length} group${duplicateGroups.length > 1 ? 's' : ''} of potential duplicates
                    </p>
                    ${duplicateGroups.map((group, index) => createDuplicateGroup(group, index)).join('')}
                `;
            }
        }
    } catch (error) {
        console.error('Error finding duplicates:', error);
        showToast('Error analyzing duplicates', 'error');
        if (loading) loading.style.display = 'none';
    }
}

function createDuplicateGroup(images, groupIndex) {
    return `
        <div class="duplicate-group" style="margin-bottom: var(--spacing-xl);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: var(--spacing-md);">
                <h3 style="font-size: 16px; color: var(--text-primary);">
                    Group ${groupIndex + 1}
                    <span style="color: var(--text-secondary); font-size: 14px; font-weight: normal;">
                        (${images.length} images, ${formatFileSize(images[0].file_size || 0)})
                    </span>
                </h3>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: var(--spacing-md);">
                ${images.map(img => createDuplicateCard(img)).join('')}
            </div>
        </div>
    `;
}

function createDuplicateCard(image) {
    const isVideo = image.media_type === 'video';

    return `
        <div class="duplicate-card" style="position: relative; cursor: pointer;" onclick="openImageModal(state.images.find(img => img.id === ${image.id}))">
            <div style="position: relative; aspect-ratio: 1; overflow: hidden; border-radius: var(--radius-md); background: var(--bg-tertiary);">
                ${isVideo ?
                    `<img
                        src="/api/images/${image.id}/thumbnail?size=300"
                        alt="${escapeHtml(image.filename)}"
                        style="width: 100%; height: 100%; object-fit: cover;"
                    >
                    <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background: rgba(0,0,0,0.7); border-radius: 50%; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center;">
                        <div style="font-size: 20px;">▶</div>
                    </div>` :
                    `<img
                        src="/api/images/${image.id}/thumbnail?size=300"
                        alt="${escapeHtml(image.filename)}"
                        style="width: 100%; height: 100%; object-fit: cover;"
                    >`
                }
            </div>
            <div style="padding: var(--spacing-sm); background: var(--bg-secondary); border-radius: 0 0 var(--radius-md) var(--radius-md);">
                <div style="font-size: 12px; color: var(--text-primary); font-weight: 600; margin-bottom: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="${escapeHtml(image.filename)}">
                    ${escapeHtml(image.filename)}
                </div>
                <div style="font-size: 11px; color: var(--text-secondary);">
                    ${image.width || 0}×${image.height || 0}
                </div>
            </div>
        </div>
    `;
}

function closeDuplicatesModal() {
    const modal = document.getElementById('duplicatesModal');
    if (modal) modal.style.display = 'none';
}

// ============ EXPORT/IMPORT DATA ============

function openExportImportModal() {
    const modal = document.getElementById('exportImportModal');
    if (modal) modal.style.display = 'flex';
}

function closeExportImportModal() {
    const modal = document.getElementById('exportImportModal');
    if (modal) modal.style.display = 'none';
}

async function exportData(format) {
    const includeImages = document.getElementById('exportIncludeImages')?.checked ?? true;
    const includeBoards = document.getElementById('exportIncludeBoards')?.checked ?? true;

    try {
        showToast(`Exporting data as ${format.toUpperCase()}...`, 'info');

        const params = new URLSearchParams({
            format: format,
            include_images: includeImages,
            include_boards: includeBoards
        });

        const response = await fetch(`/api/export?${params}`);

        if (!response.ok) {
            throw new Error(`Export failed: ${response.statusText}`);
        }

        // Get filename from Content-Disposition header or use default
        const contentDisposition = response.headers.get('Content-Disposition');
        let filename = `gallery_export.${format}`;
        if (contentDisposition) {
            const matches = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/.exec(contentDisposition);
            if (matches != null && matches[1]) {
                filename = matches[1].replace(/['"]/g, '');
            }
        }

        // Get the blob data
        const blob = await response.blob();

        // Create download link
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

        showToast(`Data exported successfully as ${filename}`, 'success');
    } catch (error) {
        console.error('Export error:', error);
        showToast(`Export failed: ${error.message}`, 'error');
    }
}

async function importData() {
    const jsonDataElement = document.getElementById('importJsonData');
    const importBoardsElement = document.getElementById('importBoards');
    const importBoardAssignmentsElement = document.getElementById('importBoardAssignments');
    const updateExistingElement = document.getElementById('importUpdateExisting');

    if (!jsonDataElement) {
        showToast('Import form not found', 'error');
        return;
    }

    const jsonData = jsonDataElement.value.trim();

    if (!jsonData) {
        showToast('Please paste JSON data to import', 'error');
        return;
    }

    const importBoards = importBoardsElement?.checked ?? true;
    const importBoardAssignments = importBoardAssignmentsElement?.checked ?? true;
    const updateExisting = updateExistingElement?.checked ?? false;

    try {
        showToast('Importing data...', 'info');

        const response = await fetch('/api/import', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                data: jsonData,
                import_boards: importBoards,
                import_board_assignments: importBoardAssignments,
                update_existing: updateExisting
            })
        });

        const result = await response.json();

        if (!response.ok || !result.success) {
            throw new Error(result.error || 'Import failed');
        }

        // Show success message with stats
        const stats = [];
        if (result.boards_created > 0) stats.push(`${result.boards_created} boards created`);
        if (result.boards_updated > 0) stats.push(`${result.boards_updated} boards updated`);
        if (result.images_updated > 0) stats.push(`${result.images_updated} images updated`);
        if (result.board_assignments_created > 0) stats.push(`${result.board_assignments_created} assignments created`);

        const message = stats.length > 0
            ? `Import successful: ${stats.join(', ')}`
            : 'Import completed';

        showToast(message, 'success');

        // Clear the textarea
        jsonDataElement.value = '';

        // Refresh the UI
        await loadBoards();
        await loadImages();
        await loadStats();

        // Close modal after a short delay
        setTimeout(() => {
            closeExportImportModal();
        }, 1500);

    } catch (error) {
        console.error('Import error:', error);
        showToast(`Import failed: ${error.message}`, 'error');
    }
}

// Event listeners for export/import
document.getElementById('exportImportBtn')?.addEventListener('click', openExportImportModal);
document.getElementById('exportImportClose')?.addEventListener('click', closeExportImportModal);
document.getElementById('exportImportOverlay')?.addEventListener('click', closeExportImportModal);

document.getElementById('exportJsonBtn')?.addEventListener('click', () => exportData('json'));
document.getElementById('exportMarkdownBtn')?.addEventListener('click', () => exportData('markdown'));
document.getElementById('exportCsvBtn')?.addEventListener('click', () => exportData('csv'));

document.getElementById('importDataBtn')?.addEventListener('click', importData);

// ============ PRIVACY FUNCTIONS ============

// Privacy mode state
let privacyModeEnabled = false;

function openPrivacyModal() {
    const modal = document.getElementById('privacyModal');
    if (modal) {
        modal.style.display = 'flex';
        loadPrivacyStats();
    }
}

function closePrivacyModal() {
    const modal = document.getElementById('privacyModal');
    if (modal) modal.style.display = 'none';
}

async function loadPrivacyStats() {
    try {
        const response = await fetch('/api/privacy/stats');
        const stats = await response.json();

        document.getElementById('privacyStatFaces').textContent = stats.images_with_faces;
        document.getElementById('privacyStatNSFW').textContent = stats.nsfw_images;
        document.getElementById('privacyStatUnanalyzed').textContent = stats.unanalyzed_images;
    } catch (error) {
        console.error('Error loading privacy stats:', error);
    }
}

async function analyzeImagePrivacy(imageId) {
    try {
        const response = await fetch(`/api/images/${imageId}/privacy/analyze`, {
            method: 'POST'
        });

        const result = await response.json();

        if (result.success) {
            return result;
        } else {
            throw new Error(result.error || 'Privacy analysis failed');
        }
    } catch (error) {
        console.error('Error analyzing privacy:', error);
        throw error;
    }
}

async function analyzeAllPrivacy() {
    try {
        showToast('Starting privacy analysis...', 'info');

        // Get all unanalyzed images
        const images = state.images.filter(img => !img.privacy_analyzed_at);

        if (images.length === 0) {
            showToast('All images already analyzed', 'info');
            return;
        }

        const imageIds = images.map(img => img.id);

        const response = await fetch('/api/privacy/batch-analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image_ids: imageIds })
        });

        const result = await response.json();

        showToast(
            `Privacy analysis complete: ${result.success.length} analyzed, ${result.failed.length} failed`,
            result.failed.length > 0 ? 'warning' : 'success'
        );

        // Refresh stats
        loadPrivacyStats();
        await loadImages();

    } catch (error) {
        console.error('Error in batch privacy analysis:', error);
        showToast(`Privacy analysis failed: ${error.message}`, 'error');
    }
}

async function analyzeSelectedPrivacy() {
    if (!state.selectionMode || state.selectedImages.size === 0) {
        showToast('Please select images first', 'warning');
        return;
    }

    try {
        showToast('Analyzing selected images...', 'info');

        const imageIds = Array.from(state.selectedImages);

        const response = await fetch('/api/privacy/batch-analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image_ids: imageIds })
        });

        const result = await response.json();

        showToast(
            `Privacy analysis complete: ${result.success.length} analyzed, ${result.failed.length} failed`,
            result.failed.length > 0 ? 'warning' : 'success'
        );

        // Refresh stats
        loadPrivacyStats();
        await loadImages();

    } catch (error) {
        console.error('Error in batch privacy analysis:', error);
        showToast(`Privacy analysis failed: ${error.message}`, 'error');
    }
}

function togglePrivacyMode() {
    privacyModeEnabled = !privacyModeEnabled;

    const checkbox = document.getElementById('privacyModeEnabled');
    if (checkbox) {
        checkbox.checked = privacyModeEnabled;
    }

    // Reload images with/without blur
    renderImages();

    showToast(
        privacyModeEnabled ? 'Privacy mode enabled' : 'Privacy mode disabled',
        'success'
    );
}

// Update blur strength value display
document.getElementById('blurStrength')?.addEventListener('input', (e) => {
    document.getElementById('blurStrengthValue').textContent = e.target.value;
});

// Event listeners for stats dashboard
document.getElementById('statsBtn')?.addEventListener('click', openStatsModal);
document.getElementById('statsClose')?.addEventListener('click', closeStatsModal);
document.getElementById('statsOverlay')?.addEventListener('click', closeStatsModal);

// Event listeners for privacy
document.getElementById('privacyBtn')?.addEventListener('click', openPrivacyModal);
document.getElementById('privacyClose')?.addEventListener('click', closePrivacyModal);
document.getElementById('privacyOverlay')?.addEventListener('click', closePrivacyModal);

document.getElementById('analyzeAllPrivacyBtn')?.addEventListener('click', analyzeAllPrivacy);
document.getElementById('analyzeSelectedPrivacyBtn')?.addEventListener('click', analyzeSelectedPrivacy);

document.getElementById('privacyModeEnabled')?.addEventListener('change', (e) => {
    privacyModeEnabled = e.target.checked;
    renderImages();
    showToast(
        privacyModeEnabled ? 'Privacy mode enabled' : 'Privacy mode disabled',
        'success'
    );
});

// ============ TOOLS DROPDOWN MENU ============

const toolsMenuBtn = document.getElementById('toolsMenuBtn');
const toolsDropdown = document.getElementById('toolsDropdown');

if (toolsMenuBtn && toolsDropdown) {
    // Toggle dropdown
    toolsMenuBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        const isVisible = toolsDropdown.style.display === 'block';
        toolsDropdown.style.display = isVisible ? 'none' : 'block';
    });

    // Close dropdown when clicking outside
    document.addEventListener('click', (e) => {
        if (!toolsMenuBtn.contains(e.target) && !toolsDropdown.contains(e.target)) {
            toolsDropdown.style.display = 'none';
        }
    });

    // Hover effects for dropdown items
    const dropdownItems = toolsDropdown.querySelectorAll('.dropdown-item');
    dropdownItems.forEach(item => {
        item.addEventListener('mouseenter', () => {
            item.style.background = 'var(--bg-tertiary)';
        });
        item.addEventListener('mouseleave', () => {
            item.style.background = 'none';
        });
        item.addEventListener('click', () => {
            toolsDropdown.style.display = 'none';
        });
    });
}

// ============ BOARD DRAG & DROP (for sub-boards) ============

function enableBoardDragDrop() {
    const boardsList = document.getElementById('boardsList');
    if (!boardsList) return;

    let draggedBoard = null;

    // Add dragstart listener to all board pills
    boardsList.addEventListener('dragstart', (e) => {
        const boardPill = e.target.closest('.board-pill[data-board-id]');
        if (boardPill) {
            draggedBoard = parseInt(boardPill.dataset.boardId);
            e.dataTransfer.effectAllowed = 'move';
            boardPill.style.opacity = '0.5';
        }
    });

    // Add dragend listener
    boardsList.addEventListener('dragend', (e) => {
        const boardPill = e.target.closest('.board-pill[data-board-id]');
        if (boardPill) {
            boardPill.style.opacity = '1';
        }
    });

    // Add dragover listener
    boardsList.addEventListener('dragover', (e) => {
        e.preventDefault();
        const boardPill = e.target.closest('.board-pill[data-board-id]');
        if (boardPill && draggedBoard) {
            const targetBoardId = parseInt(boardPill.dataset.boardId);
            if (draggedBoard !== targetBoardId) {
                boardPill.style.background = 'var(--primary-color)';
                boardPill.style.opacity = '0.7';
            }
        }
    });

    // Add dragleave listener
    boardsList.addEventListener('dragleave', (e) => {
        const boardPill = e.target.closest('.board-pill[data-board-id]');
        if (boardPill) {
            boardPill.style.background = '';
            boardPill.style.opacity = '1';
        }
    });

    // Add drop listener
    boardsList.addEventListener('drop', async (e) => {
        e.preventDefault();
        const boardPill = e.target.closest('.board-pill[data-board-id]');

        if (boardPill && draggedBoard) {
            const targetBoardId = parseInt(boardPill.dataset.boardId);

            // Reset styles
            boardPill.style.background = '';
            boardPill.style.opacity = '1';

            if (draggedBoard !== targetBoardId) {
                // Ask for confirmation
                if (confirm(`Move board to become a sub-board of the target board?`)) {
                    await makeBoardSubBoard(draggedBoard, targetBoardId);
                }
            }
        }

        draggedBoard = null;
    });

    // Make board pills draggable
    const updateBoardsDraggable = () => {
        const pills = boardsList.querySelectorAll('.board-pill[data-board-id]');
        pills.forEach(pill => {
            pill.setAttribute('draggable', 'true');
        });
    };

    // Call initially and after board list updates
    updateBoardsDraggable();

    // Observe for changes in boards list
    const observer = new MutationObserver(updateBoardsDraggable);
    observer.observe(boardsList, { childList: true, subtree: true });
}

async function makeBoardSubBoard(boardId, newParentId) {
    try {
        const response = await fetch(`/api/boards/${boardId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                parent_id: newParentId
            })
        });

        const data = await response.json();

        if (data.success) {
            showMessage('Board moved to sub-board!', 'success');
            await loadBoards();
        } else {
            showMessage('Error: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('Error moving board:', error);
        showMessage('Error moving board', 'error');
    }
}

// Initialize board drag & drop after boards are loaded
setTimeout(() => {
    enableBoardDragDrop();
}, 1000);

// ============ SEND TO TELEGRAM ============

async function sendImageToTelegram(imageId) {
    showMessage('Telegram integration: Please configure your Telegram bot in Settings first. Then you can send images via the bot commands.', 'info');
    // Future: API endpoint for sending images directly from UI
}

console.log('AI Gallery initialized ✨');