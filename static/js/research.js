/**
 * Research & Education Module
 * Handles annotations, citations, and dataset export
 */

// ============ STATE MANAGEMENT ============

const researchState = {
    currentImageForAnnotation: null,
    annotations: [],
    drawingAnnotation: null,
    selectedAnnotationIndex: null,
    classes: [],
    datasetStats: null
};

// ============ MODAL & TAB MANAGEMENT ============

function openResearchModal() {
    document.getElementById('researchModal').style.display = 'block';
    loadResearchData();
}

function closeResearchModal() {
    document.getElementById('researchModal').style.display = 'none';
}

function switchResearchTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.research-tab-content').forEach(tab => {
        tab.style.display = 'none';
    });

    // Remove active class from all tab buttons
    document.querySelectorAll('.research-tab').forEach(btn => {
        btn.classList.remove('active');
        btn.style.borderBottomColor = 'transparent';
        btn.style.color = 'var(--text-muted)';
    });

    // Show selected tab
    document.getElementById(tabName + 'Tab').style.display = 'block';

    // Mark button as active
    const activeBtn = document.querySelector(`[data-tab="${tabName}"]`);
    activeBtn.classList.add('active');
    activeBtn.style.borderBottomColor = 'var(--primary-color)';
    activeBtn.style.color = 'var(--text-color)';

    // Load tab-specific data
    if (tabName === 'annotations') {
        loadDatasetStats();
        loadDatasetClasses();
    } else if (tabName === 'dataset') {
        loadDatasetStats();
    }
}

async function loadResearchData() {
    await loadDatasetStats();
    await loadDatasetClasses();
}

// ============ DATASET STATISTICS ============

async function loadDatasetStats() {
    try {
        const response = await fetch('/api/dataset/stats');
        const data = await response.json();

        if (data.success) {
            researchState.datasetStats = data.stats;

            // Update UI
            document.getElementById('statAnnotatedImages').textContent = data.stats.annotated_images;
            document.getElementById('statTotalAnnotations').textContent = data.stats.total_annotations;
            document.getElementById('statTotalClasses').textContent = data.stats.class_distribution.length;
            document.getElementById('statExportReady').textContent = data.stats.annotated_images;
            document.getElementById('statExportAnnotations').textContent = data.stats.total_annotations;

            // Update class distribution
            const distList = document.getElementById('classDistributionList');
            if (data.stats.class_distribution.length > 0) {
                distList.innerHTML = data.stats.class_distribution.map(cls => `
                    <div style="display: flex; justify-content: space-between; padding: var(--spacing-xs); border-bottom: 1px solid var(--border-color);">
                        <span>${cls.class_name}</span>
                        <span style="color: var(--primary-color); font-weight: bold;">${cls.count}</span>
                    </div>
                `).join('');
            } else {
                distList.innerHTML = '<p style="color: var(--text-muted); text-align: center;">No annotations yet</p>';
            }
        }
    } catch (error) {
        console.error('Error loading dataset stats:', error);
    }
}

// ============ DATASET CLASSES ============

async function loadDatasetClasses() {
    try {
        const response = await fetch('/api/dataset/classes');
        const data = await response.json();

        if (data.success) {
            researchState.classes = data.classes;

            // Update class selects
            const selects = [
                document.getElementById('annotationClassSelect'),
                document.getElementById('annotationCanvasClass')
            ];

            selects.forEach(select => {
                select.innerHTML = '<option value="">Select class...</option>' +
                    data.classes.map(cls => `<option value="${cls.name}">${cls.name}</option>`).join('');
            });
        }
    } catch (error) {
        console.error('Error loading classes:', error);
    }
}

async function addNewClass() {
    const className = prompt('Enter new class name:');
    if (!className) return;

    try {
        const response = await fetch('/api/dataset/classes', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name: className,
                color: '#' + Math.floor(Math.random() * 16777215).toString(16)
            })
        });

        const data = await response.json();

        if (data.success) {
            showMessage('Class added successfully!', 'success');
            await loadDatasetClasses();
        } else {
            showMessage('Error adding class: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('Error adding class:', error);
        showMessage('Error adding class', 'error');
    }
}

// ============ ANNOTATIONS ============

async function startAnnotation() {
    const selectedImage = getFirstSelectedImage();

    if (!selectedImage) {
        showMessage('Please select an image first', 'error');
        return;
    }

    researchState.currentImageForAnnotation = selectedImage;

    // Load existing annotations
    await loadImageAnnotations(selectedImage.id);

    // Open annotation canvas
    openAnnotationCanvas(selectedImage);
}

async function loadImageAnnotations(imageId) {
    try {
        const response = await fetch(`/api/images/${imageId}/annotations`);
        const data = await response.json();

        if (data.success) {
            researchState.annotations = data.annotations;
        }
    } catch (error) {
        console.error('Error loading annotations:', error);
    }
}

function openAnnotationCanvas(image) {
    document.getElementById('annotationCanvasModal').style.display = 'block';

    // Load image into canvas
    const canvas = document.getElementById('annotationCanvas');
    const ctx = canvas.getContext('2d');

    const img = new Image();
    img.onload = function () {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);

        // Draw existing annotations
        drawAnnotations(ctx);

        // Setup canvas interactions
        setupCanvasInteractions(canvas, ctx, img);
    };
    img.src = `/api/images/${image.id}/file?t=${Date.now()}`;
}

function closeAnnotationCanvas() {
    document.getElementById('annotationCanvasModal').style.display = 'none';
    researchState.currentImageForAnnotation = null;
    researchState.annotations = [];
}

function setupCanvasInteractions(canvas, ctx, img) {
    let isDrawing = false;
    let startX, startY;

    canvas.onmousedown = function (e) {
        const rect = canvas.getBoundingClientRect();
        startX = e.clientX - rect.left;
        startY = e.clientY - rect.top;
        isDrawing = true;
    };

    canvas.onmousemove = function (e) {
        if (!isDrawing) return;

        const rect = canvas.getBoundingClientRect();
        const currentX = e.clientX - rect.left;
        const currentY = e.clientY - rect.top;

        // Redraw image and existing annotations
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0);
        drawAnnotations(ctx);

        // Draw current box
        ctx.strokeStyle = '#FF5722';
        ctx.lineWidth = 2;
        ctx.strokeRect(startX, startY, currentX - startX, currentY - startY);
    };

    canvas.onmouseup = function (e) {
        if (!isDrawing) return;
        isDrawing = false;

        const rect = canvas.getBoundingClientRect();
        const endX = e.clientX - rect.left;
        const endY = e.clientY - rect.top;

        const width = Math.abs(endX - startX);
        const height = Math.abs(endY - startY);

        // Only create annotation if box is large enough
        if (width > 5 && height > 5) {
            const className = document.getElementById('annotationCanvasClass').value;

            if (!className) {
                showMessage('Please select a class first', 'error');
                return;
            }

            const annotation = {
                class_name: className,
                x: Math.min(startX, endX),
                y: Math.min(startY, endY),
                width: width,
                height: height,
                temp_id: Date.now()
            };

            researchState.annotations.push(annotation);

            // Redraw
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0);
            drawAnnotations(ctx);

            updateAnnotationsList();
        }
    };

    // Delete key to remove selected annotation
    document.addEventListener('keydown', function (e) {
        if (e.key === 'Delete' && researchState.selectedAnnotationIndex !== null) {
            researchState.annotations.splice(researchState.selectedAnnotationIndex, 1);
            researchState.selectedAnnotationIndex = null;

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0);
            drawAnnotations(ctx);
            updateAnnotationsList();
        }
    });
}

function drawAnnotations(ctx) {
    researchState.annotations.forEach((ann, index) => {
        ctx.strokeStyle = index === researchState.selectedAnnotationIndex ? '#00FF00' : '#FF5722';
        ctx.lineWidth = 2;
        ctx.strokeRect(ann.x, ann.y, ann.width, ann.height);

        // Draw label
        ctx.fillStyle = 'rgba(255, 87, 34, 0.8)';
        ctx.fillRect(ann.x, ann.y - 20, ctx.measureText(ann.class_name).width + 10, 20);
        ctx.fillStyle = '#FFFFFF';
        ctx.font = '12px Arial';
        ctx.fillText(ann.class_name, ann.x + 5, ann.y - 5);
    });
}

function updateAnnotationsList() {
    const listContent = document.getElementById('annotationsListContent');

    if (researchState.annotations.length === 0) {
        listContent.innerHTML = '<p style="color: var(--text-muted); text-align: center;">No annotations</p>';
        return;
    }

    listContent.innerHTML = researchState.annotations.map((ann, index) => `
        <div style="display: flex; justify-content: space-between; padding: var(--spacing-xs); border-bottom: 1px solid var(--border-color); cursor: pointer;"
             onclick="selectAnnotation(${index})">
            <span><strong>${ann.class_name}</strong> (${Math.round(ann.x)}, ${Math.round(ann.y)}, ${Math.round(ann.width)}×${Math.round(ann.height)})</span>
            <button onclick="deleteAnnotation(${index}); event.stopPropagation();" style="background: var(--error-color); color: white; border: none; padding: 2px 8px; border-radius: 4px; cursor: pointer;">Delete</button>
        </div>
    `).join('');
}

function selectAnnotation(index) {
    researchState.selectedAnnotationIndex = index;
    const canvas = document.getElementById('annotationCanvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();
    img.onload = function () {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0);
        drawAnnotations(ctx);
    };
    img.src = `/api/images/${researchState.currentImageForAnnotation.id}/file?t=${Date.now()}`;
}

function deleteAnnotation(index) {
    researchState.annotations.splice(index, 1);
    researchState.selectedAnnotationIndex = null;

    const canvas = document.getElementById('annotationCanvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();
    img.onload = function () {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0);
        drawAnnotations(ctx);
        updateAnnotationsList();
    };
    img.src = `/api/images/${researchState.currentImageForAnnotation.id}/file?t=${Date.now()}`;
}

async function saveAnnotations() {
    if (!researchState.currentImageForAnnotation) return;

    try {
        for (const ann of researchState.annotations) {
            // Skip if already saved (has id instead of temp_id)
            if (ann.id) continue;

            const response = await fetch('/api/annotations', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_id: researchState.currentImageForAnnotation.id,
                    class_name: ann.class_name,
                    x: ann.x,
                    y: ann.y,
                    width: ann.width,
                    height: ann.height
                })
            });

            const data = await response.json();

            if (data.success) {
                ann.id = data.annotation_id;
                delete ann.temp_id;
            }
        }

        showMessage(`Saved ${researchState.annotations.length} annotations!`, 'success');
        await loadDatasetStats();
        closeAnnotationCanvas();

    } catch (error) {
        console.error('Error saving annotations:', error);
        showMessage('Error saving annotations', 'error');
    }
}

function clearAllAnnotations() {
    if (!confirm('Clear all annotations for this image?')) return;
    researchState.annotations = [];
    researchState.selectedAnnotationIndex = null;

    const canvas = document.getElementById('annotationCanvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();
    img.onload = function () {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0);
        updateAnnotationsList();
    };
    img.src = `/api/images/${researchState.currentImageForAnnotation.id}/file?t=${Date.now()}`;
}

async function importFromPrivacy() {
    const selectedImage = getFirstSelectedImage();

    if (!selectedImage) {
        showMessage('Please select an image first', 'error');
        return;
    }

    try {
        const response = await fetch('/api/annotations/import-from-privacy', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_id: selectedImage.id })
        });

        const data = await response.json();

        if (data.success) {
            showMessage(`Imported ${data.imported} annotations from privacy zones!`, 'success');
            await loadDatasetStats();
        } else {
            showMessage('Error: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('Error importing from privacy:', error);
        showMessage('Error importing from privacy', 'error');
    }
}

// ============ CITATIONS ============

async function generateCitation() {
    const selectedImage = getFirstSelectedImage();

    if (!selectedImage) {
        showMessage('Please select an image first', 'error');
        return;
    }

    const format = document.getElementById('citationFormat').value;

    try {
        const response = await fetch('/api/citation', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_id: selectedImage.id,
                format: format
            })
        });

        const data = await response.json();

        if (data.success) {
            document.getElementById('citationOutput').style.display = 'block';
            document.getElementById('citationsList').textContent = data.citation;
        } else {
            showMessage('Error: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('Error generating citation:', error);
        showMessage('Error generating citation', 'error');
    }
}

async function generateBatchCitations() {
    const imageIds = Array.from(state.selectedImages);

    if (imageIds.length === 0) {
        showMessage('Please select at least one image', 'error');
        return;
    }

    const format = document.getElementById('citationFormat').value;

    try {
        const response = await fetch('/api/citation/batch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_ids: imageIds,
                format: format
            })
        });

        const data = await response.json();

        if (data.success) {
            document.getElementById('citationOutput').style.display = 'block';
            const citations = data.citations.map(c =>
                `${c.filename}:\n${c.citation}\n`
            ).join('\n---\n\n');
            document.getElementById('citationsList').textContent = citations;
        } else {
            showMessage('Error: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('Error generating citations:', error);
        showMessage('Error generating batch citations', 'error');
    }
}

function copyCitations() {
    const citations = document.getElementById('citationsList').textContent;
    navigator.clipboard.writeText(citations);
    showMessage('Citations copied to clipboard!', 'success');
}

// ============ DATASET EXPORT ============

async function exportDataset() {
    const format = document.getElementById('datasetFormat').value;
    const split = document.getElementById('splitDataset').checked;

    let url = `/api/dataset/export?format=${format}`;

    if (format === 'zip') {
        url += `&export_format=coco&split=${split}`;
    }

    try {
        const response = await fetch(url);

        if (format === 'csv' || format === 'zip') {
            // Download file
            const blob = await response.blob();
            const downloadUrl = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = downloadUrl;
            a.download = `dataset_${format}_${Date.now()}.${format === 'zip' ? 'zip' : 'csv'}`;
            document.body.appendChild(a);
            a.click();
            a.remove();
            showMessage('Dataset exported successfully!', 'success');
        } else {
            // Show JSON
            const data = await response.json();
            document.getElementById('datasetPreview').style.display = 'block';
            document.getElementById('datasetPreviewContent').textContent = JSON.stringify(data, null, 2);
        }
    } catch (error) {
        console.error('Error exporting dataset:', error);
        showMessage('Error exporting dataset', 'error');
    }
}

async function previewDataset() {
    const format = document.getElementById('datasetFormat').value;

    if (format === 'zip') {
        showMessage('Preview not available for ZIP format. Use Export instead.', 'info');
        return;
    }

    try {
        const response = await fetch(`/api/dataset/export?format=${format}`);
        const data = format === 'csv' ? await response.text() : await response.json();

        document.getElementById('datasetPreview').style.display = 'block';
        document.getElementById('datasetPreviewContent').textContent =
            format === 'csv' ? data : JSON.stringify(data, null, 2);
    } catch (error) {
        console.error('Error previewing dataset:', error);
        showMessage('Error previewing dataset', 'error');
    }
}

// ============ HELPER FUNCTIONS ============

function getFirstSelectedImage() {
    const imageId = state.selectedImages.values().next().value;
    return state.images.find(img => img.id === imageId);
}

// ============ EVENT LISTENERS ============

// Research modal
document.getElementById('researchBtn')?.addEventListener('click', openResearchModal);
document.getElementById('researchClose')?.addEventListener('click', closeResearchModal);
document.getElementById('researchOverlay')?.addEventListener('click', closeResearchModal);

// Tab switching
document.querySelectorAll('.research-tab').forEach(btn => {
    btn.addEventListener('click', () => {
        switchResearchTab(btn.dataset.tab);
    });
});

// Annotations
document.getElementById('addNewClassBtn')?.addEventListener('click', addNewClass);
document.getElementById('startAnnotationBtn')?.addEventListener('click', startAnnotation);
document.getElementById('importFromPrivacyBtn')?.addEventListener('click', importFromPrivacy);

// Annotation canvas
document.getElementById('annotationCanvasClose')?.addEventListener('click', closeAnnotationCanvas);
document.getElementById('annotationCanvasOverlay')?.addEventListener('click', closeAnnotationCanvas);
document.getElementById('saveAnnotationsBtn')?.addEventListener('click', saveAnnotations);
document.getElementById('clearAnnotationsBtn')?.addEventListener('click', clearAllAnnotations);

// Citations
document.getElementById('generateCitationBtn')?.addEventListener('click', generateCitation);
document.getElementById('generateBatchCitationBtn')?.addEventListener('click', generateBatchCitations);
document.getElementById('copyCitationsBtn')?.addEventListener('click', copyCitations);

// Dataset export
document.getElementById('exportDatasetBtn')?.addEventListener('click', exportDataset);
document.getElementById('previewDatasetBtn')?.addEventListener('click', previewDataset);

console.log('Research & Education module loaded');
