/**
 * Workflow Automation Module
 * Handles pipeline creation, execution, and monitoring
 */

// ============ STATE MANAGEMENT ============

const workflowState = {
    pipelines: [],
    templates: [],
    availableActions: [],
    executions: [],
    currentPipeline: null,
    editingPipeline: null
};

// ============ MODAL MANAGEMENT ============

function openWorkflowsModal() {
    document.getElementById('workflowsModal').style.display = 'block';
    loadWorkflowData();
}

function closeWorkflowsModal() {
    document.getElementById('workflowsModal').style.display = 'none';
}

function switchWorkflowTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.workflow-tab-content').forEach(tab => {
        tab.style.display = 'none';
    });

    // Remove active class
    document.querySelectorAll('.workflow-tab').forEach(btn => {
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

    // Load tab data
    if (tabName === 'pipelines') {
        loadPipelines();
    } else if (tabName === 'templates') {
        loadTemplates();
    } else if (tabName === 'logs') {
        loadExecutionLogs();
    }
}

async function loadWorkflowData() {
    await Promise.all([
        loadPipelines(),
        loadAvailableActions()
    ]);
}

// ============ PIPELINES ============

async function loadPipelines() {
    try {
        const response = await fetch('/api/pipelines');
        const data = await response.json();

        if (data.success) {
            workflowState.pipelines = data.pipelines;
            renderPipelines();
        }
    } catch (error) {
        console.error('Error loading pipelines:', error);
    }
}

function renderPipelines() {
    const listEl = document.getElementById('pipelinesList');
    const noMsgEl = document.getElementById('noPipelinesMessage');

    if (workflowState.pipelines.length === 0) {
        listEl.innerHTML = '';
        noMsgEl.style.display = 'block';
        return;
    }

    noMsgEl.style.display = 'none';

    listEl.innerHTML = workflowState.pipelines.map(pipeline => `
        <div style="background: var(--bg-secondary); padding: var(--spacing-md); border-radius: var(--border-radius); border-left: 4px solid ${pipeline.enabled ? 'var(--success-color)' : 'var(--text-muted)'};">
            <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: var(--spacing-sm);">
                <div style="flex: 1;">
                    <h4 style="margin: 0; margin-bottom: var(--spacing-xs);">${pipeline.name}</h4>
                    <p style="margin: 0; color: var(--text-muted); font-size: 0.875rem;">${pipeline.description || 'No description'}</p>
                </div>
                <div style="display: flex; gap: var(--spacing-xs);">
                    <button onclick="runPipeline(${pipeline.id})" class="btn btn-primary btn-sm" style="padding: 4px 8px; font-size: 0.875rem;">▶️ Run</button>
                    <button onclick="editPipeline(${pipeline.id})" class="btn btn-secondary btn-sm" style="padding: 4px 8px; font-size: 0.875rem;">✏️ Edit</button>
                    <button onclick="deletePipeline(${pipeline.id})" class="btn btn-sm" style="padding: 4px 8px; font-size: 0.875rem; background: var(--error-color); color: white;">🗑️</button>
                </div>
            </div>
            <div style="display: flex; gap: var(--spacing-md); font-size: 0.875rem; color: var(--text-muted);">
                <span>📍 Trigger: <strong>${getTriggerLabel(pipeline.trigger_type)}</strong></span>
                <span>⚙️ Actions: <strong>${pipeline.actions.length}</strong></span>
                <span>▶️ Runs: <strong>${pipeline.run_count || 0}</strong></span>
                <span>${pipeline.enabled ? '✅ Enabled' : '⏸️ Disabled'}</span>
            </div>
        </div>
    `).join('');
}

function getTriggerLabel(triggerType) {
    const labels = {
        'manual': 'Manual',
        'on_scan': 'On Scan',
        'on_upload': 'On Upload'
    };
    return labels[triggerType] || triggerType;
}

async function runPipeline(pipelineId) {
    const imageIds = Array.from(state.selectedImages);

    if (imageIds.length === 0) {
        showMessage('Please select at least one image', 'error');
        return;
    }

    if (!confirm(`Run pipeline on ${imageIds.length} image(s)?`)) return;

    try {
        const response = await fetch(`/api/pipelines/${pipelineId}/execute`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_ids: imageIds })
        });

        const data = await response.json();

        if (data.success) {
            showMessage(`Pipeline completed! ${data.successful} successful, ${data.failed} failed`, 'success');
            loadPipelines();
        } else {
            showMessage('Error: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('Error running pipeline:', error);
        showMessage('Error running pipeline', 'error');
    }
}

async function deletePipeline(pipelineId) {
    if (!confirm('Delete this pipeline?')) return;

    try {
        const response = await fetch(`/api/pipelines/${pipelineId}`, {
            method: 'DELETE'
        });

        const data = await response.json();

        if (data.success) {
            showMessage('Pipeline deleted', 'success');
            loadPipelines();
        } else {
            showMessage('Error deleting pipeline', 'error');
        }
    } catch (error) {
        console.error('Error deleting pipeline:', error);
        showMessage('Error deleting pipeline', 'error');
    }
}

// ============ PIPELINE BUILDER ============

function openPipelineBuilder() {
    document.getElementById('pipelineBuilderModal').style.display = 'block';
    document.getElementById('pipelineBuilderTitle').textContent = 'Create Pipeline';
    workflowState.editingPipeline = null;

    // Reset form
    document.getElementById('pipelineName').value = '';
    document.getElementById('pipelineDescription').value = '';
    document.getElementById('pipelineTrigger').value = 'manual';
    document.getElementById('pipelineEnabled').checked = true;
    workflowState.currentPipeline = { actions: [] };
    renderActionsList();
}

async function editPipeline(pipelineId) {
    try {
        const response = await fetch(`/api/pipelines/${pipelineId}`);
        const data = await response.json();

        if (data.success) {
            const pipeline = data.pipeline;
            workflowState.editingPipeline = pipeline;
            workflowState.currentPipeline = { actions: [...pipeline.actions] };

            document.getElementById('pipelineBuilderModal').style.display = 'block';
            document.getElementById('pipelineBuilderTitle').textContent = 'Edit Pipeline';
            document.getElementById('pipelineName').value = pipeline.name;
            document.getElementById('pipelineDescription').value = pipeline.description || '';
            document.getElementById('pipelineTrigger').value = pipeline.trigger_type;
            document.getElementById('pipelineEnabled').checked = pipeline.enabled;

            renderActionsList();
        }
    } catch (error) {
        console.error('Error loading pipeline:', error);
        showMessage('Error loading pipeline', 'error');
    }
}

function closePipelineBuilder() {
    document.getElementById('pipelineBuilderModal').style.display = 'none';
    workflowState.currentPipeline = null;
    workflowState.editingPipeline = null;
}

async function savePipeline(event) {
    event.preventDefault();

    const name = document.getElementById('pipelineName').value.trim();
    const description = document.getElementById('pipelineDescription').value.trim();
    const trigger_type = document.getElementById('pipelineTrigger').value;
    const enabled = document.getElementById('pipelineEnabled').checked;
    const actions = workflowState.currentPipeline.actions;

    if (!name) {
        showMessage('Pipeline name is required', 'error');
        return;
    }

    if (actions.length === 0) {
        showMessage('Add at least one action', 'error');
        return;
    }

    const pipelineData = {
        name,
        description,
        trigger_type,
        trigger_config: {},
        actions,
        enabled
    };

    try {
        let response;

        if (workflowState.editingPipeline) {
            // Update existing
            response = await fetch(`/api/pipelines/${workflowState.editingPipeline.id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(pipelineData)
            });
        } else {
            // Create new
            response = await fetch('/api/pipelines', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(pipelineData)
            });
        }

        const data = await response.json();

        if (data.success) {
            showMessage('Pipeline saved!', 'success');
            closePipelineBuilder();
            loadPipelines();
        } else {
            showMessage('Error: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('Error saving pipeline:', error);
        showMessage('Error saving pipeline', 'error');
    }
}

// ============ ACTIONS ============

async function loadAvailableActions() {
    try {
        const response = await fetch('/api/pipelines/actions');
        const data = await response.json();

        if (data.success) {
            workflowState.availableActions = data.actions;
        }
    } catch (error) {
        console.error('Error loading actions:', error);
    }
}

function renderActionsList() {
    const listEl = document.getElementById('actionsList');

    if (!workflowState.currentPipeline || workflowState.currentPipeline.actions.length === 0) {
        listEl.innerHTML = '<p style="color: var(--text-muted); text-align: center; padding: var(--spacing-md);">No actions yet. Click "+ Add Action" to get started.</p>';
        return;
    }

    listEl.innerHTML = workflowState.currentPipeline.actions.map((action, index) => `
        <div style="background: var(--bg-tertiary); padding: var(--spacing-sm); border-radius: var(--border-radius); display: flex; justify-content: space-between; align-items: center;">
            <div>
                <strong>${index + 1}. ${getActionLabel(action.type)}</strong>
                <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 2px;">
                    ${JSON.stringify(action.params)}
                </div>
            </div>
            <div style="display: flex; gap: var(--spacing-xs);">
                ${index > 0 ? `<button onclick="moveAction(${index}, -1)" class="btn btn-sm" style="padding: 2px 6px;">↑</button>` : ''}
                ${index < workflowState.currentPipeline.actions.length - 1 ? `<button onclick="moveAction(${index}, 1)" class="btn btn-sm" style="padding: 2px 6px;">↓</button>` : ''}
                <button onclick="removeAction(${index})" class="btn btn-sm" style="padding: 2px 6px; background: var(--error-color); color: white;">✕</button>
            </div>
        </div>
    `).join('');
}

function getActionLabel(actionType) {
    const action = workflowState.availableActions.find(a => a.type === actionType);
    return action ? action.description : actionType;
}

function openActionSelector() {
    document.getElementById('actionSelectorModal').style.display = 'block';
    renderAvailableActions();
}

function closeActionSelector() {
    document.getElementById('actionSelectorModal').style.display = 'none';
}

function renderAvailableActions() {
    const listEl = document.getElementById('availableActionsList');

    listEl.innerHTML = workflowState.availableActions.map(action => `
        <div onclick="selectAction('${action.type}')" style="background: var(--bg-secondary); padding: var(--spacing-sm); border-radius: var(--border-radius); cursor: pointer; transition: background 0.2s;" onmouseover="this.style.background='var(--bg-tertiary)'" onmouseout="this.style.background='var(--bg-secondary)'">
            <strong>${action.description}</strong>
            <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 2px;">
                ${action.type}
            </div>
        </div>
    `).join('');
}

function selectAction(actionType) {
    const action = workflowState.availableActions.find(a => a.type === actionType);

    if (action) {
        // Add action with default params
        workflowState.currentPipeline.actions.push({
            type: actionType,
            params: { ...action.default_params }
        });

        renderActionsList();
        closeActionSelector();
    }
}

function removeAction(index) {
    workflowState.currentPipeline.actions.splice(index, 1);
    renderActionsList();
}

function moveAction(index, direction) {
    const newIndex = index + direction;
    if (newIndex < 0 || newIndex >= workflowState.currentPipeline.actions.length) return;

    const temp = workflowState.currentPipeline.actions[index];
    workflowState.currentPipeline.actions[index] = workflowState.currentPipeline.actions[newIndex];
    workflowState.currentPipeline.actions[newIndex] = temp;

    renderActionsList();
}

// ============ TEMPLATES ============

async function loadTemplates() {
    try {
        const response = await fetch('/api/pipelines/templates');
        const data = await response.json();

        if (data.success) {
            workflowState.templates = data.templates;
            renderTemplates();
        }
    } catch (error) {
        console.error('Error loading templates:', error);
    }
}

function renderTemplates() {
    const listEl = document.getElementById('templatesList');

    listEl.innerHTML = workflowState.templates.map(template => `
        <div style="background: var(--bg-secondary); padding: var(--spacing-md); border-radius: var(--border-radius);">
            <h4 style="margin: 0; margin-bottom: var(--spacing-sm);">${template.name}</h4>
            <p style="margin: 0; margin-bottom: var(--spacing-sm); color: var(--text-muted); font-size: 0.875rem;">${template.description}</p>
            <div style="margin-bottom: var(--spacing-sm); font-size: 0.875rem;">
                <strong>Actions:</strong>
                <ul style="margin: var(--spacing-xs) 0; padding-left: var(--spacing-lg);">
                    ${template.actions.map(a => `<li>${getActionLabel(a.type)}</li>`).join('')}
                </ul>
            </div>
            <button onclick="useTemplate('${template.id}')" class="btn btn-primary btn-sm">Use This Template</button>
        </div>
    `).join('');
}

async function useTemplate(templateId) {
    try {
        const response = await fetch(`/api/pipelines/templates/${templateId}`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            showMessage('Pipeline created from template!', 'success');
            switchWorkflowTab('pipelines');
        } else {
            showMessage('Error: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('Error using template:', error);
        showMessage('Error using template', 'error');
    }
}

// ============ EXECUTION LOGS ============

async function loadExecutionLogs() {
    try {
        const response = await fetch('/api/executions/recent?limit=50');
        const data = await response.json();

        if (data.success) {
            workflowState.executions = data.executions;
            renderExecutionLogs();
        }
    } catch (error) {
        console.error('Error loading logs:', error);
    }
}

function renderExecutionLogs() {
    const listEl = document.getElementById('executionLogsList');
    const noMsgEl = document.getElementById('noLogsMessage');

    if (workflowState.executions.length === 0) {
        listEl.innerHTML = '';
        noMsgEl.style.display = 'block';
        return;
    }

    noMsgEl.style.display = 'none';

    listEl.innerHTML = workflowState.executions.map(exec => {
        const statusColor = exec.status === 'completed' ? 'var(--success-color)' :
                          exec.status === 'failed' ? 'var(--error-color)' :
                          'var(--warning-color)';

        return `
            <div style="background: var(--bg-secondary); padding: var(--spacing-sm); border-radius: var(--border-radius); border-left: 4px solid ${statusColor};">
                <div style="display: flex; justify-content: space-between; font-size: 0.875rem;">
                    <div>
                        <strong>${exec.pipeline_name}</strong>
                        <span style="color: var(--text-muted); margin-left: var(--spacing-sm);">${new Date(exec.started_at).toLocaleString()}</span>
                    </div>
                    <div>
                        <span style="color: ${statusColor}; font-weight: bold;">${exec.status}</span>
                    </div>
                </div>
                <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 4px;">
                    ${exec.completed_actions}/${exec.total_actions} actions completed
                    ${exec.failed_actions > 0 ? `• ${exec.failed_actions} failed` : ''}
                    ${exec.trigger_source ? `• Triggered by: ${exec.trigger_source}` : ''}
                </div>
            </div>
        `;
    }).join('');
}

// ============ EVENT LISTENERS ============

// Workflows modal
document.getElementById('workflowsBtn')?.addEventListener('click', openWorkflowsModal);
document.getElementById('workflowsClose')?.addEventListener('click', closeWorkflowsModal);
document.getElementById('workflowsOverlay')?.addEventListener('click', closeWorkflowsModal);

// Tab switching
document.querySelectorAll('.workflow-tab').forEach(btn => {
    btn.addEventListener('click', () => {
        switchWorkflowTab(btn.dataset.tab);
    });
});

// Pipeline builder
document.getElementById('createPipelineBtn')?.addEventListener('click', openPipelineBuilder);
document.getElementById('pipelineBuilderClose')?.addEventListener('click', closePipelineBuilder);
document.getElementById('pipelineBuilderOverlay')?.addEventListener('click', closePipelineBuilder);
document.getElementById('cancelPipelineBtn')?.addEventListener('click', closePipelineBuilder);
document.getElementById('pipelineBuilderForm')?.addEventListener('submit', savePipeline);

// Action selector
document.getElementById('addActionBtn')?.addEventListener('click', openActionSelector);
document.getElementById('actionSelectorClose')?.addEventListener('click', closeActionSelector);
document.getElementById('actionSelectorOverlay')?.addEventListener('click', closeActionSelector);

console.log('Workflow Automation module loaded');
