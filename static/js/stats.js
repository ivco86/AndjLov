/**
 * Stats Dashboard
 * Interactive analytics and visualizations
 */

const statsState = {
    overview: null,
    timeline: null,
    storage: null,
    activity: null,
    charts: {}
};

// ============ Load Stats Data ============

async function loadStatsOverview() {
    try {
        const response = await fetch('/api/stats/overview');
        const data = await response.json();

        if (data.success) {
            statsState.overview = data.stats;
            renderOverviewStats();
        }
    } catch (error) {
        console.error('Error loading overview stats:', error);
    }
}

async function loadTimelineStats() {
    try {
        const response = await fetch('/api/stats/timeline');
        const data = await response.json();

        if (data.success) {
            statsState.timeline = data.timeline;
            renderTimelineChart();
        }
    } catch (error) {
        console.error('Error loading timeline stats:', error);
    }
}

async function loadStorageStats() {
    try {
        const response = await fetch('/api/stats/storage');
        const data = await response.json();

        if (data.success) {
            statsState.storage = data.storage;
            renderStorageChart();
        }
    } catch (error) {
        console.error('Error loading storage stats:', error);
    }
}

async function loadActivityStats() {
    try {
        const response = await fetch('/api/stats/activity');
        const data = await response.json();

        if (data.success) {
            statsState.activity = data.activity;
            renderActivityStats();
        }
    } catch (error) {
        console.error('Error loading activity stats:', error);
    }
}

// ============ Render Stats ============

function renderOverviewStats() {
    const stats = statsState.overview;
    if (!stats) return;

    // Update stat cards
    document.getElementById('statTotalImages').textContent = stats.total_images.toLocaleString();
    document.getElementById('statFavorites').textContent = stats.favorites.toLocaleString();
    document.getElementById('statAnalyzed').textContent = stats.analyzed.toLocaleString();
    document.getElementById('statStorage').textContent = stats.total_size_mb.toLocaleString() + ' MB';
    document.getElementById('statBoards').textContent = stats.total_boards.toLocaleString();
    document.getElementById('statTags').textContent = stats.total_tags.toLocaleString();

    // Render tags chart
    renderTagsChart(stats.top_tags);
}

function renderTagsChart(topTags) {
    const canvas = document.getElementById('tagsChart');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');

    // Destroy existing chart
    if (statsState.charts.tags) {
        statsState.charts.tags.destroy();
    }

    statsState.charts.tags = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: topTags.map(t => t.tag),
            datasets: [{
                label: 'Images',
                data: topTags.map(t => t.count),
                backgroundColor: 'rgba(33, 150, 243, 0.6)',
                borderColor: 'rgba(33, 150, 243, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Top 10 Tags'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1
                    }
                }
            }
        }
    });
}

function renderTimelineChart() {
    const timeline = statsState.timeline;
    if (!timeline) return;

    const canvas = document.getElementById('timelineChart');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');

    // Destroy existing chart
    if (statsState.charts.timeline) {
        statsState.charts.timeline.destroy();
    }

    // Use monthly data for better visualization
    const data = timeline.monthly.length > 0 ? timeline.monthly : timeline.daily.slice(-30);
    const labels = data.map(d => d.month || d.date);
    const counts = data.map(d => d.count);

    statsState.charts.timeline = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Images Added',
                data: counts,
                borderColor: 'rgba(76, 175, 80, 1)',
                backgroundColor: 'rgba(76, 175, 80, 0.2)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Images Over Time'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1
                    }
                }
            }
        }
    });
}

function renderStorageChart() {
    const storage = statsState.storage;
    if (!storage || storage.length === 0) return;

    const canvas = document.getElementById('storageChart');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');

    // Destroy existing chart
    if (statsState.charts.storage) {
        statsState.charts.storage.destroy();
    }

    const colors = [
        'rgba(255, 99, 132, 0.6)',
        'rgba(54, 162, 235, 0.6)',
        'rgba(255, 206, 86, 0.6)',
        'rgba(75, 192, 192, 0.6)',
        'rgba(153, 102, 255, 0.6)'
    ];

    statsState.charts.storage = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: storage.map(s => s.type.toUpperCase()),
            datasets: [{
                data: storage.map(s => s.size_mb),
                backgroundColor: colors.slice(0, storage.length),
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right'
                },
                title: {
                    display: true,
                    text: 'Storage by Type (MB)'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.parsed || 0;
                            const item = storage[context.dataIndex];
                            return `${label}: ${value.toFixed(2)} MB (${item.count} files)`;
                        }
                    }
                }
            }
        }
    });
}

function renderActivityStats() {
    const activity = statsState.activity;
    if (!activity) return;

    document.getElementById('statRecentImages').textContent = activity.images_last_7_days.toLocaleString();

    // Render recent executions
    const container = document.getElementById('recentExecutionsList');
    if (!container) return;

    if (activity.recent_executions.length === 0) {
        container.innerHTML = '<p style="color: var(--text-secondary); text-align: center;">No recent pipeline executions</p>';
        return;
    }

    container.innerHTML = activity.recent_executions.map(exec => `
        <div style="padding: var(--spacing-sm); border-bottom: 1px solid var(--border-color);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <strong>${exec.pipeline_name || 'Pipeline #' + exec.pipeline_id}</strong>
                <span style="font-size: 0.85em; color: var(--text-secondary);">
                    ${new Date(exec.started_at).toLocaleString()}
                </span>
            </div>
            <div style="margin-top: 4px; font-size: 0.9em; color: var(--text-secondary);">
                Status: <span style="color: ${exec.status === 'completed' ? 'var(--success-color)' : 'var(--warning-color)'}">
                    ${exec.status}
                </span>
                - ${exec.completed_actions}/${exec.total_actions} actions
            </div>
        </div>
    `).join('');
}

// ============ Open Stats Dashboard ============

function openStatsModal() {
    const modal = document.getElementById('statsModal');
    modal.style.display = 'block';

    // Load all stats
    loadStatsOverview();
    loadTimelineStats();
    loadStorageStats();
    loadActivityStats();
}

function closeStatsModal() {
    closeModal('statsModal');

    // Destroy charts to prevent memory leaks
    Object.values(statsState.charts).forEach(chart => {
        if (chart) chart.destroy();
    });
    statsState.charts = {};
}

// ============ Refresh Stats ============

function refreshStats() {
    showMessage('Refreshing statistics...', 'info');

    loadStatsOverview();
    loadTimelineStats();
    loadStorageStats();
    loadActivityStats();

    setTimeout(() => {
        showMessage('Statistics refreshed!', 'success');
    }, 500);
}

// ============ Initialize Event Listeners ============

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initStatsEventListeners);
} else {
    initStatsEventListeners();
}

function initStatsEventListeners() {
    // Stats modal controls
    document.getElementById('statsBtn')?.addEventListener('click', openStatsModal);
    document.getElementById('statsClose')?.addEventListener('click', closeStatsModal);
    document.getElementById('statsOverlay')?.addEventListener('click', closeStatsModal);

    // Refresh button if it exists
    document.getElementById('statsRefresh')?.addEventListener('click', refreshStats);
}
