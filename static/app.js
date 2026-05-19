/**
 * Face Sorter V3 - Application JavaScript
 * ========================================
 * Clean, modular JavaScript with proper state management.
 */

// ============================================================================
// STATE
// ============================================================================
const State = {
    pageNamed: 1,
    pageUnnamed: 1,
    totalPagesNamed: 1,
    totalPagesUnnamed: 1,
    searchTimeout: null,

    // Modal state
    modalFaces: [],
    modalIndex: 0,

    // Context menu state
    contextFaceId: null,
    contextFaceEl: null,

    // Multi-select state
    selectedFaces: [],  // Array of {id, el} objects
    lastSelectedFaceId: null,  // For shift-click range select

    // Filters
    filterSuggestions: false,

    // People cache for search
    peopleData: [],

    // Autocomplete
    activeInput: null,
    autocompleteTimeout: null,
    contextFaceId: null, // Fixed previous typo/missing property if any
};

// ============================================================================
// INITIALIZATION
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
    // Bind slider events
    bindSliders();

    // Bind cluster button
    document.getElementById('cluster-btn').addEventListener('click', runClustering);

    // Bind context menu events
    document.addEventListener('click', handleGlobalClick);
    document.getElementById('ctx-new-name').addEventListener('keydown', handleCtxNewName);

    // Bind modal keyboard navigation
    document.addEventListener('keydown', handleKeyboard);

    // Initialize resize handle for unsorted section
    initResizeHandle();

    // Initial load
    loadClusters();
});

/**
 * Bind slider input events to update display values
 */
function bindSliders() {
    const epsSlider = document.getElementById('eps');
    const epsVal = document.getElementById('eps-val');
    const minSlider = document.getElementById('min-samples');
    const minVal = document.getElementById('min-samples-val');

    epsSlider.addEventListener('input', () => {
        epsVal.textContent = epsSlider.value;
    });

    minSlider.addEventListener('input', () => {
        minVal.textContent = minSlider.value;
    });
}

// ============================================================================
// API FUNCTIONS
// ============================================================================

/**
 * Load clusters from the API
 */
async function loadClusters() {
    const previewSize = document.getElementById('preview-size').value || 24;
    const perPage = document.getElementById('per-page').value || 20;
    const searchNamed = document.getElementById('search-verified').value || '';
    const searchUnnamed = document.getElementById('search-workbench').value || '';

    const params = new URLSearchParams({
        page_named: State.pageNamed,
        page_unnamed: State.pageUnnamed,
        per_page: perPage,
        preview_size: previewSize,
        search_named: searchNamed,
        search_unnamed: searchUnnamed,
        only_suggestions: State.filterSuggestions
    });

    try {
        const res = await fetch(`/api/clusters?${params}`);
        const data = await res.json();

        // Update state
        State.totalPagesNamed = data.pagination.named.total;
        State.totalPagesUnnamed = data.pagination.unnamed.total;

        // Render columns
        renderVerifiedColumn(data.named_clusters, data.pagination.named);
        renderWorkbenchColumn(data.unnamed_clusters, data.pagination.unnamed);
        renderUnsortedSection(data.unsorted_faces, data.unsorted_total);

        // Update URL
        updateURL();
    } catch (err) {
        console.error('Failed to load clusters:', err);
        showStatus('Failed to load clusters', 'error');
    }
}

/**
 * Run DBSCAN clustering
 */
async function runClustering() {
    const btn = document.getElementById('cluster-btn');
    const eps = document.getElementById('eps').value;
    const minSamples = document.getElementById('min-samples').value;

    btn.disabled = true;
    btn.classList.add('loading');
    btn.innerHTML = '<span class="btn-icon">⏳</span> Clustering...';

    try {
        const res = await fetch('/api/cluster', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ eps: parseFloat(eps), min_samples: parseInt(minSamples) })
        });

        const data = await res.json();

        if (data.status === 'ok') {
            showStatus(`Created ${data.clusters} clusters, ${data.noise} unsorted`, 'success');
            // Reset to page 1 and reload
            State.pageNamed = 1;
            State.pageUnnamed = 1;
            await loadClusters();
        } else {
            showStatus(data.message || 'Clustering failed', 'error');
        }
    } catch (err) {
        console.error('Clustering error:', err);
        showStatus('Clustering failed: ' + err.message, 'error');
    } finally {
        btn.disabled = false;
        btn.classList.remove('loading');
        btn.innerHTML = '<span class="btn-icon">⚙️</span> Re-Cluster Photos';
    }
}

/**
 * Rename a cluster
 */
async function renameCluster(clusterId, inputEl) {
    const name = inputEl.value.trim();
    if (!name) return;

    const btn = inputEl.nextElementSibling;
    const originalText = btn.textContent;
    btn.textContent = '...';
    btn.disabled = true;

    try {
        const res = await fetch('/api/rename', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id: clusterId, name: name })
        });

        const data = await res.json();

        if (data.status === 'ok') {
            btn.textContent = '✓';
            btn.style.background = 'var(--accent-green)';
            setTimeout(() => loadClusters(), 500);
        } else {
            throw new Error(data.error);
        }
    } catch (err) {
        console.error('Rename error:', err);
        btn.textContent = '✗';
        btn.style.background = 'var(--accent-red)';
        setTimeout(() => {
            btn.textContent = originalText;
            btn.style.background = '';
            btn.disabled = false;
        }, 1500);
    }
}

/**
 * Move a face to another cluster
 */
async function moveFace(faceId, targetClusterId, targetName) {
    try {
        const body = targetName
            ? { target_name: targetName }
            : { target_cluster_id: targetClusterId };

        const res = await fetch(`/api/face/${faceId}/move`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });

        const data = await res.json();

        if (data.status === 'ok') {
            // Remove the face element from DOM
            if (State.contextFaceEl) {
                State.contextFaceEl.remove();
            }
            // Optionally reload to update counts
            loadClusters();
        } else {
            throw new Error(data.error);
        }
    } catch (err) {
        console.error('Move error:', err);
        showStatus('Failed to move face', 'error');
    }

    hideContextMenu();
}

/**
 * Remove a face (mark as unsorted)
 */
async function removeFace(faceId, el) {
    try {
        const res = await fetch(`/api/face/${faceId}/remove`, {
            method: 'POST'
        });

        const data = await res.json();

        if (data.status === 'ok') {
            el.closest('.face-thumb').remove();
        }
    } catch (err) {
        console.error('Remove error:', err);
    }
}

/**
 * Load all unsorted faces
 */
async function loadAllUnsorted() {
    const btn = document.getElementById('load-all-unsorted');
    btn.textContent = 'Loading...';
    btn.disabled = true;

    try {
        const res = await fetch('/api/unsorted/all');
        const data = await res.json();

        const container = document.getElementById('unsorted-faces');
        container.innerHTML = '';

        data.faces.forEach(face => {
            container.appendChild(createFaceThumb(face, data.faces, data.faces.indexOf(face)));
        });

        btn.classList.add('hidden');
    } catch (err) {
        console.error('Load unsorted error:', err);
        btn.textContent = 'Error - Retry';
        btn.disabled = false;
    }
}

/**
 * Load all unsorted faces silently (for automatic loading)
 */
async function loadAllUnsortedSilent() {
    try {
        const res = await fetch('/api/unsorted/all');
        const data = await res.json();

        const container = document.getElementById('unsorted-faces');
        container.innerHTML = '';

        data.faces.forEach((face, i) => {
            container.appendChild(createFaceThumb(face, data.faces, i));
        });

        // Update count badge
        document.getElementById('unsorted-count').textContent = data.faces.length;
    } catch (err) {
        console.error('Load unsorted error:', err);
    }
}

// ============================================================================
// RENDER FUNCTIONS
// ============================================================================

/**
 * Render the Verified Library column
 */
function renderVerifiedColumn(clusters, pagination) {
    const list = document.getElementById('list-verified');
    const countBadge = document.getElementById('count-verified');
    const paginationEl = document.getElementById('pagination-verified');

    countBadge.textContent = pagination.count;

    // Render pagination
    paginationEl.innerHTML = renderPagination(
        pagination.page,
        pagination.total,
        (p) => { State.pageNamed = p; loadClusters(); }
    );

    // Render clusters
    list.innerHTML = '';
    clusters.forEach(cluster => {
        list.appendChild(createClusterCard(cluster, true));
    });

    if (clusters.length === 0) {
        list.innerHTML = '<div class="empty-state">No verified people yet</div>';
    }
}

/**
 * Render the Workbench column
 */
function renderWorkbenchColumn(clusters, pagination) {
    const list = document.getElementById('list-workbench');
    const countBadge = document.getElementById('count-workbench');
    const paginationEl = document.getElementById('pagination-workbench');

    countBadge.textContent = pagination.count;

    // Render pagination
    paginationEl.innerHTML = renderPagination(
        pagination.page,
        pagination.total,
        (p) => { State.pageUnnamed = p; loadClusters(); }
    );

    // Render clusters
    list.innerHTML = '';
    clusters.forEach(cluster => {
        list.appendChild(createClusterCard(cluster, false));
    });

    if (clusters.length === 0) {
        list.innerHTML = '<div class="empty-state">No suggestions - run clustering</div>';
    }
}

/**
 * Render the unsorted faces section
 */
function renderUnsortedSection(faces, total) {
    const section = document.getElementById('unsorted-section');
    const container = document.getElementById('unsorted-faces');
    const countBadge = document.getElementById('unsorted-count');
    const loadMoreBtn = document.getElementById('load-all-unsorted');

    // Always hide the load more button - we'll load all automatically
    loadMoreBtn.classList.add('hidden');

    if (total === 0) {
        section.classList.add('hidden');
        return;
    }

    section.classList.remove('hidden');
    countBadge.textContent = total;

    // If we have more faces than shown, load all of them
    if (total > faces.length) {
        // Trigger automatic load of all unsorted faces
        loadAllUnsortedSilent();
    } else {
        // We have all faces, just render them
        container.innerHTML = '';
        faces.forEach((face, i) => {
            container.appendChild(createFaceThumb(face, faces, i));
        });
    }
}

/**
 * Create a cluster card element
 */
function createClusterCard(cluster, isVerified) {
    const card = document.createElement('div');
    card.className = 'cluster-card';
    card.dataset.clusterId = cluster.id;

    // Header with name input
    const header = document.createElement('div');
    header.className = 'cluster-header';

    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'cluster-name-input';

    if (cluster.suggested_name) {
        // Pre-fill suggestion (remove trailing ?) to allow 1-click save
        input.value = cluster.suggested_name.replace(/\?$/, '');
        input.style.color = 'var(--accent-blue)'; // Visual cue
    } else {
        input.value = cluster.name || '';
    }

    input.placeholder = 'Name this person...';

    // Autocomplete events
    input.addEventListener('focus', () => showSuggestions(input, cluster.id));
    input.addEventListener('input', () => showSuggestions(input, cluster.id));
    input.addEventListener('blur', () => setTimeout(hideSuggestions, 200));

    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            renameCluster(cluster.id, input);
            hideSuggestions();
        }
    });

    const saveBtn = document.createElement('button');
    saveBtn.className = 'btn-save';
    saveBtn.textContent = 'Save';
    saveBtn.addEventListener('click', () => renameCluster(cluster.id, input));

    const count = document.createElement('span');
    count.className = 'cluster-count';
    count.textContent = `${cluster.count} faces`;

    const trashBtn = document.createElement('button');
    trashBtn.className = 'btn-cluster-trash';
    trashBtn.title = 'Dismiss entire cluster';
    trashBtn.innerHTML = '&times;';
    trashBtn.onclick = () => trashCluster(cluster.id, cluster.count);

    header.appendChild(input);
    
    // Render Race Badge
    if (cluster.race) {
        const raceBadge = document.createElement('span');
        raceBadge.className = 'badge'; // Uses native UI badge shape
        raceBadge.style.background = 'rgba(88, 166, 255, 0.1)'; // Soft blue
        raceBadge.style.color = 'var(--accent-blue)';
        raceBadge.style.border = '1px solid rgba(88, 166, 255, 0.2)';
        raceBadge.style.whiteSpace = 'nowrap';
        raceBadge.textContent = cluster.race;
        header.appendChild(raceBadge);
    }
    
    header.appendChild(saveBtn);
    header.appendChild(trashBtn);
    header.appendChild(count);

    // Face grid
    const grid = document.createElement('div');
    grid.className = 'face-grid';

    cluster.faces.forEach((face, i) => {
        grid.appendChild(createFaceThumb(face, cluster.faces, i));
    });

    // Show more button if there are more faces
    if (cluster.count > cluster.faces.length) {
        const showMore = document.createElement('button');
        showMore.className = 'show-more-btn';
        showMore.textContent = `Show ${cluster.count - cluster.faces.length} more`;
        showMore.addEventListener('click', () => expandCluster(cluster.id, grid, showMore));
        card.appendChild(header);
        card.appendChild(grid);
        card.appendChild(showMore);
    } else {
        card.appendChild(header);
        card.appendChild(grid);
    }

    return card;
}

/**
 * Create a face thumbnail element
 */
function createFaceThumb(face, allFaces, index) {
    const thumb = document.createElement('div');
    thumb.className = 'face-thumb';
    thumb.dataset.faceId = face.id;

    const img = document.createElement('img');
    img.src = `/api/thumbnail/${face.id}`;
    img.alt = 'Face';
    img.loading = 'lazy';

    // Click handling: Ctrl+click for multi-select, Shift+click for range, regular click for modal
    img.addEventListener('click', (e) => {
        if (e.shiftKey) {
            if (State.lastSelectedFaceId !== null) {
                // Range select: select all faces between last selected and this one
                selectFaceRange(State.lastSelectedFaceId, face.id);
            } else {
                // No anchor yet, just select this face as anchor
                toggleFaceSelection(face.id, thumb);
            }
        } else if (e.ctrlKey || e.metaKey) {
            // Toggle selection
            toggleFaceSelection(face.id, thumb);
        } else {
            // Clear selection and open modal
            clearFaceSelection();
            openModal(allFaces, index);
        }
    });

    // Right-click for context menu
    thumb.addEventListener('contextmenu', (e) => {
        e.preventDefault();

        // If this face is not selected, clear selection and select only this one
        if (!thumb.classList.contains('selected')) {
            clearFaceSelection();
            toggleFaceSelection(face.id, thumb);
        }

        showContextMenu(e);
    });

    // Remove button
    const removeBtn = document.createElement('button');
    removeBtn.className = 'remove-btn';
    removeBtn.textContent = '×';
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        removeFace(face.id, removeBtn);
    });

    thumb.appendChild(img);
    thumb.appendChild(removeBtn);

    return thumb;
}

/**
 * Expand a cluster to show all faces
 */
async function expandCluster(clusterId, grid, btn) {
    btn.textContent = 'Loading...';
    btn.disabled = true;

    try {
        const res = await fetch(`/api/cluster/${clusterId}/faces`);
        const data = await res.json();

        grid.innerHTML = '';
        data.faces.forEach((face, i) => {
            grid.appendChild(createFaceThumb(face, data.faces, i));
        });

        btn.remove();
    } catch (err) {
        console.error('Expand error:', err);
        btn.textContent = 'Error - Retry';
        btn.disabled = false;
    }
}

/**
 * Render pagination controls
 */
function renderPagination(current, total, onPageChange) {
    if (total <= 1) return '';

    return `
        <button ${current <= 1 ? 'disabled' : ''} onclick="(${onPageChange})(${current - 1})">&lt;</button>
        <span>${current} / ${total}</span>
        <button ${current >= total ? 'disabled' : ''} onclick="(${onPageChange})(${current + 1})">&gt;</button>
    `;
}

// ============================================================================
// MODAL FUNCTIONS
// ============================================================================

/**
 * Open the modal with a specific image
 */
function openModal(faces, index) {
    State.modalFaces = faces;
    State.modalIndex = index;

    const modal = document.getElementById('modal');
    const img = document.getElementById('modal-img');

    modal.classList.remove('hidden');
    updateModalImage();
}

/**
 * Close the modal
 */
function closeModal() {
    document.getElementById('modal').classList.add('hidden');
    State.modalFaces = [];
    State.modalIndex = 0;
}

/**
 * Navigate to previous image in modal
 */
function modalPrev() {
    if (State.modalIndex > 0) {
        State.modalIndex--;
        updateModalImage();
    }
}

/**
 * Navigate to next image in modal
 */
function modalNext() {
    if (State.modalIndex < State.modalFaces.length - 1) {
        State.modalIndex++;
        updateModalImage();
    }
}

/**
 * Update the modal image
 */
function updateModalImage() {
    const face = State.modalFaces[State.modalIndex];
    if (face) {
        // Use the original image, not the thumbnail
        document.getElementById('modal-img').src = `/api/image/${face.id}`;
        // Show path
        document.getElementById('modal-path').textContent = face.path || '';
    }
}

// ============================================================================
// CONTEXT MENU FUNCTIONS
// ============================================================================

/**
 * Show the context menu
 */
async function showContextMenu(e) {
    const menu = document.getElementById('context-menu');
    const list = document.getElementById('ctx-people-list');
    const input = document.getElementById('ctx-new-name');
    const searchInput = document.getElementById('ctx-search');
    const header = menu.querySelector('.ctx-header');

    // Update header to show selection count
    const count = State.selectedFaces.length;
    header.textContent = count > 1 ? `Move ${count} faces to...` : 'Move to...';

    // Show menu first (off-screen) to get its dimensions
    menu.style.left = '-9999px';
    menu.style.top = '-9999px';
    menu.classList.remove('hidden');

    // Wait a tick for the menu to render
    await new Promise(r => setTimeout(r, 10));

    // Now calculate position
    const rect = menu.getBoundingClientRect();
    let x = e.clientX;
    let y = e.clientY;

    // Adjust if near right edge
    if (x + rect.width > window.innerWidth) {
        x = window.innerWidth - rect.width - 10;
    }

    // Adjust if near bottom edge - show ABOVE the cursor if not enough space
    if (y + rect.height > window.innerHeight) {
        y = Math.max(10, window.innerHeight - rect.height - 10);
    }

    // Ensure minimum bounds
    x = Math.max(10, x);
    y = Math.max(10, y);

    menu.style.left = x + 'px';
    menu.style.top = y + 'px';

    // Clear inputs
    input.value = '';
    searchInput.value = '';

    // Load people list
    list.innerHTML = '<div class="ctx-item">Loading...</div>';

    try {
        const res = await fetch('/api/people');
        const data = await res.json();
        State.peopleData = data.people;

        list.innerHTML = '';

        // If single unsorted face selected, show AI matches first
        if (State.selectedFaces.length === 1) {
            const faceObj = State.selectedFaces[0];
            console.log('Selected face object:', faceObj);
            const faceId = faceObj.id || faceObj;  // Handle both {id, el} and just id
            console.log('Face ID:', faceId);
            try {
                const matchRes = await fetch(`/api/face/${faceId}/matches`);
                const matchData = await matchRes.json();
                console.log('Match data:', matchData);

                if (matchData.matches && matchData.matches.length > 0) {
                    // Add AI suggestions header
                    const aiHeader = document.createElement('div');
                    aiHeader.className = 'ctx-section-header';
                    aiHeader.innerHTML = '✨ AI Suggestions';
                    list.appendChild(aiHeader);

                    matchData.matches.forEach(match => {
                        const item = document.createElement('div');
                        item.className = 'ctx-item ctx-match';

                        // Color based on confidence
                        let badgeClass = 'badge-low';
                        if (match.confidence >= 70) badgeClass = 'badge-high';
                        else if (match.confidence >= 40) badgeClass = 'badge-medium';

                        item.innerHTML = `
                            <span class="match-name">${match.name}</span>
                            <span class="match-confidence ${badgeClass}">${match.confidence}%</span>
                        `;
                        item.dataset.clusterId = match.cluster_id;
                        item.dataset.name = match.name;
                        list.appendChild(item);
                    });

                    // Separator
                    const sep = document.createElement('div');
                    sep.className = 'ctx-separator';
                    list.appendChild(sep);

                    // All people header
                    const allHeader = document.createElement('div');
                    allHeader.className = 'ctx-section-header';
                    allHeader.innerHTML = 'All People';
                    list.appendChild(allHeader);
                }
            } catch (matchErr) {
                console.error('Failed to load matches:', matchErr);
            }
        }

        // Add all people
        data.people.forEach(person => {
            const item = document.createElement('div');
            item.className = 'ctx-item';
            item.textContent = person.name;
            item.dataset.clusterId = person.cluster_id;
            item.dataset.name = person.name;
            list.appendChild(item);
        });

        // Use event delegation for clicks
        list.onclick = (e) => {
            const item = e.target.closest('.ctx-item');
            if (item && item.dataset.clusterId) {
                moveSelectedFaces(parseInt(item.dataset.clusterId), null);
            }
        };

        if (data.people.length === 0) {
            list.innerHTML = '<div class="ctx-item" style="color: var(--text-muted)">No people yet</div>';
        }
    } catch (err) {
        list.innerHTML = '<div class="ctx-item" style="color: var(--accent-red)">Failed to load</div>';
    }
}

/**
 * Hide the context menu
 */
function hideContextMenu() {
    document.getElementById('context-menu').classList.add('hidden');
}

/**
 * Trash selected faces
 */
function trashSelectedFaces() {
    const count = State.selectedFaces.length;

    showDialog({
        title: 'Dismiss Faces',
        message: `Dismiss ${count} selected face(s)?\nThey will be hidden from view.`,
        isConfirm: true,
        onConfirm: async () => {
            hideContextMenu();

            let successCount = 0;

            // Process in parallel
            const promises = State.selectedFaces.map(async (faceObj) => {
                const faceId = faceObj.id || faceObj;
                try {
                    const res = await fetch(`/api/face/${faceId}/trash`, { method: 'POST' });
                    if (res.ok) {
                        successCount++;
                        const el = document.querySelector(`.face-thumb[data-face-id="${faceId}"]`);
                        if (el) {
                            el.remove();
                        }
                    }
                } catch (err) {
                    console.error(err);
                }
            });

            await Promise.all(promises);

            showStatus(`Dismissed ${successCount} faces`, 'success');
            State.selectedFaces = [];
        }
    });
}

/**
 * Trash an entire cluster
 */
function trashCluster(clusterId, count) {
    showDialog({
        title: 'Dismiss Cluster',
        message: `Dismiss entire cluster (${count} faces)?\nThey will be hidden from view.`,
        isConfirm: true,
        onConfirm: async () => {
            try {
                const res = await fetch(`/api/cluster/${clusterId}/trash`, { method: 'POST' });
                if (res.ok) {
                    // Remove card from UI
                    const card = document.querySelector(`.cluster-card[data-cluster-id="${clusterId}"]`);
                    if (card) {
                        card.remove();
                    }
                    showStatus(`Dismissed cluster`, 'success');
                } else {
                    const data = await res.json();
                    showStatus('Error: ' + (data.error || 'Failed'), 'error');
                }
            } catch (err) {
                console.error(err);
                showStatus('Failed to dismiss cluster', 'error');
            }
        }
    });
}

/**
 * Handle new name input in context menu
 */
function handleCtxNewName(e) {
    if (e.key === 'Enter') {
        const name = e.target.value.trim();
        if (name && State.selectedFaces.length > 0) {
            moveSelectedFaces(null, name);
        }
    } else if (e.key === 'Escape') {
        hideContextMenu();
    }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Handle search input with debounce
 */
function handleSearch() {
    clearTimeout(State.searchTimeout);
    State.searchTimeout = setTimeout(() => {
        State.pageNamed = 1;
        State.pageUnnamed = 1;
        loadClusters();
    }, 300);
}

/**
 * Toggle the suggestions-only filter
 */
function toggleSuggestionFilter() {
    State.filterSuggestions = !State.filterSuggestions;

    // Update button UI
    const btn = document.getElementById('filter-suggestions');
    if (State.filterSuggestions) {
        btn.classList.add('active');
    } else {
        btn.classList.remove('active');
    }

    // Reset page and reload
    State.pageUnnamed = 1;
    loadClusters();
}

/**
 * Handle keyboard events
 */
function handleKeyboard(e) {
    const modal = document.getElementById('modal');

    if (!modal.classList.contains('hidden')) {
        if (e.key === 'ArrowLeft') {
            modalPrev();
        } else if (e.key === 'ArrowRight') {
            modalNext();
        } else if (e.key === 'Escape') {
            closeModal();
        }
    }
}

/**
 * Show status message
 */
function showStatus(message, type = 'info') {
    const statusBox = document.getElementById('cluster-status');
    statusBox.textContent = message;
    statusBox.className = `status-box ${type}`;
    statusBox.classList.remove('hidden');

    // Auto-hide after 5 seconds
    setTimeout(() => {
        statusBox.classList.add('hidden');
    }, 5000);
}

/**
 * Update URL with current state
 */
function updateURL() {
    const params = new URLSearchParams({
        page_named: State.pageNamed,
        page_unnamed: State.pageUnnamed,
        search_named: document.getElementById('search-verified').value || '',
        search_unnamed: document.getElementById('search-workbench').value || ''
    });

    window.history.replaceState({}, '', `?${params}`);
}

// ============================================================================
// MULTI-SELECT FUNCTIONS
// ============================================================================

/**
 * Toggle face selection
 */
function toggleFaceSelection(faceId, el) {
    const index = State.selectedFaces.findIndex(f => f.id === faceId);

    if (index === -1) {
        // Add to selection
        State.selectedFaces.push({ id: faceId, el: el });
        el.classList.add('selected');
        State.lastSelectedFaceId = faceId;
    } else {
        // Remove from selection
        State.selectedFaces.splice(index, 1);
        el.classList.remove('selected');
    }
}

/**
 * Clear all face selections
 */
function clearFaceSelection() {
    State.selectedFaces.forEach(f => {
        if (f.el) f.el.classList.remove('selected');
    });
    State.selectedFaces = [];
    State.lastSelectedFaceId = null;
}

/**
 * Select a range of faces between two face IDs
 */
function selectFaceRange(startId, endId) {
    // Get all visible face thumbnails in order
    const allThumbs = Array.from(document.querySelectorAll('.face-thumb[data-face-id]'));

    // Find indices of start and end
    const startIdx = allThumbs.findIndex(t => parseInt(t.dataset.faceId) === startId);
    const endIdx = allThumbs.findIndex(t => parseInt(t.dataset.faceId) === endId);

    if (startIdx === -1 || endIdx === -1) return;

    // Determine range (handle both directions)
    const minIdx = Math.min(startIdx, endIdx);
    const maxIdx = Math.max(startIdx, endIdx);

    // Select all faces in range
    for (let i = minIdx; i <= maxIdx; i++) {
        const thumb = allThumbs[i];
        const faceId = parseInt(thumb.dataset.faceId);

        // Only add if not already selected
        if (!State.selectedFaces.some(f => f.id === faceId)) {
            State.selectedFaces.push({ id: faceId, el: thumb });
            thumb.classList.add('selected');
        }
    }

    State.lastSelectedFaceId = endId;
}

/**
 * Handle global click (for clearing selection and hiding context menu)
 */
function handleGlobalClick(e) {
    const ctxMenu = document.getElementById('context-menu');

    // If clicking inside the context menu, don't do anything
    if (ctxMenu.contains(e.target)) {
        return;
    }

    // Hide context menu if clicking outside
    hideContextMenu();

    // Clear selection if clicking outside faces (unless Ctrl is held)
    if (!e.ctrlKey && !e.metaKey) {
        const clickedFace = e.target.closest('.face-thumb');
        if (!clickedFace) {
            clearFaceSelection();
        }
    }
}

/**
 * Move all selected faces to a cluster
 */
async function moveSelectedFaces(targetClusterId, targetName) {
    const faceIds = State.selectedFaces.map(f => f.id);

    if (faceIds.length === 0) return;

    showStatus(`Moving ${faceIds.length} face(s)...`, 'info');

    try {
        // Move each face
        for (const faceId of faceIds) {
            const body = targetName
                ? { target_name: targetName }
                : { target_cluster_id: targetClusterId };

            await fetch(`/api/face/${faceId}/move`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
        }

        // Remove elements from DOM
        State.selectedFaces.forEach(f => {
            if (f.el) f.el.remove();
        });

        clearFaceSelection();
        showStatus(`Moved ${faceIds.length} face(s) successfully`, 'success');
        loadClusters();
    } catch (err) {
        console.error('Bulk move error:', err);
        showStatus('Failed to move faces', 'error');
    }

    hideContextMenu();
}

// ============================================================================
// RESIZE HANDLE
// ============================================================================

/**
 * Initialize the resize handle for the unsorted section
 */
function initResizeHandle() {
    const handle = document.getElementById('unsorted-resize-handle');
    const section = document.getElementById('unsorted-section');

    if (!handle || !section) return;

    let startY = 0;
    let startHeight = 0;
    let isDragging = false;

    handle.addEventListener('mousedown', (e) => {
        isDragging = true;
        startY = e.clientY;
        startHeight = section.offsetHeight;
        handle.classList.add('dragging');
        document.body.style.cursor = 'ns-resize';
        document.body.style.userSelect = 'none';
        e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
        if (!isDragging) return;

        // Calculate new height (dragging up increases height)
        const deltaY = startY - e.clientY;
        const newHeight = Math.max(100, Math.min(600, startHeight + deltaY));
        section.style.maxHeight = newHeight + 'px';
    });

    document.addEventListener('mouseup', () => {
        if (isDragging) {
            isDragging = false;
            handle.classList.remove('dragging');
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        }
    });
}

// ============================================================================
// CONTEXT MENU SEARCH
// ============================================================================

/**
 * Filter the context menu people list based on search input
 */
function filterContextMenu() {
    const searchInput = document.getElementById('ctx-search');
    const query = searchInput.value.toLowerCase().trim();
    const list = document.getElementById('ctx-people-list');

    // Filter existing items
    const items = list.querySelectorAll('.ctx-item');
    items.forEach(item => {
        const name = (item.dataset.name || item.textContent).toLowerCase();
        if (name.includes(query)) {
            item.style.display = '';
        } else {
            item.style.display = 'none';
        }
    });
}

// ============================================================================
// MANAGE PEOPLE (SIDEBAR)
// ============================================================================

/**
 * Load people into the sidebar manage list
 */
async function loadManageList() {
    const list = document.getElementById('manage-list');
    if (!list) return;

    list.innerHTML = '<div class="manage-item">Loading...</div>';

    try {
        const res = await fetch('/api/people');
        const data = await res.json();

        list.innerHTML = '';
        data.people.forEach(person => {
            const item = document.createElement('div');
            item.className = 'manage-item';
            item.dataset.name = person.name.toLowerCase();

            const nameSpan = document.createElement('span');
            // Show name and count
            nameSpan.innerHTML = `${person.name} <span class="text-muted" style="font-size:0.8em">(${person.count})</span>`;

            const actions = document.createElement('div');
            actions.className = 'manage-actions';

            // Edit Button
            const editBtn = document.createElement('button');
            editBtn.className = 'manage-btn edit';
            editBtn.innerHTML = '✎';
            editBtn.title = 'Rename';
            editBtn.onclick = () => renamePerson(person.cluster_id, person.name);

            // Delete Button
            const delBtn = document.createElement('button');
            delBtn.className = 'manage-btn delete';
            delBtn.innerHTML = '🗑';
            delBtn.title = 'Delete Name';
            delBtn.onclick = () => deletePerson(person.cluster_id, person.name);

            actions.appendChild(editBtn);
            actions.appendChild(delBtn);

            item.appendChild(nameSpan);
            item.appendChild(actions);
            list.appendChild(item);
        });

        if (data.people.length === 0) {
            list.innerHTML = '<div class="manage-item">No named people</div>';
        }
    } catch (err) {
        console.error('Failed to load manage list:', err);
        list.innerHTML = '<div class="manage-item" style="color:var(--accent-red)">Error loading</div>';
    }
}

/**
 * Filter the manage list
 */
function filterManageList() {
    const query = document.getElementById('manage-search').value.toLowerCase().trim();
    const items = document.querySelectorAll('.manage-item');

    items.forEach(item => {
        if (!item.dataset.name) return;
        if (item.dataset.name.includes(query)) {
            item.style.display = 'flex';
        } else {
            item.style.display = 'none';
        }
    });
}

/**
 * Delete a person (un-verify / remove name)
 */
async function deletePerson(clusterId, name) {
    showDialog({
        title: 'Delete Person',
        message: `Are you sure you want to delete "${name}"?\nAssociated faces will become unnamed/unsorted.`,
        isConfirm: true,
        onConfirm: async () => {
            try {
                const res = await fetch(`/api/person/${clusterId}`, { method: 'DELETE' });
                const data = await res.json();

                if (res.ok) {
                    showStatus(`Deleted "${name}"`, 'success');
                    loadManageList(); // Refresh list
                    loadClusters();   // Refresh main view
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (err) {
                alert('Failed to delete: ' + err);
            }
        }
    });
}

/**
 * Rename a person
 */
async function renamePerson(clusterId, currentName) {
    showDialog({
        title: 'Rename Person',
        message: `Enter a new name for "${currentName}":`,
        isInput: true,
        inputValue: currentName,
        onConfirm: async (newName) => {
            if (!newName || newName === currentName) return;

            try {
                const res = await fetch('/api/rename', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        id: clusterId,
                        name: newName
                    })
                });

                if (res.ok) {
                    showStatus(`Renamed to "${newName}"`, 'success');
                    loadManageList();
                    loadClusters();
                } else {
                    const data = await res.json();
                    alert('Error: ' + (data.error || 'Failed to rename'));
                }
            } catch (err) {
                alert('Rename failed: ' + err);
            }
        }
    });
}

/**
 * Show autocomplete suggestions
 */
function showSuggestions(input, clusterId) {
    clearTimeout(State.autocompleteTimeout);
    State.activeInput = input;

    // Delay slightly to debounce
    State.autocompleteTimeout = setTimeout(async () => {
        const query = input.value.trim().toLowerCase();
        const dropdown = document.getElementById('autocomplete-dropdown');
        dropdown.innerHTML = '';

        let items = [];

        if (query.length === 0) {
            // Empty input: Fetch AI matches
            dropdown.innerHTML = '<div class="ac-item" style="color:var(--text-muted)">Loading suggestions...</div>';
            setPosition(input, dropdown);
            dropdown.classList.remove('hidden');

            try {
                const res = await fetch(`/api/cluster/${clusterId}/matches`);
                const data = await res.json();

                if (data.matches && data.matches.length > 0) {
                    items = data.matches.map(m => ({
                        name: m.name,
                        type: 'match',
                        confidence: m.confidence
                    }));
                } else {
                    // fall through to empty
                    items = [];
                }
            } catch (err) {
                console.error('Failed to fetch matches', err);
            }
        } else {
            // Text input: Filter existing people
            if (State.peopleData.length === 0) {
                // Ensure we have data
                try {
                    const res = await fetch('/api/people');
                    const data = await res.json();
                    State.peopleData = data.people;
                } catch (e) { }
            }

            items = State.peopleData
                .filter(p => p.name.toLowerCase().includes(query))
                .slice(0, 10) // Limit to 10
                .map(p => ({ name: p.name, type: 'person' }));
        }

        // Render items
        dropdown.innerHTML = '';

        if (items.length > 0) {
            if (query.length === 0) {
                const header = document.createElement('div');
                header.className = 'ctx-section-header';
                header.textContent = '✨ AI Suggestions';
                dropdown.appendChild(header);
            }

            items.forEach(item => {
                const el = document.createElement('div');
                el.className = 'ac-item';

                if (item.type === 'match') {
                    // Color based on confidence
                    let badgeClass = 'badge-low';
                    if (item.confidence >= 70) badgeClass = 'badge-high';
                    else if (item.confidence >= 40) badgeClass = 'badge-medium';

                    el.innerHTML = `
                        <span>${item.name}</span>
                        <span class="match-confidence ${badgeClass}">${item.confidence}%</span>
                    `;
                } else {
                    el.textContent = item.name;
                }

                el.addEventListener('mousedown', (e) => { // mousedown fires before blur
                    e.preventDefault(); // Prevent blur
                    input.value = item.name;
                    // Auto-save? Maybe just fill for now to let user confirm
                    // renameCluster(clusterId, input); // Uncomment to auto-save
                    hideSuggestions();
                    input.focus(); // Keep focus
                });

                dropdown.appendChild(el);
            });

            setPosition(input, dropdown);
            dropdown.classList.remove('hidden');
        } else {
            if (query.length > 0) dropdown.classList.add('hidden');
            else dropdown.classList.remove('hidden'); // Keep showing even if empty to say "No suggestions" if we want
            if (items.length === 0 && query.length === 0) dropdown.innerHTML = '<div class="ac-item" style="color:var(--text-muted)">No suggestions</div>';
        }

    }, 150);
}

function setPosition(input, dropdown) {
    const rect = input.getBoundingClientRect();
    dropdown.style.left = rect.left + 'px';
    dropdown.style.top = (rect.bottom + 5) + 'px';
    dropdown.style.width = Math.max(200, rect.width) + 'px';
}

function hideSuggestions() {
    clearTimeout(State.autocompleteTimeout);
    setTimeout(() => {
        document.getElementById('autocomplete-dropdown').classList.add('hidden');
    }, 100);
}

/**
 * Show custom dialog
 */
function showDialog({ title, message, isInput = false, isConfirm = false, inputValue = '', onConfirm }) {
    const overlay = document.getElementById('dialog-overlay');
    const titleEl = document.getElementById('dialog-title');
    const msgEl = document.getElementById('dialog-message');
    const inputEl = document.getElementById('dialog-input');
    const confirmBtn = document.getElementById('dialog-confirm');
    const cancelBtn = document.getElementById('dialog-cancel');

    // Setup content
    titleEl.textContent = title;
    msgEl.textContent = message; // Note: simplified newline handling? Pre-wrap css would be good
    msgEl.style.whiteSpace = 'pre-wrap';

    // Input state
    if (isInput) {
        inputEl.classList.remove('hidden');
        inputEl.value = inputValue;
        setTimeout(() => inputEl.focus(), 100);
    } else {
        inputEl.classList.add('hidden');
    }

    // Show overlay
    overlay.classList.remove('hidden');

    // Button Handlers
    const cleanup = () => {
        overlay.classList.add('hidden');
        confirmBtn.onclick = null;
        cancelBtn.onclick = null;
        inputEl.onkeydown = null;
    };

    confirmBtn.onclick = () => {
        const value = isInput ? inputEl.value.trim() : true;
        cleanup();
        if (onConfirm) onConfirm(value);
    };

    cancelBtn.onclick = cleanup;

    // Close on overlay click
    overlay.onclick = (e) => {
        if (e.target === overlay) cleanup();
    };

    // Enter key support
    inputEl.onkeydown = (e) => {
        if (e.key === 'Enter') confirmBtn.click();
        if (e.key === 'Escape') cleanup();
    };
}

// Make functions available globally for inline handlers
window.loadClusters = loadClusters;
window.handleSearch = handleSearch;
window.closeModal = closeModal;
window.modalPrev = modalPrev;
window.modalNext = modalNext;
window.loadAllUnsorted = loadAllUnsorted;
window.filterContextMenu = filterContextMenu;
window.filterManageList = filterManageList;

// Load manage list on init
document.addEventListener('DOMContentLoaded', loadManageList);
