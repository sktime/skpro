.. _estimator_overview:

===================
Estimator Overview
===================

Use the below search table to find estimators and distributions by property.

**Features:**

* type into the search box to subset rows by substring search
* choose a type (regressor, distribution, …) in the dropdown
* if type is selected, check object tags to display in table
* for explanation of tags, see the :ref:`tags reference <tags_reference>`

.. raw:: html

    <script src="_static/estimator_data.js"></script>

    <style>
        .estimator-box {
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 12px;
            margin-bottom: 12px;
            background-color: #fff;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            transition: all 0.2s ease;
        }
        .estimator-box:hover {
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
            border-color: #0066cc;
        }
        .estimator-name {
            font-weight: bold;
            font-size: 1.05em;
            color: #0066cc;
            margin-bottom: 6px;
        }
        .estimator-name a {
            color: inherit;
            text-decoration: none;
        }
        .estimator-name a:hover {
            text-decoration: underline;
        }
        .estimator-info {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 6px;
        }
        .estimator-module {
            font-family: monospace;
            background-color: #f5f5f5;
            padding: 2px 4px;
            border-radius: 2px;
        }
        .tag {
            display: inline-block;
            background-color: #e8f4f8;
            border: 1px solid #b3dbe0;
            padding: 3px 8px;
            margin: 2px 4px 2px 0;
            border-radius: 3px;
            font-size: 0.85em;
            color: #333;
        }
        .tag-label {
            font-weight: 500;
            color: #0066cc;
        }
        .tag-label a {
            color: inherit;
            text-decoration: none;
        }
        .tag-label a:hover {
            text-decoration: underline;
        }
        .tag-value {
            color: #555;
        }
        .estimator-tags {
            margin-top: 8px;
            line-height: 1.6;
        }
        .no-results {
            padding: 20px;
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 4px;
            text-align: center;
            color: #666;
        }
        .filters-section {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 20px;
            border: 1px solid #ddd;
        }
        .filter-group {
            margin-bottom: 12px;
        }
        .filter-group label {
            display: block;
            font-weight: 500;
            margin-bottom: 4px;
            color: #333;
        }
        .filter-group input,
        .filter-group select {
            width: 100%;
            padding: 8px;
            border: 1px solid #ccc;
            border-radius: 3px;
            font-size: 0.9em;
        }
        .filter-group select {
            min-height: 80px;
        }
        .estimators-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 12px;
            margin-top: 20px;
        }
    </style>

    <div class="filters-section">
        <div class="filter-group">
            <label for="search-input">Search by name:</label>
            <input type="text" id="search-input" placeholder="Type estimator name..." />
        </div>

        <div class="filter-group">
            <label for="estimator-type-select">Filter by type:</label>
            <select id="estimator-type-select">
                <option value="">All types</option>
                <option value="regressor_proba">Probabilistic Regressors</option>
                <option value="distribution">Distributions</option>
                <option value="metric">Metrics</option>
                <option value="survival">Survival Prediction</option>
            </select>
        </div>

        <div class="filter-group">
            <label for="tags-select">Filter by tags (Ctrl+click for multiple):</label>
            <select id="tags-select" multiple></select>
        </div>
    </div>

    <div id="estimators-container" class="estimators-grid"></div>
    <div id="no-results" class="no-results" style="display: none;">No estimators found matching your filters.</div>

    <script>
        // Helper function to format tag values
        function formatTagValue(value) {
            if (typeof value === 'boolean') {
                return value ? 'Yes' : 'No';
            }
            return String(value);
        }

        // Helper function to get estimator documentation URL
        function getEstimatorDocUrl(est) {
            // Map object types to API reference sections
            const typeMap = {
                'regressor_proba': 'regression',
                'distribution': 'distributions',
                'metric': 'metrics',
                'survival': 'survival'
            };

            const section = typeMap[est.object_type] || 'auto_generated';
            const modulePath = est.module.replace(/\./g, '/');

            return `api_reference/${section}/${modulePath}.html`;
        }

        // Helper function to get tag documentation URL
        function getTagDocUrl(tag) {
            return `tags_reference.html#tag_${tag.replace(/[^a-zA-Z0-9_]/g, '_').toLowerCase()}`;
        }

        // Function to get valid tags for a specific object type
        function getValidTagsForType(objectType) {
            const tagsByType = {
                'regressor_proba': ['object_type', 'estimator_type', 'capability:survival', 'handles_missing_data', 'requires_y', 'handles_multioutput'],
                'distribution': ['object_type', 'distr:measuretype', 'capabilities:approx', 'capabilities:exact'],
                'metric': ['object_type', 'metric_type'],
                'survival': ['object_type', 'estimator_type', 'handles_missing_data']
            };

            return tagsByType[objectType] || [];
        }

        // Render estimators based on current filters
        function renderEstimators() {
            const searchTerm = document.getElementById('search-input').value.toLowerCase();
            const typeFilter = document.getElementById('estimator-type-select').value;
            const selectedTags = Array.from(document.getElementById('tags-select').selectedOptions).map(o => o.value);

            // Filter estimators
            const filtered = estimatorData.filter(est => {
                // Search filter
                if (searchTerm && !est.name.toLowerCase().includes(searchTerm)) {
                    return false;
                }

                // Type filter
                if (typeFilter && est.object_type !== typeFilter) {
                    return false;
                }

                // Tag filters
                if (selectedTags.length > 0) {
                    return selectedTags.every(tagFilter => {
                        if (tagFilter.includes('=')) {
                            // String-valued tag: key=value
                            const [key, value] = tagFilter.split('=');
                            return est.tags[key] === value;
                        } else {
                            // Boolean tag
                            return est.tags[tagFilter] === true;
                        }
                    });
                }

                return true;
            });

            // Render results
            const grid = document.getElementById('estimators-container');
            const noResults = document.getElementById('no-results');

            if (filtered.length === 0) {
                grid.innerHTML = '';
                noResults.style.display = 'block';
                return;
            }

            noResults.style.display = 'none';
            grid.innerHTML = filtered.map(est => {
                const tagsHtml = Object.entries(est.tags)
                    .filter(([key]) => key !== 'object_type')
                    .filter(([key, value]) => value === true || (typeof value === 'string' && value))
                    .map(([key, value]) => `
                        <span class="tag">
                            <span class="tag-label"><a href="${getTagDocUrl(key)}">${key}</a>:</span>
                            <span class="tag-value">${formatTagValue(value)}</span>
                        </span>
                    `)
                    .join('');

                return `
                    <div class="estimator-box">
                        <div class="estimator-name"><a href="${getEstimatorDocUrl(est)}">${est.name}</a></div>
                        <div class="estimator-info">
                            <strong>Type:</strong> ${est.object_type}
                        </div>
                        <div class="estimator-info">
                            <strong>Module:</strong> <span class="estimator-module">${est.module}</span>
                        </div>
                        <div class="estimator-tags">
                            ${tagsHtml}
                        </div>
                    </div>
                `;
            }).join('');

            // Save filter state to URL hash
            saveFilterState();
        }

        function saveFilterState() {
            const typeFilter = document.getElementById('estimator-type-select').value;
            const searchTerm = document.getElementById('search-input').value;
            const selectedTags = Array.from(document.getElementById('tags-select').selectedOptions).map(o => o.value);

            const state = {
                type: typeFilter,
                search: searchTerm,
                tags: selectedTags
            };

            window.location.hash = encodeURIComponent(JSON.stringify(state));
        }

        function restoreFilterState() {
            try {
                if (!window.location.hash) return;

                const state = JSON.parse(decodeURIComponent(window.location.hash.substring(1)));

                if (state.type) {
                    document.getElementById('estimator-type-select').value = state.type;
                }
                if (state.search) {
                    document.getElementById('search-input').value = state.search;
                }
                if (state.tags && state.tags.length > 0) {
                    const tagsSelect = document.getElementById('tags-select');
                    Array.from(tagsSelect.options).forEach(option => {
                        option.selected = state.tags.includes(option.value);
                    });
                }
            } catch (e) {
                console.log('Could not restore filter state:', e);
            }
        }

        function initializeTags() {
            const tagsSelect = document.getElementById('tags-select');
            const typeFilter = document.getElementById('estimator-type-select').value;

            // Clear existing options
            tagsSelect.innerHTML = '';

            // Get valid tags for current type
            const validTags = typeFilter ? getValidTagsForType(typeFilter) : null;

            const booleanTags = new Set();
            const stringTagValues = new Map(); // key -> Set of values

            estimatorData.forEach(est => {
                // If type filter is set, only consider matching estimators
                if (typeFilter && est.object_type !== typeFilter) return;

                Object.entries(est.tags).forEach(([key, value]) => {
                    if (key === 'object_type') return;

                    // If type filter is set, only show valid tags for that type
                    if (validTags && !validTags.includes(key)) return;

                    if (typeof value === 'boolean') {
                        booleanTags.add(key);
                    } else if (typeof value === 'string' && value) {
                        if (!stringTagValues.has(key)) stringTagValues.set(key, new Set());
                        stringTagValues.get(key).add(value);
                    }
                });
            });

            // Add boolean tags (filter means tag is True)
            Array.from(booleanTags).sort().forEach(tag => {
                const option = document.createElement('option');
                option.value = tag;
                option.textContent = tag;
                tagsSelect.appendChild(option);
            });

            // Add string-valued tag options as key=value pairs
            Array.from(stringTagValues.keys()).sort().forEach(tag => {
                const values = Array.from(stringTagValues.get(tag)).sort();
                values.forEach(val => {
                    const option = document.createElement('option');
                    option.value = `${tag}=${val}`;
                    option.textContent = `${tag}=${val}`;
                    tagsSelect.appendChild(option);
                });
            });
        }

        // Event listeners
        document.getElementById('estimator-type-select').addEventListener('change', function() {
            initializeTags();  // Rebuild tags for new type
            renderEstimators();
        });
        document.getElementById('search-input').addEventListener('input', renderEstimators);
        document.getElementById('tags-select').addEventListener('change', renderEstimators);

        // Initialize on page load
        document.addEventListener('DOMContentLoaded', function() {
            if (estimatorData.length > 0) {
                restoreFilterState();  // Restore from URL hash if present
                initializeTags();
                renderEstimators();
            } else {
                document.getElementById('no-results').style.display = 'block';
                document.getElementById('no-results').textContent = 'No estimators found. If this is unexpected, ensure the documentation was built with the estimator data generation enabled.';
            }
        });
    </script>

Estimator Reference
===================

For detailed documentation of each estimator, see the API reference:

* :doc:`Probabilistic Regressors <api_reference/regression>`
* :doc:`Distributions <api_reference/distributions>`
* :doc:`Metrics <api_reference/metrics>`
* :doc:`Survival Prediction <api_reference/survival>`

