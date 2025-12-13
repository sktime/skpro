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

    <div id="estimator-overview-container" style="margin-top: 20px;">
        <div style="margin-bottom: 20px;">
            <label for="estimator-type-select">
                <strong>Filter by Estimator Type:</strong>
            </label>
            <select id="estimator-type-select" style="margin-left: 10px; padding: 5px;">
                <option value="">All Types</option>
                <option value="regressor_proba">Probabilistic Regressor</option>
                <option value="distribution">Distribution</option>
                <option value="metric">Metric</option>
            </select>
        </div>

        <div style="margin-bottom: 20px;">
            <label for="search-input">
                <strong>Search Estimators:</strong>
            </label>
            <input type="text" id="search-input" placeholder="Type to search..." style="margin-left: 10px; padding: 5px; width: 300px;">
        </div>

        <div style="margin-bottom: 20px;">
            <label for="tags-select">
                <strong>Filter by Tags:</strong>
            </label>
            <select id="tags-select" multiple style="margin-left: 10px; padding: 5px; width: 400px; height: 100px;">
            </select>
            <p style="font-size: 0.9em; color: #666; margin-top: 5px;">
                Hold Ctrl (Cmd on Mac) to select multiple tags
            </p>
        </div>

        <div id="table-container" style="overflow-x: auto; margin-top: 20px;">
            <table id="estimator-table" style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr style="background-color: #f5f5f5; border-bottom: 2px solid #ccc;">
                        <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">Name</th>
                        <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">Type</th>
                        <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">Module</th>
                        <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">Key Tags</th>
                    </tr>
                </thead>
                <tbody id="table-body">
                </tbody>
            </table>
        </div>

        <div id="no-results" style="margin-top: 20px; padding: 20px; background-color: #f9f9f9; border: 1px solid #ddd; display: none;">
            <p><strong>No estimators match the current filters.</strong></p>
        </div>
    </div>

.. raw:: html

    <script>
        // Estimator data will be injected here by conf.py during build
        const estimatorData = window.estimatorData || [];
        const tagsData = window.tagsData || {};

        function renderTable() {
            const typeFilter = document.getElementById('estimator-type-select').value;
            const searchTerm = document.getElementById('search-input').value.toLowerCase();
            const selectedTags = Array.from(document.getElementById('tags-select').selectedOptions).map(o => o.value);

            let filtered = estimatorData.filter(est => {
                // Type filter
                if (typeFilter && est.object_type !== typeFilter) return false;

                // Search filter
                if (searchTerm && !est.name.toLowerCase().includes(searchTerm) &&
                    !est.module.toLowerCase().includes(searchTerm)) return false;

                // Tags filter
                if (selectedTags.length > 0) {
                    return selectedTags.some(tag => est.tags.includes(tag));
                }

                return true;
            });

            const tbody = document.getElementById('table-body');
            const noResults = document.getElementById('no-results');

            if (filtered.length === 0) {
                tbody.innerHTML = '';
                noResults.style.display = 'block';
            } else {
                noResults.style.display = 'none';
                tbody.innerHTML = filtered.map(est => `
                    <tr style="border-bottom: 1px solid #ddd;">
                        <td style="padding: 10px; border: 1px solid #ddd;"><strong>${est.name}</strong></td>
                        <td style="padding: 10px; border: 1px solid #ddd;">${est.object_type}</td>
                        <td style="padding: 10px; border: 1px solid #ddd;"><code>${est.module}</code></td>
                        <td style="padding: 10px; border: 1px solid #ddd;">
                            ${est.tags.map(tag => `<span style="background-color: #e8f4f8; padding: 2px 6px; margin: 2px; border-radius: 3px; display: inline-block; font-size: 0.85em;">${tag}</span>`).join('')}
                        </td>
                    </tr>
                `).join('');
            }
        }

        function initializeTags() {
            const tagsSelect = document.getElementById('tags-select');
            const uniqueTags = new Set();

            estimatorData.forEach(est => {
                est.tags.forEach(tag => uniqueTags.add(tag));
            });

            Array.from(uniqueTags).sort().forEach(tag => {
                const option = document.createElement('option');
                option.value = tag;
                option.textContent = tag;
                tagsSelect.appendChild(option);
            });
        }

        // Event listeners
        document.getElementById('estimator-type-select').addEventListener('change', renderTable);
        document.getElementById('search-input').addEventListener('input', renderTable);
        document.getElementById('tags-select').addEventListener('change', renderTable);

        // Initialize on page load
        document.addEventListener('DOMContentLoaded', function() {
            if (estimatorData.length > 0) {
                initializeTags();
                renderTable();
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
