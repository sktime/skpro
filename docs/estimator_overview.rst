.. _estimator_overview:

Estimator Overview
==================

Use the below search grid to find estimators by property.

* type into the search box to subset cards by substring search
* choose a type (distribution, regressor, survival, metric, ...) in the dropdown
* if type is selected, check object tags to display in cards
* for explanation of tags, see the :ref:`tags reference <tags_reference>`

.. raw:: html

    <style>
    .bd-article-container { max-width: 100em !important; }
    .bd-sidebar-secondary { display: none; }
    .top-container { display: flex; justify-content: space-between; align-items: center; width: 100%; padding: 10px; border: 1px solid #ccc; margin-bottom: 10px; box-sizing: border-box; }
    #dropdownContainer { flex: 1; }
    #checkboxContainer { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; width: 100%; padding: 10px; border: 1px solid #ccc; margin-bottom: 10px; box-sizing: border-box; }
    #checkboxContainer input[type="checkbox"] { margin-right: 5px; }
    #checkboxContainer label { white-space: nowrap; color: black; text-decoration: none; cursor: default; }
    .grid-container { width: 100%; }
    #cardContainer { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }
    .card { border: 1px solid #ccc; padding: 15px; background-color: #f9f9f9; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .card:hover { background-color: #e9e9e9; transform: translateY(-2px); transition: all 0.2s; }
    .card h3 { margin: 0 0 10px 0; font-size: 1.2em; color: #333; }
    .card p { margin: 5px 0; font-size: 0.9em; }
    .card .tags { margin-top: 10px; }
    .card .tag { display: inline-block; background-color: #e0e0e0; padding: 2px 6px; margin: 2px; border-radius: 4px; font-size: 0.8em; }
    </style>

    <div class="top-container">
        <input type="text" id="searchInput" placeholder="Search the cards ..." />
        <div id="dropdownContainer">
            <select id="filterOptions">
                <option value="all" selected>ALL</option>
                <option value="distribution">Distribution</option>
                <option value="regressor">Regressor</option>
                <option value="survival">Survival</option>
                <option value="metric">Metric</option>
            </select>
        </div>
    </div>

    <div id="checkboxContainer"></div>

    <div class="grid-container">
        <div id="cardContainer"></div>
    </div>
