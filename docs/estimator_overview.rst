.. _estimator_overview:

Estimator Overview
==================

Use the below search table to find estimators by property.

* type into the search box to subset rows by substring search
* choose a type (distribution, regressor, survival, metric, ...) in the dropdown
* if type is selected, check object tags to display in table
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
    .table-container { width: 100%; overflow-x: auto; }
    #tableContainer { float: left; table-layout: fixed; border-collapse: collapse; overflow-x: auto; }
    #tableContainer th, #tableContainer td { border: 2px solid #888; text-align: center; word-break: break-word; width: 15vw; color: black; text-decoration: none; }
    #tableContainer td:hover { background-color: #f5f5f5; }
    </style>

    <div class="top-container">
        <input type="text" id="searchInput" placeholder="Search the table ..." />
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

    <div class="table-container">
        <table id="tableContainer"></table>
    </div>
