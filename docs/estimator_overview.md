# Estimator Overview (skpro)

Use the table below to find skpro estimators by property.

- Type into the search box to subset rows by substring search.
- Choose a type (distribution, regressor, survival, etc.) in the dropdown.
- If a type is selected, check object tags to display in the table.
- For explanation of tags, see the [skpro tags reference](https://skpro.readthedocs.io/en/latest/api_reference/tags.html).

<style>
.bd-article-container { max-width: 100em !important; }
.bd-sidebar-secondary { display: none; }
.top-container { display: flex; justify-content: space-between; align-items: center; width: 100%; padding: 10px; border: 1px solid #ccc; margin-bottom: 10px; box-sizing: border-box; }
#dropdownContainer { flex: 1; }
#checkboxContainer { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; width: 100%; padding: 10px; border: 1px solid #ccc; margin-bottom: 10px; box-sizing: border-box; }
#checkboxContainer input[type="checkbox"] { margin-right: 5px; }
#checkboxContainer label { white-space: nowrap; }
.table-container { width: 100%; overflow-x: auto; }
#tableContainer { float: left; table-layout: fixed; border-collapse: collapse; overflow-x: auto; }
#tableContainer th, #tableContainer td { border: 2px solid #888; text-align: center; word-break: break-word; width: 15vw; }
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
            <!-- Add more skpro estimator types as needed -->
        </select>
    </div>
</div>

<div id="checkboxContainer"></div>

<div class="table-container">
    <table id="tableContainer"></table>
</div>

<script>
// The script from your original example can be reused, just update the estimator types and tag logic for skpro as needed.
// The script expects a JSON file at docs/_static/estimator_overview_db.json and optionally a static HTML at docs/_static/table_all.html
</script>

<!-- Optionally include a static HTML table for "ALL" estimators -->
<!-- ```{include} estimator_overview_table.html ``` -->
