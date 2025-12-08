Estimator Overview Implementation (Issue #248)
==============================================

This document summarizes the implementation of an interactive estimator overview page for skpro, similar to sktime's estimator overview.

## Overview

The estimator overview provides a searchable, filterable table of all estimators, distributions, and other objects in skpro, accessible via the documentation website.

## Implementation Details

### 1. Core Components

#### docs/source/estimator_overview.rst
- Main documentation page with embedded HTML/JavaScript
- Interactive table with filters by:
  - Estimator type (dropdown)
  - Search terms (text input)
  - Tags (multi-select)
- Real-time statistics showing matching estimators
- Beautiful CSS styling with:
  - Color-coded tags
  - Hover effects
  - Sticky header for scrolling
  - Responsive design

#### build_tools/generate_estimator_data.py
- Python script to collect estimator metadata
- Uses skpro.registry.all_objects() to retrieve:
  - Estimator names
  - Object types
  - Module paths
  - Key tags
- Generates JSON-formatted data
- Can be run standalone or integrated into documentation build

#### docs/source/conf.py (modified)
- Added Sphinx setup function to auto-generate estimator data during build
- Integrates with "config-inited" event
- Creates estimator_data.js file in _static directory
- Error handling for import failures
- Suppresses stdout during data collection

#### docs/source/tags_reference.rst
- New page documenting the tag system
- Explains common tags and their meanings
- Shows how to use tags programmatically
- References the registry module documentation

### 2. Integration Points

#### docs/source/index.rst
- Added estimator_overview to toctree (hidden)
- Added tags_reference to toctree (hidden)
- Added "Estimator Overview" card to home page navigation grid

#### docs/source/api_reference.rst
- Reorganized to include Estimator Overview section first
- Separates overview from detailed API documentation

### 3. JavaScript Features

The embedded JavaScript provides:

- **Dynamic rendering**: Populates table based on window.estimatorData
- **Real-time filtering**: Updates as users type or change filters
- **Tag population**: Automatically discovers and populates available tags
- **Statistics**: Shows count of matching vs total estimators
- **No results handling**: Displays helpful message when no estimators match
- **Accessibility**: Works without external dependencies

### 4. Data Collection Process

During Sphinx build:

1. Import skpro.registry.all_objects()
2. Query all objects with selected tags:
   - object_type
   - estimator_type
   - capability:survival
   - handles_missing_data
   - requires_y
   - handles_multioutput
3. Format data as JSON array
4. Generate estimator_data.js in _static directory
5. Log success with count of estimators

### 5. Features

✓ **Search functionality**: Real-time substring search across estimator names and modules
✓ **Type filtering**: Filter by object type (regressor, distribution, metric)
✓ **Tag filtering**: Multi-select tag filtering with OR logic
✓ **Statistics**: Live count of visible vs total estimators
✓ **Beautiful UI**: Professional styling with colors and interactive elements
✓ **Responsive**: Works on desktop and mobile
✓ **Auto-generated**: Data automatically collected during documentation build
✓ **Programmatic access**: Users can also access data via registry.all_objects()
✓ **Documentation**: Includes reference page explaining tags

### 6. How to Use

#### From Documentation Website
1. Navigate to "Estimator Overview" link in main navigation
2. Use dropdown to filter by type
3. Use search box to find specific estimators
4. Use multi-select to filter by tags (Ctrl+Click to select multiple)
5. View matching estimators in the table
6. Click on estimator name to go to its API documentation

#### From Code
```python
from skpro.registry import all_objects

# Get all estimators with survival capability
survival_regressors = all_objects(
    object_types="regressor_proba",
    filter_tags={"capability:survival": True},
    as_dataframe=True
)

# Search for specific estimator
my_est = all_objects(
    object_types="distribution",
    as_dataframe=True
)
```

### 7. Design Decisions

1. **JavaScript in RST**: Used raw HTML/JavaScript blocks for interactive features
   - Rationale: Simpler than creating custom Sphinx extension
   - No external JavaScript dependencies required

2. **Auto-generation during build**: Data generated at build time
   - Rationale: Ensures data is always current with actual codebase
   - No runtime overhead

3. **Tag-based filtering**: Uses existing tag system from registry
   - Rationale: Consistent with skpro's architecture
   - Enables extensibility as new tags are added

4. **Similar to sktime**: Follows sktime's design patterns
   - Rationale: Familiar to sktime users
   - Consistent ecosystem experience

## Files Modified/Created

**New Files:**
- `docs/source/estimator_overview.rst` - Main overview page
- `docs/source/tags_reference.rst` - Tag documentation
- `build_tools/generate_estimator_data.py` - Data generation script

**Modified Files:**
- `docs/source/conf.py` - Added setup function for data generation
- `docs/source/index.rst` - Added navigation entries
- `docs/source/api_reference.rst` - Reorganized structure

## Testing & Verification

The implementation was verified by:
- Python syntax check on conf.py and generate_estimator_data.py
- Manual review of HTML/JavaScript code
- Verification of ReStructuredText syntax
- Git commit history confirmation

## Future Enhancements

Possible future improvements:
1. Add sorting by column headers
2. Add export functionality (CSV/JSON)
3. Add estimator comparison view
4. Add estimator recommendations based on use case
5. Add performance metrics and benchmarks
6. Integration with estimator selection/composition tools

## Related Issues

- Issue #248: [ENH] add an estimator overview like in `sktime`
- Implements feature request for interactive estimator discovery
- Follows design pattern from sktime project
