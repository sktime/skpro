import os
import re

import pytest


def test_readme_examples():
    """Test that all Python code blocks in README.md execute without error."""

    # Read README.md from repo root
    readme_path = os.path.join(os.path.dirname(__file__), "../..", "README.md")
    with open(readme_path) as f:
        readme = f.read()

    # Extract all ```python ... ``` blocks
    pattern = r"```python(.*?)```"
    code_blocks = re.findall(pattern, readme, re.DOTALL)

    assert len(code_blocks) > 0, "No Python code blocks found in README"

    # Run all blocks sequentially in shared namespace
    namespace = {}
    for i, block in enumerate(code_blocks):
        try:
            exec(block.strip(), namespace)
        except Exception as e:
            pytest.fail(f"README code block {i+1} failed:\n{block}\nError: {e}")
