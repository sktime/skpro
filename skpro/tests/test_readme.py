import os
import re

import pytest


def test_readme_examples():
    """Test that all Python code blocks in README.md execute without error."""

    readme_path = os.path.join(os.path.dirname(__file__), "../..", "README.md")
    with open(readme_path, encoding="utf-8") as f:
        readme = f.read()

    # Only match explicitly tagged ```python blocks (not bash, shell etc)
    pattern = r"```\s*python(.*?)```"
    code_blocks = re.findall(pattern, readme, re.DOTALL)

    assert len(code_blocks) > 0, "No Python code blocks found in README"

    namespace = {}
    for i, block in enumerate(code_blocks):
        # Strip output lines like >>> 32.19 — these are not executable
        cleaned = "\n".join(
            line for line in block.splitlines() if not line.strip().startswith(">>>")
        )
        try:
            exec(cleaned.strip(), namespace)
        except Exception as e:
            pytest.fail(f"README code block {i+1} failed:\n{block}\nError: {e}")
