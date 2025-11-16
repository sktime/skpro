"""Generate CONTRIBUTORS.md from .all-contributorsrc."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict
from json import JSONDecodeError
import re

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / ".all-contributorsrc"
OUT = ROOT / "CONTRIBUTORS.md"


def read_all_contributors(path: Path | str) -> List[Dict]:
    """Read .all-contributorsrc and return contributors list.

    Accepts a Path or str. Tries a tolerant fallback to remove trailing commas
    if JSON parsing fails (helps with trailing-comma errors in the file).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except JSONDecodeError as e:
        # Fallback: remove trailing commas before ] or } and retry once
        cleaned = re.sub(r",\s*(\]|})", r"\1", text)
        try:
            data = json.loads(cleaned)
        except JSONDecodeError:
            # raise informative error referencing the file
            raise ValueError(f"Failed to parse JSON in {path}: {e}") from e

    return data.get("contributors", [])


def render_md(contributors: List[Dict]) -> str:
    lines = [
        "# Contributors",
        "",
        "Thanks to all contributors! This file is generated from `.all-contributorsrc`.",
        "",
    ]
    contributors_sorted = sorted(contributors, key=lambda x: (x.get("login") or "").lower())
    for c in contributors_sorted:
        login = c.get("login", "")
        name = c.get("name") or ""
        contribs = c.get("contributions") or []
        contribs_s = ", ".join(contribs) if contribs else ""
        if name:
            lines.append(f"- @{login} — {name}" + (f" — {contribs_s}" if contribs_s else ""))
        else:
            lines.append(f"- @{login}" + (f" — {contribs_s}" if contribs_s else ""))
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    try:
        contributors = read_all_contributors(SRC)
    except FileNotFoundError:
        print(f"{SRC} not found, skipping CONTRIBUTORS.md generation.")
        return 0

    md = render_md(contributors)
    if OUT.exists() and OUT.read_text(encoding="utf-8") == md:
        print("CONTRIBUTORS.md is up to date.")
        return 0

    OUT.write_text(md, encoding="utf-8")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())