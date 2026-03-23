#!/usr/bin/env python3
"""
Sync dependencies from environment.yml into pyproject.toml [project].dependencies.

- Reads conda + pip deps from environment.yml
- Skips python and pip pseudo-entries
- Converts conda single "=" pins to "=="
- De-duplicates while preserving order
- Replaces [project].dependencies in pyproject.toml
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Missing dependency: pyyaml. Install with: pip install pyyaml", file=sys.stderr)
    raise


def is_single_equals_pin(dep: str) -> bool:
    # Matches "name=1.2.3" but not "name==1.2.3"
    return re.match(r"^[A-Za-z0-9_.-]+=[^=].*$", dep) is not None


def normalize_dep(dep: str) -> str | None:
    dep = dep.strip()
    if not dep:
        return None
    if dep == "pip":
        return None
    if dep.startswith("python"):
        return None
    if is_single_equals_pin(dep):
        name, ver = dep.split("=", 1)
        return f"{name}=={ver}"
    return dep


def collect_deps_from_environment(env_path: Path) -> list[str]:
    data = yaml.safe_load(env_path.read_text())
    raw_deps = data.get("dependencies", [])

    deps: list[str] = []

    for item in raw_deps:
        if isinstance(item, str):
            norm = normalize_dep(item)
            if norm:
                deps.append(norm)
        elif isinstance(item, dict) and "pip" in item:
            for pip_dep in item["pip"]:
                norm = normalize_dep(str(pip_dep))
                if norm:
                    deps.append(norm)

    # De-duplicate in original order
    seen = set()
    unique = []
    for d in deps:
        if d not in seen:
            seen.add(d)
            unique.append(d)

    return unique


def render_dependencies_block(deps: list[str], indent: str = "") -> str:
    lines = [f"{indent}dependencies = ["]
    for d in deps:
        lines.append(f'{indent}    "{d}",')
    lines.append(f"{indent}]")
    return "\n".join(lines)


def replace_project_dependencies(pyproject_text: str, new_deps_block: str) -> str:
    # Find [project] section
    project_match = re.search(r"(?ms)^\[project\]\n(.*?)(?=^\[|\Z)", pyproject_text)
    if not project_match:
        raise ValueError("Could not find [project] section in pyproject.toml")

    section_start = project_match.start(1)
    section_end = project_match.end(1)
    project_body = pyproject_text[section_start:section_end]

    # Replace dependencies = [...] (multi-line array)
    array_pattern = re.compile(r"(?ms)^dependencies\s*=\s*\[.*?^\]", re.MULTILINE)
    line_pattern = re.compile(r"(?m)^dependencies\s*=.*$")

    if array_pattern.search(project_body):
        new_body = array_pattern.sub(new_deps_block, project_body, count=1)
    elif line_pattern.search(project_body):
        new_body = line_pattern.sub(new_deps_block, project_body, count=1)
    else:
        # Insert near top of [project] block (after name/version if present)
        lines = project_body.splitlines()
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.startswith("name") or line.startswith("version"):
                insert_idx = i + 1
        lines.insert(insert_idx, new_deps_block)
        new_body = "\n".join(lines) + ("\n" if project_body.endswith("\n") else "")

    return pyproject_text[:section_start] + new_body + pyproject_text[section_end:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="environment.yml", help="Path to environment.yml")
    parser.add_argument("--pyproject", default="pyproject.toml", help="Path to pyproject.toml")
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the generated dependencies block instead of writing pyproject.toml",
    )
    args = parser.parse_args()

    env_path = Path(args.env)
    pyproject_path = Path(args.pyproject)

    if not env_path.exists():
        print(f"Not found: {env_path}", file=sys.stderr)
        return 1
    if not pyproject_path.exists():
        print(f"Not found: {pyproject_path}", file=sys.stderr)
        return 1

    deps = collect_deps_from_environment(env_path)
    deps_block = render_dependencies_block(deps)

    if args.print_only:
        print(deps_block)
        return 0

    text = pyproject_path.read_text()
    updated = replace_project_dependencies(text, deps_block)
    pyproject_path.write_text(updated)

    print(f"Updated {pyproject_path} with {len(deps)} dependencies from {env_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())