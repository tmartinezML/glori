import json
import subprocess
import re
import sys
from pathlib import Path
from tqdm import tqdm

REQ_FILE = "new_requirements.txt"
ENV_NAME = "cenv_glori"
CHANNEL = "conda-forge"
MICROMAMBA = "/hsopt/micromamba/micromamba"


def get_python_version():
    """Get current Python major.minor version"""
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def normalize_name(pkg):
    """Strip version specifiers for searching"""
    return re.split(r"[<>=!~]", pkg)[0].strip()


def is_conda_package(pkg_name):
    """Check if package exists in conda-forge (robust)"""
    try:
        result = subprocess.run(
            [
                MICROMAMBA,
                "repoquery",
                "search",
                pkg_name,
                "-c",
                CHANNEL,
                "--json",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )

        # print(f"{pkg_name}: {result.stdout}")  # Debug: print raw output

        if not result.stdout.strip():
            return False

        data = json.loads(result.stdout)

        # If any results exist → it's available
        res = data.get("result", {})

        return len(res.get("pkgs", [])) > 0

    except Exception as e:
        print(f"Error occurred while checking {pkg_name}: {e}")
        return False


def load_requirements():
    lines = Path(REQ_FILE).read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def main():
    python_version = get_python_version()
    print(f"🐍 Detected Python version: {python_version}")

    reqs = load_requirements()

    conda_deps = []
    pip_deps = []

    for req in tqdm(reqs, desc="Checking packages", ncols=80):
        # name = normalize_name(req)

        if is_conda_package(req):
            conda_deps.append(req)
            tqdm.write(f"[conda] {req}")
        else:
            pip_deps.append(req)
            tqdm.write(f"[pip]   {req}")

    # Build YAML
    yaml = []
    yaml.append(f"name: {ENV_NAME}")
    yaml.append("channels:")
    yaml.append(f"  - {CHANNEL}")
    yaml.append("dependencies:")
    yaml.append(f"  - python={python_version}")

    for dep in conda_deps:
        yaml.append(f"  - {dep}")

    if pip_deps:
        yaml.append("  - pip")
        yaml.append("  - pip:")
        for dep in pip_deps:
            yaml.append(f"      - {dep}")

    Path("environment.yml").write_text("\n".join(yaml) + "\n")
    print("\n✅ environment.yml generated")


if __name__ == "__main__":
    main()
