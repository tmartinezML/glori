import os
import sys
import pytest

import dependency_detective

# Main folder is the pythonpath
main_folder = os.path.join(os.path.dirname(__file__), "..", "src")
if main_folder not in sys.path:
    sys.path.insert(0, main_folder)


def list_python_modules(folder):
    modules = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".py") and not file.startswith("__init__"):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, folder)
                if not relative_path.startswith("glori" + os.sep):
                    continue
                module = relative_path.replace(".py", "").replace(os.sep, ".")
                modules.append(module)
    return modules


imports_to_test = list_python_modules(main_folder)


@pytest.mark.parametrize("module", imports_to_test)
def test_dynamic_imports(module):
    # Exclude modules that start with 'scripts.'
    excluded = ["archive."]
    if any(module.startswith(prefix) for prefix in excluded):
        return
    try:
        __import__(module)
    except ImportError as e:
        pytest.fail(f"Failed to import {module}: {str(e)}")
