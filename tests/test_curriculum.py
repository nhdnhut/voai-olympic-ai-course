"""
VOAI Olympic AI Comprehensive Training Course Test Suite
Validates all 38 session notebooks + Course Map integrity and valid JSON structure.
"""

import json
import os
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COURSE_DIR = os.path.join(REPO_ROOT, "course")

EXPECTED_MODULES = [
    "M0_AI_ORIENTATION",
    "M1_PYTHON_DATA",
    "M2_MACHINE_LEARNING",
    "M3_PYTORCH",
    "M4_APPLIED_AI",
    "M5_OLYMPIC_STRATEGY",
    "M6_LLM_ASSISTED",
    "M7_FINAL_PROJECT",
]

def test_course_directory_exists():
    assert os.path.isdir(COURSE_DIR), "Course directory must exist"

def test_all_modules_present():
    for mod in EXPECTED_MODULES:
        mod_path = os.path.join(COURSE_DIR, mod)
        assert os.path.isdir(mod_path), f"Module {mod} must exist"

def test_notebooks_count_and_validity():
    notebooks = []
    for root, _, files in os.walk(COURSE_DIR):
        for f in files:
            if f.endswith(".ipynb"):
                notebooks.append(os.path.join(root, f))
    
    # 38 session notebooks + COURSE_MAP.ipynb = 39
    assert len(notebooks) >= 39, f"Expected at least 39 notebooks, found {len(notebooks)}"
    
    for nb_path in notebooks:
        with open(nb_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            assert "cells" in data, f"Notebook {nb_path} missing 'cells' key"
            assert "metadata" in data, f"Notebook {nb_path} missing 'metadata' key"
            assert len(data["cells"]) > 0, f"Notebook {nb_path} has 0 cells"

def test_course_map_exists():
    course_map = os.path.join(COURSE_DIR, "COURSE_MAP.ipynb")
    assert os.path.isfile(course_map), "COURSE_MAP.ipynb must exist"
