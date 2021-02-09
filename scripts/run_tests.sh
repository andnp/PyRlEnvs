#!/bin/bash
set -e

export PYTHONPATH=Py2048
python3 -m unittest discover -p "*test_*.py"
