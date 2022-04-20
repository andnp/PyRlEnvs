#!/bin/bash
set -e

MYPYPATH=./typings mypy -p PyRlEnvs

export PYTHONPATH=PyRlEnvs
python3 -m unittest discover -p "*test_*.py"
