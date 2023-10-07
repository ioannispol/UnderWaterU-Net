#!/bin/bash

git config --global --add safe.directory /workspaces/UnderWaterU-Net

pip install -e .[dev]
pip install pytest-cov
pre-commit install
