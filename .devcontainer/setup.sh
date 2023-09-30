#!/bin/bash

pip install -e .[dev]
pip install pytest-cov
pre-commit install