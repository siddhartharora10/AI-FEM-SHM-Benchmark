# AI-FEM SHM Benchmarking Framework

This repository contains code for benchmarking AI-augmented Finite Element Methods (FEM) for Structural Health Monitoring (SHM).

## Features
- FEM-based synthetic dataset generation
- Moderate and realistic scenario modeling
- Machine learning-based damage classification
- Performance comparison under different conditions

## How to Run

### Generate datasets
python3 scripts/generate_dataset_moderate.py
python3 scripts/generate_dataset_realistic.py

### Train models
python3 scripts/train_model.py moderate
python3 scripts/train_model.py realistic

### Generate figures
python3 scripts/plot_results.py moderate
python3 scripts/plot_results.py realistic

## Results
- Moderate scenario: ~99% accuracy
- Realistic scenario: ~44% accuracy

## Reproducibility
All experiments are fully reproducible using provided scripts.