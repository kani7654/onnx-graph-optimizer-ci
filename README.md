# ONNX Graph Optimizer

This project demonstrates compiler-style optimization of neural network
graphs using ONNX. The entire pipeline runs on GitHub and focuses on
graph-level transformations used in edge AI toolchains.

## Overview
Neural networks exported to ONNX often contain redundant or unnecessary
operations. This project applies graph optimization passes to improve
runtime efficiency and ensure correctness before deployment.

## Pipeline
PyTorch → ONNX Export → Graph Optimization → Model Validation

## Optimizations Applied
- Dead-end node elimination
- Identity node removal
- No-op dropout removal
- Unused initializer cleanup

## Why This Matters
Graph-level optimizations are a critical part of AI compilers and
edge inference runtimes. Removing redundant operations reduces
latency, memory usage, and improves deployability on constrained devices.

## Key Technologies
- PyTorch
- ONNX
- onnxoptimizer
- ONNX Runtime

## Execution
The optimization and validation steps are executed automatically
inside GitHub, without any local setup.

## Key Learning Outcomes
- Understanding ONNX graph structure
- Applying compiler-style optimization passes
- Handling real-world API deprecations
- Building reliable automated AI pipelines
