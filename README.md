# CIFAR-10 ML System Optimization

A PyTorch-based study of training performance optimization for a Convolutional Neural Network (CNN) on the CIFAR-10 image classification dataset. This project investigates how **DataLoader worker count** and **batch size** affect training time, throughput, and validation accuracy on a single NVIDIA RTX 4050 Laptop GPU.

## Project Overview

Deep learning performance depends not only on model architecture but also on data loading, batch configuration, and hardware utilization. This project experimentally evaluates these factors using repeated training runs and statistical summaries.

### Objectives

- Establish a baseline CNN training configuration.
- Measure the impact of DataLoader workers on training performance.
- Compare batch sizes of 64, 128, and 256.
- Evaluate training time, epoch response time, throughput, and validation accuracy.
- Use repeated trials to assess variability.
- Identify performance configurations while considering the trade-off between speed and accuracy.

## Dataset

**CIFAR-10** contains 60,000 color images across 10 classes.

- Training dataset: 50,000 images
- Training subset used: 45,000 images
- Validation subset: 5,000 images
- Official test set: 10,000 images
- Image dimensions: 32 × 32 × 3
- Number of classes: 10

The training/validation split uses a fixed seed of 42. Experiment trials use seeds 42, 43, and 44.

## Model Architecture

The project uses a custom CNN implemented in PyTorch.

- Convolutional blocks: 3 → 32 → 64 → 128 channels
- Batch normalization and ReLU activations
- Max-pooling layers
- Flatten layer
- Fully connected hidden layer with 256 units
- ReLU activation and dropout (0.3)
- Output layer with 10 class scores

### Training Configuration

| Parameter | Configuration |
|---|---|
| Framework | PyTorch |
| Loss function | Cross-Entropy Loss |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Epochs per run | 15 |
| Hardware | NVIDIA RTX 4050 Laptop GPU |
| GPU memory | 6 GB |
| CUDA | Available |
| Python | 3.11.9 |

## Experiments

### Experiment 1: Baseline

The initial configuration uses batch size 128, two DataLoader workers, and seed 42.

| Metric | Measured result |
|---|---:|
| Best validation accuracy | 80.46% |
| Test accuracy | 78.99% |
| Test loss | 0.5994 |
| Total training time | 74.12 seconds |
| Test throughput | 1,619.17 images/second |

Baseline timing was measured separately from the repeated experiments. Comparisons should account for this difference in timing methodology.

### Experiment 2: DataLoader Worker Count

Batch size is fixed at 128. Worker counts 0, 2, and 4 are evaluated across three seeds.

| Workers | Mean training time (s/epoch) | Mean throughput (images/s) | Mean best validation accuracy |
|---:|---:|---:|---:|
| 0 | 12.884 ± 1.943 | 3,556 ± 502 | 79.793% ± 0.329% |
| 2 | 4.312 ± 0.222 | 10,474 ± 498 | 79.947% ± 0.662% |
| 4 | 2.654 ± 0.152 | 17,061 ± 942 | 79.640% ± 0.374% |

**Observation:** Four workers achieved the lowest measured steady-state training time and the highest throughput in this experiment. Two workers achieved the highest mean best validation accuracy.

Comparing two versus four workers:

- Measured training-time ratio: 1.624×
- Training-time reduction: 38.44%
- Throughput increase: 62.89%

These are single-GPU configuration comparisons, not multi-GPU speedup measurements.

### Experiment 3: Batch Size

DataLoader workers are fixed at four. Batch sizes 64, 128, and 256 are evaluated across three seeds.

| Batch size | Mean training time (s/epoch) | Mean throughput (images/s) | Mean best validation accuracy |
|---:|---:|---:|---:|
| 64 | 2.797 ± 0.245 | 16,262 ± 1,325 | 80.467% ± 0.665% |
| 128 | 2.512 ± 0.199 | 18,010 ± 1,382 | 79.640% ± 0.374% |
| 256 | 2.496 ± 0.144 | 18,088 ± 1,032 | 79.393% ± 1.115% |

**Observation:** Batch size 256 achieved the best mean training-time and throughput results, but batch size 64 achieved the highest mean best validation accuracy.

Comparing batch sizes 64 and 256:

- Measured training-time ratio: 1.121×
- Training-time reduction: 10.77%
- Throughput increase: 11.23%
- Difference in mean best validation accuracy: approximately 1.073 percentage points in favor of batch size 64.

The timing difference between batch sizes 128 and 256 is relatively small.

## Key Findings

1. DataLoader worker configuration had a substantial impact on measured training performance.
2. Increasing workers from two to four improved throughput in the tested configuration.
3. Increasing batch size from 64 to 256 improved throughput modestly.
4. The fastest configuration was not the one with the highest mean best validation accuracy.
5. Repeated trials help expose variability that a single run may not reveal.
6. The best configuration depends on whether the priority is training speed, throughput, or predictive performance.

## Project Structure

```text
cifar10-ml-system-optimization/
├── configs/
│   └── baseline.json
├── src/
│   ├── check_environment.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── experiment_workers.py
│   ├── experiment_workers_repeated.py
│   ├── experiment_batch_size.py
│   └── analyze_performance.py
├── results/
│   ├── experiment_A_repeated/
│   └── experiment_B_batch_size/
├── README.md
└── requirements.txt
```

Generated logs, checkpoints, aggregate CSV files, and plots may be stored under `results/`. The exact contents depend on which experiments have been run and which artifacts are retained.

## Setup and Execution

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/cifar10-ml-system-optimization.git
cd cifar10-ml-system-optimization
```

### 2. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Ensure that the installed PyTorch build supports your CUDA environment if you want to run experiments on an NVIDIA GPU.

### 4. Check the environment

```bash
python src/check_environment.py
```

### 5. Run the experiments

Run these commands from the project root. Use the corresponding scripts and configuration implemented in your local project.

```bash
python src/train.py
python src/experiment_workers.py
python src/experiment_workers_repeated.py
python src/experiment_batch_size.py
python src/analyze_performance.py
```

**Note:** Verify the argument requirements and execution flow in each script before running it. If a script expects command-line arguments or a particular configuration, supply those as defined in the source code.

## Reproducibility and Measurement Notes

- The training/validation split uses seed 42.
- Repeated experiments use seeds 42, 43, and 44.
- Mean and sample standard deviation are reported for repeated trials.
- Steady-state training time excludes the first epoch in the repeated worker experiment.
- Throughput is reported as measured by the experiment scripts.
- Timing may vary with GPU temperature, background processes, data-loading behavior, and software versions.
- Results are specific to the tested hardware, software, dataset split, and training configuration.

## Limitations

- The experiments use one GPU; multi-GPU scaling, inter-GPU communication overhead, and distributed parallel efficiency were not measured.
- Only a limited range of worker counts and batch sizes was tested.
- Results from repeated trials may vary with hardware and software conditions.
- Performance findings should not be assumed to generalize to every GPU, model, or dataset.

## Technologies Used

- Python
- PyTorch
- Torchvision
- CUDA
- CIFAR-10
- NumPy and pandas for experiment data analysis, where used
- Matplotlib for performance plots, where used

## Academic Context

**Course:** ML System Optimization  
**Project:** CIFAR-10 CNN Performance Evaluation

This project investigates empirical performance optimization for deep learning workloads. Add the student name, group number, institution, and instructor details required by your course before submission.

## License

Add a license only if appropriate for your course and the project’s source code and dataset terms. No license is specified here.
