# Performance of Quantum Annealing Machine Learning Models on ADMET Datasets

This repository contains the official implementation of the paper titled **"Performance of Quantum Annealing Machine Learning Models on ADMET Datasets"**. You can access the paper at the following link:(not available yet).

## Prerequisites

Before running the code, ensure you have the following set up:

## Datasets Used

The datasets used can be found [here](https://drive.google.com/drive/folders/1oSFcPrGwyeXtfEubGFLE5VsGpJmRvJIt?usp=sharing).

### 1. D-Wave Quantum Annealer Configuration

To use the D-Wave Quantum Annealer, you need to create a configuration file `dwave.conf` with your D-Wave API token. The file should contain the following information:

```
[defaults]
token = "add your token here"
```

### 2. Python Version

We are using **Python 3.12.4**. You can download and install this version from [here](https://www.python.org/downloads/).

### 3. Dataset Setup

Ensure all datasets are stored in the `alldatasets` directory. The method for generating these datasets is explained in the paper.

### 4. Installing Dependencies

Once you've prepared the configuration and datasets, install the required Python dependencies by running:

```
pip install -r requirements.txt
```

## Running the Code

To run the quantum machine learning experiments, execute the following command:

```
python3 Qexperiments.py
```

### Cross-Validation (Optional)

If you want to run the code with cross-validation for the **Quantum Support Vector Machine (QSVM)** method, use the `--cross-validation` flag:

```
python3 Qexperiments.py --cross-validation
```

## Datasets

The ADMET datasets used in this project can be found at the following link: [ADMET Datasets](https://tdcommons.ai/benchmark/overview/).

---

For further details on dataset generation, model training, and experiment results, please refer to the paper.