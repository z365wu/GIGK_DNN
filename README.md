# GIGK_DNN
Training DNN for solving GI/G/K Queuing systems

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview
This repository contains the source code and scripts accompanying the paper:

> **Training Neural Networks for the GI/G/K Queue**  
> *Zhenggao Wu, Haoran Wu, Haokun Zhao, Qi-Ming He*  
> *Journal/Conference, Year: TBD*  
> TBD: [DOI or preprint link]:TBD

The project provides Python implementations for training, sampling, and prediction tasks, as well as reproducible workflows to generate results.

---

## Repository Structure
```
.
├──scripts_outputs            # All Python scripts and outputs
    ├── README.md               # Detailed instructions on running codes
    ├── script_main.py          # Main entry point for running the workflow
    ├── script_training.py      # Script for model training
    ├── script_prediction.py    # Script for prediction
    ├── script_sampling.py      # Script for sampling
    │
    ├── Training/               # Training-related modules
    ├── Sampling/               # Sampling-related modules
    ├── Prediction/             # Prediction-related modules
    │
    ├── Output/                 # Generated results
    │   ├── Figures/            # Figures results
    │   ├── models/             # DNN parameters
    │   ├── Tables/             # Table results
    │   └── samples/            # Sampled data
│
├──results                  # Final results: TBD
    ├── README.md               # Short pointer to outputs: TBD
│
├── requirements.txt        # Python dependencies
├── LICENSE                 # License file
└── README.md               # Project overview, installation, workflow.
```

---

## Installation
Clone the repository and install the dependencies:

```bash
git clone https://github.com/z365wu/GIGK_DNN.git
cd YourProject
pip install -r requirements.txt
```

> **Note:** Python ≥3.12 is recommended.

---

## Usage

### 1. Training
```bash
python scripts_outputs/script_training.py
```

### 2. Prediction
```bash
python scripts_outputs/script_prediction.py
```

### 3. Sampling
```bash
python scripts_outputs/script_sampling.py
```

### 4. Full Workflow
```bash
python scripts_outputs/script_main.py
```

Generated outputs will be stored under `.scripts_outputs/Output/`.

---

## Data Availability
- **Input data:** Large result files are excluded from the repository but can be reproduced using the provided scripts. 
- **Output data:** Large result files are excluded from the repository but can be reproduced using the provided scripts.  

If you require access to the original training samples, please contact the authors.

---

## Reproducibility
To ensure reproducibility:
- Code has been tested with Python 3.12.  
- Dependencies are specified in `requirements.txt`.   

---

## Citation
If you use this code or results in your research, please cite:

```bibtex
TBD
```

---

## License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## Contact
For questions or collaboration, please contact:  
**Zhenggao Wu, Qi-Ming He**  
Email: z365wu@uwaterloo.ca, q7he@uwaterloo.ca

Affiliation: Department of Management Science and Engineering, University of Waterloo
