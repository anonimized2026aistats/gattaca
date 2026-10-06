# Graph-based ATtractor-TArget Control Algorithm (GATTACA)

Source code and trained model implementation for the GATTACA framework introduced in the associated anonymous submission.

This project extends prior work on pbn-STAC and gym-pbn-STAC, which in turn builds on the methods introduced in [G. Papagiannis, S. Moschoyiannis et al., Deep Reinforcement Learning for Stabilization of Large-Scale Probabilistic Boolean Networks (2022)](https://ieeexplore.ieee.org/document/9999487), including the public [gym-PBN](https://github.com/UoS-PLCCN/gym-PBN/tree/main) and [pbn-rl](https://github.com/UoS-PLCCN/pbn-rl) implementations.

# Environment Requirements
- CUDA 11.3+
- Python >=3.10,<3.14

Python 3.14 is not currently supported because `bang-gpu` depends on `numba==0.61.0`, which supports Python versions >=3.10,<3.14.

# Installation
## Local
- Clone the three repositories next to each other:
    ```sh
    git clone https://github.com/anonimized2026aistats/bang.git
    git clone https://github.com/anonimized2026aistats/gym-PBN-stac.git
    git clone https://github.com/anonimized2026aistats/gattaca.git
    cd gattaca
    ```
- Create and activate a Python environment:
    ```sh
    python --version
    python3 -m venv .env
    source .env/bin/activate
    ```
    Use a Python 3.10, 3.11, 3.12, or 3.13 interpreter. For the last line, use `.\.env\Scripts\Activate.ps1` if on Windows PowerShell.
- Install [PyTorch](https://pytorch.org/get-started/locally/):
    ```sh
    python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
    ```
- Install the local BANG and gym-PBN packages:
    ```sh
    python -m pip install -e ../bang
    python -m pip install -e ../gym-PBN-stac
    ```
- Install the remaining GATTACA dependencies:
    ```sh
    python -m pip install -r requirements.txt
    ```

Do not install `gym-PBN` from PyPI for this codebase; the PyPI package may pull the deprecated `sklearn` dependency and does not necessarily include the BANG-backed environment used by `train_bang.py`.

# Models
All trained models are available via google drive:
https://drive.google.com/drive/folders/1qLV0IdBfFg-MFj28WtYGdy6pYK63YfUs?usp=sharing

# Running
- Use `train_gattaca.py` to train a DDQN agent. It's a command line utility so you can check out what you can do with it using `--help`.
    E.g.:
    ```sh
     python train_gattaca.py --size 67 --assa-file  bortezomib_fixed.ispl --exp-name example
    ```

- Use `model_tester.py` to get strategies and statistics about the model.
E.g.:
```sh
python model_tester.py -n 67 --assa-file  bortezomib_fixed.ispl --model-path models/pbn67/bdq_final.pt --attractors 10 --runs 10
```

| Argument       | Description                                                          |
| -------------- | -------------------------------------------------------------------- |
| `-n`           | (Required) Number of nodes in the model.                       |
| `--assa-file`  | (Required) Path to the `.ispl` file.     |
| `--model-path` | (Required) Path to the trained PyTorch model `.pt` file.             |
| `--attractors` | (Optional) Number of source attractors to analyse. It may be smaller than the total number of attractors.             |
| `--runs`       | (Optional) Number of test runs for averaging or robustness checking. |
