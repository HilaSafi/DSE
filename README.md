# Design Space Exploration for Quantum Circuit Compilation

This repository contains the code and experimental setup for the paper:

> **Hardware–Software Co-Design in Quantum Circuit Compilation:  
> A Multi-Layer Design Space Exploration**  
> _[Authors, 2025]_

We investigate how compilation strategies and hardware characteristics
interact to affect quantum circuit performance, with a focus on **layout methods**,
**qubit routing techniques**, **optimisation levels**, **topological connectivity**, and
**device-specific noise variants** (including crosstalk).  
Our experiments span both **noisy simulations** and **quantum error correction (QEC)** scenarios.


---

## 📦 Requirements

We maintain **two separate requirement files** to keep Python dependencies reproducible for each type of experiment:

- `requirements.txt` — for **device experiments**
- `requirements_mapper.txt` — for **mapper/compiler experiments**

Both files are **fully pinned** to ensure long-term reproducibility.

---

## 🐳 Running with Docker (Recommended)

Using Docker ensures the environment is identical across machines.

### 1️⃣ Build the images

From the repository root:

```bash
# Build image for device experiments
docker build -t dse-device -f Dockerfile .

# Build image for mapper experiments
docker build -t dse-mapper -f Dockerfile_mapper .

docker run --rm -it -v "$PWD":/app dse-device \
    python device_experiments/run_device_experiment.py

docker run --rm -it -v "$PWD":/app dse-mapper \
    python mapper_experiments/run_mapper_experiment.py

```
---

## 🐳 Running locally (Without Docker)

# Activate your environment - Device Experiments
```bash
pip install -r requirements.txt
python device_experiments/run_device_experiment.py

pip install -r requirements_mapper.txt - Compilation Experiments
python mapper_experiments/run_mapper_experiment.py
```

