# CUDA Matrix Inversion using Neumann Series Approximation (NSA)

## 📌 Overview
This project presents a CUDA-based implementation of square matrix inversion
using the **Neumann Series Approximation (NSA)**.  
The objective is to efficiently compute the inverse of small to medium-sized
matrices and evaluate both **numerical accuracy** and **computational performance**
on GPU.

The CUDA implementation is compared against:
- a CPU reference implementation
- a MATLAB-based implementation

This work is conducted in the context of a **university CUDA mini-project**.

---

## 🎯 Objectives
- Implement matrix inversion using NSA on GPU (CUDA)
- Study different thread-mapping strategies
- Compare CUDA results with CPU and MATLAB references
- Analyze numerical error (Frobenius norm) and execution time
- Apply the inversion to a **Zero Forcing (ZF) detector**

---

## 🧮 Mathematical Background
Given a square matrix **A**, its inverse is approximated using the Neumann series:

\[
A^{-1} \approx \sum_{k=0}^{N} (I - A)^k
\]

where:
- \( I \) is the identity matrix
- \( N \) is the order of approximation

The convergence depends on the spectral radius of \( (I - A) \).

---

## 🚀 Implemented Features
- CUDA matrix multiplication
- Matrix inversion using NSA
- Multiple CUDA kernel strategies:
  - 1 thread per element
  - 1 thread per row
  - 1 thread per column
- CPU reference inversion
- MATLAB reference scripts
- Zero Forcing (ZF) detector implementation
- Performance and accuracy comparison

---

## 📐 Supported Matrix Sizes
- 4 × 4
- 8 × 8
- 16 × 16
- 32 × 32
- 64 × 64

All computations are performed in **single precision (float)**.

---

## 🛠️ Technologies Used
- **CUDA C**
- **MATLAB**
- **NVIDIA Nsight (profiling)**
- **Git / GitHub**

---
## 📊 Numerical Accuracy Evaluation

The numerical accuracy of the CUDA-based matrix inversion is evaluated using
the **Frobenius norm of the error matrix**, defined as:

\[
\varepsilon = \frac{\| A^{-1}_{\text{CUDA}} - A^{-1}_{\text{ref}} \|_F}
{\| A^{-1}_{\text{ref}} \|_F}
\]

where:
- \( A^{-1}_{\text{CUDA}} \) is the inverse computed on the GPU
- \( A^{-1}_{\text{ref}} \) is the reference inverse (CPU or MATLAB)
- \( \| \cdot \|_F \) denotes the Frobenius norm

This metric provides a global measure of numerical error and allows a
quantitative comparison of accuracy across different matrix sizes and NSA
approximation orders.

## 📄 License
This project is released under the MIT License.



