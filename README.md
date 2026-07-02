# Static Mixer Simulation

**Author:** Uzma Shehzadi , Hamna Shafique
**Supervisor:** Dr. Timm Treskatis  
**Institution:** TU Dortmund

---

## 📌 Project Description

2D Stokes flow and advection-diffusion simulations for a Kenics-type static mixer. Investigates mixing of two highly viscous Newtonian fluids by computing:

- Pressure drop
- Coefficient of Variation (CoV)

Simulations for **0, 4, and 6 obstacles**.

---

## Project Objectives

This project aims to:

- Develop a finite element solver for coupled Stokes and advection–diffusion equations.
- Simulate tracer transport in a two-dimensional static mixer.
- Evaluate mixing performance using the Coefficient of Variation (CoV).
- Analyze pressure drop across different mixer configurations.
- Verify the numerical implementation using manufactured solutions and unit tests.

## 🧰 Dependencies

- dolfinx (FEniCSx)
- mpi4py
- gmsh
- numpy
- ufl

---

## 🚀 How to Run

```bash
# Generate mesh
gmsh dfg_benchmark_4obstacles.geo -2 -o dfg_benchmark_4obstacles.msh

# Run simulation
python main_solver_4.py

# Run tests
python test_mms.py
python test_unit.py
