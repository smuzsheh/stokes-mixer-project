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
'''

---
## Solver Workflow

The simulation is performed using the following workflow:

1. Generate the computational mesh for the selected mixer configuration.
2. Solve the incompressible Stokes equations to compute the velocity and pressure fields.
3. Solve the advection–diffusion equation using the computed velocity field.
4. Evaluate the pressure drop across the channel.
5. Compute the Coefficient of Variation (CoV) to assess mixing performance.
6. Export simulation results for visualization and post-processing.




