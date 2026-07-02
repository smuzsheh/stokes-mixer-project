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
