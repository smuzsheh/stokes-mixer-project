# Static Mixer Simulation

**Author:** Uzma Shehzadi  ,Hamna Shafique
**Supervisor:** Dr. Timm Treskatis  
**Institution:** TU Dortmund, Fakultät für Mathematik  
**Course:** Scientific Computing Project

---

## 📌 Project Description

This project simulates 2D Stokes flow and advection-diffusion of a tracer in a channel containing a Kenics-type static mixer. The study investigates the mixing of two highly viscous Newtonian fluids (toothpaste-like) by computing:

- **Pressure drop** across the mixer
- **Coefficient of Variation (CoV)** as a measure of mixing quality

Simulations are performed for **0, 4, and 6 obstacles** to analyze the effect of mixer geometry on performance. The solver is verified using the Method of Manufactured Solutions (MMS) and unit tests.
**Repository Structure**
.
├── README.md
├── requirements.txt
├── meshes/
│   ├── dfg_benchmark_0obstacles.geo
│   ├── dfg_benchmark_0obstacles.msh
│   ├── dfg_benchmark_4obstacles.geo
│   ├── dfg_benchmark_4obstacles.msh
│   ├── dfg_benchmark_6obstacles.geo
│   └── dfg_benchmark_6obstacles.msh
├── src/
│   ├── main_solver_0.py
│   ├── main_solver_4.py
│   └── main_solver_6.py
├── tests/
│   ├── test_mms.py
│   └── test_unit.py
├── results/            # Simulation output 
└── docs/               # Notes and documentation
** How to Run**
**Generate mesh**
gmsh meshes/dfg_benchmark_4obstacles.geo -2 -o meshes/dfg_benchmark_4obstacles.msh                                           
**Run a simulation**                                                                                                   
python src/main_solver_4.py
**Run verification tests**                                                                                                       
python tests/test_mms.py
python tests/test_unit.py                                                                                              
**Conclusion:** The 2D Kenics mixer did not achieve well-mixed conditions (CoV < 0.05) for the highly viscous fluids tested. These results are a valid outcome for this geometry and provide a baseline for future design improvements.
