# simulation-tech-project
# Coupled Stokes-Advection-Diffusion Solver
Finite element solver for coupled Stokes-advection-diffusion problem in the 2-D DFG benchmark configuration using FEniCSx.

## Problem
**Governing Equations:**
```
-ν Δu + ∇p = 0      (Stokes momentum)
div u = 0           (Incompressibility)
u·∇c - D Δc = 0     (Advection-diffusion)
```

**Parameters:**
- Maximum velocity: U = 0.3
- Kinematic viscosity: ν = 0.001
- Diffusivity: D = 5×10⁻⁴

**Boundary Conditions:**
- Inlet: u = u_in (parabolic), c = c_in (step function at y = 0.2)
- Walls/obstacle: u = 0, -D∂c/∂n = 0
- Outlet: ν∂u/∂n - pn = 0, -D∂c/∂n = 0

## Solution Method
**Sequential decoupling:**
1. Solve Stokes for velocity u and pressure p
2. Solve advection-diffusion for concentration c using computed u

No nonlinear solver needed - u·∇c is linear in c when u is known.

**Discretization:**
- Velocity/pressure: Taylor-Hood P2-P1 elements
- Concentration: P1 elements
- Linear solver: Direct LU with MUMPS (default)

## Files
- `stokes_advection_diffusion.py` - Main solver
- `verification_mms.py` - Method of manufactured solutions verification
- `extension_iterative_solvers_concise.py` - Iterative solver comparison study
- `dfg_benchmark_2d.msh` - Mesh file

## Usage
**Serial:**
```bash
python3 stokes_advection_diffusion.py
```

**Parallel:**
```bash
mpirun -n 4 python3 stokes_advection_diffusion.py
```

**Output:** Results exported to `results/` directory (VTX format for ParaView)

## Verification
Run convergence study with manufactured solutions:
```bash
python3 verification_mms.py
```

Expected convergence rates:
- Velocity (P2): rate ≈ 3
- Pressure (P1): rate ≈ 2
- Concentration (P1): rate ≈ 2

## Extension Study: Iterative Solver Comparison
Compare performance of direct vs iterative solvers:
```bash
python3 extension_iterative_solvers_concise.py
```

**Solvers tested:**
- **Direct:** LU factorization (MUMPS)
- **Iterative:** GMRES and BiCGSTAB with different preconditioners
  - Jacobi (simple diagonal preconditioning)
  - ILU (Incomplete LU factorization)
  - AMG (Algebraic Multigrid - hypre BoomerAMG)

**Performance metrics:**
- Solution time (wall-clock)
- Number of iterations
- Solution accuracy (L2 error vs LU reference)

**Key findings:**
- Direct solvers (LU) are most robust for small-medium problems
- Iterative solvers with AMG/ILU preconditioning can be faster for large systems
- Preconditioning is critical: AMG/ILU significantly outperform Jacobi
- All iterative solvers achieve machine precision accuracy (~1e-8)

## Requirements
- Python 3.8+
- FEniCSx (dolfinx)
- MPI (mpi4py)
- PETSc with MUMPS
- Gmsh (for mesh generation)
- hypre (for AMG preconditioning)

## Installation
```bash
# Create virtual environment
python3 -m venv fenics_venv
source fenics_venv/bin/activate

# Install FEniCSx
pip install fenics-dolfinx
```
