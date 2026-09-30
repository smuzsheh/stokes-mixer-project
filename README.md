# 2D Stokes Flow and Advection–Diffusion in Static Mixers

**Authors:** Uzma Shehzadi, Hamna Shafique
**Supervisor:** Dr. Timm Treskatis
**Institution:** TU Dortmund University, Faculty of Mathematics

## Overview

This project implements a finite element solver for coupled, steady
Stokes flow and advection–diffusion transport in a two-dimensional
channel containing an array of cylindrical obstacles. The obstacles
introduce laminar recirculation and stretching rather than the
crossed-element stream-division mechanism used in commercial Kenics-
or SMX-type static mixers, so the geometry studied here is best
described as an **obstacle-array micromixer**, not a Kenics-type
mixer. Pressure drop and mixing quality (via the Coefficient of
Variation, CoV) are evaluated across five configurations, with
**0, 2, 4, 6, and 8 obstacles**.

## Key Objectives

- Implement a Taylor–Hood ($P_2$/$P_1$) finite element solver for the
  steady Stokes equations, coupled one-way to an SUPG-stabilised
  advection–diffusion solver for tracer transport.
- Simulate tracer transport and quantify mixing quality via the CoV
  metric for each configuration (0, 2, 4, 6, 8 obstacles).
- Quantify the pressure drop across the same five configurations.
- Verify the implementation using the Method of Manufactured
  Solutions (MMS) and unit testing.

## Repository Structure

| Path | Description |
|---|---|
| `main_solver_{0,2,4,6,8}.py` | Stokes + advection–diffusion solver for each obstacle count |
| `dfg_benchmark_{0,2,4,6,8}obstacles.geo/.msh` | Gmsh geometry and mesh files for each configuration |
| `test_mms.py` | Method of Manufactured Solutions convergence tests |
| `test_unit.py` | Unit tests for individual solver components |
| `results_{0,2,4,6,8}/` | Output (fields, figures, CoV/pressure data) for each configuration |
| `requirements.txt` | Pinned Python package versions for a reproducible environment |

## Required Dependencies

- dolfinx (FEniCSx)
- mpi4py
- PETSc
- gmsh
- numpy
- ufl

Exact, pinned versions are listed in `requirements.txt`; install them
with:

```bash
pip install -r requirements.txt
```

## Execution Steps

1. Generate the mesh for a given configuration from its `.geo` file
   using Gmsh (or use the provided `.msh` file directly).
2. Run the corresponding solver script, e.g.:
   ```bash
   python3 main_solver_4.py
   ```
3. (Optional) Run the verification suite:
   ```bash
   python3 test_mms.py
   python3 test_unit.py
   ```
4. Results (fields, CoV, pressure-drop data, and figures) are written
   to the matching `results_*` directory.

## Planned Enhancements

Future work includes implementing SMX-style crossed-element
geometries (which introduce true stream division), extending to
three-dimensional and unsteady flow, and incorporating non-Newtonian
fluid behaviour across a range of Reynolds and Péclet numbers.
