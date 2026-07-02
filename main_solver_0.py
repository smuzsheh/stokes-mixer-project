"""
Course Project: Stokes flow + tracer transport (advection–diffusion)
EMPTY CHANNEL — 0 obstacles (Reference Case)
HIGHLY VISCOUS NEWTONIAN FLUID
"""

import time
import numpy as np
from mpi4py import MPI
from dolfinx import fem, io, geometry
from dolfinx.fem.petsc import LinearProblem
from ufl import div, dx, grad, inner, TrialFunction, TestFunction, dot
from pathlib import Path

comm = MPI.COMM_WORLD


# MESH (0 obstacles)

mesh_data = io.gmsh.read_from_msh("dfg_benchmark_0obstacles.msh", comm, gdim=2)
domain = mesh_data.mesh
facet_tags = mesh_data.facet_tags

if comm.rank == 0:
    obstacle_facets = facet_tags.find(3)
    print(f"Obstacle facets found: {len(obstacle_facets)}")
    print(f"Mesh loaded")

results_folder = Path("results_0")
results_folder.mkdir(exist_ok=True, parents=True)



# BOUNDING BOX TREE

_bb_tree = geometry.bb_tree(domain, domain.topology.dim)


def get_pressure_at_point(ph, point):
    pt = np.array([point], dtype=np.float64).reshape(1, 3)
    cells = geometry.compute_collisions_points(_bb_tree, pt)
    if len(cells.array) > 0:
        try:
            return float(ph.eval(pt, [cells.array[0]])[0])
        except:
            pass
    return None


def compute_CoV_from_simulation(ch, x_position):
    y_pts = np.linspace(0.0, 0.41, 500)
    pts = np.column_stack([
        np.full(len(y_pts), x_position),
        y_pts,
        np.zeros(len(y_pts)),
    ])
    
    cells_candidate = geometry.compute_collisions_points(_bb_tree, pts)
    c_values = []
    
    for i in range(len(y_pts)):
        links = cells_candidate.links(i)
        if len(links) > 0:
            cell = links[0]
            try:
                val = float(ch.eval(pts[i:i+1], [cell]).flatten()[0])
                c_values.append(val)
            except:
                pass
    
    if len(c_values) < 50:
        return None
    
    mean = np.mean(c_values)
    std = np.std(c_values)
    
    if mean > 1e-12:
        return float(std / mean)
    else:
        return 1.0



# STOKES PROBLEM

degree = 1
V = fem.functionspace(domain, ("Lagrange", degree + 1, (2,)))
Q = fem.functionspace(domain, ("Lagrange", degree))

u, v = TrialFunction(V), TestFunction(V)
p, q = TrialFunction(Q), TestFunction(Q)

nu = fem.Constant(domain, 1e-3)
f = fem.Constant(domain, np.zeros(2))

a = [
    [nu * inner(grad(u), grad(v)) * dx, -p * div(v) * dx],
    [-q * div(u) * dx, None],
]
L = [inner(f, v) * dx, fem.Constant(domain, 0.0) * q * dx]


def inflow_velocity(x):
    U = 0.3
    ux = 4.0 * U * x[1] * (0.41 - x[1]) / 0.41**2
    uy = np.zeros_like(ux)
    return np.vstack([ux, uy])


u_in = fem.Function(V)
u_in.interpolate(inflow_velocity)
bc_in = fem.dirichletbc(u_in, fem.locate_dofs_topological(V, 1, facet_tags.find(1)))

zero_vec = fem.Constant(domain, np.zeros(2, dtype=np.float64))
bc_wall = fem.dirichletbc(zero_vec, fem.locate_dofs_topological(V, 1, facet_tags.find(2)), V)

bcs = [bc_in, bc_wall]


def solve_stokes(method: str):
    if method == "lu":
        P = None
        opts = {"ksp_type": "preonly", "pc_type": "lu"}
        pfx = "stokes_lu_"
    elif method == "gmres":
        a_p = [[a[0][0], None], [None, p * q * dx]]
        P = a_p
        pfx = "stokes_gmres_"
        opts = {
            "ksp_type": "gmres",
            "ksp_max_it": 2000,
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "additive",
            "fieldsplit_0_ksp_type": "preonly",
            "fieldsplit_0_pc_type": "hypre",
            "fieldsplit_0_pc_hypre_type": "boomeramg",
            "fieldsplit_1_ksp_type": "preonly",
            "fieldsplit_1_pc_type": "jacobi",
        }

    t0 = time.perf_counter()
    problem = LinearProblem(a, L, bcs=bcs, kind="nest", P=P, petsc_options=opts, petsc_options_prefix=pfx)
    uh, ph = problem.solve()
    uh.x.scatter_forward()
    ph.x.scatter_forward()
    dt = time.perf_counter() - t0

    its = None
    try:
        its = problem.solver.getIterationNumber()
    except Exception:
        pass

    return uh, ph, {"time_s": dt, "iterations": its}


uh_lu, ph_lu, info_lu = solve_stokes("lu")
uh_gmres, ph_gmres, info_gmres = solve_stokes("gmres")

if comm.rank == 0:
    print("\n=== STOKES SOLVER COMPARISON ===")
    print(f"LU   : {info_lu['time_s']:.3f} s")
    print(f"GMRES: {info_gmres['time_s']:.3f} s, iters = {info_gmres['iterations']}")



# ADVECTION-DIFFUSION (Highly Viscous Newtonian Fluids)

# HIGHLY VISCOUS PARAMETERS
nu_cold, nu_hot = 0.1, 0.01  # Highly viscous
nu.value = (nu_cold + nu_hot) / 2.0

degree_c = 1
W = fem.functionspace(domain, ("Lagrange", degree_c))
c, w = TrialFunction(W), TestFunction(W)

diffusivity = fem.Constant(domain, 5e-2)  # Low diffusivity


def c_inflow(x):
    return np.where(x[1] >= 0.205, 1.0, 0.0).astype(np.float64)


c_D = fem.Function(W)
c_D.interpolate(c_inflow)

fdim = domain.topology.dim - 1
domain.topology.create_connectivity(fdim, domain.topology.dim)
bc_c = fem.dirichletbc(c_D, fem.locate_dofs_topological(W, fdim, facet_tags.find(1)))

a_c = diffusivity * dot(grad(c), grad(w)) * dx + dot(uh_lu, grad(c)) * w * dx
L_c = fem.Constant(domain, 0.0) * w * dx

problem_c = LinearProblem(a_c, L_c, bcs=[bc_c], petsc_options={
    "ksp_type": "gmres", "pc_type": "hypre", "pc_hypre_type": "boomeramg",
    "pc_hypre_boomeramg_smooth_type": "ilu",
}, petsc_options_prefix="adv_diff_")

ch = problem_c.solve()
ch.name = "Concentration"



# RESULTS

if comm.rank == 0:
    p_in = get_pressure_at_point(ph_lu, (0.05, 0.2, 0.0))
    p_out = get_pressure_at_point(ph_lu, (2.15, 0.2, 0.0))
    
    print("\n" + "="*50)
    print(f"PRESSURE ANALYSIS (0 Obstacles)")
    print("="*50)
    print(f"Inlet pressure:  {p_in:.8f} Pa")
    print(f"Outlet pressure: {p_out:.8f} Pa")
    print(f"\nPressure difference: {p_in - p_out:.6f} Pa")

    print("\n" + "="*50)
    print("CoV FROM SIMULATION")
    print("="*50)
    print(f"{'Position':<12} {'CoV':<10} {'Interpretation'}")
    print("-"*50)
    
    x_positions = [0.1, 0.4, 0.7, 1.0, 1.3, 1.6, 1.9]
    
    for x in x_positions:
        cov = compute_CoV_from_simulation(ch, x)
        if cov is not None:
            if cov < 0.05:
                interp = "Well mixed ✓"
            elif cov < 0.1:
                interp = "Moderately mixed"
            elif cov < 0.2:
                interp = "Partially mixed"
            else:
                interp = "Poorly mixed"
            print(f"x = {x:.2f}:    CoV = {cov:.6f}  ({interp})")
        else:
            print(f"x = {x:.2f}:    CoV = Could not compute")
    
    cov_outlet = compute_CoV_from_simulation(ch, 1.9)
    print("\n" + "="*50)
    print("CONCLUSION")
    print("="*50)
    print(f"• Total pressure drop: {p_in - p_out:.6f} Pa")
    print(f"• Outlet CoV: {cov_outlet:.6f} (No mixing elements)")



# PARAVIEW EXPORT

uh_lu.name = "Velocity"
with io.VTXWriter(comm, results_folder / "stokes.bp", [uh_lu]) as vtx:
    vtx.write(0.0)

ph_lu.name = "Pressure"
with io.VTXWriter(comm, results_folder / "stokes_pressure.bp", [ph_lu]) as vtx:
    vtx.write(0.0)

with io.VTXWriter(comm, results_folder / "advection_diffusion.bp", [ch]) as vtx:
    vtx.write(0.0)

if comm.rank == 0:
    print("\nFiles written to results_0/")