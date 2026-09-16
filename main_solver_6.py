"""
Course Project: Stokes flow + tracer transport (advection–diffusion)
KENICS MIXER — 6 obstacles
HIGHLY VISCOUS NEWTONIAN FLUID - 
"""

import time
import numpy as np
from mpi4py import MPI
from dolfinx import fem, io, geometry, mesh as dmesh
from dolfinx.fem.petsc import LinearProblem
from ufl import div, dx, grad, inner, TrialFunction, TestFunction, dot
from pathlib import Path
from ufl import sqrt, CellDiameter, div, dot, grad, dx

comm = MPI.COMM_WORLD



# MESH (6 obstacles)
OBSTACLES_6 = [
    (0.30, 0.20),
    (0.55, 0.27),
    (0.80, 0.13),
    (1.05, 0.20),
    (1.30, 0.27),
    (1.55, 0.13),
]

mesh_data = io.gmsh.read_from_msh("dfg_benchmark_6obstacles.msh", comm, gdim=2)
domain = mesh_data.mesh
facet_tags = mesh_data.facet_tags

if comm.rank == 0:
    obstacle_facets = facet_tags.find(3)
    print(f"Obstacle facets found: {len(obstacle_facets)}")
    print(f"Mesh loaded")

results_folder = Path("results_6")
results_folder.mkdir(exist_ok=True, parents=True)



# EXPORT OBSTACLE MESH FOR PARAVIEW

domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim - 1)

obstacle_facet_indices = facet_tags.find(3)
if len(obstacle_facet_indices) > 0:
    obstacle_submesh, entity_map, vertex_map, geom_map = dmesh.create_submesh(
        domain, domain.topology.dim - 1, obstacle_facet_indices
    )
    with io.XDMFFile(comm, results_folder / "obstacle_boundary.xdmf", "w") as xf:
        xf.write_mesh(obstacle_submesh)

cell_tag_space = fem.functionspace(domain, ("DG", 0))
obstacle_marker = fem.Function(cell_tag_space)
obstacle_marker.name = "obstacle_cells"
obstacle_marker.x.array[:] = 0.0

f_to_c = domain.topology.connectivity(domain.topology.dim - 1, domain.topology.dim)
for fidx in obstacle_facet_indices:
    for cidx in f_to_c.links(fidx):
        obstacle_marker.x.array[cidx] = 1.0
obstacle_marker.x.scatter_forward()

with io.XDMFFile(comm, results_folder / "obstacle_cells.xdmf", "w") as xf:
    xf.write_mesh(domain)
    xf.write_function(obstacle_marker)



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
from dolfinx import mesh as dmesh
from dolfinx.fem import assemble_scalar, form
import ufl


def compute_CoV_from_simulation(ch, x_position, domain, obstacle_centers=None,
                                obstacle_radius=0.04, n_samples=2000,
                                c_global_mean=0.5):
    """
    Compute the Coefficient of Variation on the vertical cross-section
    x = x_position. The mean in the denominator is the GLOBAL mean
    concentration, which is conserved by the advection-diffusion
    equation (zero diffusive flux at walls and obstacles).

    Parameters
    ----------
    ...
    c_global_mean : float
        Global mean concentration. For the inlet split used here,
        c_global_mean = 0.5.
    """
    from dolfinx import geometry as geom

    y_pts = np.linspace(0.0, 0.41, n_samples)

    if obstacle_centers is not None:
        mask = np.ones_like(y_pts, dtype=bool)
        for (cx, cy) in obstacle_centers:
            if abs(cx - x_position) < obstacle_radius + 0.005:
                arg = obstacle_radius**2 - (cx - x_position)**2
                if arg > 0:
                    dy = np.sqrt(arg)
                    mask &= ~((y_pts >= cy - dy) & (y_pts <= cy + dy))
        y_pts = y_pts[mask]

    if len(y_pts) < 20:
        return None

    points = np.column_stack([
        np.full(len(y_pts), x_position),
        y_pts,
        np.zeros(len(y_pts)),
    ]).astype(np.float64)

    bb_tree = geom.bb_tree(domain, domain.topology.dim)
    cells = geom.compute_collisions_points(bb_tree, points)

    c_values = np.full(len(y_pts), np.nan)
    for i in range(len(y_pts)):
        links = cells.links(i)
        if len(links) > 0:
            try:
                val = ch.eval(points[i:i + 1], [links[0]])
                c_values[i] = float(val.flatten()[0])
            except Exception:
                pass

    valid = ~np.isnan(c_values)
    if np.sum(valid) < 20:
        return None

    y_valid = y_pts[valid]
    c_valid = np.clip(c_values[valid], 0.0, 1.0)

    L = np.trapezoid(np.ones_like(y_valid), y_valid)
    if L < 1e-14:
        return None

    # LOCAL mean (for the standard deviation only)
    c_local_mean = np.trapezoid(c_valid, y_valid) / L
    c2_mean = np.trapezoid(c_valid**2, y_valid) / L

    var = c2_mean - c_local_mean**2
    var = max(var, 0.0)
    std = np.sqrt(var)

    # GLOBAL mean in the denominator (conserved, = 0.5 by construction)
    if c_global_mean < 1e-12:
        return None
    return float(std / c_global_mean)
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
bc_obstacle = fem.dirichletbc(zero_vec, fem.locate_dofs_topological(V, 1, facet_tags.find(3)), V)

bcs = [bc_in, bc_wall, bc_obstacle]


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

# HIGHLY VISCOUS PARAMETERS - SAME AS 4 AND 6 OBSTACLES!
nu_cold, nu_hot = 0.1, 0.01  # Highly viscous (toothpaste-like)
nu.value = (nu_cold + nu_hot) / 2.0

degree_c = 1
W = fem.functionspace(domain, ("Lagrange", degree_c))
c, w = TrialFunction(W), TestFunction(W)

diffusivity = fem.Constant(domain, 5e-4)  # Low diffusivity - SAME AS OTHERS!


def c_inflow(x):
    return np.where(x[1] >= 0.205, 1.0, 0.0).astype(np.float64)


c_D = fem.Function(W)
c_D.interpolate(c_inflow)

fdim = domain.topology.dim - 1
domain.topology.create_connectivity(fdim, domain.topology.dim)
bc_c = fem.dirichletbc(c_D, fem.locate_dofs_topological(W, fdim, facet_tags.find(1)))



# Element size (UFL CellDiameter works element-wise on P1 and P2 spaces)
h_e = CellDiameter(domain)

# Velocity magnitude (regularized to avoid division by zero)
u_mag = sqrt(dot(uh_lu, uh_lu)) + 1e-12

# SUPG stabilization parameter (Brooks & Hughes 1982)
# Doubly-asymptotic form: correct in both advection- and diffusion-dominated limits
tau = 1.0 / sqrt((2.0 * u_mag / h_e) ** 2 + (4.0 * diffusivity / h_e ** 2) ** 2)

# Strong-form residual of the advection-diffusion equation
residual_c = dot(uh_lu, grad(c)) - diffusivity * div(grad(c))

# SUPG-stabilized weak form
a_c = (
    diffusivity * dot(grad(c), grad(w)) * dx
    + dot(uh_lu, grad(c)) * w * dx
    + tau * dot(uh_lu, grad(w)) * residual_c * dx
)
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
    print(f"PRESSURE ANALYSIS (6 Obstacles)")
    print("="*50)
    print(f"Inlet pressure:  {p_in:.8f} Pa")
    print(f"Outlet pressure: {p_out:.8f} Pa")
    print(f"\nPressure difference: {p_in - p_out:.6f} Pa")
    print(f"Pressure drop per obstacle: {(p_in - p_out)/6:.6f} Pa")

    print("\n" + "="*50)
    print("CoV FROM SIMULATION")
    print("="*50)
    print(f"{'Position':<12} {'CoV':<10} {'Interpretation'}")
    print("-"*50)
    
    # SAME positions as other cases for fair comparison
    x_positions = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.45, 1.6, 1.75, 1.9]
    
    for x in x_positions:
        cov = compute_CoV_from_simulation(ch, x, domain,obstacle_centers=OBSTACLES_6)
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
    
    cov_outlet = compute_CoV_from_simulation(ch, 1.9,domain,obstacle_centers=OBSTACLES_6)
    print("\n" + "="*50)
    print("CONCLUSION")
    print("="*50)
    print(f"• Total pressure drop: {p_in - p_out:.6f} Pa")
    print(f"• Pressure drop per element: {(p_in - p_out)/6:.6f} Pa")
    if cov_outlet is not None:
        if cov_outlet < 0.05:
            print(f"• Outlet CoV: {cov_outlet:.6f} (Well mixed ✓)")
            print("• Industry standard achieved ✓")
        else:
            print(f"• Outlet CoV: {cov_outlet:.6f} (Not yet well mixed)")


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
    print("\nFiles written to results_6/")