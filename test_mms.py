"""
================================================================================
METHOD OF MANUFACTURED SOLUTIONS (MMS) TEST WITH CONVERGENCE STUDY
================================================================================

Purpose:
    Verify that the Stokes + advection-diffusion solver is correctly implemented
    and determine the order of convergence.

Manufactured Solutions:
    Velocity:  u = (sin(πy), 0)
    Pressure:  p = cos(πx)
    Concentration: c = 1 + x² + y²

Expected convergence rates (P2/P1/P2 elements):
    Velocity L2:  O(h³) ≈ 3.0
    Velocity H1:  O(h²) ≈ 2.0
    Pressure L2:  O(h²) ≈ 2.0
    Concentration L2: O(h³) ≈ 3.0
    Concentration H1: O(h²) ≈ 2.0

How to run:
    python test_mms.py
================================================================================
"""

import numpy as np
from mpi4py import MPI
from dolfinx import fem, io, mesh as dmesh
from dolfinx.fem.petsc import LinearProblem
from ufl import (
    div, dx, grad, inner, TrialFunction, TestFunction, dot,
    SpatialCoordinate, as_vector, sin, cos, pi,
)

comm = MPI.COMM_WORLD


# ============================================================================
# PART 1: Manufactured Solutions
# ============================================================================

def u_exact(x):
    """Manufactured velocity: u = (sin(πy), 0)"""
    return as_vector((sin(pi * x[1]), 0.0))

def p_exact(x):
    """Manufactured pressure: p = cos(πx)"""
    return cos(pi * x[0])

def c_exact(x):
    """Manufactured concentration: c = 1 + x² + y²"""
    return 1.0 + x[0]**2 + x[1]**2

# Numpy versions for boundary conditions
def u_exact_numpy(x):
    return np.vstack((np.sin(np.pi * x[1]), np.zeros_like(x[1])))

def c_exact_numpy(x):
    return 1.0 + x[0]**2 + x[1]**2


# ============================================================================
# PART 2: Helper Functions
# ============================================================================

def compute_mesh_size(domain):
    """Compute the mesh size parameter h from the actual mesh."""
    dx_domain = dx(domain=domain)
    total_area = fem.assemble_scalar(fem.form(1 * dx_domain))
    num_cells = domain.topology.index_map(domain.topology.dim).size_global
    avg_area = total_area / num_cells
    h = np.sqrt(avg_area)
    return h


def compute_errors(domain, uh, ph, ch, u_manu, p_manu, c_manu):
    """Compute errors for velocity, pressure, and concentration."""
    dx_domain = dx(domain=domain)
    
    def L2_error(func, exact):
        form = fem.form(inner(func - exact, func - exact) * dx_domain)
        local = fem.assemble_scalar(form)
        return np.sqrt(comm.allreduce(local, op=MPI.SUM))
    
    def H1_error(func, exact):
        L2_form = fem.form(inner(func - exact, func - exact) * dx_domain)
        L2_local = fem.assemble_scalar(L2_form)
        L2_global = comm.allreduce(L2_local, op=MPI.SUM)
        
        H1_form = fem.form(inner(grad(func - exact), grad(func - exact)) * dx_domain)
        H1_local = fem.assemble_scalar(H1_form)
        H1_global = comm.allreduce(H1_local, op=MPI.SUM)
        
        return np.sqrt(L2_global + H1_global)
    
    def H2_error(func, exact):
        L2_form = fem.form(inner(func - exact, func - exact) * dx_domain)
        L2_local = fem.assemble_scalar(L2_form)
        L2_global = comm.allreduce(L2_local, op=MPI.SUM)
        
        H1_form = fem.form(inner(grad(func - exact), grad(func - exact)) * dx_domain)
        H1_local = fem.assemble_scalar(H1_form)
        H1_global = comm.allreduce(H1_local, op=MPI.SUM)
        
        if len(func.ufl_shape) == 0:
            H2_form = fem.form(inner(grad(grad(func - exact)), grad(grad(func - exact))) * dx_domain)
        else:
            H2_form = fem.form(inner(grad(func - exact), grad(func - exact)) * dx_domain)
        
        H2_local = fem.assemble_scalar(H2_form)
        H2_global = comm.allreduce(H2_local, op=MPI.SUM)
        
        return np.sqrt(L2_global + H1_global + H2_global)
    
    return {
        "velocity_L2": L2_error(uh, u_manu),
        "velocity_H1": H1_error(uh, u_manu),
        "velocity_H2": H2_error(uh, u_manu),
        "pressure_L2": L2_error(ph, p_manu),
        "concentration_L2": L2_error(ch, c_manu),
        "concentration_H1": H1_error(ch, c_manu),
        "concentration_H2": H2_error(ch, c_manu),
    }


def compute_convergence_orders(errors_list):
    """Compute convergence orders for each norm."""
    orders = {}
    norm_keys = ["velocity_L2", "velocity_H1", "velocity_H2", 
                 "pressure_L2", "concentration_L2", "concentration_H1", "concentration_H2"]
    
    for key in norm_keys:
        orders[key] = []
        for i in range(1, len(errors_list)):
            if errors_list[i-1][key] > 0 and errors_list[i][key] > 0:
                order = np.log(errors_list[i][key] / errors_list[i-1][key]) / np.log(errors_list[i]["h"] / errors_list[i-1]["h"])
                orders[key].append(order)
            else:
                orders[key].append(np.nan)
    
    return orders


# ============================================================================
# PART 3: Run MMS on a Single Mesh
# ============================================================================

def run_mms_on_mesh(mesh_file):
    """Run MMS test on a specific mesh file and return errors."""
    if comm.rank == 0:
        print(f"\n  Running on: {mesh_file}")
    
    # Load mesh
    try:
        mesh_data = io.gmsh.read_from_msh(mesh_file, comm, gdim=2)
        domain = mesh_data.mesh
        facet_tags = mesh_data.facet_tags
    except Exception as e:
        if comm.rank == 0:
            print(f"    ERROR: Could not load mesh {mesh_file}")
        return None
    
    # Function spaces: P2 for velocity, P1 for pressure, P2 for concentration
    V = fem.functionspace(domain, ("Lagrange", 2, (2,)))   # Velocity P2
    Q = fem.functionspace(domain, ("Lagrange", 1))         # Pressure P1
    W = fem.functionspace(domain, ("Lagrange", 2))         # Concentration P2
    
    # Test and trial functions
    u, v = TrialFunction(V), TestFunction(V)
    p, q = TrialFunction(Q), TestFunction(Q)
    c, w = TrialFunction(W), TestFunction(W)
    
    x = SpatialCoordinate(domain)
    
    # Parameters
    nu = fem.Constant(domain, 1e-3)
    D = fem.Constant(domain, 5e-4)
    
    # Manufactured solutions
    u_manu = u_exact(x)
    p_manu = p_exact(x)
    c_manu = c_exact(x)
    
    # Source terms
    f_stokes = -nu * div(grad(u_manu)) + grad(p_manu)
    s_advdiff = dot(u_manu, grad(c_manu)) - div(D * grad(c_manu))
    
    # Boundary conditions
    u_bc = fem.Function(V)
    u_bc.interpolate(u_exact_numpy)
    c_bc = fem.Function(W)
    c_bc.interpolate(c_exact_numpy)
    
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    boundary_facets = dmesh.locate_entities_boundary(
        domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
    )
    
    bc_u = fem.dirichletbc(u_bc, fem.locate_dofs_topological(V, fdim, boundary_facets))
    bc_c = fem.dirichletbc(c_bc, fem.locate_dofs_topological(W, fdim, boundary_facets))
    
    # STOKES WEAK FORM
    a_stokes = [
        [nu * inner(grad(u), grad(v)) * dx, -p * div(v) * dx],
        [-q * div(u) * dx, None],
    ]
    L_stokes = [inner(f_stokes, v) * dx, fem.Constant(domain, 0.0) * q * dx]
    
    # ADVECTION-DIFFUSION WEAK FORM
    a_adv = D * dot(grad(c), grad(w)) * dx + dot(u_manu, grad(c)) * w * dx
    L_adv = s_advdiff * w * dx
    
    # Solve Stokes
    problem_stokes = LinearProblem(
        a_stokes, L_stokes,
        bcs=[bc_u],
        kind="nest",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="mms_stokes_",
    )
    uh, ph = problem_stokes.solve()
    uh.x.scatter_forward()
    ph.x.scatter_forward()
    
    # Fix pressure mean to zero
    dx_domain = dx(domain=domain)
    volume = fem.assemble_scalar(fem.form(1 * dx_domain))
    p_mean = fem.assemble_scalar(fem.form(ph * dx_domain)) / volume
    ph.x.array[:] -= p_mean
    ph.x.scatter_forward()
    
    # Also fix the manufactured pressure to have zero mean
    p_manu_mean = fem.assemble_scalar(fem.form(p_manu * dx_domain)) / volume
    p_manu_zero_mean = p_manu - p_manu_mean
    
    # Solve Advection-Diffusion
    problem_adv = LinearProblem(
        a_adv, L_adv,
        bcs=[bc_c],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="mms_advdiff_",
    )
    ch = problem_adv.solve()
    ch.x.scatter_forward()
    
    # Compute errors
    h_actual = compute_mesh_size(domain)
    errors = compute_errors(domain, uh, ph, ch, u_manu, p_manu_zero_mean, c_manu)
    errors["h"] = h_actual
    errors["dofs"] = V.dofmap.index_map.size_local + Q.dofmap.index_map.size_local + W.dofmap.index_map.size_local
    
    if comm.rank == 0:
        print(f"    h = {errors['h']:.6f}, DOFs = {errors['dofs']}")
        print(f"    Velocity L2: {errors['velocity_L2']:.6e}")
        print(f"    Pressure L2: {errors['pressure_L2']:.6e}")
        print(f"    Concentration L2: {errors['concentration_L2']:.6e}")
    
    return errors


# ============================================================================
# PART 4: Print Results
# ============================================================================

def print_results(all_errors, orders):
    """Print convergence tables and summary."""
    if comm.rank != 0:
        return
    
    # TABLE 1: VELOCITY
    print("\n" + "="*90)
    print("TABLE 1: VELOCITY CONVERGENCE")
    print("="*90)
    print(f"{'h':<12} {'L2 Error':<18} {'Order':<10} {'H1 Error':<18} {'Order':<10} {'H2 Error':<18} {'Order':<10}")
    print("-"*110)
    
    for i, errors in enumerate(all_errors):
        print(f"{errors['h']:<12.6f}", end="")
        
        # L2
        print(f"{errors['velocity_L2']:<18.6e}", end="")
        if i > 0 and not np.isnan(orders['velocity_L2'][i-1]):
            print(f"{orders['velocity_L2'][i-1]:<10.3f}", end="")
        else:
            print(f"{'—':<10}", end="")
        
        # H1
        print(f"{errors['velocity_H1']:<18.6e}", end="")
        if i > 0 and not np.isnan(orders['velocity_H1'][i-1]):
            print(f"{orders['velocity_H1'][i-1]:<10.3f}", end="")
        else:
            print(f"{'—':<10}", end="")
        
        # H2
        print(f"{errors['velocity_H2']:<18.6e}", end="")
        if i > 0 and not np.isnan(orders['velocity_H2'][i-1]):
            print(f"{orders['velocity_H2'][i-1]:<10.3f}", end="")
        else:
            print(f"{'—':<10}", end="")
        
        print()
    
    print("-"*110)
    print("Expected: L2: O(h³)≈3.0, H1: O(h²)≈2.0")
    
    # TABLE 2: PRESSURE
    print("\n" + "="*90)
    print("TABLE 2: PRESSURE CONVERGENCE")
    print("="*90)
    print(f"{'h':<12} {'L2 Error':<18} {'Order':<10}")
    print("-"*50)
    
    for i, errors in enumerate(all_errors):
        print(f"{errors['h']:<12.6f}", end="")
        print(f"{errors['pressure_L2']:<18.6e}", end="")
        if i > 0 and not np.isnan(orders['pressure_L2'][i-1]):
            print(f"{orders['pressure_L2'][i-1]:<10.3f}", end="")
        else:
            print(f"{'—':<10}", end="")
        print()
    
    print("-"*50)
    print("Expected: L2: O(h²)≈2.0")
    
    # TABLE 3: CONCENTRATION
    print("\n" + "="*90)
    print("TABLE 3: CONCENTRATION CONVERGENCE")
    print("="*90)
    print(f"{'h':<12} {'L2 Error':<18} {'Order':<10} {'H1 Error':<18} {'Order':<10} {'H2 Error':<18} {'Order':<10}")
    print("-"*110)
    
    for i, errors in enumerate(all_errors):
        print(f"{errors['h']:<12.6f}", end="")
        
        # L2
        print(f"{errors['concentration_L2']:<18.6e}", end="")
        if i > 0 and not np.isnan(orders['concentration_L2'][i-1]):
            print(f"{orders['concentration_L2'][i-1]:<10.3f}", end="")
        else:
            print(f"{'—':<10}", end="")
        
        # H1
        print(f"{errors['concentration_H1']:<18.6e}", end="")
        if i > 0 and not np.isnan(orders['concentration_H1'][i-1]):
            print(f"{orders['concentration_H1'][i-1]:<10.3f}", end="")
        else:
            print(f"{'—':<10}", end="")
        
        # H2
        print(f"{errors['concentration_H2']:<18.6e}", end="")
        if i > 0 and not np.isnan(orders['concentration_H2'][i-1]):
            print(f"{orders['concentration_H2'][i-1]:<10.3f}", end="")
        else:
            print(f"{'—':<10}", end="")
        
        print()
    
    print("-"*110)
    print("Expected: L2: O(h³)≈3.0, H1: O(h²)≈2.0")
    
    # SUMMARY
    print("\n" + "="*90)
    print("SUMMARY")
    print("="*90)
    
    final_errors = all_errors[-1]
    print(f"Final errors (finest mesh, h = {final_errors['h']:.6f}):")
    print(f"  Velocity L2: {final_errors['velocity_L2']:.6e}")
    print(f"  Velocity H1: {final_errors['velocity_H1']:.6e}")
    print(f"  Velocity H2: {final_errors['velocity_H2']:.6e}")
    print(f"  Pressure L2: {final_errors['pressure_L2']:.6e}")
    print(f"  Concentration L2: {final_errors['concentration_L2']:.6e}")
    print(f"  Concentration H1: {final_errors['concentration_H1']:.6e}")
    
    print("\n" + "-"*90)
    print("CONVERGENCE CHECK:")
    
    if len(orders['pressure_L2']) > 0 and not np.isnan(orders['pressure_L2'][-1]):
        final_p_order = orders['pressure_L2'][-1]
        if final_p_order > 1.5:
            print(f"  ✅ Pressure converges: Order ≈ {final_p_order:.3f} (Expected ≈ 2.0)")
        else:
            print(f"  ❌ Pressure does NOT converge: Order ≈ {final_p_order:.3f}")
    else:
        print("  ⚠️ Pressure order not available")
    
    if len(orders['velocity_L2']) > 0 and not np.isnan(orders['velocity_L2'][-1]):
        final_v_order = orders['velocity_L2'][-1]
        if final_v_order > 2.5:
            print(f"  ✅ Velocity L2 converges: Order ≈ {final_v_order:.3f} (Expected ≈ 3.0)")
        else:
            print(f"  ❌ Velocity L2 does NOT converge: Order ≈ {final_v_order:.3f}")
    else:
        print("  ⚠️ Velocity L2 order not available")
    
    if len(orders['concentration_L2']) > 0 and not np.isnan(orders['concentration_L2'][-1]):
        final_c_order = orders['concentration_L2'][-1]
        if final_c_order > 2.5:
            print(f"  ✅ Concentration L2 converges: Order ≈ {final_c_order:.3f} (Expected ≈ 3.0)")
        else:
            print(f"  ❌ Concentration L2 does NOT converge: Order ≈ {final_c_order:.3f}")
    else:
        print("  ⚠️ Concentration L2 order not available")
    
    print("="*90)


# ============================================================================
# PART 5: Main Program
# ============================================================================

def run_convergence_study():
    """Run MMS on a sequence of refined meshes and compute convergence orders."""
    if comm.rank == 0:
        print("\n" + "="*90)
        print("MMS CONVERGENCE STUDY: Stokes + Advection-Diffusion Solver")
        print("="*90)
        print("Manufactured Solutions:")
        print("  u = (sin(πy), 0)")
        print("  p = cos(πx)")
        print("  c = 1 + x² + y²")
        print("="*90)
        print("Boundary Conditions:")
        print("  - Velocity: Dirichlet on all boundaries")
        print("  - Pressure: Natural (mean fixed to zero)")
        print("  - Concentration: Dirichlet on all boundaries")
        print("="*90)
    
    mesh_sequence = [
        "dfg_benchmark_4obstacles_coarse.msh",
        "dfg_benchmark_4obstacles_medium.msh",
        "dfg_benchmark_4obstacles_fine.msh",
        "dfg_benchmark_4obstacles_veryfine.msh",
    ]
    
    import os
    actual_meshes = [m for m in mesh_sequence if os.path.exists(m)]
    
    if len(actual_meshes) == 0:
        if comm.rank == 0:
            print("\nNo refined meshes found. Please generate them first:")
            print("  gmsh dfg_benchmark_4obstacles.geo -2 -clscale 2.0 -o dfg_benchmark_4obstacles_coarse.msh")
            print("  gmsh dfg_benchmark_4obstacles.geo -2 -clscale 1.0 -o dfg_benchmark_4obstacles_medium.msh")
            print("  gmsh dfg_benchmark_4obstacles.geo -2 -clscale 0.5 -o dfg_benchmark_4obstacles_fine.msh")
            print("  gmsh dfg_benchmark_4obstacles.geo -2 -clscale 0.25 -o dfg_benchmark_4obstacles_veryfine.msh")
        return
    
    all_errors = []
    for mesh_file in actual_meshes:
        errors = run_mms_on_mesh(mesh_file)
        if errors is not None:
            all_errors.append(errors)
    
    if len(all_errors) == 0:
        if comm.rank == 0:
            print("\nNo meshes could be loaded. Exiting.")
        return
    
    all_errors.sort(key=lambda x: x["h"], reverse=True)
    orders = compute_convergence_orders(all_errors)
    print_results(all_errors, orders)


if __name__ == "__main__":
    run_convergence_study()