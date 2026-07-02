"""
================================================================================
METHOD OF MANUFACTURED SOLUTIONS (MMS) TEST
================================================================================

Purpose:
    Verify that the Stokes + advection-diffusion solver is correctly implemented
    by testing it on different mesh geometries.

How it works:
    1. We add "exact" solution (manufactured solution)
    2. We calculate what source term would produce this solution
    3. We run our solver with this source term
    4. We compare the solver's output to the exact solution
    5. If the error is small, our solver is correct!

Manufactured Solutions:
    Velocity:  u = (sin(πy), 0)
    Pressure:  p = cos(πx)
    Concentration: c = 1 + x² + y²

Passing criteria:
    Velocity L2 < 1e-4
    Pressure L2 < 1e-1
    Concentration L2 < 1e-12

This test automatically runs on all meshes: 0, 4, and 6 obstacles.
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

# PART 1: Manufactured Solutions (The "Fake" Exact Answers)

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

# PART 2: MMS Test Function (Runs on Any Mesh)

def run_mms_test(mesh_name, expected_obstacles):
    """
    Run the MMS test on a specific mesh file.
    
    Parameters:
        mesh_name: Name of the mesh file (e.g., "dfg_benchmark_0obstacles.msh")
        expected_obstacles: Expected number of obstacle facets (for verification)
    
    Returns:
        Dictionary with error values and pass/fail status
    """
    
    if comm.rank == 0:
        print("\n" + "="*70)
        print(f"MMS TEST: {mesh_name}")
        print("="*70)
        print(f"Expected obstacles: {expected_obstacles}")
        print("-"*70)
        print("Manufactured Solutions:")
        print("  u = (sin(πy), 0)")
        print("  p = cos(πx)")
        print("  c = 1 + x² + y²")
        print("-"*70)
    
    # Load the mesh
  
    try:
        mesh_data = io.gmsh.read_from_msh(mesh_name, comm, gdim=2)
        domain = mesh_data.mesh
        facet_tags = mesh_data.facet_tags
    except Exception as e:
        if comm.rank == 0:
            print(f"❌ ERROR: Could not load mesh {mesh_name}")
            print(f"   {e}")
        return None
    
    if comm.rank == 0:
        obstacle_facets = facet_tags.find(3)
        print(f"Obstacle facets found: {len(obstacle_facets)}")
    
    # Function spaces
   
    degree = 1
    V = fem.functionspace(domain, ("Lagrange", degree + 1, (2,)))  # Velocity P2
    Q = fem.functionspace(domain, ("Lagrange", degree))            # Pressure P1
    W = fem.functionspace(domain, ("Lagrange", degree + 1))        # Concentration P2
    
    u, v = TrialFunction(V), TestFunction(V)
    p, q = TrialFunction(Q), TestFunction(Q)
    c, w = TrialFunction(W), TestFunction(W)
    
    x = SpatialCoordinate(domain)
    
    # Parameters
  
    nu = fem.Constant(domain, 1e-3)
    D = fem.Constant(domain, 5e-4)
    
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
    

    # Weak forms
   
    a_stokes = [
        [nu * inner(grad(u), grad(v)) * dx, -p * div(v) * dx],
        [-q * div(u) * dx, None],
    ]
    L_stokes = [inner(f_stokes, v) * dx, fem.Constant(domain, 0.0) * q * dx]
    
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
    
    # Fix pressure: remove the mean value
    dx_domain = dx(domain=domain)
    volume = fem.assemble_scalar(fem.form(1 * dx_domain))
    p_mean = fem.assemble_scalar(fem.form(ph * dx_domain)) / volume
    
    if comm.rank == 0:
        print(f"Pressure mean before fixing: {p_mean:.6f}")
    
    ph.x.array[:] -= p_mean
    ph.x.scatter_forward()
    
    if comm.rank == 0:
        p_mean_new = fem.assemble_scalar(fem.form(ph * dx_domain)) / volume
        print(f"Pressure mean after fixing:  {p_mean_new:.6f}")
    
    
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
    
    # Calculate all errors
    errors = {
        "velocity_L2": L2_error(uh, u_manu),
        "velocity_H1": H1_error(uh, u_manu),
        "pressure_L2": L2_error(ph, p_manu),
        "concentration_L2": L2_error(ch, c_manu),
        "concentration_H1": H1_error(ch, c_manu),
    }
    
    
    # Print results
  
    if comm.rank == 0:
        print("\n" + "="*70)
        print(f"MMS ERROR RESULTS ({mesh_name})")
        print("="*70)
        print(f"Velocity L2 error:      {errors['velocity_L2']:.6e}")
        print(f"Velocity H1 error:      {errors['velocity_H1']:.6e}")
        print(f"Pressure L2 error:      {errors['pressure_L2']:.6e}")
        print(f"Concentration L2 error: {errors['concentration_L2']:.6e}")
        print(f"Concentration H1 error: {errors['concentration_H1']:.6e}")
        print("="*70)
        
        # Check passing criteria
        pass_u = errors['velocity_L2'] < 1e-4
        pass_p = errors['pressure_L2'] < 1e-1
        pass_c = errors['concentration_L2'] < 1e-12
        
        print("\nVerification Criteria:")
        print(f"  ✓ Velocity L2 < 1e-4:  {'✓ PASS' if pass_u else '✗ FAIL'}")
        print(f"  ✓ Pressure L2 < 1e-1:  {'✓ PASS' if pass_p else '✗ FAIL'}")
        print(f"  ✓ Concentration L2 < 1e-12: {'✓ PASS' if pass_c else '✗ FAIL'}")
        
        if pass_u and pass_p and pass_c:
            print(f"\n✅ ALL TESTS PASSED! Solver verified on {mesh_name}.")
        else:
            print(f"\n⚠️  Some tests failed on {mesh_name}. Check implementation.")
        
        print("="*70)
    
    return errors


# PART 3: Main Program - Run MMS on All Meshes


if __name__ == "__main__":
    if comm.rank == 0:
        print("\n" + "="*70)
        print("MMS VERIFICATION: Stokes + Advection-Diffusion Solver")
        print("="*70)
        print("Testing on: 0 obstacles, 4 obstacles, 6 obstacles")
        print("="*70)
    
    # List of meshes to test
    meshes_to_test = [
        ("dfg_benchmark_0obstacles.msh", 0),
        ("dfg_benchmark_4obstacles.msh", 4),
        ("dfg_benchmark_6obstacles.msh", 6),
    ]
    
    # Store results
    all_results = {}
    
    # Run tests
    for mesh_name, expected in meshes_to_test:
        results = run_mms_test(mesh_name, expected)
        if results is not None:
            all_results[mesh_name] = results
    
    # Summary
    if comm.rank == 0:
        print("\n" + "="*70)
        print("SUMMARY: MMS TEST RESULTS")
        print("="*70)
        print(f"{'Mesh':<35} {'Velocity L2':<15} {'Pressure L2':<15} {'Concentration L2':<15} {'Status':<10}")
        print("-"*90)
        
        all_passed = True
        for mesh_name, errors in all_results.items():
            passed = (errors['velocity_L2'] < 1e-4 and 
                     errors['pressure_L2'] < 1e-1 and 
                     errors['concentration_L2'] < 1e-12)
            status = "✅ PASS" if passed else "❌ FAIL"
            if not passed:
                all_passed = False
            print(f"{mesh_name:<35} {errors['velocity_L2']:<15.6e} {errors['pressure_L2']:<15.6e} {errors['concentration_L2']:<15.6e} {status:<10}")
        
        print("-"*90)
        if all_passed and len(all_results) > 0:
            print("\n ALL MMS TESTS PASSED! Your solver is verified on all meshes.")
        else:
            print("\n Some tests failed. Please check the implementation.")
        print("="*70)