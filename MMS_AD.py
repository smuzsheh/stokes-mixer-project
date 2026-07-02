""""
Method of Manufactured Solutions (MMS) test for the advection–diffusion equation
for a concentration c.

We test the concentration equation on the DFG benchmark mesh:

    -div(D ∇c) + u · ∇c = s   in Ω
                c = c_exact  on ∂Ω

We prescribe:
    u_exact(x,y) = (sin(pi*y), 0)
    c_exact(x,y) = 1 + x^2 + y^2

Then we compute the matching source term s so that c_exact is the exact solution.
At the end we solve the FEM problem and compute the L2 and H1 error on the DFG mesh.
"""
import numpy as np
from mpi4py import MPI
from dolfinx import fem, io
from dolfinx.fem.petsc import LinearProblem
from ufl import (
    div,
    dx,
    grad,
    TrialFunction,
    TestFunction,
    dot,
    SpatialCoordinate,
    as_vector,
    sin,
    pi
)


def c_exact_numpy(x):
    return 1.0 + x[0]**2 + x[1]**2


#  load the DFG mesh 
mesh_data = io.gmsh.read_from_msh("dfg_benchmark_2d.msh", MPI.COMM_WORLD, gdim=2)
domain = mesh_data.mesh
facet_tags = mesh_data.facet_tags


# Function space
degree = 1
V = fem.functionspace(domain, ("Lagrange", degree))

# Trial and Testfunctions
c = TrialFunction(V)
v = TestFunction(V)

x = SpatialCoordinate(domain)
D = fem.Constant(domain, 5e-4)

# Manufactured terms
u_exact = as_vector((sin(pi * x[1]), 0.0))
c_exact = 1.0 + x[0]**2 + x[1]**2

# Source term
source_term = dot(u_exact, grad(c_exact)) - div(D * grad(c_exact))

# ── Only change in BCs: apply c_exact on ALL boundary facets 
cD = fem.Function(V)
cD.interpolate(c_exact_numpy)

fdim = domain.topology.dim - 1
domain.topology.create_connectivity(fdim, domain.topology.dim)
from dolfinx import mesh as dmesh
boundary_facets = dmesh.locate_entities_boundary(
    domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
)
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(cD, boundary_dofs)


# Weak form  
a = D * dot(grad(c), grad(v)) * dx + dot(u_exact, grad(c)) * v * dx
L = source_term * v * dx

# Solver  
problem = LinearProblem(
    a,
    L,
    bcs=[bc],
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu"
    },
    petsc_options_prefix="mms_concentration_",
)

ch = problem.solve()

# L2 error 
L2_error = fem.form((ch - c_exact) * (ch - c_exact) * dx)
error_local = fem.assemble_scalar(L2_error)
error_L2 = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))

# H1 error  
H1_error = fem.form(
    (ch - c_exact) * (ch - c_exact) * dx
    + dot(grad(ch - c_exact), grad(ch - c_exact)) * dx
)
error_local = fem.assemble_scalar(H1_error)
error_H1 = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))

if domain.comm.rank == 0:
    print("MMS advection-diffusion on DFG benchmark mesh")
    print(f"   L²-error: {error_L2:.2e}")
    print(f"   H¹-error: {error_H1:.2e}")