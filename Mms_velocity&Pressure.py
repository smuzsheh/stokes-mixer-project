""""
Method of Manufactured Solutions (MMS) test for the steady Stokes equations
(velocity u and pressure p).

We test the Stokes system on the DFG benchmark mesh:

    -nu * Δu + ∇p = f   in Ω
          div(u) = 0   in Ω
           u = u_exact on ∂Ω

We prescribe:
    u_exact(x,y) = (sin(pi*y), 0)
    p_exact(x,y) = cos(pi*x)

Then we compute the matching forcing term f so that (u_exact, p_exact) is the exact solution.
At the end we solve the FEM Stokes problem and compute the L2 and H1 error of u
on the DFG benchmark mesh.
"""

import numpy as np
from mpi4py import MPI
from dolfinx import fem, io
from dolfinx import mesh as dmesh
from dolfinx.fem.petsc import LinearProblem
from ufl import (
    as_vector,
    div,
    dx,
    grad,
    inner,
    SpatialCoordinate,
    TrialFunction,
    TestFunction,
    sin,
    cos,
    pi,
)


def u_exact_numpy(x):
    return np.vstack((np.sin(np.pi * x[1]), np.zeros_like(x[1])))


#  load the DFG mesh 
mesh_data = io.gmsh.read_from_msh("dfg_benchmark_2d.msh", MPI.COMM_WORLD, gdim=2)
domain = mesh_data.mesh
facet_tags = mesh_data.facet_tags


# Function space  
degree = 1
V = fem.functionspace(domain, ("Lagrange", degree + 1, (2,)))  # Velocity
Q = fem.functionspace(domain, ("Lagrange", degree))            # Pressure

# Trial and Testfunctions  
u = TrialFunction(V)
p = TrialFunction(Q)
v = TestFunction(V)
q = TestFunction(Q)

x = SpatialCoordinate(domain)

# Parameters  
nu = fem.Constant(domain, 1e-3)

# Manufactured terms  
u_exact = as_vector((sin(pi * x[1]), 0.0))
p_exact = cos(pi * x[0])

# Forcing term  
f = -nu * div(grad(u_exact)) + grad(p_exact)

# Only change in BCs: apply u_exact on ALL boundary facets
uD = fem.Function(V)
uD.interpolate(u_exact_numpy)

fdim = domain.topology.dim - 1
domain.topology.create_connectivity(fdim, domain.topology.dim)
boundary_facets = dmesh.locate_entities_boundary(
    domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
)
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(uD, boundary_dofs)


# Weak form  
a = [
    [nu * inner(grad(u), grad(v)) * dx, -p * div(v) * dx],
    [-q * div(u) * dx, None],
]
L = [inner(f, v) * dx, fem.Constant(domain, 0.0) * q * dx]

# Preconditioner  
a_p = [[a[0][0], None], [None, p * q * dx]]

# Solver  (unchanged)
problem = LinearProblem(
    a,
    L,
    bcs=[bc],
    kind="nest",
    P=a_p,
    petsc_options={
        "ksp_type": "gmres",
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "additive",
        "fieldsplit_0_ksp_type": "preonly",
        "fieldsplit_0_pc_type": "hypre",
        "fieldsplit_0_pc_hypre_type": "boomeramg",
        "fieldsplit_1_ksp_type": "preonly",
        "fieldsplit_1_pc_type": "jacobi",
    },
    petsc_options_prefix="mms_stokes_dfg_",
)

uh, ph = problem.solve()

# Velocity L2 error  
L2_u_form = fem.form(inner(uh - u_exact, uh - u_exact) * dx)
L2_u_local = fem.assemble_scalar(L2_u_form)
L2_u = np.sqrt(domain.comm.allreduce(L2_u_local, op=MPI.SUM))

# Velocity H1 error  
H1_u_form = fem.form(
    inner(uh - u_exact, uh - u_exact) * dx
    + inner(grad(uh - u_exact), grad(uh - u_exact)) * dx
)
H1_u_local = fem.assemble_scalar(H1_u_form)
H1_u = np.sqrt(domain.comm.allreduce(H1_u_local, op=MPI.SUM))

if domain.comm.rank == 0:
    num_cells = domain.topology.index_map(domain.topology.dim).size_global
    num_dofs_V = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    print("MMS Stokes on DFG benchmark mesh")
    print(f"   Mesh cells:          {num_cells}")
    print(f"   Velocity DOFs:       {num_dofs_V}")
    print(f"   L²-error (velocity): {L2_u:.2e}")
    print(f"   H¹-error (velocity): {H1_u:.2e}")