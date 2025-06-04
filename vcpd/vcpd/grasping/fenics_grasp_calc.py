#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# sudo apt install python3-petsc4py python3-dolfinx
"""
Created on Tue Nov  5 15:34:12 2024

@author: alexmbcm
"""


# Meshing libraries
import pygalmesh
import meshio


# Fenicsx FEA libraries
from petsc4py import PETSc
from mpi4py import MPI
import ufl
from dolfinx import mesh, fem, default_scalar_type, io
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc #,create_vector
import basix


# Timing
import time
import os

# Common libraries
import numpy as np

np.set_printoptions(formatter={'float': lambda x: format(x, '6.3E')})

class FEAGraspTester:
    def __init__(self) -> None:
        ## LOAD STL AND MAKE VOLUMETRIC MESH
        ###############################################################################

        # Check initial stl mesh
        # print(self.stl_mesh)
        # print('\n')
        pass

    def initialize_fea(self):


        # Check node numbering is correct 
        # print(self.domain.geometry.x[grasp_node_nums])
        # print('\n')

        self.E  = 3.5e9
        self.nu = 0.35

        self.lambda_ = self.E * self.nu / ((1.0 + self.nu) * (1 - 2.0*self.nu))
        mu      = 0.5 * self.E / (1.0 + self.nu)

        self.V = fem.functionspace(self.domain, ("Lagrange", 1, (self.domain.geometry.dim, )))
        self.fdim = self.domain.topology.dim - 1
        self.u_D = np.array([0, 0, 0], dtype=default_scalar_type)
        T = fem.Constant(self.domain, default_scalar_type((0, 0, 0)))
        ds = ufl.Measure("ds", domain=self.domain)

        def epsilon(u):
            return ufl.sym(ufl.grad(u)) # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

        def sigma(u):
            return self.lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)


        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
        L = ufl.dot(T, v) * ds

        ## Solution setup
        self.a_compiled = fem.form(a)
        self.L_compiled = fem.form(L)

        # Create solution function
        self.uh = fem.Function(self.V)

        # Solver
        self.solver = PETSc.KSP().create(self.domain.comm)
        self.solver.setType(PETSc.KSP.Type.PREONLY)
        self.solver.getPC().setType(PETSc.PC.Type.LU)

    def re_mesh_object(self, obj_name, mesh_dir, surface_mesh_output_dir):
        # Get new resampled surface mesh and its volume mesh
        obj_file_path = os.path.join(mesh_dir, obj_name + ".obj")
        self.stl_mesh = meshio.read(obj_file_path)
        t0 = time.time()
        
        pygalmesh_vol_mesh = pygalmesh.generate_volume_mesh_from_surface_mesh(
            obj_file_path,
            min_facet_angle=25.0,
            max_radius_surface_delaunay_ball=0.002,
            max_facet_distance=0.0004,
            max_circumradius_edge_ratio=1.2,
            verbose=False,
        )
        t1 = time.time()

        # print(self.pygalmesh_vol_mesh)
        # print('\n')
        ## MAKE FEA MESH AND EXPORT SURFACE MESH
        ###############################################################################

        # Import nodes and connectivity matrix into Fenics domain and mesh objects
        self.points = pygalmesh_vol_mesh.points
        self.cells  = pygalmesh_vol_mesh.cells[1].data

        finiteElement = basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3, ))

        self.domain = mesh.create_mesh(MPI.COMM_WORLD, self.cells, self.points, finiteElement)


        # Export mesh
        # Find facets on boundary
        fdim = self.domain.topology.dim - 1
        self.domain.topology.create_connectivity(fdim, fdim+1)
        self.free_end_facets = mesh.exterior_facet_indices(self.domain.topology)

        # Connectivity between nodes and boundary facets
        self.free_end_connectivity = mesh.entities_to_geometry(self.domain, fdim, self.free_end_facets)

        # Get only unique nodes
        self.free_end_nodes = np.unique(self.free_end_connectivity.reshape(-1))

        # Extract coordinates of nodes associated with the surface facets
        self.free_end_coords = self.domain.geometry.x[self.free_end_nodes]


        # Change connectivity numbering to match order from 0 of extracted nodes
        self.connectivity_copy = np.copy(self.free_end_connectivity)

        for n in np.arange(self.free_end_nodes.size):
            self.free_end_connectivity[self.connectivity_copy == self.free_end_nodes[n]] = n


        meshio.write_points_cells(
            surface_mesh_output_dir + f"{obj_name}.fenics_surf_mesh.obj", 
            self.free_end_coords, 
            {'triangle': self.free_end_connectivity}
        )

## FENICS SIMULATIONS BASED ON ARRAYS WITH NODES AND CORRESPONDING FACETS FROM GRASPING CODE
###############################################################################
    def find_grasp_displacement(self, grasp_facet_nums, grasp_node_nums, grasp_node_normals) -> float:

# grasp_facet_nums = np.array([234, 2386, 352, 2044, 923, 2078])

# grasp_node_nums  = np.array([50, 200, 200, 400, 500, 1050])

# grasp_node_normals = np.array([[ 1.0, 0.0, 0.0],
#                                [ 0.0, 1.0, 0.0],
#                                [ 0.0, 1.0, 0.0],
#                                [-0.08343754, 0.99600488, -0.03182245],
#                                [-1.0, 0.0, 0.0],
#                                [ 0.0, 1.0, 0.0]])

        t0 = time.time()
        # Change facet and node numbering back to Fenics numbering
        grasp_facet_nums = self.free_end_facets[grasp_facet_nums]
        grasp_node_nums  = self.free_end_nodes[grasp_node_nums]



        # Actual solver \/\/
        t2 = time.time()
        # print("GRASP FACET NUMS SIZE:", grasp_facet_nums.size)
        boundary_facets = np.array([grasp_facet_nums[0]])
        
        bc = fem.dirichletbc(self.u_D, fem.locate_dofs_topological(self.V, self.fdim, boundary_facets), self.V)
        
        # Assemble system, applying boundary conditions
        A = assemble_matrix(self.a_compiled, bcs=[bc])
        A.assemble()
        
        b = assemble_vector(self.L_compiled)
        apply_lifting(b, [self.a_compiled], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, [bc])
        
        global_node = grasp_node_nums[0]
        b.setValue(3*global_node,   -grasp_node_normals[0,0])
        b.setValue(3*global_node+1, -grasp_node_normals[0,1])
        b.setValue(3*global_node+2, -grasp_node_normals[0,2])
        
        # Solver
        self.solver.setOperators(A)
        
        # Compute solution
        self.solver.solve(b, self.uh.x.petsc_vec)
        
        #Store max displacement from solution
        max_displacement = np.max(np.linalg.norm(np.reshape(self.uh.x.array, (-1,3)), axis=1))
    
        # Disable printing of deformation to files.
        # with io.XDMFFile(self.domain.comm, "deformation" + str(n) + ".xdmf", "w") as xdmf:
        #     xdmf.write_mesh(self.domain)
        #     uh.name = "Deformation"
        #     xdmf.write_function(uh)

        t3 = time.time()

        # print(max_displacement)
        # print('\n')


        return max_displacement, t3 - t0














