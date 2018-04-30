"""Unit tests for assembly"""

# Copyright (C) 2011-2018 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pytest
import os
import numpy
import dolfin
import ufl
from ufl import dx, ds, dS

from dolfin_utils.test import skip_in_parallel, filedir, pushpop_parameters


def test_cell_size_assembly_1D():
    mesh = dolfin.generation.UnitIntervalMesh(dolfin.MPI.comm_world, 10)
    assert round(dolfin.fem.assembling.assemble_scalar(2*dolfin.Circumradius(mesh)*dx) - 0.1, 12) == 0
    assert round(dolfin.fem.assembling.assemble_scalar(dolfin.CellDiameter(mesh)*dx) - 0.1, 12) == 0
    assert round(dolfin.fem.assembling.assemble_scalar(dolfin.CellVolume(mesh)*dx) - 0.1, 12) == 0


def test_cell_assembly_1D():
    mesh = dolfin.generation.UnitIntervalMesh(dolfin.MPI.comm_world, 48)
    V = dolfin.FunctionSpace(mesh, "CG", 1)

    v = dolfin.TestFunction(V)
    u = dolfin.TrialFunction(V)
    f = dolfin.Constant(10.0)

    a = ufl.inner(ufl.grad(v), ufl.grad(u))*dx
    L = ufl.inner(v, f)*dx

    A_frobenius_norm = 811.75365721381274397572
    b_l2_norm = 1.43583841167606474087

    # Assemble A and b
    assembler = dolfin.fem.assembling.Assembler(a, L, [])
    A = dolfin.cpp.la.PETScMatrix(dolfin.MPI.comm_world)
    b = dolfin.cpp.la.PETScVector(dolfin.MPI.comm_world)

    assembler.assemble(A)
    assembler.assemble(b)

    assert round(A.norm("frobenius") - A_frobenius_norm, 10) == 0
    assert round(b.norm("l2") - b_l2_norm, 10) == 0


def test_cell_assembly():
    mesh = dolfin.generation.UnitCubeMesh(dolfin.MPI.comm_world, 4, 4, 4)
    V = dolfin.VectorFunctionSpace(mesh, "DG", 1)

    v = dolfin.TestFunction(V)
    u = dolfin.TrialFunction(V)
    f = dolfin.Constant((10, 20, 30))

    def epsilon(v):
        return 0.5*(ufl.grad(v) + ufl.grad(v).T)

    a = ufl.inner(epsilon(v), epsilon(u))*dx
    L = ufl.inner(v, f)*dx

    A_frobenius_norm = 4.3969686527582512
    b_l2_norm = 0.95470326978246278

    # Assemble A and b
    assembler = dolfin.fem.assembling.Assembler(a, L, [])
    A = dolfin.cpp.la.PETScMatrix(dolfin.MPI.comm_world)
    b = dolfin.cpp.la.PETScVector(dolfin.MPI.comm_world)

    assembler.assemble(A)
    assembler.assemble(b)

    assert round(A.norm("frobenius") - A_frobenius_norm, 10) == 0
    assert round(b.norm("l2") - b_l2_norm, 10) == 0


@skip_in_parallel  # Ghost mode not working yet
def test_facet_assembly():
    mesh = dolfin.UnitSquareMesh(dolfin.MPI.comm_world, 24, 24)
    V = dolfin.FunctionSpace(mesh, "DG", 1)

    # Define test and trial functions
    v = dolfin.TestFunction(V)
    u = dolfin.TrialFunction(V)

    # Define normal component, mesh size and right-hand side
    n = ufl.FacetNormal(mesh)
    h = 2*ufl.Circumradius(mesh)
    h_avg = (h('+') + h('-'))/2
    f = dolfin.Expression("500.0*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=1)

    # Define bilinear form
    a = ufl.dot(ufl.grad(v), ufl.grad(u))*dx \
        - ufl.dot(ufl.avg(ufl.grad(v)), ufl.jump(u, n))*dS \
        - ufl.dot(ufl.jump(v, n), ufl.avg(ufl.grad(u)))*dS \
        + 4.0/h_avg*ufl.dot(ufl.jump(v, n), ufl.jump(u, n))*dS \
        - ufl.dot(ufl.grad(v), u*n)*ds \
        - ufl.dot(v*n, ufl.grad(u))*ds \
        + 8.0/h*v*u*ds

    # Define linear form
    L = v*f*dx

    # Reference values
    A_frobenius_norm = 157.867392938645
    b_l2_norm = 1.48087142738768

    # Assemble A and b
    assembler = dolfin.fem.assembling.Assembler(a, L, [])
    A = dolfin.cpp.la.PETScMatrix(mesh.mpi_comm())
    b = dolfin.cpp.la.PETScVector(mesh.mpi_comm())
    assembler.assemble(A=A, b=b, mat_type=dolfin.cpp.fem.Assembler.BlockType.monolithic)

    assert round(A.norm("frobenius") - A_frobenius_norm, 10) == 0
    assert round(b.norm("l2") - b_l2_norm, 10) == 0


# def test_ghost_mode_handling(pushpop_parameters):
#     def _form():
#         # Return form with trivial interior facet integral
#         mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 10, 10)
#         ff = dolfin.MeshFunction('size_t', mesh, mesh.topology.dim-1, 0)
#         dolfin.AutoSubDomain(lambda x: numpy.isclose(x[0], 0.5)).mark(ff, 1)
#         return dolfin.Constant(1.0)*dS(domain=mesh, subdomain_data=ff, subdomain_id=1)
#
#     # Not-ghosted mesh won't work in parallel and assembler should raise
#     if dolfin.MPI.size(dolfin.MPI.comm_world) == 1:
#         form = _form()
#         m = dolfin.fem.assembling.assemble_scalar(form)
#         assert numpy.isclose(m, 1.0)
#     else:
#         form = _form()
#         with pytest.raises(RuntimeError) as excinfo:
#             assemble(form)
#         assert "Incorrect mesh ghost mode" in repr(excinfo.value)
#
#     # Ghosted meshes work everytime
#     parameters["ghost_mode"] = "shared_vertex"
#     assert numpy.isclose(assemble(_form()), 1.0)
#     parameters["ghost_mode"] = "shared_facet"
#     assert numpy.isclose(assemble(_form()), 1.0)


@pytest.mark.parametrize('mesh_factory, facet_area', [((dolfin.generation.UnitSquareMesh, (dolfin.MPI.comm_world, 4, 4)), 4.0),
                                                      ((dolfin.generation.UnitCubeMesh, (dolfin.MPI.comm_world, 2, 2, 2)), 6.0),
                                                      ((dolfin.generation.UnitSquareMesh, (dolfin.MPI.comm_world, 4, 4, dolfin.CellType.Type.quadrilateral)), 4.0),
                                                      ((dolfin.generation.UnitCubeMesh, (dolfin.MPI.comm_world, 2, 2, 2, dolfin.CellType.Type.hexahedron)), 6.0)])
def test_functional_assembly(mesh_factory, facet_area):
    func, args = mesh_factory
    mesh = func(*args)

    f = dolfin.Constant(1.0)
    M0 = f*dx(mesh)
    assert round(dolfin.fem.assembling.assemble_scalar(M0) - 1.0, 7) == 0

    M1 = f*ds(mesh)
    assert round(dolfin.fem.assembling.assemble_scalar(M1) - facet_area, 7) == 0


@skip_in_parallel  # Ghost mode not working yet
@pytest.mark.parametrize('mesh_factory, args', [(dolfin.generation.UnitSquareMesh, (dolfin.MPI.comm_world, 4, 4)),
                                          (dolfin.generation.UnitCubeMesh, (dolfin.MPI.comm_world, 2, 2, 2)),
                                          (dolfin.generation.UnitSquareMesh, (dolfin.MPI.comm_world, 4, 4, dolfin.CellType.Type.quadrilateral)),
                                          pytest.param(dolfin.generation.UnitCubeMesh, (dolfin.MPI.comm_world, 2, 2, 2, dolfin.CellType.Type.hexahedron), marks=pytest.mark.xfail)])
def test_functional_assembly_interior(mesh_factory, args):
    mesh = mesh_factory(*args)

    f = dolfin.Constant(1.0)

    h_sum = 0.0
    for facet in dolfin.Facets(mesh):
        if facet.exterior():
            continue
        cell = dolfin.Cell(mesh, facet.entities(mesh.topology.dim)[0])
        h_sum += cell.facet_area(cell.index(facet))
    h_sum = dolfin.MPI.sum(mesh.mpi_comm(), h_sum)

    M2 = f*dS(mesh)
    eval = dolfin.fem.assembling.assemble_scalar(M2)
    assert round(eval - h_sum, 7) == 0


@pytest.mark.parametrize('mesh_factory', [(dolfin.generation.UnitCubeMesh, (dolfin.MPI.comm_world, 4, 4, 4)),
                                          (dolfin.generation.UnitCubeMesh, (dolfin.MPI.comm_world, 4, 4, 4, dolfin.CellType.Type.hexahedron))])
def test_subdomain_and_fulldomain_assembly_meshdomains(mesh_factory):
    """Test assembly over subdomains AND the full domain with markers
    stored as part of the mesh.
    """

    # Create a mesh of the unit cube
    func, args = mesh_factory
    mesh = func(*args)

    # Define subdomains for 3 faces of the unit cube
    class F0(dolfin.SubDomain):
        def inside(self, x, inside):
            return numpy.isclose(x[:,0], 0.0)

    class F1(dolfin.SubDomain):
        def inside(self, x, inside):
            return numpy.isclose(x[:,1], 0.0)

    class F2(dolfin.SubDomain):
        def inside(self, x, inside):
            return numpy.isclose(x[:,2], 0.0)

    # Define subdomains for 3 parts of the unit cube
    class S0(dolfin.SubDomain):
        def inside(self, x, inside):
            return x[:,0] > 0.25 - dolfin.DOLFIN_EPS

    class S1(dolfin.SubDomain):
        def inside(self, x, inside):
            return x[:,0] > 0.5 - dolfin.DOLFIN_EPS

    class S2(dolfin.SubDomain):
        def inside(self, x, inside):
            return x[:,0] > 0.75 - dolfin.DOLFIN_EPS

    # Mark mesh facets
    ff = dolfin.MeshFunction("size_t", mesh, mesh.topology.dim - 1, 100)
    f0 = F0()
    f1 = F1()
    f2 = F2()
    f0.mark(ff, 0)
    f1.mark(ff, 1)
    f2.mark(ff, 3)  # NB! 3, to leave a gap

    cf = dolfin.MeshFunction("size_t", mesh, mesh.topology.dim, 100)
    s0 = S0()
    s1 = S1()
    s2 = S2()
    s0.mark(cf, 0)
    s1.mark(cf, 1)
    s2.mark(cf, 3)  # NB! 3, to leave a gap

    # Assemble forms on subdomains and full domain and compare
    krange = list(range(5))
    dx, ds = dolfin.Measure("dx", subdomain_data=cf), dolfin.Measure("ds", subdomain_data=ff)
    for dmu in (dx, ds):
        full = dolfin.fem.assembling.assemble_scalar(dolfin.Constant(3.0)*dmu(mesh))
        sub_plus_form = [dolfin.Constant(3.0)*dmu(mesh) + dolfin.Constant(1.0)*dmu(k, domain=mesh) for k in krange]
        subplusfull = list(map(dolfin.fem.assembling.assemble_scalar, sub_plus_form))

        sub_form = [dolfin.Constant(1.0)*dmu(k, domain=mesh) for k in krange]
        sub = list(map(dolfin.fem.assembling.assemble_scalar, sub_form))
        for k in krange:
            assert round(sub[k] + full - subplusfull[k], 7) == 0


@skip_in_parallel
def test_subdomain_assembly_form_1():
    "Test assembly over subdomains with markers stored as part of form"

    mesh = dolfin.UnitSquareMesh(dolfin.MPI.comm_world, 4, 4)

    # Define cell/facet function
    class Left(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return x[:,0] < 0.49
    subdomains = dolfin.MeshFunction("size_t", mesh, mesh.topology.dim, 0)
    left = Left()
    left.mark(subdomains, 1)

    class RightBoundary(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return x[:,0] > 0.95
    boundaries = dolfin.MeshFunction("size_t", mesh, mesh.topology.dim-1, 0)
    right = RightBoundary()
    right.mark(boundaries, 1)

    V = dolfin.FunctionSpace(mesh, "CG", 2)
    f = dolfin.Expression("x[0] + 2", degree=1)
    g = dolfin.Expression("x[1] + 1", degree=1)

    f = dolfin.interpolate(f, V)
    g = dolfin.interpolate(g, V)

    mesh1 = subdomains.mesh()
    mesh2 = boundaries.mesh()
    assert mesh1.id() == mesh2.id()
    assert mesh1.ufl_domain().ufl_id() == mesh2.ufl_domain().ufl_id()

    dxs = dx(subdomain_data=subdomains)
    dss = ds(subdomain_data=boundaries)
    assert dxs.ufl_domain() == None
    assert dss.ufl_domain() == None
    assert dxs.subdomain_data() == subdomains
    assert dss.subdomain_data() == boundaries

    M = f*f*dxs(0) + g*f*dxs(1) + f*f*dss(1)
    assert M.ufl_domains() == (mesh.ufl_domain(),)
    sd = M.subdomain_data()[mesh.ufl_domain()]
    assert sd["cell"] == subdomains
    assert sd["exterior_facet"] == boundaries

    assemble = dolfin.fem.assembling.assemble_scalar
    # Check that subdomains are respected
    reference = 15.0
    assert round(assemble(M) - reference, 10) == 0

    # Check that the form itself assembles as before
    assert round(assemble(M) - reference, 10) == 0

    # Take action of derivative of M on f
    df = dolfin.TestFunction(V)
    L = dolfin.derivative(M, f, df)
    dg = dolfin.TrialFunction(V)
    F = dolfin.derivative(L, g, dg)
    b = dolfin.action(F, f)

    dummy_a = df*dg*dx
    b_vector = dolfin.la.PETScVector(mesh.mpi_comm())
    assembler = dolfin.fem.assembling.Assembler(dummy_a, b, [])
    assembler.assemble(b=b_vector, mat_type=dolfin.cpp.fem.Assembler.BlockType.monolithic)

    # Check that domain data carries across transformations:
    reference = 0.136477465659
    assert round(b_vector.norm("l2") - reference, 8) == 0


def test_subdomain_assembly_form_2():
    "Test assembly over subdomains with markers stored as part of form"

    # Define mesh
    mesh = dolfin.UnitSquareMesh(dolfin.MPI.comm_world, 8, 8)

    # Define domain for lower left corner
    class MyDomain(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return numpy.logical_and(x[:,0] < (0.5 + dolfin.DOLFIN_EPS), x[:,1] < 0.5 + dolfin.DOLFIN_EPS)
    my_domain = MyDomain()

    # Define boundary for lower left corner
    class MyBoundary(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return numpy.logical_or(
                numpy.logical_and(x[:,0] < (0.5 + dolfin.DOLFIN_EPS), x[:,1] < dolfin.DOLFIN_EPS),
                numpy.logical_and(x[:,1] < (0.5 + dolfin.DOLFIN_EPS), x[:,0] < dolfin.DOLFIN_EPS)
            )
    my_boundary = MyBoundary()

    # Mark mesh functions
    D = mesh.topology.dim
    cell_domains = dolfin.MeshFunction("size_t", mesh, D, 1)
    exterior_facet_domains = dolfin.MeshFunction("size_t", mesh, D - 1, 1)

    my_domain.mark(cell_domains, 0)
    my_boundary.mark(exterior_facet_domains, 0)

    # Define forms
    c = dolfin.Constant(1.0)

    a0 = c*dx(0, domain=mesh, subdomain_data=cell_domains)
    a1 = c*ds(0, domain=mesh, subdomain_data=exterior_facet_domains)

    assert round(dolfin.fem.assembling.assemble_scalar(a0) - 0.25, 7) == 0
    assert round(dolfin.fem.assembling.assemble_scalar(a1) - 1.0, 7) == 0


def test_nonsquare_assembly():
    """Test assembly of a rectangular matrix"""

    mesh = dolfin.UnitSquareMesh(dolfin.MPI.comm_world, 16, 16)

    V = dolfin.VectorElement("Lagrange", mesh.ufl_cell(), 2)
    Q = dolfin.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    W = V*Q
    V = dolfin.FunctionSpace(mesh, V)
    Q = dolfin.FunctionSpace(mesh, Q)
    W = dolfin.FunctionSpace(mesh, W)

    (v, q) = dolfin.TestFunctions(W)
    (u, p) = dolfin.TrialFunctions(W)
    a = ufl.div(v)*p*dx
    L = v[0]*dx
    A_frobenius_norm = 9.6420303878382718e-01

    A = dolfin.cpp.la.PETScMatrix(mesh.mpi_comm())
    assembler = dolfin.fem.assembling.Assembler([[a]], [L], [])
    assembler.assemble(A=A, mat_type=dolfin.cpp.fem.Assembler.BlockType.monolithic)
    assert round(A.norm("frobenius") - A_frobenius_norm, 10) == 0

    v = dolfin.TestFunction(V)
    p = dolfin.TrialFunction(Q)
    a = ufl.inner(ufl.grad(p), v)*dx

    A = dolfin.cpp.la.PETScMatrix(mesh.mpi_comm())
    assembler = dolfin.fem.assembling.Assembler([[a]], [L], [])
    assembler.assemble(A=A, mat_type=dolfin.cpp.fem.Assembler.BlockType.monolithic)
    A_frobenius_norm = 0.935414346693
    assert round(A.norm("frobenius") - A_frobenius_norm, 10) == 0


# @skip_in_parallel
# def test_reference_assembly(filedir, pushpop_parameters):
#     "Test assembly against a reference solution"
#
#     # NOTE: This test is not robust as it relies on specific
#     #       DOF order, which cannot be guaranteed
#     parameters["reorder_dofs_serial"] = False
#
#     # Load reference mesh (just a simple tetrahedron)
#     mesh = Mesh(os.path.join(filedir, "tetrahedron.xml"))
#
#     # Assemble stiffness and mass matrices
#     V = FunctionSpace(mesh, "Lagrange", 1)
#     u, v = TrialFunction(V), TestFunction(V)
#     A, M = EigenMatrix(), EigenMatrix()
#     assemble(dot(grad(v), grad(u))*dx, tensor=A)
#     assemble(v*u*dx, tensor=M)
#
#     # Run test (requires SciPy)
#     try:
#         import scipy
#         A = A.sparray().todense()
#         M = M.sparray().todense()
#
#         # Create reference matrices and set entries
#         A0 = numpy.array([[1.0/2.0, -1.0/6.0, -1.0/6.0, -1.0/6.0],
#                           [-1.0/6.0, 1.0/6.0, 0.0, 0.0],
#                           [-1.0/6.0, 0.0, 1.0/6.0, 0.0],
#                           [-1.0/6.0, 0.0, 0.0, 1.0/6.0]])
#         M0 = numpy.array([[1.0/60.0, 1.0/120.0, 1.0/120.0, 1.0/120.0],
#                           [1.0/120.0, 1.0/60.0, 1.0/120.0, 1.0/120.0],
#                           [1.0/120.0, 1.0/120.0, 1.0/60.0, 1.0/120.0],
#                           [1.0/120.0, 1.0/120.0, 1.0/120.0, 1.0/60.0]])
#
#         C = A - A0
#         assert round(numpy.linalg.norm(C, 'fro') - 0.0, 7) == 0
#         D = M - M0
#         assert round(numpy.linalg.norm(D, 'fro') - 0.0, 7) == 0
#
#     except:
#         print("Cannot run this test without SciPy")
#
#
# def test_ways_to_pass_mesh_to_assembler():
#     mesh = UnitSquareMesh(MPI.comm_world, 16, 16)
#
#     # Geometry with mesh (ufl.Domain with mesh in domain data)
#     x = SpatialCoordinate(mesh)
#     n = FacetNormal(mesh)
#
#     # Geometry with just cell (no reference to mesh, for backwards
#     # compatibility)
#     x2 = SpatialCoordinate(mesh)
#     n2 = FacetNormal(mesh)
#
#     # A function equal to x[0] for comparison
#     V = FunctionSpace(mesh, "CG", 1)
#     f = Function(V)
#     f.interpolate(Expression("x[0]", degree=1))
#
#     # An expression equal to x[0], with different geometry info:
#     e = Expression("x[0]", degree=1)  # nothing
#     e2 = Expression("x[0]", cell=mesh.ufl_cell(), degree=1)  # cell
#     e3 = Expression("x[0]", element=V.ufl_element())  # ufl element
#     e4 = Expression("x[0]", domain=mesh, degree=1)  # mesh
#
#     # Provide mesh in measure:
#     dx2 = Measure("dx", domain=mesh)
#     assert round(1.0 - assemble(1*dx(mesh)), 7) == 0
#     assert round(1.0 - assemble(Constant(1.0)*dx(mesh)), 7) == 0
#     assert round(1.0 - assemble(Constant(1.0)*dx2), 7) == 0
#
#     # Try with cell argument to Constant as well:
#     assert round(1.0 - assemble(Constant(1.0,
#                                          cell=mesh.ufl_cell())*dx(mesh))) == 0
#     assert round(1.0 - assemble(Constant(1.0, cell=mesh.ufl_cell())*dx2)) == 0
#     assert round(1.0 - assemble(Constant(1.0,
#                                          cell=mesh.ufl_cell())*dx(mesh))) == 0
#     assert round(1.0 - assemble(Constant(1.0, cell=mesh.ufl_cell())*dx2)) == 0
#
#     # Geometric quantities with mesh in domain:
#     assert round(0.5 - assemble(x[0]*dx), 7) == 0
#     assert round(0.5 - assemble(x[0]*dx(mesh)), 7) == 0
#
#     # Geometric quantities without mesh in domain:
#     assert round(0.5 - assemble(x2[0]*dx(mesh)), 7) == 0
#
#     # Functions with mesh in domain:
#     assert round(0.5 - assemble(f*dx), 7) == 0
#     assert round(0.5 - assemble(f*dx(mesh)), 7) == 0
#
#     # Expressions with and without mesh in domain:
#     assert round(0.5 - assemble(e*dx(mesh)), 7) == 0
#     assert round(0.5 - assemble(e2*dx(mesh)), 7) == 0
#     assert round(0.5 - assemble(e3*dx(mesh)), 7) == 0
#     assert round(0.5 - assemble(e4*dx), 7) == 0  # e4 has a domain with mesh reference
#     assert round(0.5 - assemble(e4*dx(mesh)), 7) == 0
#
#     # Geometric quantities with mesh in domain:
#     assert round(0.0 - assemble(n[0]*ds), 7) == 0
#     assert round(0.0 - assemble(n[0]*ds(mesh)), 7) == 0
#
#     # Geometric quantities without mesh in domain:
#     assert round(0.0 - assemble(n2[0]*ds(mesh)), 7) == 0
