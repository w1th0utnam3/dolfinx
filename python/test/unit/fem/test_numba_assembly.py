"""Unit tests for assembly with a numba kernel"""

# Copyright (C) 2018 Chris N. Richardson
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

import pytest
import dolfin
from dolfin import *
from dolfin.la import PETScMatrix, PETScVector
from dolfin.cpp.fem import SystemAssembler
from numba import cfunc, types, carray, jit
import numpy as np
import ctypes

def tabulate_tensor_A(A_, w_, coords_, cell_orientation):
    A = carray(A_, (3, 3), dtype=np.float64)
    coordinate_dofs = carray(coords_, (6), dtype=np.float64)

    # Ke=∫Ωe BTe Be dΩ
    x0, y0 = coordinate_dofs[0:2]
    x1, y1 = coordinate_dofs[2:4]
    x2, y2 = coordinate_dofs[4:6]

    # Element area Ae
    Ae = abs((x0 - x1)*(y2 - y1) - (y0 - y1)*(x2 - x1))

    B = np.array([y1 - y2, y2 - y0, y0 - y1,
                  x2 - x1, x0 - x2, x1 - x0],
                 dtype=np.float64).reshape(2, 3)

    A[:,:] = np.dot(B.T, B)/(2*Ae)


def tabulate_tensor_b(b_, w_, coords_, cell_orientation):
    b = carray(b_, (3), dtype=np.float64)
    b *= 0.0

def test_numba_assembly():
    mesh = UnitSquareMesh(MPI.comm_world, 39, 39)
    Q = FunctionSpace(mesh, "CG", 1)

    u = TrialFunction(Q)
    v = TestFunction(Q)
    a = u*v*dx
    f = Constant(0.0)
    L = f*v*dx

    a = dolfin.cpp.fem.Form([Q._cpp_object, Q._cpp_object])
    L = dolfin.cpp.fem.Form([Q._cpp_object])

    sig = types.void(types.CPointer(types.double), types.CPointer(types.CPointer(types.double)), types.CPointer(types.double), types.intc)

    fnA = cfunc(sig, cache=True)(tabulate_tensor_A)
    a.set_cell_tabulate(0, fnA.address)

    fnb = cfunc(sig, cache=True)(tabulate_tensor_b)
    L.set_cell_tabulate(0, fnb.address)

    assembler = SystemAssembler(a, L, [])
    A = PETScMatrix(MPI.comm_world)
    b = PETScVector(MPI.comm_world)
    assembler.assemble(A, b)

    print(A.norm('frobenius'))
