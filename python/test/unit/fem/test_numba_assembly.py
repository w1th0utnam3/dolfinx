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
    A = carray(A_, (16), dtype=np.float64)
    coordinate_dofs = carray(coords_, (12), dtype=np.float64)

    J_c4 = -coordinate_dofs[1] + coordinate_dofs[7]
    J_c8 = -coordinate_dofs[2] + coordinate_dofs[11]
    J_c5 = -coordinate_dofs[1] + coordinate_dofs[10]
    J_c7 = -coordinate_dofs[2] + coordinate_dofs[8]
    J_c0 = -coordinate_dofs[0] + coordinate_dofs[3]
    J_c1 = -coordinate_dofs[0] + coordinate_dofs[6]
    J_c6 = -coordinate_dofs[2] + coordinate_dofs[5]
    J_c3 = -coordinate_dofs[1] + coordinate_dofs[4]
    J_c2 = -coordinate_dofs[0] + coordinate_dofs[9]

    sp = np.empty(80, dtype=np.float64)
    sp[0] = J_c4 * J_c8
    sp[1] = J_c5 * J_c7
    sp[2] = sp[0] - sp[1]
    sp[3] = J_c0 * sp[2]
    sp[4] = J_c5 * J_c6
    sp[5] = J_c3 * J_c8
    sp[6] = sp[4] - sp[5]
    sp[7] = J_c1 * sp[6]
    sp[8] = sp[3] + sp[7]
    sp[9] = J_c3 * J_c7
    sp[10] = J_c4 * J_c6
    sp[11] = sp[9] - sp[10]
    sp[12] = J_c2 * sp[11]
    sp[13] = sp[8] + sp[12]
    sp[14] = sp[2] / sp[13]
    sp[15] = -J_c3 * J_c8
    sp[16] = sp[4] + sp[15]
    sp[17] = sp[16] / sp[13]
    sp[18] = sp[11] / sp[13]
    sp[19] = sp[14] * sp[14]
    sp[20] = sp[14] * sp[17]
    sp[21] = sp[18] * sp[14]
    sp[22] = sp[17] * sp[17]
    sp[23] = sp[18] * sp[17]
    sp[24] = sp[18] * sp[18]
    sp[25] = J_c2 * J_c7
    sp[26] = -J_c8 * J_c1
    sp[27] = sp[25] + sp[26]
    sp[28] = sp[27] / sp[13]
    sp[29] = J_c0 * J_c8
    sp[30] = -J_c6 * J_c2
    sp[31] = sp[29] + sp[30]
    sp[32] = sp[31] / sp[13]
    sp[33] = J_c1 * J_c6
    sp[34] = J_c0 * J_c7
    sp[35] = sp[33] - sp[34]
    sp[36] = sp[35] / sp[13]
    sp[37] = sp[28] * sp[28]
    sp[38] = sp[28] * sp[32]
    sp[39] = sp[28] * sp[36]
    sp[40] = sp[32] * sp[32]
    sp[41] = sp[32] * sp[36]
    sp[42] = sp[36] * sp[36]
    sp[43] = sp[37] + sp[19]
    sp[44] = sp[38] + sp[20]
    sp[45] = sp[39] + sp[21]
    sp[46] = sp[40] + sp[22]
    sp[47] = sp[41] + sp[23]
    sp[48] = sp[24] + sp[42]
    sp[49] = J_c1 * J_c5
    sp[50] = J_c2 * J_c4
    sp[51] = sp[49] - sp[50]
    sp[52] = sp[51] / sp[13]
    sp[53] = J_c2 * J_c3
    sp[54] = J_c0 * J_c5
    sp[55] = sp[53] - sp[54]
    sp[56] = sp[55] / sp[13]
    sp[57] = J_c0 * J_c4
    sp[58] = J_c1 * J_c3
    sp[59] = sp[57] - sp[58]
    sp[60] = sp[59] / sp[13]
    sp[61] = sp[52] * sp[52]
    sp[62] = sp[52] * sp[56]
    sp[63] = sp[60] * sp[52]
    sp[64] = sp[56] * sp[56]
    sp[65] = sp[60] * sp[56]
    sp[66] = sp[60] * sp[60]
    sp[67] = sp[43] + sp[61]
    sp[68] = sp[44] + sp[62]
    sp[69] = sp[45] + sp[63]
    sp[70] = sp[46] + sp[64]
    sp[71] = sp[47] + sp[65]
    sp[72] = sp[48] + sp[66]
    sp[73] = np.abs(sp[13])
    sp[74] = sp[67] * sp[73]
    sp[75] = sp[68] * sp[73]
    sp[76] = sp[69] * sp[73]
    sp[77] = sp[70] * sp[73]
    sp[78] = sp[71] * sp[73]
    sp[79] = sp[72] * sp[73]

    A[0] = sp[74] +  sp[77] + sp[79] + 2*(sp[78] + sp[75] + sp[76])
    A[1] = -sp[74] - sp[75] - sp[76]
    A[2] = -sp[75] - sp[77] - sp[78]
    A[3] = -sp[76] - sp[78] - sp[79]
    A[4] = -sp[74] - sp[75] - sp[76]
    A[5] =  sp[74]
    A[6] =  sp[75]
    A[7] =  sp[76]
    A[8] = -sp[75] -sp[77] -sp[78]
    A[9] =  sp[75]
    A[10] = sp[77]
    A[11] = sp[78]
    A[12] = -sp[76] -sp[78] -sp[79]
    A[13] = sp[76]
    A[14] = sp[78]
    A[15] = sp[79]
    A *= 0.1666666666666667

def tabulate_tensor_b(b_, w_, coords_, cell_orientation):
    b = carray(b_, (4), dtype=np.float64)
    b *= 0.0

def test_numba_assembly():
    mesh = UnitCubeMesh(MPI.comm_world, 3,3,3)
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
