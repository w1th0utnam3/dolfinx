"""Unit tests for Expression using Numba"""

# Copyright (C) 2018 Garth N. Wells
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Benjamin Kehlet 2012

import pytest


def test_expression_attach():
    from numba import cfunc, types, carray

    c_sig = types.void(types.CPointer(types.double), types.intc,
                       types.CPointer(types.double), types.intc)

    @cfunc(c_sig, nopython=True)
    def my_callback(value, m, x, n):
        x_array = carray(x, n)
        val_array = carray(value, (m))
        val_array[0] = x_array[0] + x_array[1]

    # print("Test: ", my_callback.address)
    from dolfin import cpp
    e = cpp.function.Expression([1], my_callback.address)

    import numpy as np
    vals = np.zeros(1)
    x = np.ndarray([3, 10])
    e.eval(vals, [3, 10])
    print("Test2: ", vals)
    assert vals == 13.0

    # print(my_callback.inspect_llvm())
