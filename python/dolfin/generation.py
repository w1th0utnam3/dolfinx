# -*- coding: utf-8 -*-
# Copyright (C) 2017 Chris N. Richardson
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Simple mesh generation module"""

import dolfin.fem
from dolfin.cpp.generation import IntervalMesh, RectangleMesh, BoxMesh
from dolfin.cpp.mesh import CellType
import numpy as np

# FIXME: Remove, and use 'create' method?


def UnitIntervalMesh(comm, nx):
    """Create a mesh on the unit interval"""
    mesh = IntervalMesh.create(comm, nx, [0.0, 1.0])
    mesh.geometry.coord_mapping = dolfin.fem.create_coordinate_map(mesh)
    return mesh


def UnitSquareMesh(comm, nx, ny, cell_type=CellType.Type.triangle):
    """Create a mesh of a unit square"""
    mesh = RectangleMesh.create(
        comm,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])], [nx, ny], cell_type)
    mesh.geometry.coord_mapping = dolfin.fem.create_coordinate_map(mesh)
    return mesh


def UnitCubeMesh(comm, nx, ny, nz, cell_type=CellType.Type.tetrahedron):
    """Create a mesh of a unit cube"""
    mesh = BoxMesh.create(
        comm, [np.array([0.0, 0.0, 0.0]),
               np.array([1.0, 1.0, 1.0])], [nx, ny, nz], cell_type)
    mesh.geometry.coord_mapping = dolfin.fem.create_coordinate_map(mesh)
    return mesh
