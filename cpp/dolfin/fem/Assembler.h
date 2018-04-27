// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <boost/multi_array.hpp>
#include <memory>
#include <utility>
#include <vector>
#include <dolfin/la/Scalar.h>
#include <dolfin/common/types.h>

namespace dolfin
{
namespace la
{
class PETScMatrix;
class PETScVector;
}

namespace mesh
{
class Mesh;
class MeshEntity;
}

namespace fem
{
// Forward declarations
class DirichletBC;
class Form;

/// Assembly of LHS and RHS Forms with DirichletBC boundary conditions applied
class Assembler
{
public:
  /// Assembly type for block forms
  enum class BlockType
  {
    monolithic,
    nested
  };

  /// Constructor
  Assembler(std::vector<std::vector<std::shared_ptr<const Form>>> a,
            std::vector<std::shared_ptr<const Form>> L,
            std::vector<std::shared_ptr<const DirichletBC>> bcs);

  /// Assemble matrix. Dirichlet rows/columns are zeroed, with '1' placed on
  /// diagonal
  void assemble(la::PETScMatrix& A, BlockType type = BlockType::nested);

  /// Assemble vector
  void assemble(la::PETScVector& b, BlockType type = BlockType::nested);

  /// Assemble matrix and vector
  void assemble(la::PETScMatrix& A, la::PETScVector& b);

  /// Assemble scalar functional
  static void assemble(la::Scalar& m, const Form& M);

private:
  // Assemble matrix. Dirichlet rows/columns are zeroed, with '1' placed on
  // diagonal
  static void assemble(la::PETScMatrix& A, const Form& a,
                       std::vector<std::shared_ptr<const DirichletBC>> bcs);

  // Assemble vector
  static void assemble(la::PETScVector& b, const Form& L);

  // Apply bcs to vector (b <- b - Ax, where x holds prescribed boundary
  // values)
  static void apply_bc(la::PETScVector& b, const Form& a,
                       std::vector<std::shared_ptr<const DirichletBC>> bcs);

  // Set bcs (set entries of b to be equal to boundary value)
  static void set_bc(la::PETScVector& b, const Form& L,
                     std::vector<std::shared_ptr<const DirichletBC>> bcs);

  // Iterate over cells and assemble
  static void assemble_over_cells(const Form &form,
                                  const std::function<void(EigenRowMatrixXd& Ae)>& initialise_element_tensor,
                                  const std::function<void(EigenRowMatrixXd& Ae, mesh::MeshEntity& cell)>& insert_element_to_global_tensor);

  // Bilinear and linear forms
  std::vector<std::vector<std::shared_ptr<const Form>>> _a;
  std::vector<std::shared_ptr<const Form>> _l;

  // Dirichlet boundary conditions
  std::vector<std::shared_ptr<const DirichletBC>> _bcs;
};
}
}
