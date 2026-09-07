// Copyright (c) 2026, Lawrence Livermore National Security, LLC.
// See files LICENSE and NOTICE for details.

#ifndef REMHOS_NODAL_HPP
#define REMHOS_NODAL_HPP

#include "mfem.hpp"
#include <vector>

namespace mfem
{

struct NodalRemapOptions
{
   int order = 3;
   int ncp = 12;
   int cp_type = 1;
   int quadrature_order = -1;
   int max_sweeps = 20000;
   real_t abs_tol = 1e-12;
   real_t rel_tol = 1e-10;
   bool plbound = true;
   bool upper_constraint = false;
   real_t upper = 1.0;
   bool compare = false;
   int repetitions = 3;
   int mass_solver = 0;
   int mass_max_iterations = 100;
   real_t mass_tolerance = 1e-10;
};

struct JointFitInfo
{
   int iterations = 0;
   bool converged = false;
   real_t mass_error = 0, violation = 0, stationarity = 0, gap = 0;
};

// Solve the block-diagonal mass-norm QP with identical local inequalities
// constraints*x >= bounds and global equality sum (M_K*1)^T*x_K = target_mass.
// A failed solve returns its last iterate and converged=false, never success.
JointFitInfo FitConservative(MPI_Comm comm, const std::vector<DenseMatrix> &mass,
                            const std::vector<Vector> &target,
                            const DenseMatrix &constraints, const Vector &bounds,
                            real_t target_mass, const NodalRemapOptions &options,
                            std::vector<Vector> &result);

// Quadrature-based transfer and elementwise mass-norm fit, entirely in a
// Gauss-Legendre nodal L2 basis. Optional mass correction uses bounded
// elementwise blends, with HiOp (1) or LVPP (2) selecting the blend factors.
// Solver 4 instead minimizes the original transfer error over all DOFs with
// bounds and global mass conservation simultaneously (internal primal-dual QP).
// Returns 2 if any local QP fails to converge; its last iterate is reported.
// Returns 3 if mass correction cannot be accepted; the bounded field is kept
// for solvers 1/2, while solver 4 reports its unconverged last iterate.
int RemapNodalL2(ParMesh &source_mesh, const Vector &destination_nodes,
                 Coefficient &initial_condition,
                 const NodalRemapOptions &options, bool visualization);

} // namespace mfem

#endif
