#ifndef MFEM_POLYCLIP_HPP
#define MFEM_POLYCLIP_HPP

#include "mfem.hpp"
namespace mfem
{
void SimplexAndBound(Vector &b_min, Vector &b_max, DenseMatrix &V,
                     const real_t tol = 1e-12);
namespace SimplexAndBoundHelpers
{
// Helper function of solve_subset_sum.
// Recursive backtracking function to find subsets summing to [t_min, t_max]
void backtrack(int target_idx, int i, const real_t current_sum,
               const real_t t_min, const real_t t_max,
               const Vector &values, Array<bool> &current_mask,
               const real_t tol);

// Helper function of SimplexAndBound.
// Find all subsets of values that sum to [t_min, t_max] within tolerance.
// Returns a logical vector
void solve_subset_sum(const Vector &values, int i, real_t t_min, real_t t_max,
                      Array<bool> &subset_mask, real_t tol);
} // namespace PolyClipHelpers
} // namespace mfem

#endif // MFEM_POLYCLIP_HPP
