#include "polyclip.hpp"

namespace mfem
{

// Comparator to handle floating point uniqueness in std::set
struct VectorComparator
{
   double tol;
   VectorComparator(double t) : tol(t) {}

   bool operator()(const std::vector<real_t>& a,
                   const std::vector<real_t>& b) const
   {
      for (size_t k = 0; k < a.size(); ++k)
      {
         if (std::abs(a[k] - b[k]) > tol) { return a[k] < b[k]; }
      }
      return false;
   }
};

void SimplexAndBound(Vector &b_min, Vector &b_max, DenseMatrix &V,
                     const real_t tol)
{
   const int dim = b_min.Size();
   MFEM_VERIFY(dim == b_max.Size(), "Dimension mismatch in PolyClip");
   // Safety: Bitwise shift works for dim <= 30 (int is usually 32-bit)
   // In FEM, dim is usually 2 or 3, so this is safe.
   MFEM_VERIFY(dim <= 30, "Dimension too high for PolyClip");

   // 1. Clamp bounds to [0, 1] standard simplex range
   for (int i = 0; i < dim; i++)
   {
      b_min(i) = std::max(0.0, b_min(i));
      b_max(i) = std::min(1.0, b_max(i));
      MFEM_VERIFY(b_min(i) <= b_max(i),
                  "Infeasible bounds in PolyClip");
   }

   // 2. Precompute deltas (cost of flipping from Min to Max)
   Vector deltas(dim);
   subtract(b_max, b_min, deltas);
   double sum_min_total = b_min.Sum();

   // 3. Use std::set to automatically discard duplicate vertices
   std::set<std::vector<real_t>, VectorComparator> unique_verts((VectorComparator(
            tol)));

   // Pre-allocate temporary variables
   Array<int> map_idx(dim - 1);
   std::vector<real_t> pt(dim);

   // --- MAIN LOOP ---
   // Iterate over which dimension 'i' acts as the "Free Variable"
   for (int i = 0; i < dim; i++)
   {
      // Map indices of the "other" variables (j != i)
      int k = 0;
      for (int j = 0; j < dim; j++)
      {
         if (j == i) { continue; }
         map_idx[k++] = j;
      }

      // Calculate the valid range for the sum of the "other" variables.
      // Since x_i = 1 - sum(others), and b_min(i) <= x_i <= b_max(i),
      // the constraints on sum(others) are derived from the bounds of x_i.
      double sum_others_base = sum_min_total - b_min(i);
      double target_min = 1.0 - b_max(i) - sum_others_base;
      double target_max = 1.0 - b_min(i) - sum_others_base;

      // Bitwise Enumeration: Iterate all 2^(dim-1) subsets of the other variables
      int num_combinations = 1 << (dim - 1);

      for (int mask = 0; mask < num_combinations; mask++)
      {
         double current_delta_sum = 0.0;

         // Check which variables in the subset are flipped to Max
         for (int bit = 0; bit < (dim - 1); bit++)
         {
            if ((mask >> bit) & 1)
            {
               current_delta_sum += deltas(map_idx[bit]);
            }
         }

         // Check if this combination creates a valid vertex
         if (current_delta_sum >= target_min - tol &&
             current_delta_sum <= target_max + tol)
         {
            // Reconstruct Vertex
            double actual_sum_others = 0.0;

            for (int bit = 0; bit < (dim - 1); bit++)
            {
               int idx = map_idx[bit];
               if ((mask >> bit) & 1)
               {
                  pt[idx] = b_max(idx);
               }
               else
               {
                  pt[idx] = b_min(idx);
               }
               actual_sum_others += pt[idx];
            }

            // Set the Free Variable 'i'
            pt[i] = 1.0 - actual_sum_others;

            // Insert into set (deduplicates automatically)
            unique_verts.insert(pt);
         }
      }
   }

   // 4. Copy unique vertices to Output Matrix V
   if (unique_verts.empty())
   {
      V.SetSize(0, 0);
      return;
   }

   V.SetSize(unique_verts.size(), dim);
   int row = 0;
   for (const auto &v_data : unique_verts)
   {
      for (int col = 0; col < dim; col++)
      {
         // Round to cleanup float noise
         V(row, col) = std::round(v_data[col] / tol) * tol;
      }
      row++;
   }

}
} // namespace mfem

