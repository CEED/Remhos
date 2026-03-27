#ifndef REMHOS_LVPP_HPP
#define REMHOS_LVPP_HPP
#include "mfem.hpp"
#include "remap.hpp"
#include "legendre.hpp"

namespace mfem
{

inline real_t allreduce(MPI_Comm comm, real_t val, MPI_Op op)
{
   real_t recv;
   MPI_Allreduce(&val, &recv, 1, MPITypeMap<real_t>::mpi_type, op, comm);
   return recv;
}
class Dykstra
{
   MPI_Comm comm = MPI_COMM_NULL;
   StackedFunctional &constraints;
   StackedSharedFunctional *shared_constraints = nullptr;
   MassOperator &mass;
   const Vector &xmin;
   const Vector &xmax;
   Array<LegendreFunction*> legendre_funcs;
   Array<int> offsets;
   real_t tol;
   int max_iter;
   int max_linesearch = 30;
   real_t c1 = 1e-03; // Armijo condition constant

   bool enforce_sum_to_one = false;
   Array<int> sum_to_one_idx_start;
   int sum_to_one_block_size;
public:
   Dykstra(MPI_Comm comm, StackedFunctional &constraints, MassOperator &mass,
           Array<LegendreFunction*> &legendre_funcs_, Array<int> &offsets_,
           const Vector &xmin, const Vector &xmax, real_t tol=1e-10, int max_iter=1000)
      : comm(comm), constraints(constraints), mass(mass)
      , xmin(xmin), xmax(xmax)
      , legendre_funcs(legendre_funcs_), offsets(offsets_)
      , tol(tol), max_iter(max_iter)
   {
      shared_constraints = dynamic_cast<StackedSharedFunctional*>(&constraints);
   }
   void SetAbsTol(real_t tol) { this->tol = tol; }
   void SetMaxIter(int max_iter) { this->max_iter = max_iter; }
   void EnforceSumToOne(const Array<int> &idx_start, const int block_size)
   {
      enforce_sum_to_one = true;
      sum_to_one_idx_start = idx_start;
      sum_to_one_block_size = block_size;
   }

   // Dykstra projection with Bregman divergence
   // At each iteration, we project onto the tangent plane of each constraint
   // psi_k = inv_sigmoid(Project_{k mod N}(sigmoid(psi_{k-1} + q_{k mod N})))
   // q_{k mod N} = psi_{k-1} + q_{k - N mod N} - psi_k
   // where Project_k is the projection onto the k-th constraint (tangent plane)
   void Project(Vector &projected_x);
private:
   void ProjectSumToOne(Vector &psi, Vector &qi);

   void Project(const Functional &con, Vector &psi, const Vector &grad,
                const real_t targ, Vector &psi_aux, Vector &projected_x);

   void MapLatent(const Vector &psi_,
                  const Vector &xmin_,
                  const Vector &xmax_,
                  Vector &x_);

   void MapPrimal(const Vector &x_,
                  const Vector &xmin_,
                  const Vector &xmax_,
                  Vector &psi_);

};
}

#endif // REMHOS_LVPP_HPP
