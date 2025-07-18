
#ifndef REMHOS_LVPP_HPP
#define REMHOS_LVPP_HPP
#include "config/config.hpp"
#include "fem/fe_coll.hpp"
#include "fem/qspace.hpp"
#include "general/forall.hpp"
#include "mfem.hpp"
#include "mpi.h"
#include "./remap.hpp"

namespace mfem
{

inline real_t sigmoid(const real_t x)
{
   return x > 0 ? 1.0 / (1.0 + std::exp(-x)) : std::exp(x) / (1.0 + std::exp(x));
}

inline real_t sigmoid(const real_t x, const real_t l, const real_t u)
{
   real_t scale = (u - l);
   return l + scale*sigmoid(x);
}

inline real_t der_sigmoid(const real_t x)
{
   const real_t s = sigmoid(x);
   return s * (1.0 - s);
}

inline real_t der_sigmoid(const real_t x, const real_t l, const real_t u)
{
   real_t scale = (u - l);
   return scale*der_sigmoid(x);
}

class MappedGridFunctionCoefficient : public GridFunctionCoefficient
{
protected:
   std::function<real_t(const real_t x, ElementTransformation &T, const IntegrationPoint &ip)>
   F;
public:
   MappedGridFunctionCoefficient(GridFunction &gf,
                                 std::function<real_t(const real_t x, ElementTransformation &T, const IntegrationPoint &ip)>
                                 F)
      : GridFunctionCoefficient(&gf), F(F) {}
   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      return F(GridFunctionCoefficient::Eval(T, ip), T, ip);
   }
};

class RemapObjective : public Functional
{
public:
   // Initialize the LVPPRemap Functional.
   // space_idx[i] = -1 : quadrature function, >= 0 : finite element space index
   // x_initial : initial values before remapping
   RemapObjective(MPI_Comm comm, QuadratureSpace &qs,
                  std::vector<ParFiniteElementSpace*> &fes,
                  const Array<int> &space_idx,
                  const BlockVector &x_initial)
      : Functional(comm, x_initial.Size())
      , qs(qs), fes(fes), qf(qs), gfs(fes.size())
      , zero_cf(0.0)
      , numVars(space_idx.Size()), space_idx(space_idx)
      , x_initial(x_initial)
   {
      MFEM_VERIFY(space_idx.Max() < (int)fes.size() && space_idx.Min() >= -1,
                  "Size mismatch between fes and space_idx")

      // GridFunction initialization
      for (int i=0; i<fes.size(); i++)
      {
         gfs[i] = std::make_unique<ParGridFunction>(fes[i]);
      }

      // Compute Block offsets for each variable
      true_offsets.SetSize(0); true_offsets.Append(0);
      for (int i=0; i<numVars; i++)
      {
         if (space_idx[i] < 0) { true_offsets.Append(qs.GetSize()); }
         else { true_offsets.Append(fes[space_idx[i]]->GetTrueVSize()); }
      }
      true_offsets.PartialSum();
   }

   // Compute L2-objective, 0.5 * \sum_var ||x_var - x_initial_var||^2_{L2}
   void Mult(const Vector &x, Vector &y) const override
   {
      BlockVector x_block(const_cast<Vector&>(x), true_offsets);
      real_t result = 0.0;
      for (int i=0; i<numVars; i++)
      {
         const int sid = space_idx[i];
         if (sid < 0)
         {
            qf = x_block.GetBlock(i);
            qf -= x_initial.GetBlock(i);
            qf *= qf;
            result += qf.Integrate(); // this will take care of parallel reduction
         }
         else
         {
            add(x_block.GetBlock(i), -1.0, x_initial.GetBlock(i),
                gfs[sid]->GetTrueVector());
            gfs[sid]->SetFromTrueVector();
            real_t err = gfs[sid]->ComputeL2Error(zero_cf);
            result += err * err;
         }
      }
      y.SetSize(1);
      y[0] = 0.5 * result;
   }

   // Compute true gradient in L2 sense
   // Note that derivative is M*(x-x_initial), but we are using the true gradient.
   void EvalGradient(const Vector &x, Vector &grad) const override
   {
      add(x, -1.0, x_initial, grad);
   }

   /// Hessian is identity. So, HessianMult simply returns the directio d.
   void HessianMult(const Vector &x, const Vector &d, Vector &hess) const override
   {
      hess = d;
   }

   QuadratureSpace &GetQuadratureSpace() const { return qs; }
   std::vector<ParFiniteElementSpace*> GetFiniteElementSpaces() const { return fes; }
   Array<int> GetSpaceIdx() const { return space_idx; }


protected:
   QuadratureSpace &qs;
   std::vector<ParFiniteElementSpace*> fes;
   mutable QuadratureFunction qf;
   mutable std::vector<std::unique_ptr<ParGridFunction>> gfs;
   mutable ConstantCoefficient zero_cf;
   const int numVars;
   Array<int> space_idx;
   Array<int> true_offsets;
   const BlockVector &x_initial;
private:
};

class RemapProblem : public ConstrainedOptimizationProblem
{
public:
   RemapProblem(RemapObjective &objective,
                const BlockVector &x_min,
                const BlockVector &x_max,
                StackedSharedFunctional &C)
      : ConstrainedOptimizationProblem(objective, &C)
      , x_min(x_min), x_max(x_max)
   {}
   void Mult(const Vector &x, Vector &y) const override
   {
      objective.Mult(x, y);
      /// Do something with the constraints
   }
   void EvalGradient(const Vector &x, Vector &grad) const override
   {
      objective.EvalGradient(x, grad);
      /// Do something with the constraints
   }
   void HessianMult(const Vector &x, const Vector &d, Vector &hess) const override
   {
      objective.HessianMult(x, d, hess);
      /// Do something with the constraints
   }
   const Vector &GetLowerBounds() const { return x_min; }
   const Vector &GetUpperBounds() const { return x_max; }
   QuadratureSpace &GetQuadratureSpace() const { return static_cast<RemapObjective&>(objective).GetQuadratureSpace(); }
   std::vector<ParFiniteElementSpace*> GetFiniteElementSpaces() const
   {
      return static_cast<RemapObjective&>(objective).GetFiniteElementSpaces();
   }
   Array<int> GetSpaceIdx() const { return static_cast<RemapObjective&>(objective).GetSpaceIdx(); }
   // Compute KKT residual with grad = grad objective + <lambda, grad C>
   real_t ComputeKKT(const BlockVector &x,
                     const BlockVector &grad) const
   {
      real_t kkt = kkt_res(x, x_min, x_max, grad);
      MPI_Allreduce(MPI_IN_PLACE, &kkt, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                    GetComm());
      HYPRE_BigInt n = x.Size();
      MPI_Allreduce(MPI_IN_PLACE, &n, 1, MPITypeMap<HYPRE_BigInt>::mpi_type, MPI_SUM,
                    GetComm());
      return kkt / n;
   }

protected:
   const BlockVector &x_min; // lower bounds
   const BlockVector &x_max; // upper bounds
};

// Solve [alpha I_nn, I_nb; I_bn, -w I_bb]. As long as alpha > 0 and w >= 0, the system is well-posed.
// Here, I is a rectangular identity matrix corresponding to the bounds.
// That is, I_{bn}[i,j] = 1 when there is a bound for the i-th primal variable
class PointwiseSolver : public Operator
{
   Array<int> x_offsets, b_offsets;
   real_t alpha;
   const BlockVector *w;
   Array<bool> has_bounds;
public:

   PointwiseSolver(const Array<int> x_offsets, const Array<int> b_offsets)
      : Operator(x_offsets.Last() + b_offsets.Last())
      , x_offsets(x_offsets), b_offsets(b_offsets)
      , alpha(1.0), has_bounds(b_offsets.Size()-1)
   {
      MFEM_VERIFY(x_offsets.Size() == b_offsets.Size(),
                  "PointwiseSolver: Size mismatch between x_offsets and b_offsets");
      for (int i=0; i<b_offsets.Size()-1; i++)
      {
         has_bounds[i] = b_offsets[i+1] - b_offsets[i] > 0;
         if (has_bounds[i])
         {
            MFEM_VERIFY(b_offsets[i+1] - b_offsets[i] == x_offsets[i+1] - x_offsets[i],
                        "PointwiseSolver: Size mismatch between bounds and primal variables");
         }
      }
   }
   void Mult(const Vector &b, Vector &x) const override
   {
      const int numPrimalDof = x_offsets.Last();
      const int numLatentDof = b_offsets.Last();
      MFEM_VERIFY(numPrimalDof + numLatentDof == Width(),
                  "PointwiseSolver: Size mismatch between x and offsets");
      MFEM_VERIFY(b.Size() == numPrimalDof + b_offsets.Last(),
                  "PointwiseSolver: Size mismatch between b and offsets");
      BlockVector primal_b(const_cast<Vector&>(b), x_offsets);
      BlockVector latent_b(const_cast<Vector&>(b), numPrimalDof, b_offsets);

      x.SetSize(Width());
      BlockVector primal_x(x, x_offsets);
      BlockVector latent_x(x, numPrimalDof, b_offsets);

      const int numBlocks = primal_x.NumBlocks();

      // pointwise solve
      DenseMatrix A_point(2);
      A_point(0,0) = alpha; A_point(0,1) = 1.0; A_point(1,0) = 1.0;
      DenseMatrixInverse Ainv(A_point);
      Vector x_point(2), b_point(2);

      // for each block
      for (int i_block=0; i_block<numBlocks; i_block++)
      {
         if (!has_bounds[i_block]) // no bound
         {
            // alpha I = b
            primal_x.GetBlock(i_block) = primal_b.GetBlock(i_block);
            primal_x.GetBlock(i_block) /= alpha;
            continue;
         }

         // have bounds
         // alpha xp +   xl = bp
         //       xp - w xl = bl
         const Vector &bp = primal_b.GetBlock(i_block);
         const Vector &bl = latent_b.GetBlock(i_block);

         Vector &xp = primal_x.GetBlock(i_block);
         Vector &xl = latent_x.GetBlock(i_block);

         const Vector &wb = w->GetBlock(i_block);
         for (int i=0; i<bp.Size(); i++)
         {
            A_point(1,1) = -wb[i];
            b_point[0] = bp[i];
            b_point[1] = bl[i];
            Ainv.Factor();
            Ainv.Mult(b_point, x_point);
            xp[i] = x_point[0];
            xl[i] = x_point[1];
         }
      }
   }
   void Update(const real_t alpha, const BlockVector &w)
   {
      this->alpha = alpha;
      this->w = &w;
   };
};

class LVPPSolver : public IterativeSolver
{
public:
   // Constructor for the LVPP solver
   // problem: Remap problem
   // offsets: block offsets for the primal variables
   // b_offsets: block offsets for the bounds
   LVPPSolver(RemapProblem &problem, const Array<int> &offsets,
              const Array<int> &b_offsets)
      : IterativeSolver(problem.GetComm())
      , problem(problem), offsets(offsets), b_offsets(b_offsets)
      , pointwise_solver(offsets, b_offsets)
   {
      // input and output are primal variables
      width = height = offsets.Last();

      total_offsets = offsets;
      for (int i=0; i<b_offsets.Size()-1; i++)
      {
         total_offsets.Append(b_offsets[i+1] + width);
      }
   }

   LVPPSolver(RemapProblem &problem, const Array<int> &offsets)
      : LVPPSolver(problem, offsets, offsets)
   {}

   void IncludeConstraintHessian(bool use_hessian_approx)
   {
      this->use_hessian_approx = use_hessian_approx;
   }

   // Solves the LVPP problem
   /*
       It uses the Augmented Lagrangian method.
       For each AL iteration, it solves the subproblem
        min_x { L(x, lambda, mu) = F(x) + 0.5 mu * C(x)^T C(x) + <lambda, C(x)> }
       where F is the objective functional,
       C is the equality constraint operator,
       lambda is the Lagrange multiplier vector (initialized to zero),
       mu is the penalty parameter (initialized to 1.0).
       After each AL iteration, lambda <- lambda + mu * C(x)

       The subproblem is solved using LVPP method.

       min_x L(x, lambda, mu) + 1/alpha * D(x, x_k)
       where D is the Bregman distance operator,
       x_k is the previous iteration.
       Instead of D(x, x_k), we uses its dual form
       D(x, x_k) = D^*(psi, psi_k)
       where psi = sigmoid(x) with proper scaling.
   */
   void Mult(const Vector &x, Vector &y) const override
   {
      // Cast x to block vector.
      const BlockVector x_block(const_cast<Vector&>(x), offsets);

      // Get info from the problem
      Functional &obj = problem.GetObjective();
      StackedSharedFunctional &C = static_cast<StackedSharedFunctional&>
                                   (*problem.GetEqualityConstraints());
      // Create the Augmented Lagrangian functional
      // J(x) + 0.5 mu * C(x)^T C(x) + <lambda, C(x)>

      Array<int> space_idx = problem.GetSpaceIdx();

      const BlockVector x_max(const_cast<Vector&>(problem.GetUpperBounds()),
                              b_offsets);
      const BlockVector x_min(const_cast<Vector&>(problem.GetLowerBounds()),
                              b_offsets);
      Array<bool> has_bounds(0);
      for (int i=0; i<x_max.NumBlocks(); i++) { has_bounds.Append(x_max.GetBlock(i).Size() > 0); }

      const int numPrimalDof = x.Size(); // primal ndof
      const int numLatentDof = x_max.Size();
      const int numTotalDof = numPrimalDof + numLatentDof; // total ndof
      const int numConst = C.Height(); // num of constraints
      const int numVars = offsets.Size()-1;

      MFEM_VERIFY(numVars == x_min.NumBlocks()
                  && x.Size() == offsets.Last()
                  && x_max.Size() == b_offsets.Last(),
                  "Size mismatch between x, x_min, x_max and offsets, b_offsets");
      MFEM_VERIFY(total_offsets.Size() == numVars*2 + 1,
                  "Size mismatch between total_offsets and numVars");

      // Create an auxiliary vector for the inner solve
      BlockVector x_all(total_offsets);
      BlockVector x_primal(x_all, offsets);
      BlockVector x_latent(x_all, numPrimalDof, b_offsets);

      BlockVector dx_all(total_offsets);
      BlockVector dx_primal(dx_all, offsets);
      BlockVector dx_latent(dx_all, numPrimalDof, b_offsets);

      BlockVector G(total_offsets);
      BlockVector G_primal(G, offsets);
      BlockVector G_latent(G, numPrimalDof, b_offsets);

      // Latent auxiliary vector
      BlockVector x_latent_k(b_offsets);
      BlockVector Upsi(b_offsets); // Upsi = sigmoid(x)
      BlockVector dUpsi(b_offsets); // dUpsi = der_sigmoid(x)

      x_primal = x_block;
      x_latent = 0.0;

      // Values
      Vector obj_value(1);
      Vector C_x(numConst); // C(x)
      C.Mult(x_primal, C_x);
      // gradient. Will be used GN iteration
      BlockVector gradF(offsets);
      DenseMatrix gradC(numTotalDof, numConst);
      gradC = 0.0;
      DenseMatrix gradC_primal(numPrimalDof, numConst);
      Vector lambda(numConst);
      lambda = C_x;
      Vector new_lambda(numConst);

      real_t kkt_target = abs_tol;
      real_t alpha;

      // AL loop for Remap Problem
      for (int it_AL=0; it_AL<max_iter; it_AL++) // AL loop
      {
         real_t kkt_AL_target = prox_abs_tol;
         // PG loop for AL Subproblem
         for (int it_PG=0; it_PG<prox_max_iter; it_PG++)
         {
            x_latent_k = x_latent;
            // alpha = 1.0;
            alpha = alpha0*std::pow(it_PG + 1, 2);
            real_t res_PG_target = nonlin_abs_tol;
            // Nonlinear loop for PG subproblem.
            for (int it_GN=0; it_GN<nonlin_max_iter; it_GN++)
            {
               // Objective
               // obj.Mult(x_primal, obj_value);
               obj.GetGradient().Mult(x_primal, gradF);

               // Constraints
               C.Mult(x_primal, C_x);
               C.GetGradientMatrix(x_primal, gradC_primal);

               // (grad C(x))*(lambda + mu * C(x))
               add(lambda, mu, C_x, new_lambda);
               gradC_primal.Mult(new_lambda, G_primal);
               // grad F(x) + (grad C(x))*(lambda + mu * C(x))
               G_primal += gradF;
               // alpha F(x) + (grad C(x))*(lambda + mu * C(x))
               G_primal *= alpha;

               // [alpha F(x) + (grad C(x))*(lambda + mu*C(x)) + (psi - psi_k)]
               // [u - U(psi)]
               MapLatent(x_latent, x_min, x_max, Upsi, dUpsi);
               for (int i=0; i<numVars; i++)
               {
                  if (!has_bounds[i]) { continue; }
                  G_primal.GetBlock(i) += x_latent.GetBlock(i);
                  G_primal.GetBlock(i) -= x_latent_k.GetBlock(i);
                  add(x_primal.GetBlock(i), -1.0, Upsi.GetBlock(i), G_latent.GetBlock(i));
               }
               // Hessian solver. Hessian is I + hess (mu||C(x)||^2 + lambda^T C(x))
               // However, we approximate Hessian of the second term by mu*(grad C)(grad C)^T
               // This leads to a modified Gauss-Newton method, and can be solved
               // using Woodbury formula
               pointwise_solver.Update(alpha, dUpsi);
               if (!use_hessian_approx)
               {
                  // Linear constraints, so we can use the pointwise solver directly
                  pointwise_solver.Mult(G, dx_all);
               }
               else
               {
                  // pointwise_solver.Mult(G, dx_all);
                  gradC.SetSubMatrix(0, 0, gradC_primal);
                  Woodbury(GetComm(), pointwise_solver, mu*alpha, gradC, gradC,
                           G, dx_all);
               }
               x_all -= dx_all;
               real_t succdiff_norm = std::sqrt(Dot(dx_primal, dx_primal));
               real_t violate_norm = std::sqrt(Dot(G_latent, G_latent));

               if (it_GN == 0) { res_PG_target = std::max(res_PG_target, succdiff_norm*nonlin_rel_tol); }
               if (print_level >= 3)
               {
                  out << "    GN Iteration " << it_GN + 1 << ", ||dx||_l2 = " << succdiff_norm <<
                         ", Latent primal inconsistency: " << violate_norm << std::endl;
               }
               if (succdiff_norm < res_PG_target)
               {
                  if (print_level >= 2)
                  {
                     out << "    PG Converged in " << it_GN + 1
                         << " iterations, residual norm = " << succdiff_norm << std::endl;
                  }
                  break; // PG subproblem converged
               }
            } // end of PG subproblem loop
            obj.Mult(x_primal, obj_value);
            obj.GetGradient().Mult(x_primal, gradF);

            C.Mult(x_primal, C_x);
            C.GetGradientMatrix(x_primal, gradC_primal);

            add(lambda, mu, C_x, new_lambda);
            gradC_primal.Mult(lambda, G_primal);
            G_primal += gradF;

            real_t kkt_AL = problem.ComputeKKT(x_primal, G_primal);
            // If first iteration, setup the relative tolerance
            if (it_PG == 0) { kkt_AL_target = std::max(kkt_AL_target, kkt_AL*prox_rel_tol); }
            if (print_level >= 2)
            {
               out << "  PG Iteration " << it_PG + 1 << ", KKT residual = " << kkt_AL <<
                   std::endl;
            }
            if (kkt_AL < kkt_AL_target)
            {
               if (print_level >= 1)
               {
                  out << "  AL subproblem converged in " << it_PG + 1
                      << " PG iterations, KKT residual = " << kkt_AL << std::endl;
               }
               break; // AL subproblem solved
            }
         } // end of AL subproblem loop
         // Check KKT for Remap Problem, grad obj(x) + grad C(x)*lambda points inward
         obj.Mult(x_primal, obj_value);
         obj.GetGradient().Mult(x_primal, gradF);

         C.Mult(x_primal, C_x);
         C.GetGradientMatrix(x_primal, gradC_primal);

         lambda.Add(mu, C_x);
         gradC_primal.Mult(lambda, G_primal);
         G_primal += gradF;

         real_t kkt = problem.ComputeKKT(x_primal, G_primal);

         if (print_level >= 1)
         {
            out << "AL Iteration " << it_AL + 1 << ", KKT residual = " << kkt << std::endl;
            out << "Objective value = " << obj_value[0] << ", " << "Constraint residual = ";
            for (int i=0; i<numConst; i++) { out << C_x[i] << " "; } out << std::endl;
         }

         // If first iteration, setup the relative tolerance.
         if (it_AL == 0) { kkt_target = std::max(kkt_target, kkt*rel_tol); }
         if ((C_x.Normlinf() < kkt_target) && it_AL > 0)
         {
            if (print_level >= 0)
            {
               out << "Remap Problem converged in " << it_AL + 1
                   << " iterations, KKT residual = " << kkt << std::endl;
               out << "Objective value = " << obj_value[0] << ", " << "Constraint residual = ";
               for (int i=0; i<numConst; i++) { out << C_x[i] << " "; } out << std::endl;
            }
            break; // Remap Problem solved
         }
      } // end of AL loop
      y = x_primal;
   }
   void SetAbsTol(real_t tol) { abs_tol = tol; }
   void SetRelTol(real_t tol) { rel_tol = tol; }
   void SetProxMaxIter(int n) { prox_max_iter = n; }
   void SetProxAbsTol(real_t tol) { prox_abs_tol = tol; }
   void SetProxRelTol(real_t tol) { prox_rel_tol = tol; }
   void SetNonlinMaxIter(int n) { nonlin_max_iter = n; }
   void SetNonlinAbsTol(real_t tol) { nonlin_abs_tol = tol; }
   void SetNonlinRelTol(real_t tol) { nonlin_rel_tol = tol; }
   void SetAlpha(real_t alpha) { this->alpha0 = alpha; }
   void SetPenalty(real_t mu) { this->mu = mu; }
protected:
   // Inner solver parameters
   real_t alpha0 = 1.0;
   real_t mu = 1.0;
   bool use_hessian_approx = false; // whether the constraints are linear
   int prox_max_iter = 100;
   real_t prox_abs_tol = 1e-06;
   real_t prox_rel_tol = 1e-06;
   int nonlin_max_iter = 100;
   real_t nonlin_abs_tol = 1e-08;
   real_t nonlin_rel_tol = 1e-08;

   void MapLatent(const BlockVector &psi, const BlockVector &xmin,
                  const BlockVector &xmax, BlockVector &Upsi, BlockVector &dUpsi) const
   {
      const int N = psi.Size();
      const bool use_dev = psi.UseDevice() || xmin.UseDevice() || xmax.UseDevice() ||
                           Upsi.UseDevice();
      auto psi_ = psi.Read(use_dev);
      auto xmin_ = xmin.Read(use_dev);
      auto xmax_ = xmax.Read(use_dev);
      auto Upsi_ = Upsi.Write(use_dev);
      auto dUpsi_ = dUpsi.Write(use_dev);
      mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i)
      {
         Upsi_[i] = sigmoid(psi_[i], xmin_[i], xmax_[i]);
         dUpsi_[i] = der_sigmoid(psi_[i], xmin_[i], xmax_[i]);
      });
   }

   RemapProblem &problem; // The LVPP problem to solve

   Array<int> offsets; // primal offsets
   Array<int> b_offsets; // bound offsets
   Array<int> total_offsets; // both primal and latent
   mutable PointwiseSolver pointwise_solver;
};


}

#endif // REMHOS_LVPP_HPP
