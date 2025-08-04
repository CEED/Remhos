#ifndef REMHOS_LVPP_HPP
#define REMHOS_LVPP_HPP
#include "mfem.hpp"
#include "remap.hpp"

namespace mfem
{
inline void MapLatent(const Vector &latent, const Vector &x_min,
                      const Vector &x_max, Vector &primal)
{
   const int N = latent.Size();
   const bool use_dev = latent.UseDevice() || x_min.UseDevice() ||
                        x_max.UseDevice() || primal.UseDevice();
   auto psi_ = latent.Read(use_dev);
   auto xmin_ = x_min.Read(use_dev);
   auto xmax_ = x_max.Read(use_dev);
   auto primal_ = primal.Write(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i)
   {
      primal_[i] = sigmoid(psi_[i], xmin_[i], xmax_[i]);
   });
}

inline void MapLatent(const Vector &latent, const Vector &x_min,
                      const Vector &x_max, Vector &primal, Vector &dprimal)
{
   const int N = latent.Size();
   const bool use_dev = latent.UseDevice() || x_min.UseDevice() ||
                        x_max.UseDevice() || primal.UseDevice();
   auto psi_ = latent.Read(use_dev);
   auto xmin_ = x_min.Read(use_dev);
   auto xmax_ = x_max.Read(use_dev);
   auto primal_ = primal.Write(use_dev);
   auto dprimal_ = dprimal.Write(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i)
   {
      primal_[i] = sigmoid(psi_[i], xmin_[i], xmax_[i]);
   });
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i)
   {
      dprimal_[i] = der_sigmoid(psi_[i], xmin_[i], xmax_[i]);
   });
}

inline void MapPrimal(const Vector &primal, const Vector &x_min,
                      const Vector &x_max, Vector &latent,
                      const real_t maxval = 20.0)
{
   const int N = primal.Size();
   const bool use_dev = latent.UseDevice() || x_min.UseDevice() ||
                        x_max.UseDevice() || primal.UseDevice();
   auto primal_ = primal.Read(use_dev);
   auto xmin_ = x_min.Read(use_dev);
   auto xmax_ = x_max.Read(use_dev);
   auto latent_ = latent.Write(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i)
   {
      latent_[i] = std::clamp(logit(primal_[i], xmin_[i], xmax_[i]), -maxval, maxval);
   });
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

// Solve [alpha I_nn, I_nb; I_bn, -w I_bb]. As long as alpha > 0 and w >= 0, the system is well-posed.
// Here, I is a rectangular identity matrix corresponding to the bounds.
// That is, I_{bn}[i,j] = 1 when there is a bound for the i-th primal variable
class PointwiseSolver : public Operator
{
   Array<int> x_offsets, b_offsets;
   real_t alpha;
   const BlockVector *w;
   Array<bool> has_latent;
public:

   PointwiseSolver(const Array<int> x_offsets, const Array<int> l_offsets)
      : Operator(x_offsets.Last() + l_offsets.Last())
      , x_offsets(x_offsets), b_offsets(l_offsets)
      , alpha(1.0), has_latent(l_offsets.Size()-1)
   {
      MFEM_VERIFY(x_offsets.Size() == l_offsets.Size(),
                  "PointwiseSolver: Size mismatch between x_offsets and b_offsets");
      for (int i=0; i<l_offsets.Size()-1; i++)
      {
         has_latent[i] = (l_offsets[i+1] - l_offsets[i]) > 0;
         if (has_latent[i])
         {
            MFEM_VERIFY(l_offsets[i+1] - l_offsets[i] == x_offsets[i+1] - x_offsets[i],
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
         if (!has_latent[i_block]) // no bound
         {
            // alpha I = b
            primal_x.GetBlock(i_block).Set(1.0 / alpha, primal_b.GetBlock(i_block));
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
   LVPPSolver(remap::RemapProblem &problem, const Array<int> &offsets,
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

   LVPPSolver(remap::RemapProblem &problem, const Array<int> &offsets)
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
      QuadratureSpace &qspace = problem.GetQuadratureSpace();
      std::vector<ParFiniteElementSpace*> fespaces = problem.GetFiniteElementSpaces();
      std::vector<std::unique_ptr<MassOperator>> mass_ops(fespaces.size() + 1);
      for(int i=0; i<fespaces.size(); i++)
      {
         mass_ops[i] = std::make_unique<MassOperator>(*fespaces[i]);
      }
      mass_ops.back() = std::make_unique<MassOperator>(qspace);

      Array<int> space_idx = problem.GetSpaceIdx();

      const BlockVector x_max(const_cast<Vector&>(problem.GetUpperBounds()),
                              b_offsets);
      const BlockVector x_min(const_cast<Vector&>(problem.GetLowerBounds()),
                              b_offsets);
      MFEM_VERIFY(x_min.Size() == x_max.Size() && x_min.Size() == b_offsets.Last(),
                  "Size mismatch between x_min, x_max and b_offsets");
      Array<bool> has_bounds(x_max.NumBlocks());
      for (int i=0; i<x_max.NumBlocks(); i++)
      {
         has_bounds[i] = x_max.GetBlock(i).Size() > 0;
      }

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

      BlockVector res(total_offsets);
      BlockVector res_primal(res, offsets);
      BlockVector res_latent(res, numPrimalDof, b_offsets);

      // Latent auxiliary vector
      BlockVector x_latent_k(b_offsets);
      BlockVector Upsi(b_offsets); // Upsi = sigmoid(x)
      BlockVector dUpsi(b_offsets); // dUpsi = der_sigmoid(x)
      for (int i=0; i<numVars; i++)
      {
         if (!has_bounds[i]) { x_primal.GetBlock(i) = x_block.GetBlock(i); }
         else
         {
            add(x_min.GetBlock(i), 1.0, x_max.GetBlock(i), x_primal.GetBlock(i));
            x_primal.GetBlock(i) *= 0.5;
         }
      }

      MapPrimal(x_primal, x_min, x_max, x_latent, 20.0);

      // Values
      Vector obj_value(1);
      Vector C_x(numConst); // C(x)
      C.Mult(x_primal, C_x);
      real_t C_x_prev = C_x.Normlinf();
      // gradient. Will be used GN iteration
      BlockVector gradF(offsets);
      DenseMatrix gradC(numTotalDof, numConst);
      gradC = 0.0;
      DenseMatrix gradC_primal(numPrimalDof, numConst);
      C.GetGradientMatrix(x_primal, gradC_primal);
      Vector lambda(numConst);
      lambda = 0.0;
      Vector new_lambda(numConst);
      real_t last_min_Cx = infinity();
      real_t mu = 1.0 / gradC_primal.MaxMaxNorm();

      real_t kkt_target = abs_tol;
      real_t alpha;

      int it_AL, it_PG, it_GN_all, it_GN;
      // AL loop for Remap Problem
      real_t kkt = prox_abs_tol0;
      real_t prox_abs_tol = prox_abs_tol0;
      real_t prox_rel_tol = prox_rel_tol0;
      for (int it_AL=0; it_AL<max_iter; it_AL++) // AL loop
      {
         it_GN_all = 0;
         real_t kkt_AL_target = prox_abs_tol;
         real_t kkt_prev = 0.0;
         // PG loop for AL Subproblem
         for (it_PG=0; it_PG<prox_max_iter; it_PG++)
         {
            x_latent_k = x_latent;
            alpha = alpha0;
            // alpha = alpha0*std::pow(1.0 + it_PG, 1);

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
            for (int i=0; i<numVars; i++)
            {
               mass_ops[(space_idx[i] + mass_ops.size()) % mass_ops.size()]->Riesz(
                  G_primal.GetBlock(i), res_primal.GetBlock(i));
            }

            // [alpha F(x) + (grad C(x))*(lambda + mu*C(x)) + (psi - psi_k)]
            for (int i=0; i<numVars; i++)
            {
               if (!has_bounds[i]) { continue; }
               G_primal.GetBlock(i) += x_latent.GetBlock(i);
               G_primal.GetBlock(i) -= x_latent_k.GetBlock(i);
            }
            dx_latent = 0.0;
            real_t res_PG_target = nonlin_abs_tol;
            res_primal = G_primal;
            gradC.SetSubMatrix(0, 0, gradC_primal);
            // Nonlinear loop for PG subproblem.
            for (it_GN=0; it_GN<nonlin_max_iter; it_GN++)
            {
               // [alpha F(x) + (grad C(x))*(lambda + mu*C(x)) + (psi - psi_k)]
               // [u - U(psi)]
               MapLatent(x_latent, x_min, x_max, Upsi, dUpsi);
               for (int i=0; i<numVars; i++)
               {
                  if (!has_bounds[i]) { continue; }
                  res_primal.GetBlock(i) -= dx_latent.GetBlock(i);
                  add(x_primal.GetBlock(i), -1.0, Upsi.GetBlock(i), res_latent.GetBlock(i));
               }
               // Hessian solver. Hessian is I + hess (mu||C(x)||^2 + lambda^T C(x))
               // However, we approximate Hessian of the second term by mu*(grad C)(grad C)^T
               // This leads to a modified Gauss-Newton method, and can be solved
               // using Woodbury formula
               pointwise_solver.Update(alpha, dUpsi);
               if (!use_hessian_approx)
               {
                  pointwise_solver.Mult(res, dx_all);
               }
               else
               {
                  Woodbury(GetComm(), pointwise_solver, mu*alpha, gradC, gradC,
                           res, dx_all);
               }
               x_all -= dx_all;
               for (auto &v : x_latent) { v = std::min(std::max(v, -1e08), 1e08); }
               real_t succdiff_norm = std::sqrt(Dot(dx_primal, dx_primal));
               real_t violate_norm = std::sqrt(Dot(res_latent, res_latent));
               real_t res_norm = std::max(succdiff_norm, violate_norm);

               if (it_GN == 0) { res_PG_target = std::max(res_PG_target, res_norm*nonlin_rel_tol); }
               if (print_level >= 3)
               {
                  out << "    GN Iteration " << it_GN + 1 << ", ||dx||_l2 = " << succdiff_norm <<
                         ", Latent primal inconsistency: " << violate_norm << std::endl;
               }
               if (res_norm < res_PG_target)
               {
                  if (print_level >= 2)
                  {
                     out << "    PG Converged in " << it_GN + 1
                         << " iterations, residual norm = " << succdiff_norm << std::endl;
                  }
                  it_GN++;
                  break; // PG subproblem converged
               }
            } // end of PG subproblem loop
            it_GN_all += it_GN;
            obj.Mult(x_primal, obj_value);

            C.Mult(x_primal, C_x);
            add(lambda, mu, C_x, new_lambda);

            C.GetGradient(x_primal).Mult(new_lambda, G_primal);
            obj.GetGradient().AddMult(x_primal, G_primal);

            last_min_Cx = std::min(C_x.Normlinf(), last_min_Cx);

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
         lambda.Add(mu, C_x);
         real_t mu_prev = mu;
         if (C_x.Normlinf() > C_x_prev && C_x.Normlinf() > const_abs_tol)
         {
            mu = std::min(mu*1.05, 1e02);
         }
         C_x_prev = C_x.Normlinf();

         kkt = problem.ComputeKKT(x_primal, G_primal);

         if (print_level >= 1)
         {
            out << "AL Iteration " << it_AL + 1 << " with mu = " << mu_prev << " (" << it_PG
                + 1 << " - " << it_GN_all
                << "), KKT residual = " << kkt << std::endl;
            out << "Objective value = " << obj_value[0] << ", " << "Constraint residual = "
                << C_x.Normlinf() << " (";
            for (int i=0; i<numConst; i++) { out << C_x[i] << " "; } out << "\b)" <<
                  std::endl;
         }

         // If first iteration, setup the relative tolerance.
         if (it_AL == 0) { kkt_target = std::max(kkt_target, kkt*rel_tol); }
         if ((kkt < kkt_target && C_x.Normlinf() < const_abs_tol) && it_AL > 0)
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

   void SetConstAbsTol(real_t tol) { const_abs_tol = tol; }

   void SetProxMaxIter(int n) { prox_max_iter = n; }
   void SetProxAbsTol(real_t tol) { prox_abs_tol0 = tol; }
   void SetProxRelTol(real_t tol) { prox_rel_tol0 = tol; }

   void SetNonlinMaxIter(int n) { nonlin_max_iter = n; }
   void SetNonlinAbsTol(real_t tol) { nonlin_abs_tol = tol; }
   void SetNonlinRelTol(real_t tol) { nonlin_rel_tol = tol; }
   void SetAlpha(real_t alpha) { this->alpha0 = alpha; }
   void SetPenalty(real_t mu) { this->mu0 = mu; }
protected:
   // Inner solver parameters
   real_t alpha0 = 1.0;
   real_t mu0 = 1.0;
   bool use_hessian_approx = false; // whether the constraints are linear

   real_t const_abs_tol = 1e-06;

   int prox_max_iter = 100;
   mutable real_t prox_abs_tol0 = 1e-06;
   mutable real_t prox_rel_tol0 = 1e-06;
   int nonlin_max_iter = 100;
   real_t nonlin_abs_tol = 1e-08;
   real_t nonlin_rel_tol = 1e-08;

   remap::RemapProblem &problem; // The LVPP problem to solve

   Array<int> offsets; // primal offsets
   Array<int> b_offsets; // bound offsets
   Array<int> total_offsets; // both primal and latent
   mutable PointwiseSolver pointwise_solver;
};

// Proximal gradient functional
// Only uses latent variable for now.
// Mult will evaluate the functional without the Bregman divergence
// EvalGradient will evaluate the gradient
// alpha*J(U(x)) + x - x_k
// where x_k should be set before the first call.
class PGFunctional : public Functional
{
private:
   const Functional &f;
   const Vector &x_min, &x_max;
   Vector *latent_k;
   real_t alpha = 1.0;
   mutable Vector primal;
public:
   PGFunctional(const Functional &f, const Vector &x_min,
                const Vector &x_max)
      : Functional(f.GetComm(), f.Width())
      , f(f), x_min(x_min), x_max(x_max)
      , primal(f.Width())
   { }

   void SetLatent(Vector &latent) { this->latent_k = &latent; }
   void SetStepSize(real_t alpha) { this->alpha = alpha; }
   void Mult(const Vector &x, Vector &y) const override
   {
      MapLatent(x, x_min, x_max, primal);
      f.Mult(primal, y);
   }
   void EvalGradient(const Vector &latent, Vector &grad) const override
   {
      MFEM_VERIFY(latent_k != nullptr, "Latent vector is not set.");
      MapLatent(latent, x_min, x_max, primal);
      f.GetGradient().Mult(primal, grad);
      grad *= alpha;
      grad += latent;
      grad -= *latent_k;
   }
};

// LBFGS in the Mirror Descent Framework
// The original BFGS method uses
// the primal difference (s_k) and the object gradient difference (y_k).
// Instead, we use the latent difference
// s_k = psi_{k+1} - psi_k
// Then the mirror descent update becomes
// psi_{k+1} = psi_k + eta_k * d_k
// where d_k is the usual L-BFGS search direction
// eta_k is determined by the line search to satisfy the curvature condition
// <s_k, y_k> > 0. See LineSearch() for details.
class ProxLBFGS : public Optimizer
{
public:
   ProxLBFGS(Functional &obj,
             const Vector &x_min,
             const Vector &x_max,
             MultiL2RieszMap &riesz,
             const int m = 10)
      : Optimizer(obj.GetComm())
      , x_min(x_min), x_max(x_max)
      , g(obj.Width()), g_k(obj.Width())
      , d(obj.Width())
      , m(m)
      , s(m), y(m), rho(m)
      , riesz(riesz)
   {
      rho = 0.0;
      Optimizer::SetOperator(obj);
   }

   void Mult(const Vector &latent_0, Vector &latent) const override
   {
      latent = latent_0;

      current_primal.SetSize(latent.Size());
      g.SetSize(latent.Size());
      g_k.SetSize(latent.Size());

      real_t tol = std::max(abs_tol, rel_tol * std::sqrt(Dot(g,g)));
      PGFunctional pgf(*subproblem, x_min, x_max);

      for (final_iter=0; final_iter<max_iter; final_iter++)
      {
         g_k = g;
         g_k -= latent_0;
         g_k += latent;
         GetDirection(final_iter, g_k, d);
         rho[final_iter % m] = LineSearch(latent_0, d, g_k,
                                          latent, current_primal, g,
                                          y[final_iter % m], s[final_iter % m]);
         if (rho[final_iter % m] < 0)
         {
            if (print_level >= 0) { MFEM_WARNING("Line search failed in MirrorLBFGS. Ignore current iteration."); }
            final_iter--;
         }
         real_t kkt_res = kkt(current_primal, g);
         if (print_level >= 2) { out << "MirrorLBFGS Iteration " << final_iter + 1 << ": ||kkt||_L2 = " << kkt_res << std::endl; }
         if (kkt_res < tol)
         {
            if (print_level >= 1) { out << "MirrorLBFGS converged in " << final_iter + 1 << " iterations, gradient norm = " << kkt_res << std::endl; }
            return; // converged
         }
      }
      final_iter--;
      real_t kkt_res = kkt(current_primal, g);
      if (print_level >= 0)
      {
         MFEM_WARNING("MirrorLBFGS failed to converge in " << max_iter
                      << " iterations with gradient norm = " << kkt_res << ".");
      }
   }

   void SetPrimalVector(Vector &x) const
   {
      current_primal.MakeRef(x, 0, x.Size());
   }

protected:

   virtual real_t kkt(const Vector &x, const Vector &gradient) const
   {
      masked_grad = gradient;
      for (int i=0; i<x.Size(); i++)
      {
         if (x[i] < x_min[i] + 1e-08 && masked_grad[i] > 0.0)
         {
            masked_grad[i] = 0.0; // zero out the gradient at the lower bound
         }
         if (x[i] > x_max[i] - 1e-08 && masked_grad[i] < 0.0)
         {
            masked_grad[i] = 0.0; // zero out the gradient at the lower bound
         }
      }
      return std::sqrt(Dot(masked_grad, masked_grad));
   }

   // Simple Line search to ensure <sk, yk> > 0
   // returns positive rho = 1 / <sk, yk> when successful
   // returns negative rho when failed
   virtual bool LineSearch(const Vector &latent_k,
                           const Vector &direction_k,
                           const Vector &gradient_k,
                           Vector &latent,
                           Vector &primal,
                           Vector &gradient,
                           Vector &yk,
                           Vector &sk) const
   {
      real_t eta = 1.0; // initial step size

      yk.SetSize(latent.Size());
      sk.SetSize(latent.Size());
      Vector obj_value(1);

      MapLatent(latent, x_min, x_max, primal);
      subproblem->Mult(primal, obj_value);
      real_t obj_value0 = obj_value[0];

      real_t dk_dot_sk;
      bool success = false;
      for (int i=0; i<30; i++) // until step size > 2^-10 \approx 1e-03
      {
         // psi = psik + eta*dk
         add(latent_k, eta, direction_k, latent);
         for (auto &v : latent) { v = std::min(std::max(v, -1e02), 1e02); }
         // x = U(psi)
         MapLatent(latent, x_min, x_max, primal);
         // update the gradient
         subproblem->Mult(primal, obj_value);
         oper->Mult(primal, gradient);
         // yk = g - gk;
         subtract(gradient, gradient_k, yk);
         subtract(latent, latent_k, sk);
         // sk = psi - psi_k = eta*dk;
         // angle = Dot(yk, sk)/|yk||sk| = Dot(yk, dk)/|yk||dk|.
         // To check positivity condition, we only consider Dot(yk, dk)
         dk_dot_sk = Dot(yk, sk);
         if (dk_dot_sk > 0.0)
         {
            success = true;
            break; // found a positive curvature condition
         }
         eta *= 0.5;
      }
      // if (!success)
      // {
      // if (print_level >= 0)
      // {
      //    MFEM_WARNING("Line search failed in MirrorLBFGS. "
      //                 "Check the problem setup and the initial guess.");
      // }
      // }
      // MFEM_VERIFY(dk_dot_yk > 0,
      //             "MirrorLBFGS: Line search failed to find a positive curvature "
      //             "condition <sk, yk> > 0. Check the problem setup and the initial guess.");
      return 1.0 / dk_dot_sk; // rho = 1 / <sk, yk>
   }

   real_t Dot(const Vector &u, const Vector &v) const override
   {
      return riesz.InnerProduct(u, v);
   }

   // Get the search direction from the current gradient
   // @param k: current iteration number
   // @param g: current gradient (not derivative)
   // @param d: output primal search direction
   virtual void GetDirection(const int k, const Vector &gradient,
                             Vector &direction) const
   {
      // https://en.wikipedia.org/wiki/Limited-memory_BFGS#Algorithm
      direction = gradient;
      Vector alpha(m);
      for (int i = k - 1; i >= std::max(k - m, 0); i--)
      {
         const int idx = i % m;
         // out << "i = " << i << ", idx = i % m = " << idx << std::endl;
         // MFEM_VERIFY(s[idx].Size() == direction.Size() &&
         //             y[idx].Size() == direction.Size(),
         //             "MirrorLBFGS: s[" << idx << "] and y[" << idx
         //             << "] must have the same size as the search direction at " << k
         //             << ". Check the problem setup and the initial guess.");
         // MFEM_VERIFY(s[idx].CheckFinite() == 0,
         //             "MirrorLBFGS: s[" << idx << "] is not finite at " << k << ". "
         //             << "Check the problem setup and the initial guess.");
         // MFEM_VERIFY(y[idx].CheckFinite() == 0,
         //             "MirrorLBFGS: y[" << idx << "] is not finite at " << k << ". "
         //             << "Check the problem setup and the initial guess.");
         alpha[idx] = rho[idx] * Dot(direction, s[idx]);
         direction.Add(-alpha[idx], y[idx]);
      }

      if (k > 0)
      {
         const int idx = std::max(k - m, 0) % m;
         const real_t gamma = Dot(s[idx], y[idx]) / Dot(y[idx], y[idx]);
         // MFEM_VERIFY(CheckFinite(&gamma, 1) == 0,
         //             "MirrorLBFGS: gamma is not finite at " << k << ". "
         //             << "Check the problem setup and the initial guess.");
         direction *= gamma;
      }

      for (int i = std::max(k - m, 0); i < k; i++)
      {
         const int idx = i % m;
         // out << "i = " << i << ", idx = i % m = " << idx << std::endl;
         // MFEM_VERIFY(s[idx].CheckFinite() == 0,
         //             "MirrorLBFGS: s[" << idx << "] is not finite at " << k << ". "
         //             << "Check the problem setup and the initial guess.");
         // MFEM_VERIFY(y[idx].CheckFinite() == 0,
         //             "MirrorLBFGS: y[" << idx << "] is not finite at " << k << ". "
         //             << "Check the problem setup and the initial guess.");
         // MFEM_VERIFY(CheckFinite(&rho[idx], 1) == 0,
         //             "MirrorLBFGS: rho is not finite at " << k << ". "
         //             << "Check the problem setup and the initial guess.");
         real_t beta = rho[idx] * Dot(direction, y[idx]);
         // real_t norm = Dot(direction, y[idx]);
         // MFEM_VERIFY(CheckFinite(&norm, 1) == 0,
         //             "MirrorLBFGS: norm is not finite at " << k << ". "
         //             << "Check the problem setup and the initial guess.");
         // MFEM_VERIFY(CheckFinite(&beta, 1) == 0,
         //             "MirrorLBFGS: beta is not finite at " << k << ". "
         //             << "Check the problem setup and the initial guess.");
         direction.Add(alpha[idx] - beta, s[idx]);
      }
      direction.Neg();
   }
   const Vector &x_min; // lower bounds
   const Vector &x_max;
   mutable Vector current_primal;
   mutable Vector latent_prev;
   mutable Vector g;
   mutable Vector g_k;
   mutable Vector d;
   mutable Vector masked_grad;
   int m;
   MultiL2RieszMap &riesz;

   mutable std::vector<Vector> s;
   mutable std::vector<Vector> y;
   mutable Vector rho;
};

class LBFGSLVPPSolver : public IterativeSolver
{
public:
   // Constructor for the LVPP solver
   // problem: Remap problem
   // offsets: block offsets for the primal variables
   // b_offsets: block offsets for the bounds
   LBFGSLVPPSolver(remap::RemapProblem &problem, const Array<int> &offsets,
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

   LBFGSLVPPSolver(remap::RemapProblem &problem, const Array<int> &offsets)
      : LBFGSLVPPSolver(problem, offsets, offsets)
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
      MultiL2RieszMap riesz(problem.GetQuadratureSpace(),
                            problem.GetFiniteElementSpaces(),
                            problem.GetSpaceIdx());


      Array<int> space_idx = problem.GetSpaceIdx();

      const BlockVector x_max(const_cast<Vector&>(problem.GetUpperBounds()),
                              b_offsets);
      const BlockVector x_min(const_cast<Vector&>(problem.GetLowerBounds()),
                              b_offsets);
      MFEM_VERIFY(x_min.Size() == x_max.Size() && x_min.Size() == b_offsets.Last(),
                  "Size mismatch between x_min, x_max and b_offsets");
      Array<bool> has_bounds(x_max.NumBlocks());
      for (int i=0; i<x_max.NumBlocks(); i++)
      {
         has_bounds[i] = x_max.GetBlock(i).Size() > 0;
      }

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

      BlockVector res(total_offsets);
      BlockVector res_primal(res, offsets);
      BlockVector res_latent(res, numPrimalDof, b_offsets);

      // Latent auxiliary vector
      BlockVector x_latent_k(b_offsets);
      BlockVector Upsi(b_offsets); // Upsi = sigmoid(x)
      BlockVector dUpsi(b_offsets); // dUpsi = der_sigmoid(x)

      AugLagrangianFunctional aug_lag(obj, C);
      ProxLBFGS mlbfgs(aug_lag, x_min, x_max, riesz, 300);
      mlbfgs.SetPrintLevel(print_level - 1);
      mlbfgs.SetMaxIter(prox_max_iter);
      mlbfgs.SetAbsTol(prox_abs_tol0);
      mlbfgs.SetRelTol(prox_rel_tol0);
      mlbfgs.SetPrimalVector(
         x_primal); // x_primal will be updated when Mult() is called

      for (int i=0; i<numVars; i++)
      {
         if (!has_bounds[i]) { x_primal.GetBlock(i) = x_block.GetBlock(i); }
         else
         {
            add(x_min.GetBlock(i), 1.0, x_max.GetBlock(i), x_primal.GetBlock(i));
            x_primal.GetBlock(i) *= 0.5;
         }
      }

      x_latent = 0.0;

      // Values
      Vector obj_value(1);
      Vector C_x(numConst); // C(x)
      C.Mult(x_primal, C_x);
      real_t C_x_prev = C_x.Normlinf();
      // gradient. Will be used GN iteration
      BlockVector gradF(offsets);
      DenseMatrix gradC(numTotalDof, numConst);
      gradC = 0.0;
      DenseMatrix gradC_primal(numPrimalDof, numConst);
      C.GetGradientMatrix(x_primal, gradC_primal);
      Vector lambda(numConst);
      lambda = 0.0;
      Vector new_lambda(numConst);
      real_t last_min_Cx = infinity();
      real_t mu = 1.0 / gradC_primal.MaxMaxNorm();

      real_t kkt_target = abs_tol;
      real_t alpha;

      int it_AL, it_PG, it_GN_all, it_GN;
      // AL loop for Remap Problem
      real_t kkt = prox_abs_tol0;
      real_t prox_abs_tol = prox_abs_tol0;
      real_t prox_rel_tol = prox_rel_tol0;
      for (int it_AL=0; it_AL<max_iter; it_AL++) // AL loop
      {
         it_GN_all = 0;
         real_t kkt_AL_target = prox_abs_tol;
         real_t kkt_prev = 0.0;
         aug_lag.SetLambda(lambda); aug_lag.SetPenalty(mu);
         for (int it_PG = 0; it_PG < prox_max_iter; it_PG++)
         {
            // PG loop for AL Subproblem
            x_latent_k = x_latent;
            mlbfgs.Mult(x_latent_k, x_latent); // update x_primal too
         }

         obj.Mult(x_primal, obj_value);
         C.Mult(x_primal, C_x);

         lambda.Add(mu, C_x);
         real_t mu_prev = mu;

         // if (C_x.Normlinf() > C_x_prev && C_x.Normlinf() > const_abs_tol)
         // {
         //    mu = std::min(mu*1.5, 1e02);
         // }
         C_x_prev = C_x.Normlinf();

         C.GetGradient(x_primal).Mult(lambda, G_primal);
         obj.GetGradient().AddMult(x_primal, G_primal);

         kkt = problem.ComputeKKT(x_primal, G_primal);

         if (print_level >= 1)
         {
            out << "AL Iteration " << it_AL + 1 << " with mu = " << mu_prev << " (" << it_PG
                + 1 << " - " << it_GN_all
                << "), KKT residual = " << kkt << std::endl;
            out << "Objective value = " << obj_value[0] << ", " << "Constraint residual = "
                << C_x.Normlinf() << " (";
            for (int i=0; i<numConst; i++) { out << C_x[i] << " "; } out << "\b)" <<
                  std::endl;
         }

         // If first iteration, setup the relative tolerance.
         if (it_AL == 0) { kkt_target = std::max(kkt_target, kkt*rel_tol); }
         if ((kkt < kkt_target && C_x.Normlinf() < const_abs_tol) && it_AL > 0)
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

   void SetConstAbsTol(real_t tol) { const_abs_tol = tol; }

   void SetProxMaxIter(int n) { prox_max_iter = n; }
   void SetProxAbsTol(real_t tol) { prox_abs_tol0 = tol; }
   void SetProxRelTol(real_t tol) { prox_rel_tol0 = tol; }

   void SetNonlinMaxIter(int n) { nonlin_max_iter = n; }
   void SetNonlinAbsTol(real_t tol) { nonlin_abs_tol = tol; }
   void SetNonlinRelTol(real_t tol) { nonlin_rel_tol = tol; }
   void SetAlpha(real_t alpha) { this->alpha0 = alpha; }
   void SetPenalty(real_t mu) { this->mu0 = mu; }
protected:
   // Inner solver parameters
   real_t alpha0 = 1.0;
   real_t mu0 = 1.0;
   bool use_hessian_approx = false; // whether the constraints are linear

   real_t const_abs_tol = 1e-06;

   int prox_max_iter = 100;
   mutable real_t prox_abs_tol0 = 1e-06;
   mutable real_t prox_rel_tol0 = 1e-06;
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

   remap::RemapProblem &problem; // The LVPP problem to solve

   Array<int> offsets; // primal offsets
   Array<int> b_offsets; // bound offsets
   Array<int> total_offsets; // both primal and latent
   mutable PointwiseSolver pointwise_solver;
};

}

#endif // REMHOS_LVPP_HPP
