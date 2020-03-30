// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include "remhos_fct.hpp"
#include "remhos_tools.hpp"

using namespace std;

namespace mfem
{

void FluxBasedFCT::CalcFCTSolution(const ParGridFunction &u, const Vector &m,
                                   const Vector &du_ho, const Vector &du_lo,
                                   const Vector &u_min, const Vector &u_max,
                                   Vector &du) const
{
   MFEM_VERIFY(smth_indicator == NULL, "TODO: update SI bounds.");

   // Construct the flux matrix.
   const int s = u.Size();
   double *flux_data = flux_ij.GetData();
   const int *K_I = K.GetI(), *K_J = K.GetJ();
   const double *K_data = K.GetData();
   const double *u_np = u.FaceNbrData().GetData();
   for (int i = 0; i < s; i++)
   {
      for (int k = K_I[i]; k < K_I[i + 1]; k++)
      {
         int j = K_J[k];
         if (j <= i) { continue; }

         double kij  = K_data[k], kji = K_data[K_smap[k]];
         double dij  = max(max(0.0, -kij), -kji);
         double u_ij = (j < s) ? u(i) - u(j)
                       : u(i) - u_np[j - s];

         flux_data[k] = dt * dij * u_ij;
      }
   }
   const int NE = pfes.GetMesh()->GetNE();
   const int ndof = s / NE;
   Array<int> dofs;
   DenseMatrix Mz(ndof);
   Vector du_z(ndof);
   for (int k = 0; k < NE; k++)
   {
      pfes.GetElementDofs(k, dofs);
      M.GetSubMatrix(dofs, dofs, Mz);
      du_ho.GetSubVector(dofs, du_z);
      for (int i = 0; i < ndof; i++)
      {
         int j = 0;
         for (; j <= i; j++) { Mz(i, j) = 0.0; }
         for (; j < ndof; j++) { Mz(i, j) *= dt * (du_z(i) - du_z(j)); }
      }
      // This is the local contribution.
      flux_ij.AddSubMatrix(dofs, dofs, Mz, 0);
   }

   for (int fct_iter = 0; fct_iter < iter_cnt; fct_iter++)
   {
      // Compute sums of incoming fluxes.
      const int *flux_I = flux_ij.GetI(), *flux_J = flux_ij.GetJ();
      gp = 0.0;
      gm = 0.0;
      for (int i = 0; i < s; i++)
      {
         for (int k = flux_I[i]; k < flux_I[i + 1]; k++)
         {
            int j = flux_J[k];

            // The skipped fluxes will be added when the outer loop is at j as
            // the flux matrix is always symmetric.
            if (j <= i) { continue; }

            double f_ij = flux_data[k];

            if (f_ij >= 0.0)
            {
               gp(i) += f_ij;
               // Modify j if it's on the same MPI task (prevents x2 counting).
               if (j < s) { gm(j) -= f_ij; }
            }
            else
            {
               gm(i) += f_ij;
               // Modify j if it's on the same MPI task (prevents x2 counting).
               if (j < s) { gp(j) -= f_ij; }
            }
         }
      }

      // Compute alpha coefficients (into gp and gm).
      for (int i = 0; i < s; i++)
      {
         double u_lo = u(i) + dt * du_lo(i);
         double max_pos_diff = max((u_max(i) - u_lo) * m(i), 0.0),
                min_neg_diff = min((u_min(i) - u_lo) * m(i), 0.0);
         double sum_pos = gp(i), sum_neg = gm(i);

         gp(i) = (sum_pos > max_pos_diff) ? max_pos_diff / sum_pos : 1.0;
         gm(i) = (sum_neg < min_neg_diff) ? min_neg_diff / sum_neg : 1.0;
      }

      // Apply the alpha coefficients to get the final solution.
      Vector &a_pos_n = gp.FaceNbrData(), &a_neg_n = gm.FaceNbrData();
      gp.ExchangeFaceNbrData();
      gm.ExchangeFaceNbrData();
      du = du_lo;
      for (int i = 0; i < s; i++)
      {
         for (int k = flux_I[i]; k < flux_I[i + 1]; k++)
         {
            int j = flux_J[k];
            if (j <= i) { continue; }

            double fij = flux_data[k], a_ij;
            if (fij >= 0.0)
            {
               a_ij = (j < s) ? min(gp(i), gm(j))
                      : min(gp(i), a_neg_n(j - s));
            }
            else
            {
               a_ij = (j < s) ? min(gm(i), gp(j))
                      : min(gm(i), a_pos_n(j - s));
            }
            fij *= a_ij;

            du(i) += fij / m(i) / dt;
            if (j < s) { du(j) -= fij / m(j) / dt; }

            flux_data[k] -= fij;
         }
      }
   } // iterated FCT.
}

void ClipScaleSolver::CalcFCTSolution(const ParGridFunction &u, const Vector &m,
                                      const Vector &du_ho, const Vector &du_lo,
                                      const Vector &u_min, const Vector &u_max,
                                      Vector &du) const
{
   const int NE = pfes.GetMesh()->GetNE();
   const int nd = pfes.GetFE(0)->GetDof();
   Vector f_clip(nd);

   int dof_id;
   double sumPos, sumNeg, u_new_ho, u_new_lo, new_mass, f_clip_min, f_clip_max;
   double umin, umax;
   const double eps = 1.0e-15;

   // Smoothness indicator.
   ParGridFunction si_val;
   if (smth_indicator)
   {
      smth_indicator->ComputeSmoothnessIndicator(u, si_val);
   }

   for (int k = 0; k < NE; k++)
   {
      sumPos = sumNeg = 0.0;

      // Clip.
      for (int j = 0; j < nd; j++)
      {
         dof_id = k*nd+j;

         u_new_ho   = u(dof_id) + dt * du_ho(dof_id);
         u_new_lo   = u(dof_id) + dt * du_lo(dof_id);

         umin = u_min(dof_id);
         umax = u_max(dof_id);
         if (smth_indicator)
         {
            smth_indicator->UpdateBounds(dof_id, u_new_ho, si_val, umin, umax);
         }

         f_clip_min = m(dof_id) / dt * (umin - u_new_lo);
         f_clip_max = m(dof_id) / dt * (umax - u_new_lo);

         f_clip(j) = m(dof_id) * (du_ho(dof_id) - du_lo(dof_id));
         f_clip(j) = min(f_clip_max, max(f_clip_min, f_clip(j)));

         sumNeg += min(f_clip(j), 0.0);
         sumPos += max(f_clip(j), 0.0);
      }

      new_mass = sumNeg + sumPos;

      // Rescale.
      for (int j = 0; j < nd; j++)
      {
         if (new_mass > eps)
         {
            f_clip(j) = min(0.0, f_clip(j)) -
                        max(0.0, f_clip(j)) * sumNeg / sumPos;
         }
         if (new_mass < -eps)
         {
            f_clip(j) = max(0.0, f_clip(j)) -
                        min(0.0, f_clip(j)) * sumPos / sumNeg;
         }

         // Set du to the discrete time derivative featuring the high order
         // anti-diffusive reconstruction that leads to an forward Euler
         // updated admissible solution.
         dof_id = k*nd+j;
         du(dof_id) = du_lo(dof_id) + f_clip(j) / m(dof_id);
      }
   }
}

void NonlinearPenaltySolver::CalcFCTSolution(const ParGridFunction &u,
                                             const Vector &m,
                                             const Vector &du_ho,
                                             const Vector &du_lo,
                                             const Vector &u_min,
                                             const Vector &u_max,
                                             Vector &du) const
{
   const int size = u.Size();
   Vector du_ho_star(size);

   double umin, umax;

   // Smoothness indicator.
   ParGridFunction si_val;
   if (smth_indicator)
   {
      smth_indicator->ComputeSmoothnessIndicator(u, si_val);
   }

   // Clipped flux.
   for (int i = 0; i < size; i++)
   {
      umin = u_min(i);
      umax = u_max(i);
      if (smth_indicator)
      {
         smth_indicator->UpdateBounds(i, u(i) + dt * du_ho(i),
                                      si_val, umin, umax);
      }

      // Note that this uses u(i) at the old time.
      du_ho_star(i) = min( (umax - u(i)) / dt,
                           max(du_ho(i), (umin - u(i)) / dt) );
   }

   // Non-conservative fluxes.
   Vector fL(size), fH(size);
   for (int i = 0; i < size; i++)
   {
      fL(i) = m(i) * (du_ho_star(i) - du_lo(i));
      fH(i) = m(i) * (du_ho_star(i) - du_ho(i));
   }

   // Restore conservation.
   Vector flux_correction(size);
   CorrectFlux(fL, fH, flux_correction);

   for (int i = 0; i < size; i++)
   {
      fL(i) += flux_correction(i);
   }

   for (int i = 0; i < size; i++)
   {
      du(i) = du_lo(i) + fL(i) / m(i);
   }
}

void get_z(double lambda, const Vector &w, const Vector &flux, Vector &zz)
{
   if (lambda == 0.0)
   {
      zz = 0.0;
   }
   else
   {
      for (int j=0; j<w.Size(); j++)
      {
         if (flux(j)!=0)
         {
            zz(j) = (abs(flux(j)) >= lambda*abs(w(j))) ? w(j) : flux(j)/lambda;
         }
         else
         {
            zz(j) = 0;
         }
      }
   }
}

double get_lambda_times_sum_z(double lambda,
                              const Vector &w, const Vector &fluxL)
{
   double lambda_times_z;
   double lambda_times_sum_z = 0.0;
   for (int j=0; j<w.Size(); j++)
   {
      // In the second case, lambda * w(j) is set to fluxL(j).
      lambda_times_z = (abs(fluxL(j)) >= lambda*abs(w(j)))
                       ? lambda * w(j) : fluxL(j);

      lambda_times_sum_z += lambda_times_z;
   }
   return lambda_times_sum_z;
}

double get_lambda(double delta, const Vector &w,
                  const Vector &fluxL, Vector &zz)
{
   // solve nonlinearity F(lambda)=0
   double tol=1e-15;
   double lambdaLower=0., lambdaUpper = 0.;
   double FLower=0., FUpper=0.;
   double factor=1.;

   // compute starting F
   double lambda = 1.0;
   double F = delta - get_lambda_times_sum_z(lambda, w, fluxL);
   // check F at the extrema of lambda (0 and 1).
   double F0 = delta-get_lambda_times_sum_z(0.0, w, fluxL);
   double F1 = delta-get_lambda_times_sum_z(1.0, w, fluxL);
   if (abs(F) <= tol)
   {
      get_z(lambda, w, fluxL, zz);
      return lambda;
   }
   else if (abs(F0)<=tol)
   {
      get_z(0.0, w, fluxL,zz);
      return 0.;
   }
   else if (abs(F1)<=tol)
   {
      get_z(1,w,fluxL,zz);
      return 1.;
   }

   // solve non-linearity
   do
   {
      factor*=2;
      // look for other lambda to have opposite sign in F
      lambdaLower = lambda/factor;
      lambdaUpper = factor*lambda;
      FLower=delta-get_lambda_times_sum_z(lambdaLower,w,fluxL);
      FUpper=delta-get_lambda_times_sum_z(lambdaUpper,w,fluxL);
   }
   while ((F*FLower > 0) && (F*FUpper > 0));

   // check if either of lambdaLower or lambdaUpper hit the solution
   if (FLower==0)
   {
      get_z(lambdaLower, w, fluxL, zz);
      return lambdaLower;
   }
   else if (FUpper==0)
   {
      get_z(lambdaUpper, w, fluxL, zz);
      return lambdaUpper;
   }

   // get STARTING lower and upper bounds for lambda
   if (F*FLower < 0) // F>0
   {
      lambdaUpper = lambda;
   }
   else // F<0
   {
      lambdaLower = lambda;
   }

   // get STARTING lower and upper bounds on F
   FLower = delta - get_lambda_times_sum_z(lambdaLower,w,fluxL);
   FUpper = delta - get_lambda_times_sum_z(lambdaUpper,w,fluxL);

   do
   {
      // compute new lambda and new F
      lambda = 0.5*(lambdaLower+lambdaUpper);
      F = delta - get_lambda_times_sum_z(lambda,w,fluxL);
      if (F*FLower < 0) // F >= 0
      {
         lambdaUpper = lambda;
         FUpper = F;
      }
      else // F <= 0
      {
         lambdaLower = lambda;
         FLower = F;
      }
   }
   while (abs(F)>tol);

   lambda = 0.5*(lambdaLower+lambdaUpper);
   get_z(lambda, w, fluxL, zz);
   return lambda;
}

void NonlinearPenaltySolver::CorrectFlux(Vector &fluxL, Vector &fluxH,
                                         Vector &flux_fix) const
{
   // This consider any definition of wi. If a violation on MPP is created,
   // then wi is s.t. fi=0.
   // The idea is to relax the penalization wi in favor of MPP
   const int num_cells = pfes.GetNE();
   const int xd = pfes.GetFE(0)->GetDof();

   Array<int> ldofs;
   Vector fluxL_z(xd), fluxH_z(xd), flux_correction_z(xd);
   for (int i = 0; i < num_cells; i++)
   {
      pfes.GetElementDofs(i, ldofs);
      fluxL.GetSubVector(ldofs, fluxL_z);
      fluxH.GetSubVector(ldofs, fluxH_z);

      double fp = 0.0, fn = 0.0;
      for (int j = 0; j < xd; j++)
      {
         if (fluxL_z(j) >= 0.0) { fp += fluxL_z(j); }
         else                   { fn += fluxL_z(j); }
      }

      double delta = fp + fn;

      if (delta == 0.0)
      {
         flux_correction_z = 0.0;
         flux_fix.SetSubVector(ldofs, flux_correction_z);
         continue;
      }

      // compute penalization terms wi's as desired
      Vector w(xd);
      const double eps = pfes.GetMesh()->GetElementSize(0,0) / pfes.GetOrder(0);
      for (int j = 0; j < xd; j++)
      {
         if (delta > 0.0)
         {
            w(j) = (fluxL_z(j) > 0.0)
                   ? eps * abs(fluxL_z(j)) + abs(get_max_on_cellNi(fluxH_z))
                   : 0.0;
         }
         else
         {
            w(j) = (fluxL_z(j) < 0.0)
                   ? - eps * abs(fluxL_z(j)) - abs(get_max_on_cellNi(fluxH_z))
                   : 0.0;
         }
      }

      // compute lambda
      Vector zz(xd);
      double lambda = get_lambda(delta, w, fluxL_z, zz);

      // compute flux correction
      for (int j = 0; j < xd; j++)
      {
         flux_correction_z(j) = -lambda * zz(j);
      }

      flux_fix.SetSubVector(ldofs, flux_correction_z);
   }
}

double NonlinearPenaltySolver::get_max_on_cellNi(Vector &fluxH) const
{
   double MAX = -1.0;
   for (int i = 0; i < fluxH.Size(); i++)
   {
      MAX = max(fabs(fluxH(i)), MAX);
   }
   return MAX;
}

} // namespace mfem
