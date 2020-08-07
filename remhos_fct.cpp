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
#include "remhos_sync.hpp"

using namespace std;

namespace mfem
{

void FluxBasedFCT::CalcFCTSolution(const ParGridFunction &u, const Vector &m,
                                   const Vector &du_ho, const Vector &du_lo,
                                   const Vector &u_min, const Vector &u_max,
                                   Vector &du) const
{
   MFEM_VERIFY(smth_indicator == NULL, "TODO: update SI bounds.");

   // Construct the flux matrix (it gets recomputed every time).
   ComputeFluxMatrix(u, du_ho, flux_ij);

   // Iterated FCT correction.
   Vector du_lo_fct(du_lo);
   for (int fct_iter = 0; fct_iter < iter_cnt; fct_iter++)
   {
      // Compute sums of incoming/outgoing fluxes at each DOF.
      AddFluxesAtDofs(flux_ij, gp, gm);

      // Compute the flux coefficients (aka alphas) into gp and gm.
      ComputeFluxCoefficients(u, du_lo_fct, m, u_min, u_max, gp, gm);

      // Apply the alpha coefficients to get the final solution.
      // Update the flux matrix for iterative FCT (when iter_cnt > 1).
      UpdateSolutionAndFlux(du_lo_fct, m, gp, gm, flux_ij, du);

      du_lo_fct = du;
   }
}

void PrintCellVals(int cell_id, int NE, const Vector &vec, const char *msg)
{
   std::cout << msg << std::endl;
   const int ndofs = vec.Size() / NE;
   for (int i = 0; i < ndofs; i++)
   {
      std::cout << vec(cell_id * ndofs + i) << " ";
   }
   std::cout << endl;
}

void FluxBasedFCT::CalcFCTProduct(const ParGridFunction &us, const Vector &m,
                                  const Vector &dus_ho, const Vector &dus_lo,
                                  Vector &s_min, Vector &s_max, const Vector &u,
                                  const Vector &u_new, const Vector &u_new_LO,
                                  const Array<bool> &active_el, Vector &dus)
{
   // Construct the flux matrix (it gets recomputed every time).
   ComputeFluxMatrix(us, dus_ho, flux_ij);

   const double eps = 1e-6;
   int dof_id;

   // Update the flux matrix to a product-compatible version.
   // Compute a compatible low-order solutions.
   const int NE = us.ParFESpace()->GetNE();
   const int ndofs = us.Size() / NE;
   Vector us_new_LO_loc(ndofs), flux_loc(ndofs), beta(ndofs), dus_lo_fct(us.Size());
   Vector us_min(us.Size()), us_max(us.Size());
   DenseMatrix fij_loc(ndofs);
   fij_loc = 0.0;

   dus_lo_fct = 0.0;

   // TODO organize these arrays better.
   Array<bool> whatever, active_dofs;
   ComputeBoolIndicators(NE, u_new, whatever, active_dofs);

   for (int k = 0; k < NE; k++)
   {
      if (active_el[k] == false) { continue; }

      Array<int> dofs;
      us.ParFESpace()->GetElementDofs(k, dofs);

      double mass_us = 0.0, mass_u = 0.0, mass_u_LO = 0.0;
      for (int j = 0; j < ndofs; j++)
      {
         us_new_LO_loc(j) = us(k*ndofs + j) + dt * dus_lo(k*ndofs + j);
         mass_us += us_new_LO_loc(j) * m(k*ndofs + j);
         mass_u  += u_new(k*ndofs + j) * m(k*ndofs + j);
         mass_u_LO += u_new_LO(k*ndofs + j) * m(k*ndofs + j);
      }
      const double s_avg = mass_us / mass_u;

      Vector min_loc, max_loc, u_new_loc, u_new_LO_loc, us_old_loc, dus_lo_loc, u_old_loc;
      s_min.GetSubVector(dofs, min_loc);
      s_max.GetSubVector(dofs, max_loc);
      u.GetSubVector(dofs, u_old_loc);
      u_new.GetSubVector(dofs, u_new_loc);
      u_new_LO.GetSubVector(dofs, u_new_LO_loc);
      us.GetSubVector(dofs, us_old_loc);
      dus_lo.GetSubVector(dofs, dus_lo_loc);
      double minv = min_loc.Min(), maxv = max_loc.Max();

      // When s_avg is not in bounds for some dof (it should be within the
      // global max and min for the element), reset the bounds to s_avg.
      for (int j = 0; j < ndofs; j++)
      {
         dof_id = k*ndofs + j;
         if (active_dofs[dof_id] == false) { continue; }

         double s_LO = us_new_LO_loc(j) / u_new_LO_loc(j);
         if (s_LO + eps < minv ||
             s_LO - eps > maxv)
         {
            std::cout << "Cell " << k << std::endl;
            std::cout << "At " << j << " out of " << ndofs << std::endl;
            std::cout << "Basic theorem " << minv << " " << s_LO << " " << maxv << std::endl;
            std::cout << us_new_LO_loc(j) << " " << u_new_LO_loc(j) << std::endl;

            std::cout << "us_old_loc: " << std::endl;
            us_old_loc.Print();

            std::cout << "u_old_loc: " << std::endl;
            u_old_loc.Print();

            std::cout << "us_new_loc_LO: " << std::endl;
            us_new_LO_loc.Print();

            std::cout << "u_new_loc_LO: " << std::endl;
            u_new_LO_loc.Print();

            std::cout << "u_new_loc: " << std::endl;
            u_new_loc.Print();

            MFEM_ABORT("s_LO not in element bounds");
         }

         // Check if s_avg is within the min/max for the cell.
         if (s_avg + eps < minv ||
             s_avg - eps > maxv)
         {
            std::cout << "---\ns_avg element bounds: "
                      << minv << " " << s_avg << " " << maxv << std::endl;
            std::cout << "Element " << k << std::endl;
            std::cout << "Masses " << mass_us << " " << mass_u << " " << mass_u_LO << std::endl;
            std::cout << "u_old_loc: " << std::endl;
            us_old_loc.Print();
            std::cout << "u_loc: " << std::endl;
            u_new_loc.Print();
            std::cout << "u_loc_LO: " << std::endl;
            u_new_LO_loc.Print();
            std::cout << "us_loc_LO: " << std::endl;
            us_new_LO_loc.Print();
            MFEM_ABORT("s_avg not in element bounds");
         }

         if (s_avg + eps < s_min(dof_id) ||
             s_avg - eps > s_max(dof_id))
         {
            s_min(dof_id) = s_avg;
            s_max(dof_id) = s_avg;
         }
      }

      // Take into account the compatible low-order solution.
      for (int j = 0; j < ndofs; j++)
      {
         // In inactive dofs we get zeros for u*s, which should be fine.

         dof_id = k*ndofs + j;
         double d_us_lo_j = (u_new(dof_id) * s_avg - us(dof_id)) / dt;
         flux_loc(j) = m(dof_id) * dt * (dus_lo(dof_id) - d_us_lo_j);
         // Change the LO solution.
         dus_lo_fct(dof_id) = d_us_lo_j;

         beta(j) = m(dof_id) * u_new(dof_id);
      }

      // Make the betas sum to 1, add the new compatible fluxes.
      beta /= beta.Sum();
      for (int j = 1; j < ndofs; j++)
      {
         for (int i = 0; i < j; i++)
         {
            fij_loc(i, j) = beta(j) * flux_loc(i) - beta(i) * flux_loc(j);
         }
      }
      flux_ij.AddSubMatrix(dofs, dofs, fij_loc);

      // Rescale the bounds (s_min, s_max) -> (u*s_min, u*s_max).
      for (int j = 0; j < ndofs; j++)
      {
         dof_id = k*ndofs + j;

         // For inactive dofs, s_min and s_max are undefined (inf values).
         if (active_dofs[dof_id] == false)
         {
            us_min(dof_id) = 0.0;
            us_max(dof_id) = 0.0;
            continue;
         }

         us_min(dof_id) = s_min(dof_id) * u_new(dof_id);
         us_max(dof_id) = s_max(dof_id) * u_new(dof_id);
      }

      //
      // Check product.
      //
      us_min.GetSubVector(dofs, min_loc);
      us_max.GetSubVector(dofs, max_loc);
      minv = min_loc.Min(); maxv = max_loc.Max();
      for (int j = 0; j < ndofs; j++)
      {         
         dof_id = k*ndofs + j;
         if (active_dofs[dof_id] == false) { continue; }

         if (s_avg * u_new(k*ndofs + j) + eps < us_min(dof_id) ||
             s_avg * u_new(k*ndofs + j) - eps > us_max(dof_id))
         {
            std::cout << "---\ns_avg * u: " << k << " "
                      << us_min(dof_id) << " "
                      << s_avg * u_new(k*ndofs + j) << " "
                      << us_max(dof_id) << std::endl;
            std::cout << u_new(k*ndofs + j) << " " << s_avg << endl;
            std::cout << s_min(dof_id) << " " << s_max(dof_id) << "\n---\n";
            MFEM_ABORT("s_avg * u NOT IN BOUNDS WTF MAN");
         }
      }
   }

   dus = dus_lo_fct;
   // Check the bounds.
   Vector us_new(dus.Size());
   add(1.0, us, dt, dus, us_new);
   for (int k = 0; k < NE; k++)
   {
      if (active_el[k] == false) { continue; }

      for (int j = 0; j < ndofs; j++)
      {
         dof_id = k*ndofs + j;
         if (active_dofs[dof_id] == false) { continue; }

         if (us_new(k*ndofs + j) + eps < us_min(k*ndofs + j) ||
             us_new(k*ndofs + j) - eps > us_max(k*ndofs + j))
         {
            std::cout << "LO " << j << " " << k << " "
                      << us_min(k*ndofs + j) << " "
                      << us_new(k*ndofs + j) << " "
                      << us_max(k*ndofs + j) << std::endl;
            std::cout << "---\n";
         }
      }
   }

   ZeroOutEmptyDofs(active_el, active_dofs, dus);

   /*
   PrintCellVals(18, NE, u, "Cell 18 after update, u_old.");
   PrintCellVals(18, NE, u_new, "Cell 18 after update, u_new.");
   PrintCellVals(18, NE, us, "Cell 18 after update, u_s_old.");
   PrintCellVals(18, NE, us_new, "Cell 18 after update, u_s_new.");
   PrintCellVals(50, NE, u, "Cell 50 after update, u_old.");
   PrintCellVals(50, NE, u_new, "Cell 50 after update, u_new.");
   PrintCellVals(50, NE, us, "Cell 50 after update, u_s_old.");
   PrintCellVals(50, NE, us_new, "Cell 50 after update, u_s."); */


   /*
   // Iterated FCT correction.
   for (int fct_iter = 0; fct_iter < iter_cnt; fct_iter++)
   {
      // Compute sums of incoming fluxes at each DOF.
      AddFluxesAtDofs(flux_ij, gp, gm);

      // Compute the flux coefficients (aka alphas) into gp and gm.
      ComputeFluxCoefficients(us, dus_lo_fct, m, us_min, us_max, gp, gm);

      // Apply the alpha coefficients to get the final solution.
      // Update the fluxes for iterative FCT (when iter_cnt > 1).
      UpdateSolutionAndFlux(dus_lo_fct, m, gp, gm, flux_ij, dus);

      ZeroOutEmptyDofs(active_el, active_dofs, dus);

      dus_lo_fct = dus;
   }
   */

#ifdef REMHOS_FCT_DEBUG
   // Check the bounds.
   add(1.0, us, dt, dus, us_new);
   for (int k = 0; k < NE; k++)
   {
      if (active_el[k] == false) { continue; }

      for (int j = 0; j < ndofs; j++)
      {
         dof_id = k*ndofs + j;
         if (active_dofs[dof_id] == false) { continue; }

         if (us_new(k*ndofs + j) + eps < us_min(k*ndofs + j) ||
             us_new(k*ndofs + j) - eps > us_max(k*ndofs + j))
         {
            std::cout << "Final us " << j << " " << k << " "
                      << us_min(k*ndofs + j) << " "
                      << us_new(k*ndofs + j) << " "
                      << us_max(k*ndofs + j) << std::endl;
            std::cout << "---\n";
         }
      }
   }
#endif
}

void FluxBasedFCT::ComputeFluxMatrix(const ParGridFunction &u,
                                     const Vector &du_ho,
                                     SparseMatrix &flux_mat) const
{
   const int s = u.Size();
   double *flux_data = flux_mat.GetData();
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
      flux_mat.AddSubMatrix(dofs, dofs, Mz, 0);
   }
}

// Compute sums of incoming fluxes for every DOF.
void FluxBasedFCT::AddFluxesAtDofs(const SparseMatrix &flux_mat,
                                   Vector &flux_pos, Vector &flux_neg) const
{
   const int s = flux_pos.Size();
   const double *flux_data = flux_mat.GetData();
   const int *flux_I = flux_mat.GetI(), *flux_J = flux_mat.GetJ();
   flux_pos = 0.0;
   flux_neg = 0.0;
   for (int i = 0; i < s; i++)
   {
      for (int k = flux_I[i]; k < flux_I[i + 1]; k++)
      {
         int j = flux_J[k];

         // The skipped fluxes will be added when the outer loop is at j as
         // the flux matrix is always symmetric.
         if (j <= i) { continue; }

         const double f_ij = flux_data[k];

         if (f_ij >= 0.0)
         {
            flux_pos(i) += f_ij;
            // Modify j if it's on the same MPI task (prevents x2 counting).
            if (j < s) { flux_neg(j) -= f_ij; }
         }
         else
         {
            flux_neg(i) += f_ij;
            // Modify j if it's on the same MPI task (prevents x2 counting).
            if (j < s) { flux_pos(j) -= f_ij; }
         }
      }
   }
}

// Compute the so-called alpha coefficients that scale the fluxes into gp, gm.
void FluxBasedFCT::
ComputeFluxCoefficients(const Vector &u, const Vector &du_lo, const Vector &m,
                        const Vector &u_min, const Vector &u_max,
                        Vector &coeff_pos, Vector &coeff_neg) const
{
   const int s = u.Size();
   for (int i = 0; i < s; i++)
   {
      const double u_lo = u(i) + dt * du_lo(i);
      const double max_pos_diff = max((u_max(i) - u_lo) * m(i), 0.0),
                   min_neg_diff = min((u_min(i) - u_lo) * m(i), 0.0);
      const double sum_pos = coeff_pos(i), sum_neg = coeff_neg(i);

      coeff_pos(i) = (sum_pos > max_pos_diff) ? max_pos_diff / sum_pos : 1.0;
      coeff_neg(i) = (sum_neg < min_neg_diff) ? min_neg_diff / sum_neg : 1.0;
   }
}

void FluxBasedFCT::
UpdateSolutionAndFlux(const Vector &du_lo, const Vector &m,
                      ParGridFunction &coeff_pos, ParGridFunction &coeff_neg,
                      SparseMatrix &flux_mat, Vector &du) const
{
   Vector &a_pos_n = coeff_pos.FaceNbrData(),
          &a_neg_n = coeff_neg.FaceNbrData();
   coeff_pos.ExchangeFaceNbrData();
   coeff_neg.ExchangeFaceNbrData();
   du = du_lo;
   double *flux_data = flux_mat.GetData();
   const int *flux_I = flux_mat.GetI(), *flux_J = flux_mat.GetJ();
   const int s = du.Size();
   for (int i = 0; i < s; i++)
   {
      for (int k = flux_I[i]; k < flux_I[i + 1]; k++)
      {
         int j = flux_J[k];
         if (j <= i) { continue; }

         double fij = flux_data[k], a_ij;
         if (fij >= 0.0)
         {
            a_ij = (j < s) ? min(coeff_pos(i), coeff_neg(j))
                   : min(coeff_pos(i), a_neg_n(j - s));
         }
         else
         {
            a_ij = (j < s) ? min(coeff_neg(i), coeff_pos(j))
                   : min(coeff_neg(i), a_pos_n(j - s));
         }
         fij *= a_ij;

         du(i) += fij / m(i) / dt;
         if (j < s) { du(j) -= fij / m(j) / dt; }

         flux_data[k] -= fij;
      }
   }
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
   for (int j=0; j<w.Size(); j++)
   {
      zz(j) = (abs(flux(j)) >= lambda*abs(w(j))) ? lambda * w(j) : flux(j);
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

void get_lambda(double delta, const Vector &w,
                const Vector &fluxL, Vector &zz)
{
   // solve nonlinearity F(lambda)=0
   double tol=1e-15;
   double lambdaLower=0., lambdaUpper = 0.;
   double FLower=0., FUpper=0.;

   // compute starting F and check F at the min (lambda = 0).
   double lambda = 1.0;
   double F = delta - get_lambda_times_sum_z(lambda, w, fluxL);
   double F0 = delta-get_lambda_times_sum_z(0.0, w, fluxL);
   if (abs(F) <= tol)
   {
      get_z(lambda, w, fluxL, zz);
   }
   else if (abs(F0)<=tol)
   {
      get_z(0.0, w, fluxL, zz);
   }

   // solve non-linearity
   // Get lambda values that give Fs on both sides of the zero.
   double factor = 1.0;
   do
   {
      factor *= 2.0;
      lambdaLower = lambda/factor;
      lambdaUpper = factor*lambda;
      FLower = delta - get_lambda_times_sum_z(lambdaLower,w,fluxL);
      FUpper = delta - get_lambda_times_sum_z(lambdaUpper,w,fluxL);
   }
   while (F*FLower > 0 && F*FUpper > 0);

   // check if either of lambdaLower or lambdaUpper hit the solution
   if (FLower==0.0)
   {
      get_z(lambdaLower, w, fluxL, zz);
   }
   else if (FUpper==0.0)
   {
      get_z(lambdaUpper, w, fluxL, zz);
   }

   // get STARTING lower and upper bounds for lambda
   if (F*FLower < 0)
   {
      lambdaUpper = lambda;
   }
   else
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
      if (F*FLower < 0)
      {
         lambdaUpper = lambda;
         FUpper = F;
      }
      else
      {
         lambdaLower = lambda;
         FLower = F;
      }
   }
   while (abs(F)>tol);

   lambda = 0.5*(lambdaLower+lambdaUpper);
   get_z(lambda, w, fluxL, zz);
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

      // compute lambda and the flux correction.
      get_lambda(delta, w, fluxL_z, flux_correction_z);
      flux_correction_z.Neg();

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
