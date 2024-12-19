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

void FCTSolver::CalcCompatibleLOProduct(const ParGridFunction &us,
                                        const Vector &m, const Vector &d_us_HO,
                                        Vector &s_min, Vector &s_max,
                                        const Vector &u_new,
                                        const Array<bool> &active_el,
                                        const Array<bool> &active_dofs,
                                        Vector &d_us_LO_new)
{
   const double eps = 1e-12;
   int dof_id;

   // Compute a compatible low-order solution.
   const int NE = us.ParFESpace()->GetNE();
   const int ndofs = us.Size() / NE;

   Vector s_min_loc, s_max_loc;

   d_us_LO_new = 0.0;

   for (int k = 0; k < NE; k++)
   {
      if (active_el[k] == false) { continue; }

      double mass_us = 0.0, mass_u = 0.0;
      for (int j = 0; j < ndofs; j++)
      {
         const double us_new_HO = us(k*ndofs + j) + dt * d_us_HO(k*ndofs + j);
         mass_us += us_new_HO * m(k*ndofs + j);
         mass_u  += u_new(k*ndofs + j) * m(k*ndofs + j);
      }
      double s_avg = mass_us / mass_u;

      // Min and max of s using the full stencil of active dofs.
      s_min_loc.SetDataAndSize(s_min.GetData() + k*ndofs, ndofs);
      s_max_loc.SetDataAndSize(s_max.GetData() + k*ndofs, ndofs);
      double smin = numeric_limits<double>::infinity(),
             smax = -numeric_limits<double>::infinity();
      for (int j = 0; j < ndofs; j++)
      {
         if (active_dofs[k*ndofs + j] == false) { continue; }
         smin = min(smin, s_min_loc(j));
         smax = max(smax, s_max_loc(j));
      }

      // Fix inconsistencies due to round-off and the usage of local bounds.
      for (int j = 0; j < ndofs; j++)
      {
         if (active_dofs[k*ndofs + j] == false) { continue; }

         // Check if there's a violation, s_avg < s_min, due to round-offs that
         // are inflated by the division of a small number (the 2nd check means
         // s_avg = mass_us / mass_u > s_min up to round-off in mass_us).
         if (s_avg < smin &&
             mass_us + eps > smin * mass_u) { s_avg = smin; }
         // As above for the s_max.
         if (s_avg > smax &&
             mass_us - eps < smax * mass_u) { s_avg = smax; }

         if (verify_bounds)
         {
            // Check if s_avg = mass_us / mass_u is within the bounds of the
            // full stencil of active dofs.
            if (mass_us + eps < smin * mass_u ||
                mass_us - eps > smax * mass_u ||
                s_avg + eps < smin ||
                s_avg - eps > smax)
            {
               std::cout << "---\ns_avg element bounds: "
                         << smin << " " << s_avg << " " << smax << std::endl;
               std::cout << "Element " << k << std::endl;
               std::cout << "Masses " << mass_us << " " << mass_u << std::endl;
               PrintCellValues(k, NE, u_new, "u_loc: ");

               MFEM_ABORT("s_avg is not in the full stencil bounds!");
            }
         }

         // When s_avg is not in the local bounds for some dof (it should be
         // within the full stencil of active dofs), reset the bounds to s_avg.
         if (s_avg + eps < s_min_loc(j)) { s_min_loc(j) = s_avg; }
         if (s_avg - eps > s_max_loc(j)) { s_max_loc(j) = s_avg; }
      }

      // Take into account the compatible low-order solution.
      for (int j = 0; j < ndofs; j++)
      {
         // In inactive dofs we get u_new*s_avg ~ 0, which should be fine.

         // Compatible LO solution.
         dof_id = k*ndofs + j;
         d_us_LO_new(dof_id) = (u_new(dof_id) * s_avg - us(dof_id)) / dt;
      }
   }
}

void FCTSolver::ScaleProductBounds(const Vector &s_min, const Vector &s_max,
                                   const Vector &u_new,
                                   const Array<bool> &active_el,
                                   const Array<bool> &active_dofs,
                                   Vector &us_min, Vector &us_max)
{
   const int NE = pfes.GetNE();
   const int ndofs = u_new.Size() / NE;
   int dof_id;
   us_min = 0.0;
   us_max = 0.0;
   for (int k = 0; k < NE; k++)
   {
      if (active_el[k] == false) { continue; }

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
   }
}

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

void FluxBasedFCT::CalcFCTProduct(const ParGridFunction &us, const Vector &m,
                                  const Vector &d_us_HO, const Vector &d_us_LO,
                                  Vector &s_min, Vector &s_max,
                                  const Vector &u_new,
                                  const Array<bool> &active_el,
                                  const Array<bool> &active_dofs, Vector &d_us)
{
   // Construct the flux matrix (it gets recomputed every time).
   ComputeFluxMatrix(us, d_us_HO, flux_ij);

   us.HostRead();
   d_us_LO.HostRead();
   s_min.HostReadWrite();
   s_max.HostReadWrite();
   u_new.HostRead();
   active_el.HostRead();
   active_dofs.HostRead();

   // Compute a compatible low-order solution.
   Vector dus_lo_fct(us.Size()), us_min(us.Size()), us_max(us.Size());
   CalcCompatibleLOProduct(us, m, d_us_HO, s_min, s_max, u_new,
                           active_el, active_dofs, dus_lo_fct);
   ScaleProductBounds(s_min, s_max, u_new, active_el, active_dofs,
                      us_min, us_max);

   // Update the flux matrix to a product-compatible version.
   // Compute a compatible low-order solution.
   const int NE = us.ParFESpace()->GetNE();
   const int ndofs = us.Size() / NE;
   Vector flux_el(ndofs), beta(ndofs);
   DenseMatrix fij_el(ndofs);
   fij_el = 0.0;
   Array<int> dofs;
   int dof_id;
   for (int k = 0; k < NE; k++)
   {
      if (active_el[k] == false) { continue; }

      // Take into account the compatible low-order solution.
      for (int j = 0; j < ndofs; j++)
      {
         // In inactive dofs we get u_new*s_avg ~ 0, which should be fine.

         dof_id = k*ndofs + j;
         flux_el(j) = m(dof_id) * dt * (d_us_LO(dof_id) - dus_lo_fct(dof_id));
         beta(j) = m(dof_id) * u_new(dof_id);
      }

      // Make the betas sum to 1, add the new compatible fluxes.
      beta /= beta.Sum();
      for (int j = 1; j < ndofs; j++)
      {
         for (int i = 0; i < j; i++)
         {
            fij_el(i, j) = beta(j) * flux_el(i) - beta(i) * flux_el(j);
         }
      }
      pfes.GetElementDofs(k, dofs);
      flux_ij.AddSubMatrix(dofs, dofs, fij_el);
   }

   // Iterated FCT correction.
   // To get the LO compatible product solution (with s_avg), just do
   // d_us = dus_lo_fct instead of the loop below.
   for (int fct_iter = 0; fct_iter < iter_cnt; fct_iter++)
   {
      // Compute sums of incoming fluxes at each DOF.
      AddFluxesAtDofs(flux_ij, gp, gm);

      // Compute the flux coefficients (aka alphas) into gp and gm.
      ComputeFluxCoefficients(us, dus_lo_fct, m, us_min, us_max, gp, gm);

      // Apply the alpha coefficients to get the final solution.
      // Update the fluxes for iterative FCT (when iter_cnt > 1).
      UpdateSolutionAndFlux(dus_lo_fct, m, gp, gm, flux_ij, d_us);

      ZeroOutEmptyDofs(active_el, active_dofs, d_us);

      dus_lo_fct = d_us;
   }

   if (verify_bounds)
   {
      // Check the bounds of the final solution.
      const double eps = 1e-12;
      Vector us_new(d_us.Size());
      add(1.0, us, dt, d_us, us_new);
      for (int k = 0; k < NE; k++)
      {
         if (active_el[k] == false) { continue; }

         for (int j = 0; j < ndofs; j++)
         {
            dof_id = k*ndofs + j;
            if (active_dofs[dof_id] == false) { continue; }

            if (us_new(dof_id) + eps < us_min(dof_id) ||
                us_new(dof_id) - eps > us_max(dof_id))
            {
               std::cout << "Final us " << j << " " << k << " "
                         << us_min(dof_id) << " "
                         << us_new(dof_id) << " "
                         << us_max(dof_id) << std::endl;
               std::cout << "---\n";

               MFEM_ABORT("us not in bounds after FCT.");
            }
         }
      }
   }
}

void FluxBasedFCT::ComputeFluxMatrix(const ParGridFunction &u,
                                     const Vector &du_ho,
                                     SparseMatrix &flux_mat) const
{
   const int s = u.Size();
   double *flux_data = flux_mat.HostReadWriteData();
   flux_mat.HostReadI(); flux_mat.HostReadJ();
   const int *K_I = K.HostReadI(), *K_J = K.HostReadJ();
   const double *K_data = K.HostReadData();
   const double *u_np = u.FaceNbrData().HostRead();
   u.HostRead();
   du_ho.HostRead();
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
   flux_pos.HostReadWrite();
   flux_neg.HostReadWrite();
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
   Vector &a_pos_n = coeff_pos.FaceNbrData();
   Vector &a_neg_n = coeff_neg.FaceNbrData();
   coeff_pos.ExchangeFaceNbrData();
   coeff_neg.ExchangeFaceNbrData();

   du = du_lo;

   coeff_pos.HostReadWrite();
   coeff_neg.HostReadWrite();
   du.HostReadWrite();

   double *flux_data = flux_mat.HostReadWriteData();
   const int *flux_I = flux_mat.HostReadI(), *flux_J = flux_mat.HostReadJ();
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

   u.HostRead();
   m.HostRead();
   du.HostReadWrite();
   du_lo.HostRead(); du_ho.HostRead();
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

void ClipScaleSolver::CalcFCTProduct(const ParGridFunction &us, const Vector &m,
                                     const Vector &d_us_HO, const Vector &d_us_LO,
                                     Vector &s_min, Vector &s_max,
                                     const Vector &u_new,
                                     const Array<bool> &active_el,
                                     const Array<bool> &active_dofs, Vector &d_us)
{
   us.HostRead();
   s_min.HostReadWrite();
   s_max.HostReadWrite();
   u_new.HostRead();
   active_el.HostRead();
   active_dofs.HostRead();

   // Compute a compatible low-order solution.
   Vector dus_lo_fct(us.Size()), us_min(us.Size()), us_max(us.Size());
   CalcCompatibleLOProduct(us, m, d_us_HO, s_min, s_max, u_new,
                           active_el, active_dofs, dus_lo_fct);
   ScaleProductBounds(s_min, s_max, u_new, active_el, active_dofs,
                      us_min, us_max);

   // ClipScale solve for d_us.
   CalcFCTSolution(us, m, d_us_HO, dus_lo_fct, us_min, us_max, d_us);
   ZeroOutEmptyDofs(active_el, active_dofs, d_us);

   if (verify_bounds)
   {
      // Check the bounds of the final solution.
      const int NE = pfes.GetNE();
      const int ndofs = u_new.Size() / NE;
      int dof_id;
      const double eps = 1e-12;
      Vector us_new(d_us.Size());
      add(1.0, us, dt, d_us, us_new);
      for (int k = 0; k < NE; k++)
      {
         if (active_el[k] == false) { continue; }

         for (int j = 0; j < ndofs; j++)
         {
            dof_id = k*ndofs + j;
            if (active_dofs[dof_id] == false) { continue; }

            /* // this doesn't check round-offs in the division.
            double s = us_new(dof_id) / u_new(dof_id);
            if (s + eps < s_min(dof_id) ||
                s - eps > s_max(dof_id))
            {
               std::cout << "Final s " << j << " " << k << " "
                         << s_min(dof_id) << " "
                         << s << " "
                         << s_max(dof_id) << std::endl;
               std::cout << "---\n";
            }*/

            if (us_new(dof_id) + eps < us_min(dof_id) ||
                us_new(dof_id) - eps > us_max(dof_id))
            {
               std::cout << "Final us " << j << " " << k << " "
                         << us_min(dof_id) << " "
                         << us_new(dof_id) << " "
                         << us_max(dof_id) << std::endl;
               std::cout << "---\n";
               MFEM_ABORT("Bounds violation FCT us.");
            }
         }
      }
   }
}

void ElementFCTProjection::CalcFCTSolution(const ParGridFunction &u,
                                           const Vector &m,
                                           const Vector &du_HO,
                                           const Vector &du_LO,
                                           const Vector &u_min,
                                           const Vector &u_max,
                                           Vector &du) const
{
   const int NE = pfes.GetMesh()->GetNE();
   const int s  = pfes.GetFE(0)->GetDof();
   int dof_id;

   DenseMatrix M(s);
   Vector ML(s), rhs(s), beta(s), z(s), u_loc, du_HO_loc, du_LO_loc, du_loc,
          du_max_loc(s), du_min_loc(s);
   MassIntegrator mass_integ;

   for (int k = 0; k < NE; k++)
   {
      u_loc.SetDataAndSize(u.GetData() + k*s, s);
      du_HO_loc.SetDataAndSize(du_HO.GetData() + k*s, s);
      du_LO_loc.SetDataAndSize(du_LO.GetData() + k*s, s);
      du_loc.SetDataAndSize(du.GetData() + k*s, s);

      // Local max/min increments.
      for (int i = 0; i < s; i++)
      {
         dof_id = k*s + i;
         du_max_loc(i) = (u_max(dof_id) - u(dof_id)) / dt; // positive
         du_min_loc(i) = (u_min(dof_id) - u(dof_id)) / dt; // negative
      }

      // Construct the local mass matrix.
      ElementTransformation *T = pfes.GetMesh()->GetElementTransformation(k);
      const FiniteElement *el = pfes.GetFE(k);
      mass_integ.AssembleElementMatrix(*el, *T, M);

      M.Mult(du_HO_loc, rhs);
      M.GetRowSums(ML);

      for (int i = 0; i < s; i++)
      {
         // Some different options for beta:
         //beta(i) = 1.0;
         beta(i) = ML(i);
         //beta(i) = Mxy(i);

         // The low order flux correction
         z(i) = rhs(i) - ML(i) * du_LO_loc(i);
      }

      // Make beta_i sum to 1.
      beta /= beta.Sum();

      DenseMatrix F(s);
      for (int i = 1; i < s; i++)
      {
         for (int j = 0; j < i; j++)
         {
            F(i, j) = M(i, j) * (du_HO_loc(i) - du_HO_loc(j)) +
                      (beta(j) * z(i) - beta(i) * z(j));
         }
      }

      Vector gp(s), gm(s);
      gp = 0.0;
      gm = 0.0;
      for (int i = 1; i < s; i++)
      {
         for (int j = 0; j < i; j++)
         {
            double fij = F(i, j);
            if (fij >= 0.0)
            {
               gp(i) += fij;
               gm(j) -= fij;
            }
            else
            {
               gm(i) += fij;
               gp(j) -= fij;
            }
         }
      }

      du_loc = du_LO_loc;

      for (int i = 0; i < s; i++)
      {
         double rp = std::max(ML(i) * (du_max_loc(i) - du_loc(i)), 0.0);
         double rm = std::min(ML(i) * (du_min_loc(i) - du_loc(i)), 0.0);
         double sp = gp(i), sm = gm(i);

         gp(i) = (rp < sp) ? rp / sp : 1.0;
         gm(i) = (rm > sm) ? rm / sm : 1.0;
      }

      for (int i = 1; i < s; i++)
      {
         for (int j = 0; j < i; j++)
         {
            double fij = F(i, j), aij;

            if (fij >= 0.0)
            {
               aij = std::min(gp(i), gm(j));
            }
            else
            {
               aij = std::min(gm(i), gp(j));
            }

            fij *= aij;
            du_loc(i) += fij / ML(i);
            du_loc(j) -= fij / ML(j);
         }
      }
   } // element loop
}

void ElementFCTProjection::CalcFCTProduct(const ParGridFunction &us,
                                          const Vector &m,
                                          const Vector &d_us_HO, const Vector &d_us_LO,
                                          Vector &s_min, Vector &s_max,
                                          const Vector &u_new,
                                          const Array<bool> &active_el,
                                          const Array<bool> &active_dofs, Vector &d_us)
{
   us.HostRead();
   s_min.HostReadWrite();
   s_max.HostReadWrite();
   u_new.HostRead();
   active_el.HostRead();
   active_dofs.HostRead();

   // Compute a compatible low-order solution.
   Vector dus_lo_fct(us.Size()), us_min(us.Size()), us_max(us.Size());
   CalcCompatibleLOProduct(us, m, d_us_HO, s_min, s_max, u_new,
                           active_el, active_dofs, dus_lo_fct);
   ScaleProductBounds(s_min, s_max, u_new, active_el, active_dofs,
                      us_min, us_max);

   // ClipScale solve for d_us.
   CalcFCTSolution(us, m, d_us_HO, dus_lo_fct, us_min, us_max, d_us);
   ZeroOutEmptyDofs(active_el, active_dofs, d_us);
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

   u.HostRead();  m.HostRead();
   du_ho.HostRead(); du_lo.HostRead();
   u_min.HostRead(); u_max.HostRead();
   du.HostReadWrite();

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
