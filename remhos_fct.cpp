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

void ClipScaleSolver::CalcFCTSolution(const Vector &u, const Vector &m,
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

         f_clip_min = m(dof_id) / dt * (u_min(dof_id) - u_new_lo);
         f_clip_max = m(dof_id) / dt * (u_max(dof_id) - u_new_lo);

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

void NonlinearPenaltySolver::CalcFCTSolution(const Vector &u, const Vector &m,
   const Vector &du_ho, const Vector &du_lo,
   const Vector &u_min, const Vector &u_max, Vector &du) const
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

void get_z(double lambda, Vector &w, Vector &flux, Vector &zz)
{
   if (lambda==0)
      zz = 0.;
   else
      for (int j=0; j<w.Size(); j++)
      {
         if (flux(j)!=0)
            zz(j) = (abs(flux(j)) >= lambda*abs(w(j))) ? w(j) : flux(j)/lambda;
         else
            zz(j) = 0;
      }
}

double get_lambda_times_sum_z(double lambda, Vector &w, Vector &flux)
{
   //double tol=1e-30;
   Vector lambda_times_z(w.Size());
   double lambda_times_sum_z=0.;
   for (int j=0; j<w.Size(); j++)
   {
      //if (abs(flux(j)) > tol) //flux(j)!=0
      if (flux(j)!=0)
      {
         //lambda_times_z(j) = lambda*w(j) + flux(j)*min(0.,1-lambda*w(j)/flux(j));
         lambda_times_z(j) = ((abs(flux(j)) >= lambda*abs(w(j))) ? lambda*w(j)
                                                                 : flux(j));
      }
      else
         lambda_times_z(j) = 0;
      lambda_times_sum_z += lambda_times_z(j);
   }
   return lambda_times_sum_z;
}

double get_lambda(double lambda, double delta, Vector &w, Vector &flux,
                  Vector &zz)
{
   // solve nonlinearity F(lambda)=0
   double F=0.;
   double tol=1e-14;
   double lambdaLower=0., lambdaUpper = 0.;
   double FLower=0., FUpper=0.;
   double factor=1.;

   // compute starting F
   F= delta - get_lambda_times_sum_z(lambda,w,flux);
   // check F at extremum of lambda {0,1}
   double F0 = delta-get_lambda_times_sum_z(0,w,flux);
   double F1 = delta-get_lambda_times_sum_z(1,w,flux);
   if (abs(F)<=tol)
   {
      get_z(lambda, w, flux, zz);
      return lambda;
   }
   else if (abs(F0)<=tol)
   {
      get_z(0,w,flux,zz);
      return 0.;
   }
   else if (abs(F1)<=tol)
   {
      get_z(1,w,flux,zz);
      return 1.;
   }
   else // solve non-linearity
   {
      do {
         factor*=2;
         //look for other lambda to have opposite sign in F
         lambdaLower = lambda/factor;
         lambdaUpper = factor*lambda;
         FLower=delta-get_lambda_times_sum_z(lambdaLower,w,flux);
         FUpper=delta-get_lambda_times_sum_z(lambdaUpper,w,flux);

         //cout << "****************" << endl;
         //cout << lambdaLower << ", " << lambdaUpper << endl;
         //cout << FLower << ", " << FUpper << endl;
         //cout << "delta: " << delta << endl;
         //w.Print(cout,12);
         //flux.Print(cout,12);
         //cout << "****************" << endl;
      } while ((F*FLower > 0) && (F*FUpper > 0));
      //cout << "******************************************" << endl;

      //cout << lambdaLower << ", " << lambdaUpper << endl;
      //cout << FLower << ", " << FUpper << endl;
      //abort();

      // Check if either of lambdaLower or lambdaUpper hit the solution
      if (FLower==0)
      {
         get_z(lambdaLower, w, flux, zz);
         return lambdaLower;
      }
      else if (FUpper==0)
      {
         get_z(lambdaUpper, w, flux, zz);
         return lambdaUpper;
      }
      else
      {
         // Get STARTING lower and upper bounds for lambda
         if (F*FLower < 0) // F>0
            lambdaUpper = lambda;
         else // F<0
            lambdaLower = lambda;

         // get STARTING lower and upper bounds on F
         FLower=delta-get_lambda_times_sum_z(lambdaLower,w,flux);
         FUpper=delta-get_lambda_times_sum_z(lambdaUpper,w,flux);

         //cout << FLower << ", " << FUpper << endl;
         //abort();
         do
         {
            // compute new lambda and new F
            lambda = 0.5*(lambdaLower+lambdaUpper);
            F = delta - get_lambda_times_sum_z(lambda,w,flux);
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
         } while (abs(F)>tol);
         //cout << "*******************************************" << endl;

         lambda = 0.5*(lambdaLower+lambdaUpper);
         get_z(lambda, w, flux, zz);
         return lambda;
      }
   }
}

void NonlinearPenaltySolver::CorrectFlux(Vector &fluxL, Vector &fluxH,
                                         Vector &flux_fix) const
{
   // This consider any definition of wi. If a violation on MPP is created,
   // then wi is s.t. fi=0.
   // The idea is to relax the penalization wi in favor of MPP
   int num_cells = pfes.GetNE();
   int xd = pfes.GetFE(0)->GetDof();

   Array<int> ldofs;
   Vector flux_z(xd), fluxH_z(xd), flux_correction_z(xd);
   for (int i = 0; i < num_cells; i++)
   {
      pfes.GetElementDofs(i, ldofs);
      fluxL.GetSubVector(ldofs, flux_z);
      fluxH.GetSubVector(ldofs, fluxH_z);

      double fp = 0.0, fn= 0.0;
      for (int j = 0; j < xd; j++)
      {
         if (flux_z(j) >= 0.0) { fp += flux_z(j); }
         else                  { fn += flux_z(j); }
      }

      double delta = fp + fn;

      if (delta == 0.0)
      {
         flux_correction_z = 0.0;
         flux_fix.SetSubVector(ldofs, flux_correction_z);
         continue;
      }

      Vector w(xd), zz(xd);

      const double eps = pfes.GetMesh()->GetElementSize(0,0) / pfes.GetOrder(0);

      // compute penalization terms wi's as desired
      for (int j = 0; j < xd; j++)
      {
         if (delta > 0.0)
         {
            w(j) = (flux_z(j) > 0.0) ? eps * abs(flux_z(j)) +
                                       abs(get_max_on_cellNi(fluxH_z))
                                     : 0.0;
         }
         else
         {
            w(j) = (flux_z(j) < 0.0) ? - eps * abs(flux_z(j))
                                       - abs(get_max_on_cellNi(fluxH_z))
                                     : 0.0;
         }
      }

      // compute lamdba
      double lambda = get_lambda(1.0, delta, w, flux_z, zz);
      // compute flux correction
      for (int j = 0; j < xd; j++)
         flux_correction_z(j) = -lambda * zz(j);

      flux_fix.SetSubVector(ldofs, flux_correction_z);
   }
}

double NonlinearPenaltySolver::get_max_on_cellNi(Vector &fluxH) const
{
   double MAX = -1.0;
   for (int i = 0; i < fluxH.Size(); i++)
      MAX = max(fabs(fluxH(i)), MAX);
   return MAX;
}

} // namespace mfem
