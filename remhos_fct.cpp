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
   const double eps = 1.0e-15;

   for (int k = 0; k < NE; k++)
   {
      sumPos = sumNeg = 0.0;

      // Clip.
      for (int j = 0; j < nd; j++)
      {
         dof_id = k*nd+j;

         u_new_ho   = u(dof_id) + dt * du_ho(dof_id);
         u_new_lo   = u(dof_id) + dt * du_lo(dof_id);

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


} // namespace mfem
