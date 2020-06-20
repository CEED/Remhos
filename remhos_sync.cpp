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

#include "remhos_sync.hpp"

using namespace std;

namespace mfem
{

void ComputeBoolIndicator(const ParGridFunction &u, Array<bool> &ind)
{
   const int NE = u.ParFESpace()->GetMesh()->GetNE();
   const int ndof = u.ParFESpace()->GetFE(0)->GetDof();

   ind.SetSize(NE);
   for (int i = 0; i < NE; i++)
   {
      ind[i] = false;
      for (int j = 0; j < ndof; j++)
      {
         // This indexing assumes u is DG.
         if (u(i*ndof + j) > EMPTY_ZONE_TOL) { ind[i] = true; break; }
      }
   }
}

// This function assumes a DG space.
void ComputeGFRatio(const ParGridFunction &u_s, const ParGridFunction &u,
                    const Vector &lumpedM, ParGridFunction &s)
{
   Array<bool> u_bool;
   ComputeBoolIndicator(u, u_bool);

   const int NE = u_s.ParFESpace()->GetNE();
   const int ndof = u_s.Size() / NE;

   for (int i = 0; i < NE; i++)
   {
      if (u_bool[i] == false)
      {
         for (int j = 0; j < ndof; j++) { s(i*ndof + j) = 0.0; }
         continue;
      }

      const double *u_el = &u(i*ndof), *u_s_el = &u_s(i*ndof);
      double *s_el = &s(i*ndof);

      double mass_u_s = 0.0, mass_u = 0.0;
      for (int j = 0; j < ndof; j++)
      {
         mass_u += lumpedM(i*ndof + j) * u_el[j];
         mass_u_s += lumpedM(i*ndof + j) * u_s_el[j];
      }
      const double s_avg = mass_u_s / mass_u;

      for (int j = 0; j < ndof; j++)
      {
         if (u_el[j] <= EMPTY_ZONE_TOL)
         {
            s_el[j] = s_avg;
         }
         else
         {
            const double ss = u_s_el[j] / u_el[j];
            // Soft transition between s_avg and s.
            const double soft01 = 1.0 - std::exp(- u_el[j] / EMPTY_ZONE_TOL);
            s_el[j] = (ss - s_avg) * soft01 + s_avg;
         }
      }
   }
}

double BoolFunctionCoefficient::Eval(ElementTransformation &T,
                                     const IntegrationPoint &ip)
{
   if (ind[T.ElementNo] == true)
   {
      return FunctionCoefficient::Eval(T, ip);
   }
   else { return 0.0; }
}

} // namespace mfem
