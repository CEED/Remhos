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

void ComputeBoolIndicators(int NE, const Vector &u,
                           Array<bool> &ind_elem, Array<bool> &ind_dofs)
{
   ind_elem.SetSize(NE);
   ind_dofs.SetSize(u.Size());

   const int ndof = u.Size() / NE;
   int dof_id;
   for (int i = 0; i < NE; i++)
   {
      ind_elem[i] = false;
      for (int j = 0; j < ndof; j++)
      {
         dof_id = i*ndof + j;
         ind_dofs[dof_id] = (u(dof_id) > EMPTY_ZONE_TOL) ? true : false;

         if (u(dof_id) > EMPTY_ZONE_TOL) { ind_elem[i] = true; }
      }
   }
}

// This function assumes a DG space.
void ComputeRatio(int NE, const Vector &u_s, const Vector &u,
                  const Vector &lumpedM, Vector &s,
                  Array<bool> &bool_el, Array<bool> &bool_dof)
{
   ComputeBoolIndicators(NE, u, bool_el, bool_dof);

   const int ndof = u.Size() / NE;
   for (int i = 0; i < NE; i++)
   {
      if (bool_el[i] == false)
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
         if (u_el[j] <= 0.0)
         {
            s_el[j] = s_avg;
         }
         else
         {
            const double s_j = u_s_el[j] / u_el[j];
            if (u_el[j] > EMPTY_ZONE_TOL) { s_el[j] = s_j; }
            else
            {
               // Continuous transition between s_avg and s for u in [0, tol].
               s_el[j] = u_el[j] * (s_j - s_avg) / EMPTY_ZONE_TOL + s_avg;
            }

            // NOTE: the above transition alters slightly the values of
            // s = us / u, near u = EMPTY_ZONE_TOL. This might break the theorem
            // stating that s_min <= us_LO / u_LO <= s_max, as s_min and s_max
            // are different, due to s not being exactly us / u.
         }
      }
   }
}

void ZeroOutEmptyDofs(const Array<bool> &ind_elem,
                      const Array<bool> &ind_dofs, Vector &u)
{
   const int NE = ind_elem.Size();
   const int ndofs = u.Size() / NE;
   for (int k = 0; k < NE; k++)
   {
      if (ind_elem[k] == true) { continue; }

      for (int i = 0; i < ndofs; i++)
      {
         if (ind_dofs[k*ndofs + i] == false) { u(k*ndofs + i) = 0.0; }
      }
   }
}

void ComputeMinMaxS(int NE, const Vector &u_s, const Vector &u,
                    const Vector &lumpedM)
{
   const int size = u.Size();
   Vector s(size);
   Array<bool> bool_el, bool_dofs;
   ComputeBoolIndicators(NE, u, bool_el, bool_dofs);
   ComputeRatio(NE, u_s, u, lumpedM, s, bool_el, bool_dofs);
   double min_s = numeric_limits<double>::infinity();
   double max_s = -numeric_limits<double>::infinity();
   for (int i = 0; i < size; i++)
   {
      if (bool_dofs[i] == false) { continue; }

      min_s = min(s(i), min_s);
      max_s = max(s(i), max_s);
   }
   std::cout << min_s << " " << max_s << std::endl;
}

void PrintCellValues(int cell_id, int NE, const Vector &vec, const char *msg)
{
   std::cout << msg << std::endl;
   const int ndofs = vec.Size() / NE;
   for (int i = 0; i < ndofs; i++)
   {
      std::cout << vec(cell_id * ndofs + i) << " ";
   }
   std::cout << endl;
}

void VerifyLOProduct(int NE, const Vector &us_LO, const Vector &u_LO,
                     const Vector &s_min, const Vector &s_max,
                     const Array<bool> &active_el,
                     const Array<bool> &active_dofs)
{
   const double eps = 1.0e-12;
   const int ndofs = u_LO.Size() / NE;
   Vector s_min_loc, s_max_loc;

   for (int k = 0; k < NE; k++)
   {
      if (active_el[k] == false) { continue; }

      const double *us = &us_LO(k*ndofs), *u = &u_LO(k*ndofs);
      s_min_loc.SetDataAndSize(s_min.GetData() + k*ndofs, ndofs);
      s_max_loc.SetDataAndSize(s_max.GetData() + k*ndofs, ndofs);
      double s_min = numeric_limits<double>::infinity(),
             s_max = -numeric_limits<double>::infinity();
      for (int j = 0; j < ndofs; j++)
      {
         if (active_dofs[k*ndofs + j] == false) { continue; }
         s_min = min(s_min, s_min_loc(j));
         s_max = max(s_max, s_max_loc(j));
      }

      for (int j = 0; j < ndofs; j++)
      {
         if (active_dofs[k*ndofs + j] == false) { continue; }

         if (us[j] + eps < s_min * u[j] ||
             us[j] - eps > s_max * u[j])
         {
            const double s_LO = us[j] / u[j];
            std::cout << "Element " << k << std::endl
                      << "At " << j << " out of " << ndofs << std::endl
                      << "Basic LO product theorem is violated: " << endl
                      << s_min << " <= " << s_LO << " <= " << s_max << std::endl
                      << s_min * u[j] << " <= "
                      << us[j] << " <= " << s_max * u[j] << std::endl
                      << "s_LO = " << us[j] << " / " << u[j] << std::endl;

            PrintCellValues(k, NE, us_LO, "us_LO_loc: ");
            PrintCellValues(k, NE, u_LO, "u_LO_loc: ");

            MFEM_ABORT("[us_LO/u_LO] is not in the full stencil bounds!");
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
