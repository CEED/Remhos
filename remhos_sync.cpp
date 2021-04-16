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

   ind_elem.HostWrite();
   ind_dofs.HostWrite();
   u.HostRead();

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
void ComputeRatio(int NE, const Vector &us, const Vector &u, Vector &s,
                  Array<bool> &bool_el, Array<bool> &bool_dof)
{
   ComputeBoolIndicators(NE, u, bool_el, bool_dof);

   us.HostRead();
   u.HostRead();
   s.HostWrite();
   bool_el.HostRead();
   bool_dof.HostRead();

   const int ndof = u.Size() / NE;
   for (int i = 0; i < NE; i++)
   {
      if (bool_el[i] == false)
      {
         for (int j = 0; j < ndof; j++) { s(i*ndof + j) = 0.0; }
         continue;
      }

      const double *u_el = &u(i*ndof), *us_el = &us(i*ndof);
      double *s_el = &s(i*ndof);

      // Average of the existing values.
      int n = 0;
      double sum = 0.0;
      for (int j = 0; j < ndof; j++)
      {
         if (u_el[j] > EMPTY_ZONE_TOL)
         {
            sum += us_el[j] / u_el[j];
            n++;
         }
      }
      MFEM_VERIFY(n > 0, "Major error that makes no sense");
      const double s_avg = sum / n;

      for (int j = 0; j < ndof; j++)
      {
         if (u_el[j] <= 0.0)
         {
            s_el[j] = s_avg;
         }
         else
         {
            const double s_j = us_el[j] / u_el[j];
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
   ind_elem.HostRead();
   ind_dofs.HostRead();
   u.HostReadWrite();

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

void CorrectFCT(const Vector &x_min, const Vector &x_max, ParGridFunction &x)
{
   x.HostReadWrite();

   ParFiniteElementSpace &pfes = *x.ParFESpace();
   ConstantCoefficient one(1.0);
   MassIntegrator m_integ(one);

   const int NE = pfes.GetNE();
   const int ndofs = x.Size() / NE;
   int dof_id;

   // Q0 solutions can't be adjusted conservatively. It's what it is.
   if (ndofs == 1) { return; }

   Vector x_loc, ML_loc(ndofs), m_loc(ndofs);
   DenseMatrix M_loc;

   for (int k = 0; k < NE; k++)
   {
      bool fix = false;
      for (int j = 0; j < ndofs; j++)
      {
         dof_id = k * ndofs + j;
         if (x(dof_id) < x_min(dof_id) ||
             x(dof_id) > x_max(dof_id)) { fix = true; }
      }

      if (fix == false) { continue; }

      x_loc.SetDataAndSize(x.GetData() + k*ndofs, ndofs);

      const FiniteElement &fe = *pfes.GetFE(k);
      m_integ.AssembleElementMatrix(fe, *pfes.GetElementTransformation(k), M_loc);
      m_loc = 1.0;
      M_loc.Mult(m_loc, ML_loc);
      M_loc.Mult(x_loc, m_loc);

      const double x_avg = m_loc.Sum() / ML_loc.Sum();

#ifdef REMHOS_FCT_PRODUCT_DEBUG
      for (int j = 0; j < ndofs; j++)
      {
         dof_id = k * ndofs + j;
         if (x_avg < x_min(dof_id) || x_avg > x_min(dof_id))
         {
            std::cout << x_avg << " " << x_min << " " << x_max << std::endl;
            MFEM_ABORT("In correction: avg is not in bounds!");
         }
      }
#endif

      Vector z(ndofs);
      Vector beta(ndofs);
      for (int i = 0; i < ndofs; i++)
      {
         // Some different options for beta:
         //beta(i) = 1.0;
         beta(i) = ML_loc(i);

         // The low order flux correction
         z(i) = m_loc(i) - ML_loc(i) * x_avg;
      }

      // Make beta_i sum to 1
      beta /= beta.Sum();

      DenseMatrix F(ndofs);
      for (int i = 1; i < ndofs; i++)
      {
         for (int j = 0; j < i; j++)
         {
            F(i, j) = M_loc(i, j) * (x(i) - x(j)) +
                      (beta(j) * z(i) - beta(i) * z(j));
         }
      }

      Vector gp(ndofs), gm(ndofs);
      gp = 0.0; gm = 0.0;
      for (int i = 1; i < ndofs; i++)
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

      x_loc = x_avg;
      for (int i = 0; i < ndofs; i++)
      {
         int dof_id = k*ndofs + i;
         double mi = ML_loc(i);
         double rp = std::max(mi * (x_max(dof_id) - x_loc(i)), 0.0);
         double rm = std::min(mi * (x_min(dof_id) - x_loc(i)), 0.0);
         double sp = gp(i), sm = gm(i);

         gp(i) = (rp < sp) ? rp / sp : 1.0;
         gm(i) = (rm > sm) ? rm / sm : 1.0;
      }

      for (int i = 1; i < ndofs; i++)
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
            x_loc(i) += fij / ML_loc(i);
            x_loc(j) -= fij / ML_loc(j);
         }
      }
   }
}

void FCT_Project(DenseMatrix &M, DenseMatrixInverse &M_inv,
                 Vector &m, Vector &x,
                 double y_min, double y_max, Vector &xy)
{
   // [IN]  - M, M_inv, m, x, y_min, y_max
   // [OUT] - xy

   m.HostReadWrite();
   x.HostReadWrite();
   xy.HostReadWrite();
   const int s = M.Size();

   xy.SetSize(s);

   // Compute the lumped mass matrix in ML
   Vector ML(s);
   M.GetRowSums(ML);

   // Compute the high-order projection in xy
   M_inv.Mult(m, xy);

   // Q0 solutions can't be adjusted conservatively. It's what it is.
   if (xy.Size() == 1) { return; }

   // Ensure dot product is done on the CPU
   double dMLX(0);
   for (int i = 0; i < x.Size(); ++i)
   {
      dMLX += ML(i) * x(i);
   }

   const double y_avg = m.Sum() / dMLX;

#ifdef DEBUG
   EXO_WARNING_IF(!(y_min < y_avg + 1e-12 && y_avg < y_max + 1e-12),
                  "Average is out of bounds: "
                     << "y_min < y_avg + 1e-12 && y_avg < y_max + 1e-12 " << y_min << " " << y_avg
                     << " " << y_max);
#endif

   Vector z(s);
   Vector beta(s);
   Vector Mxy(s);
   M.Mult(xy, Mxy);
   for (int i = 0; i < s; i++)
   {
      // Some different options for beta:
      //beta(i) = 1.0;
      beta(i) = ML(i) * x(i);
      //beta(i) = ML(i)*(x(i) + 1e-14);
      //beta(i) = ML(i);
      //beta(i) = Mxy(i);

      // The low order flux correction
      z(i) = m(i) - ML(i) * x(i) * y_avg;
   }

   // Make beta_i sum to 1
   beta /= beta.Sum();

   DenseMatrix F(s);
   for (int i = 1; i < s; i++)
   {
      for (int j = 0; j < i; j++)
      {
         F(i, j) = M(i, j) * (xy(i) - xy(j)) + (beta(j) * z(i) - beta(i) * z(j));
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

   for (int i = 0; i < s; i++)
   {
      xy(i) = x(i) * y_avg;
   }

   for (int i = 0; i < s; i++)
   {
      double mi = ML(i), xyLi = xy(i);
      double rp = std::max(mi * (x(i) * y_max - xyLi), 0.0);
      double rm = std::min(mi * (x(i) * y_min - xyLi), 0.0);
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
         xy(i) += fij / ML(i);
         xy(j) -= fij / ML(j);
      }
   }
}


void ComputeMinMaxS(int NE, const Vector &u_s, const Vector &u, int myid)
{
   const int size = u.Size();
   Vector s(size);
   Array<bool> bool_el, bool_dofs;
   ComputeBoolIndicators(NE, u, bool_el, bool_dofs);
   ComputeRatio(NE, u_s, u, s, bool_el, bool_dofs);

   bool_dofs.HostRead();

   double min_s = numeric_limits<double>::infinity();
   double max_s = -numeric_limits<double>::infinity();
   for (int i = 0; i < size; i++)
   {
      if (bool_dofs[i] == false) { continue; }

      min_s = min(s(i), min_s);
      max_s = max(s(i), max_s);
   }
   double min_s_glob, max_s_glob;
   MPI_Allreduce(&min_s, &min_s_glob, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
   MPI_Allreduce(&max_s, &max_s_glob, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

   if (myid == 0)
   {
      std::cout << std::scientific << std::setprecision(5);
      std::cout << "min_s: " << min_s_glob
                << "; max_s: " << max_s_glob << std::endl;
   }
}

void ComputeMinMaxS(const Vector &s, const Array<bool> &bool_dofs, int myid)
{
   s.HostRead();
   bool_dofs.HostRead();

   const int size = s.Size();
   double min_s = numeric_limits<double>::infinity();
   double max_s = -numeric_limits<double>::infinity();
   for (int i = 0; i < size; i++)
   {
      if (bool_dofs[i] == false) { continue; }

      min_s = min(s(i), min_s);
      max_s = max(s(i), max_s);
   }
   double min_s_glob, max_s_glob;
   MPI_Allreduce(&min_s, &min_s_glob, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
   MPI_Allreduce(&max_s, &max_s_glob, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

   if (myid == 0)
   {
      std::cout << std::scientific << std::setprecision(5);
      std::cout << "min_s: " << min_s_glob
                << "; max_s: " << max_s_glob << std::endl;
   }
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
