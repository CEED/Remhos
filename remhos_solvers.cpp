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

#include "remhos_solvers.hpp"
#include "general/forall.hpp"

namespace mfem
{

void ForwardEulerIDPSolver::Init(LimitedTimeDependentOperator &f)
{
   IDPODESolver::Init(f);
   dx.SetSize(f.Height());
}

void ForwardEulerIDPSolver::Step(Vector &x, double &t, double &dt)
{
   f->SetTime(t);
   f->SetDt(dt);
   f->MultUnlimited(x, dx);
   f->LimitMult(x, dx);

   x.Add(dt, dx);
   t += dt;
}

void RKIDPSolver::ConstructD()
{
   // Convert high-order to Forward Euler factors
   d = new real_t[s*(s+1)/2];

   const real_t *a_n = a; // new coeff line
   const real_t *a_o = a; // old coeff line
   int i_o = -1; // old stage
   real_t c_o = 0.; // old time fraction

   for (int i = 0; i < s; i++)
   {
      const real_t c_n = (i<s-1)?(c[i]):(1.); // new time fraction
      const real_t dc = c_n - c_o; // time fraction diff
      real_t *di = d + i*(i+1)/2;

      for (int j = 0; j < i; j++)
      {
         const real_t a_oj = (j<=i_o)?(a_o[j]):(0.); // old coeff
         const real_t m = (a_n[j] - a_oj) / dc; // old HO update coeff
         if (m == 0.)
         {
            di[j] = 0.;
            continue;
         }
         // Express j-th HO update by Forward Euler updates
         const real_t *dj = d + j*(j+1)/2;
         const real_t dij = m / dj[j];
         for (int k = 0; k < j; k++)
         {
            di[k] -= dj[k] * dij;
         }
         di[j] = dij;
      }
      di[i] = a_n[i] / dc;

      // Update stage

      const double c_next = (i < s-2)?(c[i+1]):(1.);
      if (c_next > c_n)
      {
         i_o = i;
         c_o = c_n;
         a_o = a_n;
      }

      if (i < s-2)
      {
         a_n += i+1;
      }
      else
      {
         a_n = b;
      }
   }
}

void RKIDPSolver::AddMasked(const Array<bool> &mask, real_t b, const Vector &vb,
                            Vector &va)
{
   MFEM_ASSERT(va.Size() == vb.Size(),
               "incompatible Vectors!");

#if !defined(MFEM_USE_LEGACY_OPENMP)
   const bool use_dev = va.UseDevice() || va.UseDevice();
   const int N = va.Size();
   // Note: get read access first, in case c is the same as a/b.
   auto ad = va.ReadWrite(use_dev);
   auto bd = vb.Read(use_dev);
   auto maskd = mask.Read(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i)
   {
      ad[i] += (maskd[i])?(b * bd[i]):(0.);
   });
#else
   real_t *ap = va.GetData();
   const real_t *bp = vb.GetData();
   const bool   *maskp = mask.GetData();
   const int      s = va.Size();
   #pragma omp parallel for
   for (int i = 0; i < s; i++)
   {
      ap[i] += (maskp[i])?(b * bp[i]):(0.);
   }
#endif
}

void RKIDPSolver::UpdateMask(const Vector &x, const Vector &dx, real_t dt,
                             Array<bool> &mask)
{
   Array<bool> mask_new(mask.Size());
   if (dt != 0.)
   {
      Vector x_new(x.Size());
      add(x, dt, dx, x_new);
      f->ComputeMask(x_new, mask_new);
   }
   else
   {
      f->ComputeMask(x, mask_new);
   }
   // All intermediate updates must be on active DOFs
   // for a valid high-order update
   for (int i = 0; i < mask.Size(); i++)
   {
      mask[i] = mask[i] && mask_new[i];
   }
}

RKIDPSolver::RKIDPSolver(int s_, const real_t a_[], const real_t b_[],
                         const real_t c_[])
   : s(s_), a(a_), b(b_), c(c_)
{
   dxs = new Vector[s];
   ConstructD();
}

RKIDPSolver::~RKIDPSolver()
{
   delete[] dxs;
}

void RKIDPSolver::Init(LimitedTimeDependentOperator &f_)
{
   IDPODESolver::Init(f_);
   for (int i = 0; i < s; i++)
   {
      dxs[i].SetSize(f->Height());
   }
}

void RKIDPSolver::Step(Vector &x, double &t, double &dt)
{
   real_t c_o = 0.;

   // Perform the first step
   f->SetTime(t);
   f->SetDt(c[0] * dt);
   f->MultUnlimited(x, dxs[0]);
   f->LimitMult(x, dxs[0]);

   // Update state
   const double c_next = (s > 2)?(c[1]):(1.);
   if (c_next > c[0])// only when advancing after
   {
      x.Add(c[0] * dt, dxs[0]);
      if (use_masks) { f->ComputeMask(x, mask); }
      f->SetTime(t + c[0] * dt);
      c_o = c[0];
   }
   else
   {
      // Only initialize the mask
      Vector x_new(x.Size());
      add(x, c[0] * dt, dxs[0], x_new);
      if (use_masks) { f->ComputeMask(x_new, mask); }
   }

   // Step through higher stages

   const real_t *d_i = d + 1;

   for (int i = 1; i < s; i++)
   {
      const real_t c_n = (i<s-1)?(c[i]):(1.);
      const real_t dc = c_n - c_o;
      const real_t dct = dc * dt;

      // Explicit HO step
      f->SetDt(dct);
      f->MultUnlimited(x, dxs[i]);

      // Update mask with the HO update
      if (use_masks) { UpdateMask(x, dxs[i], dct, mask); }

      //
      // Form the unlimited update for the stage.
      // Note that it converts eq. (2.16) in JLG's paper into an update using
      // the previous limited updates.
      //
      {
         // for mask = 0, we get dxs (nothing happens).
         //               the loop below won't change it -> Forward Euler.
         // for mask = 1, we scale dxs by d_i[i].
         if (use_masks) { AddMasked(mask, d_i[i]-1., dxs[i], dxs[i]); }
         else           { dxs[i] *= d_i[i]; }
      }
      // Use all previous limited updates.
      for (int j = 0; j < i; j++)
      {
         if (use_masks) { AddMasked(mask, d_i[j], dxs[j], dxs[i]); }
         else           { dxs[i].Add(d_i[j], dxs[j]); }
      }

      // Limit the step (always a Forward Euler step).
      f->LimitMult(x, dxs[i]);

      // Update the state
      const double c_next = (i < s-2)?(c[i+1]):(1.);
      if (i == s-1 || c_next > c_n)// only when advancing after
      {
         f->SetTime(t + c_n * dt);
         x.Add(dct, dxs[i]);
         c_o = c_n;
      }
      d_i += i+1;
   }

   t += dt;
}

//2-stage, 2nd order
const real_t RK2IDPSolver::a[] = {.5};
const real_t RK2IDPSolver::b[] = {0., 1.};
const real_t RK2IDPSolver::c[] = {.5};

//3-stage, 3rd order
const real_t RK3IDPSolver::a[] = {1./3., 0., 2./3.};
const real_t RK3IDPSolver::b[] = {.25, 0., .75};
const real_t RK3IDPSolver::c[] = {1./3., 2./3.};

//4-stage, 4th order for linear, 3rd for non-linear
//const real_t RK4IDPSolver::a[] = {.25, 0., .5, 0., .25, .5};
//const real_t RK4IDPSolver::b[] = {0., 2./3., -1./3., 2./3.};
//const real_t RK4IDPSolver::c[] = {.25, .5, .75};
//4-stage, 4th order, non-equidistant
//const real_t RK4IDPSolver::a[] = {.5, 0., .5, 0., 0., 1.};
//const real_t RK4IDPSolver::b[] = {1./6., 2./6., 2./6., 1./6.};
//const real_t RK4IDPSolver::c[] = {.5, .5, 1.};
//4-stage, 4th order, equidistant (except the last)
const real_t RK4IDPSolver::a[] = {1./3., -1./3., 1., 1., -1., 1.};
const real_t RK4IDPSolver::b[] = {1./8., 3./8., 3./8., 1./8.};
const real_t RK4IDPSolver::c[] = {1./3., 2./3., 1.};

//6-stage, 5th order, equidistant (except the last)
const real_t RK6IDPSolver::a[] = {.25, 1./8., 1./8., 0., -.5, 1., 3./16., 0., 0.,
                                  9./16., -3./7., 2./7., 12./7., -12./7., 8./7.
                                 };
const real_t RK6IDPSolver::b[] = {7./90., 0., 32./90., 12./90., 32./90., 7./90.};
const real_t RK6IDPSolver::c[] = {.25, .25, .5, .75, 1.};

} // namespace mfem
