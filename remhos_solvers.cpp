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

void IDPODESolver::AddMasked(const Array<bool> &mask, real_t a,
                             const Vector &va,
                             real_t b, const Vector &vb, Vector &vc)
{
   MFEM_ASSERT(va.Size() == vb.Size() && va.Size() == vc.Size(),
               "incompatible Vectors!");

#if !defined(MFEM_USE_LEGACY_OPENMP)
   const bool use_dev = va.UseDevice() || va.UseDevice() || vc.UseDevice();
   const int N = vc.Size();
   // Note: get read access first, in case c is the same as a/b.
   auto ad = va.Read(use_dev);
   auto bd = vb.Read(use_dev);
   auto cd = vc.Write(use_dev);
   auto maskd = mask.Read(use_dev);
   mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i)
   {
      cd[i] = (maskd[i])?(a * ad[i] + b * bd[i]):(cd[i]);
   });
#else
   const real_t *ap = va.GetData();
   const real_t *bp = vb.GetData();
   real_t       *cp = vc.GetData();
   const bool   *maskp = mask.GetData();
   const int      s = vc.Size();
   #pragma omp parallel for
   for (int i = 0; i < s; i++)
   {
      cp[i] = (maskp[i])?(a * ap[i] + b * bp[i]):(cp[i]);
   }
#endif
}

void IDPODESolver::UpdateMask(const Vector &x, const Vector &dx, real_t dt,
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
   for (int i = 0; i < mask.Size(); i++)
   {
      mask[i] = mask[i] && mask_new[i];
   }
}

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

void RK2IDPSolver::Init(LimitedTimeDependentOperator &f)
{
   IDPODESolver::Init(f);
   dx12.SetSize(f.Height());
   dx.SetSize(f.Height());
}

void RK2IDPSolver::Step(Vector &x, double &t, double &dt)
{
   f->SetTime(t);
   f->SetDt(dt/2.);
   f->MultUnlimited(x, dx12);
   f->LimitMult(x, dx12);

   x.Add(dt/2., dx12);
   f->ComputeMask(x, mask);
   f->SetTime(t+dt/2.);
   //f->SetDt(dt/2.);
   f->MultUnlimited(x, dx);

   UpdateMask(x, dx, dt/2., mask);
   AddMasked(mask, 2., dx, -1., dx12, dx);
   f->LimitMult(x, dx);

   x.Add(dt/2., dx);
   t += dt;
}

} // namespace mfem
