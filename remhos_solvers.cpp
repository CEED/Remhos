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

namespace mfem
{

RK2IDPSolver::RK2IDPSolver()
{
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
   f->Mult(x, dx12);

   x.Add(dt/2., dx12);
   f->SetTime(t+dt/2.);
   f->Mult(x, dx);

   add(2., dx, -1., dx12, dx);

   x.Add(dt/2., dx);
   t += dt;
}

} // namespace mfem
