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
   constexpr double cutoff = 1.0e-12;
   const int NE = u.ParFESpace()->GetMesh()->GetNE();
   const int ndof = u.ParFESpace()->GetFE(0)->GetDof();

   ind.SetSize(NE);
   for (int i = 0; i < NE; i++)
   {
      ind[i] = false;
      for (int j = 0; j < ndof; j++)
      {
         // This indexing assumes u is DG.
         if (u(i*ndof + j) > cutoff) { ind[i] = true; break; }
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
