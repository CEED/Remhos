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

#include "remhos_ho.hpp"

using namespace std;

namespace mfem
{

void NeumannSolver::CalcHOSolution(const Vector &u, Vector &du) const
{
   const int n = u.Size();
   Vector rhs(n), res(n);

   // K multiplies a ldofs Vector, as we're always doing DG.
   K.Mult(u, rhs);

   du = 0.0;
   const double abs_tol = 1.e-4;
   const int max_iter = 20;
   for (int iter = 1; iter <= max_iter; iter++)
   {
      M.Mult(du, res);
      res -= rhs;

      double resid_loc = res.Norml2(); resid_loc *= resid_loc;
      double resid;
      MPI_Allreduce(&resid_loc, &resid, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      resid = std::sqrt(resid);
      if (resid <= abs_tol) { return; }

      for (int i = 0; i < n; i++)
      {
         du(i) -= res(i) / M_lumped(i);
      }
   }
}


} // namespace mfem
