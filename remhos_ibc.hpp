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

#ifndef MFEM_REMHOS_IBC
#define MFEM_REMHOS_IBC

#include "mfem.hpp"

namespace mfem
{

// Force zero on boundary conditions
void ZeroItOutOnBoundaries(const ParMesh *subcell_mesh,
                           const ParGridFunction *xsub,
                           ParGridFunction &v_sub_gf,
                           VectorGridFunctionCoefficient &v_sub_coef);
// Velocity coefficient
void velocity_function(const Vector &x, Vector &v);

// Initial conditions
double u0_function(const Vector &x);
double s0_function(const Vector &x);

// Inflow boundary condition
double inflow_function(const Vector &x);


} // namespace mfem

#endif // MFEM_REMHOS_IBC
