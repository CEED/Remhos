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

#ifndef MFEM_REMHOS
#define MFEM_REMHOS

#include <fstream>
#include <iostream>

#include "mfem.hpp"
using namespace mfem;

// Choice for the problem setup. The fluid velocity, initial condition and
// inflow boundary condition are chosen based on this parameter.
extern int problem_num;

// 0 is standard transport.
// 1 is standard remap (mesh moves, solution is fixed).
extern int exec_mode;

// Mesh bounding box
extern Vector bb_min, bb_max;

#include "remhos_ho.hpp"
#include "remhos_lo.hpp"
#include "remhos_ibc.hpp"
#include "remhos_fct.hpp"
#include "remhos_mono.hpp"
#include "remhos_sync.hpp"
#include "remhos_tools.hpp"
#include "remhos_adv.hpp"
#include "remhos_amr.hpp"

enum class HOSolverType {None, Neumann, CG, LocalInverse};
enum class LOSolverType {None, DiscrUpwind, DiscrUpwindPrec, ResDist, ResDistSubcell};
enum class FCTSolverType {None, FluxBased, ClipScale, NonlinearPenalty};
enum class MonolithicSolverType {None, ResDistMono, ResDistMonoSubcell};

#endif // MFEM_REMHOS
