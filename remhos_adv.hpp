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

#ifndef MFEM_REMHOS_ADV
#define MFEM_REMHOS_ADV

#include "remhos.hpp"

using namespace std;
using namespace mfem;

namespace mfem
{

class AdvectionOperator : public TimeDependentOperator
{
private:
   BilinearForm &Mbf, &ml;
   ParBilinearForm &Kbf;
   ParBilinearForm &M_HO, &K_HO;
   Vector &lumpedM;

   Vector start_mesh_pos, start_submesh_pos;
   GridFunction &mesh_pos, *submesh_pos, &mesh_vel, &submesh_vel;

   mutable ParGridFunction x_gf;

   double dt;
   Assembly &asmbl;

   LowOrderMethod &lom;
   DofInfo &dofs;

   HOSolver *ho_solver;
   LOSolver *lo_solver;
   FCTSolver *fct_solver;
   MonolithicSolver *mono_solver;

public:
   AdvectionOperator(int size, BilinearForm &Mbf_, BilinearForm &_ml,
                     Vector &_lumpedM,
                     ParBilinearForm &Kbf_,
                     ParBilinearForm &M_HO_, ParBilinearForm &K_HO_,
                     GridFunction &pos, GridFunction *sub_pos,
                     GridFunction &vel, GridFunction &sub_vel,
                     Assembly &_asmbl, LowOrderMethod &_lom, DofInfo &_dofs,
                     HOSolver *hos, LOSolver *los, FCTSolver *fct,
                     MonolithicSolver *mos);

   virtual void Mult(const Vector &x, Vector &y) const;

   virtual void SetDt(double _dt) { dt = _dt; }
   void SetRemapStartPos(const Vector &m_pos, const Vector &sm_pos)
   {
      start_mesh_pos    = m_pos;
      start_submesh_pos = sm_pos;
   }

   virtual ~AdvectionOperator() { }

   void AMRUpdate(const Vector &S,
                  ParGridFunction &u,
                  const double mass0_u);
};

} // namespace mfem

#endif // MFEM_REMHOS_ADV
