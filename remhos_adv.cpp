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

#define MFEM_DEBUG_COLOR 220
#include "debug.hpp"

#include "remhos.hpp"

using namespace std;
using namespace mfem;

namespace mfem
{

AdvectionOperator::AdvectionOperator(int size,
                                     BilinearForm &Mbf_,
                                     BilinearForm &_ml,
                                     Vector &_lumpedM,
                                     ParBilinearForm &Kbf_,
                                     ParBilinearForm &M_HO_,
                                     ParBilinearForm &K_HO_,
                                     GridFunction &pos,
                                     GridFunction *sub_pos,
                                     GridFunction &vel,
                                     GridFunction &sub_vel,
                                     Assembly &_asmbl,
                                     LowOrderMethod &_lom,
                                     DofInfo &_dofs,
                                     HOSolver *hos,
                                     LOSolver *los,
                                     FCTSolver *fct,
                                     MonolithicSolver *mos) :
   TimeDependentOperator(size), Mbf(Mbf_), ml(_ml), Kbf(Kbf_),
   M_HO(M_HO_), K_HO(K_HO_),
   lumpedM(_lumpedM),
   start_mesh_pos(pos.Size()), start_submesh_pos(sub_vel.Size()),
   mesh_pos(pos), submesh_pos(sub_pos),
   mesh_vel(vel), submesh_vel(sub_vel),
   x_gf(Kbf.ParFESpace()),
   asmbl(_asmbl), lom(_lom), dofs(_dofs),
   ho_solver(hos), lo_solver(los), fct_solver(fct), mono_solver(mos)
{
   dbg();
   assert(!lo_solver);
   assert(!fct_solver);
   assert(!mono_solver);
}

void AdvectionOperator::Mult(const Vector &X, Vector &Y) const
{
   dbg();
   if (exec_mode == 1)
   {
      assert(false);
      // Move the mesh positions.
      const double t = GetTime();
      add(start_mesh_pos, t, mesh_vel, mesh_pos);
      if (submesh_pos)
      {
         add(start_submesh_pos, t, submesh_vel, *submesh_pos);
      }
      // Reset precomputed geometric data.
      Mbf.FESpace()->GetMesh()->DeleteGeometricFactors();

      // Reassemble on the new mesh. Element contributions.
      // Currently needed to have the sparse matrices used by the LO methods.
      Mbf.BilinearForm::operator=(0.0);
      Mbf.Assemble();
      Kbf.BilinearForm::operator=(0.0);
      Kbf.Assemble(0);
      ml.BilinearForm::operator=(0.0);
      ml.Assemble();
      lumpedM.HostReadWrite();
      ml.SpMat().GetDiag(lumpedM);

      M_HO.BilinearForm::operator=(0.0);
      M_HO.Assemble();
      K_HO.BilinearForm::operator=(0.0);
      K_HO.Assemble(0);

      if (lom.pk)
      {
         lom.pk->BilinearForm::operator=(0.0);
         lom.pk->Assemble();
      }

      // Face contributions.
      asmbl.bdrInt = 0.;
      Mesh *mesh = M_HO.FESpace()->GetMesh();
      const int dim = mesh->Dimension(), ne = mesh->GetNE();
      Array<int> bdrs, orientation;
      FaceElementTransformations *Trans;

      for (int k = 0; k < ne; k++)
      {
         if (dim == 1)      { mesh->GetElementVertices(k, bdrs); }
         else if (dim == 2) { mesh->GetElementEdges(k, bdrs, orientation); }
         else if (dim == 3) { mesh->GetElementFaces(k, bdrs, orientation); }

         for (int i = 0; i < dofs.numBdrs; i++)
         {
            Trans = mesh->GetFaceElementTransformations(bdrs[i]);
            asmbl.ComputeFluxTerms(k, i, Trans, lom);
         }
      }
   }

   const int size = Kbf.ParFESpace()->GetVSize();
   const int NE   = Kbf.ParFESpace()->GetNE();

   // Needed because X and Y are allocated on the host by the ODESolver.
   X.Read(); Y.Read();

   Vector u, d_u;
   Vector* xptr = const_cast<Vector*>(&X);
   u.MakeRef(*xptr, 0, size);
   d_u.MakeRef(Y, 0, size);
   Vector du_HO(u.Size()), du_LO(u.Size());

   dbg("x_gf");
   x_gf = u;
   dbg("ExchangeFaceNbrData");
   x_gf.ExchangeFaceNbrData();
   dbg("done");

   if (mono_solver)
   {
      assert(false);
      mono_solver->CalcSolution(u, d_u);
   }
   else if (fct_solver)
   {
      assert(false);
      MFEM_VERIFY(ho_solver && lo_solver, "FCT requires HO and LO solvers.");

      lo_solver->CalcLOSolution(u, du_LO);
      ho_solver->CalcHOSolution(u, du_HO);

      dofs.ComputeElementsMinMax(u, dofs.xe_min, dofs.xe_max, NULL, NULL);
      dofs.ComputeBounds(dofs.xe_min, dofs.xe_max, dofs.xi_min, dofs.xi_max);
      fct_solver->CalcFCTSolution(x_gf, lumpedM, du_HO, du_LO,
                                  dofs.xi_min, dofs.xi_max, d_u);
   }
   else if (lo_solver) { assert(false); lo_solver->CalcLOSolution(u, d_u); }
   else if (ho_solver)
   {
      dbg("CalcHOSolution");
      ho_solver->CalcHOSolution(u, d_u);
   }
   else { MFEM_ABORT("No solver was chosen."); }

   d_u.SyncAliasMemory(Y);

   // Remap the product field, if there is a product field.
   if (X.Size() > size)
   {
      assert(false);
      Vector us, d_us;
      us.MakeRef(*xptr, size, size);
      d_us.MakeRef(Y, size, size);

      x_gf = us;
      x_gf.ExchangeFaceNbrData();

      if (mono_solver) { mono_solver->CalcSolution(us, d_us); }
      else if (fct_solver)
      {
         MFEM_VERIFY(ho_solver && lo_solver, "FCT requires HO and LO solvers.");

         Vector d_us_HO(us.Size()), d_us_LO(us.Size());
         lo_solver->CalcLOSolution(us, d_us_LO);
         ho_solver->CalcHOSolution(us, d_us_HO);

         // Compute the ratio s = us_old / u_old, and old active dofs.
         Vector s(size);
         Array<bool> s_bool_el, s_bool_dofs;
         ComputeRatio(NE, us, u, s, s_bool_el, s_bool_dofs);
#ifdef REMHOS_FCT_DEBUG
         ComputeMinMaxS(s, s_bool_dofs, x_gf.ParFESpace()->GetMyRank());
#endif

         // Bounds for s, based on the old values (and old active dofs).
         // This doesn't consider s values from the old inactive dofs, because
         // there were no bounds restriction on them at the previous time step.
         dofs.ComputeElementsMinMax(s, dofs.xe_min, dofs.xe_max,
                                    &s_bool_el, &s_bool_dofs);
         dofs.ComputeBounds(dofs.xe_min, dofs.xe_max,
                            dofs.xi_min, dofs.xi_max, &s_bool_el);

         // Evolve u and get the new active dofs.
         Vector u_new(size);
         add(1.0, u, dt, d_u, u_new);
         Array<bool> s_bool_el_new, s_bool_dofs_new;
         ComputeBoolIndicators(NE, u_new, s_bool_el_new, s_bool_dofs_new);

         fct_solver->CalcFCTProduct(x_gf, lumpedM, d_us_HO, d_us_LO,
                                    dofs.xi_min, dofs.xi_max,
                                    u_new,
                                    s_bool_el_new, s_bool_dofs_new, d_us);

#ifdef REMHOS_FCT_DEBUG
         Vector us_new(size);
         add(1.0, us, dt, d_us, us_new);
         int myid = x_gf.ParFESpace()->GetMyRank();
         ComputeMinMaxS(NE, us_new, u_new, myid);
         if (myid == 0) { std::cout << " --- " << std::endl; }
#endif
      }
      else if (lo_solver) { lo_solver->CalcLOSolution(us, d_us); }
      else if (ho_solver) { ho_solver->CalcHOSolution(us, d_us); }
      else { MFEM_ABORT("No solver was chosen."); }

      d_us.SyncAliasMemory(Y);
   }
}

void AdvectionOperator::AMRUpdate(const Vector &S,
                                  ParGridFunction &u,
                                  const double mass0_u)
{
   dbg("AdvectionOperator Update");
   const int skip_zeros = 0;

   // TimeDependentOperator
   width = height = S.Size();

   dbg("Mbf");
   Mbf.FESpace()->Update();
   Mbf.Update();
   Mbf.BilinearForm::operator=(0.0);
   Mbf.Assemble();
   Mbf.Finalize();

   dbg("ml");
   ml.FESpace()->Update();
   ml.Update();
   ml.BilinearForm::operator=(0.0);
   ml.Assemble();
   ml.Finalize();

   dbg("lumpedM");
   ml.SpMat().GetDiag(lumpedM);
   {
      MPI_Comm comm = u.ParFESpace()->GetParMesh()->GetComm();
      Vector masses(lumpedM);
      double mass_u, mass_u_loc = masses * u;
      MPI_Allreduce(&mass_u_loc, &mass_u, 1, MPI_DOUBLE, MPI_SUM, comm);
      std::cout << setprecision(10)
                << "AdvectionOperator, u size:" << u.Size() << std::endl
                << "Current mass u: " << mass_u << std::endl
                << "   Mass loss u: " << abs(mass0_u - mass_u) << std::endl;
      MFEM_VERIFY(abs(mass0_u - mass_u) < 1e-11, "Error in mass!");
   }

   dbg("Kbf");
   Kbf.ParFESpace()->Update();
   Kbf.Update();
   Kbf.Assemble(skip_zeros);
   Kbf.Finalize(skip_zeros);

   dbg("M_HO");
   M_HO.ParFESpace()->Update();
   M_HO.Update();
   M_HO.Assemble();
   M_HO.Finalize();

   dbg("K_HO");
   K_HO.ParFESpace()->Update();
   K_HO.Update();
   K_HO.KeepNbrBlock(true);
   K_HO.Assemble(skip_zeros);
   K_HO.Finalize(skip_zeros);

   // order 1: subcell_mesh = &pmesh;
   if (lom.subcell_scheme)
   {
      dbg("subcell_scheme");
      mesh_pos.Update();
      if (submesh_pos) { submesh_pos->Update(); }
      mesh_vel.Update();
      submesh_vel.Update();
   }
   else { dbg("!subcell_scheme"); }

   dbg("x_gf");
   //x_gf.Update(); // x_gf = u in adv.Mult

   dbg("asmbl");
   asmbl.Update();

   dbg("lom?");
   if (lom.SubFes0) { lom.SubFes0->Update(); }
   if (lom.SubFes1) { lom.SubFes1->Update(); }

   dbg("dofs?");
   dofs.Update();

   dbg("ho_solver");
   assert(ho_solver);
   if (ho_solver) { ho_solver->Update(); }

   //dbg("lo_solver?");
   assert(!lo_solver);
   //lo_solver update ?

   //dbg("fct_solver?");
   assert(!fct_solver);
   //if (fct_solver) fct_solver->UpdateTimeStep(dt);

   //dbg("mono_solver?");
   assert(!mono_solver);
}

} // namespace mfem
