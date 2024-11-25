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
#include "remhos_tools.hpp"

using namespace std;

namespace mfem
{

CGHOSolver::CGHOSolver(ParFiniteElementSpace &space,
                       ParBilinearForm &Mbf, ParBilinearForm &Kbf)
   : HOSolver(space), M(Mbf), K(Kbf)
{ }

void CGHOSolver::CalcHOSolution(const Vector &u, Vector &du) const
{
   Vector rhs(u.Size());
   du = 0.0;

   // Invert by preconditioned CG.
   CGSolver M_solver(pfes.GetComm());
   HypreParMatrix *M_mat = NULL, *K_mat = NULL;
   Solver *M_prec;
   Array<int> ess_tdof_list;
   if (M.GetAssemblyLevel() == AssemblyLevel::PARTIAL)
   {
      K.Mult(u, rhs);

      M_solver.SetOperator(M);
      M_prec = new OperatorJacobiSmoother(M, ess_tdof_list);
   }
   else
   {
      K.SpMat().HostReadWriteI();
      K.SpMat().HostReadWriteJ();
      K.SpMat().HostReadData();
      K_mat = K.ParallelAssemble(&K.SpMat());
      K_mat->Mult(u, rhs);

      M_mat = M.ParallelAssemble();
      M_solver.SetOperator(*M_mat);
      M_prec = new HypreSmoother(*M_mat, HypreSmoother::Jacobi);
   }
   M_solver.SetPreconditioner(*M_prec);
   M_solver.SetRelTol(1e-12);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(500);
   M_solver.SetPrintLevel(0);

   M_solver.Mult(rhs, du);

   delete M_prec;
   delete M_mat;
   delete K_mat;
}

LocalInverseHOSolver::LocalInverseHOSolver(ParFiniteElementSpace &space,
                                           ParBilinearForm &Mbf,
                                           ParBilinearForm &Kbf)
   : HOSolver(space), M(Mbf), K(Kbf)
{
   if (M.GetAssemblyLevel() == AssemblyLevel::PARTIAL)
   {
      M_inv = new DGMassInverse(space, BasisType::GaussLegendre);
      M_inv->SetAbsTol(1e-8), M_inv->SetRelTol(0.0);
   }
}

void LocalInverseHOSolver::CalcHOSolution(const Vector &u, Vector &du) const
{
   MFEM_VERIFY(timer, "Timer not set.");

   Vector rhs(u.Size());

   if (M.GetAssemblyLevel() != AssemblyLevel::PARTIAL)
   {
      timer->sw_rhs.Start();
      K.SpMat().HostReadWriteI();
      K.SpMat().HostReadWriteJ();
      K.SpMat().HostReadWriteData();
      HypreParMatrix *K_mat = K.ParallelAssemble(&K.SpMat());
      K_mat->Mult(u, rhs);
      timer->sw_rhs.Stop();

      const int ne = pfes.GetMesh()->GetNE();
      const int nd = pfes.GetFE(0)->GetDof();
      DenseMatrix M_loc(nd);
      DenseMatrixInverse M_loc_inv(&M_loc);
      Vector rhs_loc(nd), du_loc(nd);
      Array<int> dofs;
      for (int i = 0; i < ne; i++)
      {
         pfes.GetElementDofs(i, dofs);
         rhs.GetSubVector(dofs, rhs_loc);
         timer->sw_L2inv.Start();
         M.SpMat().GetSubMatrix(dofs, dofs, M_loc);
         M_loc_inv.Factor();
         M_loc_inv.Mult(rhs_loc, du_loc);
         timer->sw_L2inv.Stop();
         du.SetSubVector(dofs, du_loc);
      }
      delete K_mat;
   }
   else
   {
      timer->sw_rhs.Start();
      K.Mult(u, rhs);
      timer->sw_rhs.Stop();

      timer->sw_L2inv.Start();
      M_inv->Update(), M_inv->Mult(rhs, du);
      timer->sw_L2inv.Stop();
   }

}

NeumannHOSolver::NeumannHOSolver(ParFiniteElementSpace &space,
                                 ParBilinearForm &Mbf, ParBilinearForm &Kbf,
                                 Vector &Mlump, Assembly &a)
   : HOSolver(space), M(Mbf), K(Kbf), M_lumped(Mlump), assembly(a) { }

void NeumannHOSolver::CalcHOSolution(const Vector &u, Vector &du) const
{
   MFEM_VERIFY(K.GetAssemblyLevel() != AssemblyLevel::PARTIAL,
               "PA for DG is not supported for Neummann Solver.");
   const int n = u.Size(), ne = pfes.GetNE(), ndof = pfes.GetFE(0)->GetDof();
   Vector rhs(n), res(n);
   Vector alpha(ndof); alpha = 1.0;

   // K multiplies a ldofs Vector, as we're always doing DG.
   K.Mult(u, rhs);

   // Face contributions.
   ParGridFunction u_gf(&pfes);
   u_gf = u;
   u_gf.ExchangeFaceNbrData();
   Vector &u_nd = u_gf.FaceNbrData();
   u.HostRead();
   rhs.HostReadWrite();
   u_nd.HostRead();
   for (int k = 0; k < ne; k++)
   {
      for (int i = 0; i < assembly.dofs.numBdrs; i++)
      {
         assembly.LinearFluxLumping(k, ndof, i, u, rhs, u_nd, alpha);
      }
   }

   // Neumann iteration.
   du = 0.0;
   const double abs_tol = 1.e-4;
   const int max_iter = 20;
   du.ReadWrite();
   M_lumped.HostRead();
   for (int iter = 1; iter <= max_iter; iter++)
   {
      M.Mult(du, res);
      res -= rhs;

      double resid_loc = res.Norml2(); resid_loc *= resid_loc;
      double resid;
      MPI_Allreduce(&resid_loc, &resid, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      resid = std::sqrt(resid);
      if (resid <= abs_tol) { return; }

      res.HostReadWrite();
      du.HostReadWrite();
      for (int i = 0; i < n; i++)
      {
         du(i) -= res(i) / M_lumped(i);
      }
   }
}

} // namespace mfem
