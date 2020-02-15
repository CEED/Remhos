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

#include "remhos_tools.hpp"

using namespace std;

namespace mfem
{


SmoothnessIndicator::SmoothnessIndicator(int type_id,
                                         ParFiniteElementSpace &pfes_CG_sub_,
                                         ParFiniteElementSpace &pfes_DG_,
                                         ParGridFunction &u,
                                         DofInfo &dof_info)
   : type(type_id), param(type == 1 ? 5.0 : 3.0),
     pfes_CG_sub(pfes_CG_sub_), pfes_DG(pfes_DG_)
{
   // TODO assemble SI matrices every RK stage for remap.

   MFEM_VERIFY(type_id == 1 || type_id == 2, "Bad smoothness indicator id!");

   BilinearForm massH1(&pfes_CG_sub);
   massH1.AddDomainIntegrator(new MassIntegrator);
   massH1.Assemble();
   massH1.Finalize();
   Mmat = massH1.SpMat();

   ConstantCoefficient neg_one(-1.0);
   BilinearForm lap(&pfes_CG_sub);
   lap.AddDomainIntegrator(new DiffusionIntegrator(neg_one));
   lap.AddBdrFaceIntegrator(new DGDiffusionIntegrator(neg_one, 0., 0.));
   lap.Assemble();
   lap.Finalize();
   LaplaceOp = lap.SpMat();

   MassMixed = new SparseMatrix(pfes_CG_sub.GetVSize(), pfes_DG.GetVSize());
   MassInt = new MassIntegrator;

   BilinearForm mlH1(&pfes_CG_sub);
   mlH1.AddDomainIntegrator(new LumpedIntegrator(new MassIntegrator));
   mlH1.Assemble();
   mlH1.Finalize();
   mlH1.SpMat().GetDiag(lumpedMH1);

   Vector lumped_hv(pfes_CG_sub.GetTrueVSize());
   pfes_CG_sub.Dof_TrueDof_Matrix()->MultTranspose(lumpedMH1, lumped_hv);
   pfes_CG_sub.GetProlongationMatrix()->Mult(lumped_hv, lumpedMH1);

   // Stores the index for the dof of H1-conforming for each node.
   // If the node is on the boundary, the entry is -1.
   const int dim = pfes_DG.GetMesh()->Dimension(),
             nd = pfes_DG.GetFE(0)->GetDof(),
             ne = pfes_DG.GetMesh()->GetNE();
   DG2CG.SetSize(ne*nd);
   Array<int> vdofs;
   int e, i, j, e_id;

   for (e = 0; e < ne; e++)
   {
      for (i = 0; i < dof_info.numSubcells; i++)
      {
         e_id = e*dof_info.numSubcells + i;
         pfes_CG_sub.GetElementVDofs(e_id, vdofs);

         // Switchero - numbering CG vs DG.
         DG2CG(e*nd + dof_info.Sub2Ind(i, 0)) = vdofs[0];
         DG2CG(e*nd + dof_info.Sub2Ind(i, 1)) = vdofs[1];
         if (dim > 1)
         {
            DG2CG(e*nd + dof_info.Sub2Ind(i, 2)) = vdofs[3];
            DG2CG(e*nd + dof_info.Sub2Ind(i, 3)) = vdofs[2];
         }
         if (dim == 3)
         {
            DG2CG(e*nd + dof_info.Sub2Ind(i, 4)) = vdofs[4];
            DG2CG(e*nd + dof_info.Sub2Ind(i, 5)) = vdofs[5];
            DG2CG(e*nd + dof_info.Sub2Ind(i, 6)) = vdofs[7];
            DG2CG(e*nd + dof_info.Sub2Ind(i, 7)) = vdofs[6];
         }
      }

      // Domain boundaries.
      for (i = 0; i < dof_info.numBdrs; i++)
      {
         for (j = 0; j < dof_info.numFaceDofs; j++)
         {
            if (dof_info.NbrDof(e,i,j) < 0)
            {
               DG2CG(e*nd+dof_info.BdrDofs(j,i)) = -1;
            }
         }
      }
   }

   ComputeVariationalMatrix(dof_info);

   IntegrationRule *ir;
   IntegrationRule irX;
   QuadratureFunctions1D qf;
   qf.ClosedUniform(pfes_DG.GetFE(0)->GetOrder() + 1, &irX);
   Vector shape(nd);

   if (dim == 1) { ir = &irX; }
   if (dim == 2) { ir = new IntegrationRule(irX, irX); }
   if (dim == 3) { ir = new IntegrationRule(irX, irX, irX); }

   ShapeEval.SetSize(nd, nd); // nd equals ir->GetNPoints().
   for (i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      pfes_DG.GetFE(0)->CalcShape(ip, shape);
      ShapeEval.SetCol(i, shape);
   }
   ShapeEval.Transpose();

   // Print the values of the smoothness indicator.
   ParGridFunction si_val;
   ComputeSmoothnessIndicator(u, si_val);
   {
      ofstream smth("si_init.gf");
      smth.precision(8);
      si_val.SaveAsOne(smth);
   }

   if (dim > 1) { delete ir; }
}

SmoothnessIndicator::~SmoothnessIndicator()
{
   delete MassInt;
   delete MassMixed;
}

void SmoothnessIndicator::ComputeSmoothnessIndicator(const Vector &u,
                                                     ParGridFunction &si_vals_u)
{
   si_vals_u.SetSpace(&pfes_CG_sub);

   ParGridFunction g(&pfes_CG_sub);
   const int N = g.Size();
   Vector gmin(N), gmax(N);

   ApproximateLaplacian(u, g);
   ComputeFromSparsity(Mmat, g, gmin, gmax);

   if (type == 1)
   {
      const double eps = 1.0e-50;
      for (int e = 0; e < N; e++)
      {
         si_vals_u(e) = 1.0 - pow( (abs(gmin(e) - gmax(e)) + eps) /
                                   (abs(gmin(e)) + abs(gmax(e)) + eps), param);
      }
   }
   else if (type == 2)
   {
      const double eps = 1.0e-15;
      for (int e = 0; e < N; e++)
      {
         si_vals_u(e) = min(1.0, param *
                                 max(0., gmin(e)*gmax(e)) /
                                 (max(gmin(e)*gmin(e),gmax(e)*gmax(e)) + eps) );
      }
   }
}

void SmoothnessIndicator::ComputeVariationalMatrix(DofInfo &dof_info)
{
   Mesh *subcell_mesh = pfes_CG_sub.GetMesh();

   const int dim = subcell_mesh->Dimension();
   const int nd = pfes_DG.GetFE(0)->GetDof(), ne = pfes_DG.GetMesh()->GetNE();

   int k, m, e_id;
   DenseMatrix elmat1;

   Array <int> te_vdofs, tr_vdofs;

   tr_vdofs.SetSize(dof_info.numDofsSubcell);

   for (k = 0; k < ne; k++)
   {
      for (m = 0; m < dof_info.numSubcells; m++)
      {
         e_id = k*dof_info.numSubcells + m;
         const FiniteElement *el = pfes_CG_sub.GetFE(e_id);
         ElementTransformation *tr = subcell_mesh->GetElementTransformation(e_id);
         MassInt->AssembleElementMatrix(*el, *tr, elmat1);
         pfes_CG_sub.GetElementVDofs(e_id, te_vdofs);

         // Switchero - numbering CG vs DG.
         tr_vdofs[0] = k*nd + dof_info.Sub2Ind(m, 0);
         tr_vdofs[1] = k*nd + dof_info.Sub2Ind(m, 1);
         if (dim > 1)
         {
            tr_vdofs[2] = k*nd + dof_info.Sub2Ind(m, 3);
            tr_vdofs[3] = k*nd + dof_info.Sub2Ind(m, 2);
         }
         if (dim == 3)
         {
            tr_vdofs[4] = k*nd + dof_info.Sub2Ind(m, 4);
            tr_vdofs[5] = k*nd + dof_info.Sub2Ind(m, 5);
            tr_vdofs[6] = k*nd + dof_info.Sub2Ind(m, 7);
            tr_vdofs[7] = k*nd + dof_info.Sub2Ind(m, 6);
         }
         MassMixed->AddSubMatrix(te_vdofs, tr_vdofs, elmat1);
      }
   }
   MassMixed->Finalize();
}

void SmoothnessIndicator::ApproximateLaplacian(const Vector &x,
                                               ParGridFunction &y)
{
   const int nd = pfes_DG.GetFE(0)->GetDof(), ne = pfes_DG.GetMesh()->GetNE();
   int k, i, j, N = lumpedMH1.Size();
   Array<int> eldofs;
   Vector xDofs(nd), tmp(nd), xEval(ne*nd);
   Vector rhs_tv(pfes_CG_sub.GetTrueVSize()), z_tv(pfes_CG_sub.GetTrueVSize());

   eldofs.SetSize(nd);
   y.SetSize(N);
   Vector rhs(N), z(N);

   // Approximate inversion, corresponding to Neumann series truncated after
   // first two summands.
   int iter, max_iter = 2;
   const double abs_tol = 1.e-10;
   double resid;

   for (k = 0; k < ne; k++)
   {
      for (j = 0; j < nd; j++) { eldofs[j] = k*nd + j; }

      x.GetSubVector(eldofs, xDofs);
      ShapeEval.Mult(xDofs, tmp);
      xEval.SetSubVector(eldofs, tmp);
   }

   MassMixed->Mult(xEval, rhs);
   pfes_CG_sub.Dof_TrueDof_Matrix()->MultTranspose(rhs, rhs_tv);
   pfes_CG_sub.GetProlongationMatrix()->Mult(rhs_tv, rhs);

   y = 0.;

   // Project x to a CG space (result is in y).
   for (iter = 1; iter <= max_iter; iter++)
   {
      Mmat.Mult(y, z);
      pfes_CG_sub.Dof_TrueDof_Matrix()->MultTranspose(z, z_tv);
      z_tv -= rhs_tv;
      pfes_CG_sub.GetProlongationMatrix()->Mult(z_tv, z);

      double loc_res = z_tv.Norml2();
      loc_res *= loc_res;
      MPI_Allreduce(&loc_res, &resid, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      resid = sqrt(resid);

      if (resid <= abs_tol) { break; }

      for (i = 0; i < N; i++)
      {
         y(i) -= z(i) / lumpedMH1(i);
      }
   }

   LaplaceOp.Mult(y, rhs);
   pfes_CG_sub.Dof_TrueDof_Matrix()->MultTranspose(rhs, rhs_tv);
   pfes_CG_sub.GetProlongationMatrix()->Mult(rhs_tv, rhs);

   y = 0.;

   for (iter = 1; iter <= max_iter; iter++)
   {
      Mmat.Mult(y, z);
      pfes_CG_sub.Dof_TrueDof_Matrix()->MultTranspose(z, z_tv);
      z_tv -= rhs_tv;
      pfes_CG_sub.GetProlongationMatrix()->Mult(z_tv, z);

      double loc_res = z_tv.Norml2();
      loc_res *= loc_res;
      MPI_Allreduce(&loc_res, &resid, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      resid = sqrt(resid);

      if (resid <= abs_tol) { break; }

      for (i = 0; i < N; i++)
      {
         y(i) -= z(i) / lumpedMH1(i);
      }
   }
}

void SmoothnessIndicator::ComputeFromSparsity(const SparseMatrix &K,
                                              const ParGridFunction &x,
                                              Vector &x_min, Vector &x_max)
{
   const int *I = K.GetI(), *J = K.GetJ(), loc_size = K.Size();
   int end;

   for (int i = 0, k = 0; i < loc_size; i++)
   {
      x_min(i) = numeric_limits<double>::infinity();
      x_max(i) = -x_min(i);
      for (end = I[i+1]; k < end; k++)
      {
         const double x_j = x(J[k]);
         x_max(i) = max(x_max(i), x_j);
         x_min(i) = min(x_min(i), x_j);
      }
   }

   GroupCommunicator &gcomm = x.ParFESpace()->GroupComm();
   Array<double> minvals(x_min.GetData(), x_min.Size()),
         maxvals(x_max.GetData(), x_max.Size());
   gcomm.Reduce<double>(minvals, GroupCommunicator::Min);
   gcomm.Bcast(minvals);
   gcomm.Reduce<double>(maxvals, GroupCommunicator::Max);
   gcomm.Bcast(maxvals);
}


int GetLocalFaceDofIndex3D(int loc_face_id, int face_orient,
                           int face_dof_id, int face_dof1D_cnt)
{
   int k1, k2;
   const int kf1 = face_dof_id % face_dof1D_cnt;
   const int kf2 = face_dof_id / face_dof1D_cnt;
   switch (loc_face_id)
   {
      case 0://BOTTOM
      switch (face_orient)
      {
      case 0://{0, 1, 2, 3}
         k1 = kf1;
         k2 = face_dof1D_cnt-1-kf2;
         break;
      case 1://{0, 3, 2, 1}
         k1 = face_dof1D_cnt-1-kf2;
         k2 = kf1;
         break;
      case 2://{1, 2, 3, 0}
         k1 = face_dof1D_cnt-1-kf2;
         k2 = face_dof1D_cnt-1-kf1;
         break;
      case 3://{1, 0, 3, 2}
         k1 = face_dof1D_cnt-1-kf1;
         k2 = face_dof1D_cnt-1-kf2;
         break;
      case 4://{2, 3, 0, 1}
         k1 = face_dof1D_cnt-1-kf1;
         k2 = kf2;
         break;
      case 5://{2, 1, 0, 3}
         k1 = kf2;
         k2 = face_dof1D_cnt-1-kf1;
         break;
      case 6://{3, 0, 1, 2}
         k1 = kf2;
         k2 = kf1;
         break;
      case 7://{3, 2, 1, 0}
         k1 = kf1;
         k2 = kf2;
         break;
      default:
         mfem_error("This orientation does not exist in 3D");
         break;
      }
      break;
   case 1://SOUTH
      switch (face_orient)
      {
      case 0://{0, 1, 2, 3}
         k1 = kf1;
         k2 = kf2;
         break;
      case 1://{0, 3, 2, 1}
         k1 = kf2;
         k2 = kf1;
         break;
      case 2://{1, 2, 3, 0}
         k1 = kf2;
         k2 = face_dof1D_cnt-1-kf1;
         break;
      case 3://{1, 0, 3, 2}
         k1 = face_dof1D_cnt-1-kf1;
         k2 = kf2;
         break;
      case 4://{2, 3, 0, 1}
         k1 = face_dof1D_cnt-1-kf1;
         k2 = face_dof1D_cnt-1-kf2;
         break;
      case 5://{2, 1, 0, 3}
         k1 = face_dof1D_cnt-1-kf2;
         k2 = face_dof1D_cnt-1-kf1;
         break;
      case 6://{3, 0, 1, 2}
         k1 = face_dof1D_cnt-1-kf2;
         k2 = kf1;
         break;
      case 7://{3, 2, 1, 0}
         k1 = kf1;
         k2 = face_dof1D_cnt-1-kf2;
         break;
      default:
         mfem_error("This orientation does not exist in 3D");
         break;
      }
      break;
   case 2://EAST
      switch (face_orient)
      {
      case 0://{0, 1, 2, 3}
         k1 = kf1;
         k2 = kf2;
         break;
      case 1://{0, 3, 2, 1}
         k1 = kf2;
         k2 = kf1;
         break;
      case 2://{1, 2, 3, 0}
         k1 = kf2;
         k2 = face_dof1D_cnt-1-kf1;
         break;
      case 3://{1, 0, 3, 2}
         k1 = face_dof1D_cnt-1-kf1;
         k2 = kf2;
         break;
      case 4://{2, 3, 0, 1}
         k1 = face_dof1D_cnt-1-kf1;
         k2 = face_dof1D_cnt-1-kf2;
         break;
      case 5://{2, 1, 0, 3}
         k1 = face_dof1D_cnt-1-kf2;
         k2 = face_dof1D_cnt-1-kf1;
         break;
      case 6://{3, 0, 1, 2}
         k1 = face_dof1D_cnt-1-kf2;
         k2 = kf1;
         break;
      case 7://{3, 2, 1, 0}
         k1 = kf1;
         k2 = face_dof1D_cnt-1-kf2;
         break;
      default:
         mfem_error("This orientation does not exist in 3D");
         break;
      }
      break;
   case 3://NORTH
      switch (face_orient)
      {
      case 0://{0, 1, 2, 3}
         k1 = face_dof1D_cnt-1-kf1;
         k2 = kf2;
         break;
      case 1://{0, 3, 2, 1}
         k1 = kf2;
         k2 = face_dof1D_cnt-1-kf1;
         break;
      case 2://{1, 2, 3, 0}
         k1 = kf2;
         k2 = kf1;
         break;
      case 3://{1, 0, 3, 2}
         k1 = kf1;
         k2 = kf2;
         break;
      case 4://{2, 3, 0, 1}
         k1 = kf1;
         k2 = face_dof1D_cnt-1-kf2;
         break;
      case 5://{2, 1, 0, 3}
         k1 = face_dof1D_cnt-1-kf2;
         k2 = kf1;
         break;
      case 6://{3, 0, 1, 2}
         k1 = face_dof1D_cnt-1-kf2;
         k2 = face_dof1D_cnt-1-kf1;
         break;
      case 7://{3, 2, 1, 0}
         k1 = face_dof1D_cnt-1-kf1;
         k2 = face_dof1D_cnt-1-kf2;
         break;
      default:
         mfem_error("This orientation does not exist in 3D");
         break;
      }
      break;
   case 4://WEST
      switch (face_orient)
      {
      case 0://{0, 1, 2, 3}
         k1 = face_dof1D_cnt-1-kf1;
         k2 = kf2;
         break;
      case 1://{0, 3, 2, 1}
         k1 = kf2;
         k2 = face_dof1D_cnt-1-kf1;
         break;
      case 2://{1, 2, 3, 0}
         k1 = kf2;
         k2 = kf1;
         break;
      case 3://{1, 0, 3, 2}
         k1 = kf1;
         k2 = kf2;
         break;
      case 4://{2, 3, 0, 1}
         k1 = kf1;
         k2 = face_dof1D_cnt-1-kf2;
         break;
      case 5://{2, 1, 0, 3}
         k1 = face_dof1D_cnt-1-kf2;
         k2 = kf1;
         break;
      case 6://{3, 0, 1, 2}
         k1 = face_dof1D_cnt-1-kf2;
         k2 = face_dof1D_cnt-1-kf1;
         break;
      case 7://{3, 2, 1, 0}
         k1 = face_dof1D_cnt-1-kf1;
         k2 = face_dof1D_cnt-1-kf2;
         break;
      default:
         mfem_error("This orientation does not exist in 3D");
         break;
      }
      break;
   case 5://TOP
      switch (face_orient)
      {
      case 0://{0, 1, 2, 3}
         k1 = kf1;
         k2 = kf2;
         break;
      case 1://{0, 3, 2, 1}
         k1 = kf2;
         k2 = kf1;
         break;
      case 2://{1, 2, 3, 0}
         k1 = kf2;
         k2 = face_dof1D_cnt-1-kf1;
         break;
      case 3://{1, 0, 3, 2}
         k1 = face_dof1D_cnt-1-kf1;
         k2 = kf2;
         break;
      case 4://{2, 3, 0, 1}
         k1 = face_dof1D_cnt-1-kf1;
         k2 = face_dof1D_cnt-1-kf2;
         break;
      case 5://{2, 1, 0, 3}
         k1 = face_dof1D_cnt-1-kf2;
         k2 = face_dof1D_cnt-1-kf1;
         break;
      case 6://{3, 0, 1, 2}
         k1 = face_dof1D_cnt-1-kf2;
         k2 = kf1;
         break;
      case 7://{3, 2, 1, 0}
         k1 = kf1;
         k2 = face_dof1D_cnt-1-kf2;
         break;
      default:
         mfem_error("This orientation does not exist in 3D");
         break;
      }
      break;
   default: MFEM_ABORT("This face_id does not exist in 3D");
   }
   return k1 + face_dof1D_cnt * k2;
}

int GetLocalFaceDofIndex(int dim, int loc_face_id, int face_orient,
                         int face_dof_id, int face_dof1D_cnt)
{
   switch (dim)
   {
      case 1: return face_dof_id;
      case 2:
         if (loc_face_id <= 1)
         {
            // SOUTH or EAST (canonical ordering)
            return face_dof_id;
         }
         else
         {
            // NORTH or WEST (counter-canonical ordering)
            return face_dof1D_cnt - 1 - face_dof_id;
         }
      case 3: return GetLocalFaceDofIndex3D(loc_face_id, face_orient,
                                            face_dof_id, face_dof1D_cnt);
      default: MFEM_ABORT("Dimension too high!"); return 0;
   }
}


// Assuming L2 elements.
void ExtractBdrDofs(int p, Geometry::Type gtype, DenseMatrix &dofs)
{
   switch (gtype)
   {  
      case Geometry::SEGMENT:
      {
         dofs.SetSize(1, 2);
         dofs(0, 0) = 0;
         dofs(0, 1) = p;
         break;
      }
      case Geometry::SQUARE:
      {
         dofs.SetSize(p+1, 4);
         for (int i = 0; i <= p; i++)
         {
            dofs(i,0) = i;
            dofs(i,1) = i*(p+1) + p;
            dofs(i,2) = (p+1)*(p+1) - 1 - i;
            dofs(i,3) = (p-i)*(p+1);
         }
         break;
      }
      case Geometry::CUBE:
      {
         dofs.SetSize((p+1)*(p+1), 6);
         for (int bdrID = 0; bdrID < 6; bdrID++)
         {
            int o(0);
            switch (bdrID)
            {
            case 0:
               for (int i = 0; i < (p+1)*(p+1); i++)
               {
                  dofs(o++,bdrID) = i;
               }
               break;
            case 1:
               for (int i = 0; i <= p*(p+1)*(p+1); i+=(p+1)*(p+1))
                  for (int j = 0; j < p+1; j++)
                  {
                     dofs(o++,bdrID) = i+j;
                  }
               break;
            case 2:
               for (int i = p; i < (p+1)*(p+1)*(p+1); i+=p+1)
               {
                  dofs(o++,bdrID) = i;
               }
               break;
            case 3:
               for (int i = 0; i <= p*(p+1)*(p+1); i+=(p+1)*(p+1))
                  for (int j = p*(p+1); j < (p+1)*(p+1); j++)
                  {
                     dofs(o++,bdrID) = i+j;
                  }
               break;
            case 4:
               for (int i = 0; i <= (p+1)*((p+1)*(p+1)-1); i+=p+1)
               {
                  dofs(o++,bdrID) = i;
               }
               break;
            case 5:
               for (int i = p*(p+1)*(p+1); i < (p+1)*(p+1)*(p+1); i++)
               {
                  dofs(o++,bdrID) = i;
               }
               break;
            }
         }
         break;
      }
      default: MFEM_ABORT("Geometry not implemented.");
   }
}

} // namespace mfem
