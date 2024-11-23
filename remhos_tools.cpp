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
                                         ParMesh &subcell_mesh,
                                         ParFiniteElementSpace &pfes_DG_,
                                         ParGridFunction &u,
                                         DofInfo &dof_info)
   : type(type_id), param(type == 1 ? 5.0 : 3.0),
     fec_sub(1, pfes_DG_.GetMesh()->Dimension(), BasisType::Positive),
     pfes_CG_sub(&subcell_mesh, &fec_sub),
     pfes_DG(pfes_DG_)
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

void SmoothnessIndicator::UpdateBounds(int dof_id, double u_HO,
                                       const ParGridFunction &si_vals,
                                       double &u_min, double &u_max)
{
   const double tmp = (DG2CG(dof_id) < 0.0) ? 1.0 : si_vals(DG2CG(dof_id));
   u_min = max(0.0, tmp * u_HO + (1.0 - tmp) * u_min);
   u_max = min(1.0, tmp * u_HO + (1.0 - tmp) * u_max);
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

      y.HostReadWrite();
      z.HostReadWrite();
      const double *h_lumpedMH1 = lumpedMH1.HostRead();
      for (i = 0; i < N; i++)
      {
         y(i) -= z(i) / h_lumpedMH1[i];
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

      y.HostReadWrite();
      z.HostReadWrite();
      const double * h_lumpedMH1 = lumpedMH1.HostRead();
      for (i = 0; i < N; i++)
      {
         y(i) -= z(i) / h_lumpedMH1[i];
      }
   }
}

void SmoothnessIndicator::ComputeFromSparsity(const SparseMatrix &K,
                                              const ParGridFunction &x,
                                              Vector &x_min, Vector &x_max)
{
   const int *I = K.GetI(), *J = K.GetJ(), loc_size = K.Size();
   int end;

   x_min.HostReadWrite();
   x_max.HostReadWrite();

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
   Array<double> minvals(x_min.HostReadWrite(), x_min.Size()),
         maxvals(x_max.HostReadWrite(), x_max.Size());
   gcomm.Reduce<double>(minvals, GroupCommunicator::Min);
   gcomm.Bcast(minvals);
   gcomm.Reduce<double>(maxvals, GroupCommunicator::Max);
   gcomm.Bcast(maxvals);
}

DofInfo::DofInfo(ParFiniteElementSpace &pfes_sltn, int btype)
   : bounds_type(btype),
     pmesh(pfes_sltn.GetParMesh()), pfes(pfes_sltn),
     fec_bounds(std::max(pfes.GetOrder(0), 1),
                pmesh->Dimension(), BasisType::GaussLobatto),
     pfes_bounds(pmesh, &fec_bounds),
     x_min(&pfes_bounds), x_max(&pfes_bounds)
{
   int n = pfes.GetVSize();
   int ne = pmesh->GetNE();

   xi_min.SetSize(n);
   xi_max.SetSize(n);
   xe_min.SetSize(ne);
   xe_max.SetSize(ne);

   ExtractBdrDofs(pfes.GetOrder(0),
                  pfes.GetFE(0)->GetGeomType(), BdrDofs);
   numFaceDofs = BdrDofs.Height();
   numBdrs = BdrDofs.Width();

   FillNeighborDofs();    // Fill NbrDof.
   FillSubcell2CellDof(); // Fill Sub2Ind.
}

void DofInfo::ComputeMatrixSparsityBounds(const Vector &el_min,
                                          const Vector &el_max,
                                          Vector &dof_min, Vector &dof_max,
                                          Array<bool> *active_el)
{
   ParMesh *pmesh = pfes.GetParMesh();
   L2_FECollection fec_bounds(0, pmesh->Dimension());
   ParFiniteElementSpace pfes_bounds(pmesh, &fec_bounds);
   ParGridFunction x_min(&pfes_bounds), x_max(&pfes_bounds);
   const int NE = pmesh->GetNE();
   const int ndofs = dof_min.Size() / NE;

   el_min.HostRead(), el_max.HostRead();
   x_min.HostReadWrite(), x_max.HostReadWrite();

   x_min = el_min;
   x_max = el_max;

   x_min.ExchangeFaceNbrData(); x_max.ExchangeFaceNbrData();
   const Vector &minv = x_min.FaceNbrData(), &maxv = x_max.FaceNbrData();
   const Table &el_to_el = pmesh->ElementToElementTable();
   Array<int> face_nbr_el;
   for (int i = 0; i < NE; i++)
   {
      double el_min = x_min(i), el_max = x_max(i);

      el_to_el.GetRow(i, face_nbr_el);
      for (int n = 0; n < face_nbr_el.Size(); n++)
      {
         if (face_nbr_el[n] < NE)
         {
            // Local neighbor.
            el_min = std::min(el_min, x_min(face_nbr_el[n]));
            el_max = std::max(el_max, x_max(face_nbr_el[n]));
         }
         else
         {
            // MPI face neighbor.
            el_min = std::min(el_min, minv(face_nbr_el[n] - NE));
            el_max = std::max(el_max, maxv(face_nbr_el[n] - NE));
         }
      }

      for (int j = 0; j < ndofs; j++)
      {
         dof_min(i*ndofs + j) = el_min;
         dof_max(i*ndofs + j) = el_max;
      }
   }
}

void DofInfo::ComputeOverlapBounds(const Vector &el_min,
                                   const Vector &el_max,
                                   Vector &dof_min, Vector &dof_max,
                                   Array<bool> *active_el)
{
   GroupCommunicator &gcomm = pfes_bounds.GroupComm();
   Array<int> dofsCG;
   const int NE = pfes.GetNE();

   // Form min/max at each CG dof, considering element overlaps.
   x_min =   std::numeric_limits<double>::infinity();
   x_max = - std::numeric_limits<double>::infinity();

   el_min.HostRead(), el_max.HostRead();
   dof_min.HostReadWrite(), dof_max.HostReadWrite();
   x_min.HostReadWrite(), x_max.HostReadWrite();

   for (int i = 0; i < NE; i++)
   {
      // Inactive elements don't affect the bounds.
      if (active_el && (*active_el)[i] == false) { continue; }

      pfes_bounds.GetElementDofs(i, dofsCG);
      for (int j = 0; j < dofsCG.Size(); j++)
      {
         x_min(dofsCG[j]) = std::min(x_min(dofsCG[j]), el_min(i));
         x_max(dofsCG[j]) = std::max(x_max(dofsCG[j]), el_max(i));
      }
   }
   Array<double> minvals(x_min.HostReadWrite(), x_min.Size());
   Array<double> maxvals(x_max.HostReadWrite(), x_max.Size());
   gcomm.Reduce<double>(minvals, GroupCommunicator::Min);
   gcomm.Bcast(minvals);
   gcomm.Reduce<double>(maxvals, GroupCommunicator::Max);
   gcomm.Bcast(maxvals);

   // Use (x_min, x_max) to fill (dof_min, dof_max) for each DG dof.
   const TensorBasisElement *fe_cg =
      dynamic_cast<const TensorBasisElement *>(pfes_bounds.GetFE(0));
   const Array<int> &dof_map = fe_cg->GetDofMap();
   const int ndofs = dof_map.Size();
   for (int i = 0; i < NE; i++)
   {
      // Comment about the case when active_el != null, i.e., when this function
      // is used to compute the bounds of s:
      //
      // Note that this loop goes over all elements, including inactive ones.
      // The following happens in an inactive element:
      // - If a DOF is on the boundary with an active element, it will get the
      //   value that's propagated by the continuous functions x_min and x_max.
      // - Otherwise, the DOF would get the inf values.
      // This is the mechanism that allows new elements, that switch from
      // inactive to active, to get some valid bounds. More specifically, this
      // function is called on the old state, but the result from it is used
      // to limit the new state, which has different active elements.

      pfes_bounds.GetElementDofs(i, dofsCG);
      for (int j = 0; j < dofsCG.Size(); j++)
      {
         dof_min(i*ndofs + j) = x_min(dofsCG[dof_map[j]]);
         dof_max(i*ndofs + j) = x_max(dofsCG[dof_map[j]]);
      }
   }
}

void DofInfo::ComputeElementsMinMax(const Vector &u,
                                    Vector &u_min, Vector &u_max,
                                    Array<bool> *active_el,
                                    Array<bool> *active_dof) const
{
   const int NE = pfes.GetNE(), ndof = pfes.GetFE(0)->GetDof();
   int dof_id;
   u.HostRead(); u_min.HostReadWrite(); u_max.HostReadWrite();
   for (int k = 0; k < NE; k++)
   {
      u_min(k) = numeric_limits<double>::infinity();
      u_max(k) = -numeric_limits<double>::infinity();

      // Inactive elements don't affect the bounds.
      if (active_el && (*active_el)[k] == false) { continue; }

      for (int i = 0; i < ndof; i++)
      {
         dof_id = k*ndof + i;
         // Inactive dofs don't affect the bounds.
         if (active_dof && (*active_dof)[dof_id] == false) { continue; }

         u_min(k) = min(u_min(k), u(dof_id));
         u_max(k) = max(u_max(k), u(dof_id));
      }
   }
}

void DofInfo::FillNeighborDofs()
{
   // Use the first mesh element as indicator.
   const FiniteElement &dummy = *pfes.GetFE(0);
   const int dim = pmesh->Dimension();
   int i, j, k, ne = pmesh->GetNE();
   int nd = dummy.GetDof(), p = dummy.GetOrder();
   Array <int> bdrs, orientation;

   pmesh->ExchangeFaceNbrData();
   Table *face_to_el = pmesh->GetFaceToAllElementTable();

   NbrDof.SetSize(ne, numBdrs, numFaceDofs);

   // Permutations of BdrDofs, taking into account all possible orientations.
   // Assumes BdrDofs are ordered in xyz order, which is true for 3D hexes,
   // but it isn't true for 2D quads.
   // TODO: check other FEs, function ExtractBoundaryDofs().
   int orient_cnt = 1;
   if (dim == 2) { orient_cnt = 2; }
   if (dim == 3) { orient_cnt = 8; }
   const int dof1D_cnt = p+1;
   DenseTensor fdof_ids(numFaceDofs, numBdrs, orient_cnt);
   for (int ori = 0; ori < orient_cnt; ori++)
   {
      for (int face_id = 0; face_id < numBdrs; face_id++)
      {
         for (int fdof_id = 0; fdof_id < numFaceDofs; fdof_id++)
         {
            // Index of fdof_id in the current orientation.
            const int ori_fdof_id = GetLocalFaceDofIndex(dim, face_id, ori,
                                                         fdof_id, dof1D_cnt);
            fdof_ids(ori)(ori_fdof_id, face_id) = BdrDofs(fdof_id, face_id);
         }
      }
   }

   for (k = 0; k < ne; k++)
   {
      if (dim == 1)
      {
         pmesh->GetElementVertices(k, bdrs);

         for (i = 0; i < numBdrs; i++)
         {
            const int nbr_cnt = face_to_el->RowSize(bdrs[i]);
            if (nbr_cnt == 1)
            {
               // No neighbor element.
               NbrDof(k, i, 0) = -1;
               continue;
            }

            int el1_id, el2_id, nbr_id;
            pmesh->GetFaceElements(bdrs[i], &el1_id, &el2_id);
            if (el2_id < 0)
            {
               // This element is in a different mpi task.
               el2_id = -1 - el2_id + ne;
            }
            nbr_id = (el1_id == k) ? el2_id : el1_id;

            NbrDof(k,i,0) = nbr_id*nd + BdrDofs(0, (i+1) % 2);
         }
      }
      else if (dim==2)
      {
         pmesh->GetElementEdges(k, bdrs, orientation);

         for (i = 0; i < numBdrs; i++)
         {
            const int nbr_cnt = face_to_el->RowSize(bdrs[i]);
            if (nbr_cnt == 1)
            {
               // No neighbor element.
               for (j = 0; j < numFaceDofs; j++) { NbrDof(k, i, j) = -1; }
               continue;
            }

            int el1_id, el2_id, nbr_id;
            pmesh->GetFaceElements(bdrs[i], &el1_id, &el2_id);
            if (el2_id < 0)
            {
               // This element is in a different mpi task.
               el2_id = -1 - el2_id + ne;
            }
            nbr_id = (el1_id == k) ? el2_id : el1_id;

            int el1_info, el2_info;
            pmesh->GetFaceInfos(bdrs[i], &el1_info, &el2_info);
            const int face_id_nbr = (nbr_id == el1_id) ? el1_info / 64
                                    : el2_info / 64;
            for (j = 0; j < numFaceDofs; j++)
            {
               // Here it is utilized that the orientations of the face for
               // the two elements are opposite of each other.
               NbrDof(k,i,j) = nbr_id*nd + BdrDofs(numFaceDofs - 1 - j,
                                                   face_id_nbr);
            }
         }
      }
      else if (dim==3)
      {
         pmesh->GetElementFaces(k, bdrs, orientation);

         for (int f = 0; f < numBdrs; f++)
         {
            const int nbr_cnt = face_to_el->RowSize(bdrs[f]);
            if (nbr_cnt == 1)
            {
               // No neighbor element.
               for (j = 0; j < numFaceDofs; j++) { NbrDof(k, f, j) = -1; }
               continue;
            }

            int el1_id, el2_id, nbr_id;
            pmesh->GetFaceElements(bdrs[f], &el1_id, &el2_id);
            if (el2_id < 0)
            {
               // This element is in a different mpi task.
               el2_id = -1 - el2_id + ne;
            }
            nbr_id = (el1_id == k) ? el2_id : el1_id;

            // Local index and orientation of the face, when considered in
            // the neighbor element.
            int el1_info, el2_info;
            pmesh->GetFaceInfos(bdrs[f], &el1_info, &el2_info);
            const int face_id_nbr = (nbr_id == el1_id) ? el1_info / 64
                                    : el2_info / 64;
            const int face_or_nbr = (nbr_id == el1_id) ? el1_info % 64
                                    : el2_info % 64;
            for (j = 0; j < numFaceDofs; j++)
            {
               // What is the index of the j-th dof on the face, given its
               // orientation.
               const int loc_face_dof_id =
                  GetLocalFaceDofIndex(dim, face_id_nbr, face_or_nbr,
                                       j, dof1D_cnt);
               // What is the corresponding local dof id on the element,
               // given the face orientation.
               const int nbr_dof_id =
                  fdof_ids(face_or_nbr)(loc_face_dof_id, face_id_nbr);

               NbrDof(k, f, j) = nbr_id*nd + nbr_dof_id;
            }
         }
      }
   }

   delete face_to_el;
}

void DofInfo::FillSubcell2CellDof()
{
   const int dim = pmesh->Dimension(), p = pfes.GetFE(0)->GetOrder();

   if (dim==1)
   {
      numSubcells = p;
      numDofsSubcell = 2;
   }
   else if (dim==2)
   {
      numSubcells = p*p;
      numDofsSubcell = 4;
   }
   else if (dim==3)
   {
      numSubcells = p*p*p;
      numDofsSubcell = 8;
   }

   Sub2Ind.SetSize(numSubcells, numDofsSubcell);

   int aux;
   for (int m = 0; m < numSubcells; m++)
   {
      for (int j = 0; j < numDofsSubcell; j++)
      {
         if (dim == 1) { Sub2Ind(m,j) = m + j; }
         else if (dim == 2)
         {
            aux = m + m/p;
            switch (j)
            {
               case 0: Sub2Ind(m,j) =  aux; break;
               case 1: Sub2Ind(m,j) =  aux + 1; break;
               case 2: Sub2Ind(m,j) =  aux + p+1; break;
               case 3: Sub2Ind(m,j) =  aux + p+2; break;
            }
         }
         else if (dim == 3)
         {
            aux = m + m/p + (p+1)*(m/(p*p));
            switch (j)
            {
               case 0: Sub2Ind(m,j) = aux; break;
               case 1: Sub2Ind(m,j) = aux + 1; break;
               case 2: Sub2Ind(m,j) = aux + p+1; break;
               case 3: Sub2Ind(m,j) = aux + p+2; break;
               case 4: Sub2Ind(m,j) = aux + (p+1)*(p+1); break;
               case 5: Sub2Ind(m,j) = aux + (p+1)*(p+1)+1; break;
               case 6: Sub2Ind(m,j) = aux + (p+1)*(p+1)+p+1; break;
               case 7: Sub2Ind(m,j) = aux + (p+1)*(p+1)+p+2; break;
            }
         }
      }
   }
}

Assembly::Assembly(DofInfo &_dofs, LowOrderMethod &_lom,
                   const GridFunction &inflow,
                   ParFiniteElementSpace &pfes, ParMesh *submesh, int mode)
   : exec_mode(mode), inflow_gf(inflow), x_gf(&pfes),
     VolumeTerms(NULL),
     fes(&pfes), SubFes0(NULL), SubFes1(NULL),
     subcell_mesh(submesh), dofs(_dofs), lom(_lom)
{
   Mesh *mesh = fes->GetMesh();
   int k, i, m, dim = mesh->Dimension(), ne = fes->GetNE();

   Array <int> bdrs, orientation;
   FaceElementTransformations *Trans;

   bdrInt.SetSize(ne, dofs.numBdrs, dofs.numFaceDofs*dofs.numFaceDofs);
   bdrInt = 0.;

   if (lom.subcell_scheme)
   {
      VolumeTerms = lom.VolumeTerms;
      SubcellWeights.SetSize(dofs.numSubcells, dofs.numDofsSubcell, ne);

      SubFes0 = lom.SubFes0;
      SubFes1 = lom.SubFes1;
   }

   // Initialization for transport mode.
   if (exec_mode == 0)
   {
      for (k = 0; k < ne; k++)
      {
         if (dim==1)      { mesh->GetElementVertices(k, bdrs); }
         else if (dim==2) { mesh->GetElementEdges(k, bdrs, orientation); }
         else if (dim==3) { mesh->GetElementFaces(k, bdrs, orientation); }

         for (i = 0; i < dofs.numBdrs; i++)
         {
            Trans = mesh->GetFaceElementTransformations(bdrs[i]);
            ComputeFluxTerms(k, i, Trans, lom);
         }

         if (lom.subcell_scheme)
         {
            for (m = 0; m < dofs.numSubcells; m++)
            {
               ComputeSubcellWeights(k, m);
            }
         }
      }
   }
}

void Assembly::ComputeFluxTerms(const int e_id, const int BdrID,
                                FaceElementTransformations *Trans,
                                LowOrderMethod &lom)
{
   Mesh *mesh = fes->GetMesh();

   int i, j, l, dim = mesh->Dimension();
   double aux, vn;

   const FiniteElement &el = *fes->GetFE(e_id);

   Vector vval, nor(dim), shape(el.GetDof());

   for (l = 0; l < lom.irF->GetNPoints(); l++)
   {
      const IntegrationPoint &ip = lom.irF->IntPoint(l);
      IntegrationPoint eip1;
      Trans->Face->SetIntPoint(&ip);

      if (dim == 1)
      {
         Trans->Loc1.Transform(ip, eip1);
         nor(0) = 2.*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans->Face->Jacobian(), nor);
      }

      if (Trans->Elem1No != e_id)
      {
         Trans->Loc2.Transform(ip, eip1);
         el.CalcShape(eip1, shape);
         Trans->Elem2->SetIntPoint(&eip1);
         lom.coef->Eval(vval, *Trans->Elem2, eip1);
         nor *= -1.;
      }
      else
      {
         Trans->Loc1.Transform(ip, eip1);
         el.CalcShape(eip1, shape);
         Trans->Elem1->SetIntPoint(&eip1);
         lom.coef->Eval(vval, *Trans->Elem1, eip1);
      }

      nor /= nor.Norml2();

      if (exec_mode == 0)
      {
         // Transport.
         vn = std::min(0., vval * nor);
      }
      else
      {
         // Remap.
         vn = std::max(0., vval * nor);
         vn *= -1.0;
      }

      const double w = ip.weight * Trans->Face->Weight();
      for (i = 0; i < dofs.numFaceDofs; i++)
      {
         aux = w * shape(dofs.BdrDofs(i,BdrID)) * vn;
         for (j = 0; j < dofs.numFaceDofs; j++)
         {
            bdrInt(e_id, BdrID, i*dofs.numFaceDofs+j) -=
               aux * shape(dofs.BdrDofs(j,BdrID));
         }
      }
   }
}

void Assembly::ComputeSubcellWeights(const int k, const int m)
{
   DenseMatrix elmat; // These are essentially the same.
   const int e_id = k*dofs.numSubcells + m;
   const FiniteElement *el0 = SubFes0->GetFE(e_id);
   const FiniteElement *el1 = SubFes1->GetFE(e_id);
   ElementTransformation *tr = subcell_mesh->GetElementTransformation(e_id);
   VolumeTerms->AssembleElementMatrix2(*el1, *el0, *tr, elmat);

   for (int j = 0; j < elmat.Width(); j++)
   {
      // Using the fact that elmat has just one row.
      SubcellWeights(k)(m,j) = elmat(0,j);
   }
}

void Assembly::LinearFluxLumping(const int k, const int nd, const int BdrID,
                                 const Vector &x, Vector &y, const Vector &x_nd,
                                 const Vector &alpha) const
{
   int i, j, dofInd;
   double xNeighbor;
   Vector xDiff(dofs.numFaceDofs);
   const int size_x = x.Size();

   for (j = 0; j < dofs.numFaceDofs; j++)
   {
      dofInd = k*nd+dofs.BdrDofs(j,BdrID);
      const int nbr_dof_id = dofs.NbrDof(k, BdrID, j);
      // Note that if the boundary is outflow, we have bdrInt = 0 by definition,
      // s.t. this value will not matter.
      if (nbr_dof_id < 0) { xNeighbor = inflow_gf(dofInd); }
      else
      {
         xNeighbor = (nbr_dof_id < size_x) ? x(nbr_dof_id)
                     : x_nd(nbr_dof_id - size_x);
      }
      xDiff(j) = xNeighbor - x(dofInd);
   }

   for (i = 0; i < dofs.numFaceDofs; i++)
   {
      dofInd = k*nd+dofs.BdrDofs(i,BdrID);
      for (j = 0; j < dofs.numFaceDofs; j++)
      {
         // alpha=0 is the low order solution, alpha=1, the Galerkin solution.
         // 0 < alpha < 1 can be used for limiting within the low order method.
         y(dofInd) += bdrInt(k, BdrID, i*dofs.numFaceDofs + j) *
                      (xDiff(i) + (xDiff(j)-xDiff(i)) *
                       alpha(dofs.BdrDofs(i,BdrID)) *
                       alpha(dofs.BdrDofs(j,BdrID)));
      }
   }
}

void Assembly::NonlinFluxLumping(const int k, const int nd,
                                 const int BdrID, const Vector &x,
                                 Vector &y, const Vector &x_nd,
                                 const Vector &alpha) const
{
   int i, j, dofInd;
   double xNeighbor, SumCorrP = 0., SumCorrN = 0., eps = 1.E-15;
   const int size_x = x.Size();
   Vector xDiff(dofs.numFaceDofs), BdrTermCorr(dofs.numFaceDofs);
   BdrTermCorr = 0.;

   for (j = 0; j < dofs.numFaceDofs; j++)
   {
      dofInd = k*nd+dofs.BdrDofs(j,BdrID);
      const int nbr_dof_id = dofs.NbrDof(k, BdrID, j);
      // Note that if the boundary is outflow, we have bdrInt = 0 by definition,
      // s.t. this value will not matter.
      if (nbr_dof_id < 0) { xNeighbor = inflow_gf(dofInd); }
      else
      {
         xNeighbor = (nbr_dof_id < size_x) ? x(nbr_dof_id)
                     : x_nd(nbr_dof_id - size_x);
      }
      xDiff(j) = xNeighbor - x(dofInd);
   }

   y.HostReadWrite();
   bdrInt.HostRead();
   xDiff.HostReadWrite();
   for (i = 0; i < dofs.numFaceDofs; i++)
   {
      dofInd = k*nd+dofs.BdrDofs(i,BdrID);
      for (j = 0; j < dofs.numFaceDofs; j++)
      {
         y(dofInd) += bdrInt(k, BdrID, i*dofs.numFaceDofs + j) * xDiff(i);
         BdrTermCorr(i) += bdrInt(k, BdrID,
                                  i*dofs.numFaceDofs + j) * (xDiff(j)-xDiff(i));
      }
      BdrTermCorr(i) *= alpha(dofs.BdrDofs(i,BdrID));
      SumCorrP += max(0., BdrTermCorr(i));
      SumCorrN += min(0., BdrTermCorr(i));
   }

   for (i = 0; i < dofs.numFaceDofs; i++)
   {
      dofInd = k*nd+dofs.BdrDofs(i,BdrID);
      if (SumCorrP + SumCorrN > eps)
      {
         BdrTermCorr(i) = min(0., BdrTermCorr(i)) -
                          max(0., BdrTermCorr(i)) * SumCorrN / SumCorrP;
      }
      else if (SumCorrP + SumCorrN < -eps)
      {
         BdrTermCorr(i) = max(0., BdrTermCorr(i)) -
                          min(0., BdrTermCorr(i)) * SumCorrP / SumCorrN;
      }
      y(dofInd) += BdrTermCorr(i);
   }
}

void PrecondConvectionIntegrator::AssembleElementMatrix(
   const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
   int i, nd = el.GetDof(), dim = el.GetDim();

#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape, adjJ, Q_ir;
   Vector shape, vec2, BdFidxT;
#endif
   elmat.SetSize(nd);
   dshape.SetSize(nd,dim);
   adjJ.SetSize(dim);
   shape.SetSize(nd);
   vec2.SetSize(dim);
   BdFidxT.SetSize(nd);

   double w;
   Vector vec1;
   DenseMatrix mass(nd,nd), conv(nd,nd), lumpedM(nd,nd), tmp(nd,nd);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = Trans.OrderGrad(&el) + Trans.Order() + el.GetOrder();
      order = max(order, 2 * el.GetOrder() + Trans.OrderW());
      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   Q.Eval(Q_ir, Trans, *ir);

   conv = mass = 0.0;
   for (i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcDShape(ip, dshape);
      el.CalcShape(ip, shape);

      Trans.SetIntPoint(&ip);
      CalcAdjugate(Trans.Jacobian(), adjJ);
      Q_ir.GetColumnReference(i, vec1);
      vec1 *= alpha * ip.weight;

      adjJ.Mult(vec1, vec2);
      dshape.Mult(vec2, BdFidxT);

      AddMultVWt(shape, BdFidxT, conv);

      w = Trans.Weight() * ip.weight;
      AddMult_a_VVt(w, shape, mass);
   }
   lumpedM = mass;
   lumpedM.Lump();
   mass.Invert();

   MultABt(mass, lumpedM, tmp);
   MultAtB(tmp, conv, elmat); // using symmetry of mass matrix
}

void MixedConvectionIntegrator::AssembleElementMatrix2(
   const FiniteElement &tr_el, const FiniteElement &te_el,
   ElementTransformation &Trans, DenseMatrix &elmat)
{
   int tr_nd = tr_el.GetDof();
   int te_nd = te_el.GetDof();
   int dim = te_el.GetDim(); // Using test geometry.

#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape, adjJ, Q_ir;
   Vector shape, vec2, BdFidxT;
#endif
   elmat.SetSize(te_nd, tr_nd);
   dshape.SetSize(tr_nd,dim);
   adjJ.SetSize(dim);
   shape.SetSize(te_nd);
   vec2.SetSize(dim);
   BdFidxT.SetSize(tr_nd);

   Vector vec1;

   // Using midpoint rule and test geometry.
   const IntegrationRule *ir = &IntRules.Get(te_el.GetGeomType(), 1);

   Q.Eval(Q_ir, Trans, *ir);

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      tr_el.CalcDShape(ip, dshape);
      te_el.CalcShape(ip, shape);

      Trans.SetIntPoint(&ip);
      CalcAdjugate(Trans.Jacobian(), adjJ);
      Q_ir.GetColumnReference(i, vec1);
      vec1 *= alpha * ip.weight;

      adjJ.Mult(vec1, vec2);
      dshape.Mult(vec2, BdFidxT);

      AddMultVWt(shape, BdFidxT, elmat);
   }
}

int GetLocalFaceDofIndex3D(int loc_face_id, int face_orient,
                           int face_dof_id, int face_dof1D_cnt)
{
   int k1 = 0, k2 = 0;
   const int kf1 = face_dof_id % face_dof1D_cnt;
   const int kf2 = face_dof_id / face_dof1D_cnt;
   switch (loc_face_id)
   {
      case 0: // BOTTOM
         switch (face_orient)
         {
            case 0: // {0, 1, 2, 3}
               k1 = kf1;
               k2 = face_dof1D_cnt-1-kf2;
               break;
            case 1: // {0, 3, 2, 1}
               k1 = face_dof1D_cnt-1-kf2;
               k2 = kf1;
               break;
            case 2: // {1, 2, 3, 0}
               k1 = face_dof1D_cnt-1-kf2;
               k2 = face_dof1D_cnt-1-kf1;
               break;
            case 3: // {1, 0, 3, 2}
               k1 = face_dof1D_cnt-1-kf1;
               k2 = face_dof1D_cnt-1-kf2;
               break;
            case 4: // {2, 3, 0, 1}
               k1 = face_dof1D_cnt-1-kf1;
               k2 = kf2;
               break;
            case 5: // {2, 1, 0, 3}
               k1 = kf2;
               k2 = face_dof1D_cnt-1-kf1;
               break;
            case 6: // {3, 0, 1, 2}
               k1 = kf2;
               k2 = kf1;
               break;
            case 7: // {3, 2, 1, 0}
               k1 = kf1;
               k2 = kf2;
               break;
            default:
               mfem_error("This orientation does not exist in 3D");
               break;
         }
         break;
      case 1: // SOUTH
         switch (face_orient)
         {
            case 0: // {0, 1, 2, 3}
               k1 = kf1;
               k2 = kf2;
               break;
            case 1: // {0, 3, 2, 1}
               k1 = kf2;
               k2 = kf1;
               break;
            case 2: // {1, 2, 3, 0}
               k1 = kf2;
               k2 = face_dof1D_cnt-1-kf1;
               break;
            case 3: // {1, 0, 3, 2}
               k1 = face_dof1D_cnt-1-kf1;
               k2 = kf2;
               break;
            case 4: // {2, 3, 0, 1}
               k1 = face_dof1D_cnt-1-kf1;
               k2 = face_dof1D_cnt-1-kf2;
               break;
            case 5: // {2, 1, 0, 3}
               k1 = face_dof1D_cnt-1-kf2;
               k2 = face_dof1D_cnt-1-kf1;
               break;
            case 6: // {3, 0, 1, 2}
               k1 = face_dof1D_cnt-1-kf2;
               k2 = kf1;
               break;
            case 7: // {3, 2, 1, 0}
               k1 = kf1;
               k2 = face_dof1D_cnt-1-kf2;
               break;
            default:
               mfem_error("This orientation does not exist in 3D");
               break;
         }
         break;
      case 2: // EAST
         switch (face_orient)
         {
            case 0: // {0, 1, 2, 3}
               k1 = kf1;
               k2 = kf2;
               break;
            case 1: // {0, 3, 2, 1}
               k1 = kf2;
               k2 = kf1;
               break;
            case 2: // {1, 2, 3, 0}
               k1 = kf2;
               k2 = face_dof1D_cnt-1-kf1;
               break;
            case 3: // {1, 0, 3, 2}
               k1 = face_dof1D_cnt-1-kf1;
               k2 = kf2;
               break;
            case 4: // {2, 3, 0, 1}
               k1 = face_dof1D_cnt-1-kf1;
               k2 = face_dof1D_cnt-1-kf2;
               break;
            case 5: // {2, 1, 0, 3}
               k1 = face_dof1D_cnt-1-kf2;
               k2 = face_dof1D_cnt-1-kf1;
               break;
            case 6: // {3, 0, 1, 2}
               k1 = face_dof1D_cnt-1-kf2;
               k2 = kf1;
               break;
            case 7: // {3, 2, 1, 0}
               k1 = kf1;
               k2 = face_dof1D_cnt-1-kf2;
               break;
            default:
               mfem_error("This orientation does not exist in 3D");
               break;
         }
         break;
      case 3: // NORTH
         switch (face_orient)
         {
            case 0: // {0, 1, 2, 3}
               k1 = face_dof1D_cnt-1-kf1;
               k2 = kf2;
               break;
            case 1: // {0, 3, 2, 1}
               k1 = kf2;
               k2 = face_dof1D_cnt-1-kf1;
               break;
            case 2: // {1, 2, 3, 0}
               k1 = kf2;
               k2 = kf1;
               break;
            case 3: // {1, 0, 3, 2}
               k1 = kf1;
               k2 = kf2;
               break;
            case 4: // {2, 3, 0, 1}
               k1 = kf1;
               k2 = face_dof1D_cnt-1-kf2;
               break;
            case 5: // {2, 1, 0, 3}
               k1 = face_dof1D_cnt-1-kf2;
               k2 = kf1;
               break;
            case 6: // {3, 0, 1, 2}
               k1 = face_dof1D_cnt-1-kf2;
               k2 = face_dof1D_cnt-1-kf1;
               break;
            case 7: // {3, 2, 1, 0}
               k1 = face_dof1D_cnt-1-kf1;
               k2 = face_dof1D_cnt-1-kf2;
               break;
            default:
               mfem_error("This orientation does not exist in 3D");
               break;
         }
         break;
      case 4: // WEST
         switch (face_orient)
         {
            case 0: // {0, 1, 2, 3}
               k1 = face_dof1D_cnt-1-kf1;
               k2 = kf2;
               break;
            case 1: // {0, 3, 2, 1}
               k1 = kf2;
               k2 = face_dof1D_cnt-1-kf1;
               break;
            case 2: // {1, 2, 3, 0}
               k1 = kf2;
               k2 = kf1;
               break;
            case 3: // {1, 0, 3, 2}
               k1 = kf1;
               k2 = kf2;
               break;
            case 4: // {2, 3, 0, 1}
               k1 = kf1;
               k2 = face_dof1D_cnt-1-kf2;
               break;
            case 5: // {2, 1, 0, 3}
               k1 = face_dof1D_cnt-1-kf2;
               k2 = kf1;
               break;
            case 6: // {3, 0, 1, 2}
               k1 = face_dof1D_cnt-1-kf2;
               k2 = face_dof1D_cnt-1-kf1;
               break;
            case 7: // {3, 2, 1, 0}
               k1 = face_dof1D_cnt-1-kf1;
               k2 = face_dof1D_cnt-1-kf2;
               break;
            default:
               mfem_error("This orientation does not exist in 3D");
               break;
         }
         break;
      case 5: // TOP
         switch (face_orient)
         {
            case 0: // {0, 1, 2, 3}
               k1 = kf1;
               k2 = kf2;
               break;
            case 1: // {0, 3, 2, 1}
               k1 = kf2;
               k2 = kf1;
               break;
            case 2: // {1, 2, 3, 0}
               k1 = kf2;
               k2 = face_dof1D_cnt-1-kf1;
               break;
            case 3: // {1, 0, 3, 2}
               k1 = face_dof1D_cnt-1-kf1;
               k2 = kf2;
               break;
            case 4: // {2, 3, 0, 1}
               k1 = face_dof1D_cnt-1-kf1;
               k2 = face_dof1D_cnt-1-kf2;
               break;
            case 5: // {2, 1, 0, 3}
               k1 = face_dof1D_cnt-1-kf2;
               k2 = face_dof1D_cnt-1-kf1;
               break;
            case 6: // {3, 0, 1, 2}
               k1 = face_dof1D_cnt-1-kf2;
               k2 = kf1;
               break;
            case 7: // {3, 2, 1, 0}
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

void GetMinMax(const ParGridFunction &g, double &min, double &max)
{
   g.HostRead();
   double min_loc = g.Min(), max_loc = g.Max();
   MPI_Allreduce(&min_loc, &min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
   MPI_Allreduce(&max_loc, &max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
}

Array<int> SparseMatrix_Build_smap(const SparseMatrix &A)
{
   // Assuming that A is finalized
   const int *I = A.GetI(), *J = A.GetJ(), n = A.Size();
   Array<int> smap(I[n]);

   for (int row = 0, j = 0; row < n; row++)
   {
      for (int end = I[row+1]; j < end; j++)
      {
         int col = J[j];
         // Find the offset, _j, of the (col,row) entry and store it in smap[j].
         for (int _j = I[col], _end = I[col+1]; true; _j++)
         {
            MFEM_VERIFY(_j != _end, "Can't find the symmetric entry!");

            if (J[_j] == row) { smap[j] = _j; break; }
         }
      }
   }
   return smap;
}

void ComputeDiscreteUpwindingMatrix(const SparseMatrix &K,
                                    Array<int> smap, SparseMatrix& D)
{
   const int *Ip = K.GetI(), *Jp = K.GetJ(), n = K.Size();
   const double *Kp = K.GetData();

   double *Dp = D.GetData();

   for (int i = 0, k = 0; i < n; i++)
   {
      double rowsum = 0.;
      for (int end = Ip[i+1]; k < end; k++)
      {
         int j = Jp[k];
         double kij = Kp[k];
         double kji = Kp[smap[k]];
         double dij = fmax(fmax(0.0,-kij),-kji);
         Dp[k] = kij + dij;
         Dp[smap[k]] = kji + dij;
         if (i != j) { rowsum += dij; }
      }
      D(i,i) = K(i,i) -rowsum;
   }
}

void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    ParGridFunction &gf, const char *title,
                    int x, int y, int w, int h, const char *keys, bool vec)
{
   ParMesh &pmesh = *gf.ParFESpace()->GetParMesh();
   MPI_Comm comm = pmesh.GetComm();

   int num_procs, myid;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);

   bool newly_opened = false;
   int connection_failed;

   do
   {
      if (myid == 0)
      {
         if (!sock.is_open() || !sock)
         {
            sock.open(vishost, visport);
            sock.precision(8);
            newly_opened = true;
         }
         sock << "solution\n";
      }

      pmesh.PrintAsOne(sock);
      gf.SaveAsOne(sock);

      if (myid == 0 && newly_opened)
      {
         sock << "window_title '" << title << "'\n"
              << "window_geometry "
              << x << " " << y << " " << w << " " << h << "\n";
         if ( keys ) { sock << "keys " << keys << "\n"; }
         else { sock << "keys maaAc"; }
         if ( vec ) { sock << "vvv"; }
         sock << endl;
      }

      if (myid == 0)
      {
         connection_failed = !sock && !newly_opened;
      }
      MPI_Bcast(&connection_failed, 1, MPI_INT, 0, comm);
   }
   while (connection_failed);
}

} // namespace mfem
