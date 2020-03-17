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

#ifndef MFEM_REMHOS_TOOLS
#define MFEM_REMHOS_TOOLS

#include "mfem.hpp"

namespace mfem
{

enum MONOTYPE { None,
                DiscUpw, DiscUpw_FCT,
                ResDist, ResDist_FCT,
                ResDist_Monolithic };

int GetLocalFaceDofIndex(int dim, int loc_face_id, int face_orient,
                         int face_dof_id, int face_dof1D_cnt);
void ExtractBdrDofs(int p, Geometry::Type gtype, DenseMatrix &dofs);

class DofInfo;

class SmoothnessIndicator
{
private:
   const int type;
   const double param;
   ParFiniteElementSpace &pfes_CG_sub, &pfes_DG;
   SparseMatrix Mmat, LaplaceOp, *MassMixed;
   BilinearFormIntegrator *MassInt;
   Vector lumpedMH1;
   DenseMatrix ShapeEval;

   void ComputeVariationalMatrix(DofInfo &dof_info);
   void ApproximateLaplacian(const Vector &x, ParGridFunction &y);
   void ComputeFromSparsity(const SparseMatrix &K, const ParGridFunction &x,
                            Vector &x_min, Vector &x_max);

public:
   SmoothnessIndicator(int type_id,
                       ParFiniteElementSpace &pfes_CG_sub_,
                       ParFiniteElementSpace &pfes_DG_,
                       ParGridFunction &u,
                       DofInfo &dof_info);
   ~SmoothnessIndicator();

   void ComputeSmoothnessIndicator(const Vector &u, ParGridFunction &si_vals_u);
   void UpdateBounds(int dof_id, double u_HO,
                     const ParGridFunction &si_vals,
                     double &u_min, double &u_max);

   Vector DG2CG;
};

struct LowOrderMethod
{
   MONOTYPE MonoType;
   bool OptScheme;
   FiniteElementSpace *fes, *SubFes0, *SubFes1;
   Array <int> smap;
   SparseMatrix D;
   ParBilinearForm* pk;
   VectorCoefficient* coef;
   const IntegrationRule* irF;
   BilinearFormIntegrator* VolumeTerms;
   ParMesh* subcell_mesh;
   Vector scale;
};

// Class storing information on dofs needed for the low order methods and FCT.
class DofInfo
{
private:
   ParMesh *pmesh;
   ParFiniteElementSpace *pfes;

public:
   ParGridFunction x_min, x_max;

   Vector xi_min, xi_max; // min/max values for each dof
   Vector xe_min, xe_max; // min/max values for each element

   DenseMatrix BdrDofs, Sub2Ind;
   DenseTensor NbrDof;

   int dim, numBdrs, numFaceDofs, numSubcells, numDofsSubcell;

   DofInfo(ParFiniteElementSpace *fes_sltn, ParFiniteElementSpace *fes_bounds)
      : pmesh(fes_sltn->GetParMesh()), pfes(fes_sltn),
        x_min(fes_bounds), x_max(fes_bounds)
   {
      dim = pmesh->Dimension();

      int n = pfes->GetVSize();
      int ne = pmesh->GetNE();

      xi_min.SetSize(n);
      xi_max.SetSize(n);
      xe_min.SetSize(ne);
      xe_max.SetSize(ne);

      ExtractBdrDofs(pfes->GetFE(0)->GetOrder(),
                     pfes->GetFE(0)->GetGeomType(), BdrDofs);
      numFaceDofs = BdrDofs.Height();
      numBdrs = BdrDofs.Width();

      FillNeighborDofs();    // Fill NbrDof.
      FillSubcell2CellDof(); // Fill Sub2Ind.
   }

   // Computes the admissible interval of values for each DG dof from the values
   // of all elements that feature the dof at its physical location.
   // Assumes that xe_min and xe_max are already computed.
   void ComputeBounds()
   {
      ParFiniteElementSpace *pfesCG = x_min.ParFESpace();
      GroupCommunicator &gcomm = pfesCG->GroupComm();
      Array<int> dofsCG;

      // Form min/max at each CG dof, considering element overlaps.
      x_min =   std::numeric_limits<double>::infinity();
      x_max = - std::numeric_limits<double>::infinity();
      for (int i = 0; i < pmesh->GetNE(); i++)
      {
         x_min.FESpace()->GetElementDofs(i, dofsCG);
         for (int j = 0; j < dofsCG.Size(); j++)
         {
            x_min(dofsCG[j]) = std::min(x_min(dofsCG[j]), xe_min(i));
            x_max(dofsCG[j]) = std::max(x_max(dofsCG[j]), xe_max(i));
         }
      }
      Array<double> minvals(x_min.GetData(), x_min.Size()),
                    maxvals(x_max.GetData(), x_max.Size());
      gcomm.Reduce<double>(minvals, GroupCommunicator::Min);
      gcomm.Bcast(minvals);
      gcomm.Reduce<double>(maxvals, GroupCommunicator::Max);
      gcomm.Bcast(maxvals);

      // Use (x_min, x_max) to fill (xi_min, xi_max) for each DG dof.
      const TensorBasisElement *fe_cg =
         dynamic_cast<const TensorBasisElement *>(pfesCG->GetFE(0));
      const Array<int> &dof_map = fe_cg->GetDofMap();
      const int ndofs = dof_map.Size();
      for (int i = 0; i < pmesh->GetNE(); i++)
      {
         x_min.FESpace()->GetElementDofs(i, dofsCG);
         for (int j = 0; j < dofsCG.Size(); j++)
         {
            xi_min(i*ndofs + j) = x_min(dofsCG[dof_map[j]]);
            xi_max(i*ndofs + j) = x_max(dofsCG[dof_map[j]]);
         }
      }
   }

   ~DofInfo() { }

private:

   // For each DOF on an element boundary, the global index of the DOF on the
   // opposite site is computed and stored in a list. This is needed for lumping
   // the flux contributions as in the paper. Right now it works on 1D meshes,
   // quad meshes in 2D and 3D meshes of ordered cubes.
   // NOTE: The mesh is assumed to consist of segments, quads or hexes.
   // NOTE: This approach will not work for meshes with hanging nodes.
   void FillNeighborDofs()
   {
      // Use the first mesh element as indicator.
      const FiniteElement &dummy = *pfes->GetFE(0);
      int i, j, k, nbr, ne = pmesh->GetNE();
      int nd = dummy.GetDof(), p = dummy.GetOrder();
      Array <int> bdrs, orientation;
      FaceElementTransformations *Trans;

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

               Trans = pmesh->GetFaceElementTransformations(bdrs[i]);
               nbr = Trans->Elem1No == k ? Trans->Elem2No : Trans->Elem1No;
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
   }

   // A list is filled to later access the correct element-global indices given
   // the subcell number and subcell index.
   // NOTE: The mesh is assumed to consist of segments, quads or hexes.
   void FillSubcell2CellDof()
   {
      const FiniteElement &dummy = *pfes->GetFE(0);
      int j, m, aux, p = dummy.GetOrder();

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

      for (m = 0; m < numSubcells; m++)
      {
         for (j = 0; j < numDofsSubcell; j++)
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
};

class Assembly
{
private:
   const int exec_mode;
   const GridFunction &inflow_gf;
   mutable ParGridFunction x_gf;

public:
   Assembly(DofInfo &_dofs, LowOrderMethod &lom, const GridFunction &inflow,
            ParFiniteElementSpace &pfes, int mode)
      : exec_mode(mode), inflow_gf(inflow), x_gf(&pfes),
        fes(lom.fes), SubFes0(NULL), SubFes1(NULL),
        dofs(_dofs), subcell_mesh(NULL)
   {
      Mesh *mesh = fes->GetMesh();
      int k, i, m, dim = mesh->Dimension(), ne = fes->GetNE();

      Array <int> bdrs, orientation;
      FaceElementTransformations *Trans;

      const bool NeedSubWgts = lom.OptScheme && (lom.MonoType == ResDist ||
                                                 lom.MonoType == ResDist_FCT
                                                 || lom.MonoType == ResDist_Monolithic );

      bdrInt.SetSize(ne, dofs.numBdrs, dofs.numFaceDofs*dofs.numFaceDofs);
      bdrInt = 0.;

      if (NeedSubWgts)
      {
         VolumeTerms = lom.VolumeTerms;
         SubcellWeights.SetSize(dofs.numSubcells, dofs.numDofsSubcell, ne);

         SubFes0 = lom.SubFes0;
         SubFes1 = lom.SubFes1;
         subcell_mesh = lom.subcell_mesh;
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

            if (NeedSubWgts)
            {
               for (m = 0; m < dofs.numSubcells; m++)
               {
                  ComputeSubcellWeights(k, m);
               }
            }
         }
      }
   }

   // Destructor
   ~Assembly() { }

   // Auxiliary member variables that need to be accessed during time-stepping.
   FiniteElementSpace *fes, *SubFes0, *SubFes1;
   DofInfo &dofs;
   Mesh *subcell_mesh;
   BilinearFormIntegrator *VolumeTerms;

   // Data structures storing Galerkin contributions. These are updated for
   // remap but remain constant for transport.
   // bdrInt - eq (32).
   // SubcellWeights - above eq (49).
   DenseTensor bdrInt, SubcellWeights;

   void ComputeFluxTerms(const int e_id, const int BdrID,
                         FaceElementTransformations *Trans, LowOrderMethod &lom)
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

   void ComputeSubcellWeights(const int k, const int m)
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

   void LinearFluxLumping(const int k, const int nd,
                          const int BdrID, const Vector &x,
                          Vector &y, const Vector &x_nd,
                          const Vector &alpha) const;
   void NonlinFluxLumping(const int k, const int nd,
                          const int BdrID, const Vector &x,
                          Vector &y, const Vector &x_nd,
                          const Vector &alpha) const;
};


} // namespace mfem

#endif // MFEM_REMHOS_TOOLS
