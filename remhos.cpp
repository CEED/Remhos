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
//
//                    ____                 __
//                   / __ \___  ____ ___  / /_  ____  _____
//                  / /_/ / _ \/ __ `__ \/ __ \/ __ \/ ___/
//                 / _, _/  __/ / / / / / / / / /_/ (__  )
//                /_/ |_|\___/_/ /_/ /_/_/ /_/\____/____/
//
//                       High-order Remap Miniapp
//
// Remhos (REMap High-Order Solver) is a miniapp that solves the pure advection
// equations that are used to perform discontinuous field interpolation (remap)
// as part of the Eulerian phase in Arbitrary-Lagrangian Eulerian (ALE)
// simulations.
//
// Sample runs: see README.md, section 'Verification of Results'.
//
//    Using lua problem definition file
//    ./remhos -p balls-and-jacks.lua -r 4 -dt 0.001 -tf 5.0
//
//    Transport mode:
//    ./remhos -m ./data/periodic-segment.mesh -p 0 -r 2 -dt 0.005
//    ./remhos -m ./data/periodic-square.mesh -p 0 -r 2 -dt 0.01 -tf 10
//    ./remhos -m ./data/periodic-hexagon.mesh -p 0 -r 2 -dt 0.01 -tf 10
//    ./remhos -m ./data/periodic-square.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ./remhos -m ./data/periodic-hexagon.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ./remhos -m ./data/star-q3.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ./remhos -m ./data/disc-nurbs.mesh -p 1 -r 3 -dt 0.005 -tf 9
//    ./remhos -m ./data/disc-nurbs.mesh -p 2 -r 3 -dt 0.005 -tf 9
//    ./remhos -m ./data/periodic-square.mesh -p 3 -r 4 -dt 0.0025 -tf 9 -vs 20
//    ./remhos -m ./data/periodic-cube.mesh -p 0 -r 2 -o 2 -dt 0.02 -tf 8
//    ./remhos -m ./data/periodic-square.mesh -p 4 -r 4 -dt 0.001 -o 2 -mt 3
//    ./remhos -m ./data/periodic-square.mesh -p 3 -r 2 -dt 0.0025 -o 15 -tf 9 -mt 4
//    ./remhos -m ./data/periodic-square.mesh -p 5 -r 4 -dt 0.002 -o 2 -tf 0.8 -mt 4
//    ./remhos -m ./data/periodic-cube.mesh -p 5 -r 5 -dt 0.0001 -o 1 -tf 0.8 -mt 4
//
//    Remap mode:
//    ./remhos -m ./data/periodic-square.mesh -p 10 -r 3 -dt 0.005 -tf 0.5 -mt 4 -vs 10
//    ./remhos -m ./data/periodic-square.mesh -p 14 -r 3 -dt 0.005 -tf 0.5 -mt 4 -vs 10
//
// Description:  This example code solves the time-dependent advection equation
//               du/dt + v.grad(u) = 0, where v is a given fluid velocity, and
//               u0(x)=u(0,x) is a given initial condition.
//
//               The example demonstrates the use of Discontinuous Galerkin (DG)
//               bilinear forms in MFEM (face integrators), the use of explicit
//               ODE time integrators, the definition of periodic boundary
//               conditions through periodic meshes, as well as the use of GLVis
//               for persistent visualization of a time-evolving solution. The
//               saving of time-dependent data files for external visualization
//               with VisIt (visit.llnl.gov) is also illustrated.
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

#ifdef USE_LUA
#include "lua.hpp"
lua_State* L;
#endif

// Choice for the problem setup. The fluid velocity, initial condition and
// inflow boundary condition are chosen based on this parameter.
int problem_num;

// 0 is standard transport.
// 1 is standard remap (mesh moves, solution is fixed).
int exec_mode;

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v);

// Initial condition
double u0_function(const Vector &x);

// Inflow boundary condition
double inflow_function(const Vector &x);

// Mesh bounding box
Vector bb_min, bb_max;

int GetLocalFaceDofIndex(int dim, int loc_face_id, int face_orient,
                         int face_dof_id, int face_dof1D_cnt);

enum MONOTYPE { None, DiscUpw, DiscUpw_FCT, ResDist, ResDist_FCT };

struct LowOrderMethod
{
   MONOTYPE MonoType;
   bool OptScheme;
   FiniteElementSpace *fes, *SubFes0, *SubFes1;
   Array <int> smap;
   SparseMatrix D;
   BilinearForm* pk;
   VectorCoefficient* coef;
   const IntegrationRule* irF;
   BilinearFormIntegrator* VolumeTerms;
   Mesh* subcell_mesh;
};

// Utility function to build a map to the offset of the symmetric entry in a
// sparse matrix.
Array<int> SparseMatrix_Build_smap(const SparseMatrix &A)
{
   // Assuming that A is finalized
   const int *I = A.GetI(), *J = A.GetJ(), n = A.Size();
   Array<int> smap;
   smap.SetSize(I[n]);

   for (int row = 0, j = 0; row < n; row++)
   {
      for (int end = I[row+1]; j < end; j++)
      {
         int col = J[j];
         // Find the offset, _j, of the (col,row) entry and store it in smap[j].
         for (int _j = I[col], _end = I[col+1]; true; _j++)
         {
            if (_j == _end)
            {
               mfem_error("SparseMatrix_Build_smap");
            }

            if (J[_j] == row)
            {
               smap[j] = _j;
               break;
            }
         }
      }
   }
   return smap;
}

// Given a matrix K, matrix D (initialized with same sparsity as K) is computed,
// such that (K+D)_ij >= 0 for i != j.
void ComputeDiscreteUpwindingMatrix(const SparseMatrix& K,
                                    Array<int> smap, SparseMatrix& D)
{
   const int n = K.Size();
   int* Ip = K.GetI();
   int* Jp = K.GetJ();
   double* Kp = K.GetData();

   double* Dp = D.GetData();

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

// The mesh corresponding to Bezier subcells of order p is constructed.
// NOTE: The mesh is assumed to consist of segments, quads or hexes.
Mesh* GetSubcellMesh(Mesh *mesh, int p)
{
   MFEM_VERIFY(p > 1, "This function should not be called with p = 1.");

   Mesh *subcell_mesh = NULL;
   if (mesh->Dimension() > 1)
   {
      int basis_lor = BasisType::ClosedUniform; // Get a uniformly refined mesh.
      subcell_mesh = new Mesh(mesh, p, basis_lor);

      // Check if the mesh is periodic.
      const L2_FECollection *L2_coll = dynamic_cast<const L2_FECollection *>
                                       (mesh->GetNodes()->FESpace()->FEColl());
      if (L2_coll == NULL)
      {
         // Standard non-periodic mesh.
         // Note that the fine mesh is always linear.
         subcell_mesh->SetCurvature(1);
      }
      else
      {
         // Periodic mesh - the node positions must be corrected after the call
         // to the above Mesh constructor. Note that the fine mesh is always
         // linear.
         const bool disc_nodes = true;
         subcell_mesh->SetCurvature(1, disc_nodes);
         GridFunction *coarse = mesh->GetNodes();
         GridFunction *fine   = subcell_mesh->GetNodes();
         InterpolationGridTransfer transf(*coarse->FESpace(), *fine->FESpace());
         transf.ForwardOperator().Mult(*coarse, *fine);
      }
   }
   else
   {
      // TODO generalize to arbitrary 1D segments (different length than 1).
      subcell_mesh = new Mesh(mesh->GetNE() * p, 1.);
      subcell_mesh->SetCurvature(1);
   }
   return subcell_mesh;
}

// Appropriate quadrature rule for faces of is obtained.
// TODO: check if this gives the desired order. I use the same order for all
// faces. In DGTraceIntegrator it uses the min of OrderW, why?
const IntegrationRule *GetFaceIntRule(FiniteElementSpace *fes)
{
   int i, qOrdF;
   Mesh* mesh = fes->GetMesh();
   FaceElementTransformations *Trans;

   // Use the first mesh face with two elements as indicator.
   for (i = 0; i < mesh->GetNumFaces(); i++)
   {
      Trans = mesh->GetFaceElementTransformations(i);
      qOrdF = Trans->Elem1->OrderW();
      if (Trans->Elem2No >= 0)
      {
         // qOrdF is chosen such that L2-norm of basis functions is computed
         // accurately.
         qOrdF = max(qOrdF, Trans->Elem2->OrderW());
         break;
      }
   }
   // Use the first mesh element as indicator.
   const FiniteElement &dummy = *fes->GetFE(0);
   qOrdF += 2*dummy.GetOrder();

   return &IntRules.Get(Trans->FaceGeom, qOrdF);
}

// Class storing information on dofs needed for the low order methods and FCT.
class DofInfo
{
private:
   Mesh* mesh;
   FiniteElementSpace* fes;

public:
   GridFunction x_min, x_max;

   Vector xi_min, xi_max; // min/max values for each dof
   Vector xe_min, xe_max; // min/max values for each element

   DenseMatrix BdrDofs, Sub2Ind;
   DenseTensor NbrDof;

   int dim, numBdrs, numFaceDofs, numSubcells, numDofsSubcell;

   DofInfo(FiniteElementSpace *fes_sltn, FiniteElementSpace *fes_bounds)
      : mesh(fes_sltn->GetMesh()), fes(fes_sltn),
        x_min(fes_bounds), x_max(fes_bounds)
   {
      dim = mesh->Dimension();

      int n = fes->GetVSize();
      int ne = mesh->GetNE();

      xi_min.SetSize(n);
      xi_max.SetSize(n);
      xe_min.SetSize(ne);
      xe_max.SetSize(ne);

      // Use the first mesh element as indicator.
      const FiniteElement &dummy = *fes->GetFE(0);
      dummy.ExtractBdrDofs(BdrDofs);
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
      Array<int> dofsCG;

      // Form min/max at each CG dof, taking into account element overlaps.
      x_min =   numeric_limits<double>::infinity();
      x_max = - numeric_limits<double>::infinity();
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         x_min.FESpace()->GetElementDofs(i, dofsCG);
         for (int j = 0; j < dofsCG.Size(); j++)
         {
            x_min(dofsCG[j]) = std::min(x_min(dofsCG[j]), xe_min(i));
            x_max(dofsCG[j]) = std::max(x_max(dofsCG[j]), xe_max(i));
         }
      }

      // Use (x_min, x_max) to fill (xi_min, xi_max) for each DG dof.
      const TensorBasisElement *fe_cg =
         dynamic_cast<const TensorBasisElement *>(x_min.FESpace()->GetFE(0));
      const Array<int> &dof_map = fe_cg->GetDofMap();
      const int ndofs = dof_map.Size();
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         x_min.FESpace()->GetElementDofs(i, dofsCG);
         for(int j = 0; j < dofsCG.Size(); j++)
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
      const FiniteElement &dummy = *fes->GetFE(0);
      int i, j, k, ind, nbr, ne = mesh->GetNE();
      int nd = dummy.GetDof(), p = dummy.GetOrder();
      Array <int> bdrs, NbrBdrs, orientation;
      FaceElementTransformations *Trans;

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
         if (dim==1)
         {
            mesh->GetElementVertices(k, bdrs);

            for (i = 0; i < numBdrs; i++)
            {
               Trans = mesh->GetFaceElementTransformations(bdrs[i]);
               nbr = Trans->Elem1No == k ? Trans->Elem2No : Trans->Elem1No;
               NbrDof(k,i,0) = nbr*nd + BdrDofs(0,(i+1)%2);
            }
         }
         else if (dim==2)
         {
            mesh->GetElementEdges(k, bdrs, orientation);

            for (i = 0; i < numBdrs; i++)
            {
               Trans = mesh->GetFaceElementTransformations(bdrs[i]);
               nbr = (Trans->Elem1No == k) ? Trans->Elem2No : Trans->Elem1No;

               if (nbr < 0)
               {
                  for (j = 0; j < numFaceDofs; j++) { NbrDof(k,i,j) = -1; }
                  continue;
               }

               for (j = 0; j < numFaceDofs; j++)
               {
                  mesh->GetElementEdges(nbr, NbrBdrs, orientation);
                  // Current face's id in the neighbor element.
                  for (ind = 0; ind < numBdrs; ind++)
                  {
                     if (NbrBdrs[ind] == bdrs[i]) { break; }
                  }
                  // Here it is utilized that the orientations of the face for
                  // the two elements are opposite of each other.
                  NbrDof(k,i,j) = nbr*nd + BdrDofs(numFaceDofs-1-j,ind);
               }
            }
         }
         else if (dim==3)
         {
            mesh->GetElementFaces(k, bdrs, orientation);

            for (int f = 0; f < numBdrs; f++)
            {
               int el1_id, el2_id, nbr_id;
               mesh->GetFaceElements(bdrs[f], &el1_id, &el2_id);
               nbr_id = (el1_id == k) ? el2_id : el1_id;

               if (nbr_id < 0)
               {
                  for (j = 0; j < numFaceDofs; j++) { NbrDof(k, f, j) = -1; }
                  continue;
               }

               // Local index and orientation of the face, when considered in
               // the neighbor element.
               int el1_info, el2_info;
               mesh->GetFaceInfos(bdrs[f], &el1_info, &el2_info);
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
                  const int loc_dof_id =
                     fdof_ids(face_or_nbr)(loc_face_dof_id, face_id_nbr);

                  NbrDof(k, f, j) = nbr_id*nd + loc_dof_id;

#if 0
                  // old method
                  Trans = mesh->GetFaceElementTransformations(bdrs[0]);
                  nbr = Trans->Elem1No == k ? Trans->Elem2No : Trans->Elem1No;
                  NbrDof(k,0,j) = nbr*nd + (p+1)*(p+1)*p+j;

                  Trans = mesh->GetFaceElementTransformations(bdrs[1]);
                  nbr = Trans->Elem1No == k ? Trans->Elem2No : Trans->Elem1No;

                  NbrDof(k,1,j) = nbr*nd + (j/(p+1))*(p+1)*(p+1)+(p+1)*p+(j%(p+1));

                  Trans = mesh->GetFaceElementTransformations(bdrs[2]);
                  nbr = Trans->Elem1No == k ? Trans->Elem2No : Trans->Elem1No;

                  NbrDof(k,2,j) = nbr*nd + j*(p+1);

                  Trans = mesh->GetFaceElementTransformations(bdrs[3]);
                  nbr = Trans->Elem1No == k ? Trans->Elem2No : Trans->Elem1No;

                  NbrDof(k,3,j) = nbr*nd + (j/(p+1))*(p+1)*(p+1)+(j%(p+1));

                  Trans = mesh->GetFaceElementTransformations(bdrs[4]);
                  nbr = Trans->Elem1No == k ? Trans->Elem2No : Trans->Elem1No;

                  NbrDof(k,4,j) = nbr*nd + (j+1)*(p+1)-1;

                  Trans = mesh->GetFaceElementTransformations(bdrs[5]);
                  nbr = Trans->Elem1No == k ? Trans->Elem2No : Trans->Elem1No;

                  NbrDof(k,5,j) = nbr*nd + j;
#endif
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
      const FiniteElement &dummy = *fes->GetFE(0);
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
public:
   Assembly(DofInfo &_dofs, LowOrderMethod &lom)
      : fes(lom.fes), SubFes0(NULL), SubFes1(NULL), dofs(_dofs),
        subcell_mesh(NULL)
   {
      Mesh *mesh = fes->GetMesh();
      int k, i, m, dim = mesh->Dimension(), ne = fes->GetNE();

      Array <int> bdrs, orientation;
      FaceElementTransformations *Trans;

      const bool NeedBdr = lom.OptScheme || (lom.MonoType != DiscUpw &&
                                             lom.MonoType != DiscUpw_FCT);

      const bool NeedSubcells = lom.OptScheme && (lom.MonoType == ResDist ||
                                                  lom.MonoType == ResDist_FCT);

      if (NeedBdr)
      {
         bdrInt.SetSize(ne, dofs.numBdrs, dofs.numFaceDofs*dofs.numFaceDofs);
         bdrInt = 0.;
      }
      if (NeedSubcells)
      {
         VolumeTerms = lom.VolumeTerms;
         SubcellWeights.SetSize(dofs.numSubcells, dofs.numDofsSubcell, ne);

         SubFes0 = lom.SubFes0;
         SubFes1 = lom.SubFes1;
         subcell_mesh = lom.subcell_mesh;
      }

      // Initialization for transport mode.
      if (exec_mode == 0 && (NeedBdr || NeedSubcells))
      {
         for (k = 0; k < ne; k++)
         {
            if (NeedBdr)
            {
               if (dim==1)      { mesh->GetElementVertices(k, bdrs); }
               else if (dim==2) { mesh->GetElementEdges(k, bdrs, orientation); }
               else if (dim==3) { mesh->GetElementFaces(k, bdrs, orientation); }

               for (i = 0; i < dofs.numBdrs; i++)
               {
                  Trans = mesh->GetFaceElementTransformations(bdrs[i]);
                  ComputeFluxTerms(k, i, Trans, lom);
               }
            }
            if (NeedSubcells)
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
            vn = min(0., vval * nor);
         }
         else
         {
            // Remap.
            vn = max(0., vval * nor);
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
};


/** A time-dependent operator for the right-hand side of the ODE. The DG weak
    form of du/dt = -v.grad(u) is M du/dt = K u + b, where M and K are the mass
    and advection matrices, and b describes the flow on the boundary. This can
    be written as a general ODE, du/dt = M^{-1} (K u + b), and this class is
    used to evaluate the right-hand side. */
class FE_Evolution : public TimeDependentOperator
{
private:
   BilinearForm &Mbf, &Kbf, &ml;
   SparseMatrix &M, &K;
   Vector &lumpedM;
   const GridFunction &inflow_gf;
   const Vector &b;

   Vector start_mesh_pos, start_submesh_pos;
   GridFunction &mesh_pos, *submesh_pos, &mesh_vel, &submesh_vel;

   mutable Vector z;

   double dt;
   Assembly &asmbl;

   LowOrderMethod &lom;
   DofInfo &dofs;

public:
   FE_Evolution(BilinearForm &Mbf_, SparseMatrix &_M, BilinearForm &_ml,
                Vector &_lumpedM, BilinearForm &Kbf_, SparseMatrix &_K,
                const Vector &_b, const GridFunction &inflow,
                GridFunction &pos, GridFunction *sub_pos,
                GridFunction &vel, GridFunction &sub_vel,
                Assembly &_asmbl, LowOrderMethod &_lom, DofInfo &_dofs);

   virtual void Mult(const Vector &x, Vector &y) const;

   virtual void SetDt(double _dt) { dt = _dt; }
   void SetRemapStartPos(const Vector &m_pos, const Vector &sm_pos)
   {
      start_mesh_pos    = m_pos;
      start_submesh_pos = sm_pos;
   }

   // Mass matrix solve, addressing the bad Bernstein condition number.
   virtual void NeumannSolve(const Vector &b, Vector &x) const;

   virtual void LinearFluxLumping(const int k, const int nd,
                                  const int BdrID, const Vector &x,
                                  Vector &y, const Vector &alpha) const;

   virtual void ComputeHighOrderSolution(const Vector &x, Vector &y) const;
   virtual void ComputeLowOrderSolution(const Vector &x, Vector &y) const;
   virtual void ComputeFCTSolution(const Vector &x, const Vector &yH,
                                   const Vector &yL, Vector &y) const;

   virtual ~FE_Evolution() { }
};

FE_Evolution* adv;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.

#ifdef USE_LUA
   L = luaL_newstate();
   luaL_openlibs(L);
   const char* problem_file = "problem.lua";
#else
   problem_num = 4;
#endif
   const char *mesh_file = "./data/unit-square.mesh";
   int ref_levels = 2;
   int order = 3;
   int mesh_order = 2;
   int ode_solver_type = 3;
   MONOTYPE MonoType = ResDist_FCT;
   bool OptScheme = true;
   double t_final = 2.0;
   double dt = 0.0025;
   bool visualization = true;
   bool visit = false;
   bool binary = false;
   int vis_steps = 20;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
#ifdef USE_LUA
   args.AddOption(&problem_file, "-p", "--problem",
                  "lua problem definition file.");
#else
   args.AddOption(&problem_num, "-p", "--problem",
                  "Problem setup to use. See options in velocity_function().");
#endif
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite element solution.");
   args.AddOption(&mesh_order, "-mo", "--mesh-order",
                  "Order (degree) of the mesh.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6.");
   args.AddOption((int*)(&MonoType), "-mt", "--MonoType",
                  "Monotonicity scheme: 0 - no monotonicity treatment,\n\t"
                  "                     1 - discrete upwinding - LO,\n\t"
                  "                     2 - discrete upwinding - FCT,\n\t"
                  "                     3 - residual distribution - LO,\n\t"
                  "                     4 - residual distribution - FCT.");
   args.AddOption(&OptScheme, "-sc", "--subcell", "-el", "--element (basic)",
                  "Optimized scheme: PDU / subcell (optimized).");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.AddOption(&binary, "-binary", "--binary-datafiles", "-ascii",
                  "--ascii-datafiles",
                  "Use binary (Sidre) or ascii format for VisIt data files.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // When not using lua, exec mode is derived from problem number convention
   if (problem_num < 10)      { exec_mode = 0; }
   else if (problem_num < 20) { exec_mode = 1; }
   else { MFEM_ABORT("Unspecified execution mode."); }

#ifdef USE_LUA
   // When using lua, exec mode is read from lua file
   if (luaL_dofile(L, problem_file))
   {
      printf("Error opening lua file: %s\n",problem_file);
      exit(1);
   }

   lua_getglobal(L, "exec_mode");
   if (!lua_isnumber(L, -1))
   {
      printf("Did not find exec_mode in lua input.\n");
      return 1;
   }
   exec_mode = (int)lua_tonumber(L, -1);
#endif

   // 2. Read the mesh from the given mesh file. We can handle geometrically
   //    periodic meshes in this code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);

   const int dim = mesh->Dimension();


   // 3. Define the ODE solver used for time integration. Several explicit
   //    Runge-Kutta methods are available.
   ODESolver *ode_solver = NULL;
   switch (ode_solver_type)
   {
      case 1: ode_solver = new ForwardEulerSolver; break;
      case 2: ode_solver = new RK2Solver(1.0); break;
      case 3: ode_solver = new RK3SSPSolver; break;
      case 4: ode_solver = new RK4Solver; break;
      case 6: ode_solver = new RK6Solver; break;
      default:
         cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         delete mesh;
         return 3;
   }

   // Refine the mesh and set the required curvature.
   for (int lev = 0; lev < ref_levels; lev++) { mesh->UniformRefinement(); }
   // Check if the input mesh is periodic.
   const L2_FECollection *L2_coll = dynamic_cast<const L2_FECollection *>
                                    (mesh->GetNodes()->FESpace()->FEColl());
   const bool periodic = (L2_coll != NULL);
   mesh->SetCurvature(mesh_order, periodic);
   mesh->GetBoundingBox(bb_min, bb_max, max(order, 1));

   // Current mesh positions.
   GridFunction *x = mesh->GetNodes();

   // Store initial mesh positions.
   Vector x0(x->Size());
   x0 = *x;

   // 5. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   const int btype = BasisType::Positive;
   DG_FECollection fec(order, dim, btype);
   FiniteElementSpace fes(mesh, &fec);

   // The min and max bounds are represented as CG functions of the same order
   // as the solution, thus having 1:1 dof correspondence inside each element.
   H1_FECollection fec_bounds(order, dim, BasisType::GaussLobatto);
   FiniteElementSpace fes_bounds(mesh, &fec_bounds);

   // Check for meaningful combinations of parameters.
   bool fail = false;
   if (MonoType != None)
   {
      if (((int)MonoType != MonoType) || (MonoType < 0) || (MonoType > 4))
      {
         cout << "Unsupported option for monotonicity treatment." << endl;
         fail = true;
      }
      if (btype != 2)
      {
         cout << "Monotonicity treatment requires Bernstein basis." << endl;
         fail = true;
      }
      if (order == 0)
      {
         // Disable monotonicity treatment for piecewise constants.
         mfem_warning("For -o 0, monotonicity treatment is disabled.");
         MonoType = None;
         OptScheme = false;
      }
   }
   else { OptScheme = false; }

   if ((MonoType > 2) && (order==1) && OptScheme)
   {
      // Avoid subcell methods for linear elements.
      mfem_warning("For -o 1, subcell scheme is disabled.");
      OptScheme = false;
   }
   if (fail)
   {
      delete mesh;
      delete ode_solver;
      return 5;
   }

   cout << "Number of unknowns: " << fes.GetVSize() << endl;

   // Fields related to inflow BC.
   FunctionCoefficient inflow(inflow_function);
   GridFunction inflow_gf(&fes);
   inflow_gf.ProjectCoefficient(inflow);

   // Velocity for the problem. Depending on the execution mode, this is the
   // advective velocity (transport) or mesh velocity (remap).
   VectorFunctionCoefficient velocity(dim, velocity_function);

   // Mesh velocity. Note that the resulting coefficient will be evaluated in
   // logical space.
   GridFunction v_gf(x->FESpace());
   v_gf.ProjectCoefficient(velocity);
   if (mesh->bdr_attributes.Size() > 0)
   {
      // Zero it out on boundaries (not moving boundaries).
      Array<int> ess_bdr(mesh->bdr_attributes.Max()), ess_vdofs;
      ess_bdr = 1;
      x->FESpace()->GetEssentialVDofs(ess_bdr, ess_vdofs);
      for (int i = 0; i < v_gf.Size(); i++)
      {
         if (ess_vdofs[i] == -1) { v_gf(i) = 0.0; }
      }
   }
   VectorGridFunctionCoefficient v_coef(&v_gf);

   // Set up the bilinear and linear forms corresponding to the DG
   // discretization.
   BilinearForm m(&fes);
   m.AddDomainIntegrator(new MassIntegrator);

   BilinearForm k(&fes);
   if (exec_mode == 0)
   {
      k.AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));
   }
   else if (exec_mode == 1)
   {
      k.AddDomainIntegrator(new ConvectionIntegrator(v_coef));
   }

   // In case of basic discrete upwinding, add boundary terms.
   if (((MonoType == DiscUpw) || (MonoType == DiscUpw_FCT)) && (!OptScheme))
   {
      if (exec_mode == 0)
      {
         k.AddInteriorFaceIntegrator( new TransposeIntegrator(
                                         new DGTraceIntegrator(velocity, 1.0, -0.5)) );
         k.AddBdrFaceIntegrator( new TransposeIntegrator(
                                    new DGTraceIntegrator(velocity, 1.0, -0.5)) );
      }
      else if (exec_mode == 1)
      {
         k.AddInteriorFaceIntegrator(new TransposeIntegrator(
                                        new DGTraceIntegrator(v_coef, -1.0, -0.5)) );
         k.AddBdrFaceIntegrator( new TransposeIntegrator(
                                    new DGTraceIntegrator(v_coef, -1.0, -0.5)) );
      }
   }

   LinearForm b(&fes);
   b.AddBdrFaceIntegrator(
      new BoundaryFlowIntegrator(inflow, v_coef, -1.0, -0.5));

   // Compute the lumped mass matrix algebraically
   Vector lumpedM;
   BilinearForm ml(&fes);
   ml.AddDomainIntegrator(new LumpedIntegrator(new MassIntegrator));
   ml.Assemble();
   ml.Finalize();
   ml.SpMat().GetDiag(lumpedM);

   m.Assemble();
   m.Finalize();
   int skip_zeros = 0;
   k.Assemble(skip_zeros);
   k.Finalize(skip_zeros);
   b.Assemble();

   // Store topological dof data.
   DofInfo dofs(&fes, &fes_bounds);

   // Precompute data required for high and low order schemes. This could be put
   // into a separate routine. I am using a struct now because the various
   // schemes require quite different information.
   LowOrderMethod lom;
   lom.MonoType = MonoType;
   lom.OptScheme = OptScheme;
   lom.fes = &fes;

   lom.pk = NULL;
   if ((lom.MonoType == DiscUpw) || (lom.MonoType == DiscUpw_FCT))
   {
      if (!lom.OptScheme)
      {
         lom.smap = SparseMatrix_Build_smap(k.SpMat());
         lom.D = k.SpMat();

         if (exec_mode == 0)
         {
            ComputeDiscreteUpwindingMatrix(k.SpMat(), lom.smap, lom.D);
         }
      }
      else
      {
         lom.pk = new BilinearForm(&fes);
         if (exec_mode == 0)
         {
            lom.pk->AddDomainIntegrator(
               new PrecondConvectionIntegrator(velocity, -1.0) );
         }
         else if (exec_mode == 1)
         {
            lom.pk->AddDomainIntegrator(
               new PrecondConvectionIntegrator(v_coef) );
         }
         lom.pk->Assemble(skip_zeros);
         lom.pk->Finalize(skip_zeros);

         lom.smap = SparseMatrix_Build_smap(lom.pk->SpMat());
         lom.D = lom.pk->SpMat();

         if (exec_mode == 0)
         {
            ComputeDiscreteUpwindingMatrix(lom.pk->SpMat(), lom.smap, lom.D);
         }
      }
   }
   if (exec_mode == 1) { lom.coef = &v_coef; }
   else                { lom.coef = &velocity; }

   lom.irF = GetFaceIntRule(&fes);

   DG_FECollection fec0(0, dim, btype);
   DG_FECollection fec1(1, dim, btype);

   // For linear elements, Opt scheme has already been disabled.
   const bool NeedSubcells = lom.OptScheme && (lom.MonoType == ResDist ||
                                               lom.MonoType == ResDist_FCT);
   lom.subcell_mesh = NULL;
   lom.SubFes0 = NULL;
   lom.SubFes1 = NULL;
   GridFunction *xsub = NULL;
   GridFunction v_sub_gf;
   VectorGridFunctionCoefficient v_sub_coef;
   Vector x0_sub;
   if (NeedSubcells)
   {
      // Create the low order refined submesh.
      lom.subcell_mesh = GetSubcellMesh(mesh, order);
      lom.SubFes0 = new FiniteElementSpace(lom.subcell_mesh, &fec0);
      lom.SubFes1 = new FiniteElementSpace(lom.subcell_mesh, &fec1);

      // Submesh positions.
      xsub = lom.subcell_mesh->GetNodes();

      // Submesh velocity.
      v_sub_gf.SetSpace(xsub->FESpace());
      v_sub_gf.ProjectCoefficient(velocity);
      if (lom.subcell_mesh->bdr_attributes.Size() > 0)
      {
         // Zero it out on boundaries (not moving boundaries).
         Array<int> ess_bdr(lom.subcell_mesh->bdr_attributes.Max()), ess_vdofs;
         ess_bdr = 1;
         xsub->FESpace()->GetEssentialVDofs(ess_bdr, ess_vdofs);
         for (int i = 0; i < v_sub_gf.Size(); i++)
         {
            if (ess_vdofs[i] == -1) { v_sub_gf(i) = 0.0; }
         }
      }
      v_sub_coef.SetGridFunction(&v_sub_gf);

      // Store initial submesh positions.
      x0_sub = *xsub;

      // Integrator on the submesh.
      if (exec_mode == 0)
      {
         lom.VolumeTerms = new MixedConvectionIntegrator(velocity, -1.0);
      }
      else if (exec_mode == 1)
      {
         lom.VolumeTerms = new MixedConvectionIntegrator(v_sub_coef);
      }
   }

   Assembly asmbl(dofs, lom);

   // Initial condition.
   GridFunction u(&fes);
   FunctionCoefficient u0(u0_function);
   u.ProjectCoefficient(u0);

   // Print the starting meshes and initial condition.
   {
      ofstream meshHO("meshHO_init.mesh");
      meshHO.precision(precision);
      mesh->Print(meshHO);
      if (lom.subcell_mesh)
      {
         ofstream meshLO("meshLO_init.mesh");
         meshLO.precision(precision);
         lom.subcell_mesh->Print(meshLO);
      }
      ofstream sltn("sltn_init.gf");
      sltn.precision(precision);
      u.Save(sltn);
   }

   // Create data collection for solution output: either VisItDataCollection for
   // ascii data files, or SidreDataCollection for binary data files.
   DataCollection *dc = NULL;
   if (visit)
   {
      if (binary)
      {
#ifdef MFEM_USE_SIDRE
         dc = new SidreDataCollection("Example9", mesh);
#else
         MFEM_ABORT("Must build with MFEM_USE_SIDRE=YES for binary output.");
#endif
      }
      else
      {
         dc = new VisItDataCollection("Example9", mesh);
         dc->SetPrecision(precision);
      }
      dc->RegisterField("solution", &u);
      dc->SetCycle(0);
      dc->SetTime(0.0);
      dc->Save();
   }

   socketstream sout;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      sout.open(vishost, visport);
      if (!sout)
      {
         cout << "Unable to connect to GLVis server at "
              << vishost << ':' << visport << endl;
         visualization = false;
         cout << "GLVis visualization disabled.\n";
      }
      else
      {
         sout.precision(precision);
         sout << "solution\n" << *mesh << u;
         sout << "pause\n";
         sout << flush;
         cout << "GLVis visualization paused."
              << " Press space (in the GLVis window) to resume it.\n";
      }
   }

   // check for conservation
   Vector mass(lumpedM);
   double initialMass = lumpedM * u;

   // 8. Define the time-dependent evolution operator describing the ODE
   //    right-hand side, and perform time-integration (looping over the time
   //    iterations, ti, with a time-step dt).

   FE_Evolution* adv = new FE_Evolution(m, m.SpMat(), ml, lumpedM, k, k.SpMat(),
                                        b, inflow_gf, *x, xsub, v_gf, v_sub_gf,
                                        asmbl, lom, dofs);

   double t = 0.0;
   adv->SetTime(t);
   ode_solver->Init(*adv);

   bool done = false;
   for (int ti = 0; !done; )
   {
      double dt_real = min(dt, t_final - t);

      adv->SetDt(dt_real);

      if (exec_mode == 1) { adv->SetRemapStartPos(x0, x0_sub); }

      ode_solver->Step(u, t, dt_real);
      ti++;

      if (exec_mode == 1)
      {
         add(x0, t, v_gf, *x);
         if (NeedSubcells) { add(x0_sub, t, v_sub_gf, *xsub); }
      }

      done = (t >= t_final - 1.e-8*dt);

      if (done || ti % vis_steps == 0)
      {
         cout << "time step: " << ti << ", time: " << t << endl;

         if (visualization) { sout << "solution\n" << *mesh << u << flush; }
         if (visit)
         {
            dc->SetCycle(ti);
            dc->SetTime(t);
            dc->Save();
         }
      }
   }

   // Print the final meshes and solution.
   {
      ofstream meshHO("meshHO_final.mesh");
      meshHO.precision(precision);
      mesh->Print(meshHO);
      if (asmbl.subcell_mesh)
      {
         ofstream meshLO("meshLO_final.mesh");
         meshLO.precision(precision);
         asmbl.subcell_mesh->Print(meshLO);
      }
      ofstream sltn("sltn_final.gf");
      sltn.precision(precision);
      u.Save(sltn);
   }

   // check for conservation
   double finalMass;
   if (exec_mode == 1)
   {
      ml.BilinearForm::operator=(0.0);
      ml.Assemble();
      ml.SpMat().GetDiag(lumpedM);
      finalMass = lumpedM * u;
   }
   else { finalMass = mass * u; }
   cout << setprecision(10)
        << "Final mass: " << finalMass << endl
        << "Max value:  " << u.Max() << endl << setprecision(6)
        << "Mass loss:  " << abs(initialMass - finalMass) << endl;

   // Compute errors, if the initial condition is equal to the final solution
   if (problem_num == 4) // solid body rotation
   {
      cout << "L1-error: " << u.ComputeLpError(1., u0) << ", L-Inf-error: "
           << u.ComputeLpError(numeric_limits<double>::infinity(), u0)
           << "." << endl;
   }

   // 10. Free the used memory.
   delete mesh;
   delete ode_solver;
   delete dc;

   delete lom.pk;

   if (NeedSubcells)
   {
      delete asmbl.SubFes0;
      delete asmbl.SubFes1;
      delete asmbl.VolumeTerms;
      delete lom.subcell_mesh;
   }

   return 0;
}


void FE_Evolution::NeumannSolve(const Vector &f, Vector &x) const
{
   int i, iter, n = f.Size(), max_iter = 20;
   Vector y(n);
   const double abs_tol = 1.e-4;

   x = 0.;

   double resid = f.Norml2();
   for (iter = 1; iter <= max_iter; iter++)
   {
      M.Mult(x, y);
      y -= f;
      resid = y.Norml2();
      if (resid <= abs_tol)
      {
         return;
      }
      for (i = 0; i < n; i++)
      {
         x(i) -= y(i) / lumpedM(i);
      }
   }
}

void FE_Evolution::LinearFluxLumping(const int k, const int nd,
                                     const int BdrID, const Vector &x,
                                     Vector &y, const Vector &alpha) const
{
   int i, j, idx, dofInd;
   double xNeighbor;
   Vector xDiff(dofs.numFaceDofs);

   for (j = 0; j < dofs.numFaceDofs; j++)
   {
      dofInd = k*nd+dofs.BdrDofs(j,BdrID);
      idx = dofs.NbrDof(k, BdrID, j);
      // Note that if the boundary is outflow, we have bdrInt = 0 by definition,
      // s.t. this value will not matter.
      xNeighbor = (idx < 0) ? inflow_gf(dofInd) : x(idx);
      xDiff(j) = xNeighbor - x(dofInd);
   }

   for (i = 0; i < dofs.numFaceDofs; i++)
   {
      dofInd = k*nd+dofs.BdrDofs(i,BdrID);
      for (j = 0; j < dofs.numFaceDofs; j++)
      {
         // alpha=0 is the low order solution, alpha=1, the Galerkin solution.
         // 0 < alpha < 1 can be used for limiting within the low order method.
         y(dofInd) += asmbl.bdrInt(k, BdrID, i*dofs.numFaceDofs + j) *
                      (xDiff(i) + (xDiff(j)-xDiff(i)) *
                                  alpha(dofs.BdrDofs(i,BdrID)) *
                                  alpha(dofs.BdrDofs(j,BdrID)));
      }
   }
}

void FE_Evolution::ComputeLowOrderSolution(const Vector &x, Vector &y) const
{
   const FiniteElement* dummy = lom.fes->GetFE(0);
   int i, j, k, dofInd, nd = dummy->GetDof(), ne = lom.fes->GetNE();
   Vector alpha(nd); alpha = 0.;

   if (lom.MonoType == DiscUpw || lom.MonoType == DiscUpw_FCT)
   {
      // Reassemble on the new mesh (given by mesh_pos).
      if (exec_mode == 1)
      {
         if (!lom.OptScheme)
         {
            ComputeDiscreteUpwindingMatrix(K, lom.smap, lom.D);
         }
         else
         {
            lom.pk->BilinearForm::operator=(0.0);
            lom.pk->Assemble();
            ComputeDiscreteUpwindingMatrix(lom.pk->SpMat(), lom.smap, lom.D);
         }
      }

      // Discretization and monotonicity terms.
      lom.D.Mult(x, y);
      y += b;

      // Lump fluxes (for PDU), compute min/max, and invert lumped mass matrix.
      for (k = 0; k < ne; k++)
      {
         // Boundary contributions
         if (lom.OptScheme)
         {
            for (i = 0; i < dofs.numBdrs; i++)
            {
               LinearFluxLumping(k, nd, i, x, y, alpha);
            }
         }

         dofs.xe_min(k) = numeric_limits<double>::infinity();
         dofs.xe_max(k) = -dofs.xe_min(k);

         for (j = 0; j < nd; j++)
         {
            dofInd = k*nd+j;
            dofs.xe_max(k) = max(dofs.xe_max(k), x(dofInd));
            dofs.xe_min(k) = min(dofs.xe_min(k), x(dofInd));
            y(dofInd) /= lumpedM(dofInd);
         }
      }
   }
   else // RD(S)
   {
      int m, dofInd2, loc;
      double xSum, sumFluctSubcellP, sumFluctSubcellN, sumWeightsP,
             sumWeightsN, weightP, weightN, rhoP, rhoN,
             aux, fluct, gamma = 10., eps = 1.E-15;
      Vector xMaxSubcell, xMinSubcell, sumWeightsSubcellP, sumWeightsSubcellN,
             fluctSubcellP, fluctSubcellN, nodalWeightsP, nodalWeightsN;

      // Discretization terms
      y = b;
      K.Mult(x, z);

      // Monotonicity terms
      for (k = 0; k < ne; k++)
      {
         // Boundary contributions
         for (i = 0; i < dofs.numBdrs; i++)
         {
            LinearFluxLumping(k, nd, i, x, y, alpha);
         }

         // Element contributions
         dofs.xe_min(k) = numeric_limits<double>::infinity();
         dofs.xe_max(k) = -dofs.xe_min(k);
         rhoP = rhoN = xSum = 0.;

         for (j = 0; j < nd; j++)
         {
            dofInd = k*nd+j;
            dofs.xe_max(k) = max(dofs.xe_max(k), x(dofInd));
            dofs.xe_min(k) = min(dofs.xe_min(k), x(dofInd));
            xSum += x(dofInd);
            if (lom.OptScheme)
            {
               rhoP += max(0., z(dofInd));
               rhoN += min(0., z(dofInd));
            }
         }

         sumWeightsP = nd*dofs.xe_max(k) - xSum + eps;
         sumWeightsN = nd*dofs.xe_min(k) - xSum - eps;

         if (lom.OptScheme)
         {
            fluctSubcellP.SetSize(dofs.numSubcells);
            fluctSubcellN.SetSize(dofs.numSubcells);
            xMaxSubcell.SetSize(dofs.numSubcells);
            xMinSubcell.SetSize(dofs.numSubcells);
            sumWeightsSubcellP.SetSize(dofs.numSubcells);
            sumWeightsSubcellN.SetSize(dofs.numSubcells);
            nodalWeightsP.SetSize(nd);
            nodalWeightsN.SetSize(nd);
            sumFluctSubcellP = sumFluctSubcellN = 0.;
            nodalWeightsP = 0.; nodalWeightsN = 0.;

            // compute min-/max-values and the fluctuation for subcells
            for (m = 0; m < dofs.numSubcells; m++)
            {
               xMinSubcell(m) = numeric_limits<double>::infinity();
               xMaxSubcell(m) = -xMinSubcell(m);
               fluct = xSum = 0.;

               if (exec_mode == 1)
               {
                  asmbl.ComputeSubcellWeights(k, m);
               }

               for (i = 0; i < dofs.numDofsSubcell; i++)
               {
                  dofInd = k*nd + dofs.Sub2Ind(m, i);
                  fluct += asmbl.SubcellWeights(k)(m,i) * x(dofInd);
                  xMaxSubcell(m) = max(xMaxSubcell(m), x(dofInd));
                  xMinSubcell(m) = min(xMinSubcell(m), x(dofInd));
                  xSum += x(dofInd);
               }
               sumWeightsSubcellP(m) = dofs.numDofsSubcell
                                       * xMaxSubcell(m) - xSum + eps;
               sumWeightsSubcellN(m) = dofs.numDofsSubcell
                                       * xMinSubcell(m) - xSum - eps;

               fluctSubcellP(m) = max(0., fluct);
               fluctSubcellN(m) = min(0., fluct);
               sumFluctSubcellP += fluctSubcellP(m);
               sumFluctSubcellN += fluctSubcellN(m);
            }

            for (m = 0; m < dofs.numSubcells; m++)
            {
               for (i = 0; i < dofs.numDofsSubcell; i++)
               {
                  loc = dofs.Sub2Ind(m, i);
                  dofInd = k*nd + loc;
                  nodalWeightsP(loc) += fluctSubcellP(m)
                                        * ((xMaxSubcell(m) - x(dofInd))
                                           / sumWeightsSubcellP(m)); // eq. (10)
                  nodalWeightsN(loc) += fluctSubcellN(m)
                                        * ((xMinSubcell(m) - x(dofInd))
                                           / sumWeightsSubcellN(m)); // eq. (11)
               }
            }
         }

         for (i = 0; i < nd; i++)
         {
            dofInd = k*nd+i;
            weightP = (dofs.xe_max(k) - x(dofInd)) / sumWeightsP;
            weightN = (dofs.xe_min(k) - x(dofInd)) / sumWeightsN;

            if (lom.OptScheme)
            {
               aux = gamma / (rhoP + eps);
               weightP *= 1. - min(aux * sumFluctSubcellP, 1.);
               weightP += min(aux, 1./(sumFluctSubcellP+eps))*nodalWeightsP(i);

               aux = gamma / (rhoN - eps);
               weightN *= 1. - min(aux * sumFluctSubcellN, 1.);
               weightN += max(aux, 1./(sumFluctSubcellN-eps))*nodalWeightsN(i);
            }

            for (j = 0; j < nd; j++)
            {
               dofInd2 = k*nd+j;
               if (z(dofInd2) > eps)
               {
                  y(dofInd) += weightP * z(dofInd2);
               }
               else if (z(dofInd2) < -eps)
               {
                  y(dofInd) += weightN * z(dofInd2);
               }
            }
            y(dofInd) /= lumpedM(dofInd);
         }
      }
   }
}

// No monotonicity treatment, straightforward high-order scheme
// ydot = M^{-1} (K x + b).
void FE_Evolution::ComputeHighOrderSolution(const Vector &x, Vector &y) const
{
   int i, k, nd = lom.fes->GetFE(0)->GetDof(), ne = lom.fes->GetNE();
   Vector alpha(nd); alpha = 1.;

   K.Mult(x, z);
   z += b;

   // Incorporate flux terms only if the low order scheme is PDU, RD, or RDS.
   if (lom.MonoType != DiscUpw_FCT || lom.OptScheme)
   {
      // The boundary contributions have been computed in the low order scheme.
      for (k = 0; k < ne; k++)
      {
         for (i = 0; i < dofs.numBdrs; i++)
         {
            LinearFluxLumping(k, nd, i, x, z, alpha);
         }
      }
   }

   NeumannSolve(z, y);
}

// High order reconstruction that yields an updated admissible solution by means
// of clipping the solution coefficients within certain bounds and scaling the
// antidiffusive fluxes in a way that leads to local conservation of mass. yH,
// yL are the high and low order discrete time derivatives.
void FE_Evolution::ComputeFCTSolution(const Vector &x, const Vector &yH,
                                      const Vector &yL, Vector &y) const
{
   int j, k, nd, dofInd;
   double sumPos, sumNeg, eps = 1.E-15;
   Vector uClipped, fClipped;

   dofs.ComputeBounds();

   // Monotonicity terms
   for (k = 0; k < lom.fes->GetMesh()->GetNE(); k++)
   {
      const FiniteElement* el = lom.fes->GetFE(k);
      nd = el->GetDof();

      uClipped.SetSize(nd); uClipped = 0.;
      fClipped.SetSize(nd); fClipped = 0.;
      sumPos = sumNeg = 0.;

      for (j = 0; j < nd; j++)
      {
         dofInd = k*nd+j;

         uClipped(j) = min(dofs.xi_max(dofInd), max(x(dofInd) + dt*yH(dofInd),
                                                    dofs.xi_min(dofInd)) );

         fClipped(j) = lumpedM(dofInd) / dt
                       * ( uClipped(j) - (x(dofInd) + dt * yL(dofInd)) );

         sumPos += max(fClipped(j), 0.);
         sumNeg += min(fClipped(j), 0.);
      }

      for (j = 0; j < nd; j++)
      {
         if ((sumPos + sumNeg > eps) && (fClipped(j) > eps))
         {
            fClipped(j) *= - sumNeg / sumPos;
         }
         if ((sumPos + sumNeg < -eps) && (fClipped(j) < -eps))
         {
            fClipped(j) *= - sumPos / sumNeg;
         }

         // Set y to the discrete time derivative featuring the high order anti-
         // diffusive reconstruction that leads to an forward Euler updated
         // admissible solution.
         dofInd = k*nd+j;
         y(dofInd) = yL(dofInd) + fClipped(j) / lumpedM(dofInd);
      }
   }
}


// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(BilinearForm &Mbf_, SparseMatrix &_M,
                           BilinearForm &_ml, Vector &_lumpedM,
                           BilinearForm &Kbf_, SparseMatrix &_K,
                           const Vector &_b, const GridFunction &inflow,
                           GridFunction &pos, GridFunction *sub_pos,
                           GridFunction &vel, GridFunction &sub_vel,
                           Assembly &_asmbl,
                           LowOrderMethod &_lom, DofInfo &_dofs) :
   TimeDependentOperator(_M.Size()), Mbf(Mbf_), Kbf(Kbf_), ml(_ml),
   M(_M), K(_K), lumpedM(_lumpedM), inflow_gf(inflow), b(_b),
   start_mesh_pos(pos.Size()), start_submesh_pos(sub_vel.Size()),
   mesh_pos(pos), submesh_pos(sub_pos),
   mesh_vel(vel), submesh_vel(sub_vel),
   z(_M.Size()),
   asmbl(_asmbl), lom(_lom), dofs(_dofs) { }

void FE_Evolution::Mult(const Vector &x, Vector &y) const
{
   if (exec_mode == 1)
   {
      // Move the mesh positions.
      const double t = GetTime();
      add(start_mesh_pos, t, mesh_vel, mesh_pos);
      if (submesh_pos)
      {
         add(start_submesh_pos, t, submesh_vel, *submesh_pos);
      }

      // Reassemble on the new mesh. Element contributions.
      Mbf.BilinearForm::operator=(0.0);
      Mbf.Assemble();
      Kbf.BilinearForm::operator=(0.0);
      Kbf.Assemble(0);
      ml.BilinearForm::operator=(0.0);
      ml.Assemble();
      ml.SpMat().GetDiag(lumpedM);

      // Boundary contributions.
      const bool NeedBdr = lom.OptScheme || (lom.MonoType != DiscUpw &&
                                             lom.MonoType != DiscUpw_FCT);
      if (NeedBdr)
      {
         asmbl.bdrInt = 0.;

         Mesh *mesh = lom.fes->GetMesh();
         const int dim = mesh->Dimension(), ne = lom.fes->GetNE();
         Array<int> bdrs, orientation;
         FaceElementTransformations *Trans;

         for (int k = 0; k < ne; k++)
         {
            if (dim==1)
            {
               mesh->GetElementVertices(k, bdrs);
            }
            else if (dim==2)
            {
               mesh->GetElementEdges(k, bdrs, orientation);
            }
            else if (dim==3)
            {
               mesh->GetElementFaces(k, bdrs, orientation);
            }

            for (int i = 0; i < dofs.numBdrs; i++)
            {
               Trans = mesh->GetFaceElementTransformations(bdrs[i]);
               asmbl.ComputeFluxTerms(k, i, Trans, lom);
            }
         }
      }
   }

   if (lom.MonoType == 0)
   {
      ComputeHighOrderSolution(x, y);
   }
   else
   {
      if (lom.MonoType % 2 == 1)
      {
         ComputeLowOrderSolution(x, y);
      }
      else if (lom.MonoType % 2 == 0)
      {
         Vector yH(x.Size()), yL(x.Size());

         ComputeLowOrderSolution(x, yL);
         ComputeHighOrderSolution(x, yH);
         ComputeFCTSolution(x, yH, yL, y);
      }
   }
}

#ifdef USE_LUA
void lua_velocity_function(const Vector &x, Vector &v)
{
   lua_getglobal(L, "velocity_function");
   int dim = x.Size();

   lua_pushnumber(L, x(0));
   if (dim > 1)
   {
      lua_pushnumber(L, x(1));
   }
   if (dim > 2)
   {
      lua_pushnumber(L, x(2));
   }

   double v0 = 0;
   double v1 = 0;
   double v2 = 0;
   lua_call(L, dim, dim);
   v0 = (double)lua_tonumber(L, -1);
   lua_pop(L, 1);
   if (dim > 1)
   {
      v1 = (double)lua_tonumber(L, -1);
      lua_pop(L, 1);
   }
   if (dim > 2)
   {
      v2 = (double)lua_tonumber(L, -1);
      lua_pop(L, 1);
   }

   v(0) = v0;
   if (dim > 1)
   {
      v(0) = v1;
      v(1) = v0;
   }
   if (dim > 2)
   {
      v(0) = v2;
      v(1) = v1;
      v(2) = v0;
   }
}
#endif

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v)
{
#ifdef USE_LUA
   lua_velocity_function(x, v);
   return;
#endif

   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }

   int ProbExec = problem_num % 20;

   switch (ProbExec)
   {
      case 0:
      {
         // Translations in 1D, 2D, and 3D
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = sqrt(2./3.); v(1) = sqrt(1./3.); break;
            case 3: v(0) = sqrt(3./6.); v(1) = sqrt(2./6.); v(2) = sqrt(1./6.);
               break;
         }
         break;
      }
      case 1:
      case 2:
      case 4:
      {
         // Clockwise rotation in 2D around the origin
         const double w = M_PI/2;
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = -w*X(1); v(1) = w*X(0); break;
            case 3: v(0) = -w*X(1); v(1) = w*X(0); v(2) = 0.0; break;
         }
         break;
      }
      case 3:
      {
         // Clockwise twisting rotation in 2D around the origin
         const double w = M_PI/2;
         double d = max((X(0)+1.)*(1.-X(0)),0.) * max((X(1)+1.)*(1.-X(1)),0.);
         d = d*d;
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = d*w*X(1); v(1) = -d*w*X(0); break;
            case 3: v(0) = d*w*X(1); v(1) = -d*w*X(0); v(2) = 0.0; break;
         }
         break;
      }
      case 5:
      {
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = 1.0; v(1) = 1.0; break;
            case 3: v(0) = 1.0; v(1) = 1.0; v(2) = 1.0; break;
         }
         break;
      }
      case 10:
      case 11:
      case 12:
      case 13:
      case 14:
      case 15:
      {
         // Taylor-Green velocity, used for mesh motion in remap tests used for
         // all possible initial conditions.

         // Map [-1,1] to [0,1].
         for (int d = 0; d < dim; d++) { X(d) = X(d) * 0.5 + 0.5; }

         if (dim == 1) { MFEM_ABORT("Not implemented."); }
         v(0) =  sin(M_PI*X(0)) * cos(M_PI*X(1));
         v(1) = -cos(M_PI*X(0)) * sin(M_PI*X(1));
         if (dim == 3)
         {
            v(0) *= cos(M_PI*X(2));
            v(1) *= cos(M_PI*X(2));
            v(2) = 0.0;
         }
         break;
      }
   }
}

double box(std::pair<double,double> p1, std::pair<double,double> p2,
           double theta,
           std::pair<double,double> origin, double x, double y)
{
   double xmin=p1.first;
   double xmax=p2.first;
   double ymin=p1.second;
   double ymax=p2.second;
   double ox=origin.first;
   double oy=origin.second;

   double pi = M_PI;
   double s=std::sin(theta*pi/180);
   double c=std::cos(theta*pi/180);

   double xn=c*(x-ox)-s*(y-oy)+ox;
   double yn=s*(x-ox)+c*(y-oy)+oy;

   if (xn>xmin && xn<xmax && yn>ymin && yn<ymax)
   {
      return 1.0;
   }
   else
   {
      return 0.0;
   }
}

double box3D(double xmin, double xmax, double ymin, double ymax, double zmin,
             double zmax, double theta, double ox, double oy, double x,
             double y, double z)
{
   double pi = M_PI;
   double s=std::sin(theta*pi/180);
   double c=std::cos(theta*pi/180);

   double xn=c*(x-ox)-s*(y-oy)+ox;
   double yn=s*(x-ox)+c*(y-oy)+oy;

   if (xn>xmin && xn<xmax && yn>ymin && yn<ymax && z>zmin && z<zmax)
   {
      return 1.0;
   }
   else
   {
      return 0.0;
   }
}

double get_cross(double rect1, double rect2)
{
   double intersection=rect1*rect2;
   return rect1+rect2-intersection; //union
}

double ring(double rin, double rout, Vector c, Vector y)
{
   double r = 0.;
   int dim = c.Size();
   if (dim != y.Size())
   {
      mfem_error("Origin vector and variable have to be of the same size.");
   }
   for (int i = 0; i < dim; i++)
   {
      r += pow(y(i)-c(i), 2.);
   }
   r = sqrt(r);
   if (r>rin && r<rout)
   {
      return 1.0;
   }
   else
   {
      return 0.0;
   }
}

// Initial condition as defined by lua function
#ifdef USE_LUA
double lua_u0_function(const Vector &x)
{
   lua_getglobal(L, "initial_function");
   int dim = x.Size();

   lua_pushnumber(L, x(0));
   if (dim > 1)
   {
      lua_pushnumber(L, x(1));
   }
   if (dim > 2)
   {
      lua_pushnumber(L, x(2));
   }

   lua_call(L, dim, 1);
   double u = (double)lua_tonumber(L, -1);
   lua_pop(L, 1);

   return u;
}
#endif

// Initial condition: lua function or hard-coded functions
double u0_function(const Vector &x)
{
#ifdef USE_LUA
   return lua_u0_function(x);
#endif

   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }

   int ProbExec = problem_num % 10;

   switch (ProbExec)
   {
      case 0:
      case 1:
      {
         switch (dim)
         {
            case 1:
               return exp(-40.*pow(X(0)-0.5,2));
            case 2:
            case 3:
            {
               double rx = 0.45, ry = 0.25, cx = 0., cy = -0.2, w = 10.;
               if (dim == 3)
               {
                  const double s = (1. + 0.25*cos(2*M_PI*X(2)));
                  rx *= s;
                  ry *= s;
               }
               return ( erfc(w*(X(0)-cx-rx))*erfc(-w*(X(0)-cx+rx)) *
                        erfc(w*(X(1)-cy-ry))*erfc(-w*(X(1)-cy+ry)) )/16;
            }
         }
      }
      case 2:
      {
         double x_ = X(0), y_ = X(1), rho, phi;
         rho = hypot(x_, y_);
         phi = atan2(y_, x_);
         return pow(sin(M_PI*rho),2)*sin(3*phi);
      }
      case 3:
      {
         const double f = M_PI;
         return .5*(sin(f*X(0))*sin(f*X(1)) + 1.); // modified by Hennes
      }
      case 4:
      {
         double scale = 0.0225;
         double coef = (0.5/sqrt(scale));
         double slit = (X(0) <= -0.05) || (X(0) >= 0.05) || (X(1) >= 0.7);
         double cone = coef * sqrt(pow(X(0), 2.) + pow(X(1) + 0.5, 2.));
         double hump = coef * sqrt(pow(X(0) + 0.5, 2.) + pow(X(1), 2.));

         return (slit && ((pow(X(0),2.) + pow(X(1)-.5,2.))<=4.*scale)) ? 1. : 0.
                + (1. - cone) * (pow(X(0), 2.) + pow(X(1)+.5, 2.) <= 4.*scale)
                + .25 * (1. + cos(M_PI*hump))
                * ((pow(X(0)+.5, 2.) + pow(X(1), 2.)) <= 4.*scale);
      }
      case 5:
      {
         Vector y(dim);
         for (int i = 0; i < dim; i++) { y(i) = 50. * (x(i) + 1.); }

         if (dim==1)
         {
            mfem_error("This test is not supported in 1D.");
         }
         else if (dim==2)
         {
            std::pair<double, double> p1;
            std::pair<double, double> p2;
            std::pair<double, double> origin;

            // cross
            p1.first=14.; p1.second=3.;
            p2.first=17.; p2.second=26.;
            origin.first = 15.5;
            origin.second = 11.5;
            double rect1=box(p1,p2,-45.,origin,y(0),y(1));
            p1.first=7.; p1.second=10.;
            p2.first=32.; p2.second=13.;
            double rect2=box(p1,p2,-45.,origin,y(0),y(1));
            double cross=get_cross(rect1,rect2);
            // rings
            Vector c(dim);
            c(0) = 40.; c(1) = 40;
            double ring1 = ring(7., 10., c, y);
            c(1) = 20.;
            double ring2 = ring(3., 7., c, y);

            return cross + ring1 + ring2;
         }
         else
         {
            // cross
            double rect1 = box3D(7.,32.,10.,13.,10.,13.,-45.,15.5,11.5,
                                 y(0),y(1),y(2));
            double rect2 = box3D(14.,17.,3.,26.,10.,13.,-45.,15.5,11.5,
                                 y(0),y(1),y(2));
            double rect3 = box3D(14.,17.,10.,13.,3.,26.,-45.,15.5,11.5,
                                 y(0),y(1),y(2));

            double cross = get_cross(get_cross(rect1, rect2), rect3);

            // rings
            Vector c1(dim), c2(dim);
            c1(0) = 40.; c1(1) = 40; c1(2) = 40.;
            c2(0) = 40.; c2(1) = 20; c2(2) = 20.;

            double shell1 = ring(7., 10., c1, y);
            double shell2 = ring(3., 7., c2, y);

            double dom2 = cross + shell1 + shell2;

            // cross
            rect1 = box3D(2.,27.,30.,33.,30.,33.,0.,0.,0.,y(0),y(1),y(2));
            rect2 = box3D(9.,12.,23.,46.,30.,33.,0.,0.,0.,y(0),y(1),y(2));
            rect3 = box3D(9.,12.,30.,33.,23.,46.,0.,0.,0.,y(0),y(1),y(2));

            cross = get_cross(get_cross(rect1, rect2), rect3);

            double ball1 = ring(0., 7., c1, y);
            double ball2 = ring(0., 3., c2, y);
            double shell3 = ring(7., 10., c2, y);

            double dom3 = cross + ball1 + ball2 + shell3;

            double dom1 = 1. - get_cross(dom2, dom3);

            return dom1 + 2.*dom2 + 3.*dom3;
         }
      }
   }
   return 0.0;
}

#ifdef USE_LUA
double lua_inflow_function(const Vector& x)
{
   lua_getglobal(L, "boundary_condition");

   int dim = x.Size();

   double t;
   adv ? t = adv->GetTime() : t = 0.0;

   for (int d = 0; d < dim; d++)
   {
      lua_pushnumber(L, x(d));
   }
   lua_pushnumber(L, t);

   lua_call(L, dim+1, 1);
   double u = (double)lua_tonumber(L, -1);
   lua_pop(L, 1);

   return u;
}
#endif

// Inflow boundary condition (zero for the problems considered in this example)
double inflow_function(const Vector &x)
{
#ifdef USE_LUA
   return lua_inflow_function(x);
#endif

   return 0.0;
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
      case 1:
         return face_dof_id;
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
      case 3:
         return GetLocalFaceDofIndex3D(loc_face_id, face_orient,
                                       face_dof_id, face_dof1D_cnt);
      default: MFEM_ABORT("Dimension too high!"); return 0;
   }
}
