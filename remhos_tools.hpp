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
#include "general/forall.hpp"

namespace mfem
{

Mesh *CartesianMesh(int dim, int mpi_cnt, int elem_per_mpi, bool print,
                    int &par_ref, int **partitioning);

int GetLocalFaceDofIndex(int dim, int loc_face_id, int face_orient,
                         int face_dof_id, int face_dof1D_cnt);

void ExtractBdrDofs(int p, Geometry::Type gtype, DenseMatrix &dofs);

void GetMinMax(const ParGridFunction &g, double &min, double &max);

// Utility function to build a map to the offset of the symmetric entry in a
// sparse matrix.
Array<int> SparseMatrix_Build_smap(const SparseMatrix &A);

// Given a matrix K, matrix D (initialized with same sparsity as K) is computed,
// such that (K+D)_ij >= 0 for i != j.
void ComputeDiscreteUpwindingMatrix(const SparseMatrix &K,
                                    Array<int> smap, SparseMatrix& D);

void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    ParGridFunction &gf, const char *title,
                    int x, int y, int w, int h,
                    const char *keys = NULL, bool vec = false);

class DofInfo;

struct TimingData
{
   // Total times for all major computations.
   StopWatch sw_rhs, sw_L2inv, sw_LO, sw_FCT;

   // Store the number of local L2 dofs.
   const HYPRE_Int L2dof;

   // Iterations for the L2 mass inversion.
   HYPRE_Int L2iter;

   TimingData(const HYPRE_Int l2d) : L2dof(l2d) { }
};

class SmoothnessIndicator
{
private:
   const int type;
   const double param;
   H1_FECollection fec_sub;
   ParFiniteElementSpace pfes_CG_sub;
   ParFiniteElementSpace &pfes_DG;
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
                       ParMesh &subcell_mesh,
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
   bool subcell_scheme;
   FiniteElementSpace *SubFes0, *SubFes1;
   Array <int> smap;
   SparseMatrix D;
   ParBilinearForm* pk;
   VectorCoefficient* coef;
   VectorCoefficient* subcellCoeff;
   const IntegrationRule* irF;
   BilinearFormIntegrator* VolumeTerms;
};

// Class storing information on dofs needed for the low order methods and FCT.
class DofInfo
{
private:
   // 0 is overlap, see ComputeOverlapBounds().
   // 1 is sparcity, see ComputeMatrixSparcityBounds().
   int bounds_type;
   ParMesh *pmesh;
   ParFiniteElementSpace &pfes;

   // The min and max bounds are represented as CG functions of the same order
   // as the solution, thus having 1:1 dof correspondence inside each element.
   H1_FECollection fec_bounds;
   ParFiniteElementSpace pfes_bounds;
   ParGridFunction x_min, x_max;

   // For each DOF on an element boundary, the global index of the DOF on the
   // opposite site is computed and stored in a list. This is needed for lumping
   // the flux contributions, as in the paper. Right now it works on 1D meshes,
   // quad meshes in 2D and 3D meshes of ordered cubes.
   // NOTE: The mesh is assumed to consist of segments, quads or hexes.
   // NOTE: This approach will not work for meshes with hanging nodes.
   void FillNeighborDofs();

   // A list is filled to later access the correct element-global indices given
   // the subcell number and subcell index.
   // NOTE: The mesh is assumed to consist of segments, quads or hexes.
   void FillSubcell2CellDof();

   // Computes the admissible interval of values for each DG dof from the values
   // of all elements that feature the dof at its physical location.
   // A given DOF gets bounds from the elements it touches (in Gauss-Lobatto
   // sense, i.e., a face dof touches two elements, vertex dofs can touch many).
   void ComputeOverlapBounds(const Vector &el_min, const Vector &el_max,
                             Vector &dof_min, Vector &dof_max,
                             Array<bool> *active_el = NULL);

   // A given DOF gets bounds from its own element and its face-neighbors.
   void ComputeMatrixSparsityBounds(const Vector &el_min, const Vector &el_max,
                                    Vector &dof_min, Vector &dof_max,
                                    Array<bool> *active_el = NULL);

public:
   Vector xi_min, xi_max; // min/max values for each dof
   Vector xe_min, xe_max; // min/max values for each element

   DenseMatrix BdrDofs, Sub2Ind;
   DenseTensor NbrDof;

   int numBdrs, numFaceDofs, numSubcells, numDofsSubcell;

   DofInfo(ParFiniteElementSpace &pfes_sltn, int btype = 0);

   // Computes the admissible interval of values for each DG dof from the values
   // of all elements that feature the dof at its physical location.
   void ComputeBounds(const Vector &el_min, const Vector &el_max,
                      Vector &dof_min, Vector &dof_max,
                      Array<bool> *active_el = NULL)
   {
      if (bounds_type == 0)
      {
         ComputeOverlapBounds(el_min, el_max, dof_min, dof_max, active_el);
      }
      else if (bounds_type == 1)
      {
         ComputeMatrixSparsityBounds(el_min, el_max,
                                     dof_min, dof_max, active_el);
      }
      else { MFEM_ABORT("Wrong option for bounds computation."); }
   }

   // Computes the min and max values of u over each element.
   void ComputeElementsMinMax(const Vector &u,
                              Vector &u_min, Vector &u_max,
                              Array<bool> *active_el,
                              Array<bool> *active_dof) const;
};

class Assembly
{
private:
   const int exec_mode;
   const GridFunction &inflow_gf;
   mutable ParGridFunction x_gf;
   BilinearFormIntegrator *VolumeTerms;
   FiniteElementSpace *fes, *SubFes0, *SubFes1;
   Mesh *subcell_mesh;

public:
   Assembly(DofInfo &_dofs, LowOrderMethod &inlom, const GridFunction &inflow,
            ParFiniteElementSpace &pfes, ParMesh *submesh, int mode);

   // Auxiliary member variables that need to be accessed during time-stepping.
   DofInfo &dofs;

   LowOrderMethod &lom;
   // Data structures storing Galerkin contributions. These are updated for
   // remap but remain constant for transport.
   // bdrInt - eq (32).
   // SubcellWeights - above eq (49).
   DenseTensor bdrInt, SubcellWeights;

   void ComputeFluxTerms(const int e_id, const int BdrID,
                         FaceElementTransformations *Trans,
                         LowOrderMethod &lom);

   void ComputeSubcellWeights(const int k, const int m);

   void LinearFluxLumping(const int k, const int nd,
                          const int BdrID, const Vector &x,
                          Vector &y, const Vector &x_nd,
                          const Vector &alpha) const;
   void NonlinFluxLumping(const int k, const int nd,
                          const int BdrID, const Vector &x,
                          Vector &y, const Vector &x_nd,
                          const Vector &alpha) const;

   const FiniteElementSpace * GetFes() {return fes;}

   int GetExecMode() const { return exec_mode;}

   Mesh *GetSubCellMesh() { return subcell_mesh;}
};

// Class for local assembly of M_L M_C^-1 K, where M_L and M_C are the lumped
// and consistent mass matrices and K is the convection matrix. The spaces are
// assumed to be L2 conforming.
class PrecondConvectionIntegrator: public BilinearFormIntegrator
{
private:
#ifndef MFEM_THREAD_SAFE
   DenseMatrix dshape, adjJ, Q_ir;
   Vector shape, vec2, BdFidxT;
#endif
   VectorCoefficient &Q;
   double alpha;

public:
   PrecondConvectionIntegrator(VectorCoefficient &q, double a = 1.0)
      : Q(q) { alpha = a; }
   virtual void AssembleElementMatrix(const FiniteElement &,
                                      ElementTransformation &,
                                      DenseMatrix &);
};

// alpha (q . grad u, v)
class MixedConvectionIntegrator : public BilinearFormIntegrator
{
private:
#ifndef MFEM_THREAD_SAFE
   DenseMatrix dshape, adjJ, Q_ir;
   Vector shape, vec2, BdFidxT;
#endif
   VectorCoefficient &Q;
   double alpha;

public:
   MixedConvectionIntegrator(VectorCoefficient &q, double a = 1.0)
      : Q(q) { alpha = a; }
   virtual void AssembleElementMatrix2(const FiniteElement &tr_el,
                                       const FiniteElement &te_el,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};

} // namespace mfem

#endif // MFEM_REMHOS_TOOLS
