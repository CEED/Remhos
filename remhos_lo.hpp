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

#ifndef MFEM_REMHOS_LO
#define MFEM_REMHOS_LO

#include <string>
#include "mfem.hpp"
#include "remhos_tools.hpp"

namespace mfem
{

// Low-Order Solver.
class LOSolver
{
protected:
   ParFiniteElementSpace &pfes;
   double dt = -1.0; // usually not known at creation, updated later.

public:
   LOSolver(ParFiniteElementSpace &space) : pfes(space) { }

   virtual ~LOSolver() { }

   virtual void UpdateTimeStep(double dt_new) { dt = dt_new; }

   virtual void CalcLOSolution(const Vector &u, Vector &du) const = 0;
};

class Assembly;

class DiscreteUpwind : public LOSolver
{
protected:
   const SparseMatrix &K;
   mutable SparseMatrix D;
   const Array<int> &K_smap;
   const Vector &M_lumped;
   Assembly &assembly;
   const bool update_D, lump_flux;

   void ComputeDiscreteUpwindMatrix() const;
   void ApplyDiscreteUpwindMatrix(ParGridFunction &u, Vector &du) const;

public:
   DiscreteUpwind(ParFiniteElementSpace &space, const SparseMatrix &adv,
                  const Array<int> &adv_smap, const Vector &Mlump,
                  Assembly &asmbly, bool updateD, bool lumpFlux);

   virtual void CalcLOSolution(const Vector &u, Vector &du) const;
};

class ResidualDistribution : public LOSolver
{
protected:
   ParBilinearForm &K;
   Assembly &assembly;
   const Vector &M_lumped;
   const bool subcell_scheme;
   const bool time_dep;

public:
   ResidualDistribution(ParFiniteElementSpace &space, ParBilinearForm &Kbf,
                        Assembly &asmbly, const Vector &Mlump,
                        bool subcell, bool timedep);

   virtual void CalcLOSolution(const Vector &u, Vector &du) const;
};

class HOSolver;

class MassBasedAvg : public LOSolver
{
protected:
   HOSolver &ho_solver;
   const GridFunction *mesh_v;
   bool &dt_check_loc;

   void MassesAndVolumesAtPosition(const ParGridFunction &u,
                                   const GridFunction &x,
                                   Vector &el_mass, Vector &el_vol) const;

public:
  MassBasedAvg(ParFiniteElementSpace &space, HOSolver &hos,
               const GridFunction *mesh_vel, bool &dt_check_loc)
     : LOSolver(space), ho_solver(hos), mesh_v(mesh_vel),
       dt_check_loc(dt_check_loc) { }

  virtual void CalcLOSolution(const Vector &u, Vector &du) const;
};

// Low-Order Refined Solver.
class MassBasedAvgLOR : public LOSolver
{
protected:
  HOSolver &ho_solver;
  GridFunction &mesh_pos;
  GridFunction *submesh_pos;
  const GridFunction *mesh_v;
  GridFunction &submesh_vel;
  Vector start_mesh_pos, start_submesh_pos;
  int lref;
  const int mesh_order;
  bool &dt_check_loc;

public:
  MassBasedAvgLOR(ParFiniteElementSpace &space, HOSolver &hos,
		              GridFunction &pos, GridFunction *sub_pos,
                  const GridFunction *mesh_vel, GridFunction &sub_vel,
                  int lref, const int mesh_order, bool &dt_check_loc)
    : LOSolver(space), ho_solver(hos),
      start_mesh_pos(pos.Size()), start_submesh_pos(sub_vel.Size()),
      mesh_pos(pos), submesh_pos(sub_pos),
      mesh_v(mesh_vel), submesh_vel(sub_vel), lref(lref),
      mesh_order(mesh_order), dt_check_loc(dt_check_loc) { }

  virtual void FCT_Project(DenseMatrix &M,
                           DenseMatrixInverse &M_inv,
                           Vector &m, Vector &x, double y_min,
                           double y_max, Vector &xy) const;

  virtual void NodeShift(const IntegrationPoint &ip,
                         const int &s, Vector &ip_trans,
                         const int &dim, const int &lref) const;

  virtual void CalcLOSolution(const Vector &u, Vector &du) const;

  virtual void CalcLORSolution(ParGridFunction &u_HO,
                               ParGridFunction &u_LOR,
                               ParFiniteElementSpace &fes,
                               ParFiniteElementSpace &fes_LOR,
                               ParMesh &mesh,
                               ParMesh &mesh_lor) const;

  virtual void CalcLORProjection(const GridFunction &x,
                                 ParGridFunction &u_LOR,
                                 ParFiniteElementSpace &fes,
                                 ParFiniteElementSpace &fes_LOR,
                                 const int &order, const int &lref,
                                 int &dim,
                                 DofInfo &dofs, Vector &u_Proj_vec) const;

  virtual double compute_mass(FiniteElementSpace *L2, double massL2,
                              VisItDataCollection &dc, std::string prefix) const;

};


class LumpedHO : public LOSolver
{
protected:
  HOSolver &ho_solver;
  const GridFunction *mesh_v;

public:
  LumpedHO(ParFiniteElementSpace &space, HOSolver &hos,
               const GridFunction *mesh_vel)
     : LOSolver(space), ho_solver(hos), mesh_v(mesh_vel) { }

  virtual void CalcLOSolution(const Vector &u, Vector &du) const;
};


//PA based Residual Distribution
class PAResidualDistribution : public ResidualDistribution
{
protected:
   // Data at quadrature points
   const int quad1D, dofs1D, face_dofs;
   mutable Array<double> D_int, D_bdry;
   mutable Array<double> IntVelocity, BdryVelocity;

public:
   PAResidualDistribution(ParFiniteElementSpace &space, ParBilinearForm &Kbf,
                          Assembly &asmbly, const Vector &Mlump,
                          bool subcell, bool timedep);

   void SampleVelocity(FaceType type) const;

   void SetupPA(FaceType type) const;

   void SetupPA2D(FaceType) const;

   void SetupPA3D(FaceType) const;

   void ApplyFaceTerms(const Vector &x, Vector &y, FaceType type) const;

   void ApplyFaceTerms2D(const Vector &x, Vector &y, FaceType type) const;

   void ApplyFaceTerms3D(const Vector &x, Vector &y, FaceType type) const;

   virtual void CalcLOSolution(const Vector &u, Vector &du) const;
};

class PAResidualDistributionSubcell : virtual public PAResidualDistribution
{

private:
   mutable Array<double> SubCellVel;
   mutable Array<double> subCell_pa_data;
   mutable Array<double> subCellWeights;

   void SampleSubCellVelocity() const;
   mutable bool init_weights;

public:

   PAResidualDistributionSubcell(ParFiniteElementSpace &space,
                                 ParBilinearForm &Kbf,
                                 Assembly &asmbly, const Vector &Mlump,
                                 bool subcell, bool timedep);

   void SetupSubCellPA3D() const;

   void SetupSubCellPA2D() const;

   void SetupSubCellPA() const;

   void ComputeSubCellWeights(Array<double> &subWeights) const;

   void ApplySubCellWeights(const Vector &u, Vector &y) const;

   virtual void CalcLOSolution(const Vector &u, Vector &du) const;
};

} // namespace mfem

#endif // MFEM_REMHOS_LO
