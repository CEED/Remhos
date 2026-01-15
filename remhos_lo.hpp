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

#include "mfem.hpp"

namespace mfem
{

struct TimingData;

// Low-Order Solver.
class LOSolver
{
protected:
   ParFiniteElementSpace &pfes;
   real_t dt = -1.0; // usually not known at creation, updated later.

public:
   LOSolver(ParFiniteElementSpace &space) : pfes(space) { }

   virtual ~LOSolver() { }

   virtual void UpdateTimeStep(real_t dt_new) { dt = dt_new; }

   virtual void CalcLOSolution(const Vector &u, Vector &du) const = 0;

   TimingData *timer = nullptr;
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
   const bool update_D;

   void ComputeDiscreteUpwindMatrix() const;

public:
   DiscreteUpwind(ParFiniteElementSpace &space, const SparseMatrix &adv,
                  const Array<int> &adv_smap, const Vector &Mlump,
                  Assembly &asmbly, bool updateD);

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

   // Temporary HO solution, used only in the next call to CalcLOSolution().
   mutable const Vector *du_HO = nullptr;

   void MassesAndVolumesAtPosition(const ParGridFunction &u,
                                   const GridFunction &x,
                                   Vector &el_mass, Vector &el_vol) const;

public:
   MassBasedAvg(ParFiniteElementSpace &space, HOSolver &hos,
       const GridFunction *mesh_vel)
       : LOSolver(space), ho_solver(hos), mesh_v(mesh_vel) { }

   // Temporary HO solution, used only in the next call to CalcLOSolution().
   void SetHOSolution(Vector &du) { du_HO = &du; }

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
