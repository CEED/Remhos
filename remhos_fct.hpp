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

#ifndef MFEM_REMHOS_FCT
#define MFEM_REMHOS_FCT

#define REMHOS_FCT_DEBUG

#include "mfem.hpp"

namespace mfem
{

class SmoothnessIndicator;

// Monotone, High-order, Conservative Solver.
class FCTSolver
{
protected:
   ParFiniteElementSpace &pfes;
   SmoothnessIndicator *smth_indicator;
   double dt;

public:
   FCTSolver(ParFiniteElementSpace &space,
             SmoothnessIndicator *si, double dt_)
      : pfes(space), smth_indicator(si), dt(dt_) { }

   virtual ~FCTSolver() { }

   virtual void UpdateTimeStep(double dt_) { dt = dt_; }

   // Calculate du that satisfies the following:
   // bounds preservation: u_min_i <= u_i + dt du_i <= u_max_i,
   // conservation:        sum m_i (u_i + dt du_ho_i) = sum m_i (u_i + dt du_i).
   // Some methods utilize du_lo as a backup choice, as it satisfies the above.
   virtual void CalcFCTSolution(const ParGridFunction &u, const Vector &m,
                                const Vector &du_ho, const Vector &du_lo,
                                const Vector &u_min, const Vector &u_max,
                                Vector &du) const = 0;

   virtual void CalcFCTProduct(const ParGridFunction &us, const Vector &m,
                               const Vector &d_us_HO, const Vector &d_us_LO,
                               Vector &s_min, Vector &s_max,
                               const Vector &u_new,
                               const Array<bool> &active_el,
                               const Array<bool> &active_dofs, Vector &d_us)
   {
      MFEM_ABORT("Not implemented for the chosen solver");
   }
};

class FluxBasedFCT : public FCTSolver
{
protected:
   const SparseMatrix &K, &M;
   const Array<int> &K_smap;

   // Temporary computation objects.
   mutable SparseMatrix flux_ij;
   mutable ParGridFunction gp, gm;

   const int iter_cnt;

   void ComputeFluxMatrix(const ParGridFunction &u, const Vector &du_ho,
                          SparseMatrix &flux_mat) const;
   void AddFluxesAtDofs(const SparseMatrix &flux_mat,
                        Vector &flux_pos, Vector &flux_neg) const;
   void ComputeFluxCoefficients(const Vector &u, const Vector &du_lo,
      const Vector &m, const Vector &u_min, const Vector &u_max,
      Vector &coeff_pos, Vector &coeff_neg) const;
   void UpdateSolutionAndFlux(const Vector &du_lo, const Vector &m,
      ParGridFunction &coeff_pos, ParGridFunction &coeff_neg,
      SparseMatrix &flux_mat, Vector &du) const;

public:
   FluxBasedFCT(ParFiniteElementSpace &space,
                SmoothnessIndicator *si, double delta_t,
                const SparseMatrix &adv_mat, const Array<int> &adv_smap,
                const SparseMatrix &mass_mat, int fct_iterations = 1)
      : FCTSolver(space, si, delta_t),
        K(adv_mat), M(mass_mat), K_smap(adv_smap), flux_ij(adv_mat),
        gp(&pfes), gm(&pfes),
        iter_cnt(fct_iterations) { }

   virtual void CalcFCTSolution(const ParGridFunction &u, const Vector &m,
                                const Vector &du_ho, const Vector &du_lo,
                                const Vector &u_min, const Vector &u_max,
                                Vector &du) const;

   virtual void CalcFCTProduct(const ParGridFunction &us, const Vector &m,
                               const Vector &d_us_HO, const Vector &d_us_LO,
                               Vector &s_min, Vector &s_max,
                               const Vector &u_new,
                               const Array<bool> &active_el,
                               const Array<bool> &active_dofs, Vector &d_us);
};

class ClipScaleSolver : public FCTSolver
{
public:
   ClipScaleSolver(ParFiniteElementSpace &space,
                   SmoothnessIndicator *si, double dt_)
      : FCTSolver(space, si, dt_) { }

   virtual void CalcFCTSolution(const ParGridFunction &u, const Vector &m,
                                const Vector &du_ho, const Vector &du_lo,
                                const Vector &u_min, const Vector &u_max,
                                Vector &du) const;
};

// TODO doesn't conserve mass exactly for some reason.
class NonlinearPenaltySolver : public FCTSolver
{
protected:
   void CorrectFlux(Vector &fluxL, Vector &fluxH, Vector &flux_fix) const;
   double get_max_on_cellNi(Vector &fluxH) const;

public:
   NonlinearPenaltySolver(ParFiniteElementSpace &space,
                          SmoothnessIndicator *si, double dt_)
      : FCTSolver(space, si, dt_) { }

   virtual void CalcFCTSolution(const ParGridFunction &u, const Vector &m,
                                const Vector &du_ho, const Vector &du_lo,
                                const Vector &u_min, const Vector &u_max,
                                Vector &du) const;
};

} // namespace mfem

#endif // MFEM_LAGHOS_FCT
