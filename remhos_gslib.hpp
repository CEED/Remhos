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

#ifndef MFEM_REMHOS_GSLIB
#define MFEM_REMHOS_GSLIB

#include "mfem.hpp"
#include "remhos_bregman.hpp"

namespace mfem
{

void InitializeQuadratureFunction(Coefficient &c,
                                  const Vector &pos_mesh,
                                  QuadratureFunction &q,
                                  const Array<bool> *bool_quads = nullptr);

void VisQuadratureFunction(ParMesh &pmesh, QuadratureFunction &q,
                           std::string info, int x, int y);

// How to choose the bounds for a given DOF on the final mesh.
// ELEM_INIT:  find its corresponding elem on the initial mesh, take min / max.
//             * piecewise constant initial solution -> no room to move.
//             * DOFs inside one final element can have varying bounds.
// ELEM_FINAL: interpolate, go over its elem on the final mesh, take min / max.
//             * might be restrictive - sees only some of the initial values.
//             * DOFs inside one final element have the same bounds.
// ELEM_BOTH:  take min / max over both ELEM_INIT and ELEM_FINAL bounds.
//             * this is the most diffusive approach (widest bounds).
enum BoundsType {ELEM_INIT, ELEM_FINAL, ELEM_BOTH};

class InterpolationRemap
{
private:
   int myid;

   // This class will not change the node positions of this Mesh object.
   ParMesh &pmesh_init;
   // Mesh on the final mesh positions.
   ParMesh pmesh_final;

   // Initial mesh node positions.
   const Vector pos_init;

   ParFiniteElementSpace *pfes_e = nullptr, *pfes_v = nullptr;
   QuadratureSpace *qspace = nullptr;

   // Positions of the DOFs of pfes, for the given mesh positions.
   void GetDOFPositions(const ParFiniteElementSpace &pfes,
                        const Vector &pos_mesh, Vector &pos_dofs);
   // Positions of the quads of qspace, for the given mesh positions.
   void GetQuadPositions(const QuadratureSpace &qspace,
                         const Vector &pos_mesh,
                         Vector &pos_quads);

   // Mass of g for the given mesh positions.
   double Mass(const Vector &pos, const ParGridFunction &g);

   real_t ObjectiveGF(const ParGridFunction &g_interp, const ParGridFunction &g);
   real_t ObjectiveVecGF(const ParGridFunction &g_interp, const ParGridFunction &g);
   real_t ObjectiveQF(const Vector &g_interp, const Vector &g);

   // Computes volume / mass / internal energy / total energy:
   //   integral(ind * rho * e + 0.5 ind rho v^2).
   //   * whenever ind/rho/e is nullptr, their values are taken as 1.
   //   * if v is nullptr, its value if taken as 0 (no second term).
   //
   // Computes momentum in some direction when v != nullptr, e = nullptr:
   //   integral(ind * rho * v_comp).
   //
   // Uses the given mesh positions.
   // Uses the IntegrationRule of the QuadratureFunctions if these are given.
   double Integrate(const Vector &pos, const QuadratureFunction *ind,
                    const QuadratureFunction *rho,
                    const ParGridFunction *e,
                    const ParGridFunction *v, int comp = 0);

   // Computes bounds for the DOFs of pfes, at the mesh positions given
   // by pos_final. The bounds are determined by the values of g_init, which
   // is defined with respect on the initial mesh.
   void CalcDOFBounds(const ParGridFunction &g_init,
                      const ParFiniteElementSpace &pfes,
                      const Vector &pos_final,
                      Vector &g_min, Vector &g_max, bool use_el_nbr,
                      Array<bool> *active_el = nullptr);
   // Computes bounds for quadrature values, at the mesh positions given
   // by pos_final. The bounds are determined by the values of qf_init, which
   // is defined with respect on the initial mesh.
   void CalcQuadBounds(const QuadratureFunction &qf_init,
                       const QuadratureFunction &qf_interp,
                       const Vector &pos_final,
                       Vector &g_min, Vector &g_max, BoundsType bounds_type);

   void CleanEmptyZones(QuadratureFunction &ind_interp,
                        Vector &ind_min, Vector &ind_max);

   void CalcRhoBounds(const QuadratureFunction &rho_interp,
                      const QuadratureFunction &ind_interp,
                      const Vector &ind_max,
                      Vector &rho_min, Vector &rho_max);

   void UpdateRhoInterp(QuadratureFunction &rho_interp,
                        Vector &rho_min, Vector &rho_max);

   void CalcEBounds(const ParGridFunction &e_interp,
                    const Vector &ind_max,
                    Vector &e_min, Vector &e_max);

   void UpdateEInterp(ParGridFunction &e_interp,
                      Vector &e_min, Vector &e_max);

   void CalcVBounds(const ParGridFunction &v_interp,
                    Vector &v_min, Vector &v_max);

   void CheckBounds(int myid, const Vector &v,
                    const Vector &v_min, const Vector &v_max);

   void GetTargetValues(const Vector &interp,
                    const Vector &min, const Vector &max, Vector &target);

   void ComputePressure(const Vector &pos,
                           const QuadratureFunction &rho_,
                           const ParGridFunction &e_,
                           QuadratureFunction &pressure);

public:
   InterpolationRemap(ParMesh &m)
       : myid(m.GetMyRank()), pmesh_init(m), pmesh_final(pmesh_init, true),
         pos_init(*pmesh_init.GetNodes()) { }

   void SetQuadratureSpace(QuadratureSpace &qs) { qspace = &qs; }
   void SetEnergyFESpace(ParFiniteElementSpace &es) { pfes_e = &es; }
   void SetVelocityFESpace(ParFiniteElementSpace &vs) { pfes_v = &vs; }

   // Remap of an L2 ParGridFunction.
   void Remap(const ParGridFunction &u_init, const Vector &pos_final,
              Vector &u_final, int opt_type);

   // Remap of a QuadratureFunction.
   void Remap(const QuadratureFunction &u_init, const Vector &pos_final,
              Vector &u_final, int opt_type);

   // Remap of an analytic function.
   // Same as projecting the function to the final mesh.
   void Remap(std::function<real_t(const Vector &)> func, double mass,
              const Vector &pos_final, ParGridFunction &u_final,
              int opt_type);

   // Remap of coupled
   // (indicator, density, specific internal energy, velocity) if remap_v or
   // (indicator, density, specific internal energy) if remap_v = false,
   // for a single material (no coupling between materials).
   void RemapHydro(const Vector &ind_rho_e_v_0, bool remap_v,
                   Array<bool> &active_el_0,
                   const Vector &pos_final,
                   Vector &ind_rho_e_v, int opt_type);

   bool visualization = true;
   bool h1_seminorm   = false;
   bool subprob       = true;
   int  max_iter      = 100;
   real_t atol        = 1e-08;
   real_t rtol        = 1e-08;
   hiop::hiopInterfaceBase::WeightedSpaceType weightedSpace =
       hiop::hiopInterfaceBase::WeightedSpaceType::Euclidean;
};

} // namespace mfem

#endif // MFEM_REMHOS_GSLIB
