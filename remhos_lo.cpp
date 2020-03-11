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

#include "remhos_lo.hpp"
#include "remhos_tools.hpp"

using namespace std;

namespace mfem
{

DiscreteUpwind::DiscreteUpwind(ParFiniteElementSpace &space,
                               const SparseMatrix &adv,
                               const Array<int> &adv_smap, const Vector &Mlump,
                               Assembly &asmbly, bool updateD)
   : LOSolver(space),
     K(adv), D(), K_smap(adv_smap), M_lumped(Mlump),
     assembly(asmbly), update_D(updateD)
{
   D = K;
   ComputeDiscreteUpwindMatrix();
}

void DiscreteUpwind::CalcLOSolution(const Vector &u, Vector &du) const
{
   const int ndof = pfes.GetFE(0)->GetDof();
   Vector alpha(ndof); alpha = 0.0;

   // Recompute D due to mesh changes (K changes) in remap mode.
   if (update_D) { ComputeDiscreteUpwindMatrix(); }

   // Discretization and monotonicity terms.
   D.Mult(u, du);

   // Lump fluxes (for PDU), compute min/max, and invert lumped mass matrix.
   ParGridFunction u_gf(&pfes);
   u_gf = u;
   u_gf.ExchangeFaceNbrData();
   Vector &u_nd = u_gf.FaceNbrData();
   const int ne = pfes.GetMesh()->GetNE();
   for (int k = 0; k < ne; k++)
   {
      // Face contributions.
      for (int f = 0; f < assembly.dofs.numBdrs; f++)
      {
         assembly.LinearFluxLumping(k, ndof, f, u, du, u_nd, alpha);
      }

      // Compute min / max over elements (needed for FCT).
      assembly.dofs.xe_min(k) = numeric_limits<double>::infinity();
      assembly.dofs.xe_max(k) = -numeric_limits<double>::infinity();
      for (int j = 0; j < ndof; j++)
      {
         int dof_id = k*ndof + j;
         assembly.dofs.xe_max(k) = max(assembly.dofs.xe_max(k), u(dof_id));
         assembly.dofs.xe_min(k) = min(assembly.dofs.xe_min(k), u(dof_id));
         du(dof_id) /= M_lumped(dof_id);
      }
   }
}

void DiscreteUpwind::ComputeDiscreteUpwindMatrix() const
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
         double kji = Kp[K_smap[k]];
         double dij = fmax(fmax(0.0,-kij),-kji);
         Dp[k] = kij + dij;
         Dp[K_smap[k]] = kji + dij;
         if (i != j) { rowsum += dij; }
      }
      D(i,i) = K(i,i) - rowsum;
   }
}

ResidualDistribution::ResidualDistribution(ParFiniteElementSpace &space,
                                           ParBilinearForm &Kbf,
                                           Assembly &asmbly, const Vector &Mlump,
                                           bool subcell, bool timedep)
   : LOSolver(space),
     K(Kbf), assembly(asmbly),
     M_lumped(Mlump), subcell_scheme(subcell), time_dep(timedep)
{ }

void ResidualDistribution::CalcLOSolution(const Vector &u, Vector &du) const
{
   const int ndof = pfes.GetFE(0)->GetDof();
   const int ne = pfes.GetMesh()->GetNE();
   Vector alpha(ndof); alpha = 0.0;
   Vector z(u.Size());

   int dof_id;
   double xSum, sumFluctSubcellP, sumFluctSubcellN, sumWeightsP,
          sumWeightsN, weightP, weightN, rhoP, rhoN, aux, fluct,
          gamma = 10., eps = 1.E-15;
   Vector xMaxSubcell, xMinSubcell, sumWeightsSubcellP, sumWeightsSubcellN,
          fluctSubcellP, fluctSubcellN, nodalWeightsP, nodalWeightsN;

   // Discretization terms
   du = 0.;
   K.Mult(u, z);

   ParGridFunction u_gf(&pfes);
   u_gf = u;
   u_gf.ExchangeFaceNbrData();
   Vector &u_nd = u_gf.FaceNbrData();

   // Monotonicity terms
   for (int k = 0; k < ne; k++)
   {
      // Boundary contributions
      for (int f = 0; f < assembly.dofs.numBdrs; f++)
      {
         assembly.LinearFluxLumping(k, ndof, f, u, du, u_nd, alpha);
      }

      // Element contributions
      rhoP = rhoN = xSum = 0.;
      assembly.dofs.xe_min(k) =   numeric_limits<double>::infinity();
      assembly.dofs.xe_max(k) = - numeric_limits<double>::infinity();
      for (int j = 0; j < ndof; j++)
      {
         dof_id = k*ndof+j;
         assembly.dofs.xe_max(k) = max(assembly.dofs.xe_max(k), u(dof_id));
         assembly.dofs.xe_min(k) = min(assembly.dofs.xe_min(k), u(dof_id));
         xSum += u(dof_id);
         rhoP += max(0., z(dof_id));
         rhoN += min(0., z(dof_id));
      }

      sumWeightsP = ndof*assembly.dofs.xe_max(k) - xSum + eps;
      sumWeightsN = ndof*assembly.dofs.xe_min(k) - xSum - eps;

      if (subcell_scheme)
      {
         fluctSubcellP.SetSize(assembly.dofs.numSubcells);
         fluctSubcellN.SetSize(assembly.dofs.numSubcells);
         xMaxSubcell.SetSize(assembly.dofs.numSubcells);
         xMinSubcell.SetSize(assembly.dofs.numSubcells);
         sumWeightsSubcellP.SetSize(assembly.dofs.numSubcells);
         sumWeightsSubcellN.SetSize(assembly.dofs.numSubcells);
         nodalWeightsP.SetSize(ndof);
         nodalWeightsN.SetSize(ndof);
         sumFluctSubcellP = sumFluctSubcellN = 0.;
         nodalWeightsP = 0.; nodalWeightsN = 0.;

         // compute min-/max-values and the fluctuation for subcells
         for (int m = 0; m < assembly.dofs.numSubcells; m++)
         {
            xMinSubcell(m) =   numeric_limits<double>::infinity();
            xMaxSubcell(m) = - numeric_limits<double>::infinity();;
            fluct = xSum = 0.;

            if (time_dep)
            {
               assembly.ComputeSubcellWeights(k, m);
            }

            for (int i = 0; i < assembly.dofs.numDofsSubcell; i++)
            {
               dof_id = k*ndof + assembly.dofs.Sub2Ind(m, i);
               fluct += assembly.SubcellWeights(k)(m,i) * u(dof_id);
               xMaxSubcell(m) = max(xMaxSubcell(m), u(dof_id));
               xMinSubcell(m) = min(xMinSubcell(m), u(dof_id));
               xSum += u(dof_id);
            }
            sumWeightsSubcellP(m) = assembly.dofs.numDofsSubcell
                                    * xMaxSubcell(m) - xSum + eps;
            sumWeightsSubcellN(m) = assembly.dofs.numDofsSubcell
                                    * xMinSubcell(m) - xSum - eps;

            fluctSubcellP(m) = max(0., fluct);
            fluctSubcellN(m) = min(0., fluct);
            sumFluctSubcellP += fluctSubcellP(m);
            sumFluctSubcellN += fluctSubcellN(m);
         }

         for (int m = 0; m < assembly.dofs.numSubcells; m++)
         {
            for (int i = 0; i < assembly.dofs.numDofsSubcell; i++)
            {
               const int loc_id = assembly.dofs.Sub2Ind(m, i);
               dof_id = k*ndof + loc_id;
               nodalWeightsP(loc_id) += fluctSubcellP(m)
                                     * ((xMaxSubcell(m) - u(dof_id))
                                        / sumWeightsSubcellP(m)); // eq. (58)
               nodalWeightsN(loc_id) += fluctSubcellN(m)
                                     * ((xMinSubcell(m) - u(dof_id))
                                        / sumWeightsSubcellN(m)); // eq. (59)
            }
         }
      }

      for (int i = 0; i < ndof; i++)
      {
         dof_id = k*ndof+i;
         weightP = (assembly.dofs.xe_max(k) - u(dof_id)) / sumWeightsP;
         weightN = (assembly.dofs.xe_min(k) - u(dof_id)) / sumWeightsN;

         if (subcell_scheme)
         {
            aux = gamma / (rhoP + eps);
            weightP *= 1. - min(aux * sumFluctSubcellP, 1.);
            weightP += min(aux, 1./(sumFluctSubcellP+eps))*nodalWeightsP(i);

            aux = gamma / (rhoN - eps);
            weightN *= 1. - min(aux * sumFluctSubcellN, 1.);
            weightN += max(aux, 1./(sumFluctSubcellN-eps))*nodalWeightsN(i);
         }

         du(dof_id) = (du(dof_id) + weightP * rhoP + weightN * rhoN) /
                      M_lumped(dof_id);
      }
   }
}

} // namespace mfem
