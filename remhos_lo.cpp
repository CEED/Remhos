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

   // Lump fluxes (for PDU).
   ParGridFunction u_gf(&pfes);
   u_gf = u;
   u_gf.ExchangeFaceNbrData();
   Vector &u_nd = u_gf.FaceNbrData();
   const int ne = pfes.GetNE();
   u.HostRead();
   du.HostReadWrite();
   M_lumped.HostRead();
   for (int k = 0; k < ne; k++)
   {
      // Face contributions.
      for (int f = 0; f < assembly.dofs.numBdrs; f++)
      {
         assembly.LinearFluxLumping(k, ndof, f, u, du, u_nd, alpha);
      }
   }

   const int s = du.Size();
   for (int i = 0; i < s; i++) { du(i) /= M_lumped(i); }
}

void DiscreteUpwind::ComputeDiscreteUpwindMatrix() const
{
   const int *Ip = K.HostReadI(), *Jp = K.HostReadJ(), n = K.Size();

   const double *Kp = K.HostReadData();

   double *Dp = D.HostReadWriteData();
   D.HostReadWriteI(); D.HostReadWriteJ();

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

   const double gamma = 1.0;
   //int dof_id;
   double /*xSum,*/ sumFluctSubcellP, sumFluctSubcellN,
          /*sumWeightsP,
          sumWeightsN, weightP, weightN, rhoP, rhoN,*/ aux, fluct, eps = 1.E-15;
   Vector xMaxSubcell, xMinSubcell, sumWeightsSubcellP, sumWeightsSubcellN,
          fluctSubcellP, fluctSubcellN, nodalWeightsP, nodalWeightsN;

   double infinity = numeric_limits<double>::infinity();

   // Discretization terms
   du = 0.;
   K.Mult(u, z);

   //z = Conv * u

   ParGridFunction u_gf(&pfes);
   u_gf = u;
   u_gf.ExchangeFaceNbrData();
   Vector &u_nd = u_gf.FaceNbrData();

   z.HostReadWrite();
   u.HostRead();
   du.HostReadWrite();
   M_lumped.HostRead();

   // Boundary contributions - stored in du
   //will want this in a seperate kernel to do forall elements
   /*
   for (int k=0; k < ne; ++k)
   {
      for (int f = 0; f < assembly.dofs.numBdrs; f++)
      {
        //assembly.LinearFluxLumping(k, ndof, f, u, du, u_nd, alpha);
      }
   }
   */

   //Linear Flux Lumping forall elements/faces //alpha is 0 for us here
   assembly.LinearFluxLumping_all(ndof, u, du, u_nd, alpha);

#if 1

   //initialize to infinity
   assembly.dofs.xe_min =  infinity;
   assembly.dofs.xe_max = -infinity;

   double *xe_min = assembly.dofs.xe_min.ReadWrite();
   double *xe_max = assembly.dofs.xe_max.ReadWrite();

   const double *d_u = u.Read();
   const double *d_z = z.Read();
   const double *d_M_lumped = M_lumped.Read();

   double *d_du = du.ReadWrite();

   MFEM_FORALL(k, ne,
   {

      // Boundary contributions - stored in du
      // done before this loop

      // Element contributions
      double rhoP(0.), rhoN(0.), xSum(0.);
      for (int j = 0; j < ndof; ++j)
      {
         int dof_id = k*ndof+j;
         xe_max[k] = max(xe_max[k], d_u[dof_id]);
         xe_min[k] = min(xe_min[k], d_u[dof_id]);
         xSum += d_u[dof_id];
         rhoP += max(0., d_z[dof_id]);
         rhoN += min(0., d_z[dof_id]);
      }

      //denominator of equation 47
      double sumWeightsP = ndof*xe_max[k] - xSum + eps;
      double sumWeightsN = ndof*xe_min[k] - xSum - eps;

      for (int i = 0; i < ndof; i++)
      {
         int dof_id = k*ndof+i;
         //eq 46
         double weightP = (xe_max[k] - d_u[dof_id]) / sumWeightsP;
         double weightN = (xe_min[k] - d_u[dof_id]) / sumWeightsN;

         // (lumpped trace term  + LED convection )/lumpped mass matrix
         d_du[dof_id] = (d_du[dof_id] + weightP * rhoP + weightN * rhoN) /
         d_M_lumped[dof_id];
      }

   });

#else //Reference version
   // Monotonicity terms
   for (int k = 0; k < ne; k++)
   {
      // Boundary contributions - stored in du
      // done before this loop

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

      //denominator of equation 47
      sumWeightsP = ndof*assembly.dofs.xe_max(k) - xSum + eps;
      sumWeightsN = ndof*assembly.dofs.xe_min(k) - xSum - eps;

      for (int i = 0; i < ndof; i++)
      {
         dof_id = k*ndof+i;
         //eq 46
         weightP = (assembly.dofs.xe_max(k) - u(dof_id)) / sumWeightsP;
         weightN = (assembly.dofs.xe_min(k) - u(dof_id)) / sumWeightsN;

         // (lumpped trace term  + LED convection )/lumpped mass matrix
         du(dof_id) = (du(dof_id) + weightP * rhoP + weightN * rhoN) /
                      M_lumped(dof_id);
      }
   }
#endif

}

} // namespace mfem
