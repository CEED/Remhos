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

#include "remhos_mono.hpp"
#include "remhos_tools.hpp"

using namespace std;

namespace mfem
{

MonoRDSolver::MonoRDSolver(ParFiniteElementSpace &space,
                           const SparseMatrix &adv_mat,
                           const SparseMatrix &mass_mat, const Vector &Mlump,
                           Assembly &asmbly,
                           SmoothnessIndicator *si,
                           VectorFunctionCoefficient &velocity,
                           bool subcell, bool timedep, bool masslim)
   : MonolithicSolver(space),
     K_mat(adv_mat), M_mat(mass_mat), M_lumped(Mlump),
     assembly(asmbly), smth_indicator(si), scale(pfes.GetNE()),
     subcell_scheme(subcell), time_dep(timedep), mass_lim(masslim)
{
   const int ne = pfes.GetNE(), dim = pfes.GetMesh()->Dimension();
   const int order = pfes.GetOrder(0);
   for (int e = 0; e < ne; e++)
   {
      const FiniteElement* el = pfes.GetFE(e);
      DenseMatrix velEval;
      Vector vval;
      double vmax = 0.;
      ElementTransformation *tr = pfes.GetMesh()->GetElementTransformation(e);
      int qOrdE = tr->OrderW() + 2*el->GetOrder() + 2*max(tr->OrderGrad(el), 0);
      const IntegrationRule *irE = &IntRules.Get(el->GetGeomType(), qOrdE);
      velocity.Eval(velEval, *tr, *irE);

      for (int l = 0; l < irE->GetNPoints(); l++)
      {
         velEval.GetColumnReference(l, vval);
         vmax = max(vmax, vval.Norml2());
      }
      const double el_size = pfes.GetMesh()->GetElementSize(e);
      scale(e) = vmax / (2. * (sqrt(dim) * el_size / order));
   }
}

void MonoRDSolver::CalcSolution(const Vector &u, Vector &du) const
{
   const int ndof = pfes.GetFE(0)->GetDof();
   int dof_id;
   int loc, CtrIt, ctr = 0, max_iter = 100;
   double xSum, sumFluctSubcellP, sumFluctSubcellN, sumWeightsP,
          sumWeightsN, weightP, weightN, rhoP, rhoN, aux, fluct,
          uDotMin, uDotMax, diff, MassP, MassN, alphaGlob, tmp, bndN, bndP,
          gamma = 10., beta = 10., tol = 1.E-8, eps = 1.E-15;
   Vector xMaxSubcell, xMinSubcell, sumWeightsSubcellP, sumWeightsSubcellN,
          fluctSubcellP, fluctSubcellN, nodalWeightsP, nodalWeightsN, d,
          m_it(ndof), uDot(ndof), res(ndof);
   Vector alpha(ndof), alpha1(ndof);
   alpha = 0.0; alpha1 = 1.0;
   Vector z(u.Size());

   const double *Mij = M_mat.GetData();

   if (!mass_lim) { max_iter = -1; }
   const int ne = pfes.GetMesh()->GetNE();

   assembly.dofs.xe_max.HostReadWrite();
   assembly.dofs.xe_min.HostReadWrite();
   u.HostRead();
   for (int k = 0; k < ne; k++)
   {
      assembly.dofs.xe_min(k) = numeric_limits<double>::infinity();
      assembly.dofs.xe_max(k) = -numeric_limits<double>::infinity();

      for (int i = 0; i < ndof; i++)
      {
         dof_id = k*ndof+i;
         assembly.dofs.xe_max(k) = max(assembly.dofs.xe_max(k), u(dof_id));
         assembly.dofs.xe_min(k) = min(assembly.dofs.xe_min(k), u(dof_id));
      }
   }
   assembly.dofs.ComputeElementsMinMax(u, assembly.dofs.xe_min,
                                       assembly.dofs.xe_max, NULL, NULL);
   assembly.dofs.ComputeBounds(assembly.dofs.xe_min, assembly.dofs.xe_max,
                               assembly.dofs.xi_min, assembly.dofs.xi_max);

   // Smoothness indicator.
   ParGridFunction si_val;
   if (smth_indicator)
   {
      smth_indicator->ComputeSmoothnessIndicator(u, si_val);
   }

   // Discretization terms.
   du = 0.;
   K_mat.Mult(u, z);
   d = z;

   ParGridFunction u_gf(&pfes);
   u_gf = u;
   u_gf.ExchangeFaceNbrData();
   Vector &u_nd = u_gf.FaceNbrData();

   // Monotonicity terms
   du.HostReadWrite();
   alpha.HostReadWrite();
   z.HostReadWrite();
   M_lumped.HostRead();
   for (int k = 0; k < ne; k++)
   {
      for (int j = 0; j < ndof; j++)
      {
         dof_id = k*ndof+j;
         alpha(j) = min( 1., beta * min(assembly.dofs.xi_max(dof_id) - u(dof_id),
                                        u(dof_id) - assembly.dofs.xi_min(dof_id))
                         / (max(assembly.dofs.xi_max(dof_id) - u(dof_id),
                                u(dof_id) - assembly.dofs.xi_min(dof_id)) + eps) );

         if (smth_indicator)
         {
            tmp = smth_indicator->DG2CG(dof_id) < 0. ? 1. : si_val(smth_indicator->DG2CG(
                                                                      dof_id));
            bndN = max( 0., tmp * (2.*u(dof_id) - assembly.dofs.xi_max(dof_id)) +
                        (1.-tmp) * assembly.dofs.xi_min(dof_id) );
            bndP = min( 1., tmp * (2.*u(dof_id) - assembly.dofs.xi_min(dof_id)) +
                        (1.-tmp) * assembly.dofs.xi_max(dof_id) );

            if (assembly.dofs.xi_min(dof_id)+assembly.dofs.xi_max(dof_id) > 2.*u(
                   dof_id) + eps)
            {
               alpha(j) = min(1., beta*(u(dof_id) - bndN) /
                              (assembly.dofs.xi_max(dof_id) - u(dof_id) + eps));
            }
            else if (assembly.dofs.xi_min(dof_id)+assembly.dofs.xi_max(dof_id) < 2.*u(
                        dof_id) - eps)
            {
               alpha(j) = min(1., beta*(bndP - u(dof_id)) /
                              (u(dof_id) - assembly.dofs.xi_min(dof_id) + eps));
            }
         }

         // Splitting for volume term.
         du(dof_id) += alpha(j) * z(dof_id);
         z(dof_id) -= alpha(j) * z(dof_id);
      }

      // Face contributions.
      for (int i = 0; i < assembly.dofs.numBdrs; i++)
      {
         assembly.NonlinFluxLumping(k, ndof, i, u, du, u_nd, alpha);
         assembly.NonlinFluxLumping(k, ndof, i, u, d, u_nd, alpha1);
      }

      // Element contributions
      rhoP = rhoN = xSum = 0.;

      for (int j = 0; j < ndof; j++)
      {
         dof_id = k*ndof+j;
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
               loc = assembly.dofs.Sub2Ind(m, i);
               dof_id = k*ndof + loc;
               nodalWeightsP(loc) += fluctSubcellP(m)
                                     * ((xMaxSubcell(m) - u(dof_id))
                                        / sumWeightsSubcellP(m)); // eq. (58)
               nodalWeightsN(loc) += fluctSubcellN(m)
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

         du(dof_id) += weightP * rhoP + weightN * rhoN;
      }

      // Time derivative and mass matrix, if mass_lim = false, max_iter has
      // been set to -1, and the iteration loop is not entered.
      m_it = uDot = 0.;
      for (int it = 0; it <= max_iter; it++)
      {
         for (int i = 0; i < ndof; i++)
         {
            dof_id = k*ndof+i;
            uDot(i) = (du(dof_id) + m_it(i)) / M_lumped(dof_id);
         }

         CtrIt = ctr;

         uDotMin = numeric_limits<double>::infinity();
         uDotMax = -uDotMin;

         for (int i = 0; i < ndof; i++) // eq. (28)
         {
            uDotMin = min(uDotMin, uDot(i));
            uDotMax = max(uDotMax, uDot(i));

            dof_id = k*ndof+i;
            m_it(i) = 0.;
            // NOTE: This will only work in serial.
            for (int j = ndof-1; j >= 0; j--) // run backwards through columns
            {
               // use knowledge of how M looks like
               m_it(i) += Mij[ctr] * (uDot(i) - uDot(j));
               ctr++;
            }
            diff = d(dof_id) - du(dof_id);

            tmp = 0.;
            if (smth_indicator)
            {
               tmp = smth_indicator->DG2CG(dof_id) < 0. ? 1. : si_val(smth_indicator->DG2CG(
                                                                         dof_id));
            }
            m_it(i) += min( 1., max(tmp, abs(m_it(i)) / (abs(diff) + eps)) )
                       * diff; // eq. (27) - (29)
         }

         ctr = CtrIt;
         MassP = MassN = 0.;

         for (int i = 0; i < ndof; i++)
         {
            dof_id = k*ndof+i;
            alpha(i) = min(1., beta * scale(k) * min(assembly.dofs.xi_max(dof_id) - u(
                                                        dof_id),
                                                     u(dof_id) - assembly.dofs.xi_min(dof_id))
                           / (max(uDotMax - uDot(i), uDot(i) - uDotMin) + eps) );

            if (smth_indicator)
            {
               alphaGlob = min( 1., beta * scale(k) * min(1. - u(dof_id), u(dof_id) - 0.)
                                / (max(uDotMax - uDot(i), uDot(i) - uDotMin) + eps) );
               tmp = smth_indicator->DG2CG(dof_id) < 0. ? 1. : si_val(smth_indicator->DG2CG(
                                                                         dof_id));
               alpha(i) = min(max(tmp, alpha(i)), alphaGlob);
            }

            m_it(i) *= alpha(i);
            MassP += max(0., m_it(i));
            MassN += min(0., m_it(i));
         }

         for (int i = 0; i < ndof; i++)
         {
            if (MassP + MassN > eps)
            {
               m_it(i) = min(0., m_it(i)) - max(0., m_it(i)) * MassN / MassP;
            }
            else if (MassP + MassN < -eps)
            {
               m_it(i) = max(0., m_it(i)) - min(0., m_it(i)) * MassP / MassN;
            }
         }

         for (int i = 0; i < ndof; i++)
         {
            dof_id = k*ndof+i;
            res(i) = m_it(i) + du(dof_id) - M_lumped(dof_id) * uDot(i);
         }

         if (res.Norml2() <= tol) { break; }
      }

      for (int i = 0; i < ndof; i++)
      {
         dof_id = k*ndof+i;
         du(dof_id) = (du(dof_id) + m_it(i)) / M_lumped(dof_id);
      }
   }
}

} // namespace mfem
