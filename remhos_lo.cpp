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
   int dof_id;
   double xSum, sumFluctSubcellP, sumFluctSubcellN, sumWeightsP,
          sumWeightsN, weightP, weightN, rhoP, rhoN, aux, fluct, eps = 1.E-15;
   Vector xMaxSubcell, xMinSubcell, sumWeightsSubcellP, sumWeightsSubcellN,
          fluctSubcellP, fluctSubcellN, nodalWeightsP, nodalWeightsN;

   // Discretization terms
   du = 0.;
   K.Mult(u, z);

   ParGridFunction u_gf(&pfes);
   u_gf = u;
   u_gf.ExchangeFaceNbrData();
   Vector &u_nd = u_gf.FaceNbrData();

   z.HostReadWrite();
   u.HostRead();
   du.HostReadWrite();
   M_lumped.HostRead();
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
      } //subcell scheme

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

const DofToQuad *get_maps(ParFiniteElementSpace &pfes, Assembly &asmbly)
{
   const FiniteElement *el_trace =
      pfes.GetTraceElement(0, pfes.GetParMesh()->GetFaceBaseGeometry(0));
   return &el_trace->GetDofToQuad(*asmbly.lom.irF, DofToQuad::TENSOR);
}

//====
//Residual Distribution
//
PAResidualDistribution::PAResidualDistribution(ParFiniteElementSpace &space,
                                               ParBilinearForm &Kbf,
                                               Assembly &asmbly,
                                               const Vector &Mlump,
                                               bool subcell, bool timedep)
   : ResidualDistribution(space, Kbf, asmbly, Mlump, subcell, timedep),
     quad1D(get_maps(pfes, assembly)->nqpt),
     dofs1D(get_maps(pfes, assembly)->ndof),
     face_dofs((pfes.GetMesh()->Dimension() ==2) ? quad1D : quad1D * quad1D)
{
   //MFEM_VERIFY(subcell == false, "Subcell scheme not supported with PA");
}

// Taken from DGTraceIntegrator::SetupPA L:145
void PAResidualDistribution::SampleVelocity(FaceType type) const
{
   const int nf = pfes.GetNFbyType(type);
   if (nf == 0) { return; }

   const IntegrationRule *ir = assembly.lom.irF;

   Mesh *mesh = pfes.GetMesh();
   const int dim = mesh->Dimension();
   const int nq = ir->GetNPoints();

   double *vel_ptr = nullptr;
   if (type == FaceType::Interior)
   {
      IntVelocity.SetSize(dim * nq * nf);
      vel_ptr = IntVelocity.HostWrite();
   }

   if (type == FaceType::Boundary)
   {
      BdryVelocity.SetSize(dim * nq * nf);
      vel_ptr = BdryVelocity.HostWrite();
   }

   auto C = mfem::Reshape(vel_ptr, dim, nq, nf);
   Vector Vq(dim);

   int f_idx = 0;
   for (int f = 0; f < pfes.GetNF(); ++f)
   {
      int e1, e2;
      int inf1, inf2;
      mesh->GetFaceElements(f, &e1, &e2);
      mesh->GetFaceInfos(f, &inf1, &inf2);
      int my_face_id = inf1 / 64;

      if ((type==FaceType::Interior && (e2>=0 || (e2<0 && inf2>=0))) ||
          (type==FaceType::Boundary && e2<0 && inf2<0) )
      {

         FaceElementTransformations &T =
            *mesh->GetFaceElementTransformations(f);
         for (int q = 0; q < nq; ++q)
         {
            // Convert to lexicographic ordering
            int iq = ToLexOrdering(dim, my_face_id, quad1D, q);
            T.SetAllIntPoints(&ir->IntPoint(q));
            const IntegrationPoint &eip1 = T.GetElement1IntPoint();
            assembly.lom.coef->Eval(Vq, *T.Elem1, eip1);
            for (int i = 0; i < dim; ++i)
            {
               C(i,iq,f_idx) = Vq(i);
            }
         }
         f_idx++;
      }
   }
}

void PAResidualDistribution::SetupPA(FaceType type) const
{
   const FiniteElementSpace *fes = assembly.GetFes();
   int nf = fes->GetNFbyType(type);
   if (nf == 0) {return;}

   Mesh *mesh = fes->GetMesh();
   int dim = mesh->Dimension();
   mesh->DeleteGeometricFactors();

   if (dim == 2) { return SetupPA2D(type); }
   if (dim == 3) { return SetupPA3D(type); }
}

void PAResidualDistribution::SetupPA2D(FaceType type) const
{
   const int nf = pfes.GetNFbyType(type);
   Mesh *mesh = pfes.GetMesh();
   const int dim = mesh->Dimension();

   const IntegrationRule *ir = assembly.lom.irF;

   const FaceGeometricFactors *geom =
      mesh->GetFaceGeometricFactors(*ir,
                                    FaceGeometricFactors::DETERMINANTS |
                                    FaceGeometricFactors::NORMALS, type);

   auto n = mfem::Reshape(geom->normal.Read(), quad1D, dim, nf);
   auto detJ = mfem::Reshape(geom->detJ.Read(), quad1D, nf);
   const double *w = ir->GetWeights().Read();

   const double *vel_ptr;
   if (type == FaceType::Interior) { vel_ptr = IntVelocity.Read(); }
   if (type == FaceType::Boundary) { vel_ptr = BdryVelocity.Read(); }

   auto vel = mfem::Reshape(vel_ptr, dim, ir->GetNPoints(), nf);

   const int execMode = (int) assembly.GetExecMode();

   if (type == FaceType::Interior)
   {
      //two sides per face
      D_int.SetSize(quad1D*2*nf);
      auto D = mfem::Reshape(D_int.Write(), quad1D, 2, nf);

      MFEM_FORALL(f, nf,
      {
         for (int f_side=0; f_side<2; ++f_side)
         {
            for (int k1 = 0; k1 < quad1D; ++k1)
            {
               int direction = 1 - 2*f_side;

               double vvalnor =
               vel(0,k1,f)*direction*n(k1, 0, f) +
               vel(1,k1,f)*direction*n(k1, 1, f);

               if (execMode == 0)
               {
                  vvalnor = fmin(0., vvalnor); //advection
               }
               else
               {
                  vvalnor = -fmax(0., vvalnor);
               }

               double t_vn = vvalnor* w[k1] * detJ(k1, f);
               D(k1, f_side, f) = - t_vn;
            }

         }//f_side

      });
   }//Interior

   if (type == FaceType::Boundary)
   {
      //Only one side per face on the boundary
      D_bdry.SetSize(quad1D*nf);
      auto D = mfem::Reshape(D_bdry.Write(), quad1D, nf);

      MFEM_FORALL(f, nf,
      {
         for (int k1 = 0; k1 < quad1D; ++k1)
         {
            double vvalnor =
            vel(0,k1,f)*n(k1, 0, f) +
            vel(1,k1,f)*n(k1, 1, f);

            if (execMode == 0)
            {
               vvalnor = fmin(0., vvalnor);
            }
            else
            {
               vvalnor = -fmax(0., vvalnor);
            }

            double t_vn = vvalnor* w[k1] * detJ(k1, f);
            D(k1, f) = - t_vn;
         }
      });
   }//boundary
}

void PAResidualDistribution::SetupPA3D(FaceType type) const
{
   const int nf = pfes.GetNFbyType(type);
   Mesh *mesh = pfes.GetMesh();
   const int dim = mesh->Dimension();

   const IntegrationRule *ir = assembly.lom.irF;

   const FaceGeometricFactors *geom =
      mesh->GetFaceGeometricFactors(*ir,
                                    FaceGeometricFactors::DETERMINANTS |
                                    FaceGeometricFactors::NORMALS, type);

   auto n = mfem::Reshape(geom->normal.Read(), quad1D, quad1D, dim, nf);
   auto detJ = mfem::Reshape(geom->detJ.Read(), quad1D, quad1D, nf);
   const double *w = ir->GetWeights().Read();

   const double *vel_ptr;
   if (type == FaceType::Interior) { vel_ptr = IntVelocity.Read(); }
   if (type == FaceType::Boundary) { vel_ptr = BdryVelocity.Read(); }

   auto vel = mfem::Reshape(vel_ptr, dim,  quad1D, quad1D, nf);

   const int execMode = (int) assembly.GetExecMode();

   if (type == FaceType::Interior)
   {
      D_int.SetSize(quad1D*quad1D*2*nf);
      auto D = mfem::Reshape(D_int.Write(), quad1D, quad1D, 2, nf);
      auto int_weights = mfem::Reshape(w, quad1D, quad1D);

      MFEM_FORALL(f, nf,
      {
         for (int f_side=0; f_side<2; ++f_side)
         {
            for (int k2 = 0; k2 < quad1D; ++k2)
            {
               for (int k1 = 0; k1 < quad1D; ++k1)
               {

                  int direction = 1 - 2*f_side;

                  double vvalnor =
                  vel(0, k1, k2, f)*direction*n(k1, k2, 0, f) +
                  vel(1, k1, k2, f)*direction*n(k1, k2, 1, f) +
                  vel(2, k1, k2, f)*direction*n(k1, k2, 2, f);

                  if (execMode == 0)
                  {
                     vvalnor = fmin(0., vvalnor); //advection
                  }
                  else
                  {
                     vvalnor = -fmax(0., vvalnor);
                  }

                  double t_vn = vvalnor* int_weights(k1,k2) * detJ(k1, k2, f);
                  D(k1, k2, f_side, f) = -t_vn;
               }//k1
            }//k2
         }//f_side
      });
   }//interior

   //boundary
   if (type == FaceType::Boundary)
   {
      D_bdry.SetSize(quad1D*quad1D*nf);
      auto D = mfem::Reshape(D_bdry.Write(), quad1D, quad1D, nf);
      auto int_weights = mfem::Reshape(w, quad1D, quad1D);

      MFEM_FORALL(f, nf,
      {
         for (int k2 = 0; k2 < quad1D; ++k2)
         {
            for (int k1 = 0; k1 < quad1D; ++k1)
            {
               double vvalnor =
               vel(0, k1, k2, f)*n(k1, k2, 0, f) +
               vel(1, k1, k2, f)*n(k1, k2, 1, f) +
               vel(2, k1, k2, f)*n(k1, k2, 2, f);

               if (execMode == 0)
               {
                  vvalnor = fmin(0., vvalnor); //advection
               }
               else
               {
                  vvalnor = -fmax(0., vvalnor);
               }

               double t_vn = vvalnor* int_weights(k1,k2) * detJ(k1, k2, f);
               D(k1, k2, f) = -t_vn;
            }//k2
         }//k1
      });
   }//bdry
}

void PAResidualDistribution::ApplyFaceTerms(const Vector &x, Vector &y,
                                            FaceType type) const
{
   int nf = pfes.GetNFbyType(type);
   if (nf == 0) {return;}

   Mesh *mesh = pfes.GetMesh();
   int dim = mesh->Dimension();

   if (dim == 2) { return ApplyFaceTerms2D(x, y, type); }
   if (dim == 3) { return ApplyFaceTerms3D(x, y, type); }
}

void PAResidualDistribution::ApplyFaceTerms2D(const Vector &x, Vector &y,
                                              FaceType type) const
{
   const int Q1D = quad1D;
   const int D1D = dofs1D;
   const int nf = pfes.GetNFbyType(type);
   const Operator * face_restrict_lex = nullptr;

   const IntegrationRule *ir = assembly.lom.irF;
   const FiniteElement &el_trace =
      *pfes.GetTraceElement(0, pfes.GetMesh()->GetFaceBaseGeometry(0));
   const DofToQuad *maps = &el_trace.GetDofToQuad(*ir, DofToQuad::TENSOR);

   auto B = mfem::Reshape(maps->B.Read(), quad1D, dofs1D);

   if (type == FaceType::Interior)
   {
      face_restrict_lex =
         pfes.GetFaceRestriction(ElementDofOrdering::LEXICOGRAPHIC,type);

      Vector x_loc(face_restrict_lex->Height());
      Vector y_loc(face_restrict_lex->Height());

      face_restrict_lex->Mult(x, x_loc);
      y_loc = 0.0;

      auto X = mfem::Reshape(x_loc.Read(), dofs1D, 2, nf);
      auto Y = mfem::Reshape(y_loc.ReadWrite(), dofs1D, 2, nf);
      auto D = mfem::Reshape(D_int.Read(), quad1D, 2, nf);

      MFEM_FORALL(f, nf,
      {
         constexpr int max_Q1D = MAX_Q1D;

         double Bu0[max_Q1D];
         double Bu1[max_Q1D];

         for (int q=0; q<Q1D; ++q)
         {
            Bu0[q] = 0.0;
            Bu1[q] = 0.0;

            //we are lumping the terms
            for (int d=0; d < D1D; ++d)
            {
               Bu0[q] += B(q,d) * 1.0;
               Bu1[q] += B(q,d) * 1.0;
            }
         }

         //Scale with quadrature data
         double DBu0[max_Q1D];
         double DBu1[max_Q1D];
         for (int q=0; q<Q1D; ++q)
         {
            DBu0[q] = Bu0[q] * D(q, 0, f);
            DBu1[q] = Bu1[q] * D(q, 1, f);
         }

         //Apply Bt
         for (int d=0; d<D1D; ++d)
         {

            double res0(0.0), res1(0.0);
            for (int q=0; q<Q1D; ++q)
            {
               res0 += B(q,d) * DBu0[q];
               res1 += B(q,d) * DBu1[q];
            }

            Y(d, 0, f) = res0 * (X(d,1,f) - X(d,0,f));
            Y(d, 1, f) = res1 * (X(d,0,f) - X(d,1,f));
         }
      });

      face_restrict_lex->MultTranspose(y_loc,y);
   }


   if (type == FaceType::Boundary)
   {

      face_restrict_lex =
         pfes.GetFaceRestriction(ElementDofOrdering::LEXICOGRAPHIC,type,
                                 L2FaceValues::SingleValued);

      Vector x_loc(face_restrict_lex->Height());
      Vector y_loc(face_restrict_lex->Height());

      face_restrict_lex->Mult(x, x_loc);
      y_loc = 0.0;

      auto X = mfem::Reshape(x_loc.Read(), dofs1D, nf);
      auto Y = mfem::Reshape(y_loc.ReadWrite(), dofs1D, nf);
      auto D = mfem::Reshape(D_bdry.Read(), quad1D, nf);

      MFEM_FORALL(f, nf,
      {
         constexpr int max_Q1D = MAX_Q1D;

         double Bu0[max_Q1D];
         for (int q=0; q<Q1D; ++q)
         {
            Bu0[q] = 0.0;

            //we are lumping the terms
            for (int d=0; d < D1D; ++d)
            {
               Bu0[q] += B(q,d) * 1.0;
            }
         }

         //Scale with quadrature data
         double DBu0[max_Q1D];
         for (int q=0; q<Q1D; ++q)
         {
            DBu0[q] = Bu0[q] * D(q, f);
         }

         //Apply Bt
         for (int d=0; d<D1D; ++d)
         {

            double res0(0.0);
            for (int q=0; q<Q1D; ++q)
            {
               res0 += B(q,d) * DBu0[q];
            }

            Y(d, f) = -res0 * X(d,f);
         }
      });

      face_restrict_lex->MultTranspose(y_loc,y);
   }
}

void PAResidualDistribution::ApplyFaceTerms3D(const Vector &x, Vector &y,
                                              FaceType type) const
{
   const int Q1D = quad1D;
   const int D1D = dofs1D;
   const int nf = pfes.GetNFbyType(type);
   const Operator * face_restrict_lex = nullptr;

   const IntegrationRule *ir = assembly.lom.irF;
   const FiniteElement &el_trace =
      *pfes.GetTraceElement(0, pfes.GetMesh()->GetFaceBaseGeometry(0));
   const DofToQuad *maps = &el_trace.GetDofToQuad(*ir, DofToQuad::TENSOR);

   auto B = mfem::Reshape(maps->B.Read(), quad1D, dofs1D);

   if (type == FaceType::Interior)
   {
      face_restrict_lex =
         pfes.GetFaceRestriction(ElementDofOrdering::LEXICOGRAPHIC,type);

      Vector x_loc(face_restrict_lex->Height());
      Vector y_loc(face_restrict_lex->Height());

      face_restrict_lex->Mult(x, x_loc);
      y_loc = 0.0;

      auto X = mfem::Reshape(x_loc.Read(), dofs1D, dofs1D, 2, nf);
      auto Y = mfem::Reshape(y_loc.ReadWrite(), dofs1D, dofs1D, 2, nf);
      auto D = mfem::Reshape(D_int.Read(), quad1D, quad1D, 2, nf);

      MFEM_FORALL(f, nf,
      {
         constexpr int max_Q1D = MAX_Q1D;
         constexpr int max_D1D = MAX_D1D;

         double BX0[max_Q1D][max_D1D];
         double BX1[max_Q1D][max_D1D];
         for (int k2=0; k2<Q1D; ++k2)
         {
            for (int i1=0; i1<D1D; ++i1)
            {

               double res0(0), res1(0);
               for (int i2=0; i2<D1D; ++i2)
               {
                  res0 += B(k2,i2)*1.0;
                  res1 += B(k2,i2)*1.0;
               }
               BX0[k2][i1] = res0;
               BX1[k2][i1] = res1;
            }
         }

         double BBX0[max_Q1D][max_Q1D];
         double BBX1[max_Q1D][max_Q1D];
         for (int k2=0; k2<Q1D; ++k2)
         {
            for (int k1=0; k1<Q1D; ++k1)
            {

               double res0(0.0), res1(0.0);
               for (int i1=0; i1<D1D; ++i1)
               {
                  res0 += B(k1,i1)*BX0[k2][i1];
                  res1 += B(k1,i1)*BX1[k2][i1];
               }
               BBX0[k2][k1] = res0 * D(k1, k2, 0, f);
               BBX1[k2][k1] = res1 * D(k1, k2, 1, f);
            }
         }

         double BDBBX0[max_D1D][max_Q1D];
         double BDBBX1[max_D1D][max_Q1D];
         for (int j2=0; j2<D1D; ++j2)
         {
            for (int k1=0; k1<Q1D; ++k1)
            {

               double res0(0.0), res1(0.0);
               for (int k2=0; k2<Q1D; ++k2)
               {
                  res0 += B(k2, j2) * BBX0[k2][k1];
                  res1 += B(k2, j2) * BBX1[k2][k1];
               }
               BDBBX0[j2][k1] = res0;
               BDBBX1[j2][k1] = res1;
            }
         }

         for (int j2=0; j2<D1D; ++j2)
         {
            for (int j1=0; j1<D1D; ++j1)
            {

               double res0(0.0), res1(0.0);
               for (int k1=0; k1<Q1D; ++k1)
               {
                  res0 += B(k1, j1) * BDBBX0[j2][k1];
                  res1 += B(k1, j1) * BDBBX1[j2][k1];
               }

               Y(j1, j2, 0, f) = res0 * (X(j1,j2,1,f) - X(j1,j2,0,f));
               Y(j1, j2, 1, f) = res1 * (X(j1,j2,0,f) - X(j1,j2,1,f));
            }
         }

      });

      face_restrict_lex->MultTranspose(y_loc,y);
   }


   if (type == FaceType::Boundary)
   {

      face_restrict_lex =
         pfes.GetFaceRestriction(ElementDofOrdering::LEXICOGRAPHIC,type,
                                 L2FaceValues::SingleValued);

      Vector x_loc(face_restrict_lex->Height());
      Vector y_loc(face_restrict_lex->Height());

      face_restrict_lex->Mult(x, x_loc);
      y_loc = 0.0;

      auto X = mfem::Reshape(x_loc.Read(), dofs1D, dofs1D, nf);
      auto Y = mfem::Reshape(y_loc.ReadWrite(), dofs1D, dofs1D, nf);
      auto D = mfem::Reshape(D_bdry.Read(), quad1D, quad1D, nf);

      MFEM_FORALL(f, nf,
      {
         constexpr int max_Q1D = MAX_Q1D;
         constexpr int max_D1D = MAX_D1D;

         double BX0[max_Q1D][max_D1D];
         for (int k2=0; k2<Q1D; ++k2)
         {
            for (int i1=0; i1<D1D; ++i1)
            {

               double res0(0);
               for (int i2=0; i2<D1D; ++i2)
               {
                  res0 += B(k2,i2)*1.0;
               }
               BX0[k2][i1] = res0;
            }
         }

         double BBX0[max_Q1D][max_Q1D];
         for (int k2=0; k2<Q1D; ++k2)
         {
            for (int k1=0; k1<Q1D; ++k1)
            {

               double res0(0.0);
               for (int i1=0; i1<D1D; ++i1)
               {
                  res0 += B(k1,i1)*BX0[k2][i1];
               }
               BBX0[k2][k1] = res0 * D(k1, k2, f);
            }
         }

         double BDBBX0[max_D1D][max_Q1D];
         for (int j2=0; j2<D1D; ++j2)
         {
            for (int k1=0; k1<Q1D; ++k1)
            {

               double res0(0.0);
               for (int k2=0; k2<Q1D; ++k2)
               {
                  res0 += B(k2, j2) * BBX0[k2][k1];
               }
               BDBBX0[j2][k1] = res0;
            }
         }

         for (int j2=0; j2<D1D; ++j2)
         {
            for (int j1=0; j1<D1D; ++j1)
            {

               double res0(0.0);
               for (int k1=0; k1<Q1D; ++k1)
               {
                  res0 += B(k1, j1) * BDBBX0[j2][k1];
               }

               Y(j1, j2, f) = -res0 * X(j1,j2,f);
            }
         }

      });

      face_restrict_lex->MultTranspose(y_loc,y);
   }
}

void PAResidualDistribution::CalcLOSolution(const Vector &u, Vector &du) const
{
   const int ndof = pfes.GetFE(0)->GetDof();
   const int ne = pfes.GetMesh()->GetNE();
   Vector z(u.Size());

   const double eps = 1.E-15;
   const double infinity = numeric_limits<double>::infinity();

   // Discretization terms
   du.UseDevice(true);
   du = 0.;

   //z = Conv * u
   K.Mult(u, z);
   ParGridFunction u_gf(&pfes);
   u_gf = u;
   u_gf.ExchangeFaceNbrData();

   ApplyFaceTerms(u, du, FaceType::Interior);
   ApplyFaceTerms(u, du, FaceType::Boundary);

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
}

//====
//PA Subcell Residual Distribution
//
PASubcellResidualDistribution::PASubcellResidualDistribution
(ParFiniteElementSpace &space,
 ParBilinearForm &Kbf,
 Assembly &asmbly,
 const Vector &Mlump,
 bool subcell, bool timedep)
  : init_weights(true), PAResidualDistribution(space, Kbf, asmbly, Mlump, subcell, timedep)
{
   printf("Using subcell residual distribution scheme! \n");
   //MFEM_VERIFY(subcell == true, "Must be using subcell scheme");
}

void PASubcellResidualDistribution::SampleSubCellVelocity() const
{
   const int dim = assembly.GetSubCellMesh()->Dimension();
   const IntegrationRule *ir;
   if (dim == 2) { ir = &IntRules.Get(Geometry::SQUARE, 1); }
   if (dim == 3) { ir = &IntRules.Get(Geometry::CUBE, 1); }

   Mesh *mesh = assembly.GetSubCellMesh();
   const int NE = mesh->GetNE();
   const int nq = ir->GetNPoints();

   //Q: which to use 0, or 1?
   FiniteElementSpace *SubFes = assembly.lom.SubFes0;

   SubCellVel.SetSize(dim*nq*NE);
   auto V = mfem::Reshape(SubCellVel.HostWrite(), dim, nq, NE);

   DenseMatrix Q_ir;
   for (int e=0; e<NE; ++e)
   {
      ElementTransformation& T = *SubFes->GetElementTransformation(e);
      //Q->Eval
      assembly.lom.coef->Eval(Q_ir, T, *ir);
      for (int q=0; q<nq; ++q)
      {
         for (int i=0; i<dim; ++i)
         {
            V(i,q,e) = Q_ir(i,q);
         }
      }
   }
}

void PASubcellResidualDistribution::SetupSubCellPA() const
{

   const int dim = assembly.GetSubCellMesh()->Dimension();
   if (dim == 2) { return SetupSubCellPA2D(); }
   if (dim == 3) { return SetupSubCellPA3D(); }
   mfem_error("PA Subcell Residual Distribution not supported in 1D \n");
}

//Same as
//bilininteg_convection_pa.cpp::PAConvectionSetup2D
void PASubcellResidualDistribution::SetupSubCellPA2D() const
{
   Mesh *mesh = assembly.GetSubCellMesh();
   const int DIM = mesh->Dimension();
   const int NE = mesh->GetNE();
   const IntegrationRule *ir = &IntRules.Get(Geometry::SQUARE, 1);
   const int NQ = ir->GetNPoints();

   //Q Should this be SubFes0 or SubFes1?
   FiniteElementSpace *SubFes = assembly.lom.SubFes0;
   const FiniteElement &el = *SubFes->GetFE(0);
   ElementTransformation &Trans = *SubFes->GetElementTransformation(0);

   const GeometricFactors *geom = mesh->GetGeometricFactors(*ir,
                                                            GeometricFactors::JACOBIANS);
   const DofToQuad * map = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);

   subCell_pa_data.SetSize(DIM*NQ*NE);
   auto J = Reshape(geom->J.Read(), NQ, DIM, DIM, NE);
   auto q_data = Reshape(subCell_pa_data.Write(), NQ, DIM, NE);
   auto V = Reshape(SubCellVel.Read(), DIM, NQ, NE);
   const double alpha = -1.0;
   const double *W = ir->GetWeights().Read();

   MFEM_FORALL(e, NE,
   {
      for (int q=0; q<NQ; ++q)
      {
         const double J11 = J(q,0,0,e);
         const double J21 = J(q,1,0,e);
         const double J12 = J(q,0,1,e);
         const double J22 = J(q,1,1,e);
         double w = alpha * W[q];


         const double v0 = V(0,q,e);
         const double v1 = V(1,q,e);
         const double wx = w * v0;
         const double wy = w * v1;
         //w*inv(J)
         q_data(q, 0, e) = wx * J22 - wy * J12;
         q_data(q, 1, e) = -wx * J21 + wy * J11;
      }
   });
}

//Same as
//bilininteg_convection_pa.cpp::PAConvectionSetup3D
void PASubcellResidualDistribution::SetupSubCellPA3D() const
{
   Mesh *mesh = assembly.GetSubCellMesh();
   const int DIM = mesh->Dimension();
   const int NE = mesh->GetNE();
   const IntegrationRule *ir = &IntRules.Get(Geometry::CUBE, 1);
   const int NQ = ir->GetNPoints();

   //Q Should this be SubFes0 or SubFes1?
   FiniteElementSpace *SubFes = assembly.lom.SubFes0;
   const FiniteElement &el = *SubFes->GetFE(0);
   ElementTransformation &Trans = *SubFes->GetElementTransformation(0);

   const GeometricFactors *geom = mesh->GetGeometricFactors(*ir,
                                                            GeometricFactors::JACOBIANS);
   const DofToQuad * map = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);

   subCell_pa_data.SetSize(DIM*NQ*NE);
   auto J = Reshape(geom->J.Read(), NQ, DIM, DIM, NE);
   auto q_data = Reshape(subCell_pa_data.Write(), NQ, DIM, NE);
   auto V = Reshape(SubCellVel.Read(), DIM, NQ, NE);
   const double alpha = -1.0;
   const double *W = ir->GetWeights().Read();

   MFEM_FORALL(e, NE,
   {
      for (int q = 0; q < NQ; ++q)
      {
         const double J11 = J(q,0,0,e);
         const double J21 = J(q,1,0,e);
         const double J31 = J(q,2,0,e);
         const double J12 = J(q,0,1,e);
         const double J22 = J(q,1,1,e);
         const double J32 = J(q,2,1,e);
         const double J13 = J(q,0,2,e);
         const double J23 = J(q,1,2,e);
         const double J33 = J(q,2,2,e);
         const double w = alpha * W[q];
         const double v0 = V(0, q, e);
         const double v1 = V(1, q, e);
         const double v2 = V(2, q, e);
         const double wx = w * v0;
         const double wy = w * v1;
         const double wz = w * v2;
         // adj(J)
         const double A11 = (J22 * J33) - (J23 * J32);
         const double A12 = (J32 * J13) - (J12 * J33);
         const double A13 = (J12 * J23) - (J22 * J13);
         const double A21 = (J31 * J23) - (J21 * J33);
         const double A22 = (J11 * J33) - (J13 * J31);
         const double A23 = (J21 * J13) - (J11 * J23);
         const double A31 = (J21 * J32) - (J31 * J22);
         const double A32 = (J31 * J12) - (J11 * J32);
         const double A33 = (J11 * J22) - (J12 * J21);
         // q . J^{-1} = q . adj(J)
         q_data(q,0,e) =  wx * A11 + wy * A12 + wz * A13;
         q_data(q,1,e) =  wx * A21 + wy * A22 + wz * A23;
         q_data(q,2,e) =  wx * A31 + wy * A32 + wz * A33;
      }
   });

}

void PASubcellResidualDistribution::SubCellWeights(Array<double> &subWeights)
const
{
   FiniteElementSpace *SubFes0 = assembly.lom.SubFes0;
   FiniteElementSpace *SubFes1 = assembly.lom.SubFes1;

   Mesh *mesh = assembly.GetSubCellMesh();
   const int DIM = mesh->Dimension();
   const int NE = mesh->GetNE();
   const IntegrationRule *ir = nullptr;
   if (DIM == 2) { ir = &IntRules.Get(Geometry::SQUARE, 1); }
   if (DIM == 3) { ir = &IntRules.Get(Geometry::CUBE, 1); }

   const GeometricFactors *geom =
      mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);

   //Data for subspace 0
   const FiniteElement &el_0 = *SubFes0->GetFE(0);
   ElementTransformation &Trans_0 = *SubFes0->GetElementTransformation(0);
   const DofToQuad *maps_0 = &el_0.GetDofToQuad(*ir, DofToQuad::TENSOR);

   const int dofs1D_0 = maps_0->ndof;
   const int quad1D = maps_0->nqpt; //same in both spaces

   //Data for subspace 1
   const FiniteElement &el_1 = *SubFes1->GetFE(0);
   ElementTransformation &Trans_1 = *SubFes1->GetElementTransformation(0);
   const DofToQuad *maps_1 = &el_1.GetDofToQuad(*ir, DofToQuad::TENSOR);

   const int dofs1D_1 = maps_1->ndof;

   auto B_0 = Reshape(maps_0->B.Read(), quad1D, dofs1D_0);
   auto G_0 = Reshape(maps_0->G.Read(), quad1D, dofs1D_0);

   auto B_1 = Reshape(maps_1->B.Read(), quad1D, dofs1D_1);
   auto G_1 = Reshape(maps_1->G.Read(), quad1D, dofs1D_1);

   if (DIM == 2)
   {
      subWeights.SetSize(dofs1D_1*dofs1D_1*dofs1D_0*dofs1D_0*NE);
      auto D = Reshape(subCell_pa_data.Read(), quad1D, quad1D, DIM, NE);
      auto subWeights_view = Reshape(subWeights.Write(),
                                     dofs1D_1, dofs1D_1, dofs1D_0, dofs1D_0, NE);
      MFEM_FORALL(e, NE,
      {

         for (int j2=0; j2<dofs1D_0; ++j2)
         {
            for (int j1=0; j1<dofs1D_0; ++j1)
            {

               for (int i2=0; i2<dofs1D_1; ++i2)
               {
                  for (int i1=0; i1<dofs1D_1; ++i1)
                  {
                     double val=0;
                     for (int k1=0; k1<quad1D; ++k1)
                     {
                        for (int k2=0; k2<quad1D; ++k2)
                        {
                           val +=  (G_1(k1,i1)*B_1(k2,i2)*D(k1,k2,0,e)
                           + B_1(k1,i1) * G_1(k2, i2) * D(k1,k2,1,e))
                           *B_0(k1,j1)*B_0(k2,j2);
                        }
                     }
                     subWeights_view(i1,i2,j1,j2,e) = val;
                  }
               }
            }
         }
      });
   }

   if (DIM == 3)
   {
      subWeights.SetSize(dofs1D_1*dofs1D_1*dofs1D_1*dofs1D_0*dofs1D_0*dofs1D_0);
      auto D = Reshape(subCell_pa_data.Read(), quad1D, quad1D,quad1D, DIM, NE);
      auto subWeights_view = Reshape(subWeights.Write(),
                                     dofs1D_1, dofs1D_1, dofs1D_1, dofs1D_0, dofs1D_0, dofs1D_0, NE);
      MFEM_FORALL(e, NE,
      {

         for (int j3=0; j3<dofs1D_0; ++j3)
         {
            for (int j2=0; j2<dofs1D_0; ++j2)
            {
               for (int j1=0; j1<dofs1D_0; ++j1)
               {

                  for (int i3=0; i3<dofs1D_1; ++i3)
                  {

                     for (int i2=0; i2<dofs1D_1; ++i2)
                     {
                        for (int i1=0; i1<dofs1D_1; ++i1)
                        {
                           double val=0;
                           for (int k1=0; k1<quad1D; ++k1)
                           {
                              for (int k2=0; k2<quad1D; ++k2)
                              {
                                for (int k3=0; k3<quad1D; ++k3)
                                  {
                                    val +=  (G_1(k1,i1)*B_1(k2,i2)*B_1(k3,i3)*D(k1,k2,k3,0,e)
                                             + B_1(k1,i1) * G_1(k2, i2) * B_1(k3,i3) * D(k1,k2,k3,1,e)
                                             + B_1(k1,i1) * B_1(k2,i2) * G_1(k3, i3) * D(k1,k2,k3,2,e)
                                             )
                                      *B_0(k1,j1) * B_0(k2,j2) * B_0(k3, j3);
                                  }
                              }
                           }
                           subWeights_view(i1,i2,i3,j1,j2,j3,e) = val;
                        }
                     }

                  }
               }
            }
         }
      });
   }

}

void PASubcellResidualDistribution::CalcLOSolution(const Vector &u,
                                                   Vector &du) const
{

   const int ndof = pfes.GetFE(0)->GetDof();
   const int ne = pfes.GetMesh()->GetNE();
   Vector z(u.Size());

   const double gamma = 1.0;
   const double eps = 1.E-15;
   const double infinity = numeric_limits<double>::infinity();

   //Part of subcell scheme
   Vector xMaxSubcell, xMinSubcell, sumWeightsSubcellP, sumWeightsSubcellN,
          fluctSubcellP, fluctSubcellN, nodalWeightsP, nodalWeightsN;
   ///---------------------------------

   // Discretization terms
   du.UseDevice(true);
   du = 0.;

   //z = Conv * u
   K.Mult(u, z);
   ParGridFunction u_gf(&pfes);
   u_gf = u;
   u_gf.ExchangeFaceNbrData();

   ApplyFaceTerms(u, du, FaceType::Interior);
   ApplyFaceTerms(u, du, FaceType::Boundary);

   //initialize to infinity
   assembly.dofs.xe_min =  infinity;
   assembly.dofs.xe_max = -infinity;

   double *xe_min = assembly.dofs.xe_min.ReadWrite();
   double *xe_max = assembly.dofs.xe_max.ReadWrite();

   const double *d_u = u.Read();
   const double *d_z = z.Read();
   const double *d_M_lumped = M_lumped.Read();

   double *d_du = du.ReadWrite();

   const int numSubcells = assembly.dofs.numSubcells;
   const int numDofsSubcell = assembly.dofs.numDofsSubcell;

   //Setup vector sizes for subcell
   fluctSubcellP.SetSize(numSubcells);
   fluctSubcellN.SetSize(numSubcells);
   xMaxSubcell.SetSize(numSubcells);
   xMinSubcell.SetSize(numSubcells);
   sumWeightsSubcellP.SetSize(numSubcells);
   sumWeightsSubcellN.SetSize(numSubcells);
   nodalWeightsP.SetSize(ndof);
   nodalWeightsN.SetSize(ndof);

   for (int k=0; k<ne; ++k)
   {
      for (int m = 0; m < numSubcells; m++)
      {

         if (time_dep)
         {
            assembly.ComputeSubcellWeights(k, m);
         }
      }
   }

   if(time_dep || init_weights)
   {
      SampleSubCellVelocity();
      SetupSubCellPA();  
      SubCellWeights(subCellWeights);
      init_weights = false;
   }

   /*
   printf("Total Size %d %d \n",
          subCellWeights.Size(), assembly.SubcellWeights.TotalSize());
   printf("ne * numSubcells * numSubcells = %d %d %d %d \n",
          ne*numSubcells*numDofsSubcell, ne, numSubcells, numDofsSubcell);
   */
   double error = 0;
   int idx = 0;
   //for (int k=0; k<assembly.SubcellWeights.SizeK(); ++k)
   for (int k=0; k<ne; ++k)
   {
      for (int m = 0; m < numSubcells; m++)
      {
         for (int i = 0; i <numDofsSubcell; i++)
         {
           // printf("idx  = %d \n",idx);
           if(idx >= subCellWeights.Size()) { printf("out of range %d out of %d \n", idx, subCellWeights.Size()); exit(-1); }
            error += (subCellWeights.Read()[idx] - assembly.SubcellWeights(k)(m,i)) *
                     (subCellWeights.Read()[idx] - assembly.SubcellWeights(k)(m,i));
            idx++;
         }

      }
   }

   error = sqrt(error);
   if (error > 1e-12) { printf("error is %g \n", error); exit(-1); }


   //MFEM_FORALL(k, ne,
   for (int k=0; k<ne; ++k)
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

      int dof_id;
      double sumFluctSubcellP = 0.; double sumFluctSubcellN = 0.;
      if (subcell_scheme)
      {
         sumFluctSubcellP = 0.; sumFluctSubcellN = 0.;
         nodalWeightsP = 0.; nodalWeightsN = 0.;

         // compute min-/max-values and the fluctuation for subcells
         for (int m = 0; m < numSubcells; m++)
         {
            xMinSubcell(m) =   numeric_limits<double>::infinity();
            xMaxSubcell(m) = - numeric_limits<double>::infinity();;
            double fluct = 0; double xSum = 0.;


            for (int i = 0; i <numDofsSubcell; i++)
            {
               dof_id = k*ndof + assembly.dofs.Sub2Ind(m, i);
               fluct += assembly.SubcellWeights(k)(m,i) * u(dof_id);
               xMaxSubcell(m) = max(xMaxSubcell(m), u(dof_id));
               xMinSubcell(m) = min(xMinSubcell(m), u(dof_id));
               xSum += u(dof_id);
            }
            sumWeightsSubcellP(m) =numDofsSubcell
                                    * xMaxSubcell(m) - xSum + eps;
            sumWeightsSubcellN(m) =numDofsSubcell
                                    * xMinSubcell(m) - xSum - eps;

            fluctSubcellP(m) = max(0., fluct);
            fluctSubcellN(m) = min(0., fluct);
            sumFluctSubcellP += fluctSubcellP(m);
            sumFluctSubcellN += fluctSubcellN(m);
         }

         for (int m = 0; m < numSubcells; m++)
         {
            for (int i = 0; i <numDofsSubcell; i++)
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
      } //subcell scheme


      for (int i = 0; i < ndof; i++)
      {
         int dof_id = k*ndof+i;
         //eq 46
         double weightP = (xe_max[k] - d_u[dof_id]) / sumWeightsP;
         double weightN = (xe_min[k] - d_u[dof_id]) / sumWeightsN;

         if (subcell_scheme)
         {
            double aux = gamma / (rhoP + eps);
            weightP *= 1. - min(aux * sumFluctSubcellP, 1.);
            weightP += min(aux, 1./(sumFluctSubcellP+eps))*nodalWeightsP(i);

            aux = gamma / (rhoN - eps);
            weightN *= 1. - min(aux * sumFluctSubcellN, 1.);
            weightN += max(aux, 1./(sumFluctSubcellN-eps))*nodalWeightsN(i);
         }


         // (lumpped trace term  + LED convection )/lumpped mass matrix
         d_du[dof_id] = (d_du[dof_id] + weightP * rhoP + weightN * rhoN) /
                        d_M_lumped[dof_id];
      }
   }//);

}

} // namespace mfem
