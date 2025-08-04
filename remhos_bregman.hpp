#ifndef REMHOS_BREGMAN_HPP
#define REMHOS_BREGMAN_HPP
#include "mfem.hpp"
#include "remap.hpp"

namespace mfem
{

class SigmoidCoefficient : public GridFunctionCoefficient
{
protected:
   GridFunction &lower, &upper;
public:
   SigmoidCoefficient(GridFunction &latent_gf, GridFunction &lower,
                      GridFunction &upper)
      : GridFunctionCoefficient(&latent_gf), lower(lower), upper(upper) {}
   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      const real_t l = lower.GetValue(T, ip);
      const real_t u = upper.GetValue(T, ip);
      const real_t x = GridFunctionCoefficient::Eval(T, ip);
      return sigmoid(x, l, u);
   }
};

class LogitCoefficient : public GridFunctionCoefficient
{
protected:
   GridFunction &lower, &upper;
public:
   LogitCoefficient(GridFunction &primal_gf, GridFunction &lower,
                    GridFunction &upper)
      : GridFunctionCoefficient(&primal_gf), lower(lower), upper(upper) {}
   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      const real_t l = lower.GetValue(T, ip);
      const real_t u = upper.GetValue(T, ip);
      const real_t x = GridFunctionCoefficient::Eval(T, ip);
      return logit(x, l, u);
   }
};

class DerSigmoidCoefficient : public GridFunctionCoefficient
{
protected:
   GridFunction &lower, &upper;
public:
   DerSigmoidCoefficient(GridFunction &latent_gf, GridFunction &lower,
                         GridFunction &upper)
      : GridFunctionCoefficient(&latent_gf), lower(lower), upper(upper) {}
   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      const real_t l = lower.GetValue(T, ip);
      const real_t u = upper.GetValue(T, ip);
      const real_t x = GridFunctionCoefficient::Eval(T, ip);
      return der_sigmoid(x, l, u);
   }
};

/// @brief Hessian of Differentiable Objective
/// @details At: x -> H_x(F)
///          Mult: x -> y = H_x(F)*y
///          InvMult: x -> y = H_x(F)^{-1}*y
class ObjectiveHessian : public Operator
{
public:
   using Operator::Operator;
   virtual void At(const Vector &x) = 0;
   virtual void InvMult(const Vector &x, Vector &y) const
   {
      MFEM_ABORT("Not Implemented");
   }
};

/// @brief F(u)
/// @details Eval: x -> F(u)
///          Mult: x -> y = grad F(x)
///          GetGradient: x -> y = H_x(F)
class DifferentiableObjective : public Operator
{
protected:
   std::unique_ptr<ObjectiveHessian> hessian;
   MPI_Comm comm;
public:
   using Operator::Operator;
   virtual real_t Eval(const Vector &x) = 0;
   Operator & GetGradient(const Vector &x) const override
   {
      MFEM_ASSERT(hessian, "Hessian is not set");
      hessian->At(x);
      return *hessian;
   }
   MPI_Comm GetComm() { return comm; }
};

class L2Obj : public DifferentiableObjective
{
private:
protected:
   Coefficient &targ_cf;
   Vector proj_targ;
   Vector diff_targ;
   std::unique_ptr<ParGridFunction> targ_gf;
   std::unique_ptr<ParBilinearForm> mass_form;
   std::unique_ptr<QuadratureFunction> targ_qf;
   MPI_Comm comm;
public:

private:
protected:
public:
   L2Obj(ParFiniteElementSpace &fes, Coefficient &targ_cf): targ_cf(targ_cf)
   {
      proj_targ.SetSize(fes.GetVSize());
      targ_gf.reset(new ParGridFunction(&fes, proj_targ.GetData()));
      targ_gf->ProjectCoefficient(targ_cf);
      mass_form.reset(new ParBilinearForm(&fes));
      comm = fes.GetComm();
      mass_form->AddDomainIntegrator(new MassIntegrator());
      mass_form->Assemble();
   }

   L2Obj(QuadratureSpaceBase &qspace, Coefficient &targ_cf): targ_cf(targ_cf)
   {
      proj_targ.SetSize(qspace.GetSize());
      targ_qf.reset(new QuadratureFunction(&qspace, proj_targ.GetData()));
      targ_cf.Project(*targ_qf);
      comm = dynamic_cast<ParMesh*>(qspace.GetMesh())->GetComm();
   }

   real_t Eval(const Vector &x) override
   {
      real_t obj = 0.0;
      if (targ_gf)
      {
         ParGridFunction x_gf(targ_gf->ParFESpace(), x.GetData());
         obj = x_gf.ComputeL2Error(targ_cf);
         obj = obj * obj * 0.5;
      }
      else
      {
         const Vector &weights = targ_qf->GetSpace()->GetWeights();
         for (int i=0; i<proj_targ.Size(); i++)
         {
            const real_t val = proj_targ[i] - x[i];
            obj += weights[i] * val*val;
         }
         obj *= 0.5;
         MPI_Allreduce(MPI_IN_PLACE, &obj, 1, MFEM_MPI_REAL_T,
                       MPI_SUM, comm);
      }
      return obj;
   }
   void Mult(const Vector &x, Vector &y) const override
   {
      subtract(x, proj_targ, y);
   }
};

class LatentVolumeProjector
{
private:
protected:
   const int vdim;
   const Vector &targetVolume;
   ParFiniteElementSpace *fespace=nullptr;
   QuadratureSpace *qspace=nullptr;
   MPI_Comm comm;
   int verbose;
public:
   enum PrimalType
   {
      GF, QF
   };
   PrimalType ptype;

private:
protected:
public:
   void SetVerbose(int lv=0) { verbose = Mpi::Root() ? lv : 0; }

   LatentVolumeProjector(const Vector &targetVolume,
                         ParFiniteElementSpace &fes):vdim(targetVolume.Size()),
      targetVolume(targetVolume),
      fespace(&fes), ptype(GF),
      comm(dynamic_cast<ParFiniteElementSpace*>(&fes)->GetComm()),
      verbose(0)
   { }
   LatentVolumeProjector(const Vector &targetVolume,
                         QuadratureSpaceBase &qspace):vdim(targetVolume.Size()),
      targetVolume(targetVolume), qspace(static_cast<QuadratureSpace*>(&qspace)),
      comm(static_cast<ParMesh*>(qspace.GetMesh())->GetComm()), ptype(QF), verbose(0)
   { }
   MPI_Comm GetComm() { return comm; }
   virtual void Apply(Vector &x, const Vector &lower, const Vector &upper,
                      const real_t step_size,
                      const Vector &search_l, const Vector &search_r,
                      Vector &lambda, int max_iter) = 0;
};


/**
   * @brief Projector for scalar volume int u = targetVolume
**/
class ScalarLatentVolumeProjector : public LatentVolumeProjector
{
private:
   Vector &primal;
   const Vector &pos;
   std::unique_ptr<ParGridFunction> primal_gf;
   std::unique_ptr<QuadratureFunction> primal_qf;
public:
   ScalarLatentVolumeProjector(const Vector &targetVolume,
                               const Vector &pos,
                               ParFiniteElementSpace &fes,
                               Vector &primal)
      : LatentVolumeProjector(targetVolume, fes), primal(primal),
        pos(pos),
        primal_gf(new ParGridFunction(&fes, primal))
   { }

   ScalarLatentVolumeProjector(const Vector &targetVolume,
                               const Vector &pos,
                               QuadratureSpaceBase &qspace,
                               Vector &primal)
      : LatentVolumeProjector(targetVolume, qspace), primal(primal),
        pos(pos)
   { primal_qf.reset(new QuadratureFunction(this->qspace, primal.GetData()));}

   real_t calculateMass(const QuadratureFunction &q1)
   {
      Mesh *mesh = qspace->GetMesh();
      const int NE = mesh->GetNE();
      real_t integral = 0.0;
      for (int e = 0; e < NE; e++)
      {
         const IntegrationRule &ir = qspace->GetElementIntRule(e);
         const int nqp = ir.GetNPoints();

         // Transformation w.r.t. the given mesh positions.
         IsoparametricTransformation Tr;
         mesh->GetElementTransformation(e, pos, &Tr);

         Vector q1_vals(nqp);
         q1.GetValues(e, q1_vals);

         for (int q = 0; q < nqp; q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            Tr.SetIntPoint(&ip);
            integral += Tr.Weight() * ip.weight * q1_vals(q);
         }
      }
      MPI_Allreduce(MPI_IN_PLACE, &integral, 1, MPI_DOUBLE, MPI_SUM, comm);
      return integral;
   }

   real_t calculateMass(const ParGridFunction &g) const
   {
      real_t mass = 0.0;
      const int NE = g.ParFESpace()->GetNE();
      for (int e = 0; e < NE; e++)
      {
         auto el = g.ParFESpace()->GetFE(e);
         auto ir = IntRules.Get(el->GetGeomType(), el->GetOrder() + 2);
         IsoparametricTransformation Tr;
         // Must be w.r.t. the given positions.
         g.ParFESpace()->GetParMesh()->GetElementTransformation(e, pos, &Tr);

         Vector g_vals(ir.GetNPoints());
         g.GetValues(Tr, ir, g_vals);

         for (int q = 0; q < ir.GetNPoints(); q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            Tr.SetIntPoint(&ip);
            mass += Tr.Weight() * ip.weight * g_vals(q);
         }
      }
      MPI_Allreduce(MPI_IN_PLACE, &mass, 1, MPI_DOUBLE, MPI_SUM,
                    g.ParFESpace()->GetComm());
      return mass;
   }

   real_t calculateShiftedMass(const real_t shift, const Vector &x,
                               const Vector &lower, const Vector &upper)
   {
      const bool use_dev = primal.UseDevice() || x.UseDevice();
      const int N = primal.Size();
      auto primal_rw = primal.ReadWrite(use_dev);
      auto x_r = x.Read(use_dev);
      auto l_r = lower.Read(use_dev);
      auto u_r = upper.Read(use_dev);
      mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i) { primal_rw[i] = sigmoid(x_r[i] + shift, l_r[i], u_r[i]); });
      if (qspace)
      {
         return calculateMass(*primal_qf);
      }
      else
      {
         return calculateMass(*primal_gf);
      }
   }

   void Apply(Vector &x, const Vector &lower, const Vector &upper,
              const real_t step_size, const Vector &search_l, const Vector &search_r,
              Vector &lambda, int max_iter) override
   {
      lambda.SetSize(1);
      real_t lambda_lower = search_l[0];
      real_t lambda_upper = search_r[0];

      real_t vol, vol_lower, vol_upper;
      real_t mid;
      while (true)
      {
         mid = lambda_lower;
         vol_lower = calculateShiftedMass(mid, x, lower, upper);
         if (vol_lower > targetVolume[0])
         {
            lambda_lower = lambda_lower + (lambda_lower - lambda_upper);
         }
         else { break;}
      }
      while (true)
      {
         mid = lambda_upper;
         vol_upper = calculateShiftedMass(mid, x, lower, upper);
         if (vol_upper < targetVolume[0])
         {
            lambda_upper = lambda_upper + (lambda_upper - lambda_lower);
         }
         else { break; }
      }
      vol_lower = vol_lower - targetVolume[0];
      vol_upper = vol_upper - targetVolume[0];
      int iter = 0;
      bool was_negative = false;

      // Regula Falsi method with Illinois update
      while (lambda_upper - lambda_lower > 1e-08 && iter < max_iter)
      {
         iter++;
         // convex combination of upper and lower bracket
         mid = (vol_upper*lambda_lower - vol_lower*lambda_upper)/(vol_upper - vol_lower);

         if (verbose > 1)
         {
            out << " mid : " << mid
                << " ( interval: " << lambda_upper - lambda_lower << " )"
                << ": vol-diff = " << std::flush;
         }
         vol = calculateShiftedMass(mid, x, lower, upper);
         if (verbose > 1)
         {
            out << vol - targetVolume[0] << std::endl;
         }
         if (vol < targetVolume[0])
         {
            vol_lower = vol - targetVolume[0];
            lambda_lower = mid;
            // Illinois update
            if (iter > 1 && was_negative)
            {
               vol_upper = 0.5*vol_upper;
            }
            was_negative = true;
         }
         else
         {
            lambda_upper = mid;
            vol_upper = vol - targetVolume[0];
            if (iter > 1 && !was_negative)
            {
               vol_lower = 0.5*vol_lower;
            }
            was_negative = false;
         }
      }
      if (verbose)
      {
         out << "Volume projection converged in " << iter << " iterations\n"
             << "  with volume diff: " << vol - targetVolume[0]
             << " (" << std::fixed << std::setprecision(4)
             << (vol - targetVolume[0])/targetVolume[0]*100 << "%)" << std::endl;
      }

      lambda = mid;
      x += step_size*mid;
   }
};

class IndRhoEVolumeProjector : public LatentVolumeProjector
{
private:
   Vector &primal;
   Vector ind_vec;
   Vector rho_vec;
   Vector E_vec;
   Array<int> offsets;
   const Vector &pos;
   std::unique_ptr<QuadratureFunction> ind_qf;
   std::unique_ptr<QuadratureFunction> rho_qf;
   std::unique_ptr<ParGridFunction> E_gf;
public:

   IndRhoEVolumeProjector(const Vector &targetVolume,
                          const Vector &pos,
                          QuadratureSpaceBase &qspace,
                          ParFiniteElementSpace &fes,
                          Vector &primal)
      : LatentVolumeProjector(targetVolume, qspace), primal(primal),
        pos(pos)
   {
      offsets.SetSize(4);
      offsets[0] = 0;
      offsets[1] = qspace.GetSize();
      offsets[2] = qspace.GetSize();
      offsets[3] = fes.GetVSize();
      offsets.PartialSum();
      int qsize = qspace.GetSize();
      ind_qf.reset(new QuadratureFunction(this->qspace,
                                          primal.GetData() + offsets[0]));
      rho_qf.reset(new QuadratureFunction(this->qspace,
                                          primal.GetData() + offsets[1]));
      E_gf.reset(new ParGridFunction(&fes, primal.GetData() + offsets[2]));
      ind_vec.SetDataAndSize(primal.GetData() + offsets[0], offsets[1] - offsets[0]);
      rho_vec.SetDataAndSize(primal.GetData() + offsets[1], offsets[2] - offsets[1]);
      E_vec.SetDataAndSize(primal.GetData() + offsets[2], offsets[3] - offsets[2]);
   }

   real_t calculateIndMass(const QuadratureFunction &ind)
   {
      Mesh *mesh = qspace->GetMesh();
      const int NE = mesh->GetNE();
      real_t integral = 0.0;
      for (int e = 0; e < NE; e++)
      {
         const IntegrationRule &ir = qspace->GetElementIntRule(e);
         const int nqp = ir.GetNPoints();

         // Transformation w.r.t. the given mesh positions.
         IsoparametricTransformation Tr;
         mesh->GetElementTransformation(e, pos, &Tr);

         Vector ind_vals(nqp);
         ind.GetValues(e, ind_vals);

         for (int q = 0; q < nqp; q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            Tr.SetIntPoint(&ip);
            integral += Tr.Weight() * ip.weight * ind_vals(q);
         }
      }
      MPI_Allreduce(MPI_IN_PLACE, &integral, 1, MPI_DOUBLE, MPI_SUM, comm);
      return integral;
   }

   real_t calculateRhoMass(const QuadratureFunction &ind,
                           const QuadratureFunction &rho)
   {
      Mesh *mesh = qspace->GetMesh();
      const int NE = mesh->GetNE();
      real_t integral = 0.0;
      for (int e = 0; e < NE; e++)
      {
         const IntegrationRule &ir = qspace->GetElementIntRule(e);
         const int nqp = ir.GetNPoints();

         // Transformation w.r.t. the given mesh positions.
         IsoparametricTransformation Tr;
         mesh->GetElementTransformation(e, pos, &Tr);

         Vector ind_vals(nqp);
         Vector rho_vals(nqp);
         ind.GetValues(e, ind_vals);
         rho.GetValues(e, rho_vals);

         for (int q = 0; q < nqp; q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            Tr.SetIntPoint(&ip);
            integral += Tr.Weight() * ip.weight * rho_vals(q)*ind_vals(q);
         }
      }
      MPI_Allreduce(MPI_IN_PLACE, &integral, 1, MPI_DOUBLE, MPI_SUM, comm);
      return integral;
   }

   real_t calculateEMass(const QuadratureFunction &ind,
                         const QuadratureFunction &rho, const ParGridFunction &E)
   {
      Mesh *mesh = qspace->GetMesh();
      const int NE = mesh->GetNE();
      real_t integral = 0.0;
      for (int e = 0; e < NE; e++)
      {
         const IntegrationRule &ir = qspace->GetElementIntRule(e);
         const int nqp = ir.GetNPoints();

         // Transformation w.r.t. the given mesh positions.
         IsoparametricTransformation Tr;
         mesh->GetElementTransformation(e, pos, &Tr);

         Vector ind_vals(nqp);
         Vector rho_vals(nqp);
         Vector E_vals(nqp);
         rho.GetValues(e, rho_vals);
         ind.GetValues(e, ind_vals);
         E.GetValues(e, ir, E_vals);

         for (int q = 0; q < nqp; q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            Tr.SetIntPoint(&ip);
            integral += Tr.Weight() * ip.weight * rho_vals(q)*ind_vals(q)*E_vals(q);
         }
      }
      MPI_Allreduce(MPI_IN_PLACE, &integral, 1, MPI_DOUBLE, MPI_SUM, comm);
      return integral;
   }

   real_t calculateShiftedIndMass(const real_t shift, const Vector &ind_latent,
                                  const Vector &ind_lower, const Vector &ind_upper)
   {
      const bool use_dev = primal.UseDevice() || ind_latent.UseDevice();
      const int N = ind_vec.Size();
      auto ind_rw = ind_vec.ReadWrite(use_dev);
      auto x_r = ind_latent.Read(use_dev);
      auto l_r = ind_lower.Read(use_dev);
      auto u_r = ind_upper.Read(use_dev);
      mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i) { ind_rw[i] = sigmoid(x_r[i] + shift, l_r[i], u_r[i]); });
      return calculateIndMass(*ind_qf);
   }

   real_t calculateShiftedRhoMass(const real_t shift, const Vector &rho_latent,
                                  const Vector &rho_lower, const Vector &rho_upper)
   {
      const bool use_dev = primal.UseDevice() || rho_latent.UseDevice();
      const int N = rho_vec.Size();
      auto rho_rw = rho_vec.ReadWrite(use_dev);
      auto x_r = rho_latent.Read(use_dev);
      auto l_r = rho_lower.Read(use_dev);
      auto u_r = rho_upper.Read(use_dev);
      mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i) { rho_rw[i] = sigmoid(x_r[i] + shift, l_r[i], u_r[i]); });
      return calculateRhoMass(*ind_qf, *rho_qf);
   }

   real_t calculateShiftedEMass(const real_t shift, const Vector &E_latent,
                                const Vector &E_lower, const Vector &E_upper)
   {
      const bool use_dev = primal.UseDevice() || E_latent.UseDevice();
      const int N = rho_vec.Size();
      auto E_rw = E_vec.ReadWrite(use_dev);
      auto x_r = E_latent.Read(use_dev);
      auto l_r = E_lower.Read(use_dev);
      auto u_r = E_upper.Read(use_dev);
      mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i) { E_rw[i] = sigmoid(x_r[i] + shift, l_r[i], u_r[i]); });
      return calculateEMass(*ind_qf, *rho_qf, *E_gf);
   }

   real_t calculateShiftedMass(const int i, const real_t shift,
                               const Vector &latent,
                               const Vector &lower, const Vector &upper)
   {
      if (i == 0)
      {
         return calculateShiftedIndMass(shift, latent, lower, upper);
      }
      else if (i == 1)
      {
         return calculateShiftedRhoMass(shift, latent, lower, upper);
      }
      else
      {
         return calculateShiftedEMass(shift, latent, lower, upper);
      }
   }

   void Apply(Vector &x_all, const Vector &lower_all, const Vector &upper_all,
              const real_t step_size, const Vector &search_l, const Vector &search_r,
              Vector &lambda, int max_iter) override
   {
      lambda.SetSize(3);
      int qsize = this->qspace->GetSize();

      Vector x, lower, upper;
      for (int i=0; i<offsets.Size()-1; i++)
      {
         x.SetDataAndSize(x_all.GetData() + offsets[i], offsets[i+1] - offsets[i]);
         lower.SetDataAndSize(lower_all.GetData() + offsets[i],
                              offsets[i+1] - offsets[i]);
         upper.SetDataAndSize(upper_all.GetData() + offsets[i],
                              offsets[i+1] - offsets[i]);
         real_t lambda_lower = search_l[i];
         real_t lambda_upper = search_r[i];

         real_t vol, vol_lower, vol_upper;
         real_t mid;
         int trial = 0;
         int max_trial = 3;
         for (; trial < max_trial; trial++)
         {
            mid = lambda_lower;
            vol_lower = calculateShiftedMass(i, mid, x, lower, upper);
            if (vol_lower > targetVolume[i])
            {
               if (Mpi::Root())
               {

                  out << i << ": Initial lower bound not feasible. Lower: " << vol_lower << ", "
                      <<
                  "Target: " << targetVolume[i] << std::endl;
               }
               lambda_lower = lambda_lower + (lambda_lower - lambda_upper);
            }
            else { break;}
         }
         if (trial == max_trial)
         {
            if (Mpi::Root())
            {
               out << i << ": Initial lower bound not feasible. Stop Searching" << std::endl;
            }
            lambda_upper = lambda_lower;
         }
         for (; trial < max_trial; trial++)
         {
            mid = lambda_upper;
            vol_upper = calculateShiftedMass(i, mid, x, lower, upper);
            if (vol_upper < targetVolume[i])
            {
               if (Mpi::Root())
               {
                  out << i << ": Initial upper bound not feasible. Upper: " << vol_upper << ", "
                      <<
                  "Target: " << targetVolume[i] << std::endl;
               }
               lambda_upper = lambda_upper + (lambda_upper - lambda_lower);
            }
            else { break; }
         }
         if (trial == max_trial)
         {
            if (Mpi::Root())
            {
               out << i << ": Failed to find feasible bound. " << vol_lower << ", "
                   << vol_upper << ", " <<
                               "Target: " << targetVolume[i] << std::endl;
            }
            lambda_lower = lambda_upper;
         }
         vol_lower = vol_lower - targetVolume[i];
         vol_upper = vol_upper - targetVolume[i];
         int iter = 0;
         bool was_negative = false;
         mid = lambda_lower + lambda_upper;

         // Regula Falsi method with Illinois update
         while (lambda_upper - lambda_lower > 1e-08 && iter < max_iter)
         {
            iter++;
            // convex combination of upper and lower bracket
            mid = (vol_upper*lambda_lower - vol_lower*lambda_upper)/(vol_upper - vol_lower);

            if (verbose > 1 && Mpi::Root())
            {
               out << " mid : " << mid
                   << " ( interval: " << lambda_upper - lambda_lower << " )"
                   << ": vol-diff = " << std::flush;
            }
            vol = calculateShiftedMass(i, mid, x, lower, upper);
            if (verbose > 1 && Mpi::Root())
            {
               out << vol - targetVolume[i] << std::endl;
            }
            if (vol < targetVolume[i])
            {
               vol_lower = vol - targetVolume[i];
               lambda_lower = mid;
               // Illinois update
               if (iter > 1 && was_negative)
               {
                  vol_upper = 0.5*vol_upper;
               }
               was_negative = true;
            }
            else
            {
               lambda_upper = mid;
               vol_upper = vol - targetVolume[i];
               if (iter > 1 && !was_negative)
               {
                  vol_lower = 0.5*vol_lower;
               }
               was_negative = false;
            }
         }
         if (verbose && Mpi::Root())
         {
            out << "Volume projection converged in " << iter << " iterations\n"
                << "  with volume diff: " << vol - targetVolume[i]
                << " (" << std::fixed << std::setprecision(4)
                << (vol - targetVolume[i])/targetVolume[i]*100 << "%)" << std::endl;
         }

         lambda[i] = mid;
         x += step_size*mid;
      }
   }
};

class IndRhoEVolumeProjectorCorrect : public LatentVolumeProjector
{
private:
   Vector &primal;
   Vector ind_vec;
   Vector rho_vec;
   Vector E_vec;
   Array<int> offsets;
   const Vector &pos;
   std::unique_ptr<QuadratureFunction> ind_qf;
   std::unique_ptr<QuadratureFunction> rho_qf;
   std::unique_ptr<ParGridFunction> indrho_gf;
   std::unique_ptr<ParGridFunction> E_gf;
   std::unique_ptr<QuadratureFunction> E_qf;
   std::unique_ptr<L2_FECollection> nodal_fec;
   std::unique_ptr<ParFiniteElementSpace> nodal_fespace;

public:

   IndRhoEVolumeProjectorCorrect(const Vector &targetVolume,
                                 const Vector &pos,
                                 QuadratureSpaceBase &qspace,
                                 ParFiniteElementSpace &fes,
                                 Vector &primal)
      : LatentVolumeProjector(targetVolume, qspace), primal(primal),
        pos(pos)
   {
      offsets.SetSize(4);
      offsets[0] = 0;
      offsets[1] = qspace.GetSize();
      offsets[2] = qspace.GetSize();
      offsets[3] = fes.GetVSize();
      offsets.PartialSum();
      int qsize = qspace.GetSize();
      ind_qf.reset(new QuadratureFunction(this->qspace,
                                          primal.GetData() + offsets[0]));
      rho_qf.reset(new QuadratureFunction(this->qspace,
                                          primal.GetData() + offsets[1]));
      E_gf.reset(new ParGridFunction(&fes, primal.GetData() + offsets[2]));
      ind_vec.SetDataAndSize(primal.GetData() + offsets[0], offsets[1] - offsets[0]);
      rho_vec.SetDataAndSize(primal.GetData() + offsets[1], offsets[2] - offsets[1]);
      E_vec.SetDataAndSize(  primal.GetData() + offsets[2], offsets[3] - offsets[2]);
      E_qf.reset(new QuadratureFunction(this->qspace));
      indrho_gf.reset(new ParGridFunction(&fes));
      this->fespace = &fes;
      nodal_fec.reset(new L2_FECollection(fes.GetOrder(0),
                                          fes.GetMesh()->Dimension()));
      nodal_fespace.reset(new ParFiniteElementSpace(this->fespace->GetParMesh(),
                          nodal_fec.get()));
   }

   real_t calculateIndMass(const QuadratureFunction &ind)
   {
      Mesh *mesh = qspace->GetMesh();
      const int NE = mesh->GetNE();
      real_t integral = 0.0;
      for (int e = 0; e < NE; e++)
      {
         const IntegrationRule &ir = qspace->GetElementIntRule(e);
         const int nqp = ir.GetNPoints();

         // Transformation w.r.t. the given mesh positions.
         IsoparametricTransformation Tr;
         mesh->GetElementTransformation(e, pos, &Tr);

         Vector ind_vals(nqp);
         ind.GetValues(e, ind_vals);

         for (int q = 0; q < nqp; q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            Tr.SetIntPoint(&ip);
            integral += Tr.Weight() * ip.weight * ind_vals(q);
         }
      }
      MPI_Allreduce(MPI_IN_PLACE, &integral, 1, MPI_DOUBLE, MPI_SUM, comm);
      return integral;
   }
   void calculateJacobian(DenseMatrix &J, const Vector &lambda,
                          const Vector &latent, const Vector &lower,
                          const Vector &upper)
   {
      J.SetSize(3);
      J = 0.0;
      DenseMatrix pJ(3); // point Jacobian
      DenseMatrix pJ_prv(3);
      Mesh *mesh = qspace->GetMesh();
      const int NE = mesh->GetNE();
      const int qsize = qspace->GetSize();
      Vector ind_latent_vals, rho_latent_vals, E_latent_vals;
      Vector ind_mins, rho_mins, E_mins;
      Vector ind_maxs, rho_maxs, E_maxs;
      QuadratureFunction ind_latent(qspace, latent.GetData()),
                         ind_min(qspace, lower.GetData()),
                         ind_max(qspace, upper.GetData());
      QuadratureFunction rho_latent(qspace, latent.GetData() + qsize),
                         rho_min(qspace, lower.GetData() + qsize),
                         rho_max(qspace, upper.GetData() + qsize);
      ParGridFunction E_latent(fespace, latent.GetData() + 2*qsize),
                      E_min(fespace, lower.GetData() + 2*qsize),
                      E_max(fespace, upper.GetData() + 2*qsize);
      for (int e = 0; e < NE; e++)
      {
         const IntegrationRule &ir = qspace->GetElementIntRule(e);
         const int nqp = ir.GetNPoints();
         ind_latent_vals.SetSize(nqp); ind_mins.SetSize(nqp); ind_maxs.SetSize(nqp);
         rho_latent_vals.SetSize(nqp); rho_mins.SetSize(nqp); rho_maxs.SetSize(nqp);
         E_latent_vals.SetSize(nqp); E_mins.SetSize(nqp); E_maxs.SetSize(nqp);

         // Transformation w.r.t. the given mesh positions.
         IsoparametricTransformation Tr;
         mesh->GetElementTransformation(e, pos, &Tr);
         ind_latent.GetValues(e, ind_latent_vals);
         ind_min.GetValues(e, ind_mins);
         ind_max.GetValues(e, ind_maxs);
         rho_latent.GetValues(e, rho_latent_vals);
         rho_min.GetValues(e, rho_mins);
         rho_max.GetValues(e, rho_maxs);
         E_latent.GetValues(e, ir, E_latent_vals);
         E_min.GetValues(e, ir, E_mins);
         E_max.GetValues(e, ir, E_maxs);

         for (int q = 0; q < nqp; q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            Tr.SetIntPoint(&ip);
            const real_t w = Tr.Weight() * ip.weight;

            const real_t eta = sigmoid(ind_latent_vals(q), ind_mins(q), ind_maxs(q));
            const real_t rho = sigmoid(rho_latent_vals(q), rho_mins(q), rho_maxs(q));
            const real_t E = sigmoid(E_latent_vals(q), E_mins(q), E_maxs(q));

            const real_t deta = der_sigmoid(ind_latent_vals(q), ind_mins(q),
                                            ind_maxs(q));
            const real_t drho = der_sigmoid(rho_latent_vals(q), rho_mins(q),
                                            rho_maxs(q));
            const real_t dE = der_sigmoid(E_latent_vals(q), E_mins(q), E_maxs(q));

            pJ = 0.0;
            pJ_prv = 0.0;
            real_t pointerr = mfem::infinity();
            int pJ_it = 0;
            while (pointerr > 1e-08)
            {
               pJ_prv = pJ;

               pJ(0,0) = (1     +       lambda[1]*pJ_prv(1, 0) +         lambda[2]*(pJ_prv(1,
                          0)*E   + pJ_prv(2,0)*rho));
               pJ(0,1) = (        rho + lambda[1]*pJ_prv(1, 1) +         lambda[2]*(pJ_prv(1,
                                  1)*E   + pJ_prv(2,1)*rho));
               pJ(0,2) = (              lambda[1]*pJ_prv(1, 2) + rho*E + lambda[2]*(pJ_prv(1,
                                        2)*E   + pJ_prv(2,2)*rho));

               pJ(1,0) = (              lambda[1]*pJ_prv(0, 0) +         lambda[2]*(pJ_prv(0,
                                        0)*E   + pJ_prv(2,0)*eta));
               pJ(1,1) = (        eta + lambda[1]*pJ_prv(0, 1) +         lambda[2]*(pJ_prv(0,
                                  1)*E   + pJ_prv(2,1)*eta));
               pJ(1,2) = (              lambda[1]*pJ_prv(0, 2) + eta*E + lambda[2]*(pJ_prv(0,
                                        2)*E   + pJ_prv(2,2)*eta));

               pJ(2,0) = (                                               lambda[2]*(pJ_prv(0,
                         0)*rho + pJ_prv(1,0)*eta));
               pJ(2,1) = (                                               lambda[2]*(pJ_prv(0,
                         1)*rho + pJ_prv(1,1)*eta));
               pJ(2,2) = (                                     rho*eta + lambda[2]*(pJ_prv(0,
                         2)*rho + pJ_prv(1,2)*eta));
               pJ_prv -= pJ;
               pointerr = pJ_prv.FNorm();
               if (pJ_it++ > 1e04)
               {
                  pJ = 0.0;
                  break;
               }
            }
            Vector dU({deta, drho, dE});
            pJ.LeftScaling(dU);
            J.Add(w, pJ);
         }
      }
      MPI_Allreduce(MPI_IN_PLACE, J.GetData(), 9, MFEM_MPI_REAL_T, MPI_SUM, comm);
      // pJ = J;
      // MultAtB(pJ, pJ, J);
   }

   real_t calculateRhoMass(const QuadratureFunction &ind,
                           const QuadratureFunction &rho)
   {
      Mesh *mesh = qspace->GetMesh();
      const int NE = mesh->GetNE();
      real_t integral = 0.0;
      for (int e = 0; e < NE; e++)
      {
         const IntegrationRule &ir = qspace->GetElementIntRule(e);
         const int nqp = ir.GetNPoints();

         // Transformation w.r.t. the given mesh positions.
         IsoparametricTransformation Tr;
         mesh->GetElementTransformation(e, pos, &Tr);

         Vector ind_vals(nqp);
         Vector rho_vals(nqp);
         ind.GetValues(e, ind_vals);
         rho.GetValues(e, rho_vals);

         for (int q = 0; q < nqp; q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            Tr.SetIntPoint(&ip);
            integral += Tr.Weight() * ip.weight * rho_vals(q)*ind_vals(q);
         }
      }
      MPI_Allreduce(MPI_IN_PLACE, &integral, 1, MPI_DOUBLE, MPI_SUM, comm);
      return integral;
   }

   real_t calculateEMass(const QuadratureFunction &ind,
                         const QuadratureFunction &rho, const ParGridFunction &E)
   {
      Mesh *mesh = qspace->GetMesh();
      const int NE = mesh->GetNE();
      real_t integral = 0.0;
      for (int e = 0; e < NE; e++)
      {
         const IntegrationRule &ir = qspace->GetElementIntRule(e);
         const int nqp = ir.GetNPoints();

         // Transformation w.r.t. the given mesh positions.
         IsoparametricTransformation Tr;
         mesh->GetElementTransformation(e, pos, &Tr);

         Vector ind_vals(nqp);
         Vector rho_vals(nqp);
         Vector E_vals(nqp);
         rho.GetValues(e, rho_vals);
         ind.GetValues(e, ind_vals);
         E.GetValues(e, ir, E_vals);

         for (int q = 0; q < nqp; q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            Tr.SetIntPoint(&ip);
            integral += Tr.Weight() * ip.weight * rho_vals(q)*ind_vals(q)*E_vals(q);
         }
      }
      MPI_Allreduce(MPI_IN_PLACE, &integral, 1, MPI_DOUBLE, MPI_SUM, comm);
      return integral;
   }

   void evaluatePrimalIndValues(const Vector &ind_latent,
                                const Vector &ind_lower, const Vector &ind_upper)
   {
      const bool use_dev = primal.UseDevice() || ind_latent.UseDevice();
      const int N = ind_vec.Size();
      auto ind_rw = ind_vec.ReadWrite(use_dev);
      auto x_r = ind_latent.Read(use_dev);
      auto l_r = ind_lower.Read(use_dev);
      auto u_r = ind_upper.Read(use_dev);
      mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i) { ind_rw[i] = sigmoid(x_r[i], l_r[i], u_r[i]); });
   }

   void evaluatePrimalRhoValues(const Vector &rho_latent,
                                const Vector &rho_lower, const Vector &rho_upper)
   {
      const bool use_dev = primal.UseDevice() || rho_latent.UseDevice();
      const int N = rho_vec.Size();
      auto rho_rw = rho_vec.ReadWrite(use_dev);
      auto x_r = rho_latent.Read(use_dev);
      auto l_r = rho_lower.Read(use_dev);
      auto u_r = rho_upper.Read(use_dev);
      mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i) { rho_rw[i] = sigmoid(x_r[i], l_r[i], u_r[i]); });
   }

   void evaluatePrimalEValues(const Vector &E_latent,
                              const Vector &E_lower, const Vector &E_upper)
   {
      ParGridFunction E_latent_gf(nodal_fespace.get(), (real_t*)nullptr),
                      E_lower_gf(nodal_fespace.get(), (real_t*)nullptr),
                      E_upper_gf(nodal_fespace.get(), (real_t*)nullptr);
      E_latent_gf.SetData(E_latent.GetData());
      E_lower_gf.SetData(E_lower.GetData());
      E_upper_gf.SetData(E_upper.GetData());
      SigmoidCoefficient E_cf(E_latent_gf, E_lower_gf, E_upper_gf);
      E_gf->ProjectCoefficient(E_cf);
   }

   int CalcFixedPoint(const Vector &lambda, const Vector &lower,
                      const Vector &upper,
                      Vector &latent)
   {
      int qsize = this->qspace->GetSize();
      int vsize = this->fespace->GetVSize();
      Vector latent_init(latent.Size());
      latent_init = latent;

      real_t err = mfem::infinity();
      Vector ind_latent(latent.GetData()+offsets[0], offsets[1]-offsets[0]);
      Vector ind_latent_init(latent_init.GetData()+offsets[0], offsets[1]-offsets[0]);
      Vector ind_lower(lower.GetData()+offsets[0], offsets[1]-offsets[0]);
      Vector ind_upper(upper.GetData()+offsets[0], offsets[1]-offsets[0]);

      Vector rho_latent(latent.GetData()+offsets[1], offsets[2]-offsets[1]);
      Vector rho_latent_init(latent_init.GetData()+offsets[1], offsets[2]-offsets[1]);
      Vector rho_lower(lower.GetData()+offsets[1], offsets[2]-offsets[1]);
      Vector rho_upper(upper.GetData()+offsets[1], offsets[2]-offsets[1]);

      Vector E_latent(latent.GetData()+offsets[2], offsets[3]-offsets[2]);
      Vector E_latent_init(latent_init.GetData()+offsets[2], offsets[3]-offsets[2]);
      Vector E_lower(lower.GetData()+offsets[2], offsets[3]-offsets[2]);
      Vector E_upper(upper.GetData()+offsets[2], offsets[3]-offsets[2]);
      MFEM_VERIFY(offsets[3] - offsets[2] - nodal_fespace->GetVSize() == 0,
                  "E latent vector size does not match the grid function size");

      QuadratureFunctionCoefficient ind_cf(*ind_qf);
      QuadratureFunctionCoefficient rho_cf(*rho_qf);
      ProductCoefficient indrho_cf(ind_cf, rho_cf);
      ParGridFunction E_latent_init_gf(nodal_fespace.get(), (real_t*)nullptr);
      E_latent_init_gf.SetData(E_latent_init.GetData());
      GridFunctionCoefficient E_latent_init_cf(&E_latent_init_gf);
      SumCoefficient E_latent_updated_cf(E_latent_init_cf, indrho_cf, 1.0, lambda[2]);

      ParGridFunction E_latent_gf(nodal_fespace.get(), E_latent);
      int it = 0;
      Vector latent_prev(latent_init.Size());
      while (err > 1e-08 && it < 100)
      {
         it++;
         latent_prev = latent;

         // E = sigmoid(psi_E^0 + lambda[2]*rho*ind)
         E_latent_gf.ProjectCoefficient(E_latent_updated_cf);
         evaluatePrimalEValues(E_latent, E_lower, E_upper);
         E_qf->ProjectGridFunction(*E_gf);

         // rho = sigmoid(psi_rho^0 + lambda[1]*ind + lambda[2]*E*ind)
         *rho_qf = *E_qf;
         *rho_qf *= *ind_qf;
         *rho_qf *= lambda[2];
         rho_qf->Add(lambda[1], *ind_qf);
         add(*rho_qf, 1.0, rho_latent_init, rho_latent);
         evaluatePrimalRhoValues(rho_latent, rho_lower, rho_upper);

         // ind = sigmoid(psi_ind^0 + lambda[0] + lambda[1]*rho + lambda[2]*E*rho)
         *ind_qf = *E_qf;
         *ind_qf *= *rho_qf;
         *ind_qf *= lambda[2];
         ind_qf->Add(lambda[1], *rho_qf);
         *ind_qf += lambda[0];
         add(*ind_qf, 1.0, ind_latent_init, ind_latent);
         evaluatePrimalIndValues(ind_latent, ind_lower, ind_upper);

         err = latent.DistanceSquaredTo(latent_prev);
         MPI_Allreduce(MPI_IN_PLACE, &err, 1, MFEM_MPI_REAL_T, MPI_SUM, comm);
         err = std::sqrt(err);

         real_t latent_max = latent.Max();
         MPI_Allreduce(MPI_IN_PLACE, &latent_max, 1, MFEM_MPI_REAL_T, MPI_MAX, comm);
         if (verbose>1) { out << "\tFixedPoint Iter: " << it << " lam: " << lambda[0] << ", " << lambda[1] << ", " << lambda[2] << std::endl; }
         if (verbose>1) { out << "\tFixedPoint Iter: " << it << " err: " << err << std::endl; }
         if (verbose>1) { out << "\tFixedPoint Iter: " << it << " max: " << latent_max << std::endl; }
      }
      return it;
   }

   void Apply(Vector &x_all, const Vector &lower_all, const Vector &upper_all,
              const real_t step_size, const Vector &search_l, const Vector &search_r,
              Vector &lambda, int max_iter) override
   {
      Vector x_all_init(x_all.Size());
      x_all_init = x_all;
      int qsize = this->qspace->GetSize();
      lambda.SetSize(3);
      lambda = 0.0;
      real_t mass_err = mfem::infinity();
      Vector currErr(3);
      int it = 0;
      DenseMatrix Jacobian(3);
      while (mass_err > 1e-10)
      {
         it++;
         if (verbose) { out << "Iter: " << it << std::endl; }
         x_all = x_all_init;
         CalcFixedPoint(lambda, lower_all, upper_all, x_all);
         currErr[0] = calculateIndMass(*ind_qf)-targetVolume[0];
         currErr[1] = calculateRhoMass(*ind_qf, *rho_qf)-targetVolume[1];
         currErr[2] = calculateEMass(*ind_qf, *rho_qf, *E_gf)-targetVolume[2];
         if (verbose) { out << "Ind: " << currErr[0] << " Rho: " << currErr[1] << " E: " << currErr[2] << std::endl; }
         // lambda.Add(-step_size, currErr);
         // Second order Correction. Need to calculate the Jacobian
         // The Jacobian is a 3x3 matrix. It is calculated using integration of point Jacobian.
         // This is not stable.. Need a better way to calculate the Jacobian or use a better nonlinear solver
         calculateJacobian(Jacobian, lambda, x_all, lower_all, upper_all);
         if (verbose) { out << "Jacobian Norm: " << Jacobian.FNorm() << std::endl; }

         bool isSingular = false;
         for (int d=0; d<3; d++)
         {
            if (std::fabs(Jacobian(d,d)) < 1e-9)
            {
               if (verbose) { out << d + 1 << "th Diagonal Component is Zero:" << Jacobian(d,d) << std::endl; }
               isSingular = true;
            }
         }

         if (isSingular)
         {
            if (verbose) { out << "Jacobian is singular. Add Regularization" << std::endl; }
            for (int d=0; d<3; d++) { Jacobian(d,d) += 10; }
         }
         if (Jacobian.CheckFinite()) { Jacobian = 0.0; for (int d=0; d<3; d++) {Jacobian(d,d)=1.0;}}
         else {Jacobian.Invert();}
         // damped Newton step
         Jacobian.AddMult(currErr, lambda, -std::min(1.0, step_size*it));
         // lambda.Add(-step_size, currErr);
         mass_err = currErr.Norml2();
         if (it > max_iter)
         {
            return;
         }
      }
   }
};

class BoxMirrorDescent
{
private:
protected:
   DifferentiableObjective &obj;
   const Vector &lower, &upper;
   Vector &primal;
   Vector grad, xnew;
   LatentVolumeProjector *projector;
   int max_iter;
   real_t tol;
   int verbose = 0;
public:

private:
protected:
public:
   void SetVerbose(int lv=1) { verbose = Mpi::Root() ? lv : 0; }
   BoxMirrorDescent(DifferentiableObjective &obj, Vector &primal,
                    const Vector &lower, const Vector &upper,
                    int max_iter = 1000, real_t tol = 1e-08)
      : obj(obj), primal(primal), lower(lower), upper(upper),
        max_iter(max_iter), tol(tol) { }


   void AddProjector(LatentVolumeProjector &projector)
   {
      this->projector = &projector;
   }

   void UpdatePrimal(const Vector &x)
   {
      for (int i=0; i<x.Size(); i++)
      {
         primal[i] = sigmoid(x[i], lower[i], upper[i]);
      }
   }

   void Step(const Vector &x, real_t step_size, Vector &y)
   {
      UpdatePrimal(x);
      grad.SetSize(primal.Size());
      obj.Mult(primal, grad);

      add(x, -step_size, grad, y);
   }
   void Step(Vector &x, real_t step_size)
   {
      UpdatePrimal(x);
      grad.SetSize(primal.Size());
      obj.Mult(primal, grad);

      x.Add(-step_size, grad);
   }

   void Optimize(Vector &x)
   {
      xnew.SetSize(x.Size());
      real_t step_size = 1.0;
      Vector lambda(1);
      lambda = 0.0;
      if (projector)
      {
         Vector search_l(1), search_r(1);
         search_l[0] = -1e6;
         search_r[0] = 1e6;
         projector->Apply(x, lower, upper, 1.0, search_l, search_r, lambda, max_iter);
      }
      int N = x.Size();
      MPI_Allreduce(MPI_IN_PLACE, &N, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      for (int i=0; i<max_iter; i++)
      {
         /* step_size = i+1.0; */
         if (verbose) { std::cout << "Iter: " << i << " step-size: " << step_size << std::flush; }
         Step(x, step_size);
         if (projector)
         {
            Vector search_l(1), search_r(1);
            // roughly estimate the search range
            // TODO: Update search interval
            search_l[0] = -grad.Normlinf();
            search_r[0] = +grad.Normlinf();
            MPI_Allreduce(MPI_IN_PLACE, search_l.GetData(), 1, MFEM_MPI_REAL_T, MPI_MIN,
                          projector->GetComm());
            MPI_Allreduce(MPI_IN_PLACE, search_r.GetData(), 1, MFEM_MPI_REAL_T, MPI_MAX,
                          projector->GetComm());
            projector->Apply(x, lower, upper, step_size, search_l, search_r, lambda,
                             max_iter);
         }
         UpdatePrimal(x);
         real_t val = obj.Eval(primal);
         if (verbose) { std::cout << " val: " << val; }
         // TDOO: Check convergence using KKT
         real_t kkt_residual = 0.0;
         for (int i=0; i<grad.Size(); i++)
         {
            const real_t direction = -grad[i] + lambda[0];
            const real_t eta = std::max((lower[i] - primal[i])*direction,
                                        (upper[i] - primal[i])*direction);
            kkt_residual += std::fabs(eta);
         }
         kkt_residual /= N;
         MPI_Allreduce(MPI_IN_PLACE, &kkt_residual, 1, MFEM_MPI_REAL_T, MPI_SUM,
                       projector->GetComm());
         if (kkt_residual < tol)
         {
            if (verbose) { std::cout << "Converged: " << kkt_residual << std::endl; }
            break;
         }
         else
         {
            if (verbose) { std::cout << " KKT: " << kkt_residual << std::endl; }
         }
      }
   }
};


}

#endif // REMHOS_LVPP_HPP
