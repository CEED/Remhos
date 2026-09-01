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

#include "remhos_lvpp.hpp"

#include <limits>

namespace mfem
{

void Dykstra::Project(Vector &projected_x)
{
   const int num_con = constraints.NumRows();
   Vector con_res(num_con);
   const int N = projected_x.Size();
   int total_dof;
   MPI_Allreduce(&N, &total_dof, 1, MPI_INT, MPI_SUM, comm);
   if (Mpi::Root())
   {
      out << "Dykstra::Project with " << num_con << " constraints over " << total_dof
          << std::endl;
   }

   // constraints.Mult already updates the shared processed point (manual_update
   // is never enabled), so no explicit Update() is needed here.
   constraints.Mult(projected_x, con_res);
   if (con_res.Normlinf() < tol)
   {
      if (Mpi::Root())
      {
         out << "Constraint violation is already small enough. Terminate without iterations"
             << std::endl;
      }
      return;
   }

   std::vector<std::unique_ptr<Vector>> q(num_con + enforce_sum_to_one);
   Vector grad(N), deriv(N);
   for (int i = 0; i < num_con + enforce_sum_to_one; ++i)
   {
      q[i] = std::make_unique<Vector>(N);
      *q[i] = 0.0;
   }

   Vector psi(N);
   psi = 0.0;
   MapPrimal(projected_x, xmin, xmax, psi);
   Vector psi_aux(psi);
   Vector psi_prev(psi);
   Vector psi_full_prev(psi);
   Vector x_full_prev(psi);
   Vector primal_dir(N);
   Vector latent_dir(N);
   Vector merit_grad(N);
   Vector con_val(1);
   Vector con_sgn(num_con);
   // return;
   for (int iter=0; iter<max_iter; iter++)
   {
      psi_full_prev = psi;
      x_full_prev = projected_x;
      // Update residual before projection
      if (shared_constraints) { shared_constraints->Update(projected_x); }
      constraints.Mult(projected_x, con_res);
      // Baseline from the previous iteration. L1 norm.
      // We also compute the sign so that we can obtain the gradient direction
      real_t baseline = 0;
      for (int i=0; i<num_con; i++)
      {
         con_sgn[i] = con_res[i] < 0 ? -1.0 : con_res[i] > 0 ? 1.0 : 0.0;
         baseline += std::abs(con_res[i]);
      }
      constraints.GetGradient(projected_x).Mult(con_sgn, merit_grad);
      // Cyclic projection
      for (int i=0; i<num_con; i++)
      {
         // Get current set information
         Vector &qi = *q[i];
         Functional &con = constraints.GetFunctional(i);

         // Store previous
         psi_prev = psi;
         MapLatent(psi, xmin, xmax, projected_x); // update x
         psi += qi;
         // apply a few iteration of tangential projection
         for (int j=0; j<6; j++)
         {
            // Projection target:
            // c(x) + <grad c(x), x_proj - x> = 0
            // -> <grad c(x), x_proj> = <grad c(x), x> - c(x)
            if (shared_constraints) { shared_constraints->Update(projected_x); }
            con.Mult(projected_x, con_val);
            con.GetGradient().Mult(projected_x, deriv);
            mass.Riesz(deriv, grad);
            real_t targ = mass.InnerProduct(grad, projected_x) - con_val(0);

            Project(con, psi, grad, targ, psi_aux, projected_x);
            if (shared_constraints) { shared_constraints->Update(projected_x); }
            con.Mult(projected_x, con_val);
            if (std::abs(con_val[0])<tol) { break; }
         }
         // TODO: Remove duplication of velocity blocks in the input data.
         // This is a temporary workaround
         if (duplicated_velocity && velocity_related_constraints[i])
         {
            // input data has duplicated velocity blocks.
            // Sync the velocity blocks after projection.
            // Velocity blocks are contiguous,
            // so we can just copy the projected values psi, projected_x.
            // Starting index of velocity: velocity_idx_start, block size: velocity_block_size.
            // Master material index: master_material_idx.
            // qi will be updated after this, so we don't need to update qi.
            const int num_vel_blocks = velocity_idx_start.Size();
            const int master = master_material_idx[i];
            for (int k = 0; k < num_vel_blocks; k++)
            {
               if (master == k) { continue; }
               for (int j = 0; j < velocity_block_size; j++)
               {
                  psi[velocity_idx_start[k] + j] = psi[velocity_idx_start[master] + j];
               }
            }
            MapLatent(psi, xmin, xmax, projected_x);
         }

         // Update perturbation
         qi += psi_prev;
         qi -= psi;
      } // full cycle done
      if (enforce_sum_to_one)
      {
         ProjectSumToOne(psi, *q[num_con]);
      }

      MapLatent(psi, xmin, xmax, projected_x);
      if (shared_constraints) { shared_constraints->Update(projected_x); }
      constraints.Mult(projected_x, con_res);

      if (Mpi::Root())
      {
         out << "  Dykstra iteration " << iter << ": constraint violations = (";
         for (int i=0; i<con_res.Size(); i++)
         {
            out << con_res[i] << " ";
         }
         out << "\b)\n" << std::flush;
      }
      if (con_res.Normlinf() < tol)
      {
         if (Mpi::Root())
         {
            out << "Dykstra converged in " << iter << " iterations" << std::endl;
         }
         break;
      }
   }
}
void Dykstra::ProjectSumToOne(Vector &psi, Vector &qi)
{
   const int num_materials = sum_to_one_idx_start.Size();
   Vector curr_psi(num_materials),
          curr_min(num_materials), curr_max(num_materials);
   for (int i=0; i<sum_to_one_block_size; i++) // per quadrature
   {
      real_t psimax = 0.0;
      for (int j=0; j<num_materials; j++)
      {
         const int idx = sum_to_one_idx_start[j] + i;
         // Trial psi: apply Dykstra correction.
         curr_psi[j] = psi[idx] + qi[idx];
         psimax = std::max(psimax, std::abs(curr_psi[j]));
         curr_min[j] = xmin[idx];
         curr_max[j] = xmax[idx];
      }
      // min D_R(x, x0) s.t. sum_j x_j = 1
      // The optimality condition is
      // grad R(x) - grad R(x0) + lambda 1 = 0
      // -> psi = psi0 - lambda 1
      // Here, grad R^* is the scaled sigmoid function.
      // We can solve for lambda using Illinois method
      // f(lambda) = sum_j sigmoid(curr_psi[j] - lambda, ...) - 1
      // f is monotone decreasing in lambda.
      // For the initial bracket, we use (-1,1)*||psi||_inf.
      real_t a = -psimax, b = psimax;
      auto eval = [&](real_t lam) -> real_t
      {
         real_t s = 0.0;
         for (int j=0; j<num_materials; j++)
         {
            s += sigmoid(curr_psi[j] - lam, curr_min[j], curr_max[j]);
         }
         return s - 1.0;
      };
      if (fabs(eval(0.0)) < tol)
      {
         continue;
      }

      real_t fa = eval(a), fb = eval(b);
      // f is decreasing, so fa >= fb. Expand bracket until root is enclosed.
      real_t diff = b - a;
      while (fa * fb > 0)
      {
         diff *= 2;
         if (fa > 0) // both positive: root is to the right, expand right
         {
            a = b; fa = fb;
            b += diff; fb = eval(b);
         }
         else        // both negative: root is to the left, expand left
         {
            b = a; fb = fa;
            a -= diff; fa = eval(a);
         }
      }
      real_t a0 = a, b0 = b; // for debugging
      real_t fa0 = fa, fb0 = fb;

      // Illinois method (same structure as Dykstra::Project).
      int side = 0; real_t lambda;
      for (int i=0; i<100; i++)
      {
         lambda = (fa*b - fb*a)/(fa - fb);
         real_t f = eval(lambda);
         if (f * fb > 0)
         {
            b = lambda; fb = f;
            if (side == -1) { fa *= 0.5; }
            side = -1;
         }
         else
         {
            a = lambda; fa = f;
            if (side == 1) { fb *= 0.5; }
            side = 1;
         }
         if (std::abs(f) < tol)
         {
            break;
         }
      }
      // out << "      ProjectSumToOne: lambda = " << lambda << ", f(lambda) = " << eval(
      //        lambda)
      //     << ", initial bracket = (" << a0 << ", " << b0 << ") with "
      //     << "f(a0) = " << fa0 << ", f(b0) = " << fb0 << std::endl;

      // Apply shift and update psi and qi (Dykstra pattern).
      // qi_new = old_qi + psi_prev - psi_new = lambda (simplifies algebraically)
      for (int j=0; j<num_materials; j++)
      {
         const int idx = sum_to_one_idx_start[j] + i;
         const real_t psi_prev = psi[idx];
         psi[idx] = curr_psi[j] - lambda;
         qi[idx] += psi_prev - psi[idx];
      }
   }
}

void Dykstra::Project(const Functional &con, Vector &psi, const Vector &grad,
                      const real_t targ, Vector &psi_aux, Vector &projected_x)
{
   MapLatent(psi, xmin, xmax, projected_x);
   real_t b = 1e03;
   real_t a = -1e03;
   real_t diff = b - a;

   // update projected_x,
   // psi_aux = psi + shift*grad
   // and compute the linear residual
   // int g dot (sigmoid(psi + shift*grad)) - targ
   auto eval = [&](real_t shift) -> real_t
   {
      add(psi, shift, grad, psi_aux);
      MapLatent(psi_aux, xmin, xmax, projected_x);
      return mass.InnerProduct(grad, projected_x) - targ;
   };
   if (fabs(eval(0.0)) < tol)
   {
      return;
   }

   real_t fa = eval(a);
   real_t fb = eval(b);
   MFEM_VERIFY(fa <= fb,
               "Dykstra::Project: Initial values must satisfy f(a) < f(b). "
               "This is not the case for a = " << a << ", b = " << b
               << ", f(a) = " << fa << ", f(b) = " << fb);

   // if there is no root between a and b
   // shift the interval
   while (fa*fb > 0)
   {
      diff *= 2;
      // Note that int g dot sigmoid(psi + mu*grad) is stritly increasing.
      // Therefore, either (0<fa<fb) or (fa<fb<0)
      if (fa > 0)
      {
         b = a; fb = fa;
         a -= diff; fa = eval(a);
      }
      else
      {
         a = b; fa = fb;
         b += diff; fb = eval(b);
      }
      if (Mpi::Root())
      {
         out << "      Initial values: a = " << a << ", b = " << b
             << ", f(a) = " << fa << ", f(b) = " << fb << std::endl;
      }
   }
   // Now we have a root in (a,b).
   // Search using Illinois method
   int side = 0;
   for (int i=0; i<100; i++)
   {
      real_t c = (fa*b - fb*a)/(fa - fb);
      real_t fc = eval(c);
      if (fc * fb > 0)
      {
         b = c; fb = fc;
         if (side == -1) { fa *= 0.5; }
         side = -1;
      }
      else
      {
         a = c; fa = fc;
         if (side == 1) { fb *= 0.5; }
         side = 1;
      }
      if (std::abs(fc) < tol)
      {
         psi.Add(c, grad);
         break;
      }
   }
}

void Dykstra::MapLatent(const Vector &psi_,
                        const Vector &xmin_,
                        const Vector &xmax_,
                        Vector &x_)
{
   Vector curr_psi;
   Vector curr_x;
   for (int j=0; j<offsets.Size()-1; j++)
   {
      curr_psi.MakeRef(const_cast<Vector&>(psi_), offsets[j],
                       offsets[j+1]-offsets[j]);
      curr_x.MakeRef(x_, offsets[j], offsets[j+1]-offsets[j]);
      legendre_funcs[j]->gradinv(curr_psi, curr_x);
   }
}

void Dykstra::MapPrimal(const Vector &x_,
                        const Vector &xmin_,
                        const Vector &xmax_,
                        Vector &psi_)
{
   Vector curr_psi;
   Vector curr_x;
   for (int j=0; j<offsets.Size()-1; j++)
   {
      curr_x.MakeRef(const_cast<Vector&>(x_), offsets[j],
                     offsets[j+1]-offsets[j]);
      curr_psi.MakeRef(psi_, offsets[j], offsets[j+1]-offsets[j]);
      legendre_funcs[j]->grad(curr_x, curr_psi);
   }
}



EnergyBoxReport IntersectEnergyBoxWithPressure(MPI_Comm comm,
                                               const Vector &rho_q,
                                               const Vector &p_min_q,
                                               const Vector &p_max_q,
                                               const Vector &ind_q,
                                               real_t gm1,
                                               real_t ind_tol, real_t rho_tol_rel,
                                               Vector &e_min, Vector &e_max)
{
   EnergyBoxReport rep;
   // Energy and the pressure box share the quadrature points, so the pressure
   // bound is applied at each point with that point's own density -- pointwise
   // exact, no dof-to-quad association needed.
   const int n = e_min.Size();
   MFEM_VERIFY(e_max.Size() == n && rho_q.Size() == n,
               "IntersectEnergyBoxWithPressure: size mismatch.");

   real_t rho_scale = 0.0;
   for (int q = 0; q < rho_q.Size(); q++)
   { rho_scale = std::max(rho_scale, std::abs(rho_q(q))); }
   // Global, so the floor does not depend on the MPI partition.
   rho_scale = allreduce(comm, rho_scale, MPI_MAX);
   const real_t rho_floor = rho_tol_rel * rho_scale;

   for (int i = 0; i < n; i++)
   {
      const real_t dmp_lo = e_min(i), dmp_hi = e_max(i);
      const real_t dmp_w  = dmp_hi - dmp_lo;
      if (dmp_w <= 0.0) { continue; }
      // EXPERIMENT: eta check removed; only skip where rho is negligible.
      if (rho_q(i) <= rho_floor) { continue; }

      // Pressure requirement at this quadrature point, from its own density.
      const real_t scale = 1.0 / (gm1 * rho_q(i));
      const real_t p_lo = p_min_q(i) * scale, p_hi = p_max_q(i) * scale;

      real_t lo = std::max(dmp_lo, p_lo), hi = std::min(dmp_hi, p_hi);
      if (hi < lo)
      {
         // Incompatible: the pressure bound wins, so e leaves its DMP box
         // rather than the pressure leaving its own.
         rep.n_empty++;
         rep.max_clip = std::max(rep.max_clip, lo - hi);
         lo = p_lo; hi = p_hi;
      }

      // A thin or degenerate box is issued as is: the Fermi-Dirac generator
      // handles zero width directly (diff <= tol -> the point is pinned), the
      // same path the empty points [0,0] already take, so no widening is needed.

      // Internal energy is non-negative: e < 0 is a negative temperature and
      // gives a negative pressure. Applied only when the DMP box is itself
      // non-negative, so a genuinely negative-e problem is left alone.
      if (dmp_lo >= 0.0)
      {
         lo = std::max(lo, 0.0);
         hi = std::max(hi, lo);
      }

      // Report the box as finally issued -- after the widening and clamp above.
      // Measuring before them understates both the tightening and the excursion.
      rep.max_tighten = std::max(rep.max_tighten, dmp_w - (hi - lo));
      rep.max_dmp_excursion = std::max(rep.max_dmp_excursion,
                                       std::max(std::max(dmp_lo - lo, hi - dmp_hi),
                                                0.0));
      rep.max_p_excursion = std::max(rep.max_p_excursion,
                                     std::max(std::max(p_lo - lo, hi - p_hi),
                                              0.0));
      e_min(i) = lo;
      e_max(i) = hi;
   }

   return rep;
}

// Pressure p = (gamma-1) * rho * e at the quadrature points, with rho and e
// both quadrature functions (gamma-1 = 1 here). Pointwise, since e now lives at
// the quadrature points next to rho.
static void PressureQF(real_t gm1, const QuadratureFunction &rho,
                       const Vector &e, QuadratureFunction &p)
{
   const int n = rho.Size();
   p.SetSize(n);
   for (int i = 0; i < n; i++) { p(i) = gm1 * rho(i) * e(i); }
}

TwoStagePressureRemap::TwoStagePressureRemap(
   QuadratureSpace &qspace_,
   ParFiniteElementSpace &pfes_v_scalar_,
   MassOperator &mass_q_, MassOperator &mass_h1_,
   int num_materials_, int dim_, bool remap_v_, const Options &opts_)
   : qspace(qspace_), pfes_v_scalar(pfes_v_scalar_)
   , mass_q(mass_q_), mass_h1(mass_h1_)
   , num_materials(num_materials_), dim(dim_), remap_v(remap_v_), opts(opts_)
   , size_qf(qspace_.GetSize()), size_e(qspace_.GetSize())
   , size_v1(pfes_v_scalar_.GetTrueVSize())
   , num_vars(3 + dim_ * (int)remap_v_)
   , fes({&pfes_v_scalar_})
{
   // Per-material block sizes and space indices. Both layouts differ only in
   // slot 2: the energy layout carries e, the pressure layout carries p, and
   // both now live at the quadrature points, next to rho.
   const int n_vel = dim * (int)remap_v;
   blk_e.Append(size_qf); blk_e.Append(size_qf); blk_e.Append(size_e);
   blk_p.Append(size_qf); blk_p.Append(size_qf); blk_p.Append(size_qf);
   for (int d = 0; d < n_vel; d++) { blk_e.Append(size_v1); blk_p.Append(size_v1); }
   per_mat_e = blk_e.Sum();
   per_mat_p = blk_p.Sum();

   // Space index -1 marks a quadrature function (eta, rho, e, p); velocity is
   // the only finite-element variable, at index 0 in fes.
   for (int k = 0; k < num_materials; k++)
   {
      space_idx_e.Append(-1); space_idx_e.Append(-1); space_idx_e.Append(-1);
      space_idx_p.Append(-1); space_idx_p.Append(-1); space_idx_p.Append(-1);
      for (int d = 0; d < n_vel; d++)
      {
         space_idx_e.Append(0); space_idx_p.Append(0);
      }
   }
}

// Slice the k-th material's pointwise values out of the global value vector
// before handing them to a single-material functional.
static std::function<real_t(const Vector &)>
ts_shift_f(std::function<real_t(const Vector &)> f, int k, int num_vars)
{
   return [f, k, num_vars](const Vector &x) -> real_t
   {
      const Vector x_curr(x.GetData() + k*num_vars, num_vars);
      return f(x_curr);
   };
}

static std::function<void(const Vector &, Vector &)>
ts_shift_df(std::function<void(const Vector &, Vector &)> df, int k,
            int num_vars)
{
   return [df, k, num_vars](const Vector &x, Vector &y) -> void
   {
      y = 0.0;
      const Vector x_curr(x.GetData() + k*num_vars, num_vars);
      Vector y_curr(y.GetData() + k*num_vars, num_vars);
      df(x_curr, y_curr);
   };
}

void TwoStagePressureRemap::Solve(const Vector &x_min, const Vector &x_max,
                                  const std::vector<Vector> &p_min,
                                  const std::vector<Vector> &p_max,
                                  const Vector &volume_0, const Vector &mass_0,
                                  const Vector &energy_0, const Vector &moment_0,
                                  Vector &x)
{
   MFEM_VERIFY(x.Size() == num_materials*per_mat_e &&
               x_min.Size() == x.Size() && x_max.Size() == x.Size(),
               "TwoStagePressureRemap::Solve: unexpected state size.");
   MFEM_VERIFY((int)p_min.size() == num_materials &&
               (int)p_max.size() == num_materials,
               "TwoStagePressureRemap::Solve: pressure box size mismatch.");

   // The interpolated energy is needed in stage 2 as a fallback target, so
   // keep a copy before x is overwritten.
   Vector e_interp(num_materials*size_e);
   for (int k = 0; k < num_materials; k++)
   {
      for (int i = 0; i < size_e; i++)
      {
         e_interp(k*size_e + i) = x(k*per_mat_e + 2*size_qf + i);
      }
   }

   Vector xp(num_materials*per_mat_p);
   SolveStage1(x_min, x_max, p_min, p_max, volume_0, mass_0, energy_0, moment_0,
               x, xp);
   SolveStage2(x_min, x_max, p_min, p_max, energy_0, xp, e_interp, x);
}

void TwoStagePressureRemap::SolveStage1(const Vector &x_min,
                                        const Vector &x_max,
                                        const std::vector<Vector> &p_min,
                                        const std::vector<Vector> &p_max,
                                        const Vector &volume_0,
                                        const Vector &mass_0,
                                        const Vector &energy_0,
                                        const Vector &moment_0,
                                        const Vector &x_interp, Vector &xp)
{
   const real_t gm1 = opts.gamma_minus_one;
   MPI_Comm comm = pfes_v_scalar.GetComm();
   if (Mpi::Root())
   {
      out << "\n=== Pressure control, stage 1: project (ind, rho, p"
          << (remap_v ? ", v)" : ")") << " ===" << std::endl;
   }

   Array<int> gt_off({0}), gt_off_e({0});
   for (int k = 0; k < num_materials; k++)
   {
      gt_off.Append(blk_p); gt_off_e.Append(blk_e);
   }
   gt_off.PartialSum(); gt_off_e.PartialSum();

   BlockVector xp_b(xp.GetData(), gt_off);
   BlockVector xp_min(gt_off), xp_max(gt_off);
   const BlockVector xi_b(const_cast<Vector&>(x_interp).GetData(), gt_off_e),
         xn_b(const_cast<Vector&>(x_min).GetData(), gt_off_e),
         xx_b(const_cast<Vector&>(x_max).GetData(), gt_off_e);

   // Fill the p-layout state: slots 0, 1 and the velocity come over unchanged;
   // slot 2 becomes the interpolated pressure rho_interp*e_interp, clamped
   // into the pressure box, with the box itself as the bound.
   real_t pviol_interp = 0.0;
   for (int k = 0; k < num_materials; k++)
   {
      const int b = k*num_vars;
      for (int i = 0; i < num_vars; i++)
      {
         if (i == 2) { continue; }
         xp_b.GetBlock(b+i)   = xi_b.GetBlock(b+i);
         xp_min.GetBlock(b+i) = xn_b.GetBlock(b+i);
         xp_max.GetBlock(b+i) = xx_b.GetBlock(b+i);
      }

      QuadratureFunction rho_k(&qspace, xi_b.GetBlock(b+1).GetData());
      QuadratureFunction p_init(&qspace);
      PressureQF(gm1, rho_k, xi_b.GetBlock(b+2), p_init);

      Vector &p0 = xp_b.GetBlock(b+2);
      const Vector &ind_k = xi_b.GetBlock(b+0);
      for (int i = 0; i < size_qf; i++)
      {
         if (ind_k(i) > opts.ind_tol)
         {
            pviol_interp = std::max(pviol_interp,
                                    std::max(p_init(i) - p_max[k](i),
                                             p_min[k](i) - p_init(i)));
         }
         p0(i) = std::min(std::max(p_init(i), p_min[k](i)), p_max[k](i));
      }
      xp_min.GetBlock(b+2) = p_min[k];
      xp_max.GetBlock(b+2) = p_max[k];
   }
   pviol_interp = allreduce(comm, pviol_interp, MPI_MAX);
   if (Mpi::Root())
   {
      out << "  Interpolated pressure-bound violation (where material present)"
          << " = " << pviol_interp
          << std::endl;
   }

   // Conservation functionals in the (eta, rho, p, v) variables.
   const int funcs_per_mat = 3 + dim * (int)remap_v;
   std::vector<std::unique_ptr<ComposedFunctional>> funcs(funcs_per_mat *
                                                          num_materials);
   Array<int> vel_related(funcs_per_mat*num_materials); vel_related = 0;
   Array<int> master_mat(funcs_per_mat*num_materials);  master_mat = -1;

   auto make = [&](int idx, std::function<real_t(const Vector &)> f,
                   std::function<void(const Vector &, Vector &)> df,
                   int k, real_t target)
   {
      funcs[idx] = std::make_unique<ComposedFunctional>(
                      ts_shift_f(f, k, num_vars),
                      ts_shift_df(df, k, num_vars),
                      qspace, fes, space_idx_p);
      funcs[idx]->SetTarget(target);
   };

   for (int k = 0; k < num_materials; k++)
   {
      const int f0 = funcs_per_mat*k;
      make(f0 + 0, remap::volume_f, remap::volume_df, k, volume_0(k));
      make(f0 + 1, remap::mass_f,   remap::mass_df,   k, mass_0(k));
      if (!remap_v)
      {
         make(f0 + 2,
         [gm1](const Vector &u) { return remap::p_potential_f(u, gm1); },
         [gm1](const Vector &u, Vector &g) { remap::p_potential_df(u, g, gm1); },
         k, energy_0(k));
      }
      else
      {
         vel_related[f0 + 2] = 1; master_mat[f0 + 2] = k;
         make(f0 + 2,
         [gm1](const Vector &u) { return remap::p_energy_f(u, gm1); },
         [gm1](const Vector &u, Vector &g) { remap::p_energy_df(u, g, gm1); },
         k, energy_0(k));
         for (int d = 0; d < dim; d++)
         {
            vel_related[f0 + 3 + d] = 1; master_mat[f0 + 3 + d] = k;
            make(f0 + 3 + d,
            [d](const Vector &u) { return remap::momentum_f(u, d); },
            [d](const Vector &u, Vector &g) { remap::momentum_df(u, g, d); },
            k, moment_0(k*dim + d));
         }
      }
   }

   StackedSharedFunctional C(num_materials*per_mat_p);
   for (auto &f : funcs) { f->SetComm(comm); C.AddFunctional(*f); }

   MultiMassOperator mass;
   for (int k = 0; k < num_materials; k++)
   {
      mass.Append(mass_q); mass.Append(mass_q); mass.Append(mass_q);
      if (remap_v) { for (int d = 0; d < dim; d++) { mass.Append(mass_h1); } }
   }

   PointwiseFermiDirac sigmoid(xp_min, xp_max);
   Array<LegendreFunction*> legendre_funcs({&sigmoid});
   Array<int> dummy_offset({0, xp.Size()});
   Dykstra projector(comm, C, mass, legendre_funcs, dummy_offset,
                     xp_min, xp_max, opts.atol, opts.max_iter);

   // Sum-to-one couples the per-material indicators; with a single material it
   // would force eta == 1 everywhere, contradicting the volume constraint.
   if (num_materials > 1)
   {
      Array<int> ind_idx(num_materials);
      for (int k = 0; k < num_materials; k++) { ind_idx[k] = k*per_mat_p; }
      projector.EnforceSumToOne(ind_idx, size_qf);
   }

   if (remap_v && num_materials > 1)
   {
      Array<int> vel_idx(num_materials);
      for (int k = 0; k < num_materials; k++)
      {
         vel_idx[k] = k*per_mat_p + 3*size_qf;
      }
      projector.SetDuplicatedVelocity(vel_idx, vel_related, master_mat,
                                      dim*size_v1);
   }

   projector.Project(xp);

   // The pressure is bounded by construction; report it as a sanity check.
   real_t pviol = 0.0;
   for (int k = 0; k < num_materials; k++)
   {
      const Vector &pk = xp_b.GetBlock(k*num_vars + 2);
      for (int i = 0; i < size_qf; i++)
      {
         pviol = std::max(pviol, std::max(pk(i) - p_max[k](i),
                                          p_min[k](i) - pk(i)));
      }
   }
   pviol = allreduce(comm, pviol, MPI_MAX);
   if (Mpi::Root())
   {
      out << "  Stage 1 auxiliary-pressure bound violation = " << pviol
          << std::endl;
   }
}

void TwoStagePressureRemap::SolveStage2(const Vector &x_min,
                                        const Vector &x_max,
                                        const std::vector<Vector> &p_min,
                                        const std::vector<Vector> &p_max,
                                        const Vector &energy_0, const Vector &xp,
                                        const Vector &e_interp, Vector &x)
{
   const real_t gm1 = opts.gamma_minus_one;
   MPI_Comm comm = pfes_v_scalar.GetComm();
   if (Mpi::Root())
   {
      out << "\n=== Pressure control, stage 2: project e at frozen (ind, rho"
          << (remap_v ? ", v)" : ")") << " ===" << std::endl;
   }

   Array<int> gt_off({0}), gt_off_p({0});
   for (int k = 0; k < num_materials; k++)
   {
      gt_off.Append(blk_e); gt_off_p.Append(blk_p);
   }
   gt_off.PartialSum(); gt_off_p.PartialSum();

   BlockVector x_b(x.GetData(), gt_off);
   BlockVector e_min_all(gt_off), e_max_all(gt_off);
   const BlockVector xp_b(const_cast<Vector&>(xp).GetData(), gt_off_p),
         xn_b(const_cast<Vector&>(x_min).GetData(), gt_off),
         xx_b(const_cast<Vector&>(x_max).GetData(), gt_off);

   // Copy the frozen stage-1 state back into the energy layout, and pin every
   // block but the energy by collapsing its box onto the value itself.
   for (int k = 0; k < num_materials; k++)
   {
      const int b = k*num_vars;
      for (int i = 0; i < num_vars; i++)
      {
         if (i == 2) { continue; }
         x_b.GetBlock(b+i)     = xp_b.GetBlock(b+i);
         e_min_all.GetBlock(b+i) = xp_b.GetBlock(b+i);
         e_max_all.GetBlock(b+i) = xp_b.GetBlock(b+i);
      }
   }

   EnergyBoxReport rep_all;
   Vector g_quad(size_qf);

   for (int k = 0; k < num_materials; k++)
   {
      const int b = k*num_vars;
      const Vector &ind_star = xp_b.GetBlock(b+0);
      const Vector &rho_star = xp_b.GetBlock(b+1);
      const Vector &p_star   = xp_b.GetBlock(b+2);
      Vector &e_lo = e_min_all.GetBlock(b+2), &e_hi = e_max_all.GetBlock(b+2);

      // Density floor, relative to the material's largest density so it does
      // not depend on units. Same convention as the box construction.
      real_t rho_scale = 0.0;
      for (int i = 0; i < size_qf; i++)
      { rho_scale = std::max(rho_scale, std::abs(rho_star(i))); }
      rho_scale = allreduce(comm, rho_scale, MPI_MAX);
      const real_t rho_floor = opts.rho_tol_rel * rho_scale;

      // Target first, since the box is required to keep it feasible. The energy
      // consistent with the stage-1 pressure, evaluated at the quadrature
      // points. Where the material is absent the pressure carries no
      // information and the interpolated energy is used instead.
      Vector &e_k = x_b.GetBlock(b+2);
      const Vector e_int_k(const_cast<Vector&>(e_interp).GetData() + k*size_e,
                           size_e);
      if (opts.e_target_from_pressure)
      {
         // g = p/((gamma-1)*rho) at the quadrature points (gamma-1 = 1 here).
         // Where the material is absent, or the density is negligible, p = rho*e
         // says nothing about e and dividing by rho would give a wild target, so
         // the interpolated energy is used there instead. Energy lives at the
         // quadrature points, so g is the target directly (no L2 projection).
         for (int i = 0; i < size_qf; i++)
         {
            const bool informative = rho_star(i) > rho_floor; // EXPERIMENT: no eta
            g_quad(i) = informative ? p_star(i) / (gm1 * rho_star(i)) : e_int_k(i);
         }
         e_k = g_quad;
      }
      else { e_k = e_int_k; }

      // Box: the energy values admissible for both the energy DMP bound and
      // the pressure bound at the frozen density.
      e_lo = xn_b.GetBlock(b+2);
      e_hi = xx_b.GetBlock(b+2);
      const EnergyBoxReport r =
         IntersectEnergyBoxWithPressure(comm,
                                        rho_star, p_min[k], p_max[k], ind_star,
                                        gm1,
                                        opts.ind_tol, opts.rho_tol_rel,
                                        e_lo, e_hi);
      rep_all.max_tighten = std::max(rep_all.max_tighten, r.max_tighten);
      rep_all.max_clip    = std::max(rep_all.max_clip, r.max_clip);
      rep_all.max_dmp_excursion = std::max(rep_all.max_dmp_excursion,
                                           r.max_dmp_excursion);
      rep_all.max_p_excursion = std::max(rep_all.max_p_excursion,
                                         r.max_p_excursion);
      rep_all.n_empty    += r.n_empty;

      for (int i = 0; i < size_e; i++)
      {
         e_k(i) = std::min(std::max(e_k(i), e_lo(i)), e_hi(i));
      }
   }

   int n_empty = (int)allreduce(comm, (real_t)rep_all.n_empty, MPI_SUM);
   real_t tighten = allreduce(comm, rep_all.max_tighten, MPI_MAX);
   real_t clip    = allreduce(comm, rep_all.max_clip, MPI_MAX);
   real_t excur   = allreduce(comm, rep_all.max_dmp_excursion, MPI_MAX);
   real_t p_excur = allreduce(comm, rep_all.max_p_excursion, MPI_MAX);
   if (Mpi::Root())
   {
      out << "  Energy box: max tightening by pressure = " << tighten
          << ", empty intersections = " << n_empty
          << " (max gap " << clip << ")\n"
          << "  Energy box resolved in favour of pressure: max excursion "
          << "outside the energy DMP box = " << excur
          << ", max excursion outside the pressure box = "
          << p_excur << std::endl;
   }

   // Single energy-conservation constraint per material; everything else pinned.
   std::vector<std::unique_ptr<ComposedFunctional>> funcs(num_materials);
   for (int k = 0; k < num_materials; k++)
   {
      funcs[k] = std::make_unique<ComposedFunctional>(
                    ts_shift_f(remap_v ? remap::energy_f : remap::potential_f,
                               k, num_vars),
                    ts_shift_df(remap_v ? remap::energy_df : remap::potential_df,
                                k, num_vars),
                    qspace, fes, space_idx_e);
      funcs[k]->SetTarget(energy_0(k));
      funcs[k]->SetComm(comm);
   }

   StackedSharedFunctional C(num_materials*per_mat_e);
   for (auto &f : funcs) { C.AddFunctional(*f); }

   MultiMassOperator mass;
   for (int k = 0; k < num_materials; k++)
   {
      mass.Append(mass_q); mass.Append(mass_q); mass.Append(mass_q);
      if (remap_v) { for (int d = 0; d < dim; d++) { mass.Append(mass_h1); } }
   }

   PointwiseFermiDirac sigmoid(e_min_all, e_max_all);
   Array<LegendreFunction*> legendre_funcs({&sigmoid});
   Array<int> dummy_offset({0, x.Size()});
   Dykstra projector(comm, C, mass, legendre_funcs, dummy_offset,
                     e_min_all, e_max_all, opts.atol, opts.max_iter);

   projector.Project(x);

   // Physical pressure of the recovered state, against the pressure box. The
   // masked figure is the meaningful one: a pressure bound at a quadrature
   // point that carries no material does not constrain anything physical.
   real_t pviol = 0.0, pviol_mat = 0.0, maxP = 0.0;
   for (int k = 0; k < num_materials; k++)
   {
      const int b = k*num_vars;
      const Vector &ind_k = x_b.GetBlock(b+0);
      QuadratureFunction rho_k(&qspace, x_b.GetBlock(b+1).GetData());
      QuadratureFunction P_k(&qspace);
      PressureQF(gm1, rho_k, x_b.GetBlock(b+2), P_k);
      for (int i = 0; i < size_qf; i++)
      {
         const real_t v = std::max(P_k(i) - p_max[k](i), p_min[k](i) - P_k(i));
         pviol = std::max(pviol, v);
         if (ind_k(i) > opts.ind_tol)
         {
            pviol_mat = std::max(pviol_mat, v);
            maxP = std::max(maxP, P_k(i));
         }
      }
   }
   pviol     = allreduce(comm, pviol,     MPI_MAX);
   pviol_mat = allreduce(comm, pviol_mat, MPI_MAX);
   maxP      = allreduce(comm, maxP,      MPI_MAX);
   if (Mpi::Root())
   {
      out << "  Final pressure-bound violation = " << pviol_mat
          << " where material is present, " << pviol << " raw"
          << " (maxP = " << maxP << ")" << std::endl;
   }
}

}
