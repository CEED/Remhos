// Copyright (c) 2026, Lawrence Livermore National Security, LLC.
// See files LICENSE and NOTICE for details.

#include "remhos_nodal.hpp"
#include "remhos_lvpp.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <vector>

namespace mfem
{
namespace
{

struct LocalFitInfo
{
   int sweeps = 0;
   bool converged = false;
};

// Hildreth's dual coordinate descent for min ||x-t||_M, A*x >= b.
// Maintaining x = t + M^-1 A^T lambda avoids storing the dense dual Hessian.
LocalFitInfo FitElement(const DenseMatrix &mass, const Vector &target,
                        const DenseMatrix &A, const Vector &b,
                        const NodalRemapOptions &options, Vector &x)
{
   LocalFitInfo info;
   const int n = target.Size(), m = A.Height();
   Vector slack(m), lambda(m), diagonal(m);
   x = target;
   lambda = 0.0;
   A.Mult(x, slack);
   slack -= b;
   const real_t tol = std::max(options.abs_tol,
                             options.rel_tol * std::max<real_t>(1, slack.Normlinf()));
   if (slack.Min() >= -tol)
   {
      info.converged = true;
      return info;
   }

   DenseMatrixInverse inverse(mass, true);
   DenseMatrix directions(n, m);
   std::vector<Vector> rows(m);
   Vector direction;
   for (int i = 0; i < m; i++)
   {
      A.GetRow(i, rows[i]);
      directions.GetColumnReference(i, direction);
      inverse.Mult(rows[i], direction);
      diagonal(i) = rows[i] * direction;
      MFEM_VERIFY(std::isfinite(diagonal(i)) && diagonal(i) > 0,
                  "Invalid local constraint diagonal.");
   }

   auto Reconstruct = [&]()
   {
      directions.Mult(lambda, x);
      x += target;
   };
   auto Residual = [&]()
   {
      A.Mult(x, slack);
      slack -= b;
      real_t residual = 0;
      for (int i = 0; i < m; i++)
      {
         // Projected dual residual in constraint units; also detects
         // nonzero multipliers on inactive constraints.
         const real_t r = std::min(diagonal(i) * lambda(i), slack(i));
         if (!std::isfinite(r)) { return std::numeric_limits<real_t>::infinity(); }
         residual = std::max(residual, std::abs(r));
      }
      return residual;
   };

   for (int sweep = 0; sweep < options.max_sweeps; sweep++)
   {
      for (int i = 0; i < m; i++)
      {
         const real_t gradient = rows[i] * x - b(i);
         const real_t next = std::max<real_t>(0, lambda(i) - gradient / diagonal(i));
         const real_t delta = next - lambda(i);
         if (delta == 0) { continue; }
         lambda(i) = next;
         directions.GetColumnReference(i, direction);
         x.Add(delta, direction);
      }
      info.sweeps = sweep + 1;
      if (info.sweeps % 50 == 0) { Reconstruct(); }
      if (Residual() <= tol)
      {
         Reconstruct();
         if (Residual() <= tol)
         {
            info.converged = true;
            break;
         }
      }
   }
   Reconstruct();
   info.converged = Residual() <= tol;
   return info;
}

DenseMatrix ConstraintMatrix(const DenseMatrix &L, const DenseMatrix &U,
                              const NodalRemapOptions &options, Vector &b)
{
   const int n = L.Width();
   const int nl = options.plbound ? L.Height() : 0;
   const int nu = options.upper_constraint ? U.Height() : 0;
   DenseMatrix A(n + nl + nu, n);
   A = 0.0;
   b.SetSize(A.Height());
   b = 0.0;
   for (int i = 0; i < n; i++) { A(i, i) = 1; }
   for (int i = 0; i < nl; i++)
   {
      for (int j = 0; j < n; j++) { A(n+i, j) = L(i, j); }
   }
   for (int i = 0; i < nu; i++)
   {
      for (int j = 0; j < n; j++) { A(n+nl+i, j) = -U(i, j); }
      b(n+nl+i) = -options.upper;
   }
   return A;
}

struct FieldStats
{
   real_t minimum[3] = { std::numeric_limits<real_t>::infinity(),
                        std::numeric_limits<real_t>::infinity(),
                        std::numeric_limits<real_t>::infinity() };
   real_t maximum[3] = { -std::numeric_limits<real_t>::infinity(),
                        -std::numeric_limits<real_t>::infinity(),
                        -std::numeric_limits<real_t>::infinity() };
   real_t mass = 0;
};

FieldStats MeasureField(ParGridFunction &field, const DenseMatrix &L,
                        const DenseMatrix &U, const IntegrationRule &ir)
{
   FieldStats stats;
   Array<int> dofs;
   Vector local, lower(L.Height()), upper(U.Height()), values;
   ParFiniteElementSpace &fes = *field.ParFESpace();
   for (int e = 0; e < fes.GetNE(); e++)
   {
      fes.GetElementDofs(e, dofs);
      field.GetSubVector(dofs, local);
      L.Mult(local, lower);
      U.Mult(local, upper);
      field.GetValues(e, ir, values);
      stats.minimum[0] = std::min(stats.minimum[0], local.Min());
      stats.minimum[1] = std::min(stats.minimum[1], lower.Min());
      stats.minimum[2] = std::min(stats.minimum[2], values.Min());
      stats.maximum[0] = std::max(stats.maximum[0], local.Max());
      stats.maximum[1] = std::max(stats.maximum[1], upper.Max());
      stats.maximum[2] = std::max(stats.maximum[2], values.Max());
      auto &tr = *fes.GetElementTransformation(e);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const auto &ip = ir.IntPoint(q);
         tr.SetIntPoint(&ip);
         stats.mass += ip.weight * tr.Weight() * values(q);
      }
   }
   const MPI_Datatype type = MPITypeMap<real_t>::mpi_type;
   MPI_Allreduce(MPI_IN_PLACE, stats.minimum, 3, type, MPI_MIN, fes.GetComm());
   MPI_Allreduce(MPI_IN_PLACE, stats.maximum, 3, type, MPI_MAX, fes.GetComm());
   MPI_Allreduce(MPI_IN_PLACE, &stats.mass, 1, type, MPI_SUM, fes.GetComm());
   return stats;
}

void PrintStats(const char *name, const FieldStats &stats)
{
   out << name << " nodal min/max: " << stats.minimum[0] << " " << stats.maximum[0]
       << "\n" << name << " min(L*e_K): " << stats.minimum[1]
       << "\n" << name << " max(U*e_K): " << stats.maximum[1]
       << "\n" << name << " sampled min/max: " << stats.minimum[2] << " "
       << stats.maximum[2] << "\n" << name << " mass: " << stats.mass << "\n";
}

void ShowField(ParGridFunction &field, const char *title, int position, int y = 0)
{
   auto &mesh = *field.ParFESpace()->GetParMesh();
   socketstream socket("localhost", 19916);
   socket.precision(12);
   socket << "parallel " << mesh.GetNRanks() << " " << mesh.GetMyRank() << "\n"
          << "solution\n" << mesh << field
          << "window_title '" << title << "'\n"
          << "window_geometry " << position << " " << y << " 400 400\n"
          << "keys jRmclA\n" << std::flush;
}

void ProjectElement(const FiniteElement &fe, ElementTransformation &tr,
                     const IntegrationRule &ir, const Vector &samples,
                     int offset, DenseMatrix &mass, Vector &target)
{
   MassIntegrator integrator;
   integrator.SetIntRule(&ir);
   integrator.AssembleElementMatrix(fe, tr, mass);
   Vector rhs(fe.GetDof()), shape(fe.GetDof());
   rhs = 0.0;
   for (int q = 0; q < ir.GetNPoints(); q++)
   {
      const auto &ip = ir.IntPoint(q);
      tr.SetIntPoint(&ip);
      fe.CalcShape(ip, shape);
      rhs.Add(ip.weight * tr.Weight() * samples(offset+q), shape);
   }
   DenseMatrixInverse inverse(mass, true);
   inverse.Mult(rhs, target);
}

DenseMatrix BoxConstraints(int n, const NodalRemapOptions &options, Vector &b)
{
   DenseMatrix A(options.upper_constraint ? 2*n : n, n);
   A = 0.0;
   b.SetSize(A.Height());
   b = 0.0;
   for (int i = 0; i < n; i++)
   {
      A(i, i) = 1.0;
      if (options.upper_constraint)
      {
         A(n+i, i) = -1.0;
         b(n+i) = -options.upper;
      }
   }
   return A;
}

} // namespace

// Infeasible-start primal-dual predictor/corrector. Bounds are local; eliminating
// their slacks and multipliers leaves SPD element blocks and one global scalar
// Schur complement for conservation. No global DOF matrix is assembled.
JointFitInfo FitConservative(MPI_Comm comm, const std::vector<DenseMatrix> &mass,
                            const std::vector<Vector> &target,
                            const DenseMatrix &constraints, const Vector &bounds,
                            real_t target_mass, const NodalRemapOptions &options,
                            std::vector<Vector> &result)
{
   JointFitInfo info;
   const int ne = mass.size(), n = constraints.Width(), m = constraints.Height();
   MFEM_VERIFY(n > 0 && m > 0 && bounds.Size() == m && int(target.size()) == ne,
               "Incompatible conservative QP dimensions.");
   const MPI_Datatype type = MPITypeMap<real_t>::mpi_type;
   DenseMatrix A(constraints);
   Vector b(bounds), row_scale(m), one(n);
   one = 1.0;
   std::vector<std::vector<int>> support(m);
   for (int i = 0; i < m; i++)
   {
      real_t norm = 0;
      for (int j = 0; j < n; j++) { norm += A(i,j)*A(i,j); }
      row_scale(i) = std::sqrt(norm);
      MFEM_VERIFY(row_scale(i) > 0 && std::isfinite(row_scale(i)),
                  "Empty or nonfinite joint QP constraint.");
      b(i) /= row_scale(i);
      for (int j = 0; j < n; j++)
      {
         A(i,j) /= row_scale(i);
         if (A(i,j) != 0) { support[i].push_back(j); }
      }
   }
   struct Element
   {
      Vector c, x, s, z, rd, rp, dx, ds, dz, affine_s, affine_z, v, w, scale;
      DenseMatrix factor;
      real_t dual_scale = 1;
      Element(int n, int m) : c(n), x(n), s(m), z(m), rd(n), rp(m), dx(n),
         ds(m), dz(m), affine_s(m), affine_z(m), v(n), w(n), scale(n), factor(n) { }
      bool Factor()
      {
         // MFEM's non-LAPACK Cholesky aborts on a nonpositive pivot. Detect
         // roundoff loss of definiteness here so diagonal regularization can retry.
         const int n = factor.Height();
         for (int j = 0; j < n; j++)
         {
            real_t pivot = factor(j,j);
            for (int k = 0; k < j; k++) { pivot -= factor(j,k)*factor(j,k); }
            if (!(pivot > 0) || !std::isfinite(pivot)) { return false; }
            factor(j,j) = std::sqrt(pivot);
            for (int i = j+1; i < n; i++)
            {
               for (int k = 0; k < j; k++) { factor(i,j) -= factor(i,k)*factor(j,k); }
               factor(i,j) /= factor(j,j);
            }
         }
         return true;
      }
      void Solve(Vector &rhs)
      {
         for (int j = 0; j < rhs.Size(); j++) { rhs(j) *= scale(j); }
         CholeskyFactors chol(factor.Data());
         chol.Solve(rhs.Size(), 1, rhs.GetData());
         for (int j = 0; j < rhs.Size(); j++) { rhs(j) *= scale(j); }
      }
   };
   std::vector<Element> elements;
   elements.reserve(ne);
   real_t volume = 0, field_scale = 1;
   for (int e = 0; e < ne; e++)
   {
      MFEM_VERIFY(mass[e].Height() == n && mass[e].Width() == n && target[e].Size() == n,
                  "Incompatible conservative QP element dimensions.");
      elements.emplace_back(n, m);
      mass[e].Mult(one, elements.back().c);
      volume += one*elements.back().c;
      field_scale = std::max(field_scale, target[e].Normlinf());
   }
   volume = allreduce(comm, volume, MPI_SUM);
   field_scale = allreduce(comm, field_scale, MPI_MAX);
   const real_t count = allreduce(comm, real_t(ne)*m, MPI_SUM);
   MFEM_VERIFY(volume > 0 && count > 0, "Empty or invalid conservative QP.");
   const real_t tol = std::max(options.abs_tol, options.rel_tol);
   const real_t mass_tol = options.mass_tolerance*std::max<real_t>(1, std::abs(target_mass));
   result = target;
   real_t initial_mass = 0, initial_violation = 0;
   for (int e = 0; e < ne; e++)
   {
      Vector slack(m);
      constraints.Mult(target[e], slack);
      slack -= bounds;
      initial_violation = std::max(initial_violation, -slack.Min());
      initial_mass += elements[e].c*target[e];
   }
   initial_mass = allreduce(comm, initial_mass, MPI_SUM);
   initial_violation = allreduce(comm, initial_violation, MPI_MAX);
   if (initial_violation <= tol*field_scale && std::abs(initial_mass-target_mass) <= mass_tol)
   {
      // The unconstrained minimizer is already feasible: preserve it exactly.
      info.converged = true;
      info.mass_error = initial_mass-target_mass;
      info.violation = initial_violation;
      return info;
   }
   if (target_mass < -mass_tol ||
       (options.upper_constraint && target_mass > options.upper*volume + mass_tol))
   {
      int rank;
      MPI_Comm_rank(comm, &rank);
      if (!rank) { out << "Joint QP: source mass is outside the physical bound capacity.\n"; }
      info.mass_error = initial_mass-target_mass;
      info.violation = initial_violation;
      return info;
   }
   for (int e = 0; e < ne; e++)
   {
      auto &el = elements[e];
      el.x = target_mass/volume;
      A.Mult(el.x, el.s);
      el.s -= b;
      for (int i = 0; i < m; i++)
      {
         el.s(i) = std::max(el.s(i), 0.01*field_scale);
         el.z(i) = 0.01*volume*field_scale*field_scale/(count*el.s(i));
      }
      el.dual_scale = 0;
      for (int j = 0; j < n; j++)
      {
         real_t row_sum = 0;
         for (int k = 0; k < n; k++) { row_sum += std::abs(mass[e](j,k)); }
         el.dual_scale = std::max(el.dual_scale, row_sum*field_scale);
      }
   }
   real_t y = 0;
   for (int iteration = 0; iteration <= options.mass_max_iterations; iteration++)
   {
      real_t sums[2] = {0, 0}, maxima[3] = {0, 0, 0};
      for (int e = 0; e < ne; e++)
      {
         auto &el = elements[e];
         Vector difference(el.x), atz(n), ax(m);
         difference -= target[e];
         mass[e].Mult(difference, el.rd);
         A.MultTranspose(el.z, atz);
         el.rd -= atz;
         el.rd.Add(y, el.c);
         A.Mult(el.x, ax);
         ax -= b;
         el.rp = ax;
         el.rp -= el.s;
         sums[0] += el.c*el.x;
         sums[1] += el.s*el.z;
         maxima[0] = std::max(maxima[0], el.rd.Normlinf()/el.dual_scale);
         for (int i = 0; i < m; i++)
         {
            maxima[1] = std::max(maxima[1], std::abs(el.rp(i))*row_scale(i));
            maxima[2] = std::max(maxima[2], -ax(i)*row_scale(i));
         }
      }
      MPI_Allreduce(MPI_IN_PLACE, sums, 2, type, MPI_SUM, comm);
      MPI_Allreduce(MPI_IN_PLACE, maxima, 3, type, MPI_MAX, comm);
      const real_t equality = sums[0]-target_mass, mu = sums[1]/count;
      info.iterations = iteration;
      info.mass_error = equality;
      info.violation = maxima[2];
      info.stationarity = maxima[0];
      info.gap = sums[1];
      if (std::abs(equality) <= mass_tol && maxima[0] <= tol &&
          maxima[1] <= tol*field_scale && maxima[2] <= tol*field_scale &&
          sums[1] <= tol*volume*field_scale*field_scale)
      {
         info.converged = true;
         break;
      }
      if (iteration == options.mass_max_iterations || !std::isfinite(mu)) { break; }

      int failed = 0;
      for (int e = 0; e < ne; e++)
      {
         auto &el = elements[e];
         DenseMatrix K(mass[e]);
         for (int i = 0; i < m; i++)
         {
            const real_t weight = el.z(i)/el.s(i);
            const auto &indices = support[i];
            for (int jj = 0; jj < int(indices.size()); jj++)
            {
               const int j = indices[jj];
               const real_t value = weight*A(i,j);
               for (int kk = 0; kk <= jj; kk++)
               {
                  const int k = indices[kk];
                  K(j,k) += value*A(i,k);
               }
            }
         }
         for (int j = 0; j < n; j++)
         {
            el.scale(j) = 1/std::sqrt(K(j,j));
            for (int k = 0; k < j; k++) { K(k,j) = K(j,k); }
         }
         for (int j = 0; j < n; j++)
         {
            for (int k = 0; k < n; k++) { K(j,k) *= el.scale(j)*el.scale(k); }
         }
         bool factored = false;
         for (int attempt = 0; attempt < 3 && !factored; attempt++)
         {
            el.factor = K;
            if (attempt)
            {
               const real_t regularization = attempt == 1 ? 1e-14 : 1e-12;
               for (int j = 0; j < n; j++) { el.factor(j,j) += regularization; }
            }
            factored = el.Factor();
         }
         if (!factored) { failed = 1; continue; }
         el.w = el.c;
         el.Solve(el.w);
      }
      failed = allreduce(comm, failed, MPI_MAX);
      if (failed) { break; }

      auto Direction = [&](real_t sigma, bool corrector, real_t &dy)
      {
         // Do not drive complementarity far below the requested accuracy while
         // stationarity is still being resolved: that makes active rows singular
         // to working precision without improving the requested solution.
         const real_t central_mu = corrector ?
            std::max(sigma*mu, 0.01*tol*volume*field_scale*field_scale/count) : 0;
         real_t schur[2] = {0, 0};
         for (auto &el : elements)
         {
            Vector rhs(m), atrhs(n);
            for (int i = 0; i < m; i++)
            {
               const real_t cross = corrector ? el.affine_s(i)*el.affine_z(i) : 0;
               rhs(i) = (el.s(i)*el.z(i) + cross - central_mu + el.z(i)*el.rp(i))/el.s(i);
            }
            A.MultTranspose(rhs, atrhs);
            el.v = el.rd;
            el.v += atrhs;
            el.v *= -1;
            el.Solve(el.v);
            schur[0] += el.c*el.v;
            schur[1] += el.c*el.w;
         }
         MPI_Allreduce(MPI_IN_PLACE, schur, 2, type, MPI_SUM, comm);
         if (!(schur[1] > 0) || !std::isfinite(schur[0])) { return false; }
         dy = (equality+schur[0])/schur[1];
         for (auto &el : elements)
         {
            el.dx = el.v;
            el.dx.Add(-dy, el.w);
            A.Mult(el.dx, el.ds);
            el.ds += el.rp;
            for (int i = 0; i < m; i++)
            {
               const real_t cross = corrector ? el.affine_s(i)*el.affine_z(i) : 0;
               el.dz(i) = -(el.s(i)*el.z(i) + cross - central_mu + el.z(i)*el.ds(i))/el.s(i);
            }
         }
         return true;
      };
      auto StepLengths = [&]()
      {
         Vector step(2);
         step = 1.0;
         for (const auto &el : elements)
         {
            for (int i = 0; i < m; i++)
            {
               if (el.ds(i) < 0) { step(0) = std::min(step(0), -el.s(i)/el.ds(i)); }
               if (el.dz(i) < 0) { step(1) = std::min(step(1), -el.z(i)/el.dz(i)); }
            }
         }
         MPI_Allreduce(MPI_IN_PLACE, step.GetData(), 2, type, MPI_MIN, comm);
         return step;
      };
      real_t dy = 0;
      if (!Direction(0, false, dy)) { break; }
      Vector step = StepLengths();
      real_t affine_gap = 0;
      for (auto &el : elements)
      {
         el.affine_s = el.ds;
         el.affine_z = el.dz;
         for (int i = 0; i < m; i++)
         { affine_gap += (el.s(i)+step(0)*el.ds(i))*(el.z(i)+step(1)*el.dz(i)); }
      }
      affine_gap = allreduce(comm, affine_gap, MPI_SUM);
      const real_t ratio = std::max<real_t>(0, std::min<real_t>(1, affine_gap/sums[1]));
      if (!Direction(ratio*ratio*ratio, true, dy)) { break; }
      step = StepLengths();
      step *= 0.995;
      for (auto &el : elements)
      {
         el.x.Add(step(0), el.dx);
         el.s.Add(step(0), el.ds);
         el.z.Add(step(1), el.dz);
      }
      y += step(1)*dy;
   }
   for (int e = 0; e < ne; e++) { result[e] = elements[e].x; }
   return info;
}

namespace
{

void PrintJointFit(const char *name, const JointFitInfo &info)
{
   out << name << " joint full-DOF QP converged/iterations: " << info.converged
       << " " << info.iterations << "\n"
       << name << " joint mass residual: " << info.mass_error << "\n"
       << name << " joint bound violation: " << info.violation << "\n"
       << name << " joint scaled stationarity: " << info.stationarity << "\n"
       << name << " joint complementarity gap: " << info.gap << "\n";
}

class BlendMassFunctional : public Functional
{
   const Vector &weights;
   real_t target;
public:
   BlendMassFunctional(MPI_Comm comm, const Vector &w, real_t t)
      : Functional(comm, w.Size()), weights(w), target(t) { }
   void Mult(const Vector &x, Vector &y) const override
   {
      y.SetSize(1);
      y(0) = allreduce(GetComm(), weights*x, MPI_SUM) - target;
   }
   void EvalGradient(const Vector &, Vector &y) const override { y = weights; }
};

class BlendMetric : public MassOperator
{
   MPI_Comm comm;
   const Vector &diagonal;
   mutable SparseMatrix matrix;
public:
   BlendMetric(MPI_Comm c, const Vector &d) : comm(c), diagonal(d), matrix(d.Size())
   {
      height = width = d.Size();
      for (int i = 0; i < d.Size(); i++) { matrix.Add(i, i, d(i)); }
      matrix.Finalize();
   }
   void Mult(const Vector &x, Vector &y) const override
   {
      y.SetSize(x.Size());
      matrix.Mult(x, y);
   }
   void Riesz(const Vector &x, Vector &y) const override
   {
      y.SetSize(x.Size());
      for (int i = 0; i < x.Size(); i++) { y(i) = x(i) / diagonal(i); }
   }
   Operator &GetGradient(const Vector &) const override { return matrix; }
   real_t InnerProduct(const Vector &x, const Vector &y) const override
   {
      real_t value = 0;
      for (int i = 0; i < x.Size(); i++) { value += diagonal(i)*x(i)*y(i); }
      return allreduce(comm, value, MPI_SUM);
   }
   void MultDiff(const Vector &x, const Vector &y, Vector &z) const override
   {
      Vector difference(x);
      difference -= y;
      z.SetSize(x.Size());
      Mult(difference, z);
   }
   real_t DistanceSquaredTo(const Vector &x, const Vector &y) const override
   {
      Vector difference(x);
      difference -= y;
      return InnerProduct(difference, difference);
   }
};

#ifdef MFEM_USE_HIOP
class BlendHiOpProblem : public OptimizationProblem
{
   MPI_Comm comm;
   Vector weights;
   Vector equality, lower, upper;
public:
   BlendHiOpProblem(MPI_Comm c, const Vector &w, const Vector &d, real_t target)
      : OptimizationProblem(w.Size(), nullptr, nullptr), comm(c), weights(w),
        equality(1), lower(w.Size()), upper(w.Size())
   {
      numConstraints = 1;
      equality(0) = target;
      lower = 0.0;
      // z_K = sqrt(diagonal_K)*theta_K makes the L2 objective Euclidean,
      // avoiding a poorly conditioned diagonal objective in the HiOp solve.
      for (int i = 0; i < w.Size(); i++)
      {
         upper(i) = std::sqrt(d(i));
         weights(i) /= upper(i);
      }
      SetEqualityConstraint(equality);
      SetSolutionBounds(lower, upper);
   }
   real_t CalcObjective(const Vector &x) const override
   {
      real_t value = 0;
      for (int i = 0; i < x.Size(); i++) { value += 0.5*x(i)*x(i); }
      return allreduce(comm, value, MPI_SUM);
   }
   void CalcObjectiveGrad(const Vector &x, Vector &y) const override
   {
      y = x;
   }
   void CalcConstraint(int, const Vector &x, Vector &y) const override
   { y(0) = allreduce(comm, weights*x, MPI_SUM); }
   void CalcConstraintGrad(int, const Vector &, Vector &y) const override { y = weights; }
   hiop::hiopInterfaceBase::WeightedSpaceType getWeightedSpaceType() const override
   { return hiop::hiopInterfaceBase::WeightedSpaceType::Euclidean; }
};
#endif

// Correct mass in a restricted, convex subset of A*x >= b. Every element
// follows x_K(theta) = (1-theta)*x_bounded + theta*x_reference, 0 <= theta <= 1.
// This allows the existing box-based HiOp/LVPP solvers to retain PLBound
// certificates without interpreting nodal DOFs as Bernstein coefficients.
int CorrectMass(ParGridFunction &field, real_t target_mass, const DenseMatrix &A,
                 const Vector &b, const IntegrationRule &ir,
                 const NodalRemapOptions &options, const char *name)
{
   if (options.mass_solver == 0) { return 0; }
   auto &fes = *field.ParFESpace();
   const MPI_Comm comm = fes.GetComm();
   const bool root = fes.GetMyRank() == 0;
   const int ne = fes.GetNE(), n = A.Width();
   const real_t tolerance = options.mass_tolerance * std::max<real_t>(1, std::abs(target_mass));
   std::vector<Vector> old(ne), direction(ne), mass_row(ne);
   Vector weights(ne), diagonal(ne), theta(ne), one(n);
   one = 1.0;
   real_t current_mass = 0, volume = 0, maximum = 0;
   MassIntegrator integrator;
   integrator.SetIntRule(&ir);
   std::vector<DenseMatrix> mass(ne);
   for (int e = 0; e < ne; e++)
   {
      Array<int> dofs;
      fes.GetElementDofs(e, dofs);
      field.GetSubVector(dofs, old[e]);
      integrator.AssembleElementMatrix(*fes.GetFE(e), *fes.GetElementTransformation(e), mass[e]);
      mass_row[e].SetSize(n);
      mass[e].Mult(one, mass_row[e]);
      current_mass += mass_row[e]*old[e];
      volume += mass_row[e]*one;
      maximum = std::max(maximum, old[e].Max());
   }
   current_mass = allreduce(comm, current_mass, MPI_SUM);
   volume = allreduce(comm, volume, MPI_SUM);
   maximum = allreduce(comm, maximum, MPI_MAX);
   const real_t gap = target_mass-current_mass;
   if (root)
   {
      out << name << " mass correction: " << (options.mass_solver == 1 ? "HiOp" : "LVPP")
          << " on bounded elementwise blend factors.\n"
          << name << " mass before/target: " << current_mass << " " << target_mass << "\n";
   }
   if (std::abs(gap) <= tolerance) { return 0; }
   if (target_mass < -tolerance || volume <= 0)
   {
      if (root) { out << "Mass correction failed: invalid target mass or mesh volume.\n"; }
      return 3;
   }

   real_t reference = 0;
   if (gap > 0)
   {
      Vector slopes(A.Height());
      A.Mult(one, slopes);
      real_t cap = std::numeric_limits<real_t>::infinity();
      for (int i = 0; i < slopes.Size(); i++)
      {
         if (slopes(i) < 0) { cap = std::min(cap, b(i)/slopes(i)); }
      }
      reference = std::min(2*std::max(maximum, target_mass/volume), cap);
      reference *= 1 - 64*std::numeric_limits<real_t>::epsilon();
      if (!std::isfinite(reference) || reference <= 0)
      {
         if (root)
         {
            out << "Mass correction failed: no positive constant reference satisfies these bounds. "
                << "Try increasing -ncp.\n";
         }
         return 3;
      }
   }

   real_t capacity = 0, metric_scale = 0;
   for (int e = 0; e < ne; e++)
   {
      direction[e].SetSize(n);
      direction[e] = reference;
      direction[e] -= old[e];
      real_t change = mass_row[e]*direction[e];
      if (change*gap <= 0) { direction[e] = 0.0; change = 0; }
      weights(e) = std::abs(change);
      Vector md(n);
      mass[e].Mult(direction[e], md);
      diagonal(e) = std::max<real_t>(0, direction[e]*md);
      capacity += weights(e);
      metric_scale += diagonal(e);
   }
   capacity = allreduce(comm, capacity, MPI_SUM);
   metric_scale = allreduce(comm, metric_scale, MPI_SUM);
   if (root) { out << name << " available/requested mass change: " << capacity << " " << std::abs(gap) << "\n"; }
   if (capacity + tolerance < std::abs(gap) || capacity <= 0)
   {
      if (root)
      {
         out << "Mass correction failed: insufficient capacity in the chosen bounded blends. "
             << "This does not prove infeasibility of the full nodal optimization.\n";
      }
      return 3;
   }
   const real_t fraction = std::min<real_t>(1, std::abs(gap)/capacity);
   weights /= capacity;
   for (int e = 0; e < ne; e++)
   {
      diagonal(e) = diagonal(e) > 0 ? diagonal(e)/metric_scale : 1.0;
   }
   const real_t scaled_tol = 0.1*tolerance/capacity;
   theta = 0.0;
   if (std::abs(capacity-std::abs(gap)) <= tolerance) { theta = 1.0; }
   else if (options.mass_solver == 1)
   {
#ifdef MFEM_USE_HIOP
      BlendHiOpProblem problem(comm, weights, diagonal, fraction);
      HiopNlpOptimizer optimizer(comm);
      optimizer.SetOptimizationProblem(problem);
      optimizer.SetMaxIter(options.mass_max_iterations);
      optimizer.SetAbsTol(scaled_tol);
      optimizer.SetRelTol(scaled_tol);
      optimizer.SetPrintLevel(0);
      Vector start(ne);
      for (int e = 0; e < ne; e++) { start(e) = fraction*std::sqrt(diagonal(e)); }
      optimizer.Mult(start, theta);
      for (int e = 0; e < ne; e++) { theta(e) /= std::sqrt(diagonal(e)); }
      if (root)
      {
         out << name << " HiOp optimality converged/iterations: "
             << optimizer.GetConverged() << " " << optimizer.GetNumIterations() << "\n";
      }
#else
      MFEM_ABORT("HiOp mass correction requires MFEM_USE_HIOP.");
#endif
   }
   else
   {
      BlendMassFunctional functional(comm, weights, fraction);
      StackedFunctional constraints(functional);
      BlendMetric metric(comm, diagonal);
      Vector lower(ne), upper(ne);
      lower = 0.0;
      upper = 1.0;
      PointwiseFermiDirac generator(lower, upper);
      Array<LegendreFunction*> generators({&generator});
      Array<int> offsets({0, ne});
      Dykstra projector(comm, constraints, metric, generators, offsets, lower, upper,
                        scaled_tol, options.mass_max_iterations);
      projector.Project(theta);
   }

   int invalid = 0;
   for (int e = 0; e < ne; e++)
   {
      if (!std::isfinite(theta(e))) { invalid = 1; }
      // Solver iterates may miss their box slightly; clamping retains the
      // certificate, and the mass is checked again on the actual candidate.
      theta(e) = std::max<real_t>(0, std::min<real_t>(1, theta(e)));
   }
   invalid = allreduce(comm, invalid, MPI_MAX);
   const real_t blend_mass = allreduce(comm, weights*theta, MPI_SUM);
   const real_t weight_sum = allreduce(comm, weights.Sum(), MPI_SUM);
   const real_t closure_error = std::abs(blend_mass-fraction)*capacity;
   if (!invalid && closure_error > tolerance && closure_error <= 1000*tolerance)
   {
      // HiOp's absolute box tolerance in scaled variables can be amplified
      // when recovering tiny blend factors. Close a small residual along a
      // convex direction inside the same box, then validate the actual field.
      if (blend_mass > fraction)
      {
         theta *= fraction/blend_mass;
      }
      else
      {
         const real_t step = (fraction-blend_mass)/(weight_sum-blend_mass);
         for (int e = 0; e < ne; e++) { theta(e) += step*(1-theta(e)); }
      }
      if (root) { out << name << " mass residual before bounded closure: " << closure_error << "\n"; }
   }
   real_t corrected_mass = 0, violation = 0;
   for (int e = 0; e < ne; e++)
   {
      Vector candidate(old[e]), slack(A.Height());
      candidate.Add(theta(e), direction[e]);
      corrected_mass += mass_row[e]*candidate;
      A.Mult(candidate, slack);
      slack -= b;
      violation = std::max(violation, -slack.Min());
   }
   corrected_mass = allreduce(comm, corrected_mass, MPI_SUM);
   violation = allreduce(comm, violation, MPI_MAX);
   const real_t bounds_tol = 10*std::max(options.abs_tol,
                                        options.rel_tol * std::max<real_t>(1,
                                           std::max(maximum, options.upper_constraint ? options.upper : 0)));
   if (root)
   {
      out << name << " mass correction residual: " << corrected_mass-target_mass << "\n"
          << name << " post-correction constraint violation: " << violation << "\n";
   }
   if (invalid || std::abs(corrected_mass-target_mass) > tolerance || violation > bounds_tol)
   {
      if (root) { out << "Mass correction failed validation; retaining the bounded input field.\n"; }
      return 3;
   }
   for (int e = 0; e < ne; e++)
   {
      old[e].Add(theta(e), direction[e]);
      Array<int> dofs;
      fes.GetElementDofs(e, dofs);
      field.SetSubVector(dofs, old[e]);
   }
   return 0;
}

struct ComparisonStats
{
   const char *name;
   int rows = 0;
   int failures = 0;
   int max_sweeps = 0;
   long long total_sweeps = 0;
   real_t setup_seconds = 0;
   real_t fit_seconds = 0;
   real_t min_fit_seconds = 0;
   real_t max_fit_seconds = 0;
   real_t errors[3] = {0, 0, 0}; // transfer, correction, analytic reference
   real_t constraint_violation = 0;
   real_t sampled_min = std::numeric_limits<real_t>::infinity();
   real_t sampled_max = -std::numeric_limits<real_t>::infinity();
   real_t roundtrip_error = 0;
   real_t mass_seconds = 0;
   int mass_status = 0;
   FieldStats field;
};

// All methods see the same discrete source and the same target polynomial.
// For Bernstein, x_n = V*x_b, t_b = V^-1*t_n, and M_b = V^T*M_n*V.
int CompareFits(ParGridFunction &source, ParGridFunction &target,
                 const Vector &samples, const IntegrationRule &ir,
                 Coefficient &initial_condition, const DenseMatrix &L,
                 const DenseMatrix &U, int ncp, real_t bound_setup_seconds,
                 real_t sampling_seconds, const NodalRemapOptions &options,
                 bool visualization)
{
   MFEM_VERIFY(options.repetitions > 0, "Use rep >= 1.");
   ParFiniteElementSpace &fes = *target.ParFESpace();
   const MPI_Comm comm = fes.GetComm();
   const MPI_Datatype type = MPITypeMap<real_t>::mpi_type;
   const bool root = fes.GetMyRank() == 0;
   const auto geometry = fes.GetTypicalFE()->GetGeomType();
   const int ne = fes.GetNE(), n = fes.GetTypicalFE()->GetDof();
   std::vector<DenseMatrix> mass(ne);
   std::vector<Vector> targets(ne);
   StopWatch timer;
   MPI_Barrier(comm);
   timer.Start();
   for (int e = 0; e < ne; e++)
   {
      targets[e].SetSize(n);
      ProjectElement(*fes.GetFE(e), *fes.GetElementTransformation(e), ir,
                     samples, e*ir.GetNPoints(), mass[e], targets[e]);
      Array<int> dofs;
      fes.GetElementDofs(e, dofs);
      target.SetSubVector(dofs, targets[e]);
   }
   timer.Stop();
   real_t projection_seconds = timer.RealTime();
   MPI_Allreduce(MPI_IN_PLACE, &projection_seconds, 1, type, MPI_MAX, comm);
   MPI_Allreduce(MPI_IN_PLACE, &sampling_seconds, 1, type, MPI_MAX, comm);

   const IntegrationRule &diagnostic_ir =
      IntRules.Get(geometry, std::max(ir.GetOrder(), 8*options.order));
   const IntegrationRule *error_rules[Geometry::NumGeom] = {};
   error_rules[geometry] = &diagnostic_ir;
   GeometryRefiner refiner;
   const IntegrationRule &sample_ir =
      refiner.Refine(geometry, std::max(16, 4*options.order))->RefPts;
   const FieldStats source_stats = MeasureField(source, L, U, diagnostic_ir);
   const FieldStats target_stats = MeasureField(target, L, U, diagnostic_ir);
   const real_t source_analytic_error = source.ComputeL2Error(initial_condition, error_rules);
   const real_t target_analytic_error = target.ComputeL2Error(initial_condition, error_rules);
   const auto global_size = fes.GlobalTrueVSize();
   if (root)
   {
      out << std::setprecision(12)
          << "Basis comparison: identical nodal source, destination mesh, and L2 target.\n"
          << (options.mass_solver == 4 ?
              "All three fits use the full-DOF joint conservative QP.\n" :
              "All three local fits use dual coordinate descent.\n")
          << "Conservation solver (0=none, 1=HiOp blend, 2=LVPP blend, 4=joint QP): " << options.mass_solver << "\n"
          << "Nodal-only: nodal bounds. PLBound: nodal positivity and L bounds.\n"
          << "Bernstein: coefficient bounds on the same polynomial space.\n"
          << "Upper constraints: " << (options.upper_constraint ? "enabled" : "disabled")
          << " (nodal upper / U upper / Bernstein coefficient upper).\n"
          << "Upper bound: " << options.upper << "\n"
          << "Number of unknowns: " << global_size << "\n"
          << "PLBound control points in 1D: " << ncp << "\n"
          << "Transfer quadrature order: " << ir.GetOrder() << "\n"
          << "Sampling time (max rank, seconds): " << sampling_seconds << "\n"
          << "Common target projection time (max rank, seconds): " << projection_seconds << "\n"
          << "Source analytic L2 error (quadrature): " << source_analytic_error << "\n"
          << "Target analytic L2 error (quadrature): " << target_analytic_error << "\n";
      PrintStats("Source", source_stats);
      PrintStats("Target", target_stats);
   }

   ComparisonStats stats[3];
   stats[0].name = "Nodal-only";
   stats[1].name = "PLBound";
   stats[2].name = "Bernstein";
   ParGridFunction result(&fes);
   int all_failures = 0;
   int mass_failures = 0;
   for (int method = 0; method < 3; method++)
   {
      auto &s = stats[method];
      Vector b;
      DenseMatrix A, V, inverse_V;
      timer.Clear();
      timer.Start();
      if (method == 1)
      {
         NodalRemapOptions plb_options = options;
         plb_options.plbound = true;
         A = ConstraintMatrix(L, U, plb_options, b);
      }
      else { A = BoxConstraints(n, options, b); }
      if (method == 2)
      {
         L2_FECollection bernstein(options.order, fes.GetMesh()->Dimension(), BasisType::Positive);
         const auto &bern_fe = *bernstein.FiniteElementForGeometry(geometry);
         const auto &nodes = fes.GetTypicalFE()->GetNodes();
         V.SetSize(n);
         Vector shape(n);
         for (int i = 0; i < n; i++)
         {
            bern_fe.CalcShape(nodes.IntPoint(i), shape);
            for (int j = 0; j < n; j++) { V(i,j) = shape(j); }
         }
         DenseMatrixInverse inverse(V, false);
         inverse.GetInverseMatrix(inverse_V);
      }
      timer.Stop();
      s.setup_seconds = timer.RealTime() + (method == 1 ? bound_setup_seconds : 0);
      MPI_Allreduce(MPI_IN_PLACE, &s.setup_seconds, 1, type, MPI_MAX, comm);
      s.rows = A.Height();
      std::vector<real_t> times;
      JointFitInfo joint_info;
      for (int rep = -1; rep < options.repetitions; rep++)
      {
         s.failures = s.max_sweeps = 0;
         s.total_sweeps = 0;
         s.constraint_violation = s.roundtrip_error = 0;
         MPI_Barrier(comm);
         timer.Clear();
         timer.Start();
         if (options.mass_solver == 4)
         {
            std::vector<DenseMatrix> solve_mass(mass);
            std::vector<Vector> solve_target(targets), solved;
            if (method == 2)
            {
               for (int e = 0; e < ne; e++)
               {
                  inverse_V.Mult(targets[e], solve_target[e]);
                  RAP(mass[e], V, solve_mass[e]);
               }
            }
            joint_info = FitConservative(comm, solve_mass, solve_target, A, b,
                                         source_stats.mass, options, solved);
            // The joint status is global, not a count of failed local solves.
            s.mass_status = joint_info.converged ? 0 : 3;
            for (int e = 0; e < ne; e++)
            {
               Vector nodal(n);
               if (method == 2) { V.Mult(solved[e], nodal); }
               else { nodal = solved[e]; }
               Array<int> dofs;
               fes.GetElementDofs(e, dofs);
               result.SetSubVector(dofs, nodal);
            }
         }
         else for (int e = 0; e < ne; e++)
         {
            Vector local_target(n), fitted(n), nodal(n);
            DenseMatrix bern_mass;
            const DenseMatrix *local_mass = &mass[e];
            local_target = targets[e];
            if (method == 2)
            {
               inverse_V.Mult(targets[e], local_target);
               RAP(mass[e], V, bern_mass);
               local_mass = &bern_mass;
            }
            // Use the same element tolerance, determined in the common nodal
            // representation, instead of a basis-dependent relative scale.
            NodalRemapOptions local_options = options;
            real_t scale = std::max<real_t>(1, targets[e].Normlinf());
            if (options.upper_constraint) { scale = std::max(scale, options.upper); }
            local_options.abs_tol = std::max(options.abs_tol, options.rel_tol * scale);
            local_options.rel_tol = 0;
            const LocalFitInfo info = FitElement(*local_mass, local_target, A, b,
                                                local_options, fitted);
            s.failures += !info.converged;
            s.max_sweeps = std::max(s.max_sweeps, info.sweeps);
            s.total_sweeps += info.sweeps;
            if (method == 2) { V.Mult(fitted, nodal); }
            else { nodal = fitted; }
            Array<int> dofs;
            fes.GetElementDofs(e, dofs);
            result.SetSubVector(dofs, nodal);
         }
         timer.Stop();
         real_t seconds = timer.RealTime();
         MPI_Allreduce(MPI_IN_PLACE, &seconds, 1, type, MPI_MAX, comm);
         if (rep >= 0) { times.push_back(seconds); }
      }
      std::sort(times.begin(), times.end());
      s.fit_seconds = (times[(times.size()-1)/2] + times[times.size()/2]) / 2;
      s.min_fit_seconds = times.front();
      s.max_fit_seconds = times.back();
      MPI_Allreduce(MPI_IN_PLACE, &s.failures, 1, MPI_INT, MPI_SUM, comm);
      MPI_Allreduce(MPI_IN_PLACE, &s.max_sweeps, 1, MPI_INT, MPI_MAX, comm);
      MPI_Allreduce(MPI_IN_PLACE, &s.total_sweeps, 1, MPI_LONG_LONG_INT, MPI_SUM, comm);
      all_failures += s.failures;
      if (options.mass_solver && options.mass_solver != 4)
      {
         if (s.failures)
         {
            s.mass_status = 3;
            if (root) { out << s.name << ": skipping mass correction after unconverged bounded fits.\n"; }
         }
         else
         {
            timer.Clear();
            MPI_Barrier(comm);
            timer.Start();
            DenseMatrix nodal_A(A);
            if (method == 2) { Mult(A, inverse_V, nodal_A); }
            s.mass_status = CorrectMass(result, source_stats.mass, nodal_A, b,
                                        diagnostic_ir, options, s.name);
            timer.Stop();
            s.mass_seconds = timer.RealTime();
            MPI_Allreduce(MPI_IN_PLACE, &s.mass_seconds, 1, type, MPI_MAX, comm);
         }
         mass_failures += s.mass_status != 0;
      }
      if (options.mass_solver == 4)
      {
         mass_failures += s.mass_status != 0;
         if (root) { PrintJointFit(s.name, joint_info); }
      }
      s.field = MeasureField(result, L, U, diagnostic_ir);
      s.errors[2] = result.ComputeL2Error(initial_condition, error_rules);
      for (int e = 0; e < ne; e++)
      {
         Array<int> dofs;
         fes.GetElementDofs(e, dofs);
         Vector nodal, coefficients(n), slack(A.Height()), roundtrip(n), values, target_values;
         result.GetSubVector(dofs, nodal);
         if (method == 2)
         {
            inverse_V.Mult(targets[e], coefficients);
            V.Mult(coefficients, roundtrip);
            roundtrip -= targets[e];
            s.roundtrip_error = std::max(s.roundtrip_error, roundtrip.Normlinf());
            inverse_V.Mult(nodal, coefficients);
            V.Mult(coefficients, roundtrip);
            roundtrip -= nodal;
            s.roundtrip_error = std::max(s.roundtrip_error, roundtrip.Normlinf());
         }
         else { coefficients = nodal; }
         A.Mult(coefficients, slack);
         slack -= b;
         s.constraint_violation = std::max(s.constraint_violation, -slack.Min());
         result.GetValues(e, sample_ir, values);
         s.sampled_min = std::min(s.sampled_min, values.Min());
         s.sampled_max = std::max(s.sampled_max, values.Max());
         result.GetValues(e, ir, values);
         target.GetValues(e, ir, target_values);
         auto &tr = *fes.GetElementTransformation(e);
         for (int q = 0; q < ir.GetNPoints(); q++)
         {
            const auto &ip = ir.IntPoint(q);
            tr.SetIntPoint(&ip);
            const real_t weight = ip.weight * tr.Weight();
            const real_t difference = values(q) - samples(e*ir.GetNPoints()+q);
            const real_t correction = values(q) - target_values(q);
            s.errors[0] += weight * difference * difference;
            s.errors[1] += weight * correction * correction;
         }
      }
      MPI_Allreduce(MPI_IN_PLACE, s.errors, 2, type, MPI_SUM, comm);
      MPI_Allreduce(MPI_IN_PLACE, &s.constraint_violation, 1, type, MPI_MAX, comm);
      MPI_Allreduce(MPI_IN_PLACE, &s.roundtrip_error, 1, type, MPI_MAX, comm);
      MPI_Allreduce(MPI_IN_PLACE, &s.sampled_min, 1, type, MPI_MIN, comm);
      MPI_Allreduce(MPI_IN_PLACE, &s.sampled_max, 1, type, MPI_MAX, comm);
      s.errors[0] = std::sqrt(s.errors[0]);
      s.errors[1] = std::sqrt(s.errors[1]);
      if (root)
      {
         PrintStats(s.name, s.field);
         out << s.name << " constraint violation: " << s.constraint_violation << "\n"
             << s.name << " endpoint-inclusive sampled min/max: " << s.sampled_min
             << " " << s.sampled_max << "\n"
             << s.name << " L2 correction norm: " << s.errors[1] << "\n"
             << s.name << " total/max sweeps: " << s.total_sweeps << " " << s.max_sweeps << "\n"
             << s.name << " fit time min/median/max (seconds): " << s.min_fit_seconds
             << " " << s.fit_seconds << " " << s.max_fit_seconds << "\n";
         if (method == 2)
         {
            out << "Bernstein/nodal roundtrip error: " << s.roundtrip_error << "\n";
         }
      }
      if (visualization) { ShowField(result, s.name, 400*method, 450); }
   }
   if (root)
   {
      out << "\nComparison summary (seconds; median of " << options.repetitions
          << " max-rank timings after one warm-up per method)\n"
          << "method,rows,setup_s,fit_s,L2_transfer,L2_analytic,mass_change,unconverged,mass_s,mass_status\n";
      for (const auto &s : stats)
      {
         out << s.name << "," << s.rows << "," << s.setup_seconds << "," << s.fit_seconds
             << "," << s.errors[0] << "," << s.errors[2] << ","
             << s.field.mass - source_stats.mass << "," << s.failures << ","
             << s.mass_seconds << "," << s.mass_status << "\n";
      }
      out << "Setup includes constraint matrices and, for Bernstein, basis matrices.\n"
          << "Fit includes local factorization/iterations and Bernstein target, mass, and result conversion.\n"
          << "Common GSLIB sampling/projection, diagnostics, and GLVis are excluded from fit_s.\n"
          << "L2_transfer compares to the discrete source at transfer quadrature points.\n"
          << "L2_analytic compares to the analytic initial condition with finer quadrature.\n"
          << "min(L*e_K) is a common diagnostic, not the Bernstein feasibility test.\n"
          << "Mass correction, when selected, is a separate single-pass timing in mass_s.\n"
          << "Errors describe the final field; mass_status=0 means success (or no correction requested).\n"
          << "Number of locally unconverged solves (all methods): " << all_failures << "\n";
      if (all_failures)
      {
         out << "Unconverged rows are last iterates, not completed optimization comparisons.\n";
      }
      if (options.mass_solver == 4)
      {
         out << "For -opt 4, fit_s includes the joint bounds-and-mass solve; mass_s is zero.\n";
         if (mass_failures) { out << "Joint QP failures: " << mass_failures << "; failed rows are last iterates.\n"; }
      }
   }
   if (visualization)
   {
      ShowField(source, "Original nodal L2 field", 0);
      ShowField(target, "Common unconstrained L2 target", 400);
   }
   return all_failures ? 2 : (mass_failures ? 3 : 0);
}

} // namespace

int RemapNodalL2(ParMesh &source_mesh, const Vector &destination_nodes,
                 Coefficient &initial_condition,
                 const NodalRemapOptions &options, bool visualization)
{
#ifndef MFEM_USE_GSLIB
   MFEM_ABORT("Nodal L2 remap requires MFEM built with GSLIB.");
   return 1;
#else
   const int dim = source_mesh.Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "Nodal remap requires 2D or 3D.");
   MFEM_VERIFY(options.order >= 1 && options.order <= 11,
               "Nodal PLBound remap currently supports orders 1 through 11.");
   MFEM_VERIFY(options.ncp >= 2 && (options.cp_type == 0 || options.cp_type == 1),
               "Use ncp >= 2 and cpt = 0 or 1.");
   MFEM_VERIFY(options.max_sweeps > 0 && options.abs_tol > 0 &&
               options.rel_tol >= 0 && std::isfinite(options.abs_tol) &&
               std::isfinite(options.rel_tol), "Invalid local solver tolerances.");
   MFEM_VERIFY(!options.upper_constraint ||
               (options.upper >= 0 && std::isfinite(options.upper)),
               "The upper bound must be finite and nonnegative.");
   MFEM_VERIFY(((options.mass_solver >= 0 && options.mass_solver <= 2) || options.mass_solver == 4) &&
               options.mass_max_iterations > 0 && options.mass_tolerance > 0 &&
               std::isfinite(options.mass_tolerance), "Invalid mass correction options.");
   const Geometry::Type geometry = dim == 2 ? Geometry::SQUARE : Geometry::CUBE;
   for (int e = 0; e < source_mesh.GetNE(); e++)
   {
      MFEM_VERIFY(source_mesh.GetElementBaseGeometry(e) == geometry,
                  "Nodal PLBound remap requires a quadrilateral or hexahedral mesh.");
   }

   ParMesh destination_mesh(source_mesh, true);
   destination_mesh.SetNodes(destination_nodes);
   L2_FECollection fec(options.order, dim, BasisType::GaussLegendre);
   ParFiniteElementSpace source_fes(&source_mesh, &fec);
   ParFiniteElementSpace destination_fes(&destination_mesh, &fec);
   ParGridFunction source(&source_fes), target(&destination_fes), fitted(&destination_fes);
   source.ProjectCoefficient(initial_condition);

   StopWatch setup_timer;
   setup_timer.Start();
   PLBound bounds(&destination_fes, options.ncp, options.cp_type);
   DenseMatrix L = bounds.GetLowerBoundMatrix(dim, &destination_fes);
   setup_timer.Stop();
   real_t bound_setup_seconds = setup_timer.RealTime();
   setup_timer.Start();
   DenseMatrix U = bounds.GetUpperBoundMatrix(dim, &destination_fes);
   setup_timer.Stop();
   if (options.upper_constraint) { bound_setup_seconds = setup_timer.RealTime(); }
   const int qorder = options.quadrature_order < 0 ?
                      2 * options.order + 8 : options.quadrature_order;
   MFEM_VERIFY(qorder >= 2 * options.order, "Use qo >= 2*order for the mass matrix.");
   const IntegrationRule &ir = IntRules.Get(geometry, qorder);
   const int nq = ir.GetNPoints(), ne = destination_fes.GetNE();
   const int count = ne * nq;
   StopWatch sampling_timer;
   MPI_Barrier(source_mesh.GetComm());
   sampling_timer.Start();
   Vector points(dim * count);
   for (int e = 0; e < ne; e++)
   {
      auto &tr = *destination_fes.GetElementTransformation(e);
      DenseMatrix physical;
      tr.Transform(ir, physical);
      for (int q = 0; q < nq; q++)
      {
         tr.SetIntPoint(&ir.IntPoint(q));
         MFEM_VERIFY(tr.Jacobian().Det() > 0, "Inverted destination element.");
         for (int d = 0; d < dim; d++) { points(d*count + e*nq + q) = physical(d,q); }
      }
   }

   // Sample the discrete old field, not the analytic initial condition.
   FindPointsGSLIB finder(source_mesh.GetComm());
   finder.Setup(source_mesh);
   Vector samples(count);
   finder.Interpolate(points, source, samples);
   int missing = 0;
   for (int i = 0; i < finder.GetCode().Size(); i++)
   {
      missing += finder.GetCode()[i] == 2;
   }
   MPI_Allreduce(MPI_IN_PLACE, &missing, 1, MPI_INT, MPI_SUM, source_mesh.GetComm());
   finder.FreeData();
   MFEM_VERIFY(missing == 0, "Destination quadrature points outside the source mesh: " << missing);
   sampling_timer.Stop();
   if (options.compare)
   {
      return CompareFits(source, target, samples, ir, initial_condition, L, U,
                         bounds.GetNControlPoints(), bound_setup_seconds,
                         sampling_timer.RealTime(), options, visualization);
   }

   Vector b;
   DenseMatrix A = ConstraintMatrix(L, U, options, b);
   int failures = 0, max_sweeps = 0;
   const bool joint = options.mass_solver == 4;
   std::vector<DenseMatrix> joint_mass(joint ? ne : 0);
   std::vector<Vector> joint_target(joint ? ne : 0);
   real_t errors[3] = {0, 0, 0};
   MassIntegrator integrator;
   integrator.SetIntRule(&ir);
   StopWatch timer;
   timer.Start();
   for (int e = 0; e < ne; e++)
   {
      const auto &fe = *destination_fes.GetFE(e);
      auto &tr = *destination_fes.GetElementTransformation(e);
      const int n = fe.GetDof();
      DenseMatrix mass;
      integrator.AssembleElementMatrix(fe, tr, mass);
      Vector rhs(n), shape(n), t(n), x(n);
      rhs = 0.0;
      for (int q = 0; q < nq; q++)
      {
         const auto &ip = ir.IntPoint(q);
         tr.SetIntPoint(&ip);
         fe.CalcShape(ip, shape);
         rhs.Add(ip.weight * tr.Weight() * samples(e*nq+q), shape);
      }
      DenseMatrixInverse inverse(mass, true);
      inverse.Mult(rhs, t);
      if (joint)
      {
         joint_mass[e] = mass;
         joint_target[e] = t;
         x = t;
      }
      else
      {
         const LocalFitInfo info = FitElement(mass, t, A, b, options, x);
         failures += !info.converged;
         max_sweeps = std::max(max_sweeps, info.sweeps);
      }
      Array<int> dofs;
      destination_fes.GetElementDofs(e, dofs);
      target.SetSubVector(dofs, t);
      fitted.SetSubVector(dofs, x);
      for (int q = 0; q < nq; q++)
      {
         const auto &ip = ir.IntPoint(q);
         tr.SetIntPoint(&ip);
         fe.CalcShape(ip, shape);
         const real_t tv = shape * t, xv = shape * x, old = samples(e*nq+q);
         const real_t w = ip.weight * tr.Weight();
         errors[0] += w * (tv-old) * (tv-old);
         errors[1] += w * (xv-old) * (xv-old);
         errors[2] += w * (xv-tv) * (xv-tv);
      }
   }
   timer.Stop();
   real_t seconds = timer.RealTime();
   const MPI_Comm comm = source_mesh.GetComm();
   const MPI_Datatype type = MPITypeMap<real_t>::mpi_type;
   MPI_Allreduce(MPI_IN_PLACE, errors, 3, type, MPI_SUM, comm);
   MPI_Allreduce(MPI_IN_PLACE, &seconds, 1, type, MPI_MAX, comm);
   MPI_Allreduce(MPI_IN_PLACE, &failures, 1, MPI_INT, MPI_SUM, comm);
   MPI_Allreduce(MPI_IN_PLACE, &max_sweeps, 1, MPI_INT, MPI_MAX, comm);
   // Sample more finely for diagnostics; samples alone do not certify positivity.
   const IntegrationRule &diagnostic_ir = IntRules.Get(geometry, std::max(qorder, 8*options.order));
   const FieldStats source_stats = MeasureField(source, L, U, diagnostic_ir);
   const FieldStats target_stats = MeasureField(target, L, U, diagnostic_ir);
   const FieldStats bounded_stats = MeasureField(fitted, L, U, diagnostic_ir);
   ParGridFunction bounded(fitted);
   int mass_status = 0;
   real_t mass_seconds = 0;
   if (options.mass_solver)
   {
      if (failures)
      {
         mass_status = 3;
         if (source_mesh.GetMyRank() == 0)
         { out << "Skipping mass correction after unconverged bounded fits.\n"; }
      }
      else
      {
         timer.Clear();
         MPI_Barrier(comm);
         timer.Start();
         if (joint)
         {
            std::vector<Vector> result;
            const JointFitInfo info = FitConservative(comm, joint_mass, joint_target,
                                      A, b, source_stats.mass, options, result);
            mass_status = info.converged ? 0 : 3;
            for (int e = 0; e < ne; e++)
            {
               Array<int> dofs;
               destination_fes.GetElementDofs(e, dofs);
               fitted.SetSubVector(dofs, result[e]);
            }
            if (source_mesh.GetMyRank() == 0) { PrintJointFit("Final", info); }
         }
         else
         {
            mass_status = CorrectMass(fitted, source_stats.mass, A, b, diagnostic_ir,
                                      options, "Final");
         }
         timer.Stop();
         mass_seconds = timer.RealTime();
         MPI_Allreduce(MPI_IN_PLACE, &mass_seconds, 1, type, MPI_MAX, comm);
      }
      errors[1] = errors[2] = 0;
      for (int e = 0; e < ne; e++)
      {
         Vector values, target_values;
         fitted.GetValues(e, ir, values);
         target.GetValues(e, ir, target_values);
         auto &tr = *destination_fes.GetElementTransformation(e);
         for (int q = 0; q < nq; q++)
         {
            const auto &ip = ir.IntPoint(q);
            tr.SetIntPoint(&ip);
            const real_t weight = ip.weight * tr.Weight();
            const real_t difference = values(q)-samples(e*nq+q);
            const real_t correction = values(q)-target_values(q);
            errors[1] += weight*difference*difference;
            errors[2] += weight*correction*correction;
         }
      }
      MPI_Allreduce(MPI_IN_PLACE, errors+1, 2, type, MPI_SUM, comm);
   }
   const FieldStats fitted_stats = MeasureField(fitted, L, U, diagnostic_ir);
   const auto size = destination_fes.GlobalTrueVSize();
   if (source_mesh.GetMyRank() == 0)
   {
      out << std::setprecision(12)
          << "Nodal L2 remap: Gauss-Legendre basis throughout (no Bernstein).\n"
          << "Number of unknowns: " << size << "\n"
          << "Constraints: e_K >= 0"
          << (options.plbound ? ", L*e_K >= 0" : "")
          << (options.upper_constraint ? ", U*e_K <= upper" : "") << "\n"
          << "Upper bound: " << options.upper << "\n"
          << "PLBound control points in 1D: " << bounds.GetNControlPoints() << "\n"
          << "Transfer quadrature order: " << qorder << "\n";
      PrintStats("Source", source_stats);
      PrintStats("Target", target_stats);
      if (options.mass_solver && !joint) { PrintStats("Bounded", bounded_stats); }
      PrintStats("Final", fitted_stats);
      out << "Final PLBound constraint violation: "
          << std::max<real_t>(0, -fitted_stats.minimum[1]) << "\n";
      if (options.upper_constraint)
      {
         out << "Final upper constraint violation: "
             << std::max<real_t>(0, fitted_stats.maximum[1] - options.upper) << "\n";
      }
      out << "Target L2 transfer error (quadrature): " << std::sqrt(errors[0]) << "\n"
          << "Final L2 transfer error (quadrature): " << std::sqrt(errors[1]) << "\n"
          << "L2 correction norm: " << std::sqrt(errors[2]) << "\n"
          << "Target mass change: " << target_stats.mass - source_stats.mass << "\n"
          << "Final mass change: " << fitted_stats.mass - source_stats.mass << "\n"
          << "Conservation solver (0=none, 1=HiOp blend, 2=LVPP blend, 4=joint QP): " << options.mass_solver << "\n"
          << "Mass correction status: " << mass_status << "\n"
          << "Mass correction time (max rank, seconds): " << mass_seconds << "\n"
          << "Maximum local solver sweeps: " << max_sweeps << "\n"
          << "Number of locally unconverged solves: " << failures << "\n"
          << "Projection and local fit time (max rank, seconds): " << seconds << "\n";
      if (joint && mass_status)
      { out << "Joint QP failed: final field is a last iterate, not a converged conservative remap.\n"; }
   }
   if (visualization)
   {
      ShowField(source, "Original nodal L2 field", 0);
      ShowField(target, "Unconstrained L2 remap", 400);
      if (joint) { ShowField(fitted, "Joint bounded conservative remap", 800); }
      else if (options.mass_solver)
      {
         ShowField(bounded, "Before mass correction", 0, 450);
         ShowField(fitted, "After mass correction", 400, 450);
      }
      else { ShowField(fitted, "Bounded nodal L2 remap", 800); }
   }
   return failures ? 2 : mass_status;
#endif
}

} // namespace mfem
