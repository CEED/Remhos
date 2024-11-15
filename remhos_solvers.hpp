// Copyright (c) 2024, Lawrence Livermore National Security, LLC. Produced at
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

#ifndef MFEM_REMHOS_SOLVERS
#define MFEM_REMHOS_SOLVERS

#include "mfem.hpp"

namespace mfem
{

class LimitedTimeDependentOperator : public TimeDependentOperator
{
protected:
   real_t dt;

public:
   /** @brief Construct a "square" LimitedTimeDependentOperator (u,t) -> k(u,t),
      where u and k have the same dimension @a n. */
   LimitedTimeDependentOperator(int n = 0, real_t t = 0.0)
      : TimeDependentOperator(n, t) { }

   /** @brief Construct a LimitedTimeDependentOperator (u,t) -> k(u,t), where
      u and k have dimensions @a w and @a h, respectively. */
   LimitedTimeDependentOperator(int h, int w, real_t t = 0.0)
      : TimeDependentOperator(h, w, t) { }

   virtual ~LimitedTimeDependentOperator() { }

   virtual void SetDt(double dt_) { dt = dt_; }
   virtual real_t GetDt() const { return dt; }

   void Mult(const Vector &u, Vector &k) const override
   {
      MultUnlimited(u, k);
      LimitMult(u, k);
   }

   /// Perform the unlimited action of the operator
   virtual void MultUnlimited(const Vector &u, Vector &k) const = 0;

   /// Limit the action vector @a k
   virtual void LimitMult(const Vector &u, Vector &k) const = 0;
};

class IDPODESolver : public ODESolver
{
protected:
   /// Pointer to the associated LimitedTimeDependentOperator.
   LimitedTimeDependentOperator *f;  // f(.,t) : R^n --> R^n

   void Init(TimeDependentOperator &f_) override
   { MFEM_ABORT("Limited time-dependent operator must be assigned!"); }
public:
   IDPODESolver() : ODESolver(), f(NULL) { }
   virtual ~IDPODESolver() { }

   virtual void Init(LimitedTimeDependentOperator &f_)
   { ODESolver::Init(f_); f = &f_; }
};

class ForwardEulerIDPSolver : public IDPODESolver
{
   Vector dx;
public:
   void Init(LimitedTimeDependentOperator &f) override;
   void Step(Vector &x, double &t, double &dt) override;
};

class RK2IDPSolver : public IDPODESolver
{
   Vector dx12, dx;

public:
   void Init(LimitedTimeDependentOperator &f) override;
   void Step(Vector &x, double &t, double &dt) override;
};

} // namespace mfem

#endif // MFEM_REMHOS_SOLVERS
