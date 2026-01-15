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

   virtual void SetDt(real_t dt_) { dt = dt_; }
   virtual real_t GetDt() const { return dt; }

   void Mult(const Vector &u, Vector &k) const override
   {
      MultUnlimited(u, k);
      LimitMult(u, k);
   }

   /// Perform the unlimited action of the operator
   virtual void MultUnlimited(const Vector &u, Vector &k) const = 0;

   /// Compute mask of the state for the update
   virtual void ComputeMask(const Vector &u, Array<bool> &mask) const
   { MFEM_ABORT("Mask computation not implemented!"); }

   /// Limit the action vector @a k
   /// Assumes that MultUnlimited(u, k) has been called, which has computed the
   /// unlimited solution in @a k.
   virtual void LimitMult(const Vector &u, Vector &k) const = 0;
};

class IDPODESolver : public ODESolver
{
protected:
   /// Pointer to the associated LimitedTimeDependentOperator.
   LimitedTimeDependentOperator *f;  // f(.,t) : R^n --> R^n

   void Init(TimeDependentOperator &f_) override
   {
      auto lo = dynamic_cast<LimitedTimeDependentOperator *>(&f_);
      if (lo) { Init(*lo); }
      else    { MFEM_ABORT("LimitedTimeDependentOperator must be assigned!"); }
   }

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

class RKIDPSolver : public IDPODESolver
{
   const int s;
   const real_t *a, *b, *c;
   real_t *d;
   Vector *dxs;
   bool use_masks = false;
   Array<bool> mask;

   // This function constructs coefficients that transform eq. (2.16) from
   // JLG's paper to an update that only uses the previous limited updates.
   // This function does not depend on the Operator f in any way.
   void ConstructD();

   void UpdateMask(const Vector &x, const Vector &dx, real_t dt,
                   Array<bool> &mask);

public:
   RKIDPSolver(int s_, const real_t a_[], const real_t b_[], const real_t c_[]);
   ~RKIDPSolver();

   void UseMask(bool mask_on) { use_masks = mask_on; }

   /// Adds only DOFs that have mask = true.
   /// Must be public due to NVCC
   void AddMasked(const Array<bool> &mask, real_t b,
                  const Vector &vb, Vector &va);

   void Init(LimitedTimeDependentOperator &f) override;
   void Step(Vector &x, double &t, double &dt) override;
};

class RK2IDPSolver : public RKIDPSolver
{
   static const real_t a[], b[], c[];
public:
   RK2IDPSolver() : RKIDPSolver(2, a, b, c) { }
};

class RK3IDPSolver : public RKIDPSolver
{
   static const real_t a[], b[], c[];
public:
   RK3IDPSolver() : RKIDPSolver(3, a, b, c) { }
};

class RK4IDPSolver : public RKIDPSolver
{
   static const real_t a[], b[], c[];
public:
   RK4IDPSolver() : RKIDPSolver(4, a, b, c) { }
};

class RK6IDPSolver : public RKIDPSolver
{
   static const real_t a[], b[], c[];
public:
   RK6IDPSolver() : RKIDPSolver(6, a, b, c) { }
};

} // namespace mfem

#endif // MFEM_REMHOS_SOLVERS
