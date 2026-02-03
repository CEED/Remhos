#ifndef MFEM_REMAP_HPP
#define MFEM_REMAP_HPP

#include "mfem.hpp"
#include "general/forall.hpp"
#include "miniapps/autodiff/admfem.hpp"
#include "linalg/functional.hpp"

namespace mfem
{
MFEM_HOST_DEVICE
inline real_t sigmoid(const real_t x)
{
   return x > 0 ? 1.0 / (1.0 + std::exp(-x)) : std::exp(x) / (1.0 + std::exp(x));
}

MFEM_HOST_DEVICE
inline real_t sigmoid(const real_t x, const real_t l, const real_t u)
{
   const real_t scale = (u - l);
   return l + scale*sigmoid(x);
}

MFEM_HOST_DEVICE
inline real_t logit(const real_t x)
{
   const real_t x_clamped = std::min(std::max(x, 1e-10), 1.0 - 1e-10);
   return std::log(x_clamped / (1.0 - x_clamped));
}

MFEM_HOST_DEVICE
inline real_t logit(const real_t x, const real_t l, const real_t u)
{
   const real_t scale = (u - l);
   return scale < 1e-08 ? 0.0 : logit((x - l) / scale);
}

MFEM_HOST_DEVICE
inline real_t der_sigmoid(const real_t x)
{
   const real_t s = sigmoid(x);
   return s * (1.0 - s);
}

inline real_t der_sigmoid(const real_t x, const real_t l, const real_t u)
{
   real_t scale = (u - l);
   return scale*der_sigmoid(x);
}
// Mask Gradient g using normal cone conditions:
// g_i = min { g_i, 0 } if x_i < xmin_i - tol
//       max { g_i, 0 } if x_i > xmax_i + tol
//       g_i otherwise
inline real_t kkt_res(const BlockVector &x,
                      const BlockVector &xmin, const BlockVector &xmax,
                      const BlockVector &g, const real_t tol=1e-10)
{
   real_t res = 0.0;
   for (int iblock = 0; iblock < x.NumBlocks(); iblock++)
   {
      const Vector &x_block = x.GetBlock(iblock);
      const Vector &xmin_block = xmin.GetBlock(iblock);
      const Vector &xmax_block = xmax.GetBlock(iblock);
      const Vector &g_block = g.GetBlock(iblock);
      if (xmin_block.Size() == 0)
      {
         for (auto &gval : g_block)
         {
            res += std::min(abs(gval), 1.0);
         }
         continue;
      }
      for (int i = 0; i < x_block.Size(); i++)
      {
         real_t gval = g_block[i];
         if (x_block[i] < xmin_block[i] + tol && g_block[i] > 0.0)
         {
            gval = 0.0;
         }
         if (x_block[i] > xmax_block[i] - tol && g_block[i] < 0.0)
         {
            gval = 0.0;
         }
         res += std::min(abs(gval), 1.0);
      }
   }
   return res;
}

inline std::vector<FiniteElementSpace*> par2normal(
   std::vector<ParFiniteElementSpace*> &par_spaces)
{
   std::vector<FiniteElementSpace*> fespaces(0);
   for (auto &p : par_spaces) { fespaces.push_back(p); }
   return fespaces;
}


template <typename T>
inline std::vector<T*> ToRawPtrVector(const std::vector<std::unique_ptr<T>> &v)
{
   std::vector<T*> raw_ptrs;
   for (const auto &ptr : v) { raw_ptrs.push_back(ptr.get()); }
   return raw_ptrs;
}

inline std::vector<Vector*> ToRawPtrVector(const std::vector<Vector> &v)
{
   std::vector<Vector*> raw_ptrs;
   for (const auto &vec : v) { raw_ptrs.push_back(const_cast<Vector*>(&vec)); }
   return raw_ptrs;
}

// Extract a single component from a vector-valued QuadratureFunction
inline void VecQF2QF(const QuadratureFunction &qf_vec,
                     const int comp,
                     QuadratureFunction &qf)
{
   MFEM_VERIFY(qf_vec.GetSpace() == qf.GetSpace(),
               "QuadratureFunction spaces do not match.");
   MFEM_VERIFY(qf_vec.GetVDim() > comp,
               "QuadratureFunction dimension is smaller than component index.");
   MFEM_VERIFY(qf.GetVDim() == 1,
               "QuadratureFunction vector dimension is not 1.");
   const int dim = qf_vec.GetVDim();
   const int N = qf_vec.GetSpace()->GetSize();
   const real_t *qf_vec_data = qf_vec.GetData();
   real_t *qf_data = qf.GetData();
   for (int i=comp, j=0; j<N; i+=dim, j++)
   {
      qf_data[j] = qf_vec_data[i];
   }
}

// Set a single component of a vector-valued QuadratureFunction
inline void QF2VecQF(const QuadratureFunction &qf,
                     const int comp,
                     QuadratureFunction &qf_vec)
{
   MFEM_VERIFY(qf_vec.GetSpace() == qf.GetSpace(),
               "QuadratureFunction spaces do not match.");
   MFEM_VERIFY(qf_vec.GetVDim() > comp,
               "QuadratureFunction dimension is smaller than component index.");
   MFEM_VERIFY(qf.GetVDim() == 1,
               "QuadratureFunction vector dimension is not 1.");
   const int dim = qf_vec.GetVDim();
   const int N = qf_vec.GetSpace()->GetSize();
   real_t *qf_vec_data = qf_vec.GetData();
   const real_t *qf_data = qf.GetData();
   for (int i=comp, j=0; j<N; i+=dim, j++)
   {
      qf_vec_data[i] = qf_data[j];
   }
}


class QuadratureDomainLFIntegrator : public LinearFormIntegrator
{
private:
   QuadratureFunction qf;
   const QuadratureSpace *qspace;
   FiniteElementSpace *fespace;
   mutable Vector qvals;
   std::vector<std::vector<std::unique_ptr<DenseMatrix>>> dof2q;
public:
   QuadratureDomainLFIntegrator(const QuadratureFunction &qf,
                                FiniteElementSpace &fes);

   void AssembleRHSElementVect(const FiniteElement &el,
                               ElementTransformation &Tr,
                               Vector &elvect) override;
};

// QuadratureFunction to dual vector, <qf, v> = \int qf v dx
// The returned vector is a T-vector in dual space.
class QuadratureLinearForm : public Operator
{
private:
   QuadratureSpace &qspace;
   FiniteElementSpace &fespace;
   // Mapping from dof 2 quadrature points
   // [max fe order]x[max ir order]
   std::vector<std::vector<std::unique_ptr<DenseMatrix>>> dof2q;
   mutable Vector Q_vec; // quadrature vector
   mutable Vector L_vec; // subdomain vector
   bool parallel;
public:
   QuadratureLinearForm(QuadratureSpace &qs, FiniteElementSpace &fes);

   void Mult(const Vector &x, Vector &y) const override;
};

/// @brief A composition of analytic function f:R^n->R
/// F: (u1, ..., un) |-> int f(u1, u2, ..., un) dx - target
/// with discrete functions ui, where ui is either a QuadratureFunction or a GridFunction
/// target is defaults to 0.0, and can be set by SetTarget() and SetTargets()
///
/// User must provide the composition function f and its gradient df,
/// f(const Vector &u)->real_t and
/// df(const Vector &u, const Vector &grad_u)
/// For example, f(u1, u2) = u1 + u2
///   f(const Vector &u) { return u[0] + u[1]; }
///   df(const Vector &u, Vector &grad_u) { grad_u[0] = 1.0, grad_u[1] = 1.0; }
///
/// A quadrature space must be provided to evaluate the integral.
/// At each call of Mult() or EvalGradient(),
/// the input vector x will be parsed into multiple components,
/// and evaluated at quadrature points.
/// To avoid re-evaluating the input vector at each call,
/// EnableManualUpdate() can be called to freeze the discrete variables,
/// and Update(const Vector &x) can be called to update the variables manually.
///
/// This class also supports multiple functions and gradients,
/// where choosing the function index is done by SetIndex(int idx).
/// Also, see
///
/// The evaluation point x is a block vector ordered as ByVDim. That is,
/// x = [u1_dofs, u2_dofs, ..., un_dofs]
///
/// For evaluation, the input variables will be evaluated at quadrature points.
/// F: (u1, ..., un) = \int f(u1, u2, ..., un) dx
/// \nabla F: (u1, ..., un) = [\int f_i(u1, u2, ..., un) * v1 dx, ..., \int f_n(u1, u2, ..., un) * vn dx]
/// where f_i is the derivative of f with respect to u_i, provided by .
/// When ui is a QuadratureFunction, vi is just 1. Otherwise, vi is the FE basis function
///
/// The support for Hessian will be added in the future.
///
class ComposedFunctional : public SharedFunctional
{
   typedef std::function<real_t(const Vector &)> FuncType;
   typedef std::function<void(const Vector &, Vector &)> GradType;
public:
   /// @brief Construct an CompsedFunctional with multiple functions and gradients.
   /// This is useful for evaluating multiple functionals at once.
   /// @param funcs A vector of functions f:R^n->R
   /// @param grads A vector of gradients df:R^n->R^n
   /// @param qspace A QuadratureSpace defining the quadrature points
   /// @param fes A vector of FiniteElementSpace pointers, one for each variable
   /// @param space_idx Space index, where -1 indicates a QuadratureFunction
   /// and >= 0 indicates a FiniteElementSpace index.
   ComposedFunctional(FuncType f,
                      GradType g,
                      QuadratureSpace &qspace,
                      std::vector<FiniteElementSpace*> fes,
                      const Array<int> space_idx);

   ComposedFunctional(FuncType f,
                      GradType g,
                      QuadratureSpace &qspace,
                      std::vector<ParFiniteElementSpace*> fes,
                      const Array<int> space_idx)
      : ComposedFunctional(f, g, qspace, par2normal(fes), space_idx)
   {}

   const Array<int> &GetOffsets() const { return offsets; }
   void SetTarget(real_t target) { this->target = target; }
   void SetFunction(FuncType f, GradType g) { this->f = f; this->df = g; }

   /// Evaluate the derivative of <f, v> = \int f(u1, u2, ..., un) dx
   /// That is, y = [int f_i(u1, ..., un) * v1 dx, ..., int f_n(u1, ..., un) * vn dx]
   /// where f_i is the derivative of f with respect to u_i.
   void EvalGradientCurrent(Vector &y) const override;

   void MultCurrent(Vector &y) const override;

protected:
   mutable FuncType f;
   mutable GradType df;
private:
   real_t target;

   Array<int> offsets;
   const Array<int> space_idx; // -1: QF, >= 0: FESpace index
   const int num_vars; // number of variables, i.e., the input size of f

   QuadratureSpace &qspace;
   std::vector<FiniteElementSpace*> fespace;
   mutable std::vector<std::unique_ptr<GridFunction>> gfs;
   mutable std::vector<std::unique_ptr<QuadratureLinearForm>> qlf; // int qf*v

   mutable QuadratureFunction qf; // for integration
   mutable QuadratureFunction
   qf_in; // store input quadrature function vdim = num_vars
   mutable QuadratureFunction
   qf_out; // store output quadrature function vdim = num_vars

   mutable bool is_input_frozen; // ignore input until FreezeInput is called
#ifdef MFEM_USE_MPI
   std::vector<ParFiniteElementSpace*> par_fespace;
#endif

   void Initialize();

   // convert evaluation point x to quadrature functions and store in qf_in
   void ProcessX(const Vector &x) const override;

   void ShallowCopyProcessedX(SharedFunctional &owner) override;
};

template <int n>
class ComposedADFunctional : public ComposedFunctional
{
public:
   /// @brief Construct an CompsedFunctional with multiple functions and gradients.
   /// This is useful for evaluating multiple functionals at once.
   /// @param funcs A vector of functions f:R^n->R
   /// @param grads A vector of gradients df:R^n->R^n
   /// @param qspace A QuadratureSpace defining the quadrature points
   /// @param fes A vector of FiniteElementSpace pointers, one for each variable
   /// @param space_idx Space index, where -1 indicates a QuadratureFunction
   /// and >= 0 indicates a FiniteElementSpace index.
   ComposedADFunctional(
      std::function<void(Vector &, ad::ADVectorType&, ad::ADVectorType&)> f,
      QuadratureSpace &qspace,
      std::vector<FiniteElementSpace*> fes,
      const Array<int> space_idx)
      : ComposedFunctional(nullptr, nullptr, qspace, fes, space_idx)
      , funct(f)
      , ad_dummy(0) // dummy vector for AD param
   {
      this->f = [this](const Vector &x)
      {
         ad::ADVectorType ad_x(x);
         ad::ADVectorType y(1);
         this->funct(this->ad_dummy, ad_x, y); return y[0].value;
      };
      VectorFuncAutoDiff<1,n,0> adfunc(this->funct);
      this->df = [this, adfunc](const Vector &x, Vector &grad_x)
      {
         // Vector &xview = const_cast<Vector&>(x);
         Vector &xx = const_cast<Vector&>(x);
         grad_x.SetSize(n);
         DenseMatrix grad_mat(grad_x.GetData(), 1, n);
         adfunc.Jacobian(this->ad_dummy, xx, grad_mat);
      };
   }
private:
   std::function<void(Vector &, ad::ADVectorType&, ad::ADVectorType&)> funct;
   mutable Vector ad_dummy;
};


// // A matrix operator that represents a collection of vectors as column of a matrix.
// // For parallel use, each processor owns a part of each column vector.
// // That is, each processor owns rows of the matrix.
// // Returned vectors will be distributed when Mult() is called.
// // Returned vectors will be synchronized when MultTranspose() is called
// class ColArrayMatrix : public Operator
// {
// private:
//    std::vector<const real_t*> cols;
//    mutable Vector col; // column view
// public:
//    ColArrayMatrix(const std::vector<Vector*> &cols)
//       : Operator(cols[0]->Size(), cols.size()), parallel(false)
//    {
//       for (const auto &col : cols) { this->cols.push_back(col->GetData()); }
//    }
//
//    ColArrayMatrix(const std::vector<std::unique_ptr<Vector>> &cols)
//       : Operator(cols[0]->Size(), cols.size()), parallel(false)
//    {
//       for (const auto &col : cols) { this->cols.push_back(col->GetData()); }
//    }
//
//    ColArrayMatrix(const std::vector<Vector> &cols)
//       : Operator(cols.size(), cols[0].Size()), parallel(false)
//    {
//       for (const auto &col : cols) { this->cols.push_back(col.GetData()); }
//    }
//
//    ColArrayMatrix(const DenseMatrix &mat)
//       : Operator(mat.Height(), mat.Width()), parallel(false)
//    {
//       for (int i=0; i<mat.Width(); i++) { cols.push_back(mat.GetColumn(i)); }
//    }
//
//    // return the i-th column vector
//    const Vector &GetColumn(int i) const
//    {
//       col.SetDataAndSize(const_cast<real_t*>(cols[i]), height);
//       return col;
//    }
//    // return the i-th column vector
//    const Vector &operator[](int i) const { return GetColumn(i); }
//
//    void Mult(const Vector &x, Vector &y) const override
//    {
//       y.SetSize(height);
//       y = 0.0;
//       for (int i=0; i<width; i++)
//       {
//          col.SetDataAndSize(const_cast<real_t*>(cols[i]), height);
//          y.Add(x[i], col);
//       }
//    }
//
//    void MultTranspose(const Vector &x, Vector &y) const override
//    {
//       y.SetSize(width);
//       y = 0.0; // clear output vector
//       for (int i=0; i<width; i++)
//       {
//          col.SetDataAndSize(const_cast<real_t*>(cols[i]), height);
//          y[i] += x * col;
//       }
// #ifdef MFEM_USE_MPI
//       if (IsParallel())
//       {
//          MPI_Allreduce(MPI_IN_PLACE, y.GetData(), width,
//                        MPITypeMap<real_t>::mpi_type, MPI_SUM, GetComm());
//       }
// #endif
//    }
//
//    // parallel support
// public:
//    bool IsParallel() const { return parallel; }
// #ifdef MFEM_USE_MPI
//    void SetComm(MPI_Comm comm) { parallel = true; this->comm = comm; }
//    MPI_Comm GetComm() const { return comm; }
// #endif
//
// protected:
//    bool parallel = false;
// #ifdef MFEM_USE_MPI
//    MPI_Comm comm;
// #endif
// };
//
// // Solve rank-k perturbed linear system, (A + c UV^T)x = b
// // using the Woodbury formula:
// // (A + c UV^T)^{-1} = A^{-1} - c A^{-1}U(I_k + c V^TA^{-1}U)^{-1}V^TA^{-1}
// //
// // We solve Ay = z for k + 1 times and K w = t for 1 time.
// // We allocate O(m*k + k^2) memory for the algorithm.
// //
// // input:
// //    Ainv - inverse of the operator A (m x m)
// //    V, U - matrices of size (m, k)
// //    b - right-hand side vector of size (m)
// // output:
// //    x - solution vector of size (m), does not need to be initialized
// template <typename T>
// inline void Woodbury(MPI_Comm comm, const Operator &Ainv, const real_t c,
//                      const std::vector<T> &Uptr,
//                      const std::vector<T> &Vptr,
//                      const Vector &b, Vector &x)
// {
//    int rank;
//    MPI_Comm_rank(comm, &rank);
//    // Convert pointers to raw pointers
//    std::vector<Vector*> U = ToRawPtrVector(Uptr);
//    std::vector<Vector*> V = ToRawPtrVector(Vptr);
//    MFEM_VERIFY(Ainv.Height() == Ainv.Width(),
//                "Ainv must be a square operator");
//    MFEM_VERIFY(V.size() == U.size(),
//                "V and U must have the same number of columns");
//    for (int i=0; i<V.size(); i++)
//    {
//       MFEM_VERIFY(V[i]->Size() == Ainv.Height(),
//                   "V columns must have the same size as Ainv height");
//       MFEM_VERIFY(U[i]->Size() == Ainv.Height(),
//                   "U columns must have the same size as Ainv height");
//    }
//    MFEM_VERIFY(b.Size() == Ainv.Height(),
//                "b (" << b.Size() << ") must have the same size as Ainv (" << Ainv.Height() <<
//                ") height");
//
//    const int m = Ainv.Height();
//    const int k = V.size();
//    x.SetSize(b.Size());
//
//    // 1. Solve Ay = b
//    Vector y(b); // y = Ainv * b
//    Ainv.Mult(b, y);
//
//    // 2. Solve AZ = U
//    std::vector<Vector> Z;
//    for (int i=0; i<k; i++)
//    {
//       Z.emplace_back(m);
//       MPI_Barrier(comm);
//       Ainv.Mult(*U[i], Z[i]);
//       Z[i] *= c;
//    }
//
//    // 3. Compute K = I + V^T * Z
//    DenseMatrix K(k); // K = I + V^T * Z
//    if (rank == 0) { K.Diag(1.0, k); }
//    else { K = 0.0; }
//    // 4. Solve K w = V^T * y
//    Vector Vt_y(k);
//    for (int i=0; i<k; i++)
//    {
//       for (int j=0; j<k; j++)
//       {
//          K(i,j) += *V[i]*Z[j];
//       }
//       Vt_y[i] = *V[i]*y;
//    }
//    MPI_Allreduce(MPI_IN_PLACE, K.GetData(), k*k, MPITypeMap<real_t>::mpi_type,
//                  MPI_SUM, comm);
//    MPI_Allreduce(MPI_IN_PLACE, Vt_y.GetData(), k, MPITypeMap<real_t>::mpi_type,
//                  MPI_SUM, comm);
//    Vector w(k);
//    DenseMatrixInverse Kinv(K);
//    Kinv.Factor();
//    Kinv.Mult(Vt_y, w);
//
//    // 5. Compute x = y - Z * w
//    x = y;
//    for (int i=0; i<k; i++)
//    {
//       x.Add(-w[i], Z[i]);
//    }
// }
// inline void Woodbury(MPI_Comm comm, const Operator &Ainv, const real_t c,
//                      const DenseMatrix &U,
//                      const DenseMatrix &V,
//                      const Vector &b, Vector &x)
// {
//    std::vector<std::unique_ptr<Vector>> Uptr(U.Width());
//    std::vector<std::unique_ptr<Vector>> Vptr(V.Width());
//    for (int i=0; i<U.Width(); i++)
//    {
//       Uptr[i] = std::make_unique<Vector>(const_cast<real_t*>(U.GetColumn(i)),
//                                          U.Height());
//       Vptr[i] = std::make_unique<Vector>(const_cast<real_t*>(V.GetColumn(i)),
//                                          V.Height());
//    }
//    Woodbury(comm, Ainv, c, Uptr, Vptr, b, x);
// }
//
// template <typename T>
// inline void Woodbury(const Operator &Ainv, const real_t c,
//                      const std::vector<T> &Uptr,
//                      const std::vector<T> &Vptr,
//                      const Vector &b, Vector &x)
// {
//    // Convert pointers to raw pointers
//    std::vector<Vector*> V = ToRawPtrVector(Vptr);
//    std::vector<Vector*> U = ToRawPtrVector(Uptr);
//
//    MFEM_ASSERT(Ainv.Height() == Ainv.Width(),
//                "Ainv must be a square operator");
//    MFEM_ASSERT(V.size() == U.size(),
//                "V and U must have the same number of columns");
//    for (int i=0; i<V.size(); i++)
//    {
//       MFEM_ASSERT(V[i]->Size() == Ainv.Height(),
//                   "V columns must have the same size as Ainv height");
//       MFEM_ASSERT(U[i]->Size() == Ainv.Height(),
//                   "U columns must have the same size as Ainv height");
//    }
//    MFEM_ASSERT(b.Size() == Ainv.Height(),
//                "b must have the same size as Ainv height");
//
//    const int m = Ainv.Height();
//    const int k = V.size();
//    x.SetSize(b.Size());
//
//    // 1. Solve Ay = b
//    Vector y(m); // y = Ainv * b
//    Ainv.Mult(b, y);
//
//    // 2. Solve AZ = U
//    std::vector<Vector> Z;
//    for (int i=0; i<k; i++)
//    {
//       Z.emplace_back(m);
//       Ainv.Mult(*U[i], Z[i]);
//       Z[i] *= c;
//    }
//
//    // 3. Compute K = I + V^T * Z
//    DenseMatrix K; // K = I + V^T * Z
//    // 4. Solve K w = V^T * y
//    Vector Vt_y(k);
//    K.Diag(1.0, k);
//    for (int i=0; i<k; i++)
//    {
//       for (int j=0; j<k; j++)
//       {
//          K(i,j) += *V[i]*Z[j];
//       }
//       Vt_y[i] = *V[i]*y;
//    }
//
//    DenseMatrixInverse Kinv(K);
//    Vector w(k); // w = K^{-1} * V^T * y
//    Kinv.Factor();
//    Kinv.Mult(Vt_y, w);
//
//    // 5. Compute x = y - Z * w
//    x = y;
//    for (int i=0; i<k; i++)
//    {
//       x.Add(-w[i], Z[i]);
//    }
// }
// inline void Woodbury(const Operator &Ainv, const real_t c,
//                      const DenseMatrix &U,
//                      const DenseMatrix &V,
//                      const Vector &b, Vector &x)
// {
//    std::vector<std::unique_ptr<Vector>> Uptr(U.Width());
//    std::vector<std::unique_ptr<Vector>> Vptr(V.Width());
//    for (int i=0; i<U.Width(); i++)
//    {
//       Uptr[i] = std::make_unique<Vector>(const_cast<real_t*>(U.GetColumn(i)),
//                                          U.Height());
//       Vptr[i] = std::make_unique<Vector>(const_cast<real_t*>(V.GetColumn(i)),
//                                          V.Height());
//    }
//    Woodbury(Ainv, c, Uptr, Vptr, b, x);
// }


class MassOperator : public Operator
{
private:
   mutable Vector aux, aux2;
   std::unique_ptr<Operator> M;
   std::unique_ptr<Operator> M_inv;
   std::unique_ptr<Solver> M_prec;
#ifdef MFEM_USE_MPI
   Array<HYPRE_BigInt> cols;
   MPI_Comm comm=MPI_COMM_NULL;
#endif
public:
   MassOperator() = default;
   MassOperator(QuadratureSpace &qspace);
   MassOperator(FiniteElementSpace &fespace);

   // y = M*x
   void Mult(const Vector &x, Vector &y) const override
   {
      y.SetSize(height);
      M->Mult(x, y);
   }

   // y = M^{-1}*x
   virtual void Riesz(const Vector &x, Vector &y) const
   {
      y.SetSize(height);
      M_inv->Mult(x, y);
   }

   // M
   Operator &GetGradient(const Vector &x) const override
   { return *M; }

   // z = M*(x-y)
   virtual void MultDiff(const Vector &x, const Vector &y, Vector &z) const;

   // y^T*M*x
   virtual real_t InnerProduct(const Vector &x, const Vector &y) const;

   // ||(x-y)||_M^2 = sqrt((x-y)^T*M*(x-y))^2
   virtual real_t DistanceSquaredTo(const Vector &x, const Vector &y) const;

   // ||(x-y)||_M = sqrt((x-y)^T*M*(x-y))
   virtual real_t DistanceTo(const Vector &x, const Vector &y) const
   {
      return std::sqrt(DistanceSquaredTo(x, y));
   }
};

class MultiMassOperator : public MassOperator
{
private:
   Array<MassOperator*> mass;
   Array<int> offsets;
   mutable std::unique_ptr<BlockOperator> M;
public:
   MultiMassOperator(): MassOperator(), offsets{0} {}
   void Append(MassOperator &m);

   // y = M*x
   void Mult(const Vector &x, Vector &y) const override;

   // y = M^{-1}*x
   void Riesz(const Vector &x, Vector &y) const override;

   // M
   Operator &GetGradient(const Vector &x) const override;

   // z = M*(x-y)
   void MultDiff(const Vector &x, const Vector &y, Vector &z) const override;

   // y^T*M*x
   real_t InnerProduct(const Vector &x, const Vector &y) const override;

   // ||(x-y)||_M^2 = sqrt((x-y)^T*M*(x-y))^2
   real_t DistanceSquaredTo(const Vector &x, const Vector &y) const override;

   // ||(x-y)||_M = sqrt((x-y)^T*M*(x-y))
   real_t DistanceTo(const Vector &x, const Vector &y) const override
   {
      return std::sqrt(DistanceSquaredTo(x, y));
   }
};

class MultiL2RieszMap : public Operator
{
private:
   QuadratureSpace &qspace;
   std::vector<ParFiniteElementSpace*> fespace;
   Array<int> offsets;
   const Array<int> space_idx;
   const int num_vars;

   std::vector<std::unique_ptr<HypreParMatrix>> mass;
   std::vector<std::unique_ptr<HypreBoomerAMG>> mass_prec;
   std::vector<std::unique_ptr<Operator>> projector;
public:
   MultiL2RieszMap(QuadratureSpace &qspace,
                   std::vector<ParFiniteElementSpace*> fes,
                   const Array<int> space_idx);

   // From Dual to Primal (mass inverse)
   void Mult(const Vector &x, Vector &y) const override;
   // From Primal to Dual (mass)
   void MultTranspose(const Vector &x, Vector &y) const override;
   // return u^T M v
   real_t InnerProduct(const Vector &x, const Vector &y) const;
private:
   mutable std::vector<Vector> aux;
};

namespace remap
{
/// @brief A collection of conservative quantities that are considered in remap problems.

/// @brief int eta dx
inline real_t volume_f(const Vector &u) { return u[0]; }
inline void volume_df(const Vector &u, Vector &grad_u)
{
   grad_u.SetSize(u.Size()); grad_u = 0.0;
   grad_u[0] = 1.0;
}

/// @brief int eta * rho dx
inline real_t mass_f(const Vector &u) { return u[0]*u[1]; }
inline void mass_df(const Vector &u, Vector &grad_u)
{
   grad_u.SetSize(u.Size()); grad_u = 0.0;
   grad_u[0] = u[1];
   grad_u[1] = u[0];
}

/// @brief int eta * rho * e dx
inline real_t potential_f(const Vector &u) { return u[0]*u[1]*u[2]; }
inline void potential_df(const Vector &u, Vector &grad_u)
{
   grad_u.SetSize(u.Size()); grad_u = 0.0;
   grad_u[0] = u[1]*u[2];
   grad_u[1] = u[0]*u[2];
   grad_u[2] = u[0]*u[1];

}

/// @brief int eta * rho * e + 0.5 * eta * rho * |v|^2 dx
inline real_t energy_f(const Vector &u)
{
   MFEM_ASSERT(u.Size() > 3,
               "energy_f: Energy functional requires at least 4 components: [eta, rho, e, v1, v2, ...]");
   real_t energy = u[2];
   for (int i=3; i<u.Size(); i++) { energy += u[i]*u[i]*0.5; }
   return u[0]*u[1]*energy;
}
inline void energy_df(const Vector &u, Vector &grad_u)
{
   MFEM_ASSERT(u.Size() > 3,
               "energy_df: Energy functional requires at least 4 components: [eta, rho, e, v1, v2, ...]");
   grad_u.SetSize(u.Size()); grad_u = 0.0;
   real_t potential = u[2];
   real_t kinetic_energy = 0.0;
   for (int i=3; i<u.Size(); i++) { kinetic_energy += u[i]*u[i]*0.5; }
   real_t energy = potential + kinetic_energy;
   real_t mass = u[0]*u[1];
   grad_u[0] = u[1]*energy;
   grad_u[1] = u[0]*energy;
   grad_u[2] = u[0]*u[1];
   for (int i=3; i<u.Size(); i++)
   {
      grad_u[i] = mass*u[i];
   }
}

/// @brief int eta * rho * v[comp] dx
inline real_t momentum_f(const Vector &u, const int comp)
{
   MFEM_ASSERT(u.Size() > 3,
               "momentum_f: Momentum functional requires at least 4 components: [eta, rho, e, v1, v2, ...]");
   return u[0]*u[1]*u[3+comp];
}
inline void momentum_df(const Vector &u, Vector &grad_u, const int comp)
{
   MFEM_ASSERT(u.Size() > 3,
               "momentum_df: Momentum functional requires at least 4 components: [eta, rho, e, v1, v2, ...]");
   grad_u.SetSize(u.Size()); grad_u = 0.0;
   grad_u[0] = u[1]*u[3+comp];
   grad_u[1] = u[0]*u[3+comp];
   grad_u[3+comp] = u[0]*u[1];
}

void remap_functionals(const int optType, const int dim,
                       std::vector<std::function<real_t(const Vector &)>> &f,
                       std::vector<std::function<void(const Vector &, Vector &)>> &df,
                       Array<int> &space_idx);


/// @brief A functional that computes ||u - target||^2
/// where || || is the L2-norm.
/// Here, constraints are not considered.
/// GetGradient() is not the derivative, but the gradient of the functional.
/// that is, \nabla F = u - target
/// Riesz map can be applied to another derivatives using ApplyRieszMap()
class RemapObjectiveFunctional : public Functional
{
public:
   RemapObjectiveFunctional(QuadratureSpace &qspace,
                            const std::vector<FiniteElementSpace*> &fes,
                            const Vector &target,
                            const Array<int> &space_idx);
   RemapObjectiveFunctional(QuadratureSpace &qspace,
                            const std::vector<ParFiniteElementSpace*> &fes,
                            const Vector &target,
                            const Array<int> &space_idx);
   const Array<int> GetOffsets() const { return offsets; }
   QuadratureSpace &GetQuadratureSpace() const { return qspace; }
   const std::vector<FiniteElementSpace*> &GetFiniteElementSpaces() const { return fespace; }
#ifdef MFEM_USE_MPI
   const std::vector<ParFiniteElementSpace*> &GetParFiniteElementSpaces() const { return par_fespace; }
#endif

   const Array<int> &GetSpaceIdx() const { return space_idx; }

   // return ||u - target||^2 / 2 (in L2-norm)
   void Mult(const Vector &x, Vector &y) const override;

   // return M * (x - target)
   void EvalGradient(const Vector &x, Vector &y) const override;

   // return global block mass operator, M
   Operator &GetHessian(const Vector &x) const override;

private:
   QuadratureSpace &qspace;
   std::vector<FiniteElementSpace*> fespace;
#ifdef MFEM_USE_MPI
   std::vector<ParFiniteElementSpace*> par_fespace;
#endif
   const Vector &target;
   Array<int> space_idx;
   const int num_vars;
   std::vector<std::unique_ptr<MassOperator>> mass;
   Array<int> offsets;
   mutable std::unique_ptr<BlockOperator> hessian;

   void Initialize();
};

class RemapProblem : public ConstrainedOptimizationProblem
{
public:
   RemapProblem(remap::RemapObjectiveFunctional &objective,
                const BlockVector &x_min,
                const BlockVector &x_max,
                StackedSharedFunctional &C)
      : ConstrainedOptimizationProblem(objective, &C)
      , x_min(x_min), x_max(x_max)
   {}
   void Mult(const Vector &x, Vector &y) const override
   {
      objective.Mult(x, y);
      /// Do something with the constraints
   }
   void EvalGradient(const Vector &x, Vector &grad) const override
   {
      objective.EvalGradient(x, grad);
      /// Do something with the constraints
   }

   // This will not be used because GetHessian() is overridden
   // To use different Hessian, override GetHessian() back to Functional::GetHessian()
   void HessianMult(const Vector &x, const Vector &d, Vector &Hd) const override
   {
      Hd.SetSize(x.Size());
      objective.GetHessian(x).Mult(d, Hd);
      /// Do something with the constraints
   }
   const Vector &GetLowerBounds() const { return x_min; }
   const Vector &GetUpperBounds() const { return x_max; }
   QuadratureSpace &GetQuadratureSpace() const { return static_cast<RemapObjectiveFunctional&>(objective).GetQuadratureSpace(); }
   std::vector<ParFiniteElementSpace*> GetFiniteElementSpaces() const
   {
      return static_cast<RemapObjectiveFunctional&>
             (objective).GetParFiniteElementSpaces();
   }
   Array<int> GetSpaceIdx() const { return static_cast<RemapObjectiveFunctional&>(objective).GetSpaceIdx(); }
   // Compute KKT residual with grad = grad objective + <lambda, grad C>
   real_t ComputeKKT(const BlockVector &x,
                     const BlockVector &grad) const
   {
      real_t kkt = kkt_res(x, x_min, x_max, grad);
      MPI_Allreduce(MPI_IN_PLACE, &kkt, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                    GetComm());
      HYPRE_BigInt n = x.Size();
      MPI_Allreduce(MPI_IN_PLACE, &n, 1, MPITypeMap<HYPRE_BigInt>::mpi_type, MPI_SUM,
                    GetComm());
      return kkt / n;
   }

protected:
   const BlockVector &x_min; // lower bounds
   const BlockVector &x_max; // upper bounds
};

} // namespace remap

class MappedGridFunctionCoefficient : public GridFunctionCoefficient
{
protected:
   std::function<real_t(const real_t x, ElementTransformation &T, const IntegrationPoint &ip)>
   F;
public:
   MappedGridFunctionCoefficient(GridFunction &gf,
                                 std::function<real_t(const real_t x, ElementTransformation &T, const IntegrationPoint &ip)>
                                 F)
      : GridFunctionCoefficient(&gf), F(F) {}
   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      return F(GridFunctionCoefficient::Eval(T, ip), T, ip);
   }
};

} // namespace mfem

#endif // MFEM_REMAP_HPP
