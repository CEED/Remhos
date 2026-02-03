#include "remap.hpp"

namespace mfem
{


QuadratureDomainLFIntegrator::QuadratureDomainLFIntegrator(
   const QuadratureFunction &qf,
   FiniteElementSpace &fes)
   : qf(qf)
   , qspace(dynamic_cast<const QuadratureSpace*>(qf.GetSpace()))
   , fespace(&fes)
{
   MFEM_VERIFY(fes.FEColl()->GetMapType(fes.GetMesh()->Dimension())
               == FiniteElement::MapType::VALUE,
               "QuadratureLinearForm only supports finite element space with value maps.");
   MFEM_VERIFY(fes.GetVDim() == 1,
               "QuadratureLinearForm only supports scalar finite element spaces.");
   MFEM_VERIFY(qspace != nullptr,
               "QuadratureFunction must be defined on a QuadratureSpace.");
   dof2q.resize(fes.GetMaxElementOrder()+1);
   for (int elem=0; elem<fes.GetNE(); elem++)
   {
      const FiniteElement &fe = *fespace->GetFE(elem);
      const IntegrationRule &ir = qspace->GetIntRule(elem);
      int fe_order = fe.GetOrder();
      int ir_order = ir.GetOrder();
      if (ir_order >= dof2q[fe_order].size())
      {
         dof2q[fe_order].resize(ir_order+1);
      }
      if (dof2q[fe_order][ir_order]) { continue; }

      // Create the matrix that maps quadrature points to element dofs
      const int dof = fe.GetDof();
      const int nq = ir.GetNPoints();
      dof2q[fe_order][ir_order]  = std::make_unique<DenseMatrix>(dof, nq);
      DenseMatrix &Q2E = *dof2q[fe_order][ir_order];
      Vector shape;
      for (int i=0; i<nq; i++)
      {
         Q2E.GetColumnReference(i, shape);
         fe.CalcShape(ir.IntPoint(i), shape);
      }
   }
   this->qf *= qspace->GetWeights();
}

void QuadratureDomainLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el,
   ElementTransformation &Tr,
   Vector &elvect)
{
   const int i = Tr.ElementNo;
   const int fe_order = el.GetOrder();
   const int dof = el.GetDof();
   elvect.SetSize(dof);

   const IntegrationRule &ir = qspace->GetIntRule(i);

   const int ir_order = ir.GetOrder();
   qf.GetValues(i, qvals);

   const DenseMatrix &Q2E = *dof2q[fe_order][ir_order];
   Q2E.Mult(qvals, elvect);
}

// QuadratureFunction to dual vector, <qf, v> = \int qf v dx
// The returned vector is a T-vector in dual space.
QuadratureLinearForm::QuadratureLinearForm(QuadratureSpace &qs,
      FiniteElementSpace &fes)
   : Operator(fes.GetTrueVSize(), qs.GetSize())
   , qspace(qs)
   , fespace(fes)
   , parallel(false)
{
   MFEM_VERIFY(fes.FEColl()->GetMapType(fes.GetMesh()->Dimension())
               == FiniteElement::MapType::VALUE,
               "QuadratureLinearForm only supports finite element space with value maps.");
   MFEM_VERIFY(fes.GetVDim() == 1,
               "QuadratureLinearForm only supports scalar finite element spaces.");
   dof2q.resize(fes.GetMaxElementOrder()+1);
   for (int elem=0; elem<fes.GetNE(); elem++)
   {
      const FiniteElement &fe = *fespace.GetFE(elem);
      const IntegrationRule &ir = qspace.GetIntRule(elem);
      int fe_order = fe.GetOrder();
      int ir_order = ir.GetOrder();
      if (ir_order >= dof2q[fe_order].size())
      {
         dof2q[fe_order].resize(ir_order+1);
      }
      if (dof2q[fe_order][ir_order]) { continue; }

      // Create the matrix that maps quadrature points to element dofs
      const int dof = fe.GetDof();
      const int nq = ir.GetNPoints();
      dof2q[fe_order][ir_order]  = std::make_unique<DenseMatrix>(dof, nq);
      DenseMatrix &Q2E = *dof2q[fe_order][ir_order];
      Vector shape;
      for (int i=0; i<nq; i++)
      {
         Q2E.GetColumnReference(i, shape);
         fe.CalcShape(ir.IntPoint(i), shape);
      }
   }
#ifdef MFEM_USE_MPI
   ParFiniteElementSpace *pfes =
      dynamic_cast<ParFiniteElementSpace*>(&fes);
   if (pfes)
   {
      parallel = true;
      L_vec.SetSize(fes.GetVSize());
   }
#endif
}

void QuadratureLinearForm::Mult(const Vector &x, Vector &y) const
{
   y.SetSize(height);
   if (!parallel) { y = 0.0; L_vec.MakeRef(y, 0); }
   else { L_vec = 0.0; }

   Q_vec = x;
   Q_vec *= qspace.GetWeights();
   QuadratureFunction qf(&qspace, Q_vec.GetData());
   Array<int> vdofs;
   Vector elvect, qvals;
   for (int i=0; i<fespace.GetNE(); i++)
   {
      const FiniteElement &fe = *fespace.GetFE(i);
      fespace.GetElementVDofs(i, vdofs);
      const int fe_order = fe.GetOrder();
      const int dof = fe.GetDof();
      elvect.SetSize(dof);

      const IntegrationRule &ir = qspace.GetIntRule(i);
      const int ir_order = ir.GetOrder();
      qf.GetValues(i, qvals);

      const DenseMatrix &Q2E = *dof2q[fe_order][ir_order];
      Q2E.Mult(qvals, elvect);
      L_vec.AddElementVector(vdofs, elvect);
   }
   if (parallel)
   {
      const Operator* prolong = fespace.GetProlongationMatrix();
      prolong->MultTranspose(L_vec, y);
   }
}

ComposedFunctional::ComposedFunctional(FuncType f,
                                       GradType g,
                                       QuadratureSpace &qspace,
                                       std::vector<FiniteElementSpace*> fes,
                                       const Array<int> space_idx)
   : SharedFunctional(0)
   , f(f)
   , df(g)
   , space_idx(space_idx)
   , num_vars(space_idx.Size())
   , qspace(qspace)
   , fespace(fes)
   , gfs(fespace.size())
   , qlf(fespace.size())
   , qf(qspace)
   , qf_in(qspace, space_idx.Size())
   , qf_out(qspace, space_idx.Size())
{
   MFEM_VERIFY(space_idx.Max() < (int)fespace.size() && space_idx.Min() >= -1,
               "CompsedFunctional: Space index out of range.");
   Initialize();
}

/// Evaluate the derivative of <f, v> = \int f(u1, u2, ..., un) dx
/// That is, y = [int f_i(u1, ..., un) * v1 dx, ..., int f_n(u1, ..., un) * vn dx]
/// where f_i is the derivative of f with respect to u_i.
void ComposedFunctional::EvalGradientCurrent(Vector &y) const
{
   y.SetSize(width);
   BlockVector y_block(y, offsets);

   Vector qf_in_view((real_t*)nullptr, num_vars);
   Vector qf_out_view((real_t*)nullptr, num_vars);

   // quadrature point evaluation
   for (int qid=0; qid<qspace.GetSize(); qid++)
   {
      qf_in_view.MakeRef(qf_in, qid*num_vars, num_vars);
      qf_out_view.MakeRef(qf_out, qid*num_vars, num_vars);
      df(qf_in_view, qf_out_view);
   }

   // integration
   const Vector &w = qspace.GetWeights();
   real_t *qf_owned_data;
   qf.StealData(&qf_owned_data);
   for (int vid=0; vid<num_vars; vid++)
   {
      const int sid = space_idx[vid];
      if (sid < 0) // QuadratureFunction
      {
         // Set output as a quadarture function
         qf.MakeRef(y_block.GetBlock(vid), 0, qspace.GetSize());
         // Copy ith component to output
         VecQF2QF(qf_out, vid, qf);
         // Scale by weights
         y_block.GetBlock(vid) *= w;
      }
      else // FiniteElementSpace
      {
         // Set qf data to original data
         qf.SetDataAndSize(qf_owned_data, qspace.GetSize());
         // Copy ith component to quadrature function
         VecQF2QF(qf_out, vid, qf);
         // Integrate using QuadratureLinearForm
         // No need to handle parallel case
         qlf[sid]->Mult(qf, y_block.GetBlock(vid));
      }
   }
   // Restore qf data
   qf.SetDataAndSize(qf_owned_data, qspace.GetSize());
   qf.MakeDataOwner();
}

void ComposedFunctional::MultCurrent(Vector &y) const
{
   y.SetSize(1);
   real_t &y0 = *y.GetData();
   y0 = 0.0;

   // view of [u1, ..., un] at given quadrature point
   Vector qf_in_view((real_t*)nullptr, num_vars);
   real_t *qf_in_data = qf_in.GetData();

   // integration
   const Vector &w = qspace.GetWeights();
   for (int i=0; i<qspace.GetSize(); i++) // for all quadrature points
   {
      // Update viewpoint
      qf_in_view.SetData(qf_in_data + i*num_vars);
      // Evaluate and accumulate with weight
      y0 += f(qf_in_view)*w[i];
   }
   if (IsParallel()) // if parallel, reduce the result
   {
#ifdef MFEM_USE_MPI
      MPI_Allreduce(MPI_IN_PLACE, &y0, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                    GetComm());
#endif
   }
   y0 -= target;
}
void ComposedFunctional::Initialize()
{

   is_input_frozen = false;
   target = 0.0;
#ifdef MFEM_USE_MPI
   // Check FESpaces are parallel or serial
   par_fespace.resize(0);
   for (auto &fes : fespace)
   {
      par_fespace.push_back(dynamic_cast<ParFiniteElementSpace*>(fes));
   }
   if (par_fespace.size() > 0 && par_fespace[0] != nullptr)
   {
      SetComm(par_fespace[0]->GetComm());
   }
#endif

   // for each FESpace, create a GridFunction and QuadratureLinearForm
   for (int sid=0; sid<fespace.size(); sid++)
   {
      MFEM_VERIFY(fespace[sid] != nullptr,
                  "CompsedFunctional::Initialize(): FiniteElementSpace pointer is null at index "
                  << sid);
      if (IsParallel())
      {
#ifdef MFEM_USE_MPI
         MFEM_VERIFY(par_fespace[sid] != nullptr,
                     "CompsedFunctional::Initialize(): ParFiniteElementSpace pointer is null at index "
                     << sid);
         gfs[sid] = std::make_unique<ParGridFunction>(par_fespace[sid]);
         qlf[sid] = std::make_unique<QuadratureLinearForm>(qspace, *par_fespace[sid]);
#endif
      }
      else
      {
         gfs[sid] = std::make_unique<GridFunction>(fespace[sid]);
         qlf[sid] = std::make_unique<QuadratureLinearForm>(qspace, *fespace[sid]);
      }
   }

   // Count degrees of freedom for each variable
   offsets.SetSize(0);
   offsets.Append(0);
   for (int vid=0; vid<num_vars; vid++)
   {
      const int sid = space_idx[vid];
      if (sid < 0)
      {
         offsets.Append(qspace.GetSize());
      }
      else
      {
         MFEM_VERIFY(sid < fespace.size(),
                     "CompsedFunctional::Initialize(): FiniteElementSpace index out of range.");
         offsets.Append(fespace[sid]->GetTrueVSize());
      }
   }
   offsets.PartialSum();
   width = offsets.Last();
}

// convert evaluation point x to quadrature functions and store in qf_in
void ComposedFunctional::ProcessX(const Vector &x) const
{
   // make a view of x with offsets
   // x_block is not modifiable!
   BlockVector x_block(const_cast<Vector&>(x), offsets);
   // store qf's data
   real_t *qf_owned_data;
   qf.StealData(&qf_owned_data);

   // Project if needed, and copy to qf_in
   for (int vid=0; vid<num_vars; vid++)
   {
      const int sid = space_idx[vid];
      if (sid < 0) // QuadratureFunction
      {
         // no need to project, just make a reference
         qf.SetDataAndSize(x_block.GetBlock(vid).GetData(), qspace.GetSize());
      }
      else // FiniteElementSpace
      {
         // T-Vector to L-Vector
         gfs[sid]->MakeTRef(fespace[sid], x_block.GetBlock(vid).GetData());
         gfs[sid]->SetFromTrueVector();
         // L-Vector to Q-Vector
         qf.SetDataAndSize(qf_owned_data, qspace.GetSize());
         qf.ProjectGridFunction(*gfs[sid]);
      }
      // Copy to qf_in. qf_in[i*num_vars + j] = qf[j]
      QF2VecQF(qf, vid, qf_in);
   }
   // restore the data
   qf.SetDataAndSize(qf_owned_data, qspace.GetSize());
   qf.MakeDataOwner();
}

void ComposedFunctional::ShallowCopyProcessedX(SharedFunctional &owner)
{
   ComposedFunctional * owner_comp = dynamic_cast<ComposedFunctional*>(&owner);
   if (owner_comp)
   {
      qf_in.MakeRef(owner_comp->qf_in, 0);
      return;
   }
   SharedFunctional::ShallowCopyProcessedX(owner); // this will abort
}


MassOperator::MassOperator(QuadratureSpace &qspace)
   : Operator(qspace.GetSize())
{
   M = std::make_unique<SparseMatrix>(qspace.GetWeights());
   static_cast<SparseMatrix&>(*M).Finalize();
   Vector w_inv = qspace.GetWeights();
   w_inv.Reciprocal();
   M_inv = std::make_unique<SparseMatrix>(w_inv);
   static_cast<SparseMatrix&>(*M_inv).Finalize();
#ifdef MFEM_USE_MPI
   ParMesh *pm = dynamic_cast<ParMesh*>(qspace.GetMesh());
   if (pm)
   {
      comm = pm->GetComm();

      std::unique_ptr<Operator> M_ser(std::move(M));
      std::unique_ptr<Operator> M_inv_ser(std::move(M_inv));
      Array<HYPRE_BigInt> * glb_cols(&cols);

      HYPRE_BigInt n = qspace.GetSize();
      pm->GenerateOffsets(1, &n, &glb_cols);
      int * Jloc = static_cast<SparseMatrix&>(*M_ser).GetJ();
      Array<HYPRE_BigInt> Jglb(M_ser->Width());
      for (int i=0; i<n; i++) { Jglb[i] = Jloc[i] + cols[0]; }

      M = std::make_unique<HypreParMatrix>(
             comm, n, cols.Last(), cols.Last(),
             static_cast<SparseMatrix&>(*M_ser).GetI(),
             Jglb.GetData(),
             static_cast<SparseMatrix&>(*M_ser).GetData(),
             cols.GetData(), cols.GetData()); // constructor with 9 arguments
      M_inv = std::make_unique<HypreParMatrix>(
                 comm, n, cols.Last(), cols.Last(),
                 static_cast<SparseMatrix&>(*M_inv_ser).GetI(),
                 Jglb.GetData(),
                 static_cast<SparseMatrix&>(*M_inv_ser).GetData(),
                 cols.GetData(), cols.GetData()); // constructor with 9 arguments
   }
#endif
}

MassOperator::MassOperator(FiniteElementSpace &fespace)
   : Operator(fespace.GetTrueVSize())
{
   std::unique_ptr<BilinearForm> mass;
   std::unique_ptr<BilinearForm> inv_mass;
   BilinearFormIntegrator *mass_intg = nullptr;
   BilinearFormIntegrator *inv_mass_intg = nullptr;
   if (fespace.GetVDim() == 1)
   {
      mass_intg = new MassIntegrator();
      if (fespace.IsDGSpace())
      {
         inv_mass_intg = new InverseIntegrator(new MassIntegrator());
      }
   }
   else if (fespace.FEColl()->GetMapType(fespace.GetMesh()->Dimension()) ==
            FiniteElement::MapType::VALUE)
   {
      mass_intg = new VectorMassIntegrator();
      if (fespace.IsDGSpace())
      {
         inv_mass_intg = new InverseIntegrator(new VectorMassIntegrator());
      }
   }
   else
   {
      mass_intg = new VectorFEMassIntegrator();
      if (fespace.IsDGSpace())
      {
         inv_mass_intg = new InverseIntegrator(new VectorFEMassIntegrator());
      }
   }
#ifdef MFEM_USE_MPI
   ParFiniteElementSpace *pfes =
      dynamic_cast<ParFiniteElementSpace*>(&fespace);
   if (pfes)
   {
      comm = pfes->GetComm();
      mass = std::make_unique<ParBilinearForm>(pfes);
      mass->AddDomainIntegrator(mass_intg);
      mass->Assemble();
      mass->Finalize();
      M.reset(static_cast<ParBilinearForm&>(*mass).ParallelAssemble());
      if (pfes->IsDGSpace())
      {
         inv_mass = std::make_unique<ParBilinearForm>(pfes);
         inv_mass->AddDomainIntegrator(inv_mass_intg);
         inv_mass->Assemble();
         inv_mass->Finalize();
         M_inv.reset(static_cast<ParBilinearForm&>(*inv_mass).ParallelAssemble());
      }
      else
      {
         M_prec = std::make_unique<HypreBoomerAMG>(static_cast<HypreParMatrix&>(*M));
         static_cast<HypreBoomerAMG&>(*M_prec).SetPrintLevel(0);

         M_inv = std::make_unique<HyprePCG>(comm);
         static_cast<HyprePCG&>(*M_inv).SetPrintLevel(0);
         static_cast<HyprePCG&>(*M_inv).SetPreconditioner(
            static_cast<HypreBoomerAMG&>(*M_prec));
         static_cast<HyprePCG&>(*M_inv).SetOperator(
            static_cast<HypreParMatrix&>(*M));
         static_cast<HyprePCG&>(*M_inv).SetAbsTol(1e-10);
         static_cast<HyprePCG&>(*M_inv).SetMaxIter(1e06);
         static_cast<HyprePCG&>(*M_inv).iterative_mode = true;
      }
   }
#endif
   if (!mass) // either serial build or non-parallel space
   {
      mass = std::make_unique<BilinearForm>(&fespace);
      mass->AddDomainIntegrator(mass_intg);
      mass->Assemble();
      mass->Finalize();
      M.reset(mass->LoseMat());
      if (fespace.IsDGSpace())
      {
         inv_mass = std::make_unique<BilinearForm>(&fespace);
         inv_mass->AddDomainIntegrator(inv_mass_intg);
         inv_mass->Assemble();
         inv_mass->Finalize();
         M_inv.reset(inv_mass->LoseMat());
      }
      else
      {
         M_prec = std::make_unique<GSSmoother>();
         M_inv = std::make_unique<CGSolver>();
         static_cast<CGSolver&>(*M_inv).SetPrintLevel(0);
         static_cast<CGSolver&>(*M_inv).SetPreconditioner(static_cast<GSSmoother&>
               (*M_prec));
         static_cast<CGSolver&>(*M_inv).SetOperator(static_cast<const SparseMatrix&>
               (*M));
         static_cast<CGSolver&>(*M_inv).SetAbsTol(1e-10);
         static_cast<CGSolver&>(*M_inv).SetRelTol(1e-10);
         static_cast<CGSolver&>(*M_inv).SetMaxIter(1e06);
      }
   }


}

// z = M*(x-y)
void MassOperator::MultDiff(const Vector &x, const Vector &y, Vector &z) const
{
   MPI_Barrier(comm);
   aux = x;
   aux -= y;
   z = y;
   z.SetSize(height);
   M->Mult(aux, z);
}
// y^T*M*x
real_t MassOperator::InnerProduct(const Vector &x, const Vector &y) const
{
   aux.SetSize(height);
   M->Mult(x, aux);
   real_t result = aux*y;
#ifdef MFEM_USE_MPI
   if (comm != MPI_COMM_NULL)
   {
      MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_SUM, comm);
   }
#endif
   return result;
}

// ||(x-y)||_M^2 = sqrt((x-y)^T*M*(x-y))^2
real_t MassOperator::DistanceSquaredTo(const Vector &x, const Vector &y) const
{
   MultDiff(x, y, aux2);
   real_t result = aux2*aux;
#ifdef MFEM_USE_MPI
   if (comm != MPI_COMM_NULL)
   {
      MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_SUM, comm);
   }
#endif
   return result;
}
void MultiMassOperator::Append(MassOperator &m)
{
   mass.Append(&m);
   offsets.Append(offsets.Last() + m.Height());
   height += m.Height();
   width += m.Height();
}
// y = M*x
void MultiMassOperator::Mult(const Vector &x, Vector &y) const
{
   const BlockVector x_block(x.GetData(), offsets);
   y.SetSize(height);
   BlockVector y_block(y.GetData(), offsets);
   for (int i=0; i<mass.Size(); i++)
   {
      mass[i]->Mult(x_block.GetBlock(i), y_block.GetBlock(i));
   }
}
// y = M^{-1}*x
void MultiMassOperator::Riesz(const Vector &x, Vector &y) const
{
   const BlockVector x_block(x.GetData(), offsets);
   y.SetSize(height);
   BlockVector y_block(y.GetData(), offsets);
   for (int i=0; i<mass.Size(); i++)
   {
      mass[i]->Riesz(x_block.GetBlock(i), y_block.GetBlock(i));
   }
}
// M
Operator &MultiMassOperator::GetGradient(const Vector &x) const
{
   M = std::make_unique<BlockOperator>(offsets);
   for (int i=0; i<mass.Size(); i++)
   {
      M->SetBlock(i, i, mass[i]);
   }
   return *M;
}
// z = M*(x-y)
void MultiMassOperator::MultDiff(const Vector &x, const Vector &y,
                                 Vector &z) const
{
   const BlockVector x_block(x.GetData(), offsets);
   const BlockVector y_block(y.GetData(), offsets);
   z.SetSize(height);
   BlockVector z_block(z.GetData(), offsets);
   for (int i=0; i<mass.Size(); i++)
   {
      mass[i]->MultDiff(x_block.GetBlock(i), y_block.GetBlock(i),
                        z_block.GetBlock(i));
   }
}
// y^T*M*x
real_t MultiMassOperator::InnerProduct(const Vector &x, const Vector &y) const
{
   const BlockVector x_block(x.GetData(), offsets);
   const BlockVector y_block(y.GetData(), offsets);
   real_t result = 0.0;
   for (int i=0; i<mass.Size(); i++)
   {
      result += mass[i]->InnerProduct(x_block.GetBlock(i),
                                      y_block.GetBlock(i));
   }
   return result;
}
// ||(x-y)||_M^2 = sqrt((x-y)^T*M*(x-y))^2
real_t MultiMassOperator::DistanceSquaredTo(const Vector &x,
      const Vector &y) const
{
   const BlockVector x_block(x.GetData(), offsets);
   const BlockVector y_block(y.GetData(), offsets);
   real_t result = 0.0;
   for (int i=0; i<mass.Size(); i++)
   {
      result += mass[i]->DistanceSquaredTo(x_block.GetBlock(i),
                                           y_block.GetBlock(i));
   }
   return result;
}
// ||(x-y)||_M = sqrt((x-y)^T*M*(x-y))

MultiL2RieszMap::MultiL2RieszMap(QuadratureSpace &qspace,
                                 std::vector<ParFiniteElementSpace*> fes,
                                 const Array<int> space_idx)
   : qspace(qspace)
   , fespace(fes)
   , space_idx(space_idx)
   , num_vars(space_idx.Size())
   , mass(fes.size())
   , mass_prec(fes.size())
   , projector(fes.size())
{
   for (int i=0; i<fespace.size(); i++)
   {
      ParBilinearForm curr_mass(fespace[i]);
      curr_mass.AddDomainIntegrator(new MassIntegrator());
      curr_mass.Assemble();
      curr_mass.Finalize();
      mass[i].reset(curr_mass.ParallelAssemble());
      if (dynamic_cast<const L2_FECollection*>(fespace[i]->FEColl()))
      {
         ParBilinearForm curr_mass_inv(fespace[i]);
         curr_mass_inv.AddDomainIntegrator(new InverseIntegrator(new MassIntegrator()));
         curr_mass_inv.Assemble();
         curr_mass_inv.Finalize();
         projector[i].reset(curr_mass_inv.ParallelAssemble());
      }
      else if (fespace[i]->FEColl()->GetMapType(fes[i]->GetMesh()->Dimension()) ==
               FiniteElement::VALUE)
      {
         mass_prec[i] = std::make_unique<HypreBoomerAMG>(static_cast<HypreParMatrix&>
                        (*mass[i]));
         projector[i] = std::make_unique<HyprePCG>(fespace[i]->GetComm());
         HyprePCG *solver = static_cast<HyprePCG*>(projector[i].get());
         mass_prec[i]->SetPrintLevel(0);
         solver->SetAbsTol(1e-10);
         solver->SetMaxIter(1e06);
         solver->SetPrintLevel(0);
         solver->SetPreconditioner(*mass_prec[i]);
         solver->SetOperator(static_cast<const HypreParMatrix&>(*mass[i]));
      }
      else
      {
         MFEM_ABORT("MultiL2Projector: Only L2 or H1 spaces are supported.");
      }
   }
   offsets.SetSize(0);
   offsets.Append(0);
   for (auto s : space_idx)
   {
      s == -1 ? offsets.Append(qspace.GetSize())
      : offsets.Append(fespace[s]->GetTrueVSize());
   }
   offsets.PartialSum();
   width = height = offsets.Last();
}

// From Dual to Primal (mass inverse)
void MultiL2RieszMap::Mult(const Vector &x, Vector &y) const
{
   y.SetSize(height);
   BlockVector x_block(const_cast<Vector&>(x), offsets);
   BlockVector y_block(y, offsets);

   for (int i=0; i<num_vars; i++)
   {
      const int sid = space_idx[i];
      if (sid < 0) // QuadratureFunction
      {
         // just copy the data
         y_block.GetBlock(i) = x_block.GetBlock(i);
         y_block.GetBlock(i) /= qspace.GetWeights();
      }
      else // FiniteElementSpace
      {
         projector[sid]->Mult(x_block.GetBlock(i), y_block.GetBlock(i));
      }
   }
}
// From Primal to Dual (mass)
void MultiL2RieszMap::MultTranspose(const Vector &x, Vector &y) const
{
   y.SetSize(height);
   BlockVector x_block(const_cast<Vector&>(x), offsets);
   BlockVector y_block(y, offsets);

   for (int i=0; i<num_vars; i++)
   {
      const int sid = space_idx[i];
      if (sid < 0) // QuadratureFunction
      {
         // just copy the data
         y_block.GetBlock(i) = x_block.GetBlock(i);
         y_block.GetBlock(i) *= qspace.GetWeights();
      }
      else // FiniteElementSpace
      {
         mass[sid]->Mult(x_block.GetBlock(i), y_block.GetBlock(i));
      }
   }
}
// return u^T M v
real_t MultiL2RieszMap::InnerProduct(const Vector &x, const Vector &y) const
{
   BlockVector x_block(const_cast<Vector&>(x), offsets);
   BlockVector y_block(const_cast<Vector&>(y), offsets);
   real_t result = 0.0;
   aux.resize(fespace.size() + 1);

   for (int i=0; i<num_vars; i++)
   {
      const int sid = space_idx[i];
      if (sid < 0) // QuadratureFunction
      {
         aux.back() = x_block.GetBlock(i);
         aux.back() *= qspace.GetWeights();
         result += aux.back() * y_block.GetBlock(i);
      }
      else // FiniteElementSpace
      {
         aux[sid].SetSize(fespace[sid]->GetTrueVSize());
         mass[sid]->Mult(x_block.GetBlock(i), aux[sid]);
         result += aux[sid]*y_block.GetBlock(i);
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &result, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);
   return result;
}

namespace remap
{
/// @brief A collection of conservative quantities that are considered in remap problems.


void remap_functionals(const int optType, const int dim,
                       std::vector<std::function<real_t(const Vector &)>> &f,
                       std::vector<std::function<void(const Vector &, Vector &)>> &df,
                       Array<int> &space_idx)
{
   f.resize(0);
   df.resize(0);
   space_idx.SetSize(0);

   f.push_back(volume_f);
   df.push_back(volume_df);
   space_idx.Append(-1);
   if (optType == 0) { return; }

   f.push_back(mass_f);
   df.push_back(mass_df);
   space_idx.Append(-1);
   if (optType == 1) { return; }

   if (optType == 2)
   {
      f.push_back(potential_f);
      df.push_back(potential_df);
      space_idx.Append(0); // L2
      return;
   }
   f.push_back(energy_f);
   df.push_back(energy_df);
   space_idx.Append(0); // L2
   for (int i=0; i<dim; i++)
   {
      f.push_back([i](const Vector &u) { return momentum_f(u, i); });
      df.push_back([i](const Vector &u, Vector &g) { momentum_df(u, g, i); });
      space_idx.Append(1); // H1
   }
}


/// @brief A functional that computes ||u - target||^2
/// where || || is the L2-norm.
/// Here, constraints are not considered.
/// GetGradient() is not the derivative, but the gradient of the functional.
/// that is, \nabla F = u - target
/// Riesz map can be applied to another derivatives using ApplyRieszMap()
RemapObjectiveFunctional::RemapObjectiveFunctional(QuadratureSpace &qspace,
      const std::vector<FiniteElementSpace*> &fes,
      const Vector &target,
      const Array<int> &space_idx)
   : Functional(target.Size())
   , qspace(qspace)
   , fespace(fes)
   , target(target)
   , space_idx(space_idx)
   , num_vars(space_idx.Size())
{
   Initialize();
}
RemapObjectiveFunctional::RemapObjectiveFunctional(QuadratureSpace &qspace,
      const std::vector<ParFiniteElementSpace*> &fes,
      const Vector &target,
      const Array<int> &space_idx)
   : Functional(target.Size())
   , qspace(qspace)
   , target(target)
   , space_idx(space_idx)
   , num_vars(space_idx.Size())
{
   fespace.resize(fes.size());
   for (int i=0; i<fes.size(); i++) { fespace[i] = fes[i]; }
   ParMesh *pm = dynamic_cast<ParMesh*>(qspace.GetMesh());
   if (pm)
   {
      SetComm(pm->GetComm());
   }
   Initialize();
}

// return ||u - target||^2 / 2 (in L2-norm)
void RemapObjectiveFunctional::Mult(const Vector &x, Vector &y) const
{
   y.SetSize(1);
   y[0] = 0.0;
   const BlockVector x_block(const_cast<Vector&>(x), offsets);
   const BlockVector target_block(const_cast<Vector&>(target), offsets);
   for (int i=0; i<num_vars; i++)
   {
      // wrap space index, so that qspace(==-1) get the last
      const int sid = (space_idx[i] + mass.size()) % mass.size();
      y[0] += mass[sid]->DistanceSquaredTo(x_block.GetBlock(i),
                                           target_block.GetBlock(i));
   }
   y[0] *= 0.5;
}

// return M * (x - target)
void RemapObjectiveFunctional::EvalGradient(const Vector &x,
      Vector &y) const
{
   MFEM_VERIFY(x.Size() == width && offsets.Last() == width &&
               target.Size() == width,
               "RemapObjectiveFunctional::EvalGradient(): y size does not match target size.");

   y.SetSize(width);
   const BlockVector x_block(const_cast<Vector&>(x), offsets);
   const BlockVector target_block(const_cast<Vector&>(target), offsets);
   BlockVector y_block(y, offsets);
   for (int i=0; i<num_vars; i++)
   {
      // wrap space index, so that qspace(==-1) get the last
      const int sid = (space_idx[i] + mass.size()) % mass.size();
      mass[sid]->MultDiff(x_block.GetBlock(i), target_block.GetBlock(i),
                          y_block.GetBlock(i));
   }
}

// return global block mass operator, M
Operator &RemapObjectiveFunctional::GetHessian(const Vector &x) const
{
   if (hessian) { return *hessian; }
   hessian = std::make_unique<BlockOperator>(offsets);
   for (int i=0; i<num_vars; i++)
   {
      const int sid = space_idx[i];
      hessian->SetBlock(i, i, mass[(sid+mass.size()) % mass.size()].get());
   }
   hessian->owns_blocks = false;
   return *hessian;
}

void RemapObjectiveFunctional::Initialize()
{
   // Count degrees of freedom for each variable
   offsets.SetSize(0);
   offsets.Append(0);
   for (int vid=0; vid<num_vars; vid++)
   {
      const int sid = space_idx[vid];
      if (sid < 0)
      {
         offsets.Append(qspace.GetSize());
      }
      else
      {
         MFEM_VERIFY(sid < fespace.size(),
                     "RemapObjective::Initialize(): FiniteElementSpace index out of range.");
         offsets.Append(fespace[sid]->GetTrueVSize());
      }
   }
   offsets.PartialSum();
   width = offsets.Last();
   MFEM_VERIFY(width == target.Size(),
               "RemapObjective::Initialize(): Target vector size does not match the functional size.");

#ifdef MFEM_USE_MPI
   par_fespace.resize(fespace.size());
#endif
   mass.resize(fespace.size() + 1);
   for (int sid=0; sid<fespace.size(); sid++)
   {
      MFEM_VERIFY(fespace[sid] != nullptr,
                  "RemapObjective::Initialize(): FiniteElementSpace pointer is null at index "
                  << sid);
      MFEM_VERIFY(fespace[sid]->GetVDim() == 1,
                  "RemapObjective::Initialize(): FiniteElementSpace must be a scalar FESpace");
#ifdef MFEM_USE_MPI
      par_fespace[sid] = dynamic_cast<ParFiniteElementSpace*>(fespace[sid]);
#endif
      mass[sid] = std::make_unique<MassOperator>(*fespace[sid]);
   }
   mass.back() = std::make_unique<MassOperator>(qspace);
}


} // namespace remap

} // namespace mfem

