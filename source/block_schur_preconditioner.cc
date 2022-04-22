/*
 * block_schur_preconditioner.cc
 *
 *  Created on: Apr 7, 2022
 *      Author: sg
 */

#include <block_schur_preconditioner.h>

#include <deal.II/lac/solver_selector.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_direct.h>

namespace Preconditioning {

template <class PreconditionerTypeA, class PreconditionerTypeS>
BlockSchurPreconditioner<PreconditionerTypeA, PreconditionerTypeS>::
BlockSchurPreconditioner
(const LA::BlockSparseMatrix &system_matrix,
 const LA::SparseMatrix      &schur_complement_matrix,
 const PreconditionerTypeA   &preconditioner_A,
 const PreconditionerTypeS   &preconditioner_S,
 const bool                   do_solve_A,
 const bool                   A_symmetric,
 const bool                   S_symmetric)
:
n_iterations_A(0),
n_iterations_S(0),
system_matrix(system_matrix),
schur_complement_matrix(schur_complement_matrix),
preconditioner_A(preconditioner_A),
preconditioner_S(preconditioner_S),
do_solve_A(do_solve_A),
A_symmetric(A_symmetric),
S_symmetric(S_symmetric)
{}

template <class PreconditionerAType, class PreconditionerSType>
void
BlockSchurPreconditioner<PreconditionerAType, PreconditionerSType>::
vmult
(LA::BlockVector        &dst,
 const LA::BlockVector  &src) const
{
  // The linear system reads
  //  ----------- -----    -----
  //  |  A   B  | | u |  = | f |
  //  |  C   D  | | v |    | g |
  //  ----------- -----    -----
  //
  //  The preconditioner reads
  //      ---------------
  //  P = |  A     B    |
  //      |  0   -S_A   |
  //      ---------------
  // The Schur complement with respect to A is given by S_A = C * A^-1 * B - D.
  // The inverse of the preconditioner is given by
  //         -----------------------------
  //  P^-1 = |  A^-1   A^-1 * B * S_A^-1 |
  //         |   0        -S_A^-1        |
  //         -----------------------------
  //         ------------- ----------- ---------------
  //       = |  A^-1  0  | |  I  -B  | |  I    0     |
  //         |   0    I  | |  0   I  | |  0  -S_A^-1 |
  //         ------------- ----------- ---------------
  //
  // A right-multiplication of the linear system with P^-1 yields
  //
  //  -----------         ---------------
  //  |  A   B  |  P^-1 = |   I       0 |
  //  |  C   D  |         | C * A^-1  I |
  //  -----------         ---------------
  //
  // Instead of the matrix S_A, a Schur complement approximation is used,
  // which will be denoted by S.

  using VectorType = LA::BlockVector::BlockType;

  // 1.) Solve Schur complement system: -S * y = v
  {
    SolverControl   solver_control(1000, 1e-6 * src.block(1).l2_norm());
    SolverSelector<VectorType>  solver;
    if (S_symmetric)
      solver.select("cg");
    else
      solver.select("gmres");
    solver.set_control(solver_control);

    dst.block(1) = 0.0;

    try
    {
      solver.solve(schur_complement_matrix,
                   dst.block(1),
                   src.block(1),
                   preconditioner_S);
    }
    catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception in solving the Schur complement equation:" << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::abort();
    }
    catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception in solving the Schur complement equation!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::abort();
    }

    n_iterations_S += solver_control.last_step();
    // change sign
    dst.block(1) *= -1.0;
  }
  // 2.) Compute auxiliary right-hand side: f_aux = f - B * v
  VectorType  aux_vector(src.block(0).size());
  {
    system_matrix.block(0, 1).vmult(aux_vector, dst.block(1));
    aux_vector *= -1.0;
    aux_vector += src.block(0);
  }
  // 3.) Solve the linear system: A u = f - B * v
  if (do_solve_A == true)
  {
      SolverControl solver_control(10000, 1e-4 * aux_vector.l2_norm());
      SolverSelector<VectorType>  solver;
      if (S_symmetric)
        solver.select("cg");
      else
        solver.select("gmres");
      solver.set_control(solver_control);

      dst.block(0) = 0.0;

      try
      {
        solver.solve(system_matrix.block(0, 0),
                     dst.block(0),
                     aux_vector,
                     preconditioner_A);
      }
      catch (std::exception &exc)
      {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception in the solve method of the diffusion step: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::abort();
      }
      catch (...)
      {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception in the solve method of the diffusion step!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::abort();
      }

      n_iterations_A += solver_control.last_step();
  }
  else
  {
    preconditioner_A.vmult(dst.block(0), aux_vector);
    n_iterations_A += 1;
  }
}

template class BlockSchurPreconditioner<SparseILU<double>, SparseILU<double>>;
template class BlockSchurPreconditioner<SparseDirectUMFPACK, SparseILU<double>>;
template class BlockSchurPreconditioner<SparseILU<double>, SparseDirectUMFPACK>;
template class BlockSchurPreconditioner<SparseDirectUMFPACK, SparseDirectUMFPACK>;

}  // namespace Preconditioning


