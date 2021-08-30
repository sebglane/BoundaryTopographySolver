/*
 * solver_base.h
 *
 *  Created on: Aug 30, 2021
 *      Author: sg
 */

#ifndef INCLUDE_SOLVER_BASE_H_
#define INCLUDE_SOLVER_BASE_H_

#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/tria.h>

#include <deal.II/fe/fe_system.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>

#include <vector>

namespace TopographyProblem {

using namespace dealii;

/*
 * @class SolverBase
 *
 */
template <int dim>
class SolverBase
{
public:
    SolverBase(const unsigned int  n_refinements = 3,
               const double        newton_tolerance = 1e-9,
               const unsigned int  n_maximum_iterations = 10);

    void run();

    virtual void make_grid() = 0;

private:
    virtual void setup_fe_system() = 0;

    virtual void setup_dofs();

    void setup_system_matrix
    (const std::vector<types::global_dof_index> &dofs_per_block,
     const Table<2, DoFTools::Coupling>         &coupling_table);

    void setup_vectors(const std::vector<types::global_dof_index> &dofs_per_block);

    virtual void assemble_system(const bool initial_step) = 0;

    virtual void assemble_rhs(const bool initial_step) = 0;

    void solve(const bool initial_step);

    void newton_iteration(const bool is_initial_step);

    virtual void postprocess_solution(const unsigned int cycle = 0) = 0;

    virtual void output_results(const unsigned int cycle = 0) const = 0;

    virtual void refine_mesh();

    Triangulation<dim>          triangulation;
    FESystem<dim>              *fe_system;
    DoFHandler<dim>             dof_handler;

    // constraints
    AffineConstraints<double>   nonzero_constraints;
    AffineConstraints<double>   zero_constraints;

    // system matrix
    BlockSparsityPattern        sparsity_pattern;
    BlockSparseMatrix<double>   system_matrix;

    // vectors
    BlockVector<double>         evaluation_point;
    BlockVector<double>         present_solution;
    BlockVector<double>         solution_update;
    BlockVector<double>         system_rhs;

    // monitor of computing times
    TimerOutput                 computing_timer;

    const unsigned int  n_refinements;

    const double        newton_tolerance;
    const unsigned int  n_maximum_iterations;

};

}  // namespace TopographyProblem

#endif /* INCLUDE_SOLVER_BASE_H_ */
