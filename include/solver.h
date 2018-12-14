/*
 * solver.h
 *
 *  Created on: Nov 7, 2018
 *      Author: sg
 */

#ifndef INCLUDE_SOLVER_H_
#define INCLUDE_SOLVER_H_

#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>

//#include "assembly_data.h"
#include "parameters.h"

namespace TopographyProblem {

using namespace dealii;

/*
 *
 *
 */
template <int dim>
class TopographySolver
{
public:
    TopographySolver(Parameters &parameters_);

    void run();

private:
    void make_grid();

    void setup_dofs();

    void setup_system_matrix(const std::vector<types::global_dof_index> &dofs_per_block);

    void assemble_system();

    void solve();

    void output_results(const unsigned int level = 0) const;

    void refine_mesh();

    std::pair<double,double>                get_entropy_variation(const double average_density,
                                                                  const double average_velocity) const;
    std::vector<std::pair<double,double>>   get_range() const;

    double                      compute_density_viscosity
    (const std::vector<double>          &density_values,
     const std::vector<Tensor<1,dim>>   &density_gradients,
     const std::vector<Tensor<1,dim>>   &velocity_values,
     const std::vector<Tensor<2,dim>>   &velocity_gradients,
     const double                        average_density,
     const double                        global_entropy_variation,
     const double                        cell_diameter) const;

    double                      compute_velocity_viscosity
    (const std::vector<double>          &density_values,
     const std::vector<Tensor<1,dim>>   &density_gradients,
     const std::vector<Tensor<1,dim>>   &velocity_values,
     const std::vector<Tensor<2,dim>>   &velocity_gradients,
     const double                        average_density,
     const double                        global_entropy_variation,
     const double                        cell_diameter) const;

    Tensor<1,dim>   compute_boundary_traction() const;

    Parameters                 &parameters;

    std::vector<double>         equation_coefficients;

    const Tensor<1,dim>         gravity_vector;
    const Tensor<1,dim>         background_velocity_value;
    const Tensor<1,dim>         background_density_gradient;
    const Tensor<2,dim>         background_velocity_gradient;

    Triangulation<dim>          triangulation;

    // finite elements and dof handler
    FESystem<dim>               fe_system;
    DoFHandler<dim>             dof_handler;

    // constraints
    ConstraintMatrix            constraints;

    // system matrix
    BlockSparsityPattern        sparsity_pattern;
    BlockSparseMatrix<double>   system_matrix;

    // vectors
    BlockVector<double>         solution;
    BlockVector<double>         system_rhs;

    // monitor of computing times
    TimerOutput                 computing_timer;
};

}  // namespace BouyantFluid

#endif /* INCLUDE_SOLVER_H_ */
