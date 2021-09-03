/*
 * assemble_system.cc
 *
 *  Created on: Nov 23, 2018
 *      Author: sg
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/work_stream.h>

#include <solver.h>

#include <equation_data.h>

namespace TopographyProblem {

template<int dim>
void TopographySolver<dim>::assemble_system(const bool initial_step)
{
    TimerOutput::Scope timer_section(computing_timer, "assembly");

    const QGauss<dim>   quadrature_formula(parameters.velocity_degree + 1);
    const QGauss<dim-1> face_quadrature(parameters.velocity_degree + 1);

    system_matrix = 0;
    system_rhs = 0;

    WorkStream::run(dof_handler.begin_active(),
                    dof_handler.end(),
                    std::bind(&TopographySolver<dim>::local_assemble,
                              this,
                              std::placeholders::_1,
                              std::placeholders::_2,
                              std::placeholders::_3,
                              true,
                              initial_step),
                    std::bind(&TopographySolver<dim>::copy_local_to_global,
                              this,
                              std::placeholders::_1,
                              true,
                              initial_step),
                    Assembly::Scratch<dim>(fe_system,
                                           quadrature_formula,
                                           face_quadrature,
                                           update_values|
                                           update_gradients|
                                           update_JxW_values,
                                           update_values|
                                           update_gradients|
                                           update_normal_vectors|
                                           update_JxW_values),
                    Assembly::CopyData<dim>(fe_system));
}

template<int dim>
void TopographySolver<dim>::assemble_rhs(const bool initial_step)
{
    TimerOutput::Scope timer_section(computing_timer, "assembly");

    const QGauss<dim>   quadrature_formula(parameters.velocity_degree + 1);
    const QGauss<dim-1> face_quadrature(parameters.velocity_degree + 1);

    system_rhs = 0;

    WorkStream::run(dof_handler.begin_active(),
                    dof_handler.end(),
                    std::bind(&TopographySolver<dim>::local_assemble,
                              this,
                              std::placeholders::_1,
                              std::placeholders::_2,
                              std::placeholders::_3,
                              false,
                              initial_step),
                    std::bind(&TopographySolver<dim>::copy_local_to_global,
                              this,
                              std::placeholders::_1,
                              false,
                              initial_step),
                    Assembly::Scratch<dim>(fe_system,
                                           quadrature_formula,
                                           face_quadrature,
                                           update_values|
                                           update_gradients|
                                           update_JxW_values,
                                           update_values|
                                           update_gradients|
                                           update_normal_vectors|
                                           update_JxW_values),
                    Assembly::CopyData<dim>(fe_system));
}

}  // namespace TopographyProblem

// explicit instantiation
template void TopographyProblem::TopographySolver<3>::assemble_system(const bool);

template void TopographyProblem::TopographySolver<3>::assemble_rhs(const bool);

