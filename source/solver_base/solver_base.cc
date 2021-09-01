/*
 * solver_base.cc
 *
 *  Created on: Aug 30, 2021
 *      Author: sg
 */

#include <solver_base.h>

#include <iostream>

namespace TopographyProblem {

template <int dim>
SolverBase<dim>::SolverBase
(Triangulation<dim>  &tria,
 const Mapping<dim>  &mapping,
 const unsigned int   n_refinements,
 const double         newton_tolerance,
 const unsigned int   n_maximum_iterations)
:
triangulation(tria),
mapping_ptr(&mapping),
dof_handler(triangulation),
computing_timer(std::cout, TimerOutput::summary, TimerOutput::wall_times),
n_refinements(n_refinements),
newton_tolerance(newton_tolerance),
n_maximum_iterations(n_maximum_iterations)
{}



template <int dim>
void SolverBase<dim>::solve()
{
  bool initial_step = true;

  this->setup_fe_system();

  this->setup_dofs();

  for (unsigned int cycle = 0; cycle < n_refinements; ++cycle)
  {
    std::cout << "Cycle " << cycle << ':' << std::endl;

    newton_iteration(initial_step);

    this->postprocess_solution(cycle);

    this->output_results(cycle);

    this->refine_mesh();

    if (cycle == 0)
        initial_step = false;
  }
}


template<int dim>
void SolverBase<dim>::postprocess_solution(const unsigned int cycle) const
{
  postprocessor_ptr->set_cycle(cycle);

  (*postprocessor_ptr)(*mapping_ptr,
                       *fe_system,
                       dof_handler,
                       solution_update);
}


template <int dim>
void SolverBase<dim>::newton_iteration(const bool is_initial_step)
{
  double current_residual = 1.0;
  double last_residual = 1.0;
  bool first_step = is_initial_step;

  unsigned int iteration = 0;

  while ((first_step || (current_residual > newton_tolerance)) &&
         iteration < n_maximum_iterations)
  {
    if (first_step)
    {
      // solve problem
      evaluation_point = present_solution;
      this->assemble_system(first_step);
      solve_linear_system(first_step);
      present_solution = solution_update;
      nonzero_constraints.distribute(present_solution);
      first_step = false;
      // compute residual
      evaluation_point = present_solution;
      this->assemble_rhs(first_step);
      current_residual = system_rhs.l2_norm();
    }
    else
    {
      // solve problem
      evaluation_point = present_solution;
      this->assemble_system(first_step);
      solve_linear_system(first_step);
      // line search
      std::cout << "   Line search: " << std::endl;
      for (double alpha = 1.0; alpha > 1e-4; alpha *= 0.5)
      {
        evaluation_point = present_solution;
        evaluation_point.add(alpha, solution_update);
        nonzero_constraints.distribute(evaluation_point);
        this->assemble_rhs(first_step);
        current_residual = system_rhs.l2_norm();
        std::cout << "      alpha = " << std::setw(6)
                  << std::scientific << alpha
                  << " residual = " << current_residual
                  << std::endl
                  << std::defaultfloat;
        if (current_residual < last_residual)
          break;
      }
      present_solution = evaluation_point;
    }

    // output residual
    std::cout << "Iteration: " << std::setw(4) << iteration
              << ", Current residual: "
              << std::scientific << current_residual
              << std::endl
              << std::defaultfloat;

    // update residual
    last_residual = current_residual;
    ++iteration;
  }
}

// explicit instantiations
template SolverBase<2>::SolverBase
(Triangulation<2> &, const Mapping<2> &, const unsigned int, const double, const unsigned int);
template SolverBase<3>::SolverBase
(Triangulation<3> &, const Mapping<3> &, const unsigned int, const double, const unsigned int);

template void SolverBase<2>::postprocess_solution(const unsigned int) const;
template void SolverBase<3>::postprocess_solution(const unsigned int) const;

template void SolverBase<2>::newton_iteration(const bool);
template void SolverBase<3>::newton_iteration(const bool);

template void SolverBase<2>::solve();
template void SolverBase<3>::solve();

}  // namespace TopographyProblem
