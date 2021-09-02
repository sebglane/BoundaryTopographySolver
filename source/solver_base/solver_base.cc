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
 Mapping<dim>        &mapping,
 const unsigned int   n_refinements,
 const double         newton_tolerance,
 const unsigned int   n_maximum_iterations)
:
triangulation(tria),
mapping(mapping),
dof_handler(triangulation),
print_timings(false),
computing_timer(std::cout,
                print_timings? TimerOutput::summary: TimerOutput::never,
                TimerOutput::wall_times),
postprocessor_ptr(nullptr),
n_refinements(n_refinements),
newton_tolerance(newton_tolerance),
n_maximum_iterations(n_maximum_iterations),
verbose(false)
{}



template <int dim>
void SolverBase<dim>::solve()
{
  this->setup_fe_system();

  this->setup_dofs();

  for (unsigned int cycle = 0; cycle < n_refinements; ++cycle)
  {
    std::cout << "Cycle " << cycle << ':' << std::endl;

    newton_iteration(cycle == 0? true: false);

    this->postprocess_solution(cycle);

    this->output_results(cycle);

    this->refine_mesh();
  }
}


template<int dim>
void SolverBase<dim>::postprocess_solution(const unsigned int cycle) const
{
  if (postprocessor_ptr == nullptr)
    return;

  if (verbose)
    std::cout << "    Postprocess solution..." << std::endl;

  postprocessor_ptr->set_cycle(cycle);
  (*postprocessor_ptr)(mapping,
                       *fe_system,
                       dof_handler,
                       solution_update);
}


template <int dim>
void SolverBase<dim>::newton_iteration(const bool is_initial_step)
{
  auto compute_residual = [this](const double alpha = 0.0,
                                 const bool use_homogeneous_constraints = true)
      {
        this->evaluation_point = this->present_solution;
        if (alpha != 0.0)
          this->evaluation_point.add(alpha, this->solution_update);
        this->nonzero_constraints.distribute(this->evaluation_point);
        this->assemble_rhs(use_homogeneous_constraints);
        return this->system_rhs.l2_norm();
      };

  std::cout << "Initial residual: "
            << std::scientific << compute_residual(0.0)
            << std::endl
            << std::defaultfloat;

  double current_residual = std::numeric_limits<double>::max();
  double last_residual = std::numeric_limits<double>::max();
  bool first_step = is_initial_step;

  unsigned int iteration = 0;

  while ((current_residual > newton_tolerance) &&
         iteration < n_maximum_iterations)
  {
    if (first_step)
    {
      // solve problem
      evaluation_point = present_solution;
      this->assemble_system(/* use_homogeneous_constraints ? */ false);
      solve_linear_system(/* use_homogeneous_constraints ? */ false);
      present_solution = solution_update;
      nonzero_constraints.distribute(present_solution);
      first_step = false;
      // compute residual
      current_residual = compute_residual(0.0, true);
    }
    else
    {
      // solve problem
      evaluation_point = present_solution;
      this->assemble_system(/* use_homogeneous_constraints ? */ true);
      solve_linear_system(/* use_homogeneous_constraints ? */ true);
      // line search
      if (verbose)
        std::cout << "   Line search: " << std::endl;
      for (double alpha = 1.0; alpha > 1e-2; alpha *= 0.5)
      {
        current_residual = compute_residual(alpha);
        if (verbose)
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

  AssertThrow(current_residual <= newton_tolerance,
              ExcMessage("Newton solver did not converge!"));
}

// explicit instantiations
template SolverBase<2>::SolverBase
(Triangulation<2> &, Mapping<2> &, const unsigned int, const double, const unsigned int);
template SolverBase<3>::SolverBase
(Triangulation<3> &, Mapping<3> &, const unsigned int, const double, const unsigned int);

template void SolverBase<2>::postprocess_solution(const unsigned int) const;
template void SolverBase<3>::postprocess_solution(const unsigned int) const;

template void SolverBase<2>::newton_iteration(const bool);
template void SolverBase<3>::newton_iteration(const bool);

template void SolverBase<2>::solve();
template void SolverBase<3>::solve();

}  // namespace TopographyProblem
