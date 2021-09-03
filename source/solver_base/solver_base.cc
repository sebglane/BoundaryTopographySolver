/*
 * solver_base.cc
 *
 *  Created on: Aug 30, 2021
 *      Author: sg
 */

#include <solver_base.h>

#include <iostream>
#include <filesystem>

namespace TopographyProblem {

SolverBaseParameters::SolverBaseParameters()
:
refinement_parameters(),
absolute_tolerance(1e-12),
relative_tolerance(1e-9),
verbose(false),
print_timings(false),
graphical_output_directory("./")
{}



void SolverBaseParameters::declare_parameters(ParameterHandler &prm)
{
  RefinementParameters::declare_parameters(prm);

  prm.declare_entry("Max. number of Newton iterations",
                    "15",
                    Patterns::Integer(1));

  prm.declare_entry("Absolute tolerance",
                    "1e-12",
                    Patterns::Double(std::numeric_limits<double>::epsilon()));

  prm.declare_entry("Relative tolerance",
                    "1e-9",
                    Patterns::Double(std::numeric_limits<double>::epsilon()));

  prm.declare_entry("Verbose",
                    "false",
                    Patterns::Bool());

  prm.declare_entry("Print timings",
                    "false",
                    Patterns::Bool());

  prm.declare_entry("Graphical output directory",
                    "./",
                    Patterns::DirectoryName());
}



void SolverBaseParameters::parse_parameters(ParameterHandler &prm)
{
  refinement_parameters.parse_parameters(prm);

  n_iterations = prm.get_integer("Max. number of Newton iterations");
  AssertThrow(n_iterations >= 1, ExcLowerRangeType<unsigned >(n_iterations, 0));
  AssertIsFinite(n_iterations);

  absolute_tolerance = prm.get_double("Absolute tolerance");
  AssertThrow(absolute_tolerance > 0.0, ExcLowerRangeType<double>(absolute_tolerance, 0.0));
  AssertIsFinite(absolute_tolerance);

  relative_tolerance = prm.get_double("Relative tolerance");
  AssertThrow(relative_tolerance > 0.0, ExcLowerRangeType<double>(relative_tolerance, 0.0));
  AssertIsFinite(relative_tolerance);
  AssertThrow(absolute_tolerance < relative_tolerance,
              ExcLowerRangeType<double>(absolute_tolerance, relative_tolerance));

  verbose = prm.get_bool("Verbose");

  print_timings = prm.get_bool("Print timings");

  graphical_output_directory = prm.get("Graphical output directory");
}



template <typename Stream>
Stream& operator<<(Stream &stream, const SolverBaseParameters &prm)
{
  internal::add_header(stream);
  internal::add_line(stream, "Problem parameters");
  internal::add_header(stream);

  internal::add_line(stream,
                     "Max. number of Newton iterations",
                     prm.n_iterations);


  internal::add_line(stream,
                     "Absolute tolerance",
                     prm.absolute_tolerance);

  internal::add_line(stream,
                     "Relative tolerance",
                     prm.relative_tolerance);

  internal::add_line(stream, "Verbose", (prm.verbose? "true": "false"));

  internal::add_line(stream, "Print timings", (prm.print_timings? "true": "false"));

  internal::add_line(stream,
                     "Graphical output directory",
                     prm.graphical_output_directory);

  stream << prm.refinement_parameters;

  return (stream);
}



template <int dim>
SolverBase<dim>::SolverBase
(Triangulation<dim>  &tria,
 Mapping<dim>        &mapping,
 const SolverBaseParameters &parameters)
:
triangulation(tria),
mapping(mapping),
fe_system(nullptr),
dof_handler(triangulation),
computing_timer(std::cout,
                TimerOutput::summary,
                TimerOutput::wall_times),
postprocessor_ptr(nullptr),
refinement_parameters(parameters.refinement_parameters),
n_maximum_iterations(parameters.n_iterations),
absolute_tolerance(parameters.absolute_tolerance),
relative_tolerance(parameters.relative_tolerance),
print_timings(parameters.print_timings),
graphical_output_directory(parameters.graphical_output_directory),
verbose(parameters.verbose)
{
  if (print_timings == false)
    computing_timer.disable_output();

  if (!std::filesystem::exists(graphical_output_directory))
  {
    try
    {
      std::filesystem::create_directories(graphical_output_directory);
    }
    catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception in the creation of the output directory: "
                << std::endl
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
      std::cerr << "Unknown exception in the creation of the output directory!"
                << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::abort();
    }
  }
}



template <int dim>
void SolverBase<dim>::solve()
{
  this->setup_fe_system();

  this->setup_dofs();

  for (unsigned int cycle = 0; cycle < refinement_parameters.n_cycles; ++cycle)
  {
    std::cout << "Cycle " << cycle << ':' << std::endl;

    newton_iteration(cycle == 0? true: false);

    this->postprocess_solution(cycle);

    this->output_results(cycle);

    std::cout << "End cycle " << cycle << std::endl;

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

  const double initial_residual{compute_residual(0.0, false)};

  std::cout << "Initial residual: "
            << std::scientific << initial_residual
            << std::endl
            << std::defaultfloat;

  const double tolerance{std::max(absolute_tolerance,
                                  relative_tolerance * initial_residual)};

  double current_residual = std::numeric_limits<double>::max();
  double last_residual = std::numeric_limits<double>::max();
  bool first_step = is_initial_step;

  unsigned int iteration = 0;

  while ((current_residual > tolerance) &&
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
    std::cout << "Iteration: " << std::setw(3) << std::right << iteration
              << ", Current residual: "
              << std::scientific << std::setprecision(4) << current_residual
              << " (Tolerance: "
              << std::scientific << std::setprecision(4) << tolerance
              << ")"
              << std::endl
              << std::defaultfloat;

    // update residual
    last_residual = current_residual;
    ++iteration;
  }

  AssertThrow(current_residual <= tolerance,
              ExcMessage("Newton solver did not converge!"));
}

// explicit instantiations
template std::ostream & operator<<(std::ostream &, const SolverBaseParameters &);

template SolverBase<2>::SolverBase
(Triangulation<2> &, Mapping<2> &, const SolverBaseParameters &);
template SolverBase<3>::SolverBase
(Triangulation<3> &, Mapping<3> &, const SolverBaseParameters &);

template void SolverBase<2>::postprocess_solution(const unsigned int) const;
template void SolverBase<3>::postprocess_solution(const unsigned int) const;

template void SolverBase<2>::newton_iteration(const bool);
template void SolverBase<3>::newton_iteration(const bool);

template void SolverBase<2>::solve();
template void SolverBase<3>::solve();

}  // namespace TopographyProblem
