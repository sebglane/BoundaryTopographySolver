/*
 * solver_base.cc
 *
 *  Created on: Aug 30, 2021
 *      Author: sg
 */

#include <solver_base.h>

#include <iostream>
#include <filesystem>

namespace SolverBase {

Parameters::Parameters()
:
refinement_parameters(),
absolute_tolerance(1e-12),
relative_tolerance(1e-9),
verbose(false),
print_timings(false),
graphical_output_directory("./")
{}



void Parameters::declare_parameters(ParameterHandler &prm)
{
  Utility::RefinementParameters::declare_parameters(prm);

  prm.declare_entry("Spatial dimension",
                    "2",
                    Patterns::Integer(2,3));

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



void Parameters::parse_parameters(ParameterHandler &prm)
{
  refinement_parameters.parse_parameters(prm);

  space_dim = prm.get_integer("Spatial dimension");

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
Stream& operator<<(Stream &stream, const Parameters &prm)
{
  Utility::add_header(stream);
  Utility::add_line(stream, "Problem parameters");
  Utility::add_header(stream);

  Utility::add_line(stream,
           "Spatial dimension",
           prm.space_dim);

  Utility::add_line(stream,
           "Max. number of Newton iterations",
           prm.n_iterations);


  Utility::add_line(stream,
           "Absolute tolerance",
           prm.absolute_tolerance);

  Utility::add_line(stream,
           "Relative tolerance",
           prm.relative_tolerance);

  Utility::add_line(stream, "Verbose", (prm.verbose? "true": "false"));

  Utility::add_line(stream, "Print timings", (prm.print_timings? "true": "false"));

  Utility::add_line(stream,
           "Graphical output directory",
                     prm.graphical_output_directory);

  stream << prm.refinement_parameters;

  return (stream);
}



template <int dim>
Solver<dim>::Solver
(Triangulation<dim>  &tria,
 Mapping<dim>        &mapping,
 const Parameters &parameters)
:
triangulation(tria),
mapping(mapping),
fe_system(nullptr),
dof_handler(triangulation),
computing_timer(std::cout,
                TimerOutput::summary,
                TimerOutput::wall_times),
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
void Solver<dim>::solve()
{
  this->setup_fe_system();

  this->setup_dofs();

  for (unsigned int cycle = 0; cycle < refinement_parameters.n_cycles; ++cycle)
  {
    std::cout << "Cycle " << cycle << ':' << std::endl;

    newton_iteration(cycle == 0);

    this->postprocess_solution(cycle);

    this->output_results(cycle);

    std::cout << "End cycle " << cycle << std::endl;

    this->refine_mesh();
  }
}



template<int dim>
void Solver<dim>::postprocess_solution(const unsigned int cycle) const
{
  if (postprocessor_ptrs.empty())
    return;

  if (verbose)
    std::cout << "    Postprocess solution..." << std::endl;

  for (const auto &ptr: postprocessor_ptrs)
  {
    ptr->set_cycle(cycle);
    (*ptr)(mapping,
           *fe_system,
           dof_handler,
           present_solution);
  }
}


template <int dim>
void Solver<dim>::newton_iteration(const bool is_initial_cycle)
{
  auto compute_residual = [this](const double alpha = 0.0,
                                 const bool use_homogeneous_constraints = true)
      {
        this->evaluation_point = this->present_solution;
        if (alpha != 0.0)
          this->evaluation_point.add(alpha, this->solution_update);
        this->nonzero_constraints.distribute(this->evaluation_point);
        this->assemble_rhs(use_homogeneous_constraints);

        std::vector<double> residual_components(this->system_rhs.n_blocks());
        for (unsigned int i=0; i<this->system_rhs.n_blocks(); ++i)
          residual_components[i] = this->system_rhs.block(i).l2_norm();

        return (std::make_tuple(this->system_rhs.l2_norm(), residual_components));
      };

  this->preprocess_newton_iteration(0, is_initial_cycle);
  const double initial_residual{std::get<0>(compute_residual(0.0, false))};

  std::cout << "Initial residual: "
            << std::scientific << initial_residual
            << std::endl
            << std::defaultfloat;

  const double tolerance{std::max(absolute_tolerance,
                                  relative_tolerance * initial_residual)};

  double current_residual = std::numeric_limits<double>::max();
  std::vector<double> current_residual_components(this->system_rhs.n_blocks(),
                                                  std::numeric_limits<double>::max());
  double last_residual = std::numeric_limits<double>::max();
  bool first_step = is_initial_cycle;

  unsigned int iteration = 0;

  while ((current_residual > tolerance) &&
         iteration < n_maximum_iterations)
  {

    this->preprocess_newton_iteration(iteration, is_initial_cycle);

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
      std::tie(current_residual, current_residual_components) = compute_residual(0.0, true);
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
        std::tie(current_residual, current_residual_components) = compute_residual(alpha);
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
              << "), Residual components: ";
    for (const auto residual_component: current_residual_components)
      std::cout << std::scientific << std::setprecision(4) << residual_component << ", ";
    std::cout << std::endl
              << std::defaultfloat;

    // update residual
    last_residual = current_residual;
    ++iteration;
  }

  AssertThrow(current_residual <= tolerance,
              ExcMessage("Newton solver did not converge!"));
}

// explicit instantiations
template std::ostream & operator<<(std::ostream &, const Parameters &);

template Solver<2>::Solver
(Triangulation<2> &, Mapping<2> &, const Parameters &);
template Solver<3>::Solver
(Triangulation<3> &, Mapping<3> &, const Parameters &);

template void Solver<2>::postprocess_solution(const unsigned int) const;
template void Solver<3>::postprocess_solution(const unsigned int) const;

template void Solver<2>::newton_iteration(const bool);
template void Solver<3>::newton_iteration(const bool);

template void Solver<2>::solve();
template void Solver<3>::solve();

}  // namespace SolverBase
