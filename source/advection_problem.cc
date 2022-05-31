/*
 * buoyant_hydrodynamic_problem.cc
 *
 *  Created on: Sep 8, 2021
 *      Author: sg
 */

#include <advection_problem.h>

#include <fstream>

namespace Advection {

ProblemParameters::ProblemParameters()
:
SolverParameters(),
mapping_degree(1)
{}



ProblemParameters::ProblemParameters
(const std::string &filename)
:
ProblemParameters()
{
  ParameterHandler prm;
  declare_parameters(prm);

  std::ifstream parameter_file(filename.c_str());

  if (!parameter_file)
  {
    parameter_file.close();

    std::ostringstream message;
    message << "Input parameter file <"
            << filename << "> not found. Creating a"
            << std::endl
            << "template file of the same name."
            << std::endl;

    std::ofstream parameter_out(filename.c_str());
    prm.print_parameters(parameter_out,
                         ParameterHandler::OutputStyle::Text);

    AssertThrow(false, ExcMessage(message.str().c_str()));
  }

  prm.parse_input(parameter_file);

  parse_parameters(prm);
}



void ProblemParameters::declare_parameters(ParameterHandler &prm)
{
  SolverParameters::declare_parameters(prm);

  prm.declare_entry("Mapping - Polynomial degree",
                    "1",
                    Patterns::Integer(1));

  prm.enter_subsection("Advection solver parameters");
  {
    prm.declare_entry("Stratification number",
                      "0.0",
                      Patterns::Double(0.0));
  }
  prm.leave_subsection();

}



void ProblemParameters::parse_parameters(ParameterHandler &prm)
{
  SolverParameters::parse_parameters(prm);

  mapping_degree = prm.get_integer("Mapping - Polynomial degree");
  AssertThrow(mapping_degree > 0, ExcLowerRange(mapping_degree, 0) );

  prm.enter_subsection("Advection solver parameters");
  {
    stratification_number = prm.get_double("Stratification number");
    AssertThrow(stratification_number >= 0.0, ExcLowerRangeType<double>(stratification_number, 0.0));
    AssertIsFinite(stratification_number);
  }
  prm.leave_subsection();

}



template<typename Stream>
Stream& operator<<(Stream &stream, const ProblemParameters &prm)
{
  stream << static_cast<const SolverParameters &>(prm);

  {
     std::stringstream strstream;

     strstream << "MappingQ<dim>"
               << "(" << std::to_string(prm.mapping_degree) << ")";
     Utility::add_line(stream, "Mapping", strstream.str().c_str());
   }
  if (prm.stratification_number > 0.0)
    Utility::add_line(stream, "Stratification number", prm.stratification_number);

  Utility::add_header(stream);

  return (stream);
}



template <int dim, typename TriangulationType>
AdvectionProblem<dim, TriangulationType>::AdvectionProblem(const ProblemParameters &parameters)
:
mapping(parameters.mapping_degree),
solver(triangulation, mapping, parameters, parameters.stratification_number),
n_initial_refinements(parameters.refinement_parameters.n_initial_refinements),
n_initial_bndry_refinements(parameters.refinement_parameters.n_initial_bndry_refinements)
{
  solver.get_conditional_output_stream()  << parameters << std::endl;
}



template <int dim, typename TriangulationType>
void AdvectionProblem<dim, TriangulationType>::initialize_mapping()
{
  solver.get_conditional_output_stream()  << "    Initialize mapping..." << std::endl;

  mapping.initialize(triangulation, MappingQGeneric<dim>(mapping.get_degree()));
}



template <int dim, typename TriangulationType>
void AdvectionProblem<dim, TriangulationType>::run()
{
  this->make_grid();

  initialize_mapping();

  this->set_boundary_conditions();

  this->set_advection_field();

  this->set_source_term();

  solver.solve();
}

// explicit instantiations
template std::ostream & operator<<(std::ostream &, const ProblemParameters &);
template ConditionalOStream & operator<<(ConditionalOStream &, const ProblemParameters &);

template AdvectionProblem<2>::AdvectionProblem(const ProblemParameters &);
template AdvectionProblem<3>::AdvectionProblem(const ProblemParameters &);

template void AdvectionProblem<2>::initialize_mapping();
template void AdvectionProblem<3>::initialize_mapping();

template void AdvectionProblem<2>::run();
template void AdvectionProblem<3>::run();

template class AdvectionProblem<2>;
template class AdvectionProblem<3>;


}  // namespace Advection



