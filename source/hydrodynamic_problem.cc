/*
 * hydrodynamic_problem.cc
 *
 *  Created on: Sep 1, 2021
 *      Author: sg
 */

#include <hydrodynamic_problem.h>

#include <fstream>

namespace Hydrodynamic {

ProblemParameters::ProblemParameters()
:
SolverParameters(),
mapping_degree(1),
froude_number(0.0),
reynolds_number(1.0)
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

  prm.enter_subsection("Hydrodynamic solver parameters");
  {
    prm.declare_entry("Froude number",
                      "0.0",
                      Patterns::Double(0.0));

    prm.declare_entry("Reynolds number",
                      "1.0",
                      Patterns::Double(std::numeric_limits<double>::epsilon()));
  }
  prm.leave_subsection();
}



void ProblemParameters::parse_parameters(ParameterHandler &prm)
{
  SolverParameters::parse_parameters(prm);

  mapping_degree = prm.get_integer("Mapping - Polynomial degree");
  AssertThrow(mapping_degree > 0, ExcLowerRange(mapping_degree, 0) );

  prm.enter_subsection("Hydrodynamic solver parameters");
  {
    froude_number = prm.get_double("Froude number");
    AssertThrow(froude_number >= 0.0, ExcLowerRangeType<double>(froude_number, 0.0));
    AssertIsFinite(froude_number);

    reynolds_number = prm.get_double("Reynolds number");
    AssertThrow(reynolds_number > 0.0, ExcLowerRangeType<double>(reynolds_number, 0.0));
    AssertIsFinite(reynolds_number);
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

  Utility::add_line(stream, "Reynolds number", prm.reynolds_number);

  Utility::add_line(stream, "Froude number", prm.froude_number);

  Utility::add_header(stream);

  Utility::add_header(stream);

  return (stream);
}



template <int dim>
HydrodynamicProblem<dim>::HydrodynamicProblem(const ProblemParameters &parameters)
:
mapping(parameters.mapping_degree),
solver(triangulation, mapping, parameters, parameters.reynolds_number, parameters.froude_number)
{
  std::cout << parameters << std::endl;
}



template <int dim>
void HydrodynamicProblem<dim>::initialize_mapping()
{
  std::cout << "    Initialize mapping..." << std::endl;

  mapping.initialize(triangulation, MappingQGeneric<dim>(mapping.get_degree()));
}



template <int dim>
void HydrodynamicProblem<dim>::run()
{
  this->make_grid();

  initialize_mapping();

  this->set_boundary_conditions();

  this->set_body_force_term();

  solver.solve();
}

// explicit instantiations
template std::ostream & operator<<(std::ostream &, const ProblemParameters &);

template HydrodynamicProblem<2>::HydrodynamicProblem(const ProblemParameters &);
template HydrodynamicProblem<3>::HydrodynamicProblem(const ProblemParameters &);

template void HydrodynamicProblem<2>::initialize_mapping();
template void HydrodynamicProblem<3>::initialize_mapping();

template void HydrodynamicProblem<2>::run();
template void HydrodynamicProblem<3>::run();

template class HydrodynamicProblem<2>;
template class HydrodynamicProblem<3>;


}  // namespace Hydrodynamic
