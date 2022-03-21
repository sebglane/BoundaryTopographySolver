/*
 * buoyant_hydrodynamic_problem.cc
 *
 *  Created on: Sep 8, 2021
 *      Author: sg
 */

#include <buoyant_hydrodynamic_problem.h>

#include <fstream>

namespace BuoyantHydrodynamic {

ProblemParameters::ProblemParameters()
:
SolverParameters(),
mapping_degree(1),
froude_number(0.0),
reynolds_number(1.0),
stratification_number(1.0),
rossby_number(0.0)
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

  prm.enter_subsection("Buoyant hydrodynamic solver parameters");
  {
    prm.declare_entry("Froude number",
                      "0.0",
                      Patterns::Double(0.0));

    prm.declare_entry("Reynolds number",
                      "1.0",
                      Patterns::Double(std::numeric_limits<double>::epsilon()));

    prm.declare_entry("Stratification number",
                      "1.0",
                      Patterns::Double());

    prm.declare_entry("Rossby number",
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

  prm.enter_subsection("Buoyant hydrodynamic solver parameters");
  {
    froude_number = prm.get_double("Froude number");
    AssertThrow(froude_number >= 0.0, ExcLowerRangeType<double>(0.0, froude_number));
    AssertIsFinite(froude_number);

    reynolds_number = prm.get_double("Reynolds number");
    AssertThrow(reynolds_number > 0.0, ExcLowerRangeType<double>(0.0, reynolds_number));
    AssertIsFinite(reynolds_number);

    stratification_number = prm.get_double("Stratification number");
    AssertThrow(stratification_number >= 0.0, ExcLowerRangeType<double>(0.0, stratification_number));
    AssertIsFinite(stratification_number);

    rossby_number = prm.get_double("Rossby number");
    AssertThrow(rossby_number >= 0.0, ExcLowerRangeType<double>(rossby_number, 0.0));
    AssertIsFinite(rossby_number);
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

  if (prm.froude_number > 0.0)
    Utility::add_line(stream, "Froude number", prm.froude_number);

  if (prm.stratification_number > 0.0)
    Utility::add_line(stream, "Stratification number", prm.stratification_number);

  if (prm.rossby_number > 0.0)
    Utility::add_line(stream, "Rossby number", prm.rossby_number);

  Utility::add_header(stream);

  Utility::add_header(stream);

  return (stream);
}



template <int dim, typename TriangulationType, typename LinearAlgebraContainer>
BuoyantHydrodynamicProblem<dim, TriangulationType, LinearAlgebraContainer>::
BuoyantHydrodynamicProblem(const ProblemParameters &parameters)
:
mapping(parameters.mapping_degree),
solver(triangulation, mapping, parameters,
       parameters.reynolds_number, parameters.froude_number, parameters.stratification_number,
       parameters.rossby_number),
n_initial_refinements(parameters.refinement_parameters.n_initial_refinements),
n_initial_bndry_refinements(parameters.refinement_parameters.n_initial_bndry_refinements)
{
  solver.get_conditional_output_stream()  << parameters << std::endl;
}



template <int dim, typename TriangulationType, typename LinearAlgebraContainer>
void BuoyantHydrodynamicProblem<dim, TriangulationType, LinearAlgebraContainer>::initialize_mapping()
{
  solver.get_conditional_output_stream()  << "    Initialize mapping..." << std::endl;

  mapping.initialize(triangulation, MappingQGeneric<dim>(mapping.get_degree()));
}



template <int dim, typename TriangulationType, typename LinearAlgebraContainer>
void BuoyantHydrodynamicProblem<dim, TriangulationType, LinearAlgebraContainer>::run()
{
  this->make_grid();

  initialize_mapping();

  this->set_boundary_conditions();

  this->set_angular_velocity();

  this->set_background_velocity();

  this->set_body_force_term();

  this->set_gravity_field();

  this->set_reference_density();

  this->set_postprocessor();

  solver.solve();
}

// explicit instantiations
template std::ostream & operator<<(std::ostream &, const ProblemParameters &);
template ConditionalOStream & operator<<(ConditionalOStream &, const ProblemParameters &);

template BuoyantHydrodynamicProblem<2>::BuoyantHydrodynamicProblem(const ProblemParameters &);
template BuoyantHydrodynamicProblem<3>::BuoyantHydrodynamicProblem(const ProblemParameters &);

template void BuoyantHydrodynamicProblem<2>::initialize_mapping();
template void BuoyantHydrodynamicProblem<3>::initialize_mapping();

template void BuoyantHydrodynamicProblem<2>::run();
template void BuoyantHydrodynamicProblem<3>::run();

template class BuoyantHydrodynamicProblem<2>;
template class BuoyantHydrodynamicProblem<3>;


}  // namespace Hydrodynamic



