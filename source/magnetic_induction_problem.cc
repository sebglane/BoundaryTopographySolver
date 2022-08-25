/*
 * magnetic_induction_problem.cc
 *
 *  Created on: Aug 10, 2022
 *      Author: sg
 */
#include <magnetic_induction_problem.h>

#include <fstream>

namespace MagneticInduction {

ProblemParameters::ProblemParameters()
:
SolverParameters(),
mapping_degree(1),
magnetic_reynolds_number(1.0)
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

  prm.enter_subsection("Magnetic induction solver parameters");
  {
    prm.declare_entry("Magnetic Reynolds number",
                      "1.0",
                      Patterns::Double());
  }
  prm.leave_subsection();
}



void ProblemParameters::parse_parameters(ParameterHandler &prm)
{
  SolverParameters::parse_parameters(prm);

  mapping_degree = prm.get_integer("Mapping - Polynomial degree");
  AssertThrow(mapping_degree > 0, ExcLowerRange(mapping_degree, 0) );

  prm.enter_subsection("Magnetic induction solver parameters");
  {
    magnetic_reynolds_number = prm.get_double("Magnetic Reynolds number");
    AssertThrow(magnetic_reynolds_number > 0.0, ExcLowerRangeType<double>(0.0, magnetic_reynolds_number));
    AssertIsFinite(magnetic_reynolds_number);
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

  Utility::add_line(stream, "Magnetic Reynolds number", prm.magnetic_reynolds_number);

  Utility::add_header(stream);

  Utility::add_header(stream);

  return (stream);
}



template <int dim, typename TriangulationType>
MagneticInductionProblem<dim, TriangulationType>::
MagneticInductionProblem(const ProblemParameters &parameters)
:
mapping(parameters.mapping_degree),
solver(triangulation, mapping, parameters,
       parameters.magnetic_reynolds_number),
n_initial_refinements(parameters.refinement_parameters.n_initial_refinements),
n_initial_bndry_refinements(parameters.refinement_parameters.n_initial_bndry_refinements)
{
  solver.get_conditional_output_stream()  << parameters << std::endl;
}



template <int dim, typename TriangulationType>
void MagneticInductionProblem<dim, TriangulationType>::initialize_mapping()
{
  solver.get_conditional_output_stream()  << "    Initialize mapping..." << std::endl;

  mapping.initialize(MappingQGeneric<dim>(mapping.get_degree()), triangulation);
}



template <int dim, typename TriangulationType>
void MagneticInductionProblem<dim, TriangulationType>::run()
{
  this->make_grid();

  initialize_mapping();

  this->set_boundary_conditions();

  this->set_background_magnetic_field();

  this->set_postprocessor();

  solver.solve();

}

// explicit instantiations
template std::ostream & operator<<(std::ostream &, const ProblemParameters &);
template ConditionalOStream & operator<<(ConditionalOStream &, const ProblemParameters &);

template MagneticInductionProblem<2>::MagneticInductionProblem(const ProblemParameters &);
template MagneticInductionProblem<3>::MagneticInductionProblem(const ProblemParameters &);

template void MagneticInductionProblem<2>::initialize_mapping();
template void MagneticInductionProblem<3>::initialize_mapping();

template void MagneticInductionProblem<2>::run();
template void MagneticInductionProblem<3>::run();

template class MagneticInductionProblem<2>;
template class MagneticInductionProblem<3>;


}  // namespace MagneticInduction



