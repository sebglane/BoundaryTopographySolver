/*
 * setup.cc
 *
 *  Created on: Aug 31, 2021
 *      Author: sg
 */

#include <deal.II/numerics/data_out.h>

#include <advection_solver.h>
#include <postprocessor_base.h>

#include <filesystem>
#include <fstream>
#include <string>

namespace Advection {

SolverParameters::SolverParameters()
:
SolverBase::Parameters(),
c_max(0.1),
c_entropy(0.1)
{}



void SolverParameters::declare_parameters(ParameterHandler &prm)
{
  SolverBase::Parameters::declare_parameters(prm);

  prm.enter_subsection("Buoyant hydrodynamic solver parameters");
  {
    prm.declare_entry("Entropy stabilization coefficients",
                      "0.1",
                      Patterns::Double(0.0));

    prm.declare_entry("Standard stabilization coefficient",
                      "0.1",
                      Patterns::Double(0.0));
  }
  prm.leave_subsection();
}



void SolverParameters::parse_parameters(ParameterHandler &prm)
{
  SolverBase::Parameters::parse_parameters(prm);

  prm.enter_subsection("Buoyant hydrodynamic solver parameters");
  {
    c_entropy = prm.get_double("Entropy stabilization coefficients");
    Assert(c_entropy > 0.0, ExcLowerRangeType<double>(0.0, c_entropy));

    c_max = prm.get_double("Standard stabilization coefficient");
    Assert(c_max > 0.0, ExcLowerRangeType<double>(0.0, c_max));
  }
  prm.leave_subsection();
}



template<typename Stream>
Stream& operator<<(Stream &stream, const SolverParameters &prm)
{
  stream << static_cast<const SolverBase::Parameters &>(prm);

  Utility::add_header(stream);
  Utility::add_line(stream, "Advection solver parameters");
  Utility::add_header(stream);

  Utility::add_line(stream, "Entropy stabilization coeff.", prm.c_entropy);
  Utility::add_line(stream, "Standard stabilization coeff.", prm.c_max);

  return (stream);
}



template <int dim>
Solver<dim>::Solver
(Triangulation<dim>     &tria,
 Mapping<dim>           &mapping,
 const SolverParameters &parameters)
:
SolverBase::Solver<dim>(tria, mapping, parameters),
boundary_conditions(this->triangulation),
advection_field_ptr(nullptr),
source_term_ptr(nullptr),
fe_degree(1),
c_max(parameters.c_max),
c_entropy(parameters.c_entropy),
global_entropy_variation{0.0}
{}



template<int dim>
void Solver<dim>::output_results(const unsigned int cycle) const
{
  if (this->verbose)
    std::cout << "    Output results..." << std::endl;

  // prepare data out object
  DataOut<dim, DoFHandler<dim>>    data_out;
  data_out.attach_dof_handler(this->dof_handler);
  data_out.add_data_vector(this->present_solution, "field");

  data_out.build_patches(fe_degree);

  // write output to disk
  const std::string filename = ("solution-" +
                                Utilities::int_to_string(cycle, 2) +
                                ".vtk");
  std::filesystem::path output_file(this->graphical_output_directory);
  output_file /= filename;

  std::ofstream fstream(output_file.c_str());
  data_out.write_vtk(fstream);
}

// explicit instantiation
template std::ostream & operator<<(std::ostream &, const SolverParameters &);

template Solver<2>::Solver
(Triangulation<2>  &,
 Mapping<2>        &,
 const SolverParameters &);
template Solver<3>::Solver
(Triangulation<3>  &,
 Mapping<3>        &,
 const SolverParameters &);

template void Solver<2>::output_results(const unsigned int ) const;
template void Solver<3>::output_results(const unsigned int ) const;

template class Solver<2>;
template class Solver<3>;

}  // namespace Advection
