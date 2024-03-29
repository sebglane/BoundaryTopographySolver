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
Base::Parameters(),
c(0.1),
nu(1.0e-4)
{}



void SolverParameters::declare_parameters(ParameterHandler &prm)
{
  Base::Parameters::declare_parameters(prm);

  prm.enter_subsection("Advection solver parameters");
  {
    prm.declare_entry("SUPG stabilization coefficient",
                      "0.1",
                      Patterns::Double(0.0));

    prm.declare_entry("Minimal viscosity",
                      "1.0e-4",
                      Patterns::Double(std::numeric_limits<double>::epsilon()));
  }
  prm.leave_subsection();
}



void SolverParameters::parse_parameters(ParameterHandler &prm)
{
  Base::Parameters::parse_parameters(prm);

  prm.enter_subsection("Advection solver parameters");
  {
    c = prm.get_double("SUPG stabilization coefficient");
    AssertThrow(c > 0.0, ExcLowerRangeType<double>(0.0, c));

    nu = prm.get_double("Minimal viscosity");
    AssertIsFinite(nu);
    Assert(nu > 0.0, ExcLowerRangeType<double>(nu, 0.0));
  }
  prm.leave_subsection();
}



template<typename Stream>
Stream& operator<<(Stream &stream, const SolverParameters &prm)
{
  stream << static_cast<const Base::Parameters &>(prm);

  Utility::add_header(stream);
  Utility::add_line(stream, "Advection solver parameters");
  Utility::add_header(stream);

  Utility::add_line(stream, "SUPG stabilization coeff.", prm.c);
  Utility::add_line(stream, "Minimal viscosity", prm.nu);

  return (stream);
}



template <int dim, typename TriangulationType>
Solver<dim, TriangulationType>::Solver
(TriangulationType       &tria,
 Mapping<dim>            &mapping,
 const SolverParameters  &parameters,
 const double             gradient_scaling_number)
:
Base::Solver<dim, TriangulationType>(tria, mapping, parameters),
scalar_boundary_conditions(this->triangulation),
advection_field_ptr(),
reference_field_ptr(),
source_term_ptr(),
gradient_scaling_number(gradient_scaling_number),
scalar_fe_degree(1),
c(parameters.c),
nu(parameters.nu),
scalar_fe_index(numbers::invalid_unsigned_int),
scalar_block_index(numbers::invalid_unsigned_int)
{}



template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::output_results(const unsigned int cycle) const
{
  if (this->verbose)
    this->pcout << "    Output results..." << std::endl;

  // prepare data out object
  DataOut<dim>  data_out;
  data_out.attach_dof_handler(this->dof_handler);
  data_out.add_data_vector(this->present_solution, "field");

  data_out.build_patches(scalar_fe_degree);

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
template ConditionalOStream & operator<<(ConditionalOStream &, const SolverParameters &);

template Solver<2>::Solver
(Triangulation<2>       &,
 Mapping<2>             &,
 const SolverParameters &,
 const double             );
template Solver<3>::Solver
(Triangulation<3>       &,
 Mapping<3>             &,
 const SolverParameters &,
 const double             );

template void Solver<2>::output_results(const unsigned int ) const;
template void Solver<3>::output_results(const unsigned int ) const;

template class Solver<2>;
template class Solver<3>;

}  // namespace Advection
