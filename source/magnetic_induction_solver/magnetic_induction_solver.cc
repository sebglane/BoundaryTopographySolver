/*
 * magnetic_induction_solver.cc
 *
 *  Created on: Aug 22, 2022
 *      Author: sg
 */

#include <magnetic_induction_solver.h>
#include <magnetic_postprocessor.h>

#include <deal.II/numerics/data_out.h>

#include <filesystem>
#include <fstream>
#include <string>

namespace MagneticInduction {

SolverParameters::SolverParameters()
:
Base::Parameters(),
c(1.0)
{}



void SolverParameters::declare_parameters(ParameterHandler &prm)
{
  Base::Parameters::declare_parameters(prm);

  prm.enter_subsection("Magnetic induction solver parameters");
  {
    prm.declare_entry("Stabilization parameter",
                      "1.0",
                      Patterns::Double(std::numeric_limits<double>::epsilon()));
  }
  prm.leave_subsection();
}



void SolverParameters::parse_parameters(ParameterHandler &prm)
{
  Base::Parameters::parse_parameters(prm);

  prm.enter_subsection("Magnetic induction solver parameters");
  {
    c = prm.get_double("Stabilization parameter");
    AssertIsFinite(c);
    Assert(c > 0.0, ExcLowerRangeType<double>(c, 0.0));
  }
  prm.leave_subsection();
}



template<typename Stream>
Stream& operator<<(Stream &stream, const SolverParameters &prm)
{
  stream << static_cast<const Base::Parameters &>(prm);

  Utility::add_header(stream);
  Utility::add_line(stream, "Magnetic induction solver parameters");
  Utility::add_header(stream);

  Utility::add_line(stream, "Stabilization parameter", prm.c);

  return (stream);
}



template <int dim, typename TriangulationType>
Solver<dim, TriangulationType>::Solver
(TriangulationType      &tria,
 Mapping<dim>           &mapping,
 const SolverParameters &parameters,
 const double           magnetic_reynolds_number)
:
Base::Solver<dim, TriangulationType>(tria, mapping, parameters),
magnetic_field_boundary_conditions(this->triangulation),
magnetic_pressure_boundary_conditions(this->triangulation),
background_magnetic_field_ptr(),
velocity_field_ptr(),
magnetic_fe_degree(2),
magnetic_reynolds_number(magnetic_reynolds_number),
c(parameters.c),
magnetic_field_fe_index(numbers::invalid_unsigned_int),
magnetic_pressure_fe_index(numbers::invalid_unsigned_int),
magnetic_field_block_index(numbers::invalid_unsigned_int),
magnetic_pressure_block_index(numbers::invalid_unsigned_int)
{}



template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::output_results(const unsigned int cycle) const
{
  if (this->verbose)
    this->pcout << "    Output results..." << std::endl;

  Postprocessor<dim>  postprocessor(magnetic_field_fe_index,
                                    magnetic_pressure_fe_index);

  if (background_magnetic_field_ptr)
    postprocessor.set_background_magnetic_field(background_magnetic_field_ptr);

  // prepare data out object
  DataOut<dim>  data_out;
  data_out.attach_dof_handler(this->dof_handler);
  data_out.add_data_vector(this->present_solution,
                           postprocessor);

  data_out.build_patches(magnetic_fe_degree);

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
(Triangulation<2>  &,
 Mapping<2>        &,
 const SolverParameters &,
 const double        );
template Solver<3>::Solver
(Triangulation<3>  &,
 Mapping<3>        &,
 const SolverParameters &,
 const double        );

template void Solver<2>::output_results(const unsigned int ) const;
template void Solver<3>::output_results(const unsigned int ) const;

template class Solver<2>;
template class Solver<3>;

}  // namespace MagneticInduction
