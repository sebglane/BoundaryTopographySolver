/*
 * buoyant_hydrodynamic_solver.cc
 *
 *  Created on: Aug 31, 2021
 *      Author: sg
 */

#include <deal.II/numerics/data_out.h>

#include <buoyant_hydrodynamic_solver.h>
#include <hydrodynamic_postprocessor.h>
#include <postprocessor_base.h>

#include <filesystem>
#include <fstream>
#include <string>

namespace BuoyantHydrodynamic {

SolverParameters::SolverParameters()
:
Hydrodynamic::SolverParameters(),
c_density(1.0),
nu_density(1e-4)
{}



void SolverParameters::declare_parameters(ParameterHandler &prm)
{
  Hydrodynamic::SolverParameters::declare_parameters(prm);

  prm.enter_subsection("Buoyant hydrodynamic solver parameters");
  {
    prm.declare_entry("SUPG density stabilization parameter",
                      "1.0",
                      Patterns::Double(std::numeric_limits<double>::epsilon()));

    prm.declare_entry("Minimal viscosity (density)",
                      "1.0e-4",
                      Patterns::Double(std::numeric_limits<double>::epsilon()));
  }
  prm.leave_subsection();
}



void SolverParameters::parse_parameters(ParameterHandler &prm)
{
  Hydrodynamic::SolverParameters::parse_parameters(prm);

  prm.enter_subsection("Buoyant hydrodynamic solver parameters");
  {
    c_density = prm.get_double("SUPG density stabilization parameter");
    AssertIsFinite(c_density);
    Assert(c_density > 0.0, ExcLowerRangeType<double>(c_density, 0.0));

    nu_density = prm.get_double("Minimal viscosity (density)");
    AssertIsFinite(nu_density);
    Assert(nu_density > 0.0, ExcLowerRangeType<double>(nu_density, 0.0));


  }
  prm.leave_subsection();
}



template<typename Stream>
Stream& operator<<(Stream &stream, const SolverParameters &prm)
{
  stream << static_cast<const Hydrodynamic::SolverParameters &>(prm);

  Utility::add_header(stream);
  Utility::add_line(stream, "Buoyant hydrodynamic solver parameters");
  Utility::add_line(stream, "SUPG density stab. parameter", prm.c_density);
  Utility::add_line(stream, "Minimal viscosity (density)", prm.nu_density);
  Utility::add_header(stream);

  return (stream);
}



template <int dim, typename TriangulationType>
Solver<dim, TriangulationType>::Solver
(TriangulationType      &tria,
 Mapping<dim>           &mapping,
 const SolverParameters &parameters,
 const double           reynolds,
 const double           froude,
 const double           stratification,
 const double           rossby)
:
Base::Solver<dim>(tria, mapping, parameters),
Hydrodynamic::Solver<dim, TriangulationType>(tria, mapping, parameters, reynolds, froude, rossby),
density_boundary_conditions(this->triangulation),
reference_density_ptr(),
gravity_field_ptr(),
stratification_number(stratification),
density_fe_degree(1),
c_density(parameters.c_density),
nu_density(parameters.nu_density)
{}



template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::output_results(const unsigned int cycle) const
{
  if (this->verbose)
    this->pcout << "    Output results..." << std::endl;

  Hydrodynamic::Postprocessor<dim>  postprocessor(0, dim);

  Utility::PostprocessorScalarField<dim>  density_postprocessor("density", dim+1);

  // prepare data out object
  DataOut<dim, DoFHandler<dim>>    data_out;
  data_out.attach_dof_handler(this->dof_handler);
  data_out.add_data_vector(this->present_solution,
                           postprocessor);
  data_out.add_data_vector(this->present_solution,
                           density_postprocessor);

  data_out.build_patches(this->velocity_fe_degree);

  // write output to disk
  const std::string filename = ("solution-" +
                                Utilities::int_to_string(cycle, 2) +
                                ".vtk");
  std::filesystem::path output_file(this->graphical_output_directory);
  output_file /= filename;

  std::ofstream fstream(output_file.c_str());
  data_out.write_vtk(fstream);
}



template <int dim, typename TriangulationType>
inline void Solver<dim, TriangulationType>::
postprocess_newton_iteration
(const unsigned int iteration,
 const bool         is_initial_cycle)
{
  if (iteration < 1 && is_initial_cycle)
  {
    this->pcout << "Reseting density solution..." << std::endl;

    this->evaluation_point.block(2) = 0.0;
    this->present_solution.block(2) = 0.0;
    this->solution_update.block(2) = 0.0;
  }
  return;
}


// explicit instantiation
template std::ostream & operator<<(std::ostream &, const SolverParameters &);
template ConditionalOStream & operator<<(ConditionalOStream &, const SolverParameters &);

template Solver<2>::Solver
(Triangulation<2>  &,
 Mapping<2>        &,
 const SolverParameters &,
 const double       ,
 const double       ,
 const double       ,
 const double        );
template Solver<3>::Solver
(Triangulation<3>  &,
 Mapping<3>        &,
 const SolverParameters &,
 const double       ,
 const double       ,
 const double       ,
 const double        );

template void Solver<2>::output_results(const unsigned int ) const;
template void Solver<3>::output_results(const unsigned int ) const;

template class Solver<2>;
template class Solver<3>;

}  // namespace BuoyantHydrodynamic
