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
Advection::SolverParameters()
{}



void SolverParameters::declare_parameters(ParameterHandler &prm)
{
  Hydrodynamic::SolverParameters::declare_parameters(prm);
  Advection::SolverParameters::declare_parameters(prm);
}



void SolverParameters::parse_parameters(ParameterHandler &prm)
{
  Hydrodynamic::SolverParameters::parse_parameters(prm);
  Advection::SolverParameters::parse_parameters(prm);
}



template<typename Stream>
Stream& operator<<(Stream &stream, const SolverParameters &prm)
{
  stream << static_cast<const Hydrodynamic::SolverParameters &>(prm);
  stream << static_cast<const Advection::SolverParameters &>(prm);
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
Base::Solver<dim, TriangulationType>(tria, mapping, parameters),
Hydrodynamic::Solver<dim, TriangulationType>(tria, mapping, parameters, reynolds, froude, rossby),
Advection::Solver<dim, TriangulationType>(tria, mapping, parameters, stratification),
gravity_field_ptr()
{}



template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::output_results(const unsigned int cycle) const
{
  if (this->verbose)
    this->pcout << "    Output results..." << std::endl;

  Hydrodynamic::Postprocessor<dim>  postprocessor(this->velocity_fe_index,
                                                  this->pressure_fe_index);

  Utility::PostprocessorScalarField<dim>  density_postprocessor("density", this->scalar_fe_index);

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
