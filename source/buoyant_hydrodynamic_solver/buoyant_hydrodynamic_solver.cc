/*
 * setup.cc
 *
 *  Created on: Aug 31, 2021
 *      Author: sg
 */

#include <deal.II/numerics/data_out.h>

#include <buoyant_hydrodynamic_solver.h>
#include <hydrodynamic_postprocessor.h>

#include <filesystem>
#include <fstream>
#include <string>

namespace BuoyantHydrodynamic {

template <int dim>
Solver<dim>::Solver
(Triangulation<dim>     &tria,
 Mapping<dim>           &mapping,
 const Hydrodynamic::SolverParameters &parameters,
 const double           reynolds,
 const double           froude,
 const double           stratification)
:
Hydrodynamic::Solver<dim>(tria, mapping, parameters, reynolds, froude),
density_boundary_conditions(this->triangulation),
reference_density_ptr(nullptr),
gravity_field_ptr(nullptr),
stratification_number(stratification),
density_fe_degree(1)
{}



template<int dim>
void Solver<dim>::output_results(const unsigned int cycle) const
{
  if (this->verbose)
    std::cout << "    Output results..." << std::endl;

  Hydrodynamic::Postprocessor<dim>  postprocessor(0, dim);

  // prepare data out object
  DataOut<dim, DoFHandler<dim>>    data_out;
  data_out.attach_dof_handler(this->dof_handler);
  data_out.add_data_vector(this->present_solution, postprocessor);

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

// explicit instantiation
template Solver<2>::Solver
(Triangulation<2>  &,
 Mapping<2>        &,
 const Hydrodynamic::SolverParameters &,
 const double       ,
 const double       ,
 const double        );
template Solver<3>::Solver
(Triangulation<3>  &,
 Mapping<3>        &,
 const Hydrodynamic::SolverParameters &,
 const double       ,
 const double       ,
 const double        );

template void Solver<2>::output_results(const unsigned int ) const;
template void Solver<3>::output_results(const unsigned int ) const;

template class Solver<2>;
template class Solver<3>;

}  // namespace BuoyantHydrodynamic
