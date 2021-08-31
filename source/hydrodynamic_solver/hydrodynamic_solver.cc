/*
 * setup.cc
 *
 *  Created on: Aug 31, 2021
 *      Author: sg
 */

#include <deal.II/numerics/data_out.h>

#include <hydrodynamic_solver.h>
#include <hydrodynamic_postprocessor.h>

#include <fstream>
#include <string>

namespace TopographyProblem {

template <int dim>
HydrodynamicSolver<dim>::HydrodynamicSolver
(const Triangulation<dim> &tria,
 const double       reynolds_number,
 const unsigned int velocity_fe_degree,
 const unsigned int n_refinements,
 const double       newton_tolerance,
 const unsigned int n_maximum_iterations)
:
SolverBase<dim>(tria, n_refinements, newton_tolerance, n_maximum_iterations),
velocity_fe_degree(velocity_fe_degree),
reynolds_number(reynolds_number)
{}



template<int dim>
void HydrodynamicSolver<dim>::output_results(const unsigned int cycle) const
{
  std::cout << "   Output results..." << std::endl;

  HydrodynamicPostprocessor<dim>  postprocessor(0, dim);

  // prepare data out object
  DataOut<dim, DoFHandler<dim>>    data_out;
  data_out.attach_dof_handler(this->dof_handler);
  data_out.add_data_vector(this->present_solution, postprocessor);

  data_out.build_patches(velocity_fe_degree);

  // write output to disk
  const std::string filename = ("solution-" +
                                Utilities::int_to_string(cycle, 2) +
                                ".vtk");
  std::ofstream output(filename.c_str());
  data_out.write_vtk(output);
}

// explicit instantiation
template HydrodynamicSolver<2>::HydrodynamicSolver
(const Triangulation<2>  &,
 const double       ,
 const unsigned int ,
 const unsigned int ,
 const double       ,
 const unsigned int );
template HydrodynamicSolver<3>::HydrodynamicSolver
(const Triangulation<3>  &,
 const double       ,
 const unsigned int ,
 const unsigned int ,
 const double       ,
 const unsigned int );

template void HydrodynamicSolver<2>::output_results(const unsigned int ) const;
template void HydrodynamicSolver<3>::output_results(const unsigned int ) const;

template class HydrodynamicSolver<2>;
template class HydrodynamicSolver<3>;

}  // namespace TopographyProblem
