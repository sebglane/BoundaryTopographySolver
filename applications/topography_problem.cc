/*
 * topography_problem.cc
 *
 *  Created on: Sep 1, 2021
 *      Author: sg
 */

#include <deal.II/base/function_lib.h>
#include <deal.II/base/mpi.h>
#include <deal.II/grid/grid_tools.h>

#include <evaluation_boundary_traction.h>
#include <evaluation_stabilization.h>
#include <grid_factory.h>
#include <hydrodynamic_problem.h>

#include <memory>

namespace TopographyProblem {

using namespace Hydrodynamic;

template <int dim>
class Problem : public HydrodynamicProblem<dim>
{
public:
  Problem(ProblemParameters &parameters);

protected:
  virtual void make_grid() override;

  virtual void set_boundary_conditions() override;

  virtual void set_postprocessor() override;

private:
  std::shared_ptr<EvaluationBoundaryTraction<dim>> traction_evaluation_ptr;

  std::shared_ptr<EvaluationStabilization<dim>>    stabilization_evaluation_ptr;

  types::boundary_id  left_bndry_id;
  types::boundary_id  right_bndry_id;
  types::boundary_id  bottom_bndry_id;
  types::boundary_id  top_bndry_id;
  types::boundary_id  topographic_bndry_id;
  types::boundary_id  back_bndry_id;
  types::boundary_id  front_bndry_id;

};



template <int dim>
Problem<dim>::Problem(ProblemParameters &parameters)
:
HydrodynamicProblem<dim>(parameters),
traction_evaluation_ptr(
new EvaluationBoundaryTraction<dim>{0, dim, parameters.reynolds_number}),
stabilization_evaluation_ptr(
new EvaluationStabilization<dim>{parameters.graphical_output_directory,
                                 parameters.stabilization,
                                 0,
                                 dim,
                                 parameters.reynolds_number,
                                 parameters.viscous_term_weak_form == Hydrodynamic::ViscousTermWeakForm::stress}),
left_bndry_id(numbers::invalid_boundary_id),
right_bndry_id(numbers::invalid_boundary_id),
bottom_bndry_id(numbers::invalid_boundary_id),
top_bndry_id(numbers::invalid_boundary_id),
topographic_bndry_id(numbers::invalid_boundary_id),
back_bndry_id(numbers::invalid_boundary_id),
front_bndry_id(numbers::invalid_boundary_id)
{
  std::cout << "Solving viscous topography problem" << std::endl;

  stabilization_evaluation_ptr->set_stabilization_parameters(parameters.c, parameters.mu);
}



template <int dim>
void Problem<dim>::make_grid()
{
  std::cout << "    Make grid..." << std::endl;

  GridFactory::TopographyBox<dim> topography_box(2.0 * numbers::PI, 0.1);
  left_bndry_id = topography_box.left;
  right_bndry_id = topography_box.right;
  bottom_bndry_id = topography_box.bottom;
  top_bndry_id = topography_box.top;
  back_bndry_id = topography_box.back;
  front_bndry_id = topography_box.front;

  topographic_bndry_id = topography_box.topographic_boundary;
  traction_evaluation_ptr->set_boundary_id(topographic_bndry_id);

  topography_box.create_coarse_mesh(this->triangulation);

  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
  periodicity_vector;

  GridTools::collect_periodic_faces(this->triangulation,
                                    left_bndry_id,
                                    right_bndry_id,
                                    0,
                                    periodicity_vector);
  if (dim == 3)
    GridTools::collect_periodic_faces(this->triangulation,
                                      bottom_bndry_id,
                                      top_bndry_id,
                                      1,
                                      periodicity_vector);

  this->triangulation.add_periodicity(periodicity_vector);

  this->triangulation.refine_global(this->n_initial_refinements);

  // initial boundary refinements
  if (this->n_initial_bndry_refinements > 0)
  {
    for (unsigned int step=0; step<this->n_initial_bndry_refinements; ++step)
    {
      for (const auto &cell: this->triangulation.active_cell_iterators())
        if (cell->at_boundary())
          for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
            if (cell->face(f)->boundary_id() == topographic_bndry_id)
              cell->set_refine_flag();
      this->triangulation.execute_coarsening_and_refinement();
    }
  }
}



template <int dim>
void Problem<dim>::set_boundary_conditions()
{
  std::cout << "    Set boundary conditions..." << std::endl;

  VectorBoundaryConditions<dim> &velocity_bcs = this->solver.get_velocity_bcs();
  ScalarBoundaryConditions<dim> &pressure_bcs = this->solver.get_pressure_bcs();

  velocity_bcs.clear();
  pressure_bcs.clear();

  velocity_bcs.extract_boundary_ids();
  pressure_bcs.extract_boundary_ids();

  velocity_bcs.set_periodic_bc(left_bndry_id, right_bndry_id, 0);
  pressure_bcs.set_periodic_bc(left_bndry_id, right_bndry_id, 0);

  std::vector<double> value(dim);
  value[0] = 1.0;
  const std::shared_ptr<Function<dim>> velocity_function =
      std::make_shared<Functions::ConstantFunction<dim>>(value);

  if (dim == 2)
  {
    velocity_bcs.set_dirichlet_bc(bottom_bndry_id, velocity_function);
    velocity_bcs.set_normal_flux_bc(topographic_bndry_id);
  }
  else if (dim == 3)
  {
    velocity_bcs.set_periodic_bc(bottom_bndry_id, top_bndry_id, 1);
    pressure_bcs.set_periodic_bc(bottom_bndry_id, top_bndry_id, 1);

    velocity_bcs.set_dirichlet_bc(back_bndry_id, velocity_function);
    velocity_bcs.set_normal_flux_bc(topographic_bndry_id);
  }

  velocity_bcs.close();
  pressure_bcs.close();
}



template <int dim>
void Problem<dim>::set_postprocessor()
{
  this->solver.add_postprocessor(traction_evaluation_ptr);
  this->solver.add_postprocessor(stabilization_evaluation_ptr);
}

}  // namespace TopographyProblem

int main(int argc, char *argv[])
{
  try
  {
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

    std::string parameter_filename;
    if (argc >= 2)
      parameter_filename = argv[1];
    else
      parameter_filename = "topography_problem.prm";

    TopographyProblem::ProblemParameters parameters(parameter_filename);
    if (parameters.space_dim == 2)
    {
      TopographyProblem::Problem<2> problem(parameters);
      problem.run();
    }
    else
    {
      TopographyProblem::Problem<3> problem(parameters);
      problem.run();
    }
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl << std::endl
            << "----------------------------------------------------"
            << std::endl;
    std::cerr << "Exception on processing: " << std::endl
            << exc.what() << std::endl
            << "Aborting!" << std::endl
            << "----------------------------------------------------"
            << std::endl;
    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl << std::endl
            << "----------------------------------------------------"
            << std::endl;
    std::cerr << "Unknown exception!" << std::endl
            << "Aborting!" << std::endl
            << "----------------------------------------------------"
            << std::endl;
    return 1;
  }
  return 0;
}
