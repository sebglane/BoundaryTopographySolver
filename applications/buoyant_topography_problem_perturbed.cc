/*
 * buoyant_topography_problem_perturbed.cc
 *
 *  Created on: Sep 26, 2021
 *      Author: sg
 */
#include <deal.II/base/function_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/grid/grid_tools.h>

#include <buoyant_hydrodynamic_problem.h>
#include <evaluation_boundary_traction.h>
#include <evaluation_stabilization.h>
#include <grid_factory.h>

namespace TopographyProblem {

using namespace BuoyantHydrodynamic;

template <int dim>
class ReferenceDensity : public Function<dim>
{
public:
  ReferenceDensity();

  virtual Tensor<1, dim> gradient(const Point<dim>   &point,
                                  const unsigned int  component = 0) const;

  virtual void gradient_list(const std::vector<Point<dim>> &points,
                             std::vector<Tensor<1, dim>>   &gradients,
                             const unsigned int    component = 0) const;
};



template <int dim>
ReferenceDensity<dim>::ReferenceDensity()
:
Function<dim>(1)
{}




template <int dim>
Tensor<1, dim> ReferenceDensity<dim>::gradient
(const Point<dim>   &/* point */,
 const unsigned int  /* component */) const
{
  Tensor<1, dim> gradient_value;
  gradient_value[dim-1] = -1.0;

  return gradient_value;
}



template <int dim>
void ReferenceDensity<dim>::gradient_list
(const std::vector<Point<dim>> &/* points */,
 std::vector<Tensor<1, dim>>   &gradients,
 const unsigned int /* component */) const
{
  Tensor<1, dim> gradient_value;
  gradient_value[dim-1] = -1.0;

  for (auto &gradient: gradients)
    gradient = gradient_value;

  return;
}



template <int dim>
class Problem : public BuoyantHydrodynamicProblem<dim>
{
public:
  Problem(ProblemParameters &parameters);

protected:
  virtual void make_grid() override;

  virtual void set_background_velocity() override;

  virtual void set_boundary_conditions() override;

  virtual void set_gravity_field() override;

  virtual void set_reference_density() override;

  virtual void set_postprocessor() override;

private:
  Hydrodynamic::
  EvaluationBoundaryTraction<dim> traction_evaluation;

  EvaluationStabilization<dim>    stabilization_evaluation;

  const ConstantTensorFunction<1, dim>  gravity_field;

  const ConstantTensorFunction<1, dim>  background_velocity;

  const ReferenceDensity<dim> reference_density;

  types::boundary_id  left_bndry_id;
  types::boundary_id  right_bndry_id;
  types::boundary_id  bottom_bndry_id;
  types::boundary_id  top_bndry_id;
  types::boundary_id  topographic_bndry_id;
  types::boundary_id  back_bndry_id;
  types::boundary_id  front_bndry_id;
};



template <>
Problem<2>::Problem(ProblemParameters &parameters)
:
BuoyantHydrodynamicProblem<2>(parameters),
traction_evaluation(0, 2, parameters.reynolds_number),
stabilization_evaluation(parameters.graphical_output_directory,
                         parameters.stabilization,
                         0,
                         2,
                         2 + 1,
                         parameters.reynolds_number,
                         parameters.stratification_number,
                         parameters.viscous_term_weak_form == Hydrodynamic::ViscousTermWeakForm::stress,
                         parameters.froude_number,
                         parameters.rossby_number),
gravity_field(Tensor<1, 2>({0.0, -1.0})),
background_velocity(Tensor<1, 2>({1.0, 0.0})),
reference_density(),
left_bndry_id(numbers::invalid_boundary_id),
right_bndry_id(numbers::invalid_boundary_id),
bottom_bndry_id(numbers::invalid_boundary_id),
top_bndry_id(numbers::invalid_boundary_id),
topographic_bndry_id(numbers::invalid_boundary_id),
back_bndry_id(numbers::invalid_boundary_id),
front_bndry_id(numbers::invalid_boundary_id)
{
  std::cout << "Solving perturbed buoyant topography problem" << std::endl;

  Point<2> point;
  Assert(reference_density.gradient(point) * gravity_field.value(point) >= 0.0,
         ExcMessage("Density gradient and gravity field are not co-linear."));

  stabilization_evaluation.set_stabilization_parameters(parameters.c, parameters.mu, parameters.c_density);
  stabilization_evaluation.set_background_velocity(background_velocity);
  stabilization_evaluation.set_gravity_field(gravity_field);
  stabilization_evaluation.set_reference_density(reference_density);
}



template <>
Problem<3>::Problem(ProblemParameters &parameters)
:
BuoyantHydrodynamicProblem<3>(parameters),
traction_evaluation(0, 3, parameters.reynolds_number),
stabilization_evaluation(parameters.graphical_output_directory,
                         parameters.stabilization,
                         0,
                         3,
                         3 + 1,
                         parameters.reynolds_number,
                         parameters.stratification_number,
                         parameters.viscous_term_weak_form == Hydrodynamic::ViscousTermWeakForm::stress,
                         parameters.froude_number,
                         parameters.rossby_number),
gravity_field(Tensor<1, 3>({0.0, 0.0, -1.0})),
background_velocity(Tensor<1, 3>({1.0, 0.0, 0.0})),
reference_density(),
left_bndry_id(numbers::invalid_boundary_id),
right_bndry_id(numbers::invalid_boundary_id),
bottom_bndry_id(numbers::invalid_boundary_id),
top_bndry_id(numbers::invalid_boundary_id),
topographic_bndry_id(numbers::invalid_boundary_id),
back_bndry_id(numbers::invalid_boundary_id),
front_bndry_id(numbers::invalid_boundary_id)
{
  std::cout << "Solving perturbed buoyant topography problem" << std::endl;

  Point<3> point;
  Assert(reference_density.gradient(point) * gravity_field.value(point) >= 0.0,
         ExcMessage("Density gradient and gravity field are not co-linear."));

  stabilization_evaluation.set_background_velocity(background_velocity);
  stabilization_evaluation.set_gravity_field(gravity_field);
  stabilization_evaluation.set_reference_density(reference_density);
}



template <int dim>
void Problem<dim>::set_background_velocity()
{
  this->solver.set_background_velocity(background_velocity);
}



template <int dim>
void Problem<dim>::set_gravity_field()
{
  this->solver.set_gravity_field(gravity_field);
}



template <int dim>
void Problem<dim>::set_reference_density()
{
  this->solver.set_reference_density(reference_density);
}



template <int dim>
void Problem<dim>::set_postprocessor()
{
  this->solver.add_postprocessor(traction_evaluation);
  this->solver.add_postprocessor(stabilization_evaluation);
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
  traction_evaluation.set_boundary_id(topographic_bndry_id);

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
  ScalarBoundaryConditions<dim> &density_bcs = this->solver.get_density_bcs();

  velocity_bcs.clear();
  pressure_bcs.clear();
  density_bcs.clear();

  velocity_bcs.extract_boundary_ids();
  pressure_bcs.extract_boundary_ids();
  density_bcs.extract_boundary_ids();

  velocity_bcs.set_periodic_bc(left_bndry_id, right_bndry_id, 0);
  pressure_bcs.set_periodic_bc(left_bndry_id, right_bndry_id, 0);
  density_bcs.set_periodic_bc(left_bndry_id, right_bndry_id, 0);

  const std::shared_ptr<Function<dim>> bottom_bc_fun =
      std::make_shared<Functions::ZeroFunction<dim>>(dim);
  std::vector<double> value(dim);
  value[0] = -1.0;
  const std::shared_ptr<Function<dim>> topographic_bc_fun =
      std::make_shared<Functions::ConstantFunction<dim>>(value);

  if (dim == 2)
  {
    velocity_bcs.set_dirichlet_bc(bottom_bndry_id, bottom_bc_fun);
    velocity_bcs.set_normal_flux_bc(topographic_bndry_id, topographic_bc_fun);

    pressure_bcs.set_dirichlet_bc(bottom_bndry_id);

    density_bcs.set_dirichlet_bc(bottom_bndry_id);
  }
  else if (dim == 3)
  {
    velocity_bcs.set_periodic_bc(bottom_bndry_id, top_bndry_id, 1);
    pressure_bcs.set_periodic_bc(bottom_bndry_id, top_bndry_id, 1);
    density_bcs.set_periodic_bc(bottom_bndry_id, top_bndry_id, 1);

    velocity_bcs.set_dirichlet_bc(back_bndry_id, bottom_bc_fun);
    velocity_bcs.set_normal_flux_bc(topographic_bndry_id, topographic_bc_fun);

    pressure_bcs.set_dirichlet_bc(back_bndry_id);

    density_bcs.set_dirichlet_bc(back_bndry_id);
  }

  velocity_bcs.close();
  pressure_bcs.close();
  density_bcs.close();
}

}  // namespace TopographyProblem

int main(int argc, char *argv[])
{
  try
  {
    std::string parameter_filename;
    if (argc >= 2)
      parameter_filename = argv[1];
    else
      parameter_filename = "buoyant_topography_problem_perturbed.prm";

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
