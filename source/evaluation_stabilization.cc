/*
 * evaluation_stabilization.cc
 *
 *  Created on: Sep 27, 2021
 *      Author: sg
 */
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/fe/fe_values.h>

#include <assembly_functions.h>
#include <buoyant_hydrodynamic_problem.h>
#include <hydrodynamic_problem.h>
#include <evaluation_stabilization.h>
#include <hydrodynamic_solver.h>


namespace Hydrodynamic {

using namespace dealii;

template <int dim>
EvaluationStabilization<dim>::EvaluationStabilization
(const std::filesystem::path &output_directory,
 const StabilizationFlags &stabilization_flags,
 const unsigned int velocity_start_index,
 const unsigned int pressure_index,
 const double reynolds_number,
 const bool   use_stress_form,
 const double froude_number,
 const double rossby_number,
 const bool   print_table)
:
angular_velocity_ptr(),
body_force_ptr(),
background_velocity_ptr(),
stabilization(stabilization_flags),
velocity_start_index(velocity_start_index),
pressure_index(pressure_index),
reynolds_number(reynolds_number),
froude_number(froude_number),
rossby_number(rossby_number),
use_stress_form(use_stress_form),
print_table(print_table),
c(std::numeric_limits<double>::min()),
mu(std::numeric_limits<double>::min()),
output_directory(output_directory)
{
  data_table.declare_column("cycle");

  data_table.declare_column("max point visc.");
  data_table.declare_column("max cell visc.");
  data_table.declare_column("mean visc.");
  data_table.set_scientific("max point visc.", true);
  data_table.set_scientific("max cell visc.", true);
  data_table.set_scientific("mean visc.", true);

  data_table.declare_column("max momentum point res.");
  data_table.declare_column("max momentum cell res.");
  data_table.declare_column("mean momentum res.");
  data_table.set_scientific("max momentum point res.", true);
  data_table.set_scientific("max momentum cell res.", true);
  data_table.set_scientific("mean momentum res.", true);

  data_table.declare_column("max mass point res.");
  data_table.declare_column("max mass cell res.");
  data_table.declare_column("mean mass res.");
  data_table.set_scientific("max mass point res.", true);
  data_table.set_scientific("max mass cell res.", true);
  data_table.set_scientific("mean mass res.", true);
}



template <int dim>
EvaluationStabilization<dim>::~EvaluationStabilization()
{
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::cout << std::endl;

    std::filesystem::path output_file(output_directory);
    output_file /= "Stabilization.txt";

    std::ofstream fstream(output_file.c_str());
    data_table.write_text(fstream, TableHandler::TextOutputFormat::org_mode_table);

    if (print_table)
      data_table.write_text(std::cout);
  }
}



template <int dim>
void EvaluationStabilization<dim>::operator()
(const Mapping<dim>        &mapping,
 const FiniteElement<dim>  &fe,
 const DoFHandler<dim>     &dof_handler,
 const Vector<double>      &solution)
{
  evaluate(mapping, fe, dof_handler, solution);
}



template <int dim>
void EvaluationStabilization<dim>::operator()
(const Mapping<dim>        &mapping,
 const FiniteElement<dim>  &fe,
 const DoFHandler<dim>     &dof_handler,
 const BlockVector<double> &solution)
{
  evaluate(mapping, fe, dof_handler, solution);
}



template <int dim>
void EvaluationStabilization<dim>::operator()
(const Mapping<dim>        &mapping,
 const FiniteElement<dim>  &fe,
 const DoFHandler<dim>     &dof_handler,
 const TrilinosWrappers::MPI::Vector  &solution)
{
  evaluate(mapping, fe, dof_handler, solution);
}



template <int dim>
template <typename VectorType>
void EvaluationStabilization<dim>::evaluate
(const Mapping<dim>        &mapping,
 const FiniteElement<dim>  &fe,
 const DoFHandler<dim>     &dof_handler,
 const VectorType          &solution)
{
  AssertThrow(c > std::numeric_limits<double>::min(), ExcInternalError());
  AssertThrow(mu > std::numeric_limits<double>::min(), ExcInternalError());

  if (angular_velocity_ptr)
    AssertThrow(rossby_number > 0.0,
                ExcMessage("Non-vanishing Rossby number is required if the angular "
                           "velocity vector is specified."));
  if (body_force_ptr)
    AssertThrow(froude_number > 0.0,
                ExcMessage("Non-vanishing Froude number is required if the body "
                           "force is specified."));
  AssertThrow(reynolds_number != 0.0,
              ExcMessage("The Reynolds must not vanish (stabilization is not "
                         "implemented yet)."));

  QGauss<dim>   quadrature_formula(fe.degree + 1);

  UpdateFlags update_flags{update_values|
                           update_gradients|
                           update_hessians|
                           update_JxW_values};
  if (angular_velocity_ptr || background_velocity_ptr)
    update_flags |= update_quadrature_points;

  using Scratch = AssemblyData::RightHandSide::Scratch<dim>;

  Scratch scratch(mapping,
                  quadrature_formula,
                  fe,
                  update_flags,
                  Quadrature<dim-1>(1),
                  update_default,
                  stabilization,
                  use_stress_form,
                  background_velocity_ptr != nullptr,
                  body_force_ptr != nullptr);

  const FEValuesExtractors::Vector  velocity(velocity_start_index);
  const FEValuesExtractors::Scalar  pressure(pressure_index);

  const double nu{1.0 / reynolds_number};

  OptionalArgumentsStrongForm<dim> &strong_form_options = scratch.hydrodynamic_strong_form_options;
  strong_form_options.use_stress_form = use_stress_form;

  // Coriolis term
  if (angular_velocity_ptr)
  {
    strong_form_options.angular_velocity = angular_velocity_ptr->value();
    strong_form_options.rossby_number = rossby_number;
  }

  double cell_momentum_residual;
  double mean_momentum_residual{0.0};
  double max_momentum_residual[2]{std::numeric_limits<double>::min(),
                                  std::numeric_limits<double>::min()};

  double cell_mass_residual;
  double mean_mass_residual{0.0};
  double max_mass_residual[2]{std::numeric_limits<double>::min(),
                              std::numeric_limits<double>::min()};

  double mean_momentum_viscosity{0.0};
  double max_momentum_viscosity[2]{std::numeric_limits<double>::min(),
                                   std::numeric_limits<double>::min()};

  double cell_volume;
  double volume{0};

  for (const auto &cell: dof_handler.active_cell_iterators())
  if (cell->is_locally_owned())
  {
    scratch.fe_values.reinit(cell);

    const double delta{c * std::pow(cell->diameter(), 2)};

    // solution values
    scratch.fe_values[velocity].get_function_values(solution,
                                                    scratch.present_velocity_values);
    scratch.fe_values[velocity].get_function_gradients(solution,
                                                       scratch.present_velocity_gradients);

    scratch.fe_values[pressure].get_function_values(solution,
                                                    scratch.present_pressure_values);

    // stress form
    if (use_stress_form)
      scratch.fe_values[velocity].get_function_symmetric_gradients(solution,
                                                                   scratch.present_sym_velocity_gradients);

    // stabilization related solution values
    if (stabilization & (apply_supg|apply_pspg))
    {
      scratch.fe_values[velocity].get_function_laplacians(solution,
                                                          scratch.present_velocity_laplaceans);
      scratch.fe_values[pressure].get_function_gradients(solution,
                                                         scratch.present_pressure_gradients);

      // stress form
      if (use_stress_form)
      {
        std::vector<Tensor<3, dim>> present_hessians(scratch.n_q_points);
        scratch.fe_values[velocity].get_function_hessians(solution,
                                                          present_hessians);

        std::vector<Tensor<1, dim>> &present_velocity_grad_divergences =
            strong_form_options.present_velocity_grad_divergences.value();
        for (std::size_t q=0; q<present_hessians.size(); ++q)
        {
          present_velocity_grad_divergences[q] = 0;
          for (unsigned int d=0; d<dim; ++d)
            present_velocity_grad_divergences[q] += present_hessians[q][d][d];
        }
      }
    }

    // body force
    if (body_force_ptr)
    {
      body_force_ptr->value_list(scratch.fe_values.get_quadrature_points(),
                                 *strong_form_options.body_force_values);
      strong_form_options.froude_number = froude_number;
    }

    // background field
    if (background_velocity_ptr)
    {
      background_velocity_ptr->value_list(scratch.fe_values.get_quadrature_points(),
                                          *strong_form_options.background_velocity_values);
      background_velocity_ptr->gradient_list(scratch.fe_values.get_quadrature_points(),
                                             *strong_form_options.background_velocity_gradients);
    }

    // stabilization
    if (stabilization & (apply_supg|apply_pspg))
      compute_strong_residual(scratch.present_velocity_values,
                              scratch.present_velocity_gradients,
                              scratch.present_velocity_laplaceans,
                              scratch.present_pressure_gradients,
                              scratch.present_strong_residuals,
                              nu,
                              strong_form_options);

    cell_momentum_residual = 0;
    cell_mass_residual = 0;
    cell_volume = 0;

    for (const auto q: scratch.fe_values.quadrature_point_indices())
    {
      const double JxW{scratch.fe_values.JxW(q)};

      const double mass_residual{trace(scratch.present_velocity_gradients[q])};
      max_mass_residual[0] = std::max(std::abs(mass_residual), max_mass_residual[0]);
      cell_mass_residual += mass_residual * JxW;

      cell_volume += JxW;
      volume += JxW;

      if (this->stabilization & (apply_supg|apply_pspg))
      {
        const double momentum_residual{scratch.present_strong_residuals[q].norm()};

        max_momentum_residual[0] = std::max(momentum_residual,
                                            max_momentum_residual[0]);
        max_momentum_viscosity[0] = std::max(delta * momentum_residual,
                                             max_momentum_viscosity[0]);
        cell_momentum_residual += momentum_residual * JxW;
      }
    } // end loop over cell quadrature points

    max_momentum_residual[1] = std::max(cell_momentum_residual / cell_volume,
                                        max_momentum_residual[1]);
    max_mass_residual[1] = std::max(cell_mass_residual / cell_volume,
                                    max_mass_residual[1]);
    max_momentum_viscosity[1] = std::max(delta * cell_momentum_residual / cell_volume,
                                         max_momentum_viscosity[1]);

    mean_momentum_residual += cell_momentum_residual;
    mean_mass_residual += cell_mass_residual;
    mean_momentum_viscosity += delta * cell_momentum_residual;

  } // end loop over active cells

  // compute mean value
  Assert(volume > 0.0, ExcLowerRangeType<double>(0.0, volume));

  Utilities::MPI::max(max_momentum_residual, MPI_COMM_WORLD, max_momentum_residual);
  Utilities::MPI::max(max_momentum_viscosity, MPI_COMM_WORLD, max_momentum_viscosity);
  Utilities::MPI::max(max_mass_residual, MPI_COMM_WORLD, max_mass_residual);

  volume = Utilities::MPI::sum(volume, MPI_COMM_WORLD);
  mean_momentum_residual = Utilities::MPI::sum(mean_momentum_residual, MPI_COMM_WORLD);
  mean_mass_residual = Utilities::MPI::sum(mean_mass_residual, MPI_COMM_WORLD);

  mean_momentum_residual /= volume;
  mean_mass_residual /= volume;
  mean_momentum_viscosity /= volume;

  data_table.add_value("cycle", this->cycle);

  data_table.add_value("max point visc.", max_momentum_viscosity[0]);
  data_table.add_value("max cell visc.", max_momentum_viscosity[1]);
  data_table.add_value("mean visc.", mean_momentum_viscosity);

  data_table.add_value("max momentum point res.", max_momentum_residual[0]);
  data_table.add_value("max momentum cell res.", max_momentum_residual[1]);
  data_table.add_value("mean momentum res.", mean_momentum_residual);

  data_table.add_value("max mass point res.", max_mass_residual[0]);
  data_table.add_value("max mass cell res.", max_mass_residual[1]);
  data_table.add_value("mean mass res.", mean_mass_residual);
}

// explicit instantiations
template class EvaluationStabilization<2>;
template class EvaluationStabilization<3>;

}  // namespace Hydrodynamic



namespace BuoyantHydrodynamic {

using namespace dealii;

template <int dim>
EvaluationStabilization<dim>::EvaluationStabilization
(const std::filesystem::path &output_directory,
 const StabilizationFlags &stabilization_flags,
 const unsigned int velocity_start_index,
 const unsigned int pressure_index,
 const unsigned int density_index,
 const double reynolds_number,
 const double stratification_number,
 const bool   use_stress_form,
 const double froude_number,
 const double rossby_number,
 const bool   print_table)
:
Hydrodynamic::EvaluationStabilization<dim>(output_directory,
                                           stabilization_flags,
                                           velocity_start_index,
                                           pressure_index,
                                           reynolds_number,
                                           use_stress_form,
                                           froude_number,
                                           rossby_number,
                                           print_table),
reference_density_ptr(),
gravity_field_ptr(),
density_index(density_index),
stratification_number(stratification_number),
c_density(std::numeric_limits<double>::min())
{
  this->data_table.declare_column("max density point visc.");
  this->data_table.declare_column("max density cell visc.");
  this->data_table.declare_column("mean density visc.");
  this->data_table.set_scientific("max density point visc.", true);
  this->data_table.set_scientific("max density cell visc.", true);
  this->data_table.set_scientific("mean density visc.", true);

  this->data_table.declare_column("max density point res.");
  this->data_table.declare_column("max density cell res.");
  this->data_table.declare_column("mean density res.");
  this->data_table.set_scientific("max density point res.", true);
  this->data_table.set_scientific("max density cell res.", true);
  this->data_table.set_scientific("mean density res.", true);
}



template <int dim>
void EvaluationStabilization<dim>::operator()
(const Mapping<dim>        &mapping,
 const FiniteElement<dim>  &fe,
 const DoFHandler<dim>     &dof_handler,
 const Vector<double>      &solution)
{
  evaluate(mapping, fe, dof_handler, solution);
}



template <int dim>
void EvaluationStabilization<dim>::operator()
(const Mapping<dim>        &mapping,
 const FiniteElement<dim>  &fe,
 const DoFHandler<dim>     &dof_handler,
 const BlockVector<double> &solution)
{
  evaluate(mapping, fe, dof_handler, solution);
}



template <int dim>
void EvaluationStabilization<dim>::operator()
(const Mapping<dim>        &mapping,
 const FiniteElement<dim>  &fe,
 const DoFHandler<dim>     &dof_handler,
 const TrilinosWrappers::MPI::Vector  &solution)
{
  evaluate(mapping, fe, dof_handler, solution);
}



template <int dim>
template <typename VectorType>
void EvaluationStabilization<dim>::evaluate
(const Mapping<dim>        &mapping,
 const FiniteElement<dim>  &fe,
 const DoFHandler<dim>     &dof_handler,
 const VectorType          &solution)
{
  AssertThrow(this->c > std::numeric_limits<double>::min(), ExcInternalError());
  AssertThrow(this->mu > std::numeric_limits<double>::min(), ExcInternalError());
  AssertThrow(c_density > std::numeric_limits<double>::min(), ExcInternalError());

  if (this->angular_velocity_ptr)
    AssertThrow(this->rossby_number > 0.0,
                ExcMessage("Non-vanishing Rossby number is required if the angular "
                           "velocity vector is specified."));
  if (this->body_force_ptr)
    AssertThrow(this->froude_number > 0.0,
                ExcMessage("Non-vanishing Froude number is required if the body "
                           "force is specified."));
  if (gravity_field_ptr)
    AssertThrow(this->froude_number > 0.0,
                ExcMessage("For a buoyant fluid, the Froude number must be specified."));
  AssertThrow(this->reynolds_number > 0.0,
              ExcMessage("The Reynolds must not vanish (stabilization is not "
                         "implemented yet)."));

  QGauss<dim>   quadrature_formula(fe.degree + 1);

  UpdateFlags update_flags = update_values|
                             update_gradients|
                             update_quadrature_points|
                             update_JxW_values;
  if (this->stabilization & (apply_supg|apply_pspg))
    update_flags |= update_hessians;

  using Scratch = AssemblyData::RightHandSide::Scratch<dim>;
  Scratch scratch(mapping,
                  quadrature_formula,
                  fe,
                  update_flags,
                  QGauss<dim-1>(1),
                  update_default,
                  this->stabilization,
                  this->use_stress_form,
                  this->background_velocity_ptr != nullptr,
                  this->body_force_ptr != nullptr,
                  false,
                  false,
                  gravity_field_ptr != nullptr,
                  reference_density_ptr != nullptr);

  const FEValuesExtractors::Vector  velocity(this->velocity_start_index);
  const FEValuesExtractors::Scalar  pressure(this->pressure_index);
  const FEValuesExtractors::Scalar  density(density_index);

  const double nu{1.0 / this->reynolds_number};

  Hydrodynamic::OptionalArgumentsStrongForm<dim> &strong_form_options = scratch.hydrodynamic_strong_form_options;
  strong_form_options.use_stress_form = this->use_stress_form;

  BuoyantHydrodynamic::OptionalArgumentsStrongForm<dim> &buoyancy_strong_form_options = scratch.strong_form_options;

  // Coriolis term
  if (this->angular_velocity_ptr)
  {
    strong_form_options.angular_velocity = this->angular_velocity_ptr->value();
    strong_form_options.rossby_number = this->rossby_number;
  }

  double cell_momentum_residual;
  double mean_momentum_residual{0.0};
  double max_momentum_residual[2]{std::numeric_limits<double>::min(),
                                  std::numeric_limits<double>::min()};

  double cell_mass_residual;
  double mean_mass_residual{0.0};
  double max_mass_residual[2]{std::numeric_limits<double>::min(),
                              std::numeric_limits<double>::min()};

  double cell_density_residual;
  double mean_density_residual{0.0};
  double max_density_residual[2]{std::numeric_limits<double>::min(),
                                 std::numeric_limits<double>::min()};

  double mean_momentum_viscosity{0.0};
  double max_momentum_viscosity[2]{std::numeric_limits<double>::min(),
                                   std::numeric_limits<double>::min()};

  double mean_density_viscosity{0.0};
  double max_density_viscosity[2]{std::numeric_limits<double>::min(),
                                  std::numeric_limits<double>::min()};

  double cell_volume;
  double volume{0};

  for (const auto &cell: dof_handler.active_cell_iterators())
  if (cell->is_locally_owned())
  {
    scratch.fe_values.reinit(cell);

    const double delta{this->c * std::pow(cell->diameter(), 2)};
    const double delta_density{c_density * std::pow(cell->diameter(), 2)};

    scratch.fe_values[velocity].get_function_values(solution,
                                                    scratch.present_velocity_values);
    scratch.fe_values[velocity].get_function_gradients(solution,
                                                       scratch.present_velocity_gradients);

    scratch.fe_values[pressure].get_function_values(solution,
                                                    scratch.present_pressure_values);

    // stress form
    if (this->use_stress_form)
      scratch.fe_values[velocity].get_function_symmetric_gradients(solution,
                                                                   scratch.present_sym_velocity_gradients);

    scratch.fe_values[density].get_function_values(solution,
                                                   scratch.present_density_values);
    scratch.fe_values[density].get_function_gradients(solution,
                                                      scratch.present_density_gradients);

    // stabilization related solution values
    if (this->stabilization & (apply_supg|apply_pspg))
    {
      scratch.fe_values[velocity].get_function_laplacians(solution,
                                                          scratch.present_velocity_laplaceans);
      scratch.fe_values[pressure].get_function_gradients(solution,
                                                         scratch.present_pressure_gradients);

      // stress form
      if (this->use_stress_form)
      {
        std::vector<Tensor<3, dim>> present_hessians(scratch.n_q_points);
        scratch.fe_values[velocity].get_function_hessians(solution,
                                                          present_hessians);

        std::vector<Tensor<1, dim>> &present_velocity_grad_divergences =
            strong_form_options.present_velocity_grad_divergences.value();
        for (std::size_t q=0; q<present_hessians.size(); ++q)
        {
          present_velocity_grad_divergences[q] = 0;
          for (unsigned int d=0; d<dim; ++d)
            present_velocity_grad_divergences[q] += present_hessians[q][d][d];
        }
      }
    }

    // body force
    if (this->body_force_ptr)
    {
      this->body_force_ptr->value_list(scratch.fe_values.get_quadrature_points(),
                                       *strong_form_options.body_force_values);
      strong_form_options.froude_number = this->froude_number;
    }

    // background field
    if (this->background_velocity_ptr)
    {
      this->background_velocity_ptr->value_list(scratch.fe_values.get_quadrature_points(),
                                                *strong_form_options.background_velocity_values);
      this->background_velocity_ptr->gradient_list(scratch.fe_values.get_quadrature_points(),
                                                   *strong_form_options.background_velocity_gradients);
    }

    // reference density
    if (reference_density_ptr)
    {
      reference_density_ptr->gradient_list(scratch.fe_values.get_quadrature_points(),
                                           *buoyancy_strong_form_options.reference_density_gradients);

      buoyancy_strong_form_options.stratification_number = stratification_number;
    }

    // gravity field
    if (gravity_field_ptr)
    {
      gravity_field_ptr->value_list(scratch.fe_values.get_quadrature_points(),
                                    *buoyancy_strong_form_options.gravity_field_values);

      strong_form_options.froude_number = this->froude_number;
    }

    // stabilization
    if (this->stabilization & (apply_supg|apply_pspg))
      compute_strong_hydrodynamic_residual(scratch.present_velocity_values,
                                           scratch.present_velocity_gradients,
                                           scratch.present_velocity_laplaceans,
                                           scratch.present_pressure_gradients,
                                           scratch.present_density_values,
                                           scratch.present_strong_residuals,
                                           nu,
                                           strong_form_options,
                                           buoyancy_strong_form_options);
    compute_strong_density_residual(scratch.present_density_gradients,
                                    scratch.present_velocity_values,
                                    scratch.present_strong_density_residuals,
                                    strong_form_options,
                                    buoyancy_strong_form_options);

    cell_momentum_residual = 0;
    cell_mass_residual = 0;
    cell_density_residual = 0;
    cell_volume = 0;

    for (const auto q: scratch.fe_values.quadrature_point_indices())
    {
      const double JxW{scratch.fe_values.JxW(q)};

      const double mass_residual{trace(scratch.present_velocity_gradients[q])};
      const double density_residual{scratch.present_strong_density_residuals[q]};

      max_mass_residual[0] = std::max(std::abs(mass_residual),
                                      max_mass_residual[0]);
      max_density_residual[0] = std::max(std::abs(density_residual),
                                         max_density_residual[0]);
      max_density_viscosity[0] = std::max(std::abs(delta_density * density_residual),
                                          max_density_viscosity[0]);

      cell_mass_residual += mass_residual * JxW;
      cell_density_residual += density_residual * JxW;

      cell_volume += JxW;
      volume += JxW;

      if (this->stabilization & (apply_supg|apply_pspg))
      {
        const double momentum_residual{scratch.present_strong_residuals[q].norm()};

        max_momentum_residual[0] = std::max(momentum_residual,
                                            max_momentum_residual[0]);
        max_momentum_viscosity[0] = std::max(delta * momentum_residual,
                                             max_momentum_viscosity[0]);
        cell_momentum_residual += momentum_residual * JxW;
      }

    } // end loop over cell quadrature points

    max_momentum_residual[1] = std::max(cell_momentum_residual / cell_volume,
                                        max_momentum_residual[1]);
    max_mass_residual[1] = std::max(cell_mass_residual / cell_volume,
                                    max_mass_residual[1]);
    max_density_residual[1] = std::max(cell_density_residual / cell_volume,
                                       max_density_residual[1]);
    max_momentum_viscosity[1] = std::max(delta * cell_momentum_residual / cell_volume,
                                         max_momentum_viscosity[1]);
    max_density_viscosity[1] = std::max(delta_density * cell_density_residual / cell_volume,
                                        max_density_viscosity[1]);


    mean_momentum_residual += cell_momentum_residual;
    mean_mass_residual += cell_mass_residual;
    mean_density_residual += cell_density_residual;
    mean_momentum_viscosity += delta * cell_momentum_residual;
    mean_density_residual += delta_density * cell_density_residual;

  } // end loop over active cells

  // compute mean value
  Assert(volume > 0.0, ExcLowerRangeType<double>(0.0, volume));

  Utilities::MPI::max(max_momentum_residual, MPI_COMM_WORLD, max_momentum_residual);
  Utilities::MPI::max(max_mass_residual, MPI_COMM_WORLD, max_mass_residual);
  Utilities::MPI::max(max_density_viscosity, MPI_COMM_WORLD, max_density_viscosity);
  Utilities::MPI::max(max_momentum_viscosity, MPI_COMM_WORLD, max_momentum_viscosity);

  volume = Utilities::MPI::sum(volume, MPI_COMM_WORLD);
  mean_momentum_residual = Utilities::MPI::sum(mean_momentum_residual, MPI_COMM_WORLD);
  mean_mass_residual = Utilities::MPI::sum(mean_mass_residual, MPI_COMM_WORLD);
  mean_density_viscosity = Utilities::MPI::sum(mean_density_viscosity, MPI_COMM_WORLD);

  mean_momentum_residual /= volume;
  mean_mass_residual /= volume;
  mean_momentum_viscosity /= volume;
  mean_density_viscosity /= volume;

  this->data_table.add_value("cycle", this->cycle);

  this->data_table.add_value("max point visc.", max_momentum_viscosity[0]);
  this->data_table.add_value("max cell visc.", max_momentum_viscosity[1]);
  this->data_table.add_value("mean visc.", mean_momentum_viscosity);

  this->data_table.add_value("max momentum point res.", max_momentum_residual[0]);
  this->data_table.add_value("max momentum cell res.", max_momentum_residual[1]);
  this->data_table.add_value("mean momentum res.", mean_momentum_residual);

  this->data_table.add_value("max mass point res.", max_mass_residual[0]);
  this->data_table.add_value("max mass cell res.", max_mass_residual[1]);
  this->data_table.add_value("mean mass res.", mean_mass_residual);

  this->data_table.add_value("max density point res.", max_density_residual[0]);
  this->data_table.add_value("max density cell res.", max_density_residual[1]);
  this->data_table.add_value("mean density res.", mean_density_residual);

  this->data_table.add_value("max density point visc.", max_density_viscosity[0]);
  this->data_table.add_value("max density cell visc.", max_density_viscosity[1]);
  this->data_table.add_value("mean density visc.", mean_density_viscosity);
}

// explicit instantiations
template class EvaluationStabilization<2>;
template class EvaluationStabilization<3>;

}  // namespace BuoyantHydrodynamic

