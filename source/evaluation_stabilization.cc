/*
 * evaluation_stabilization.cc
 *
 *  Created on: Sep 27, 2021
 *      Author: sg
 */
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_values.h>

#include <assembly_functions.h>
#include <evaluation_stabilization.h>

namespace Hydrodynamic {

using namespace dealii;

template <int dim>
EvaluationStabilization<dim>::EvaluationStabilization
(const StabilizationFlags &stabilization_flags,
 const unsigned int velocity_start_index,
 const unsigned int pressure_index,
 const double reynolds_number,
 const double froude_number,
 const double rossby_number,
 const bool print_table)
:
angular_velocity_ptr(nullptr),
body_force_ptr(nullptr),
background_velocity_ptr(nullptr),
stabilization(stabilization_flags),
velocity_start_index(velocity_start_index),
pressure_index(pressure_index),
reynolds_number(reynolds_number),
froude_number(froude_number),
rossby_number(rossby_number),
print_table(print_table),
c(std::numeric_limits<double>::min()),
mu(std::numeric_limits<double>::min())
{
  data_table.declare_column("cycle");

  data_table.declare_column("max viscosity");
  data_table.declare_column("mean viscosity");
  data_table.set_scientific("max viscosity", true);
  data_table.set_scientific("mean viscosity", true);
  data_table.add_column_to_supercolumn("max viscosity", "viscosity");
  data_table.add_column_to_supercolumn("mean viscosity", "viscosity");

  data_table.declare_column("max momentum residual");
  data_table.declare_column("mean momentum residual");
  data_table.set_scientific("max momentum residual", true);
  data_table.set_scientific("mean momentum residual", true);
  data_table.add_column_to_supercolumn("max momentum residual", "momentum residual");
  data_table.add_column_to_supercolumn("mean momentum residual", "momentum residual");

  data_table.declare_column("max mass residual");
  data_table.declare_column("mean mass residual");
  data_table.set_scientific("max mass residual", true);
  data_table.set_scientific("mean mass residual", true);
  data_table.add_column_to_supercolumn("max mass residual", "mass residual");
  data_table.add_column_to_supercolumn("mean mass residual", "mass residual");
}



template <int dim>
EvaluationStabilization<dim>::~EvaluationStabilization()
{
  std::cout << std::endl;
  if (print_table)
    data_table.write_text(std::cout);
}



template <int dim>
void EvaluationStabilization<dim>::operator()
(const Mapping<dim>        &/* mapping */,
 const FiniteElement<dim>  &/* fe */,
 const DoFHandler<dim>     &/* dof_handler */,
 const Vector<double>      &/* solution */)
{
  AssertThrow(false, ExcInternalError());
}



template <int dim>
void EvaluationStabilization<dim>::operator()
(const Mapping<dim>        &mapping,
 const FiniteElement<dim>  &fe,
 const DoFHandler<dim>     &dof_handler,
 const BlockVector<double> &solution)
{
  AssertThrow(c > std::numeric_limits<double>::min(), ExcInternalError());
  AssertThrow(mu > std::numeric_limits<double>::min(), ExcInternalError());

  if (angular_velocity_ptr != nullptr)
    AssertThrow(rossby_number > 0.0,
                ExcMessage("Non-vanishing Rossby number is required if the angular "
                           "velocity vector is specified."));
  if (body_force_ptr != nullptr)
    AssertThrow(froude_number > 0.0,
                ExcMessage("Non-vanishing Froude number is required if the body "
                           "force is specified."));
  AssertThrow(reynolds_number != 0.0,
              ExcMessage("The Reynolds must not vanish (stabilization is not "
                         "implemented yet)."));

  QGauss<dim>   quadrature(fe.degree + 1);

  UpdateFlags update_flags{update_values|
                           update_gradients|
                           update_hessians|
                           update_JxW_values};
  if ((angular_velocity_ptr != nullptr) ||
      (background_velocity_ptr != nullptr))
    update_flags |= update_quadrature_points;

  FEValues<dim> fe_values(mapping,
                          fe,
                          quadrature,
                          update_flags);

  const FEValuesExtractors::Vector  velocity(velocity_start_index);
  const FEValuesExtractors::Scalar  pressure(pressure_index);

  const unsigned int dofs_per_cell{fe.n_dofs_per_cell()};
  Vector<double>  cell_rhs(dofs_per_cell);

  // stabilization related shape functions
  std::vector<Tensor<2, dim>>  grad_phi_velocity;
  std::vector<Tensor<1, dim>>  grad_phi_pressure;
  if (stabilization & (apply_supg|apply_grad_div))
    grad_phi_velocity.resize(dofs_per_cell);
  if (stabilization & apply_pspg)
    grad_phi_pressure.resize(dofs_per_cell);

  const unsigned int n_q_points{quadrature.size()};
  // solution values
  std::vector<Tensor<1, dim>>  present_velocity_values(n_q_points);
  std::vector<Tensor<2, dim>>  present_velocity_gradients(n_q_points);
  std::vector<double>          present_pressure_values(n_q_points);

  // strong residual related solution values
  std::vector<Tensor<1, dim>>  present_velocity_laplaceans(n_q_points);
  std::vector<Tensor<1, dim>>  present_pressure_gradients(n_q_points);

  std::vector<Tensor<1, dim>> background_velocity_values;
  std::vector<Tensor<2, dim>> background_velocity_gradients;
  if (background_velocity_ptr != nullptr)
  {
    background_velocity_values.resize(n_q_points);
    background_velocity_gradients.resize(n_q_points);
  }

  // source term values
  std::vector<Tensor<1,dim>>  body_force_values;
  if (body_force_ptr != nullptr)
    body_force_values.resize(n_q_points);

  typename Utility::AngularVelocity<dim>::value_type angular_velocity_value;
  if (angular_velocity_ptr != nullptr)
    angular_velocity_value = angular_velocity_ptr->value();

  const double nu{1.0 / reynolds_number};

  Tensor<1, dim> cell_momentum_residual;
  double mean_momentum_residual{0.0};
  double max_momentum_residual{std::numeric_limits<double>::min()};

  double cell_mass_residual;
  double mean_mass_residual{0.0};
  double max_mass_residual{std::numeric_limits<double>::min()};

  double mean_cell_viscosity{0.0};
  double max_cell_viscosity{std::numeric_limits<double>::min()};

  double volume{0};

  for (const auto cell: dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);

    const double delta{c * std::pow(cell->diameter(), 2)};

    cell_rhs = 0;
    cell_momentum_residual = 0;
    cell_mass_residual = 0;

    fe_values[velocity].get_function_values(solution,
                                            present_velocity_values);
    fe_values[velocity].get_function_gradients(solution,
                                               present_velocity_gradients);

    fe_values[pressure].get_function_values(solution,
                                            present_pressure_values);

    // strong residual related solution values
    fe_values[velocity].get_function_laplacians(solution,
                                                present_velocity_laplaceans);
    fe_values[pressure].get_function_gradients(solution,
                                               present_pressure_gradients);

    // body force
    if (body_force_ptr != nullptr)
      body_force_ptr->value_list(fe_values.get_quadrature_points(),
                                 body_force_values);

    // background field
    if (background_velocity_ptr != nullptr)
    {
      background_velocity_ptr->value_list(fe_values.get_quadrature_points(),
                                          background_velocity_values);
      background_velocity_ptr->gradient_list(fe_values.get_quadrature_points(),
                                             background_velocity_gradients);
    }

    for (const auto q: fe_values.quadrature_point_indices())
    {
      for (const auto i: fe_values.dof_indices())
      {
        // stabilization related shape functions
        if (stabilization & (apply_supg|apply_grad_div))
          grad_phi_velocity[i] = fe_values[velocity].gradient(i, q);
        if (stabilization & apply_pspg)
          grad_phi_pressure[i] = fe_values[pressure].gradient(i, q);
      }

      const double JxW{fe_values.JxW(q)};

      double rhs{0.0};

      for (const auto i: fe_values.dof_indices())
      {
//        if (stabilization & apply_supg)
//          rhs += delta * compute_supg_rhs(grad_phi_velocity[i],
//                                          present_velocity_values[q],
//                                          present_velocity_gradients[q],
//                                          present_velocity_laplaceans[q],
//                                          present_pressure_gradients[q],
//                                          nu);
//        if (stabilization & apply_pspg)
//          rhs += delta * compute_pspg_rhs(present_velocity_values[q],
//                                          present_velocity_gradients[q],
//                                          present_velocity_laplaceans[q],
//                                          grad_phi_pressure[i],
//                                          present_pressure_gradients[q],
//                                          nu);
        if (stabilization & apply_grad_div)
          rhs += mu * compute_grad_div_rhs(present_velocity_gradients[q],
                                           grad_phi_velocity[i]);

        // body force term
        if (body_force_ptr != nullptr)
        {
          Tensor<1, dim> body_force_test_function;

          if (stabilization & apply_supg)
          {
            body_force_test_function += delta * grad_phi_velocity[i] *
                                        present_velocity_values[q];
            if (background_velocity_ptr != nullptr)
              body_force_test_function += delta * grad_phi_velocity[i] *
                                          background_velocity_values[q];
          }
          if (stabilization & apply_pspg)
            body_force_test_function += delta * grad_phi_pressure[i];

          rhs += body_force_values[q] * body_force_test_function / std::pow(this->froude_number, 2);
        }

        // background field term
        if (background_velocity_ptr != nullptr)
        {
          Tensor<1, dim> background_velocity_test_function;

          if (stabilization & apply_supg)
            background_velocity_test_function += delta * grad_phi_velocity[i] *
                                                 (present_velocity_values[q] +
                                                  background_velocity_values[q]);
          if (stabilization & apply_pspg)
            background_velocity_test_function += delta * grad_phi_pressure[i];

          rhs -= (present_velocity_gradients[q] * background_velocity_values[q] +
                  background_velocity_gradients[q] * present_velocity_values[q]) *
                 background_velocity_test_function;

          if (stabilization & apply_supg)
          {
            const Tensor<1, dim> projected_test_function_gradient(grad_phi_velocity[i] *
                                                                  background_velocity_values[q]);

            rhs -= // standard stabilization term
                   delta *
                   (present_velocity_gradients[q] * present_velocity_values[q] -
                    nu * present_velocity_laplaceans[q] +
                    present_pressure_gradients[q] ) * projected_test_function_gradient;
          }
        }

        // Coriolis term
        if (angular_velocity_ptr != nullptr)
        {
          Tensor<1, dim> coriolis_term_test_function;

          // Coriolis stabilization terms
          if (stabilization & apply_supg)
          {
            coriolis_term_test_function += delta * grad_phi_velocity[i] *
                                           present_velocity_values[q];
            if (background_velocity_ptr != nullptr)
              coriolis_term_test_function += delta * grad_phi_velocity[i] *
                                             background_velocity_values[q];
          }
          if (stabilization & apply_pspg)
            coriolis_term_test_function += delta * grad_phi_pressure[i];

          if constexpr(dim == 2)
            rhs -= 2.0 / rossby_number * angular_velocity_value[0] *
                   cross_product_2d(-present_velocity_values[q]) *
                   coriolis_term_test_function;
          else if constexpr(dim == 3)
            rhs -= 2.0 / rossby_number *
                   cross_product_3d(angular_velocity_value, present_velocity_values[q]) *
                   coriolis_term_test_function;
        }


        Tensor<1, dim> momentum_residual;
        double  mass_residual{std::numeric_limits<double>::min()};

        momentum_residual += (present_velocity_gradients[q] * present_velocity_values[q] -
                              nu * present_velocity_laplaceans[q] +
                              present_pressure_gradients[q]);
        mass_residual = trace(present_velocity_gradients[q]);

        // body force term
        if (body_force_ptr != nullptr)
          momentum_residual -= body_force_values[q] / std::pow(this->froude_number, 2);
        // background field term
        if (background_velocity_ptr != nullptr)
          momentum_residual += (present_velocity_gradients[q] * background_velocity_values[q] +
                                background_velocity_gradients[q] * present_velocity_values[q]);
        // Coriolis term
        if (angular_velocity_ptr != nullptr)
        {
          if constexpr(dim == 2)
            momentum_residual += 2.0 / rossby_number * angular_velocity_value[0] *
                                 cross_product_2d(-present_velocity_values[q]);
          else if constexpr(dim == 3)
            momentum_residual += 2.0 / rossby_number *
                                 cross_product_3d(angular_velocity_value,
                                                  present_velocity_values[q]);
        }

        max_momentum_residual = std::max(momentum_residual.norm(), max_momentum_residual);
        max_mass_residual = std::max(std::abs(mass_residual), max_mass_residual);

        cell_momentum_residual += momentum_residual * JxW;
        cell_mass_residual += mass_residual * JxW;

        cell_rhs(i) += rhs * JxW;

        volume += JxW;
      }
    } // end loop over cell quadrature points

    const double cell_viscosity{cell_rhs.l2_norm()};
    max_cell_viscosity = std::max(cell_viscosity, max_cell_viscosity);
    mean_cell_viscosity += cell_viscosity;

    mean_momentum_residual += cell_momentum_residual.norm();
    mean_mass_residual += std::abs(cell_mass_residual);

  } // end loop over active cells

  // compute mean value
  Assert(volume > 0.0, ExcLowerRangeType<double>(0.0, volume));
  mean_cell_viscosity /= volume;
  mean_momentum_residual /= volume;
  mean_mass_residual /= volume;

  data_table.add_value("cycle", this->cycle);

  data_table.add_value("max viscosity", max_cell_viscosity);
  data_table.add_value("mean viscosity", mean_cell_viscosity);

  data_table.add_value("max momentum residual", max_momentum_residual);
  data_table.add_value("mean momentum residual", mean_momentum_residual);

  data_table.add_value("max mass residual", max_mass_residual);
  data_table.add_value("mean mass residual", mean_mass_residual);
}

// explicit instantiations
template class EvaluationStabilization<2>;
template class EvaluationStabilization<3>;

}  // namespace Hydrodynamic



namespace BuoyantHydrodynamic {

using namespace dealii;

template <int dim>
EvaluationStabilization<dim>::EvaluationStabilization
(const StabilizationFlags &stabilization_flags,
 const unsigned int velocity_start_index,
 const unsigned int pressure_index,
 const unsigned int density_index,
 const double reynolds_number,
 const double stratification_number,
 const double froude_number,
 const double rossby_number)
:
Hydrodynamic::EvaluationStabilization<dim>(stabilization_flags,
                                           velocity_start_index, pressure_index,
                                           reynolds_number, froude_number, rossby_number),
reference_density_ptr(nullptr),
gravity_field_ptr(nullptr),
density_index(density_index),
stratification_number(stratification_number),
c_density(std::numeric_limits<double>::min())
{
  this->data_table.declare_column("max density viscosity");
  this->data_table.declare_column("mean density viscosity");
  this->data_table.set_scientific("max density viscosity", true);
  this->data_table.set_scientific("mean density viscosity", true);
  this->data_table.add_column_to_supercolumn("max density viscosity", "density viscosity");
  this->data_table.add_column_to_supercolumn("mean density viscosity", "density viscosity");

  this->data_table.declare_column("max density residual");
  this->data_table.declare_column("mean density residual");
  this->data_table.set_scientific("max density residual", true);
  this->data_table.set_scientific("mean density residual", true);
  this->data_table.add_column_to_supercolumn("max density residual", "density residual");
  this->data_table.add_column_to_supercolumn("mean density residual", "density residual");
}



template <int dim>
void EvaluationStabilization<dim>::operator()
(const Mapping<dim>        &/* mapping */,
 const FiniteElement<dim>  &/* fe */,
 const DoFHandler<dim>     &/* dof_handler */,
 const Vector<double>      &/* solution */)
{
  AssertThrow(false, ExcInternalError());
}



template <int dim>
void EvaluationStabilization<dim>::operator()
(const Mapping<dim>        &mapping,
 const FiniteElement<dim>  &fe,
 const DoFHandler<dim>     &dof_handler,
 const BlockVector<double> &solution)
{
  AssertThrow(this->c > std::numeric_limits<double>::min(), ExcInternalError());
  AssertThrow(this->mu > std::numeric_limits<double>::min(), ExcInternalError());
  AssertThrow(this->c_density > std::numeric_limits<double>::min(), ExcInternalError());

  if (this->angular_velocity_ptr != nullptr)
    AssertThrow(this->rossby_number > 0.0,
                ExcMessage("Non-vanishing Rossby number is required if the angular "
                           "velocity vector is specified."));
  AssertThrow(gravity_field_ptr != nullptr,
              ExcMessage("For a buoyant fluid, the gravity field must be specified."));
  AssertThrow(reference_density_ptr != nullptr,
              ExcMessage("For a buoyant fluid, the reference density field must be specified."));

  AssertThrow(this->froude_number > 0.0,
              ExcMessage("For a buoyant fluid, the Froude number must be specified."));
  AssertThrow(this->reynolds_number > 0.0,
              ExcMessage("The Reynolds must not vanish (stabilization is not "
                         "implemented yet)."));

  QGauss<dim>   quadrature(fe.degree + 1);

  FEValues<dim> fe_values(mapping,
                          fe,
                          quadrature,
                          update_values|
                          update_gradients|
                          update_hessians|
                          update_quadrature_points|
                          update_JxW_values);

  const FEValuesExtractors::Vector  velocity(this->velocity_start_index);
  const FEValuesExtractors::Scalar  pressure(this->pressure_index);
  const FEValuesExtractors::Scalar  density(density_index);

  const unsigned int dofs_per_cell{fe.n_dofs_per_cell()};
  Vector<double>  cell_rhs(dofs_per_cell);
  Vector<double>  cell_rhs_density(dofs_per_cell);

  // stabilization related shape functions
  std::vector<Tensor<2, dim>> grad_phi_velocity;
  std::vector<Tensor<1, dim>> grad_phi_pressure;
  std::vector<Tensor<1, dim>> grad_phi_density(dofs_per_cell);
  if (this->stabilization & (apply_supg|apply_grad_div))
    grad_phi_velocity.resize(dofs_per_cell);
  if (this->stabilization & apply_pspg)
    grad_phi_pressure.resize(dofs_per_cell);

  const unsigned int n_q_points{quadrature.size()};
  // solution values
  std::vector<Tensor<1, dim>>  present_velocity_values(n_q_points);
  std::vector<Tensor<2, dim>>  present_velocity_gradients(n_q_points);
  std::vector<double>          present_pressure_values(n_q_points);
  std::vector<double>         present_density_values(n_q_points);
  std::vector<Tensor<1, dim>> present_density_gradients(n_q_points);

  // strong residual related solution values
  std::vector<Tensor<1, dim>>  present_velocity_laplaceans(n_q_points);
  std::vector<Tensor<1, dim>>  present_pressure_gradients(n_q_points);

  std::vector<Tensor<1, dim>> background_velocity_values;
  std::vector<Tensor<2, dim>> background_velocity_gradients;
  if (this->background_velocity_ptr != nullptr)
  {
    background_velocity_values.resize(n_q_points);
    background_velocity_gradients.resize(n_q_points);
  }

  // source term values
  std::vector<Tensor<1,dim>>  reference_density_gradients(n_q_points);
  std::vector<Tensor<1,dim>>  gravity_field_values(n_q_points);
  std::vector<Tensor<1,dim>>  body_force_values;
  if (this->body_force_ptr != nullptr)
    body_force_values.resize(n_q_points);

  typename Utility::AngularVelocity<dim>::value_type angular_velocity_value;
  if (this->angular_velocity_ptr != nullptr)
    angular_velocity_value = this->angular_velocity_ptr->value();

  const double nu{1.0 / this->reynolds_number};

  Tensor<1, dim> cell_momentum_residual;
  double mean_momentum_residual{0.0};
  double max_momentum_residual{std::numeric_limits<double>::min()};

  double cell_mass_residual;
  double mean_mass_residual{0.0};
  double max_mass_residual{std::numeric_limits<double>::min()};

  double mean_cell_viscosity{0.0};
  double max_cell_viscosity{std::numeric_limits<double>::min()};

  double mean_cell_viscosity_density{0.0};
  double max_cell_viscosity_density{std::numeric_limits<double>::min()};

  double cell_density_residual;
  double mean_density_residual{0.0};
  double max_density_residual{std::numeric_limits<double>::min()};

  double volume{0};

  for (const auto cell: dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);

    const double delta{this->c * std::pow(cell->diameter(), 2)};
    const double delta_density{c_density * std::pow(cell->diameter(), 2)};

    cell_rhs = 0;
    cell_rhs_density = 0;
    cell_momentum_residual = 0;
    cell_mass_residual = 0;

    fe_values[velocity].get_function_values(solution,
                                            present_velocity_values);
    fe_values[velocity].get_function_gradients(solution,
                                               present_velocity_gradients);

    fe_values[pressure].get_function_values(solution,
                                            present_pressure_values);

    fe_values[density].get_function_values(solution,
                                           present_density_values);
    fe_values[density].get_function_gradients(solution,
                                              present_density_gradients);

    // strong residual related solution values
    fe_values[velocity].get_function_laplacians(solution,
                                                present_velocity_laplaceans);
    fe_values[pressure].get_function_gradients(solution,
                                               present_pressure_gradients);



    // body force
    if (this->body_force_ptr != nullptr)
      this->body_force_ptr->value_list(fe_values.get_quadrature_points(),
                                       body_force_values);

    // background field
    if (this->background_velocity_ptr != nullptr)
    {
      this->background_velocity_ptr->value_list(fe_values.get_quadrature_points(),
                                                background_velocity_values);
      this->background_velocity_ptr->gradient_list(fe_values.get_quadrature_points(),
                                                   background_velocity_gradients);
    }

    // reference density
    reference_density_ptr->gradient_list(fe_values.get_quadrature_points(),
                                         reference_density_gradients);

    // gravity field
    gravity_field_ptr->value_list(fe_values.get_quadrature_points(),
                                  gravity_field_values);


    for (const auto q: fe_values.quadrature_point_indices())
    {
      for (const auto i: fe_values.dof_indices())
      {
        grad_phi_density[i] = fe_values[density].gradient(i, q);

        // stabilization related shape functions
        if (this->stabilization & (apply_supg|apply_grad_div))
          grad_phi_velocity[i] = fe_values[velocity].gradient(i, q);
        if (this->stabilization & apply_pspg)
          grad_phi_pressure[i] = fe_values[pressure].gradient(i, q);
      }

      const double JxW{fe_values.JxW(q)};

      double rhs{0.0};
      double rhs_density{0.0};

      for (const auto i: fe_values.dof_indices())
      {
//        // rhs step 1: momentum part
//        if (this->stabilization & apply_supg)
//          rhs += delta * Hydrodynamic::
//                 compute_supg_rhs(grad_phi_velocity[i],
//                                  present_velocity_values[q],
//                                  present_velocity_gradients[q],
//                                  present_velocity_laplaceans[q],
//                                  present_pressure_gradients[q],
//                                  nu);
//        if (this->stabilization & apply_pspg)
//          rhs += delta * Hydrodynamic::
//                 compute_pspg_rhs(present_velocity_values[q],
//                                  present_velocity_gradients[q],
//                                  present_velocity_laplaceans[q],
//                                  grad_phi_pressure[i],
//                                  present_pressure_gradients[q],
//                                  nu);
        if (this->stabilization & apply_grad_div)
          rhs += this->mu * Hydrodynamic::
                 compute_grad_div_rhs(present_velocity_gradients[q],
                                      grad_phi_velocity[i]);

        // body force term
        if (this->body_force_ptr != nullptr)
        {
          Tensor<1, dim> body_force_test_function;

          if (this->stabilization & apply_supg)
          {
            body_force_test_function += delta * grad_phi_velocity[i] *
                                        present_velocity_values[q];
            if (this->background_velocity_ptr != nullptr)
              body_force_test_function += delta * grad_phi_velocity[i] *
                                          background_velocity_values[q];
          }
          if (this->stabilization & apply_pspg)
            body_force_test_function += delta * grad_phi_pressure[i];

          rhs += body_force_values[q] * body_force_test_function / std::pow(this->froude_number, 2);
        }

        // buoyancy term
        {
          Tensor<1, dim> buoyancy_test_function;

          // buoyancy stabilization terms
          if (this->stabilization & apply_supg)
          {
            buoyancy_test_function += delta * grad_phi_velocity[i] *
                                      present_velocity_values[q];
            if (this->background_velocity_ptr != nullptr)
              buoyancy_test_function += delta * grad_phi_velocity[i] *
                                        background_velocity_values[q];
          }
          if (this->stabilization & apply_pspg)
            buoyancy_test_function += delta * grad_phi_pressure[i];

          rhs += present_density_values[q] * gravity_field_values[q] *
                 buoyancy_test_function / std::pow(this->froude_number, 2);
        }


        // background field term
        if (this->background_velocity_ptr != nullptr)
        {
          Tensor<1, dim> background_velocity_test_function;

          if (this->stabilization & apply_supg)
            background_velocity_test_function += delta * grad_phi_velocity[i] *
                                                 (present_velocity_values[q] +
                                                  background_velocity_values[q]);
          if (this->stabilization & apply_pspg)
            background_velocity_test_function += delta * grad_phi_pressure[i];

          rhs -= (present_velocity_gradients[q] * background_velocity_values[q] +
                  background_velocity_gradients[q] * present_velocity_values[q]) *
                 background_velocity_test_function;

          if (this->stabilization & apply_supg)
          {
            const Tensor<1, dim> projected_test_function_gradient(grad_phi_velocity[i] *
                                                                  background_velocity_values[q]);

            rhs -= // standard stabilization term
                   delta *
                   (present_velocity_gradients[q] * present_velocity_values[q] -
                    nu * present_velocity_laplaceans[q] +
                    present_pressure_gradients[q] ) * projected_test_function_gradient;
          }
        }

        // Coriolis term
        if (this->angular_velocity_ptr != nullptr)
        {
          Tensor<1, dim> coriolis_term_test_function;

          // Coriolis stabilization terms
          if (this->stabilization & apply_supg)
          {
            coriolis_term_test_function += delta * grad_phi_velocity[i] *
                                           present_velocity_values[q];
            if (this->background_velocity_ptr != nullptr)
              coriolis_term_test_function += delta * grad_phi_velocity[i] *
                                             background_velocity_values[q];
          }
          if (this->stabilization & apply_pspg)
            coriolis_term_test_function += delta * grad_phi_pressure[i];

          if constexpr(dim == 2)
            rhs -= 2.0 / this->rossby_number * angular_velocity_value[0] *
                   cross_product_2d(-present_velocity_values[q]) *
                   coriolis_term_test_function;
          else if constexpr(dim == 3)
            rhs -= 2.0 / this->rossby_number *
                   cross_product_3d(angular_velocity_value, present_velocity_values[q]) *
                   coriolis_term_test_function;
        }

        // rhs step 2: density part
        rhs_density += delta_density *
                       compute_density_supg_rhs(grad_phi_density[i],
                                                present_density_gradients[q],
                                                present_velocity_values[q],
                                                reference_density_gradients[q],
                                                stratification_number,
                                                0.0);

        // background field term
        if (this->background_velocity_ptr != nullptr)
        {
          const double background_velocity_test_function
            = delta_density * (present_velocity_values[q] + background_velocity_values[q]) *
              grad_phi_density[i];

          rhs_density -= (stratification_number * background_velocity_values[q] * reference_density_gradients[q] +
                         background_velocity_values[q] * present_density_gradients[q] ) *
                             background_velocity_test_function +
                             delta_density *
                             (stratification_number * present_velocity_values[q] * reference_density_gradients[q] +
                                 present_velocity_values[q] * present_density_gradients[q]) *
                                 (background_velocity_values[q] * grad_phi_density[i]);
        }

        // residual step 1: momentum part
        Tensor<1, dim> momentum_residual;
        double  mass_residual{std::numeric_limits<double>::min()};

        momentum_residual += (present_velocity_gradients[q] * present_velocity_values[q] -
                              nu * present_velocity_laplaceans[q] +
                              present_pressure_gradients[q]);
        mass_residual = trace(present_velocity_gradients[q]);

        // body force term
        if (this->body_force_ptr != nullptr)
          momentum_residual -= body_force_values[q] / std::pow(this->froude_number, 2);
        // buoyancy term
        momentum_residual -= present_density_values[q] * gravity_field_values[q] / std::pow(this->froude_number, 2);
        // background field term
        if (this->background_velocity_ptr != nullptr)
          momentum_residual += (present_velocity_gradients[q] * background_velocity_values[q] +
                                background_velocity_gradients[q] * present_velocity_values[q]);
        // Coriolis term
        if (this->angular_velocity_ptr != nullptr)
        {
          if constexpr(dim == 2)
            momentum_residual += 2.0 / this->rossby_number * angular_velocity_value[0] *
                                 cross_product_2d(-present_velocity_values[q]);
          else if constexpr(dim == 3)
            momentum_residual += 2.0 / this->rossby_number *
                                 cross_product_3d(angular_velocity_value,
                                                  present_velocity_values[q]);
        }

        // residual step 2: momentum part
        double  density_residual{std::numeric_limits<double>::min()};
        density_residual = stratification_number * present_velocity_values[q] *
                           reference_density_gradients[q] +
                           present_velocity_values[q] * present_density_gradients[q];
        // background field term
        if (this->background_velocity_ptr != nullptr)
          density_residual  += (stratification_number * background_velocity_values[q] * reference_density_gradients[q] +
                                background_velocity_values[q] * present_density_gradients[q]);

        max_momentum_residual = std::max(momentum_residual.norm(), max_momentum_residual);
        max_mass_residual = std::max(std::abs(mass_residual), max_mass_residual);
        max_density_residual = std::max(std::abs(density_residual), max_density_residual);

        cell_momentum_residual += momentum_residual * JxW;
        cell_mass_residual += mass_residual * JxW;
        cell_density_residual += density_residual * JxW;

        cell_rhs(i) += rhs * JxW;
        cell_rhs_density(i) += rhs_density * JxW;

        volume += JxW;
      }
    } // end loop over cell quadrature points

    const double cell_viscosity{cell_rhs.l2_norm()};
    max_cell_viscosity = std::max(cell_viscosity, max_cell_viscosity);
    mean_cell_viscosity += cell_viscosity;

    const double cell_density_viscosity{cell_rhs_density.l2_norm()};
    max_cell_viscosity_density= std::max(cell_density_viscosity, max_cell_viscosity_density);
    mean_cell_viscosity_density += cell_density_viscosity;

    mean_momentum_residual += cell_momentum_residual.norm();
    mean_mass_residual += std::abs(cell_mass_residual);
    mean_density_residual += std::abs(cell_density_residual);

  } // end loop over active cells

  // compute mean value
  Assert(volume > 0.0, ExcLowerRangeType<double>(0.0, volume));
  mean_cell_viscosity /= volume;
  mean_momentum_residual /= volume;
  mean_mass_residual /= volume;
  mean_cell_viscosity_density /= volume;
  mean_density_residual /= volume;

  this->data_table.add_value("cycle", this->cycle);

  this->data_table.add_value("max viscosity", max_cell_viscosity);
  this->data_table.add_value("mean viscosity", mean_cell_viscosity);

  this->data_table.add_value("max momentum residual", max_momentum_residual);
  this->data_table.add_value("mean momentum residual", mean_momentum_residual);

  this->data_table.add_value("max mass residual", max_mass_residual);
  this->data_table.add_value("mean mass residual", mean_mass_residual);

  this->data_table.add_value("max density viscosity", max_cell_viscosity_density);
  this->data_table.add_value("mean density viscosity", mean_cell_viscosity_density);

  this->data_table.add_value("max density residual", max_density_residual);
  this->data_table.add_value("mean density residual", mean_density_residual);
}

// explicit instantiations
template class EvaluationStabilization<2>;
template class EvaluationStabilization<3>;

}  // namespace BuoyantHydrodynamic

