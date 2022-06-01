/*
 * assembly_functions.cc
 *
 *  Created on: Apr 20, 2022
 *      Author: sg
 */

#include <assembly_functions.h>

namespace Hydrodynamic {

template <int dim>
double compute_matrix
(const StabilizationFlags  &stabilization,
 const AssemblyData::Matrix::ScratchData<dim> &scratch,
 const unsigned int i,
 const unsigned int j,
 const unsigned int q,
 const double       nu,
 const double       delta,
 const double       mu,
 const bool         apply_newton_linearization)
{
  const Tensor<2, dim> &velocity_trial_function_gradient{scratch.grad_phi_velocity[j]};
  const Tensor<1, dim> &velocity_trial_function_value{scratch.phi_velocity[j]};

  const Tensor<2, dim> &velocity_test_function_gradient{scratch.grad_phi_velocity[i]};
  const Tensor<1, dim> &velocity_test_function_value{scratch.phi_velocity[i]};

  const double pressure_trial_function{scratch.phi_pressure[j]};
  const double pressure_test_function{scratch.phi_pressure[i]};

  const Tensor<2, dim> &present_velocity_gradient{scratch.present_velocity_gradients[q]};
  const Tensor<1, dim> &present_velocity_value{scratch.present_velocity_values[q]};

  const double velocity_trial_function_divergence{trace(velocity_trial_function_gradient)};
  const double velocity_test_function_divergence{trace(velocity_test_function_gradient)};

  double matrix{-(velocity_trial_function_divergence * pressure_test_function +
                  pressure_trial_function * velocity_test_function_divergence)};

  if (apply_newton_linearization)
    matrix += (present_velocity_gradient * velocity_trial_function_value +
               velocity_trial_function_gradient * present_velocity_value) *
               velocity_test_function_value;
  else
    matrix += velocity_trial_function_gradient * present_velocity_value *
              velocity_test_function_value;

  if (scratch.scalar_options.use_stress_form)
  {
    Assert(scratch.scalar_options.velocity_trial_function_symmetric_gradient,
           ExcMessage("Symmetric velocity trial function gradient was not assigned "
                      "in options"));
    Assert(scratch.scalar_options.velocity_test_function_symmetric_gradient,
           ExcMessage("Symmetric velocity test function gradient was not assigned "
                      "in options"));

    matrix += 2.0 * nu * scalar_product(*scratch.scalar_options.velocity_trial_function_symmetric_gradient,
                                        *scratch.scalar_options.velocity_test_function_symmetric_gradient);
  }
  else
    matrix += nu * scalar_product(velocity_trial_function_gradient,
                                  velocity_test_function_gradient);

  if (scratch.scalar_options.angular_velocity)
  {
    Assert(scratch.scalar_options.rossby_number,
           ExcMessage("Rossby number was not assigned in options."));

    if constexpr(dim == 2)
      matrix += 2.0 / *scratch.scalar_options.rossby_number * scratch.scalar_options.angular_velocity.value()[0] *
                cross_product_2d(-velocity_trial_function_value) *
                velocity_test_function_value;
    else if constexpr(dim == 3)
      matrix += 2.0 / *scratch.scalar_options.rossby_number *
                cross_product_3d(*scratch.scalar_options.angular_velocity, velocity_trial_function_value) *
                velocity_test_function_value;
  }

  matrix += compute_residual_linearization_matrix(stabilization,
                                                  scratch,
                                                  i,
                                                  j,
                                                  q,
                                                  nu,
                                                  delta,
                                                  mu,
                                                  apply_newton_linearization);

  return (matrix);
}



template <int dim>
double compute_rhs
(const StabilizationFlags  &stabilization,
 const AssemblyData::Matrix::ScratchData<dim> &scratch,
 const double       present_pressure_value,
 const unsigned int i,
 const unsigned int q,
 const double       nu,
 const double       mu,
 const double       delta)
{
  const Tensor<2, dim> &velocity_test_function_gradient{scratch.grad_phi_velocity[i]};
  const Tensor<1, dim> &velocity_test_function_value{scratch.phi_velocity[i]};

  const double pressure_test_function{scratch.phi_pressure[i]};
  const Tensor<1, dim> &pressure_test_function_gradient{scratch.grad_phi_pressure[i]};

  const Tensor<2, dim> &present_velocity_gradient{scratch.present_velocity_gradients[q]};
  const Tensor<1, dim> &present_velocity_value{scratch.present_velocity_values[q]};
  const Tensor<1, dim> &present_strong_residual{scratch.present_strong_residuals[q]};

  const double present_velocity_divergence{trace(present_velocity_gradient)};
  const double velocity_test_function_divergence{trace(velocity_test_function_gradient)};

  double rhs{present_velocity_divergence * pressure_test_function +
             present_pressure_value * velocity_test_function_divergence -
             (present_velocity_gradient * present_velocity_value) *
             velocity_test_function_value};

  if (scratch.scalar_options.use_stress_form)
  {
    Assert(scratch.scalar_options.present_symmetric_velocity_gradient,
           ExcMessage("Present symmetric velocity gradient was not assigned "
                      "in options"));
    Assert(scratch.scalar_options.velocity_test_function_symmetric_gradient,
           ExcMessage("Symmetric velocity test function gradient was not assigned "
                      "in options"));

    rhs -= 2.0 * nu * scalar_product(*scratch.scalar_options.present_symmetric_velocity_gradient,
                                     *scratch.scalar_options.velocity_test_function_symmetric_gradient);
  }
  else
    rhs -= nu * scalar_product(present_velocity_gradient,
                               velocity_test_function_gradient);

  if (scratch.scalar_options.body_force_value)
  {
    Assert(scratch.scalar_options.froude_number,
           ExcMessage("Froude number was not assigned in options."));

    rhs -= *scratch.scalar_options.body_force_value * velocity_test_function_value /
           (*scratch.scalar_options.froude_number * *scratch.scalar_options.froude_number);
  }

  if (scratch.scalar_options.angular_velocity)
  {
    Assert(scratch.scalar_options.rossby_number,
           ExcMessage("Rossby number was not assigned in options."));

    if constexpr(dim == 2)
      rhs -= 2.0 / *scratch.scalar_options.rossby_number * scratch.scalar_options.angular_velocity.value()[0] *
             cross_product_2d(-present_velocity_value) * velocity_test_function_value;
    else if constexpr(dim == 3)
      rhs -= 2.0 / *scratch.scalar_options.rossby_number *
             cross_product_3d(*scratch.scalar_options.angular_velocity, present_velocity_value) *
             velocity_test_function_value;
  }

  if (stabilization & (apply_supg|apply_pspg))
  {
    Tensor<1, dim> stabilization_test_function;

    if (stabilization & apply_supg)
      stabilization_test_function += velocity_test_function_gradient *
                                     present_velocity_value;

    if (stabilization & apply_pspg)
      stabilization_test_function += pressure_test_function_gradient;

    rhs -= delta * present_strong_residual * stabilization_test_function;
  }

  if (stabilization & apply_grad_div)
    rhs -= mu * trace(present_velocity_gradient) *
                trace(velocity_test_function_gradient);

  return (rhs);
}




template <int dim>
double compute_rhs
(const StabilizationFlags  &stabilization,
 const AssemblyData::RightHandSide::ScratchData<dim> &scratch,
 const double       present_pressure_value,
 const unsigned int i,
 const unsigned int q,
 const double       nu,
 const double       mu,
 const double       delta)
{
  const Tensor<2, dim> &velocity_test_function_gradient{scratch.grad_phi_velocity[i]};
  const Tensor<1, dim> &velocity_test_function_value{scratch.phi_velocity[i]};

  const double pressure_test_function{scratch.phi_pressure[i]};
  const Tensor<1, dim> &pressure_test_function_gradient{scratch.grad_phi_pressure[i]};

  const Tensor<2, dim> &present_velocity_gradient{scratch.present_velocity_gradients[q]};
  const Tensor<1, dim> &present_velocity_value{scratch.present_velocity_values[q]};
  const Tensor<1, dim> &present_strong_residual{scratch.present_strong_residuals[q]};

  const double present_velocity_divergence{trace(present_velocity_gradient)};
  const double velocity_test_function_divergence{trace(velocity_test_function_gradient)};

  double rhs{present_velocity_divergence * pressure_test_function +
             present_pressure_value * velocity_test_function_divergence -
             (present_velocity_gradient * present_velocity_value) *
             velocity_test_function_value};

  if (scratch.scalar_options.use_stress_form)
  {
    Assert(scratch.scalar_options.present_symmetric_velocity_gradient,
           ExcMessage("Present symmetric velocity gradient was not assigned "
                      "in options"));
    Assert(scratch.scalar_options.velocity_test_function_symmetric_gradient,
           ExcMessage("Symmetric velocity test function gradient was not assigned "
                      "in options"));

    rhs -= 2.0 * nu * scalar_product(*scratch.scalar_options.present_symmetric_velocity_gradient,
                                     *scratch.scalar_options.velocity_test_function_symmetric_gradient);
  }
  else
    rhs -= nu * scalar_product(present_velocity_gradient,
                               velocity_test_function_gradient);

  if (scratch.scalar_options.body_force_value)
  {
    Assert(scratch.scalar_options.froude_number,
           ExcMessage("Froude number was not assigned in options."));

    rhs -= *scratch.scalar_options.body_force_value * velocity_test_function_value /
           (*scratch.scalar_options.froude_number * *scratch.scalar_options.froude_number);
  }

  if (scratch.scalar_options.angular_velocity)
  {
    Assert(scratch.scalar_options.rossby_number,
           ExcMessage("Rossby number was not assigned in options."));

    if constexpr(dim == 2)
      rhs -= 2.0 / *scratch.scalar_options.rossby_number * scratch.scalar_options.angular_velocity.value()[0] *
             cross_product_2d(-present_velocity_value) * velocity_test_function_value;
    else if constexpr(dim == 3)
      rhs -= 2.0 / *scratch.scalar_options.rossby_number *
             cross_product_3d(*scratch.scalar_options.angular_velocity, present_velocity_value) *
             velocity_test_function_value;
  }

  if (stabilization & (apply_supg|apply_pspg))
  {
    Tensor<1, dim> stabilization_test_function;

    if (stabilization & apply_supg)
      stabilization_test_function += velocity_test_function_gradient *
                                     present_velocity_value;

    if (stabilization & apply_pspg)
      stabilization_test_function += pressure_test_function_gradient;

    rhs -= delta * present_strong_residual * stabilization_test_function;
  }

  if (stabilization & apply_grad_div)
    rhs -= mu * trace(present_velocity_gradient) *
                trace(velocity_test_function_gradient);

  return (rhs);
}



template <int dim>
void compute_strong_residual
(const std::vector<Tensor<1, dim>>   &present_velocity_values,
 const std::vector<Tensor<2, dim>>   &present_velocity_gradients,
 const VectorOptions<dim>  &options,
 const double                         nu,
 std::vector<Tensor<1,dim>>          &strong_residuals)
{
  const unsigned int n_q_points{(unsigned int)present_velocity_values.size()};

  AssertDimension(present_velocity_gradients.size(), n_q_points);
  AssertDimension(strong_residuals.size(), n_q_points);

  Assert(options.present_pressure_gradients,
         ExcMessage("Present pressure gradients were not assigned in options."));
  Assert(options.present_velocity_laplaceans,
         ExcMessage("Present velocity laplaceans were not assigned in options."));
  AssertDimension(options.present_pressure_gradients->size(), n_q_points);
  AssertDimension(options.present_velocity_laplaceans->size(), n_q_points);

  const auto &present_pressure_gradients{*options.present_pressure_gradients};
  const auto &present_velocity_laplaceans{*options.present_velocity_laplaceans};

  if (options.use_stress_form)
  {
    Assert(options.present_velocity_grad_divergences,
           ExcMessage("Gradient of present velocity divergences were not assigned in options."));
    AssertDimension(options.present_velocity_grad_divergences->size(), n_q_points);

    for (unsigned int q=0; q<n_q_points; ++q)
      strong_residuals[q] = (present_velocity_gradients[q] * present_velocity_values[q]) -
                            nu * present_velocity_laplaceans[q] -
                            nu * options.present_velocity_grad_divergences->at(q) +
                            present_pressure_gradients[q];
  }
  else
    for (unsigned int q=0; q<n_q_points; ++q)
      strong_residuals[q] = (present_velocity_gradients[q] * present_velocity_values[q]) -
                            nu * present_velocity_laplaceans[q] +
                            present_pressure_gradients[q];

  if (options.body_force_values)
  {
    Assert(options.froude_number,
           ExcMessage("Froude number was not assigned in options."));

    Assert(options.body_force_values,
           ExcMessage("Body force values were not assigned in options."));
    AssertDimension(options.body_force_values->size(), n_q_points);

    for (unsigned int q=0; q<n_q_points; ++q)
      strong_residuals[q] -= options.body_force_values->at(q) / std::pow(*options.froude_number, 2);
  }

  if (options.angular_velocity)
  {
    Assert(options.rossby_number,
           ExcMessage("Rossby number was not assigned in options."));
    Assert(options.angular_velocity,
           ExcMessage("Angular velocity was not assigned in options."));

    if constexpr(dim == 2)
        for (unsigned int q=0; q<n_q_points; ++q)
          strong_residuals[q] += 2.0 / *options.rossby_number * options.angular_velocity.value()[0] *
                                 cross_product_2d(-present_velocity_values[q]);
    else if constexpr(dim == 3)
        for (unsigned int q=0; q<n_q_points; ++q)
          strong_residuals[q] += 2.0 / *options.rossby_number *
                                 cross_product_3d(*options.angular_velocity, present_velocity_values[q]);
  }
}



template <int dim>
double compute_residual_linearization_matrix
(const StabilizationFlags  &stabilization,
 const AssemblyData::Matrix::ScratchData<dim> &scratch,
 const unsigned int test_function_index,
 const unsigned int trial_function_index,
 const unsigned int quadrature_point_index,
 const double       nu,
 const double       delta,
 const double       mu,
 const bool         apply_newton_linearization)
{
  if (!(stabilization & (apply_supg|apply_pspg|apply_grad_div)))
    return (0.0);

  Assert(nu > 0.0, ExcMessage("The viscosity must be positive."));
  Assert(delta > 0.0, ExcMessage("The SUPG stabilization parameter must be positive."));
  Assert(mu > 0.0, ExcMessage("The GradDiv stabilization parameter must be positive."));

  const Tensor<2, dim> &velocity_trial_function_gradient{scratch.grad_phi_velocity[trial_function_index]};
  const Tensor<1, dim> &velocity_trial_function_value{scratch.phi_velocity[trial_function_index]};
  const Tensor<1, dim> &velocity_trial_function_laplacean{scratch.laplace_phi_velocity[trial_function_index]};

  const Tensor<2, dim> &velocity_test_function_gradient{scratch.grad_phi_velocity[test_function_index]};

  const Tensor<1, dim> &pressure_trial_function_gradient{scratch.grad_phi_pressure[trial_function_index]};
  const Tensor<1, dim> &pressure_test_function_gradient{scratch.grad_phi_pressure[test_function_index]};

  const Tensor<2, dim> &present_velocity_gradient{scratch.present_velocity_gradients[quadrature_point_index]};
  const Tensor<1, dim> &present_velocity_value{scratch.present_velocity_values[quadrature_point_index]};
  const Tensor<1, dim> &present_strong_residual{scratch.present_strong_residuals[quadrature_point_index]};

  double matrix{0.0};

  if (stabilization & (apply_supg|apply_pspg))
  {
    // linearized residual
    Tensor<1, dim> linearized_residual
    {velocity_trial_function_gradient * present_velocity_value +
     pressure_trial_function_gradient};

    if (apply_newton_linearization)
      linearized_residual += present_velocity_gradient * velocity_trial_function_value;

    if (scratch.scalar_options.use_stress_form)
    {
      Assert(scratch.scalar_options.velocity_trial_function_grad_divergence,
             ExcMessage("Gradient of velocity trial function divergence was not "
                        "specified in options."));

      linearized_residual -= nu * (velocity_trial_function_laplacean +
                                   *scratch.scalar_options.velocity_trial_function_grad_divergence);
    }
    else
      linearized_residual -= nu * velocity_trial_function_laplacean;

    if (scratch.scalar_options.angular_velocity)
    {
      Assert(scratch.scalar_options.rossby_number,
             ExcMessage("Rossby number was not assigned in options."));

      if constexpr(dim == 2)
        linearized_residual += 2.0 / *scratch.scalar_options.rossby_number * scratch.scalar_options.angular_velocity.value()[0] *
                              cross_product_2d(-velocity_trial_function_value);
      else if constexpr(dim == 3)
        linearized_residual += 2.0 / *scratch.scalar_options.rossby_number *
                               cross_product_3d(*scratch.scalar_options.angular_velocity, velocity_trial_function_value);
    }

    Tensor<1, dim> test_function;
    if (stabilization & apply_supg)
      test_function += velocity_test_function_gradient *
                       present_velocity_value;

    if (stabilization & apply_pspg)
      test_function += pressure_test_function_gradient;

    matrix += delta * (linearized_residual * test_function);

    if (stabilization & apply_supg)
      matrix += delta * present_strong_residual *
                (velocity_test_function_gradient * velocity_trial_function_value);
  }

  if (stabilization & apply_grad_div)
    matrix += mu * trace(velocity_trial_function_gradient) *
                   trace(velocity_test_function_gradient);

  return (matrix);

}



// explicit instantiations
template
double compute_matrix
(const StabilizationFlags  &,
 const AssemblyData::Matrix::ScratchData<2> &,
 const unsigned int ,
 const unsigned int ,
 const unsigned int ,
 const double       ,
 const double       ,
 const double       ,
 const bool         );
template
double compute_matrix
(const StabilizationFlags  &,
 const AssemblyData::Matrix::ScratchData<3> &,
 const unsigned int ,
 const unsigned int ,
 const unsigned int ,
 const double       ,
 const double       ,
 const double       ,
 const bool         );

template
double
compute_rhs
(const StabilizationFlags  &,
 const AssemblyData::Matrix::ScratchData<2> &,
 const double       ,
 const unsigned int ,
 const unsigned int ,
 const double       ,
 const double       ,
 const double        );
template
double
compute_rhs
(const StabilizationFlags  &,
 const AssemblyData::Matrix::ScratchData<3> &,
 const double       ,
 const unsigned int ,
 const unsigned int ,
 const double       ,
 const double       ,
 const double        );

template
double
compute_rhs
(const StabilizationFlags  &,
 const AssemblyData::RightHandSide::ScratchData<2> &,
 const double       ,
 const unsigned int ,
 const unsigned int ,
 const double       ,
 const double       ,
 const double        );
template
double
compute_rhs
(const StabilizationFlags  &,
 const AssemblyData::RightHandSide::ScratchData<3> &,
 const double       ,
 const unsigned int ,
 const unsigned int ,
 const double       ,
 const double       ,
 const double        );

template
void
compute_strong_residual
(const std::vector<Tensor<1, 2>>  &,
 const std::vector<Tensor<2, 2>>  &,
 const VectorOptions<2> &,
 const double                      ,
 std::vector<Tensor<1,2>>         &);
template
void compute_strong_residual
(const std::vector<Tensor<1, 3>>   &,
 const std::vector<Tensor<2, 3>>   &,
 const VectorOptions<3>  &,
 const double                       ,
 std::vector<Tensor<1, 3>>          &);

template
double
compute_residual_linearization_matrix
(const StabilizationFlags  &,
 const AssemblyData::Matrix::ScratchData<2> &,
 const unsigned int ,
 const unsigned int ,
 const unsigned int ,
 const double       ,
 const double       ,
 const double       ,
 const bool         );
template
double
compute_residual_linearization_matrix
(const StabilizationFlags  &,
 const AssemblyData::Matrix::ScratchData<3> &,
 const unsigned int ,
 const unsigned int ,
 const unsigned int ,
 const double       ,
 const double       ,
 const double       ,
 const bool         );



}  // namespace Hydrodynamic




namespace BuoyantHydrodynamic {

template <int dim>
double compute_hydrodynamic_matrix
(const Tensor<1, dim> &velocity_trial_function_value,
 const Tensor<2, dim> &velocity_trial_function_gradient,
 const Tensor<1, dim> &velocity_test_function_value,
 const Tensor<2, dim> &velocity_test_function_gradient,
 const Tensor<1, dim> &present_velocity_value,
 const Tensor<2, dim> &present_velocity_gradient,
 const double          pressure_trial_function,
 const double          density_trial_function_value,
 const double          pressure_test_function,
 const double          nu,
 const Hydrodynamic::ScalarOptions<dim>        &options,
 const BuoyantHydrodynamic::ScalarOptions<dim> &buoyancy_options,
 const bool            apply_newton_linearization)
{
  double matrix = LegacyHydrodynamic::
                  compute_matrix(velocity_trial_function_value,
                                 velocity_trial_function_gradient,
                                 velocity_test_function_value,
                                 velocity_test_function_gradient,
                                 present_velocity_value,
                                 present_velocity_gradient,
                                 pressure_trial_function,
                                 pressure_test_function,
                                 nu,
                                 options,
                                 apply_newton_linearization);

  if (buoyancy_options.gravity_field_value)
  {
    Assert(options.froude_number, ExcInternalError());

    matrix -= density_trial_function_value * *buoyancy_options.gravity_field_value *
              velocity_test_function_value /
              (*options.froude_number * *options.froude_number);
  }

  return (matrix);
}



template <int dim>
double compute_hydrodynamic_residual_linearization_matrix
(const StabilizationFlags  &stabilization,
 const Tensor<1, dim>      &velocity_trial_function_value,
 const Tensor<2, dim>      &velocity_trial_function_gradient,
 const Tensor<1, dim>      &velocity_trial_function_laplacean,
 const Tensor<1, dim>      &pressure_trial_function_gradient,
 const Tensor<1, dim>      &present_velocity_value,
 const Tensor<2, dim>      &present_velocity_gradient,
 const Tensor<1, dim>      &present_strong_residual,
 const double               density_trial_function_value,
 const Tensor<2, dim>      &velocity_test_function_gradient,
 const Tensor<1, dim>      &pressure_test_function_gradient,
 const double               nu,
 const double               delta,
 const double               mu,
 const Hydrodynamic::ScalarOptions<dim>        &options,
 const BuoyantHydrodynamic::ScalarOptions<dim> &buoyancy_options,
 const bool                 apply_newton_linearization)
{
  if (!(stabilization & (apply_supg|apply_pspg|apply_grad_div)))
    return (0.0);

  double matrix = LegacyHydrodynamic::
                  compute_residual_linearization_matrix(stabilization,
                                                        velocity_trial_function_value,
                                                        velocity_trial_function_gradient,
                                                        velocity_trial_function_laplacean,
                                                        pressure_trial_function_gradient,
                                                        present_velocity_value,
                                                        present_velocity_gradient,
                                                        present_strong_residual,
                                                        velocity_test_function_gradient,
                                                        pressure_test_function_gradient,
                                                        nu,
                                                        delta,
                                                        mu,
                                                        options,
                                                        apply_newton_linearization);

  if (buoyancy_options.gravity_field_value)
  {
    Assert(options.froude_number, ExcInternalError());

    Tensor<1, dim> test_function;
    if (stabilization & apply_supg)
      test_function += velocity_test_function_gradient * present_velocity_value;
    if (stabilization & apply_pspg)
      test_function += pressure_test_function_gradient;

    matrix -= delta * density_trial_function_value * *buoyancy_options.gravity_field_value *
              test_function /
              (*options.froude_number * *options.froude_number);
  }

  return (matrix);

}



template <int dim>
double compute_hydrodynamic_rhs
(const StabilizationFlags & stabilization,
 const Tensor<1, dim> &velocity_test_function_value,
 const Tensor<2, dim> &velocity_test_function_gradient,
 const Tensor<1, dim> &present_velocity_value,
 const Tensor<2, dim> &present_velocity_gradient,
 const Tensor<1, dim> &present_strong_residual,
 const double          present_pressure_value,
 const double          present_density_value,
 const double          pressure_test_function,
 const Tensor<1, dim> &pressure_test_function_gradient,
 const double          nu,
 const double          mu,
 const double          delta,
 const Hydrodynamic::ScalarOptions<dim>        &options,
 const BuoyantHydrodynamic::ScalarOptions<dim> &buoyancy_options)
{
  double rhs = LegacyHydrodynamic::
               compute_rhs(stabilization,
                           velocity_test_function_value,
                           velocity_test_function_gradient,
                           present_velocity_value,
                           present_velocity_gradient,
                           present_strong_residual,
                           present_pressure_value,
                           pressure_test_function,
                           pressure_test_function_gradient,
                           nu,
                           mu,
                           delta,
                           options);

  if (buoyancy_options.gravity_field_value)
  {
    Assert(options.froude_number, ExcInternalError());

    rhs += present_density_value * *buoyancy_options.gravity_field_value *
           velocity_test_function_value /
           (*options.froude_number * *options.froude_number);
  }

  return (rhs);
}



template <int dim>
void compute_strong_hydrodynamic_residual
(const std::vector<Tensor<1, dim>> &present_velocity_values,
 const std::vector<Tensor<2, dim>> &present_velocity_gradients,
 const std::vector<double>         &present_density_values,
 std::vector<Tensor<1, dim>>       &strong_residuals,
 const double                       nu,
 const Hydrodynamic::VectorOptions<dim>        &options,
 const BuoyantHydrodynamic::VectorOptions<dim> &buoyancy_options)
{
  Hydrodynamic::
  compute_strong_residual(present_velocity_values,
                          present_velocity_gradients,
                          options,
                          nu,
                          strong_residuals);

  if (buoyancy_options.gravity_field_values)
  {
    Assert(options.froude_number, ExcInternalError());

    for (std::size_t q=0; q<present_velocity_values.size(); ++q)
      strong_residuals[q] -= present_density_values[q] *
                             buoyancy_options.gravity_field_values->at(q) /
                             (*options.froude_number * *options.froude_number);
  }
}



template <int dim>
void compute_strong_density_residual
(const std::vector<Tensor<1, dim>>             &present_density_gradients,
 const std::vector<Tensor<1, dim>>             &present_velocity_values,
 std::vector<double>                           &strong_residuals,
 const Advection::VectorOptions<dim> &advection_options)
{
  Advection::compute_strong_residual(present_density_gradients,
                                     present_velocity_values,
                                     strong_residuals,
                                     advection_options);
}



template <int dim>
double compute_density_matrix
(const Tensor<1, dim> &density_trial_function_gradient,
 const Tensor<1, dim> &velocity_trial_function_value,
 const Tensor<1, dim> &present_density_gradient,
 const Tensor<1, dim> &present_velocity_value,
 const double          density_test_function_value,
 const Advection::ScalarOptions<dim>      &advection_options,
 const bool            apply_newton_linearization)
{
  double linearized_residual =
      present_velocity_value * density_trial_function_gradient +
      (apply_newton_linearization?
          velocity_trial_function_value * present_density_gradient:
          0.0);

  if (advection_options.reference_gradient && apply_newton_linearization)
  {
    Assert(advection_options.gradient_scaling, ExcInternalError());

    linearized_residual += *advection_options.gradient_scaling *
                           velocity_trial_function_value *
                           *advection_options.reference_gradient;
  }

  return (linearized_residual * density_test_function_value);
}



template <int dim>
double compute_density_rhs
(const Tensor<1, dim> &present_density_gradient,
 const Tensor<1, dim> &present_velocity_value,
 const double          present_strong_residual,
 const double          density_test_function_value,
 const Tensor<1, dim> &density_test_function_gradient,
 const double          delta,
 const Advection::ScalarOptions<dim>  &advection_options)
{
  double residual = -(present_velocity_value * present_density_gradient);

  if (advection_options.reference_gradient)
  {
    Assert(advection_options.gradient_scaling, ExcInternalError());

    residual -= *advection_options.gradient_scaling *
                present_velocity_value *
                *advection_options.reference_gradient;
  }

  double rhs{residual * density_test_function_value};

  // standard stabilization terms
  {
    double stabilization_test_function{present_velocity_value *  density_test_function_gradient};

    rhs -= delta * present_strong_residual * stabilization_test_function;
  }

  return (rhs);
}



template <int dim>
double compute_density_residual_linearization_matrix
(const Tensor<1, dim> &density_trial_function_gradient,
 const Tensor<1, dim> &velocity_trial_function_value,
 const Tensor<1, dim> &density_test_function_gradient,
 const Tensor<1, dim> &present_velocity_value,
 const Tensor<1, dim> &present_density_gradient,
 const double          present_strong_residual,
 const double          delta,
 const double          nu,
 const Advection::ScalarOptions<dim>    &advection_options,
 const bool            apply_newton_linearization)
{
  double matrix{0.0};

  if (present_velocity_value.norm() > 0.0)
  {
    double linearized_residual =
        present_velocity_value * density_trial_function_gradient +
        (apply_newton_linearization?
            velocity_trial_function_value * present_density_gradient:
            0.0);


    if (advection_options.reference_gradient && apply_newton_linearization)
    {
      Assert(advection_options.gradient_scaling, ExcInternalError());

      linearized_residual += *advection_options.gradient_scaling *
                             velocity_trial_function_value *
                             *advection_options.reference_gradient;
    }

    matrix = delta * linearized_residual * (density_test_function_gradient * present_velocity_value);
  }
  else
    matrix = nu * density_trial_function_gradient * density_test_function_gradient;

  matrix += delta * present_strong_residual *
            (velocity_trial_function_value * density_test_function_gradient);

  return (matrix);
}



// explicit instantiations
template
double compute_hydrodynamic_matrix
(const Tensor<1, 2> &,
 const Tensor<2, 2> &,
 const Tensor<1, 2> &,
 const Tensor<2, 2> &,
 const Tensor<1, 2> &,
 const Tensor<2, 2> &,
 const double        ,
 const double        ,
 const double        ,
 const double        ,
 const Hydrodynamic::ScalarOptions<2> &,
 const ScalarOptions<2> &,
 const bool           );
template
double compute_hydrodynamic_matrix
(const Tensor<1, 3> &,
 const Tensor<2, 3> &,
 const Tensor<1, 3> &,
 const Tensor<2, 3> &,
 const Tensor<1, 3> &,
 const Tensor<2, 3> &,
 const double        ,
 const double        ,
 const double        ,
 const double        ,
 const Hydrodynamic::ScalarOptions<3> &,
 const ScalarOptions<3> &,
 const bool           );

template
double
compute_hydrodynamic_residual_linearization_matrix
(const StabilizationFlags &,
 const Tensor<1, 2>       &,
 const Tensor<2, 2>       &,
 const Tensor<1, 2>       &,
 const Tensor<1, 2>       &,
 const Tensor<1, 2>       &,
 const Tensor<2, 2>       &,
 const Tensor<1, 2>       &,
 const double              ,
 const Tensor<2, 2>       &,
 const Tensor<1, 2>       &,
 const double              ,
 const double              ,
 const double              ,
 const Hydrodynamic::ScalarOptions<2>        &,
 const BuoyantHydrodynamic::ScalarOptions<2> &,
 const bool                 );
template
double
compute_hydrodynamic_residual_linearization_matrix
(const StabilizationFlags &,
 const Tensor<1, 3>       &,
 const Tensor<2, 3>       &,
 const Tensor<1, 3>       &,
 const Tensor<1, 3>       &,
 const Tensor<1, 3>       &,
 const Tensor<2, 3>       &,
 const Tensor<1, 3>       &,
 const double              ,
 const Tensor<2, 3>       &,
 const Tensor<1, 3>       &,
 const double              ,
 const double              ,
 const double              ,
 const Hydrodynamic::ScalarOptions<3>        &,
 const BuoyantHydrodynamic::ScalarOptions<3> &,
 const bool                 );


template
double
compute_hydrodynamic_rhs
(const StabilizationFlags &,
 const Tensor<1, 2>       &,
 const Tensor<2, 2>       &,
 const Tensor<1, 2>       &,
 const Tensor<2, 2>       &,
 const Tensor<1, 2>       &,
 const double              ,
 const double              ,
 const double              ,
 const Tensor<1, 2>       &,
 const double              ,
 const double              ,
 const double              ,
 const Hydrodynamic::ScalarOptions<2>        &,
 const BuoyantHydrodynamic::ScalarOptions<2> & );
template
double
compute_hydrodynamic_rhs
(const StabilizationFlags &,
 const Tensor<1, 3>       &,
 const Tensor<2, 3>       &,
 const Tensor<1, 3>       &,
 const Tensor<2, 3>       &,
 const Tensor<1, 3>       &,
 const double              ,
 const double              ,
 const double              ,
 const Tensor<1, 3>       &,
 const double              ,
 const double              ,
 const double              ,
 const Hydrodynamic::ScalarOptions<3>        &,
 const BuoyantHydrodynamic::ScalarOptions<3> & );

template
void
compute_strong_hydrodynamic_residual
(const std::vector<Tensor<1, 2>>  &,
 const std::vector<Tensor<2, 2>>  &,
 const std::vector<double>        &,
 std::vector<Tensor<1, 2>>        &,
 const double                      ,
 const Hydrodynamic::VectorOptions<2>        &,
 const BuoyantHydrodynamic::VectorOptions<2> & );
template
void
compute_strong_hydrodynamic_residual
(const std::vector<Tensor<1, 3>>  &,
 const std::vector<Tensor<2, 3>>  &,
 const std::vector<double>        &,
 std::vector<Tensor<1, 3>>        &,
 const double                      ,
 const Hydrodynamic::VectorOptions<3>        &,
 const BuoyantHydrodynamic::VectorOptions<3> & );

template
void
compute_strong_density_residual
(const std::vector<Tensor<1, 2>>             &,
 const std::vector<Tensor<1, 2>>             &,
 std::vector<double>                         &,
 const Advection::VectorOptions<2> &);
template
void
compute_strong_density_residual
(const std::vector<Tensor<1, 3>>             &,
 const std::vector<Tensor<1, 3>>             &,
 std::vector<double>                         &,
 const Advection::VectorOptions<3> &);

template
double
compute_density_matrix
(const Tensor<1, 2> &,
 const Tensor<1, 2> &,
 const Tensor<1, 2> &,
 const Tensor<1, 2> &,
 const double        ,
 const Advection::ScalarOptions<2>  &,
 const bool                                     );
template
double
compute_density_matrix
(const Tensor<1, 3> &,
 const Tensor<1, 3> &,
 const Tensor<1, 3> &,
 const Tensor<1, 3> &,
 const double        ,
 const Advection::ScalarOptions<3>  &,
 const bool                                     );

template
double
compute_density_rhs
(const Tensor<1, 2> &,
 const Tensor<1, 2> &,
 const double        ,
 const double        ,
 const Tensor<1, 2> &,
 const double        ,
 const Advection::ScalarOptions<2> &);
template
double
compute_density_rhs
(const Tensor<1, 3> &,
 const Tensor<1, 3> &,
 const double        ,
 const double        ,
 const Tensor<1, 3> &,
 const double        ,
 const Advection::ScalarOptions<3> &);

template
double
compute_density_residual_linearization_matrix
(const Tensor<1, 2> &,
 const Tensor<1, 2> &,
 const Tensor<1, 2> &,
 const Tensor<1, 2> &,
 const Tensor<1, 2> &,
 const double        ,
 const double        ,
 const double        ,
 const Advection::ScalarOptions<2> &,
 const bool           );
template
double
compute_density_residual_linearization_matrix
(const Tensor<1, 3> &,
 const Tensor<1, 3> &,
 const Tensor<1, 3> &,
 const Tensor<1, 3> &,
 const Tensor<1, 3> &,
 const double        ,
 const double        ,
 const double        ,
 const Advection::ScalarOptions<3> &,
 const bool           );

}  // namespace BuoyantHydrodynamic



namespace Advection {

/*!
 * @brief Computes the matrix entry of the advection equation.
 *
 * @attention The advection field must include contributions due to a possible
 * background field.
 *
 * @attention The test function must include contributions related to
 * stabilization terms.
 *
 */
template <int dim>
double compute_matrix
(const AssemblyData::ScratchData<dim> &scratch,
 const unsigned int i,
 const unsigned int j,
 const unsigned int q,
 const double       delta)
{
  double matrix{(scratch.advection_field_values[q] * scratch.grad_phi[j]) * scratch.phi[i]};

  matrix += compute_residual_linearization_matrix(scratch, i, j, q, delta);

  return (matrix);
}



/*!
 * @brief Computes the right-hand side entry of the advection equation.
 *
 * @attention The advection field must include contributions due to a possible
 * background field.
 *
 * @attention The test function must include contributions related to
 * stabilization terms.
 *
 */
template <int dim>
double compute_rhs
(const AssemblyData::ScratchData<dim> &scratch,
 const Tensor<1, dim>  &present_gradient,
 const unsigned int     i,
 const unsigned int     q,
 const double           delta)
{
  const Tensor<1, dim> &advection_field_value{scratch.advection_field_values[q]};

  double rhs{-(advection_field_value * present_gradient)};

  if (scratch.scalar_options.reference_gradient)
  {
    Assert(scratch.scalar_options.gradient_scaling,
           ExcMessage("Gradient scaling number was not were not assigned in options."));

    rhs -= *scratch.scalar_options.gradient_scaling *
            (advection_field_value * *scratch.scalar_options.reference_gradient);
  }

  if (scratch.scalar_options.source_term_value)
    rhs += *scratch.scalar_options.source_term_value;

  rhs *= scratch.phi[i];

  const double stabilization_test_function{advection_field_value * scratch.grad_phi[i]};

  rhs -= delta * scratch.present_strong_residuals[q] * stabilization_test_function;

  return (rhs);
}



/*!
 * @brief Computes the strong residual of the advection equation.
 *
 * @attention The advection field must include contributions due to a possible
 * background field.
 *
 */
template<int dim>
void compute_strong_residual
(const std::vector<Tensor<1, dim>>   &present_gradients,
 const std::vector<Tensor<1, dim>>   &advection_field_values,
 std::vector<double>                 &strong_residuals,
 const VectorOptions<dim>  &options)
{
  const unsigned int n_q_points{(unsigned int)present_gradients.size()};

  AssertDimension(advection_field_values.size(), n_q_points);
  AssertDimension(strong_residuals.size(), n_q_points);

  for (unsigned int q=0; q<n_q_points; ++q)
    strong_residuals[q] = advection_field_values[q] * present_gradients[q];

  if (options.reference_gradients)
  {
    Assert(options.gradient_scaling,
           ExcMessage("Gradient scaling number was not were not assigned in options."));
    AssertDimension(options.reference_gradients->size(), n_q_points);

    for (unsigned int q=0; q<n_q_points; ++q)
      strong_residuals[q] += *options.gradient_scaling *
                              advection_field_values[q] *
                              options.reference_gradients->at(q);
  }

  if (options.source_term_values)
  {
    AssertDimension(options.source_term_values->size(), n_q_points);

    for (unsigned int q=0; q<n_q_points; ++q)
      strong_residuals[q] -= options.source_term_values->at(q);
  }

}



template <int dim>
double compute_residual_linearization_matrix
(const AssemblyData::ScratchData<dim> &scratch,
 const unsigned int j,
 const unsigned int i,
 const unsigned int q,
 const double       delta)
{
  const Tensor<1, dim> &advection_field_value{scratch.advection_field_values[q]};

  return (delta *
          (advection_field_value * scratch.grad_phi[i]) *
          (advection_field_value * scratch.grad_phi[j]));
}



// explicit instantiations
template
double
compute_matrix
(const AssemblyData::ScratchData<2> &,
 const unsigned int ,
 const unsigned int ,
 const unsigned int ,
 const double        );
template
double
compute_matrix
(const AssemblyData::ScratchData<3> &,
 const unsigned int ,
 const unsigned int ,
 const unsigned int ,
 const double        );

template
double
compute_rhs
(const AssemblyData::ScratchData<2> &,
 const Tensor<1, 2>  &,
 const unsigned int   ,
 const unsigned int   ,
 const double          );
template
double
compute_rhs
(const AssemblyData::ScratchData<3> &,
 const Tensor<1, 3>  &,
 const unsigned int   ,
 const unsigned int   ,
 const double          );

template
void
compute_strong_residual
(const std::vector<Tensor<1, 2>>    &,
 const std::vector<Tensor<1, 2>>    &,
 std::vector<double>                &,
 const VectorOptions<2>   &);
template
void
compute_strong_residual
(const std::vector<Tensor<1, 3>>    &,
 const std::vector<Tensor<1, 3>>    &,
 std::vector<double>                &,
 const VectorOptions<3>   &);

template
double
compute_residual_linearization_matrix
(const AssemblyData::ScratchData<2> &,
 const unsigned int ,
 const unsigned int ,
 const unsigned int ,
 const double       );
template
double
compute_residual_linearization_matrix
(const AssemblyData::ScratchData<3> &,
 const unsigned int ,
 const unsigned int ,
 const unsigned int ,
 const double       );

}  // namespace Advection



namespace LegacyHydrodynamic {

template <int dim>
double compute_matrix
(const Tensor<1, dim> &velocity_trial_function_value,
 const Tensor<2, dim> &velocity_trial_function_gradient,
 const Tensor<1, dim> &velocity_test_function_value,
 const Tensor<2, dim> &velocity_test_function_gradient,
 const Tensor<1, dim> &present_velocity_value,
 const Tensor<2, dim> &present_velocity_gradient,
 const double          pressure_trial_function,
 const double          pressure_test_function,
 const double          nu,
 const Hydrodynamic::ScalarOptions<dim> &options,
 const bool            apply_newton_linearization)
{
  const double velocity_trial_function_divergence{trace(velocity_trial_function_gradient)};
  const double velocity_test_function_divergence{trace(velocity_test_function_gradient)};

  double matrix{-(velocity_trial_function_divergence * pressure_test_function +
                  pressure_trial_function * velocity_test_function_divergence)};

  if (apply_newton_linearization)
    matrix += (present_velocity_gradient * velocity_trial_function_value +
               velocity_trial_function_gradient * present_velocity_value) *
               velocity_test_function_value;
  else
    matrix += velocity_trial_function_gradient * present_velocity_value *
              velocity_test_function_value;

  if (options.use_stress_form)
  {
    Assert(options.velocity_trial_function_symmetric_gradient,
           ExcMessage("Symmetric velocity trial function gradient was not assigned "
                      "in options"));
    Assert(options.velocity_test_function_symmetric_gradient,
           ExcMessage("Symmetric velocity test function gradient was not assigned "
                      "in options"));

    matrix += 2.0 * nu * scalar_product(*options.velocity_trial_function_symmetric_gradient,
                                        *options.velocity_test_function_symmetric_gradient);
  }
  else
    matrix += nu * scalar_product(velocity_trial_function_gradient,
                                  velocity_test_function_gradient);

  if (options.angular_velocity)
  {
    Assert(options.rossby_number,
           ExcMessage("Rossby number was not assigned in options."));

    if constexpr(dim == 2)
      matrix += 2.0 / *options.rossby_number * options.angular_velocity.value()[0] *
                cross_product_2d(-velocity_trial_function_value) *
                velocity_test_function_value;
    else if constexpr(dim == 3)
      matrix += 2.0 / *options.rossby_number *
                cross_product_3d(*options.angular_velocity, velocity_trial_function_value) *
                velocity_test_function_value;
  }

  return (matrix);
}



template <int dim>
double compute_rhs
(const StabilizationFlags  &stabilization,
 const Tensor<1, dim>      &velocity_test_function_value,
 const Tensor<2, dim>      &velocity_test_function_gradient,
 const Tensor<1, dim>      &present_velocity_value,
 const Tensor<2, dim>      &present_velocity_gradient,
 const Tensor<1, dim>      &present_strong_residual,
 const double               present_pressure_value,
 const double               pressure_test_function,
 const Tensor<1, dim>      &pressure_test_function_gradient,
 const double               nu,
 const double               mu,
 const double               delta,
 const Hydrodynamic::ScalarOptions<dim> &options)
{
  const double present_velocity_divergence{trace(present_velocity_gradient)};
  const double velocity_test_function_divergence{trace(velocity_test_function_gradient)};

  double rhs{present_velocity_divergence * pressure_test_function +
             present_pressure_value * velocity_test_function_divergence -
             (present_velocity_gradient * present_velocity_value) *
             velocity_test_function_value};

  if (options.use_stress_form)
  {
    Assert(options.present_symmetric_velocity_gradient,
           ExcMessage("Present symmetric velocity gradient was not assigned "
                      "in options"));
    Assert(options.velocity_test_function_symmetric_gradient,
           ExcMessage("Symmetric velocity test function gradient was not assigned "
                      "in options"));

    rhs -= 2.0 * nu * scalar_product(*options.present_symmetric_velocity_gradient,
                                     *options.velocity_test_function_symmetric_gradient);
  }
  else
    rhs -= nu * scalar_product(present_velocity_gradient,
                               velocity_test_function_gradient);

  if (options.body_force_value)
  {
    Assert(options.froude_number,
           ExcMessage("Froude number was not assigned in options."));

    rhs -= *options.body_force_value * velocity_test_function_value /
           (*options.froude_number * *options.froude_number);
  }

  if (options.angular_velocity)
  {
    Assert(options.rossby_number,
           ExcMessage("Rossby number was not assigned in options."));

    if constexpr(dim == 2)
      rhs -= 2.0 / *options.rossby_number * options.angular_velocity.value()[0] *
             cross_product_2d(-present_velocity_value) * velocity_test_function_value;
    else if constexpr(dim == 3)
      rhs -= 2.0 / *options.rossby_number *
             cross_product_3d(*options.angular_velocity, present_velocity_value) *
             velocity_test_function_value;
  }

  if (stabilization & (apply_supg|apply_pspg))
  {
    Tensor<1, dim> stabilization_test_function;

    if (stabilization & apply_supg)
      stabilization_test_function += velocity_test_function_gradient *
                                     present_velocity_value;

    if (stabilization & apply_pspg)
      stabilization_test_function += pressure_test_function_gradient;

    rhs -= delta * present_strong_residual * stabilization_test_function;
  }

  if (stabilization & apply_grad_div)
    rhs -= mu * trace(present_velocity_gradient) *
                trace(velocity_test_function_gradient);

  return (rhs);
}



template <int dim>
void compute_strong_residual
(const std::vector<Tensor<1, dim>>   &present_velocity_values,
 const std::vector<Tensor<2, dim>>   &present_velocity_gradients,
 const Hydrodynamic::VectorOptions<dim>  &options,
 const double                         nu,
 std::vector<Tensor<1,dim>>          &strong_residuals)
{
  const unsigned int n_q_points{(unsigned int)present_velocity_values.size()};

  AssertDimension(present_velocity_gradients.size(), n_q_points);
  AssertDimension(strong_residuals.size(), n_q_points);

  Assert(options.present_pressure_gradients,
         ExcMessage("Present pressure gradients were not assigned in options."));
  Assert(options.present_velocity_laplaceans,
         ExcMessage("Present velocity laplaceans were not assigned in options."));
  AssertDimension(options.present_pressure_gradients->size(), n_q_points);
  AssertDimension(options.present_velocity_laplaceans->size(), n_q_points);

  const auto &present_pressure_gradients{*options.present_pressure_gradients};
  const auto &present_velocity_laplaceans{*options.present_velocity_laplaceans};

  if (options.use_stress_form)
  {
    Assert(options.present_velocity_grad_divergences,
           ExcMessage("Gradient of present velocity divergences were not assigned in options."));
    AssertDimension(options.present_velocity_grad_divergences->size(), n_q_points);

    for (unsigned int q=0; q<n_q_points; ++q)
      strong_residuals[q] = (present_velocity_gradients[q] * present_velocity_values[q]) -
                            nu * present_velocity_laplaceans[q] -
                            nu * options.present_velocity_grad_divergences->at(q) +
                            present_pressure_gradients[q];
  }
  else
    for (unsigned int q=0; q<n_q_points; ++q)
      strong_residuals[q] = (present_velocity_gradients[q] * present_velocity_values[q]) -
                            nu * present_velocity_laplaceans[q] +
                            present_pressure_gradients[q];

  if (options.body_force_values)
  {
    Assert(options.froude_number,
           ExcMessage("Froude number was not assigned in options."));

    Assert(options.body_force_values,
           ExcMessage("Body force values were not assigned in options."));
    AssertDimension(options.body_force_values->size(), n_q_points);

    for (unsigned int q=0; q<n_q_points; ++q)
      strong_residuals[q] -= options.body_force_values->at(q) / std::pow(*options.froude_number, 2);
  }

  if (options.angular_velocity)
  {
    Assert(options.rossby_number,
           ExcMessage("Rossby number was not assigned in options."));
    Assert(options.angular_velocity,
           ExcMessage("Angular velocity was not assigned in options."));

    if constexpr(dim == 2)
        for (unsigned int q=0; q<n_q_points; ++q)
          strong_residuals[q] += 2.0 / *options.rossby_number * options.angular_velocity.value()[0] *
                                 cross_product_2d(-present_velocity_values[q]);
    else if constexpr(dim == 3)
        for (unsigned int q=0; q<n_q_points; ++q)
          strong_residuals[q] += 2.0 / *options.rossby_number *
                                 cross_product_3d(*options.angular_velocity, present_velocity_values[q]);
  }
}



template <int dim>
double compute_residual_linearization_matrix
(const StabilizationFlags  &stabilization,
 const Tensor<1, dim>      &velocity_trial_function_value,
 const Tensor<2, dim>      &velocity_trial_function_gradient,
 const Tensor<1, dim>      &velocity_trial_function_laplacean,
 const Tensor<1, dim>      &pressure_trial_function_gradient,
 const Tensor<1, dim>      &present_velocity_value,
 const Tensor<2, dim>      &present_velocity_gradient,
 const Tensor<1, dim>      &present_strong_residual,
 const Tensor<2, dim>      &velocity_test_function_gradient,
 const Tensor<1, dim>      &pressure_test_function_gradient,
 const double               nu,
 const double               delta,
 const double               mu,
 const Hydrodynamic::ScalarOptions<dim> &options,
 const bool                 apply_newton_linearization)
{
  if (!(stabilization & (apply_supg|apply_pspg|apply_grad_div)))
    return (0.0);

  Assert(nu > 0.0, ExcMessage("The viscosity must be positive."));
  Assert(delta > 0.0, ExcMessage("The SUPG stabilization parameter must be positive."));
  Assert(mu > 0.0, ExcMessage("The GradDiv stabilization parameter must be positive."));

  double matrix{0.0};

  if (stabilization & (apply_supg|apply_pspg))
  {
    // linearized residual
    Tensor<1, dim> linearized_residual
    {velocity_trial_function_gradient * present_velocity_value +
     pressure_trial_function_gradient};

    if (apply_newton_linearization)
      linearized_residual += present_velocity_gradient * velocity_trial_function_value;

    if (options.use_stress_form)
    {
      Assert(options.velocity_trial_function_grad_divergence,
             ExcMessage("Gradient of velocity trial function divergence was not "
                        "specified in options."));

      linearized_residual -= nu * (velocity_trial_function_laplacean +
                                   *options.velocity_trial_function_grad_divergence);
    }
    else
      linearized_residual -= nu * velocity_trial_function_laplacean;

    if (options.angular_velocity)
    {
      Assert(options.rossby_number,
             ExcMessage("Rossby number was not assigned in options."));

      if constexpr(dim == 2)
        linearized_residual += 2.0 / *options.rossby_number * options.angular_velocity.value()[0] *
                              cross_product_2d(-velocity_trial_function_value);
      else if constexpr(dim == 3)
        linearized_residual += 2.0 / *options.rossby_number *
                               cross_product_3d(*options.angular_velocity, velocity_trial_function_value);
    }

    Tensor<1, dim> test_function;
    if (stabilization & apply_supg)
      test_function += velocity_test_function_gradient *
                       present_velocity_value;

    if (stabilization & apply_pspg)
      test_function += pressure_test_function_gradient;

    matrix += delta * (linearized_residual * test_function);

    if (stabilization & apply_supg)
      matrix += delta * present_strong_residual *
                (velocity_test_function_gradient * velocity_trial_function_value);
  }

  if (stabilization & apply_grad_div)
    matrix += mu * trace(velocity_trial_function_gradient) *
                   trace(velocity_test_function_gradient);

  return (matrix);

}



// explicit instantiations
template
double compute_matrix
(const Tensor<1, 2> &,
 const Tensor<2, 2> &,
 const Tensor<1, 2> &,
 const Tensor<2, 2> &,
 const Tensor<1, 2> &,
 const Tensor<2, 2> &,
 const double        ,
 const double        ,
 const double        ,
 const Hydrodynamic::ScalarOptions<2> &,
 const bool           );
template
double compute_matrix
(const Tensor<1, 3> &,
 const Tensor<2, 3> &,
 const Tensor<1, 3> &,
 const Tensor<2, 3> &,
 const Tensor<1, 3> &,
 const Tensor<2, 3> &,
 const double        ,
 const double        ,
 const double        ,
 const Hydrodynamic::ScalarOptions<3> &,
 const bool           );

template
double
compute_rhs
(const StabilizationFlags &,
 const Tensor<1, 2>       &,
 const Tensor<2, 2>       &,
 const Tensor<1, 2>       &,
 const Tensor<2, 2>       &,
 const Tensor<1, 2>       &,
 const double              ,
 const double              ,
 const Tensor<1, 2>       &,
 const double              ,
 const double              ,
 const double              ,
 const Hydrodynamic::ScalarOptions<2> &);
template
double
compute_rhs
(const StabilizationFlags &,
 const Tensor<1, 3>       &,
 const Tensor<2, 3>       &,
 const Tensor<1, 3>       &,
 const Tensor<2, 3>       &,
 const Tensor<1, 3>       &,
 const double              ,
 const double              ,
 const Tensor<1, 3>       &,
 const double              ,
 const double              ,
 const double              ,
 const Hydrodynamic::ScalarOptions<3> &);

template
void
compute_strong_residual
(const std::vector<Tensor<1, 2>>  &,
 const std::vector<Tensor<2, 2>>  &,
 const Hydrodynamic::VectorOptions<2> &,
 const double                      ,
 std::vector<Tensor<1,2>>         &);
template
void compute_strong_residual
(const std::vector<Tensor<1, 3>>   &,
 const std::vector<Tensor<2, 3>>   &,
 const Hydrodynamic::VectorOptions<3>  &,
 const double                       ,
 std::vector<Tensor<1, 3>>          &);

template
double
compute_residual_linearization_matrix
(const StabilizationFlags &,
 const Tensor<1, 2>       &,
 const Tensor<2, 2>       &,
 const Tensor<1, 2>       &,
 const Tensor<1, 2>       &,
 const Tensor<1, 2>       &,
 const Tensor<2, 2>       &,
 const Tensor<1, 2>       &,
 const Tensor<2, 2>       &,
 const Tensor<1, 2>       &,
 const double              ,
 const double              ,
 const double              ,
 const Hydrodynamic::ScalarOptions<2> &,
 const bool                 );
template
double
compute_residual_linearization_matrix
(const StabilizationFlags &,
 const Tensor<1, 3>       &,
 const Tensor<2, 3>       &,
 const Tensor<1, 3>       &,
 const Tensor<1, 3>       &,
 const Tensor<1, 3>       &,
 const Tensor<2, 3>       &,
 const Tensor<1, 3>       &,
 const Tensor<2, 3>       &,
 const Tensor<1, 3>       &,
 const double              ,
 const double              ,
 const double              ,
 const Hydrodynamic::ScalarOptions<3> &,
 const bool                 );



}  // namespace Hydrodynamic

