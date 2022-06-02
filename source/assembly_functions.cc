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
(AssemblyData::Matrix::ScratchData<dim> &scratch,
 const double nu)
{
  const auto &present_velocity_values{scratch.present_velocity_values};
  const auto &present_velocity_gradients{scratch.present_velocity_gradients};
  auto &strong_residuals{scratch.present_strong_residuals};

  const unsigned int n_q_points{(unsigned int)present_velocity_values.size()};

  AssertDimension(present_velocity_gradients.size(), n_q_points);
  AssertDimension(strong_residuals.size(), n_q_points);

  Assert(scratch.vector_options.present_pressure_gradients,
         ExcMessage("Present pressure gradients were not assigned in options."));
  Assert(scratch.vector_options.present_velocity_laplaceans,
         ExcMessage("Present velocity laplaceans were not assigned in options."));
  AssertDimension(scratch.vector_options.present_pressure_gradients->size(), n_q_points);
  AssertDimension(scratch.vector_options.present_velocity_laplaceans->size(), n_q_points);

  const auto &present_pressure_gradients{*scratch.vector_options.present_pressure_gradients};
  const auto &present_velocity_laplaceans{*scratch.vector_options.present_velocity_laplaceans};

  if (scratch.vector_options.use_stress_form)
  {
    Assert(scratch.vector_options.present_velocity_grad_divergences,
           ExcMessage("Gradient of present velocity divergences were not assigned in options."));
    AssertDimension(scratch.vector_options.present_velocity_grad_divergences->size(), n_q_points);

    for (unsigned int q=0; q<n_q_points; ++q)
      strong_residuals[q] = (present_velocity_gradients[q] * present_velocity_values[q]) -
                            nu * present_velocity_laplaceans[q] -
                            nu * scratch.vector_options.present_velocity_grad_divergences->at(q) +
                            present_pressure_gradients[q];
  }
  else
    for (unsigned int q=0; q<n_q_points; ++q)
      strong_residuals[q] = (present_velocity_gradients[q] * present_velocity_values[q]) -
                            nu * present_velocity_laplaceans[q] +
                            present_pressure_gradients[q];

  if (scratch.vector_options.body_force_values)
  {
    Assert(scratch.vector_options.froude_number,
           ExcMessage("Froude number was not assigned in options."));

    Assert(scratch.vector_options.body_force_values,
           ExcMessage("Body force values were not assigned in options."));
    AssertDimension(scratch.vector_options.body_force_values->size(), n_q_points);

    for (unsigned int q=0; q<n_q_points; ++q)
      strong_residuals[q] += scratch.vector_options.body_force_values->at(q) /
                             (*scratch.vector_options.froude_number * *scratch.vector_options.froude_number);
  }

  if (scratch.vector_options.angular_velocity)
  {
    Assert(scratch.vector_options.rossby_number,
           ExcMessage("Rossby number was not assigned in options."));
    Assert(scratch.vector_options.angular_velocity,
           ExcMessage("Angular velocity was not assigned in options."));

    if constexpr(dim == 2)
        for (unsigned int q=0; q<n_q_points; ++q)
          strong_residuals[q] += 2.0 / *scratch.vector_options.rossby_number * scratch.vector_options.angular_velocity.value()[0] *
                                 cross_product_2d(-present_velocity_values[q]);
    else if constexpr(dim == 3)
        for (unsigned int q=0; q<n_q_points; ++q)
          strong_residuals[q] += 2.0 / *scratch.vector_options.rossby_number *
                                 cross_product_3d(*scratch.vector_options.angular_velocity, present_velocity_values[q]);
  }
}



template <int dim>
void compute_strong_residual
(AssemblyData::RightHandSide::ScratchData<dim> &scratch,
 const double nu)
{
  const auto &present_velocity_values{scratch.present_velocity_values};
  const auto &present_velocity_gradients{scratch.present_velocity_gradients};
  auto &strong_residuals{scratch.present_strong_residuals};

  const unsigned int n_q_points{(unsigned int)present_velocity_values.size()};

  AssertDimension(present_velocity_gradients.size(), n_q_points);
  AssertDimension(strong_residuals.size(), n_q_points);

  Assert(scratch.vector_options.present_pressure_gradients,
         ExcMessage("Present pressure gradients were not assigned in options."));
  Assert(scratch.vector_options.present_velocity_laplaceans,
         ExcMessage("Present velocity laplaceans were not assigned in options."));
  AssertDimension(scratch.vector_options.present_pressure_gradients->size(), n_q_points);
  AssertDimension(scratch.vector_options.present_velocity_laplaceans->size(), n_q_points);

  const auto &present_pressure_gradients{*scratch.vector_options.present_pressure_gradients};
  const auto &present_velocity_laplaceans{*scratch.vector_options.present_velocity_laplaceans};

  if (scratch.vector_options.use_stress_form)
  {
    Assert(scratch.vector_options.present_velocity_grad_divergences,
           ExcMessage("Gradient of present velocity divergences were not assigned in options."));
    AssertDimension(scratch.vector_options.present_velocity_grad_divergences->size(), n_q_points);

    for (unsigned int q=0; q<n_q_points; ++q)
      strong_residuals[q] = (present_velocity_gradients[q] * present_velocity_values[q]) -
                             nu * present_velocity_laplaceans[q] -
                             nu * scratch.vector_options.present_velocity_grad_divergences->at(q) +
                             present_pressure_gradients[q];
  }
  else
    for (unsigned int q=0; q<n_q_points; ++q)
      scratch.present_strong_residuals[q] = (present_velocity_gradients[q] * present_velocity_values[q]) -
                                            nu * present_velocity_laplaceans[q] +
                                            present_pressure_gradients[q];

  if (scratch.vector_options.body_force_values)
  {
    Assert(scratch.vector_options.froude_number,
           ExcMessage("Froude number was not assigned in options."));

    Assert(scratch.vector_options.body_force_values,
           ExcMessage("Body force values were not assigned in options."));
    AssertDimension(scratch.vector_options.body_force_values->size(), n_q_points);

    for (unsigned int q=0; q<n_q_points; ++q)
      strong_residuals[q] += scratch.vector_options.body_force_values->at(q) /
                             (*scratch.vector_options.froude_number * *scratch.vector_options.froude_number);
  }

  if (scratch.vector_options.angular_velocity)
  {
    Assert(scratch.vector_options.rossby_number,
           ExcMessage("Rossby number was not assigned in options."));
    Assert(scratch.vector_options.angular_velocity,
           ExcMessage("Angular velocity was not assigned in options."));

    if constexpr(dim == 2)
        for (unsigned int q=0; q<n_q_points; ++q)
          strong_residuals[q] += 2.0 / *scratch.vector_options.rossby_number * scratch.vector_options.angular_velocity.value()[0] *
                                 cross_product_2d(-present_velocity_values[q]);
    else if constexpr(dim == 3)
        for (unsigned int q=0; q<n_q_points; ++q)
          strong_residuals[q] += 2.0 / *scratch.vector_options.rossby_number *
                                 cross_product_3d(*scratch.vector_options.angular_velocity, present_velocity_values[q]);
  }
}



template <int dim>
double compute_residual_linearization_matrix
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
  if (!(stabilization & (apply_supg|apply_pspg|apply_grad_div)))
    return (0.0);

  Assert(nu > 0.0, ExcMessage("The viscosity must be positive."));
  Assert(delta > 0.0, ExcMessage("The SUPG stabilization parameter must be positive."));
  Assert(mu > 0.0, ExcMessage("The GradDiv stabilization parameter must be positive."));

  const Tensor<2, dim> &velocity_trial_function_gradient{scratch.grad_phi_velocity[j]};
  const Tensor<1, dim> &velocity_trial_function_value{scratch.phi_velocity[j]};
  const Tensor<1, dim> &velocity_trial_function_laplacean{scratch.laplace_phi_velocity[j]};

  const Tensor<2, dim> &velocity_test_function_gradient{scratch.grad_phi_velocity[i]};

  const Tensor<1, dim> &pressure_trial_function_gradient{scratch.grad_phi_pressure[j]};
  const Tensor<1, dim> &pressure_test_function_gradient{scratch.grad_phi_pressure[i]};

  const Tensor<2, dim> &present_velocity_gradient{scratch.present_velocity_gradients[q]};
  const Tensor<1, dim> &present_velocity_value{scratch.present_velocity_values[q]};
  const Tensor<1, dim> &present_strong_residual{scratch.present_strong_residuals[q]};

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
(AssemblyData::Matrix::ScratchData<2> &,
 const double );
template
void
compute_strong_residual
(AssemblyData::Matrix::ScratchData<3> &,
 const double );
template
void
compute_strong_residual
(AssemblyData::RightHandSide::ScratchData<2> &,
 const double );
template
void
compute_strong_residual
(AssemblyData::RightHandSide::ScratchData<3> &,
 const double );

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

namespace internal {

template <int dim>
double compute_hydrodynamic_matrix
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
  const Hydrodynamic::AssemblyData::Matrix::ScratchData<dim> &hydrodynamic_scratch
  {static_cast<const Hydrodynamic::AssemblyData::Matrix::ScratchData<dim> &>(scratch)};

  double matrix = Hydrodynamic::compute_matrix(stabilization,
                                               hydrodynamic_scratch,
                                               i,
                                               j,
                                               q,
                                               nu,
                                               delta,
                                               mu,
                                               apply_newton_linearization);

  if (scratch.scalar_options.gravity_field_value)
  {
    Assert(hydrodynamic_scratch.scalar_options.froude_number, ExcInternalError());

    const Advection::AssemblyData::Matrix::ScratchData<dim> &advection_scratch
    {static_cast<const Advection::AssemblyData::Matrix::ScratchData<dim> &>(scratch)};

    matrix -= advection_scratch.phi[j] *
              (*scratch.scalar_options.gravity_field_value * scratch.phi_velocity[i]) /
              (*hydrodynamic_scratch.scalar_options.froude_number *
               *hydrodynamic_scratch.scalar_options.froude_number);

    if (!(stabilization & (apply_supg|apply_pspg)))
    {
      const Tensor<2, dim> &velocity_test_function_gradient{hydrodynamic_scratch.grad_phi_velocity[i]};

      const Tensor<1, dim> &pressure_test_function_gradient{hydrodynamic_scratch.grad_phi_pressure[i]};

      const Tensor<1, dim> &present_velocity_value{hydrodynamic_scratch.present_velocity_values[q]};

      const double density_trial_function_value{advection_scratch.phi[i]};

      Tensor<1, dim> test_function;
      if (stabilization & apply_supg)
        test_function += velocity_test_function_gradient * present_velocity_value;
      if (stabilization & apply_pspg)
        test_function += pressure_test_function_gradient;

      matrix -= delta * density_trial_function_value *
                (*scratch.scalar_options.gravity_field_value * test_function) /
                (*hydrodynamic_scratch.scalar_options.froude_number *
                 *hydrodynamic_scratch.scalar_options.froude_number);
    }
  }

  return (matrix);
}



template <int dim>
double compute_hydrodynamic_rhs
(const StabilizationFlags  &stabilization,
 const AssemblyData::Matrix::ScratchData<dim> &scratch,
 const double       present_density_value,
 const double       present_pressure_value,
 const unsigned int i,
 const unsigned int q,
 const double       nu,
 const double       mu,
 const double       delta)
{
  const Hydrodynamic::AssemblyData::Matrix::ScratchData<dim> &hydrodynamic_scratch
  {static_cast<const Hydrodynamic::AssemblyData::Matrix::ScratchData<dim> &>(scratch)};

  double rhs{Hydrodynamic::compute_rhs(stabilization,
                                       hydrodynamic_scratch,
                                       present_pressure_value,
                                       i,
                                       q,
                                       nu,
                                       mu,
                                       delta)};

  if (scratch.scalar_options.gravity_field_value)
  {
    Assert(hydrodynamic_scratch.scalar_options.froude_number, ExcInternalError());

    rhs += present_density_value *
           (*scratch.scalar_options.gravity_field_value * hydrodynamic_scratch.phi_velocity[i]) /
           (*hydrodynamic_scratch.scalar_options.froude_number *
            *hydrodynamic_scratch.scalar_options.froude_number);
  }

  return (rhs);
}



template <int dim>
double compute_hydrodynamic_rhs
(const StabilizationFlags  &stabilization,
 const AssemblyData::RightHandSide::ScratchData<dim> &scratch,
 const double       present_density_value,
 const double       present_pressure_value,
 const unsigned int i,
 const unsigned int q,
 const double       nu,
 const double       mu,
 const double       delta)
{
  const Hydrodynamic::AssemblyData::RightHandSide::ScratchData<dim> &hydrodynamic_scratch
  {static_cast<const Hydrodynamic::AssemblyData::RightHandSide::ScratchData<dim> &>(scratch)};

  double rhs{Hydrodynamic::compute_rhs(stabilization,
                                       hydrodynamic_scratch,
                                       present_pressure_value,
                                       i,
                                       q,
                                       nu,
                                       mu,
                                       delta)};

  if (scratch.scalar_options.gravity_field_value)
  {
    Assert(hydrodynamic_scratch.scalar_options.froude_number, ExcInternalError());

    rhs += present_density_value *
           (*scratch.scalar_options.gravity_field_value * hydrodynamic_scratch.phi_velocity[i]) /
           (*hydrodynamic_scratch.scalar_options.froude_number *
            *hydrodynamic_scratch.scalar_options.froude_number);
  }

  return (rhs);
}



template <int dim>
double compute_density_matrix
(const AssemblyData::Matrix::ScratchData<dim> &scratch,
 const Tensor<1, dim>  &present_density_gradient,
 const unsigned int i,
 const unsigned int j,
 const unsigned int q,
 const double       delta,
 const double       nu,
 const bool         apply_newton_linearization)
{
  const Hydrodynamic::AssemblyData::Matrix::ScratchData<dim> &hydrodynamic_scratch
  {static_cast<const Hydrodynamic::AssemblyData::Matrix::ScratchData<dim> &>(scratch)};

  const Advection::AssemblyData::Matrix::ScratchData<dim> &advection_scratch
  {static_cast<const Advection::AssemblyData::Matrix::ScratchData<dim> &>(scratch)};

  const Tensor<1, dim> &velocity_trial_function_value{hydrodynamic_scratch.phi_velocity[j]};

  const Tensor<1, dim> &present_velocity_value{hydrodynamic_scratch.present_velocity_values[q]};

  const Tensor<1, dim> &density_trial_function_gradient{advection_scratch.grad_phi[j]};
  const Tensor<1, dim> &density_test_function_gradient{advection_scratch.grad_phi[i]};

  const double density_test_function_value{advection_scratch.phi[i]};

  double matrix{Advection::compute_matrix(advection_scratch,
                                          i,
                                          j,
                                          q,
                                          delta)};

  // linearization of velocity inside advection term
  if (apply_newton_linearization)
  {
    matrix += (velocity_trial_function_value * present_density_gradient) *
              density_test_function_value;

    // linearization of velocity inside reference gradient term
    if (advection_scratch.scalar_options.reference_gradient)
    {
      Assert(advection_scratch.scalar_options.gradient_scaling, ExcInternalError());

      matrix += *advection_scratch.scalar_options.gradient_scaling *
                (velocity_trial_function_value * *advection_scratch.scalar_options.reference_gradient) *
                density_test_function_value;
    }
  }

  // linearization of residual w.r.t. velocity
  if (present_velocity_value.norm() > 0.0)
  {
    double linearized_residual{0.0};

    // linearization of advection term
    if (apply_newton_linearization)
    {
      linearized_residual += velocity_trial_function_value * present_density_gradient;

      // linearization of reference gradient term
      if (advection_scratch.scalar_options.reference_gradient)
      {
        Assert(advection_scratch.scalar_options.gradient_scaling, ExcInternalError());

        linearized_residual += *advection_scratch.scalar_options.gradient_scaling *
                               velocity_trial_function_value *
                               *advection_scratch.scalar_options.reference_gradient;
      }
    }
    matrix += delta * linearized_residual * (density_test_function_gradient * present_velocity_value);
  }
  else
    matrix += nu * density_trial_function_gradient * density_test_function_gradient;

  // linearization of stabiliziation test function
  matrix += delta * advection_scratch.present_strong_residuals[q] *
            (velocity_trial_function_value * density_test_function_gradient);

  return (matrix);
}



template <int dim>
double compute_density_rhs
(const AssemblyData::Matrix::ScratchData<dim> &scratch,
 const Tensor<1,dim> &present_density_gradient,
 const unsigned int i,
 const unsigned int q,
 const double       delta)
{
  const Advection::AssemblyData::RightHandSide::ScratchData<dim> &advection_scratch
  {static_cast<const Advection::AssemblyData::RightHandSide::ScratchData<dim> &>(scratch)};

  const double rhs{Advection::compute_rhs(advection_scratch, present_density_gradient, i, q, delta)};

  return (rhs);
}



template <int dim>
double compute_density_rhs
(const AssemblyData::RightHandSide::ScratchData<dim> &scratch,
 const Tensor<1,dim> &present_density_gradient,
 const unsigned int i,
 const unsigned int q,
 const double       delta)
{
  const Advection::AssemblyData::RightHandSide::ScratchData<dim> &advection_scratch
  {static_cast<const Advection::AssemblyData::RightHandSide::ScratchData<dim> &>(scratch)};

  const double rhs{Advection::compute_rhs(advection_scratch, present_density_gradient, i, q, delta)};

  return (rhs);
}



}  // namespace internal

template <int dim>
double compute_matrix
(const StabilizationFlags  &stabilization,
 const AssemblyData::Matrix::ScratchData<dim> &scratch,
 const Tensor<1, dim>  &present_density_gradient,
 const unsigned int i,
 const unsigned int j,
 const unsigned int q,
 const double       nu,
 const double       delta,
 const double       mu,
 const double       delta_density,
 const double       nu_density,
 const bool         apply_newton_linearization)
{
  double matrix{internal::
                compute_hydrodynamic_matrix(stabilization,
                                            scratch,
                                            i,
                                            j,
                                            q,
                                            nu,
                                            delta,
                                            mu,
                                            apply_newton_linearization)};

  matrix += internal::
            compute_density_matrix(scratch,
                                   present_density_gradient,
                                   i,
                                   j,
                                   q,
                                   delta_density,
                                   nu_density,
                                   apply_newton_linearization);

  return (matrix);

}



template <int dim>
double compute_rhs
(const StabilizationFlags  &stabilization,
 const AssemblyData::Matrix::ScratchData<dim> &scratch,
 const Tensor<1,dim> &present_density_gradient,
 const double       present_density_value,
 const double       present_pressure_value,
 const unsigned int i,
 const unsigned int q,
 const double       nu,
 const double       mu,
 const double       delta,
 const double       delta_density)
{
  double rhs{internal::
             compute_hydrodynamic_rhs(stabilization,
                                      scratch,
                                      present_density_value,
                                      present_pressure_value,
                                      i,
                                      q,
                                      nu,
                                      mu,
                                      delta)};
  rhs += internal::
         compute_density_rhs(scratch,
                             present_density_gradient,
                             i,
                             q,
                             delta_density);

  return (rhs);
}
template <int dim>
double compute_rhs
(const StabilizationFlags  &stabilization,
 const AssemblyData::RightHandSide::ScratchData<dim> &scratch,
 const Tensor<1,dim> &present_density_gradient,
 const double       present_density_value,
 const double       present_pressure_value,
 const unsigned int i,
 const unsigned int q,
 const double       nu,
 const double       mu,
 const double       delta,
 const double       delta_density)
{
  double rhs{internal::
             compute_hydrodynamic_rhs(stabilization,
                                      scratch,
                                      present_density_value,
                                      present_pressure_value,
                                      i,
                                      q,
                                      nu,
                                      mu,
                                      delta)};
  rhs += internal::
         compute_density_rhs(scratch,
                             present_density_gradient,
                             i,
                             q,
                             delta_density);

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
  LegacyHydrodynamic::
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
(const std::vector<Tensor<1, dim>>      &present_density_gradients,
 AssemblyData::RightHandSide::ScratchData<dim> &scratch)
{
  Advection::AssemblyData::ScratchData<dim> &advection_scratch
  {static_cast<Advection::AssemblyData::ScratchData<dim> &>(scratch)};

  Advection::compute_strong_residual(present_density_gradients,
                                     advection_scratch);
}



template <int dim>
void compute_strong_density_residual
(const std::vector<Tensor<1, dim>>      &present_density_gradients,
 AssemblyData::Matrix::ScratchData<dim> &scratch)
{
  Advection::AssemblyData::ScratchData<dim> &advection_scratch
  {static_cast<Advection::AssemblyData::ScratchData<dim> &>(scratch)};

  Advection::compute_strong_residual(present_density_gradients,
                                     advection_scratch);
}



// explicit instantiations
template
double compute_matrix
(const StabilizationFlags                   &,
 const AssemblyData::Matrix::ScratchData<2> &,
 const Tensor<1, 2>&,
 const unsigned int ,
 const unsigned int ,
 const unsigned int ,
 const double       ,
 const double       ,
 const double       ,
 const double       ,
 const double       ,
 const bool          );
template
double compute_matrix
(const StabilizationFlags                   &,
 const AssemblyData::Matrix::ScratchData<3> &,
 const Tensor<1, 3>&,
 const unsigned int ,
 const unsigned int ,
 const unsigned int ,
 const double       ,
 const double       ,
 const double       ,
 const double       ,
 const double       ,
 const bool          );

template
double
compute_rhs
(const StabilizationFlags &,
 const AssemblyData::Matrix::ScratchData<2> &,
 const Tensor<1, 2>  &,
 const double         ,
 const double         ,
 const unsigned int   ,
 const unsigned int   ,
 const double         ,
 const double         ,
 const double         ,
 const double          );
template
double
compute_rhs
(const StabilizationFlags &,
 const AssemblyData::Matrix::ScratchData<3> &,
 const Tensor<1, 3>  &,
 const double         ,
 const double         ,
 const unsigned int   ,
 const unsigned int   ,
 const double         ,
 const double         ,
 const double         ,
 const double          );
template
double
compute_rhs
(const StabilizationFlags &,
 const AssemblyData::RightHandSide::ScratchData<2> &,
 const Tensor<1, 2>  &,
 const double         ,
 const double         ,
 const unsigned int   ,
 const unsigned int   ,
 const double         ,
 const double         ,
 const double         ,
 const double          );
template
double
compute_rhs
(const StabilizationFlags &,
 const AssemblyData::RightHandSide::ScratchData<3> &,
 const Tensor<1, 3>  &,
 const double         ,
 const double         ,
 const unsigned int   ,
 const unsigned int   ,
 const double         ,
 const double         ,
 const double         ,
 const double          );

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
(const std::vector<Tensor<1, 2>>      &,
 AssemblyData::RightHandSide::ScratchData<2> &);
template
void
compute_strong_density_residual
(const std::vector<Tensor<1, 3>>      &,
 AssemblyData::RightHandSide::ScratchData<3> &);

template
void
compute_strong_density_residual
(const std::vector<Tensor<1, 2>>      &,
 AssemblyData::Matrix::ScratchData<2> &);
template
void
compute_strong_density_residual
(const std::vector<Tensor<1, 3>>      &,
 AssemblyData::Matrix::ScratchData<3> &);

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
(const std::vector<Tensor<1, dim>> &present_gradients,
 AssemblyData::ScratchData<dim> &scratch)
{
  const unsigned int n_q_points{(unsigned int)present_gradients.size()};

  const auto &advection_field_values{scratch.advection_field_values};
  auto &strong_residuals{scratch.present_strong_residuals};
  AssertDimension(advection_field_values.size(), n_q_points);
  AssertDimension(strong_residuals.size(), n_q_points);

  for (unsigned int q=0; q<n_q_points; ++q)
    strong_residuals[q] = advection_field_values[q] * present_gradients[q];

  if (scratch.vector_options.reference_gradients)
  {
    Assert(scratch.vector_options.gradient_scaling,
           ExcMessage("Gradient scaling number was not were not assigned in options."));
    AssertDimension(scratch.vector_options.reference_gradients->size(), n_q_points);

    for (unsigned int q=0; q<n_q_points; ++q)
      strong_residuals[q] += *scratch.vector_options.gradient_scaling *
                              advection_field_values[q] *
                              scratch.vector_options.reference_gradients->at(q);
  }

  if (scratch.vector_options.source_term_values)
  {
    AssertDimension(scratch.vector_options.source_term_values->size(), n_q_points);

    for (unsigned int q=0; q<n_q_points; ++q)
      strong_residuals[q] -= scratch.vector_options.source_term_values->at(q);
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
(const std::vector<Tensor<1, 2>> &,
    AssemblyData::ScratchData<2> &);
template
void
compute_strong_residual
(const std::vector<Tensor<1, 3>> &,
 AssemblyData::ScratchData<3>    &);

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




// explicit instantiations
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




}  // namespace Hydrodynamic

