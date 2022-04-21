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
(const Tensor<1, dim> &velocity_trial_function_value,
 const Tensor<2, dim> &velocity_trial_function_gradient,
 const Tensor<1, dim> &velocity_test_function_value,
 const Tensor<2, dim> &velocity_test_function_gradient,
 const Tensor<1, dim> &present_velocity_value,
 const Tensor<2, dim> &present_velocity_gradient,
 const double          pressure_trial_function,
 const double          pressure_test_function,
 const double          nu,
 const OptionalScalarArguments<dim> &options,
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
 const OptionalScalarArguments<dim> &options)
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
 const OptionalVectorArguments<dim>  &options,
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
 const OptionalScalarArguments<dim> &options,
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
 const OptionalScalarArguments<2> &,
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
 const OptionalScalarArguments<3> &,
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
 const OptionalScalarArguments<2> &);
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
 const OptionalScalarArguments<3> &);

template
void
compute_strong_residual
(const std::vector<Tensor<1, 2>>  &,
 const std::vector<Tensor<2, 2>>  &,
 const OptionalVectorArguments<2> &,
 const double                      ,
 std::vector<Tensor<1,2>>         &);
template
void compute_strong_residual
(const std::vector<Tensor<1, 3>>   &,
 const std::vector<Tensor<2, 3>>   &,
 const OptionalVectorArguments<3>  &,
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
 const OptionalScalarArguments<2> &,
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
 const OptionalScalarArguments<3> &,
 const bool                 );



}  // namespace Hydrodynamic



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
(const Tensor<1, dim>                &trial_function_gradient,
 const Tensor<1, dim>                &advection_field_value,
 const double                         test_function_value)
{
  return ((advection_field_value * trial_function_gradient) * test_function_value);
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
(const double           test_function_value,
 const Tensor<1, dim>  &test_function_gradient,
 const Tensor<1, dim>  &present_gradient,
 const Tensor<1, dim>  &advection_field_value,
 const double           present_strong_residual,
 const double           delta,
 const OptionalScalarArguments<dim> &options)
{
  double rhs{-(advection_field_value * present_gradient)};

  if (options.reference_gradient)
  {
    Assert(options.gradient_scaling,
           ExcMessage("Gradient scaling number was not were not assigned in options."));

    rhs -= *options.gradient_scaling *
            (advection_field_value * *options.reference_gradient);
  }

  if (options.source_term_value)
    rhs += *options.source_term_value;

  rhs *= test_function_value;

  const double stabilization_test_function{advection_field_value *
                                           test_function_gradient};

  rhs -= delta * present_strong_residual * stabilization_test_function;

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
 const OptionalVectorArguments<dim>  &options)
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
(const Tensor<1, dim>  &trial_function_gradient,
 const Tensor<1, dim>  &advection_field_value,
 const Tensor<1, dim>  &test_function_gradient,
 const double           delta)
{
  return (delta *
          (advection_field_value * trial_function_gradient) *
          (advection_field_value * test_function_gradient));
}



// explicit instantiations
template
double
compute_matrix
(const Tensor<1, 2>   &,
 const Tensor<1, 2>   &,
 const double           );
template
double
compute_matrix
(const Tensor<1, 3>   &,
 const Tensor<1, 3>   &,
 const double           );

template
double
compute_rhs
(const double         ,
 const Tensor<1, 2>  &,
 const Tensor<1, 2>  &,
 const Tensor<1, 2>  &,
 const double         ,
 const double         ,
 const OptionalScalarArguments<2> &);
template
double
compute_rhs
(const double         ,
 const Tensor<1, 3>  &,
 const Tensor<1, 3>  &,
 const Tensor<1, 3>  &,
 const double         ,
 const double         ,
 const OptionalScalarArguments<3> &);


template
void
compute_strong_residual
(const std::vector<Tensor<1, 2>>    &,
 const std::vector<Tensor<1, 2>>    &,
 std::vector<double>                &,
 const OptionalVectorArguments<2>   &);
template
void
compute_strong_residual
(const std::vector<Tensor<1, 3>>    &,
 const std::vector<Tensor<1, 3>>    &,
 std::vector<double>                &,
 const OptionalVectorArguments<3>   &);

template
double
compute_residual_linearization_matrix
(const Tensor<1,2>  &,
 const Tensor<1,2>  &,
 const Tensor<1,2>  &,
 const double         );
template
double
compute_residual_linearization_matrix
(const Tensor<1, 3> &,
 const Tensor<1, 3> &,
 const Tensor<1, 3> &,
 const double         );


}  // namespace Advection
