/*
 * evaluation_residuals.h
 *
 *  Created on: Sep 27, 2021
 *      Author: sg
 */

#ifndef INCLUDE_EVALUATION_STABILIZATION_H_
#define INCLUDE_EVALUATION_STABILIZATION_H_

#include <deal.II/base/table_handler.h>
#include <deal.II/base/tensor_function.h>

#include <angular_velocity.h>
#include <evaluation_base.h>
#include <stabilization_flags.h>

namespace Hydrodynamic {

using namespace dealii;

template <int dim>
class EvaluationStabilization : public SolverBase::EvaluationBase<dim>
{
public:
  EvaluationStabilization(const StabilizationFlags  &stabilization,
                          const unsigned int velocity_start_index,
                          const unsigned int pressure_index,
                          const double reynolds_number,
                          const double froude_number = 0.0,
                          const double rossby_number = 0.0);
  ~EvaluationStabilization();

  void set_angular_velocity(const Utility::AngularVelocity<dim> &angular_velocity);

  void set_body_force(const TensorFunction<1, dim> &body_force);

  void set_background_velocity(const TensorFunction<1, dim> &background_velocity);

  void set_stabilization_parameters(const double c, const double mu);

  virtual void operator()(const Mapping<dim>        &mapping,
                          const FiniteElement<dim>  &fe,
                          const DoFHandler<dim>     &dof_handler,
                          const Vector<double>      &solution);

  virtual void operator()(const Mapping<dim>        &mapping,
                          const FiniteElement<dim>  &fe,
                          const DoFHandler<dim>     &dof_handler,
                          const BlockVector<double> &solution) override;

protected:
  TableHandler  data_table;

  const Utility::AngularVelocity<dim> *angular_velocity_ptr;

  const TensorFunction<1, dim>        *body_force_ptr;

  const TensorFunction<1, dim>        *background_velocity_ptr;

  const StabilizationFlags  stabilization;

  const unsigned int velocity_start_index;

  const unsigned int pressure_index;

  const double       reynolds_number;

  const double       froude_number;

  const double       rossby_number;

  double c;

  double mu;
};

// inline functions
template <int dim>
inline void EvaluationStabilization<dim>::set_angular_velocity
(const Utility::AngularVelocity<dim> &angular_velocity)
{
  angular_velocity_ptr = &angular_velocity;
}



template <int dim>
inline void EvaluationStabilization<dim>::set_body_force
(const TensorFunction<1, dim> &body_force)
{
  body_force_ptr = &body_force;
}



template <int dim>
inline void EvaluationStabilization<dim>::set_background_velocity
(const TensorFunction<1, dim> &velocity)
{
  background_velocity_ptr = &velocity;
}



template <int dim>
inline void EvaluationStabilization<dim>::set_stabilization_parameters
(const double c_in, const double mu_in)
{
  AssertThrow(c_in > 0.0, ExcLowerRangeType<double>(0.0, c_in));
  AssertIsFinite(c_in);
  c = c_in;

  AssertThrow(mu_in > 0.0, ExcLowerRangeType<double>(0.0, mu_in));
  AssertIsFinite(mu_in);
  mu = mu_in;
}

}  // namespace Hydrodynamic

namespace BuoyantHydrodynamic {

using namespace dealii;

template <int dim>
class EvaluationStabilization : public Hydrodynamic::EvaluationStabilization<dim>
{
public:
  EvaluationStabilization(const StabilizationFlags  &stabilization,
                          const unsigned int velocity_start_index,
                          const unsigned int pressure_index,
                          const unsigned int density_index,
                          const double reynolds_number,
                          const double stratification_number,
                          const double froude_number = 0.0,
                          const double rossby_number = 0.0);

  void set_gravity_field(const TensorFunction<1, dim> &gravity_field);

  void set_reference_density(const Function<dim> &reference_density);

  void set_stabilization_parameters(const double c, const double mu, const double c_density);

  virtual void operator()(const Mapping<dim>        &mapping,
                          const FiniteElement<dim>  &fe,
                          const DoFHandler<dim>     &dof_handler,
                          const Vector<double>      &solution) override;

  virtual void operator()(const Mapping<dim>        &mapping,
                          const FiniteElement<dim>  &fe,
                          const DoFHandler<dim>     &dof_handler,
                          const BlockVector<double> &solution) override;

private:
  const Function<dim>           *reference_density_ptr;

  const TensorFunction<1, dim>  *gravity_field_ptr;

  const unsigned int  density_index;

  const double        stratification_number;

  double c_density;

};

// inline functions
template <int dim>
inline void EvaluationStabilization<dim>::set_stabilization_parameters
(const double c_in, const double mu_in, const double c_density_in)
{
  Hydrodynamic::EvaluationStabilization<dim>::set_stabilization_parameters(c_in, mu_in);

  AssertThrow(c_density_in > 0.0, ExcLowerRangeType<double>(0.0, c_density_in));
  AssertIsFinite(c_density_in);
  c_density = c_density_in;
}



template <int dim>
inline void EvaluationStabilization<dim>::set_reference_density(const Function<dim> &reference_density)
{
  reference_density_ptr = &reference_density;
  return;
}



template <int dim>
inline void EvaluationStabilization<dim>::set_gravity_field(const TensorFunction<1, dim> &gravity_field)
{
  gravity_field_ptr = &gravity_field;
  return;
}

}  // namespace Hydrodynamic





#endif /* INCLUDE_EVALUATION_STABILIZATION_H_ */