/*
 * assembly_functions.h
 *
 *  Created on: Sep 2, 2021
 *      Author: sg
 */

#ifndef INCLUDE_ASSEMBLY_FUNCTIONS_H_
#define INCLUDE_ASSEMBLY_FUNCTIONS_H_

#include <deal.II/base/tensor.h>

#include <advection_assembly_data.h>
#include <buoyant_hydrodynamic_assembly_data.h>
#include <hydrodynamic_assembly_data.h>
#include <angular_velocity.h>
#include <buoyant_hydrodynamic_options.h>
#include <hydrodynamic_options.h>

#include <optional>

namespace Hydrodynamic {

using namespace dealii;

template <int dim>
double compute_matrix
(const StabilizationFlags  &stabilization,
 const AssemblyData::Matrix::ScratchData<dim> &scratch,
 const unsigned int test_function_index,
 const unsigned int trial_function_index,
 const unsigned int quadrature_point_index,
 const double       nu,
 const double       delta,
 const double       mu,
 const bool         apply_newton_linearization = true);



template <int dim>
double compute_rhs
(const StabilizationFlags  &stabilization,
 const AssemblyData::Matrix::ScratchData<dim> &scratch,
 const double       present_pressure_value,
 const unsigned int test_function_index,
 const unsigned int quadrature_point_index,
 const double       nu,
 const double       mu,
 const double       delta);



template <int dim>
double compute_rhs
(const StabilizationFlags  &stabilization,
 const AssemblyData::RightHandSide::ScratchData<dim> &scratch,
 const double       present_pressure_value,
 const unsigned int test_function_index,
 const unsigned int quadrature_point_index,
 const double       nu,
 const double       mu,
 const double       delta);



template <int dim>
void compute_strong_residual
(AssemblyData::Matrix::ScratchData<dim> &scratch,
 const double nu);
template <int dim>
void compute_strong_residual
(AssemblyData::RightHandSide::ScratchData<dim> &scratch,
 const double nu);



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
 const bool         apply_newton_linearization = true);

}  // namespace Hydrodynamic




namespace BuoyantHydrodynamic {

using namespace dealii;

template <int dim>
double compute_matrix
(const StabilizationFlags  &stabilization,
 const AssemblyData::Matrix::ScratchData<dim> &scratch,
 const Tensor<1, dim>  &present_density_gradient,
 const unsigned int test_function_index,
 const unsigned int trial_function_index,
 const unsigned int quadrature_point_index,
 const double       nu,
 const double       delta,
 const double       mu,
 const double       delta_density,
 const double       nu_density,
 const bool         apply_newton_linearization = true);



template <int dim>
double compute_rhs
(const StabilizationFlags  &stabilization,
 const AssemblyData::Matrix::ScratchData<dim> &scratch,
 const Tensor<1,dim> &present_density_gradient,
 const double       present_density_value,
 const double       present_pressure_value,
 const unsigned int test_function_index,
 const unsigned int quadrature_point_index,
 const double       nu,
 const double       mu,
 const double       delta,
 const double       delta_density);
template <int dim>
double compute_rhs
(const StabilizationFlags  &stabilization,
 const AssemblyData::RightHandSide::ScratchData<dim> &scratch,
 const Tensor<1,dim> &present_density_gradient,
 const double       present_density_value,
 const double       present_pressure_value,
 const unsigned int test_function_index,
 const unsigned int quadrature_point_index,
 const double       nu,
 const double       mu,
 const double       delta,
 const double       delta_density);



template <int dim>
void compute_strong_residuals
(AssemblyData::Matrix::ScratchData<dim> &scratch,
 const std::vector<Tensor<1, dim>> &present_density_gradients,
 const std::vector<double>         &present_density_values,
 const double nu);
template <int dim>
void compute_strong_residuals
(AssemblyData::RightHandSide::ScratchData<dim> &scratch,
 const std::vector<Tensor<1, dim>> &present_density_gradients,
 const std::vector<double>         &present_density_values,
 const double nu);


}  // namespace BuoyantHydrodynamic



namespace Advection {

using namespace dealii;

/*!
 * @brief Computes the matrix entry of the advection equation.
 *
 * @attention The advection field must include contributions due to a possible
 * background field.
 *
 */
template <int dim>
double compute_matrix
(const AssemblyData::ScratchData<dim> &scratch,
 const unsigned int test_function_index,
 const unsigned int trial_function_index,
 const unsigned int quadrature_point_index,
 const double       delta);


/*!
 * @brief Computes the right-hand side entry of the advection equation.
 *
 * @attention The advection field must include contributions due to a possible
 * background field.
 *
 */
template <int dim>
double compute_rhs
(const AssemblyData::ScratchData<dim> &scratch,
 const Tensor<1, dim>  &present_gradient,
 const unsigned int     test_function_index,
 const unsigned int     quadrature_point_index,
 const double           delta);




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
 AssemblyData::ScratchData<dim> &scratch);



/*!
 * @brief Computes the linearization of the strong residual of the advection
 * equation.
 *
 */
template <int dim>
double compute_residual_linearization_matrix
(const AssemblyData::ScratchData<dim> &scratch,
 const unsigned int test_function_index,
 const unsigned int trial_function_index,
 const unsigned int quadrature_point_index,
 const double       delta);

}  // namespace Advection



#endif /* INCLUDE_ASSEMBLY_FUNCTIONS_H_ */
