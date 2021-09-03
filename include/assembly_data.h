/*
 * assembly_data.h
 *
 *  Created on: Dec 24, 2018
 *      Author: sg
 */

#ifndef INCLUDE_ASSEMBLY_DATA_H_
#define INCLUDE_ASSEMBLY_DATA_H_


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

#include <deal.II/fe/fe_base.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <vector>

namespace Assembly {

using namespace dealii;

template<int dim>
struct Scratch
{
    Scratch(const FiniteElement<dim>   &finite_element,
            const Quadrature<dim>      &quadrature,
            const Quadrature<dim-1>    &face_squadrature,
            const UpdateFlags           update_flags,
            const UpdateFlags           face_update_flags);

    Scratch(const Scratch<dim>    &scratch);

    FEValues<dim>               fe_values;
    FEFaceValues<dim>           fe_face_values;

    // density part
    std::vector<double>         phi_density;
    std::vector<Tensor<1,dim>>  grad_phi_density;

    std::vector<double>         present_density_values;
    std::vector<Tensor<1,dim>>  present_density_gradients;
    std::vector<double>         present_face_density_values;

    // momentum part
    std::vector<double>         div_phi_velocity;
    std::vector<Tensor<1,dim>>  phi_velocity;
    std::vector<Tensor<2,dim>>  grad_phi_velocity;
    std::vector<double>         phi_pressure;

    std::vector<double>         present_velocity_divergences;
    std::vector<Tensor<1,dim>>  present_velocity_values;
    std::vector<Tensor<2,dim>>  present_velocity_gradients;
    std::vector<double>         present_pressure_values;

    std::vector<Tensor<1,dim>>  present_face_velocity_values;

    // magnetic part
    std::vector<double>         div_phi_field;
    std::vector<Tensor<1,dim>>  phi_field;
    std::vector<Tensor<1,dim>>  curl_phi_field;
    std::vector<Tensor<2,dim>>  grad_phi_field;
    std::vector<double>         phi_scalar;

    std::vector<double>         present_field_divergences;
    std::vector<Tensor<1,dim>>  present_field_curls;
    std::vector<double>         present_scalar_values;

    std::vector<Tensor<1,dim>>  present_face_field_curls;
    std::vector<double>         present_face_scalar_values;
};

template <int dim>
struct CopyData
{
    CopyData(const FiniteElement<dim>   &finite_element);
    CopyData(const CopyData<dim>        &data);

    FullMatrix<double>                      local_matrix;

    std::vector<types::global_dof_index>    local_dof_indices;
};

template <int dim>
struct CopyDataRightHandSide
{
    CopyDataRightHandSide(const FiniteElement<dim>          &finite_element);
    CopyDataRightHandSide(const CopyDataRightHandSide<dim>  &data);

    FullMatrix<double>                      matrix_for_bc;
    Vector<double>                          local_rhs;

    std::vector<types::global_dof_index>    local_dof_indices;
};

} // namespace Assembly


#endif /* INCLUDE_ASSEMBLY_DATA_H_ */
