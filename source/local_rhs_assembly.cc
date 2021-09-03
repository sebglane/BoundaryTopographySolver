/*
 * local_rhs_assembly.cc
 *
 *  Created on: Dec 28, 2018
 *      Author: sg
 */

#include "equation_data.h"
#include "solver.h"

namespace TopographyProblem {

template<>
void TopographySolver<3>::local_assemble_rhs(
        const typename DoFHandler<3>::active_cell_iterator &cell,
        Assembly::RightHandSideScratch<3>                  &scratch,
        Assembly::CopyDataRightHandSide<3>                 &data,
        const bool                                          initial_step)
{
    const unsigned int dim = 3;

    const ConstraintMatrix &constraints_used = (initial_step ?
                                                nonzero_constraints
                                                : zero_constraints);

    const unsigned int dofs_per_cell = scratch.fe_values.get_fe().dofs_per_cell;
    const unsigned int n_q_points = scratch.fe_values.n_quadrature_points;
    const unsigned int n_face_q_points = scratch.fe_face_values.n_quadrature_points;

    const FEValuesExtractors::Scalar    density(0);
    const FEValuesExtractors::Vector    velocity(1);
    const FEValuesExtractors::Scalar    pressure(dim+1);
    const FEValuesExtractors::Vector    field(dim+2);
    const FEValuesExtractors::Scalar    scalar(2*dim+2);

    scratch.fe_values.reinit(cell);

    cell->get_dof_indices(data.local_dof_indices);

    // reset local object
    data.matrix_for_bc = 0;
    data.local_rhs = 0;

    // compute present values for nonlinearity
    // density equation
    scratch.fe_values[density].get_function_values(
            evaluation_point,
            scratch.present_density_values);
    scratch.fe_values[density].get_function_gradients(
            evaluation_point,
            scratch.present_density_gradients);
    // momentum equation
    scratch.fe_values[velocity].get_function_divergences(
            evaluation_point,
            scratch.present_velocity_divergences);
    scratch.fe_values[velocity].get_function_values(
            evaluation_point,
            scratch.present_velocity_values);
    scratch.fe_values[velocity].get_function_gradients(
            evaluation_point,
            scratch.present_velocity_gradients);
    scratch.fe_values[pressure].get_function_values(
            evaluation_point,
            scratch.present_pressure_values);
    // induction equation
    scratch.fe_values[field].get_function_divergences(
                evaluation_point,
                scratch.present_field_divergences);
    scratch.fe_values[field].get_function_curls(
            evaluation_point,
            scratch.present_field_curls);
    scratch.fe_values[scalar].get_function_values(
            evaluation_point,
            scratch.present_scalar_values);

    // viscosity density equation
    const double nu_density = compute_density_viscosity(
            scratch.present_velocity_values,
            cell->diameter());

    // entropy velocity equation
    const double nu_velocity = compute_velocity_viscosity(
            scratch.present_velocity_values,
            cell->diameter());

    for (unsigned int q=0; q<n_q_points; ++q)
    {
        for (unsigned int k=0; k<dofs_per_cell; ++k)
        {
            // density part
            scratch.phi_density[k]      =   scratch.fe_values[density].value(k, q);
            scratch.grad_phi_density[k] =   scratch.fe_values[density].gradient(k, q);
            // momentum part
            scratch.div_phi_velocity[k] =   scratch.fe_values[velocity].divergence(k, q);
            scratch.phi_velocity[k]     =   scratch.fe_values[velocity].value(k, q);
            scratch.grad_phi_velocity[k]=   scratch.fe_values[velocity].gradient(k, q);
            scratch.phi_pressure[k]     =   scratch.fe_values[pressure].value(k, q);
            // magnetic part
            scratch.div_phi_field[k]    =   scratch.fe_values[field].divergence(k, q);
            scratch.phi_field[k]        =   scratch.fe_values[field].value(k, q);
            scratch.curl_phi_field[k]   =   scratch.fe_values[field].curl(k, q);
            scratch.phi_scalar[k]       =   scratch.fe_values[scalar].value(k, q);
        }

        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
            data.local_rhs(i) += (
                    // continuity equation
                    -  nu_density * scratch.present_density_gradients[q] * scratch.grad_phi_density[i]
                    + scratch.present_density_values[q] * background_velocity_value * scratch.grad_phi_density[i]
                    + scratch.present_density_values[q] * scratch.present_velocity_values[q] * scratch.grad_phi_density[i]
                    - equation_coefficients[0] * scratch.present_velocity_values[q] * background_density_gradient * scratch.phi_density[i]
                    // incompressibility equation
                    + scratch.present_velocity_divergences[q] * scratch.phi_pressure[i]
                    // momentum equation
                    - nu_velocity * scalar_product(scratch.present_velocity_gradients[q], scratch.grad_phi_velocity[i])
                    - (scratch.present_velocity_gradients[q] * background_velocity_value) * scratch.phi_velocity[i]
                    - (background_velocity_gradient * scratch.present_velocity_values[q]) * scratch.phi_velocity[i]
                    - (scratch.present_velocity_gradients[q] * scratch.present_velocity_values[q]) * scratch.phi_velocity[i]
                    - equation_coefficients[1] * 2. * cross_product_3d(rotation_vector, scratch.present_velocity_values[q]) * scratch.phi_velocity[i]
                    + scratch.present_pressure_values[q] * scratch.div_phi_velocity[i]
                    + equation_coefficients[2] * scratch.present_density_values[q] * gravity_vector * scratch.phi_velocity[i]
                    - equation_coefficients[3] * (
                             (scratch.grad_phi_velocity[i] * background_field_value) * scratch.present_field_curls[q]
                           + (scratch.grad_phi_velocity[i] * scratch.present_field_curls[q]) * background_field_value
                           + (scratch.grad_phi_velocity[i] * scratch.present_field_curls[q]) * scratch.present_field_curls[q])
                   // induction equation
                   + scratch.present_field_curls[q] * scratch.curl_phi_field[i]
                   - equation_coefficients[4] * (
                             cross_product_3d(background_velocity_value, scratch.present_field_curls[q]) * scratch.phi_field[i]
                           + cross_product_3d(scratch.present_velocity_values[q], background_field_curl) * scratch.phi_field[i]
                           + cross_product_3d(scratch.present_velocity_values[q], scratch.present_field_curls[q]) * scratch.phi_field[i])
                   + scratch.present_scalar_values[q] * scratch.div_phi_field[i]
                   // solenoidal constraint
                   + scratch.present_field_divergences[q] * scratch.phi_scalar[i]
                    ) * scratch.fe_values.JxW(q);

                if (constraints_used.is_inhomogeneously_constrained(data.local_dof_indices[i]))
                    for (unsigned int j=0; j<dofs_per_cell; ++j)
                        data.matrix_for_bc(j, i) += (
                                // continuity equation
                                  nu_density * scratch.grad_phi_density[i] * scratch.grad_phi_density[j]
                                - scratch.phi_density[i] * background_velocity_value * scratch.grad_phi_density[j]
                                - scratch.phi_density[i] * scratch.present_velocity_values[q] * scratch.grad_phi_density[j]
                                - scratch.present_density_values[q] * scratch.phi_velocity[i] * scratch.grad_phi_density[j]
                                + equation_coefficients[0] * scratch.phi_velocity[i] * background_density_gradient * scratch.phi_density[j]
                                // incompressibility equation
                                - scratch.div_phi_velocity[i] * scratch.phi_pressure[j]
                                // momentum equation
                                + nu_velocity * scalar_product(scratch.grad_phi_velocity[i], scratch.grad_phi_velocity[j])
                                + (scratch.grad_phi_velocity[i] * background_velocity_value) * scratch.phi_velocity[j]
                                + (background_velocity_gradient * scratch.phi_velocity[i]) * scratch.phi_velocity[j]
                                + (scratch.grad_phi_velocity[i] * scratch.present_velocity_values[q]) * scratch.phi_velocity[j]
                                + (scratch.present_velocity_gradients[q] * scratch.phi_velocity[i]) * scratch.phi_velocity[j]
                                + equation_coefficients[1] * 2. * cross_product_3d(rotation_vector, scratch.phi_velocity[i]) * scratch.phi_velocity[j]
                                - scratch.phi_pressure[i] * scratch.div_phi_velocity[j]
                                - equation_coefficients[2] * scratch.phi_density[i] * gravity_vector * scratch.phi_velocity[j]
                                + equation_coefficients[3] * (
                                          (scratch.grad_phi_velocity[j] * background_field_value) * scratch.curl_phi_field[i]
                                        + (scratch.grad_phi_velocity[j] * scratch.curl_phi_field[i]) * background_field_value
                                        + (scratch.grad_phi_velocity[j] * scratch.present_field_curls[q]) * scratch.curl_phi_field[i]
                                        + (scratch.grad_phi_velocity[j] * scratch.curl_phi_field[i]) * scratch.present_field_curls[q])
                                // induction equation
                                - scratch.curl_phi_field[i] * scratch.curl_phi_field[j]
                                + equation_coefficients[4] * (
                                          cross_product_3d(background_velocity_value, scratch.curl_phi_field[i]) * scratch.phi_field[j]
                                        + cross_product_3d(scratch.phi_velocity[i], background_field_curl) * scratch.phi_field[j]
                                        + cross_product_3d(scratch.present_velocity_values[q], scratch.curl_phi_field[i]) * scratch.phi_field[j]
                                        + cross_product_3d(scratch.phi_velocity[i], scratch.present_field_curls[q]) * scratch.phi_field[j]
                                        )
                                + scratch.phi_scalar[i] * scratch.div_phi_field[j]
                                // solenoidal constraint
                                + scratch.div_phi_field[i] * scratch.phi_scalar[j]
                               ) * scratch.fe_values.JxW(q);
        }
    }
    if (cell->at_boundary())
        for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
            if (cell->face(face_number)->at_boundary() &&
                cell->face(face_number)->boundary_id() == DomainIdentifiers::TopoBndry)
            {
                scratch.fe_face_values.reinit(cell, face_number);

                const std::vector<Tensor<1,dim>> normal_vectors = scratch.fe_face_values.get_normal_vectors();

                scratch.fe_face_values[density].get_function_values(
                        evaluation_point,
                        scratch.present_face_density_values);
                scratch.fe_face_values[velocity].get_function_values(
                        evaluation_point,
                        scratch.present_face_velocity_values);
                scratch.fe_face_values[field].get_function_curls(
                        evaluation_point,
                        scratch.present_face_field_curls);
                scratch.fe_face_values[scalar].get_function_values(
                        evaluation_point,
                        scratch.present_face_scalar_values);

                for (unsigned int q=0; q<n_face_q_points; ++q)
                {
                    for (unsigned int k=0; k<dofs_per_cell; ++k)
                    {
                        // density part
                        scratch.phi_density[k]      =   scratch.fe_face_values[density].value(k, q);
                        // momentum part
                        scratch.phi_velocity[k]     =   scratch.fe_face_values[velocity].value(k, q);
                        // magnetic part
                        scratch.curl_phi_field[k]   =   scratch.fe_face_values[field].curl(k, q);
                    }
                    for (unsigned int i=0; i<dofs_per_cell; ++i)
                    {
                        data.local_rhs(i) += (
                                // continuity equation
                                - scratch.present_face_density_values[q] * normal_vectors[q] * background_velocity_value * scratch.phi_density[i]
                                - scratch.present_face_density_values[q] * normal_vectors[q] * scratch.present_face_velocity_values[q] * scratch.phi_density[i]
                                // momentum equation
                                + equation_coefficients[3] * (
                                          (normal_vectors[q] * background_field_value) * (scratch.present_face_field_curls[q] * scratch.phi_velocity[i])
                                        + (normal_vectors[q] * scratch.present_face_field_curls[q]) * (background_field_value * scratch.phi_velocity[i])
                                        + (normal_vectors[q] * scratch.present_face_field_curls[q]) * (scratch.present_face_field_curls[q] * scratch.phi_velocity[i]))
                                // induction equation
                                + scratch.present_face_scalar_values[q] * normal_vectors[q] * scratch.curl_phi_field[i]
                                ) * scratch.fe_face_values.JxW(q);

                        if (constraints_used.is_inhomogeneously_constrained(data.local_dof_indices[i]))
                            for (unsigned int j=0; j<dofs_per_cell; ++j)
                                data.matrix_for_bc(j, i) += (
                                        // continuity equation
                                          scratch.phi_density[i] * normal_vectors[q] * background_velocity_value * scratch.phi_density[j]
                                        + scratch.phi_density[i] * normal_vectors[q] * scratch.present_face_velocity_values[q] * scratch.phi_density[j]
                                        + scratch.present_face_density_values[q] * normal_vectors[q] * scratch.phi_velocity[i] * scratch.phi_density[j]
                                        // momentum equation
                                        - equation_coefficients[3] * (
                                                  (normal_vectors[q] * background_field_value) * (scratch.curl_phi_field[i] * scratch.phi_velocity[j])
                                                + (normal_vectors[q] * scratch.curl_phi_field[i]) * (background_field_value * scratch.phi_velocity[j])
                                                + (normal_vectors[q] * scratch.present_face_field_curls[q]) * (scratch.curl_phi_field[i] * scratch.phi_velocity[j])
                                                + (normal_vectors[q] * scratch.curl_phi_field[i]) * (scratch.present_face_field_curls[q] * scratch.phi_velocity[j]))
                                        // induction equation
                                        - scratch.fe_face_values[scalar].value(i, q) * normal_vectors[q] * scratch.curl_phi_field[j]
                                        ) * scratch.fe_face_values.JxW(q);
                    }
                }
            }
}

template<int dim>
void TopographySolver<dim>::copy_local_to_global_rhs(
        const Assembly::CopyDataRightHandSide<dim> &data,
        const bool                                  initial_step)
{
    const ConstraintMatrix &constraints_used = (initial_step ?
                                                nonzero_constraints
                                                : zero_constraints);

    constraints_used.distribute_local_to_global(data.local_rhs,
                                                data.local_dof_indices,
                                                system_rhs,
                                                data.matrix_for_bc);
}
}  // namespace TopographyProblem

// explicit instantiation
template void TopographyProblem::TopographySolver<3>::local_assemble_rhs(
        const typename dealii::DoFHandler<3>::active_cell_iterator  &,
        Assembly::RightHandSideScratch<3>                           &,
        Assembly::CopyDataRightHandSide<3>                          &,
        const bool                                                    );

template void TopographyProblem::TopographySolver<3>::copy_local_to_global_rhs(
        const Assembly::CopyDataRightHandSide<3> &, const bool );

