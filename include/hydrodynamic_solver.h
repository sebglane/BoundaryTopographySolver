/*
 * hydrodynamic_solver.h
 *
 *  Created on: Aug 30, 2021
 *      Author: sg
 */

#ifndef INCLUDE_HYDRODYNAMIC_SOLVER_H_
#define INCLUDE_HYDRODYNAMIC_SOLVER_H_

#include <angular_velocity.h>
#include <assembly_base_data.h>
#include <boundary_conditions.h>
#include <solver_base.h>
#include <stabilization_flags.h>

namespace Hydrodynamic {

using namespace BoundaryConditions;

/*!
 * @brief Enumeration for the weak form of the convective term.
 *
 * @attention These definitions are the ones I see the most in the literature.
 * Nonetheless Volker John and Helene Dallmann define the skew-symmetric
 * and the divergence form differently.
 */
enum ConvectiveTermWeakForm
{
  /*!
   * @brief The standard form.
   * @details Given by
   * \f[
   * \int_\Omega \bs{w} \cdot [ \bs{v} \cdot ( \nabla \otimes \bs{v})] \dint{v}
   * \f]
   * where \f$ \bs{v} \f$ and \f$ \bs{w} \f$ are the velocity and the
   * test function respectively.
   */
  standard,

  /*!
   * @brief The skew-symmetric form.
   * @details Given by
   * \f[
   * \int_\Omega \bs{w} \cdot [ \bs{v} \cdot ( \nabla \otimes \bs{v}) +
   * (\nabla \cdot \bs{v}) \bs{v}] \dint{v}
   * \f]
   * where \f$ \bs{v} \f$ and \f$ \bs{w} \f$ are the velocity and the
   * test function respectively.
   */
  skewsymmetric,

  /*!
   * @brief The divergence form.
   * @details Given by
   * \f[
   * \int_\Omega \bs{w} \cdot [ \bs{v} \cdot ( \nabla \otimes \bs{v}) +
   * \frac{1}{2}(\nabla \cdot \bs{v}) \bs{v}] \dint{v}
   * \f]
   * where \f$ \bs{v} \f$ and \f$ \bs{w} \f$ are the velocity and the
   * test function respectively.
   */
  divergence,

  /*!
   * @brief The rotational form.
   * @details Given by
   * \f[
   * \int_\Omega \bs{w} \cdot [ ( \nabla \times\bs{v}) \times \bs{v}] \dint{v}
   * \f]
   * where \f$ \bs{v} \f$ and \f$ \bs{w} \f$ are the velocity and the
   * test function respectively.
   * @note This form modifies the pressure, *i. e.*,
   * \f[
   * \bar{p} = p + \frac{1}{2} \bs{v} \cdot \bs{v}.
   * \f]
   */
  rotational
};



/*!
 * @brief Enumeration for the weak form of the non-linear convective term.
 * @attention These definitions are the ones I see the most in the literature.
 * Nonetheless Volker John and Helene Dallmann define the skew-symmetric
 * and the divergence form differently.
 */
enum ViscousTermWeakForm
{
  /*!
   * @brief The Laplacean form.
   * @details Given by
   * \f[
   * \int_\Omega (\nabla\otimes\bs{w}) \cdott (\nabla\otimes\bs{v}) \dint{v}
   * \f]
   * where \f$ \bs{v} \f$ and \f$ \bs{w} \f$ are the velocity and the
   * test function respectively.
   */
  laplacean,

  /*!
   * @brief The stress form.
   * \f[
   * \int_\Omega (\nabla\otimes\bs{w} + \bs{w}\otimes\nabla)
   *              \cdott (\nabla\otimes\bs{v} + \bs{v}\otimes\nabla) \dint{v}
   * \f]
   */
  stress,
};



/*!
 * @struct SolverParameters
 *
 * @brief A structure containing all the parameters of the Navier-Stokes
 * solver.
 */
struct SolverParameters: SolverBase::Parameters
{
  /*!
   * Constructor which sets up the parameters with default values.
   */
  SolverParameters();

  /*!
   * @brief Static method which declares the associated parameter to the
   * ParameterHandler object @p prm.
   */
  static void declare_parameters(ParameterHandler &prm);

  /*!
   * @brief Method which parses the parameters from the ParameterHandler
   * object @p prm.
   */
  void parse_parameters(ParameterHandler &prm);

  /*!
   * @brief Method forwarding parameters to a stream object.
   *
   * @details This method does not add a `std::endl` to the stream at the end.
   *
   */
  template<typename Stream>
  friend Stream& operator<<(Stream &stream, const SolverParameters &prm);

  /*!
   * @brief Enumerator controlling which weak form of the convective
   * term is applied.
   */
  ConvectiveTermWeakForm  convective_term_weak_form;

  /*!
   * @brief Enumeration controlling which weak form of the viscous
   * term is applied.
   */
  ViscousTermWeakForm     viscous_term_weak_form;

  /*!
   * @brief Enumeration controlling which weak form of the viscous
   * term is applied.
   */
  StabilizationFlags      stabilization;

  /*!
   * @brief Stabilization parameter controlling the SUPG and PSPG terms.
   */
  double  c;

  /*!
   * @brief Stabilization parameter controlling the grad-div term.
   */
  double  mu;

};



/*!
 * @brief Method forwarding parameters to a stream object.
 */
template <typename Stream>
Stream& operator<<(Stream &stream, const SolverParameters &prm);




namespace AssemblyData {

namespace Matrix {

template <int dim>
struct Scratch : AssemblyBaseData::Matrix::Scratch<dim>
{
  Scratch(const Mapping<dim>        &mapping,
          const Quadrature<dim>     &quadrature_formula,
          const FiniteElement<dim>  &fe,
          const UpdateFlags         update_flags,
          const Quadrature<dim-1>  &face_quadrature_formula,
          const UpdateFlags         face_update_flags,
          const StabilizationFlags  stabilization_flags,
          const bool                allocate_body_force,
          const bool                allocate_traction);

  Scratch(const Scratch<dim>  &data);

  FEFaceValues<dim>   fe_face_values;

  const unsigned int  n_face_q_points;

  // shape functions
  std::vector<Tensor<1, dim>> phi_velocity;
  std::vector<Tensor<2, dim>> grad_phi_velocity;
  std::vector<double>         div_phi_velocity;
  std::vector<double>         phi_pressure;

  // stabilization related shape functions
  std::vector<Tensor<1, dim>> grad_phi_pressure;
  std::vector<Tensor<1, dim>> laplace_phi_velocity;

  // solution values
  std::vector<Tensor<1, dim>> present_velocity_values;
  std::vector<Tensor<2, dim>> present_velocity_gradients;
  std::vector<double>         present_pressure_values;

  // stabilization related solution values
  std::vector<Tensor<1, dim>> present_velocity_laplaceans;
  std::vector<Tensor<1, dim>> present_pressure_gradients;

  // source term values
  std::vector<Tensor<1,dim>>  body_force_values;
  typename Utility::AngularVelocity<dim>::value_type angular_velocity_value;

  // source term face values
  std::vector<Tensor<1, dim>> boundary_traction_values;
};

} // namespace Matrix

namespace RightHandSide
{

template <int dim>
struct Scratch : AssemblyBaseData::RightHandSide::Scratch<dim>
{
  Scratch(const Mapping<dim>        &mapping,
          const Quadrature<dim>     &quadrature_formula,
          const FiniteElement<dim>  &fe,
          const UpdateFlags         update_flags,
          const Quadrature<dim-1>  &face_quadrature_formula,
          const UpdateFlags         face_update_flags,
          const StabilizationFlags  stabilization_flags,
          const bool                allocate_body_force,
          const bool                allocate_traction);

  Scratch(const Scratch<dim>  &data);

  FEFaceValues<dim>   fe_face_values;

  const unsigned int  n_face_q_points;

  // shape functions
  std::vector<Tensor<1, dim>> phi_velocity;
  std::vector<Tensor<2, dim>> grad_phi_velocity;
  std::vector<double>         div_phi_velocity;
  std::vector<double>         phi_pressure;

  // stabilization related shape functions
  std::vector<Tensor<1, dim>> grad_phi_pressure;

  // solution values
  std::vector<Tensor<1, dim>> present_velocity_values;
  std::vector<Tensor<2, dim>> present_velocity_gradients;
  std::vector<double>         present_pressure_values;

  // stabilization related solution values
  std::vector<Tensor<1, dim>> present_velocity_laplaceans;
  std::vector<Tensor<1, dim>> present_pressure_gradients;

  // source term values
  std::vector<Tensor<1,dim>>  body_force_values;
  typename Utility::AngularVelocity<dim>::value_type angular_velocity_value;

  // source term face values
  std::vector<Tensor<1, dim>> boundary_traction_values;
};

} // namespace RightHandSide

} // namespace AssemblyData


template <int dim>
class Solver: public SolverBase::Solver<dim>
{
public:
  Solver(Triangulation<dim>  &tria,
         Mapping<dim>        &mapping,
         const SolverParameters &parameters,
         const double         reynolds_number = 1.0,
         const double         froude_number = 0.0,
         const double         rossby_number = 0.0);

  void set_angular_velocity(const Utility::AngularVelocity<dim> &angular_velocity);

  void set_body_force(const TensorFunction<1, dim> &body_force);

  VectorBoundaryConditions<dim>&  get_velocity_bcs();
  const VectorBoundaryConditions<dim>&  get_velocity_bcs() const;

  ScalarBoundaryConditions<dim>&  get_pressure_bcs();
  const ScalarBoundaryConditions<dim>&  get_pressure_bcs() const;

  double get_reynolds_number() const;

  double get_froude_number() const;

private:
  virtual void setup_fe_system();

  virtual void setup_dofs();

  virtual void apply_boundary_conditions();

  virtual void assemble_system(const bool initial_step);

  virtual void assemble_rhs(const bool initial_step);

  void assemble_local_system
  (const typename DoFHandler<dim>::active_cell_iterator &cell,
   AssemblyData::Matrix::Scratch<dim> &scratch,
   AssemblyBaseData::Matrix::Copy     &data) const;

  void copy_local_to_global_system
  (const AssemblyBaseData::Matrix::Copy     &data,
   const bool use_homogeneous_constraints);

  void assemble_local_rhs
  (const typename DoFHandler<dim>::active_cell_iterator &cell,
   AssemblyData::RightHandSide::Scratch<dim> &scratch,
   AssemblyBaseData::RightHandSide::Copy     &data) const;

  void copy_local_to_global_rhs
  (const AssemblyBaseData::RightHandSide::Copy     &data,
   const bool use_homogeneous_constraints);

  virtual void output_results(const unsigned int cycle = 0) const;

protected:
  VectorBoundaryConditions<dim> velocity_boundary_conditions;

  ScalarBoundaryConditions<dim> pressure_boundary_conditions;

  const Utility::AngularVelocity<dim> *angular_velocity_ptr;

  const TensorFunction<1, dim>        *body_force_ptr;

  const ConvectiveTermWeakForm  convective_term_weak_form;

  const ViscousTermWeakForm     viscous_term_weak_form;

  const StabilizationFlags      stabilization;

  const unsigned int  velocity_fe_degree;

  const double        reynolds_number;

  const double        froude_number;

  const double        rossby_number;

  const double        c;

  const double        mu;
};


// inline functions
template <int dim>
inline void Solver<dim>::set_angular_velocity
(const Utility::AngularVelocity<dim> &angular_velocity)
{
  angular_velocity_ptr = &angular_velocity;
}



template <int dim>
inline void Solver<dim>::set_body_force
(const TensorFunction<1, dim> &body_force)
{
  body_force_ptr = &body_force;
}



template <int dim>
inline VectorBoundaryConditions<dim> &
Solver<dim>::get_velocity_bcs()
{
  return velocity_boundary_conditions;
}



template <int dim>
inline const VectorBoundaryConditions<dim> &
Solver<dim>::get_velocity_bcs() const
{
  return velocity_boundary_conditions;
}


template <int dim>
inline ScalarBoundaryConditions<dim> &
Solver<dim>::get_pressure_bcs()
{
  return pressure_boundary_conditions;
}



template <int dim>
inline const ScalarBoundaryConditions<dim> &
Solver<dim>::get_pressure_bcs() const
{
  return pressure_boundary_conditions;
}



template <int dim>
inline double Solver<dim>::get_reynolds_number() const
{
  return reynolds_number;
}



template <int dim>
inline double Solver<dim>::get_froude_number() const
{
  return froude_number;
}

}  // namespace Hydrodynamic

#endif /* INCLUDE_HYDRODYNAMIC_SOLVER_H_ */
