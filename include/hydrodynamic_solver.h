/*
 * hydrodynamic_solver.h
 *
 *  Created on: Aug 30, 2021
 *      Author: sg
 */

#ifndef INCLUDE_HYDRODYNAMIC_SOLVER_H_
#define INCLUDE_HYDRODYNAMIC_SOLVER_H_

#include <angular_velocity.h>
#include <base.h>
#include <boundary_conditions.h>
#include <hydrodynamic_assembly_data.h>

#include <memory>
#include <optional>

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
struct SolverParameters: Base::Parameters
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
   * @brief Flag for including boundary traction terms in the weak form. These
   * are the terms on boundary where neither a Dirichlet nor a Neumann boundary
   * condition is specified.
   */
  bool  include_boundary_stress_terms;

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



template <int dim,
          typename TriangulationType = Triangulation<dim>>
class Solver: virtual public Base::Solver<dim, TriangulationType>
{
public:
  Solver(TriangulationType   &tria,
         Mapping<dim>        &mapping,
         const SolverParameters &parameters,
         const double         reynolds_number = 1.0,
         const double         froude_number = 0.0,
         const double         rossby_number = 0.0);

  void set_angular_velocity(const std::shared_ptr<const Utility::AngularVelocity<dim>> &angular_velocity);

  void set_body_force(const std::shared_ptr<const TensorFunction<1, dim>> &body_force);

  void set_background_velocity(const std::shared_ptr<const TensorFunction<1, dim>> &background_velocity);

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

  virtual void assemble_system(const bool use_homogenenous_constraints,
                               const bool use_newton_linearization);

  virtual void assemble_rhs(const bool use_homogenenous_constraints);

  void assemble_system_local_cell
  (const typename DoFHandler<dim>::active_cell_iterator  &cell,
   AssemblyData::Matrix::ScratchData<dim>                &scratch,
   MeshWorker::CopyData<1,1,1>                           &data,
   const bool                                             use_newton_linearization,
   const bool                                             use_stress_form) const;

  void assemble_system_local_boundary
  (const typename DoFHandler<dim>::active_cell_iterator  &cell,
   const unsigned int                                     face_number,
   AssemblyData::Matrix::ScratchData<dim>                &scratch,
   MeshWorker::CopyData<1,1,1>                           &data,
   const bool                                             use_stress_form) const;

  void assemble_rhs_local_cell
  (const typename DoFHandler<dim>::active_cell_iterator  &cell,
   AssemblyData::RightHandSide::ScratchData<dim>         &scratch,
   MeshWorker::CopyData<0,1,1>                           &data,
   const bool                                             use_stress_form) const;

  void assemble_rhs_local_boundary
  (const typename DoFHandler<dim>::active_cell_iterator  &cell,
   const unsigned int                                     face_number,
   AssemblyData::RightHandSide::ScratchData<dim>         &scratch,
   MeshWorker::CopyData<0,1,1>                           &data,
   const bool                                             use_stress_form) const;

  virtual void output_results(const unsigned int cycle = 0) const;

protected:
  VectorBoundaryConditions<dim> velocity_boundary_conditions;

  ScalarBoundaryConditions<dim> pressure_boundary_conditions;

  std::vector<types::boundary_id> boundary_stress_ids;

  std::shared_ptr<const Utility::AngularVelocity<dim>> angular_velocity_ptr;

  std::shared_ptr<const TensorFunction<1, dim>> body_force_ptr;

  std::shared_ptr<const TensorFunction<1, dim>> background_velocity_ptr;

  const ConvectiveTermWeakForm  convective_term_weak_form;

  const ViscousTermWeakForm     viscous_term_weak_form;

  const StabilizationFlags      stabilization;

  const unsigned int  velocity_fe_degree;

  const double        reynolds_number;

  const double        froude_number;

  const double        rossby_number;

  const double        c;

  const double        mu;

  const bool          include_boundary_stress_terms;

  unsigned int        velocity_fe_index;

  unsigned int        pressure_fe_index;

  unsigned int        velocity_block_index;

  unsigned int        pressure_block_index;

};


// inline functions
template <int dim, typename TriangulationType>
inline void Solver<dim, TriangulationType>::set_angular_velocity
(const std::shared_ptr<const Utility::AngularVelocity<dim>> &angular_velocity)
{
  angular_velocity_ptr = angular_velocity;
}



template <int dim, typename TriangulationType>
inline void Solver<dim, TriangulationType>::set_body_force
(const std::shared_ptr<const TensorFunction<1, dim>> &body_force)
{
  body_force_ptr = body_force;
}



template <int dim, typename TriangulationType>
inline void Solver<dim, TriangulationType>::set_background_velocity
(const std::shared_ptr<const TensorFunction<1, dim>> &velocity)
{
  background_velocity_ptr = velocity;
}



template <int dim, typename TriangulationType>
inline VectorBoundaryConditions<dim> &
Solver<dim, TriangulationType>::get_velocity_bcs()
{
  return velocity_boundary_conditions;
}



template <int dim, typename TriangulationType>
inline const VectorBoundaryConditions<dim> &
Solver<dim, TriangulationType>::get_velocity_bcs() const
{
  return velocity_boundary_conditions;
}


template <int dim, typename TriangulationType>
inline ScalarBoundaryConditions<dim> &
Solver<dim, TriangulationType>::get_pressure_bcs()
{
  return pressure_boundary_conditions;
}



template <int dim, typename TriangulationType>
inline const ScalarBoundaryConditions<dim> &
Solver<dim, TriangulationType>::get_pressure_bcs() const
{
  return pressure_boundary_conditions;
}



template <int dim, typename TriangulationType>
inline double Solver<dim, TriangulationType>::get_reynolds_number() const
{
  return reynolds_number;
}



template <int dim, typename TriangulationType>
inline double Solver<dim, TriangulationType>::get_froude_number() const
{
  return froude_number;
}

}  // namespace Hydrodynamic

#endif /* INCLUDE_HYDRODYNAMIC_SOLVER_H_ */
