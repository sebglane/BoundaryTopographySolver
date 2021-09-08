/*
 * hydrodynamic_solver.h
 *
 *  Created on: Aug 30, 2021
 *      Author: sg
 */

#ifndef INCLUDE_HYDRODYNAMIC_SOLVER_H_
#define INCLUDE_HYDRODYNAMIC_SOLVER_H_

#include <boundary_conditions.h>

#include <solver_base.h>

namespace Hydrodynamic {

using namespace BoundaryConditions;


/*!
 * @brief Enumeration for the weak form of the convective term.
 *
 * @attention These definitions are the ones I see the most in the literature.
 * Nonetheless Volker John and Helene Dallmann define the skew-symmetric
 * and the divergence form differently.
 */
enum class ConvectiveTermWeakForm
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
enum class ViscousTermWeakForm
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
   * term is to be implemented.
   */
  ConvectiveTermWeakForm convective_term_weak_form;

  /*!
   * @brief Enumeration controlling which weak form of the viscous
   * term is to be implemented.
   */
  ViscousTermWeakForm    viscous_term_weak_form;

};



/*!
 * @brief Method forwarding parameters to a stream object.
 */
template <typename Stream>
Stream& operator<<(Stream &stream, const SolverParameters &prm);



template <int dim>
class Solver: public SolverBase::Solver<dim>
{
public:
  Solver(Triangulation<dim>  &tria,
         Mapping<dim>        &mapping,
         const SolverParameters &parameters,
         const double         reynolds_number = 1.0,
         const double         froude_number = 0.0);

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

  virtual void output_results(const unsigned int cycle = 0) const;

protected:
  VectorBoundaryConditions<dim> velocity_boundary_conditions;

  ScalarBoundaryConditions<dim> pressure_boundary_conditions;

  const TensorFunction<1, dim> *body_force_ptr;

  ConvectiveTermWeakForm     convective_term_weak_form;

  ViscousTermWeakForm        viscous_term_weak_form;

  const unsigned int  velocity_fe_degree;

  const double        reynolds_number;

  const double        froude_number;
};

// inline functions
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
