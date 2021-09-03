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

namespace TopographyProblem {

/*!
 * @struct HydrodynamicSolverParameters
 *
 * @brief A structure containing all the parameters of the Navier-Stokes
 * solver.
 */
struct HydrodynamicSolverParameters: SolverBaseParameters
{
  /*!
   * Constructor which sets up the parameters with default values.
   */
  HydrodynamicSolverParameters();

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
  friend Stream& operator<<(Stream &stream, const HydrodynamicSolverParameters &prm);

  /*!
   * @brief Enumerator controlling which weak form of the convective
   * term is to be implemented.
   */
  ConvectiveTermWeakForm            convective_term_weak_form;

  /*!
   * @brief Enumeration controlling which weak form of the viscous
   * term is to be implemented.
   */
  ViscousTermWeakForm               viscous_term_weak_form;

};



/*!
 * @brief Method forwarding parameters to a stream object.
 */
template <typename Stream>
Stream& operator<<(Stream &stream, const HydrodynamicSolverParameters &prm);



template <int dim>
class HydrodynamicSolver: public SolverBase<dim>
{

public:
  HydrodynamicSolver(Triangulation<dim>  &tria,
                     Mapping<dim>        &mapping,
                     const HydrodynamicSolverParameters &parameters,
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

  VectorBoundaryConditions<dim> velocity_boundary_conditions;

  ScalarBoundaryConditions<dim> pressure_boundary_conditions;

  const TensorFunction<1, dim>       *body_force_ptr;

  ConvectiveTermWeakForm            convective_term_weak_form;

  ViscousTermWeakForm               viscous_term_weak_form;

  const unsigned int  velocity_fe_degree;

  const double        reynolds_number;

  const double        froude_number;

};

// inline functions
template <int dim>
inline void HydrodynamicSolver<dim>::set_body_force
(const TensorFunction<1, dim> &body_force)
{
  body_force_ptr = &body_force;
}



template <int dim>
inline VectorBoundaryConditions<dim> &
HydrodynamicSolver<dim>::get_velocity_bcs()
{
  return velocity_boundary_conditions;
}



template <int dim>
inline const VectorBoundaryConditions<dim> &
HydrodynamicSolver<dim>::get_velocity_bcs() const
{
  return velocity_boundary_conditions;
}


template <int dim>
inline ScalarBoundaryConditions<dim> &
HydrodynamicSolver<dim>::get_pressure_bcs()
{
  return pressure_boundary_conditions;
}



template <int dim>
inline const ScalarBoundaryConditions<dim> &
HydrodynamicSolver<dim>::get_pressure_bcs() const
{
  return pressure_boundary_conditions;
}



template <int dim>
inline double HydrodynamicSolver<dim>::get_reynolds_number() const
{
  return reynolds_number;
}



template <int dim>
inline double HydrodynamicSolver<dim>::get_froude_number() const
{
  return froude_number;
}

}  // namespace TopographyProblem

#endif /* INCLUDE_HYDRODYNAMIC_SOLVER_H_ */
