/*
 * buoyant_hydrodynamic_solver.h
 *
 *  Created on: Aug 30, 2021
 *      Author: sg
 */

#ifndef INCLUDE_BUOYANT_HYDRODYNAMIC_SOLVER_H_
#define INCLUDE_BUOYANT_HYDRODYNAMIC_SOLVER_H_


#include <hydrodynamic_solver.h>

namespace BuoyantHydrodynamic {

using namespace BoundaryConditions;


/*!
 * @struct SolverParameters
 *
 * @brief A structure containing all the parameters of the Navier-Stokes
 * solver.
 */
struct SolverParameters: Hydrodynamic::SolverParameters
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

};



/*!
 * @brief Method forwarding parameters to a stream object.
 */
template <typename Stream>
Stream& operator<<(Stream &stream, const SolverParameters &prm);



template <int dim>
class Solver: public Hydrodynamic::Solver<dim>
{

public:
  Solver(Triangulation<dim>  &tria,
         Mapping<dim>        &mapping,
         const SolverParameters &parameters,
         const double         reynolds_number = 1.0,
         const double         froude_number = 0.0,
         const double         stratification_number = 1.0);

  void set_reference_density(const Function<dim> &reference_density);

  void set_gravity_field(const TensorFunction<1, dim> &gravity_field);

  ScalarBoundaryConditions<dim>&  get_density_bcs();

  const ScalarBoundaryConditions<dim>&  get_density_bcs() const;

  double get_stratification_number() const;

private:
  virtual void setup_fe_system();

  virtual void setup_dofs();

  virtual void apply_boundary_conditions();

  virtual void assemble_system(const bool initial_step);

  virtual void assemble_rhs(const bool initial_step);

  virtual void output_results(const unsigned int cycle = 0) const;

  ScalarBoundaryConditions<dim> density_boundary_conditions;

  const Function<dim>          *reference_density_ptr;

  const TensorFunction<1, dim> *gravity_field_ptr;

  const double        stratification_number;

  const unsigned int  density_fe_degree;

};

// inline functions
template <int dim>
inline const ScalarBoundaryConditions<dim> &
Solver<dim>::get_density_bcs() const
{
  return density_boundary_conditions;
}



template <int dim>
inline ScalarBoundaryConditions<dim> &
Solver<dim>::get_density_bcs()
{
  return density_boundary_conditions;
}



template <int dim>
inline void Solver<dim>::set_reference_density(const Function<dim> &reference_density)
{
  reference_density_ptr = &reference_density;
  return;
}



template <int dim>
inline void Solver<dim>::set_gravity_field(const TensorFunction<1, dim> &gravity_field)
{
  gravity_field_ptr = &gravity_field;
  return;
}



template <int dim>
inline double Solver<dim>::get_stratification_number() const
{
  return (stratification_number);
}

}  // namespace TopographyProblem



#endif /* INCLUDE_BUOYANT_HYDRODYNAMIC_SOLVER_H_ */
