/*
 * hydrodynamic_problem.h
 *
 *  Created on: Sep 1, 2021
 *      Author: sg
 */

#ifndef INCLUDE_HYDRODYNAMIC_PROBLEM_H_
#define INCLUDE_HYDRODYNAMIC_PROBLEM_H_

#include <deal.II/fe/mapping_q_cache.h>

#include <hydrodynamic_solver.h>

namespace TopographyProblem {

/*!
 * @struct HydrodynamicSolverParameters
 *
 * @brief A structure containing all the parameters of the Navier-Stokes
 * solver.
 */
struct HydrodynamicProblemParameters: HydrodynamicSolverParameters
{
  /*!
   * Constructor which sets up the parameters with default values.
   */
  HydrodynamicProblemParameters();

  /*!
   * @brief Constructor which sets up the parameters as specified in the
   * parameter file with the filename @p parameter_filename.
   */
  HydrodynamicProblemParameters(const std::string &filename);

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
  friend Stream& operator<<(Stream &stream, const HydrodynamicProblemParameters &prm);

  /*!
   * @brief Polynomial degree of the mapping.
   */
  unsigned int  mapping_degree;

  /*!
   * @brief The Froude number of the problem.
   */
  double        froude_number;

  /*!
   * @brief The Reynolds number of the problem.
   */
  double        reynolds_number;

};



/*!
 * @brief Method forwarding parameters to a stream object.
 */
template <typename Stream>
Stream& operator<<(Stream &stream, const HydrodynamicProblemParameters &prm);



template <int dim>
class HydrodynamicProblem
{
public:
  HydrodynamicProblem(const HydrodynamicProblemParameters &parameters);

  void run();

protected:
  virtual void make_grid() = 0;

  void initialize_mapping();

  virtual void set_boundary_conditions() = 0;

  virtual void set_body_force_term();

  Triangulation<dim>       triangulation;

  MappingQCache<dim>       mapping;

  HydrodynamicSolver<dim>  solver;

};

// inline functions
template <int dim>
void HydrodynamicProblem<dim>::set_body_force_term()
{
  return;
}

}  // namespace TopographyProblem



#endif /* INCLUDE_HYDRODYNAMIC_PROBLEM_H_ */
