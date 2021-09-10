/*
 * buoyant_hydrodynamic_problem.h
 *
 *  Created on: Sep 8, 2021
 *      Author: sg
 */

#ifndef INCLUDE_ADVECTION_PROBLEM_H_
#define INCLUDE_ADVECTION_PROBLEM_H_

#include <deal.II/fe/mapping_q_cache.h>

#include <advection_solver.h>

namespace Advection {

/*!
 * @struct ProblemParameters
 *
 * @brief A structure containing all the parameters of the advection solver.
 */
struct ProblemParameters: SolverParameters
{
  /*!
   * Constructor which sets up the parameters with default values.
   */
  ProblemParameters();

  /*!
   * @brief Constructor which sets up the parameters as specified in the
   * parameter file with the filename @p parameter_filename.
   */
  ProblemParameters(const std::string &filename);

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
  friend Stream& operator<<(Stream &stream, const ProblemParameters &prm);

  /*!
   * @brief Polynomial degree of the mapping.
   */
  unsigned int  mapping_degree;

};



/*!
 * @brief Method forwarding parameters to a stream object.
 */
template <typename Stream>
Stream& operator<<(Stream &stream, const ProblemParameters &prm);



template <int dim>
class AdvectionProblem
{
public:
  AdvectionProblem(const ProblemParameters &parameters);

  void run();

protected:
  virtual void make_grid() = 0;

  void initialize_mapping();

  virtual void set_boundary_conditions() = 0;

  virtual void set_source_term();

  virtual void set_advection_field() = 0;

  Triangulation<dim>       triangulation;

  MappingQCache<dim>       mapping;

  Solver<dim>             solver;

};

// inline functions
template <int dim>
void AdvectionProblem<dim>::set_source_term()
{
  return;
}

}  // namespace Advection

#endif /* INCLUDE_ADVECTION_PROBLEM_H_ */
