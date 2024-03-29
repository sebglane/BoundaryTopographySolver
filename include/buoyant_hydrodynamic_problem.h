/*
 * buoyant_hydrodynamic_problem.h
 *
 *  Created on: Sep 8, 2021
 *      Author: sg
 */

#ifndef INCLUDE_BUOYANT_HYDRODYNAMIC_PROBLEM_H_
#define INCLUDE_BUOYANT_HYDRODYNAMIC_PROBLEM_H_

#include <deal.II/fe/mapping_q_cache.h>

#include <buoyant_hydrodynamic_solver.h>

namespace BuoyantHydrodynamic {

/*!
 * @struct ProblemParameters
 *
 * @brief A structure containing all the parameters of the Navier-Stokes
 * solver.
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

  /*!
   * @brief The Froude number of the problem.
   */
  double        froude_number;

  /*!
   * @brief The Reynolds number of the problem.
   */
  double        reynolds_number;

  /*!
   * @brief The stratification number of the problem.
   */
  double        stratification_number;

  /*!
   * @brief The Rossby number of the problem.
   */
  double        rossby_number;

};



/*!
 * @brief Method forwarding parameters to a stream object.
 */
template <typename Stream>
Stream& operator<<(Stream &stream, const ProblemParameters &prm);


/**
 * @todo Add documentation.
 */
template <int dim,
          typename TriangulationType = Triangulation<dim>>
class BuoyantHydrodynamicProblem
{
public:
  BuoyantHydrodynamicProblem(const ProblemParameters &parameters);

  void run();

protected:
  virtual void make_grid() = 0;

  void initialize_mapping();

  virtual void set_angular_velocity();

  virtual void set_background_velocity();

  virtual void set_boundary_conditions() = 0;

  virtual void set_body_force_term();

  virtual void set_gravity_field() = 0;

  virtual void set_reference_density() = 0;

  virtual void set_postprocessor();

  TriangulationType       triangulation;

  MappingQCache<dim>      mapping;

  Solver<dim, TriangulationType>  solver;

  const unsigned int      n_initial_refinements;

  const unsigned int      n_initial_bndry_refinements;
};

// inline functions
template <int dim, typename TriangulationType>
inline void BuoyantHydrodynamicProblem<dim, TriangulationType>::set_angular_velocity()
{
  return;
}



template <int dim, typename TriangulationType>
inline void BuoyantHydrodynamicProblem<dim, TriangulationType>::set_background_velocity()
{
  return;
}



template <int dim, typename TriangulationType>
void BuoyantHydrodynamicProblem<dim, TriangulationType>::set_body_force_term()
{
  return;
}



template <int dim, typename TriangulationType>
void BuoyantHydrodynamicProblem<dim, TriangulationType>::set_postprocessor()
{
  return;
}

}  // namespace BuoyantHydrodynamic

#endif /* INCLUDE_BUOYANT_HYDRODYNAMIC_PROBLEM_H_ */
