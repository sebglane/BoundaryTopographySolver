/*
 * parameters.h
 *
 *  Created on: Sep 2, 2021
 *      Author: sg
 */

#ifndef INCLUDE_PARAMETERS_H_
#define INCLUDE_PARAMETERS_H_

#include <deal.II/base/parameter_handler.h>

namespace Utility {

using namespace dealii;

/*!
 * @brief Prints a table row consisting of a single column with a fixed width
 * and `|` delimiters.
 */
template<typename Stream, typename A>
void add_line(Stream  &stream, const A line);

/*!
 * @brief Prints a table row consisting of a two columns with a fixed width
 * and `|` delimiters.
 */
template<typename Stream, typename A, typename B>
void add_line(Stream  &stream, const A first_column, const B second_column);

/*!
 * @brief Prints a table header with a fixed width and `|` delimiters.
 */
template<typename Stream>
void add_header(Stream  &stream);



/*!
 * @struct RefinementParameters
 *
 * @brief @ref RefinementParameters contains parameters which are
 * related to the adaptive refinement of the mesh.
 */
struct RefinementParameters
{
  /*!
   * @brief Constructor which sets up the parameters with default values.
   */
  RefinementParameters();

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
   */
  template <typename Stream>
  friend Stream& operator<<(Stream &stream,
                            const RefinementParameters &prm);

  /*!
   * @brief Boolean flag to enable or disable adaptive mesh refinement.
   */
  bool          adaptive_mesh_refinement;

  /*!
   * @brief The upper fraction of the total number of cells set to
   * coarsen.
   */
  double        cell_fraction_to_coarsen;

  /*!
   * @brief The lower fraction of the total number of cells set to
   * refine.
   */
  double        cell_fraction_to_refine;

  /*!
   * @brief The number of maximum levels of the mesh. This parameter prohibits
   * a further refinement of the mesh.
   */
  unsigned int  n_maximum_levels;

  /*!
   * @brief The number of minimum levels of the mesh. This parameter prohibits
   * a further coarsening of the mesh.
   */
  unsigned int  n_minimum_levels;

  /*!
   * @brief Boolean flag to enable or disable adaptive mesh refinement.
   */
  unsigned int  n_cycles;

  /*!
   * @brief Number of initial refinements.
   */
  unsigned int  n_initial_refinements;

  /*!
   * @brief Number of initial boundary refinements.
   */
  unsigned int  n_initial_bndry_refinements;

};



/*!
 * @brief Method forwarding parameters to a stream object.
 *
 * @details This method does not add a `std::endl` to the stream at the end.
 */
template <typename Stream>
Stream& operator<<(Stream &stream, const RefinementParameters &prm);


}  // namespace Utility


#endif /* INCLUDE_PARAMETERS_H_ */
