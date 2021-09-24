/*
 * parameters.cc
 *
 *  Created on: Sep 2, 2021
 *      Author: sg
 */

#include <parameters.h>

namespace Utility {

namespace
{

constexpr char header[] = "+------------------------------------------+"
                      "----------------------+";

constexpr size_t column_width[2] ={ 40, 20 };

constexpr size_t line_width = 63;

}



template<typename Stream, typename A>
void add_line(Stream  &stream, const A line)
{
  stream << "| "
         << std::setw(line_width)
         << line
         << " |"
         << std::endl;
  return;
}



template<typename Stream, typename A, typename B>
void add_line(Stream  &stream, const A first_column, const B second_column)
{
  stream << "| "
         << std::setw(column_width[0]) << first_column
         << " | "
         << std::setw(column_width[1]) << second_column
         << " |"
         << std::endl;
  return;
}



template<typename Stream>
void add_header(Stream  &stream)
{
  stream << std::left << header << std::endl;
  return;
}



RefinementParameters::RefinementParameters()
:
adaptive_mesh_refinement(false),
cell_fraction_to_coarsen(0.30),
cell_fraction_to_refine(0.03),
n_maximum_levels(5),
n_minimum_levels(1),
n_cycles(2),
n_initial_refinements(1),
n_initial_bndry_refinements(0)
{}



void RefinementParameters::declare_parameters(ParameterHandler &prm)
{

  prm.enter_subsection("Refinement control parameters");
  {
    prm.declare_entry("Adaptive mesh refinement",
                      "false",
                      Patterns::Bool());

    prm.declare_entry("Fraction of cells set to coarsen",
                      "0.3",
                      Patterns::Double(0));

    prm.declare_entry("Fraction of cells set to refine",
                      "0.03",
                      Patterns::Double(0));

    prm.declare_entry("Maximum number of levels",
                      "5",
                      Patterns::Integer(1));

    prm.declare_entry("Minimum number of levels",
                      "0",
                      Patterns::Integer(0));

    prm.declare_entry("Number of refinement cycles",
                      "2",
                      Patterns::Integer(1));

    prm.declare_entry("Number of initial refinements",
                      "1",
                      Patterns::Integer(1));

    prm.declare_entry("Number of initial boundary refinements",
                      "0",
                      Patterns::Integer());

  }
  prm.leave_subsection();
}



void RefinementParameters::parse_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Refinement control parameters");
  {
    adaptive_mesh_refinement = prm.get_bool("Adaptive mesh refinement");

    n_cycles = prm.get_integer("Number of refinement cycles");

    n_initial_refinements = prm.get_integer("Number of initial refinements");
    AssertThrow(n_initial_refinements > 0, ExcLowerRange(n_initial_refinements, 0) );

    n_initial_bndry_refinements = prm.get_integer("Number of initial boundary refinements");
    AssertThrow(n_initial_refinements > 0, ExcLowerRange(n_initial_refinements, 0) );

    if (adaptive_mesh_refinement)
    {
      n_minimum_levels = prm.get_integer("Minimum number of levels");
      n_maximum_levels = prm.get_integer("Maximum number of levels");
      AssertThrow(n_minimum_levels > 0,
                  ExcMessage("Minimum number of levels must be larger than zero."));
      AssertThrow(n_minimum_levels <= n_maximum_levels ,
                  ExcMessage("Maximum number of levels must be larger equal "
                             "than the minimum number of levels."));

      cell_fraction_to_coarsen = prm.get_double("Fraction of cells set to coarsen");

      cell_fraction_to_refine = prm.get_double("Fraction of cells set to refine");

      const double total_cell_fraction_to_modify =
        cell_fraction_to_coarsen + cell_fraction_to_refine;

      AssertThrow(cell_fraction_to_coarsen >= 0.0,
                  ExcLowerRangeType<double>(cell_fraction_to_coarsen, 0));

      AssertThrow(cell_fraction_to_refine >= 0.0,
                  ExcLowerRangeType<double>(cell_fraction_to_refine, 0));

      AssertThrow(1.0 > total_cell_fraction_to_modify,
                  ExcMessage("The sum of the top and bottom fractions to "
                             "coarsen and refine may not exceed 1.0"));
    }
  }
  prm.leave_subsection();
}



template <typename Stream>
Stream& operator<<(Stream &stream, const RefinementParameters &prm)
{
  add_header(stream);
  add_line(stream, "Refinement control parameters");
  add_header(stream);

  add_line(stream,"Adaptive mesh refinement", (prm.adaptive_mesh_refinement ? "True": "False"));

  add_line(stream, "Initial refinements", prm.n_initial_refinements);

  add_line(stream, "Initial boundary refinements", prm.n_initial_bndry_refinements);

  if (prm.adaptive_mesh_refinement)
  {
    add_line(stream,
             "Fraction of cells set to coarsen", prm.cell_fraction_to_coarsen);
    add_line(stream,
             "Fraction of cells set to refine",
             prm.cell_fraction_to_refine);
    add_line(stream,
             "Maximum number of levels", prm.n_maximum_levels);
    add_line(stream,
             "Minimum number of levels", prm.n_minimum_levels);
  }

  add_line(stream,
           "Number of refinement cycles",
           prm.n_cycles);

  return (stream);
}

// explicit instantiations
template void add_header(std::ostream &);

template void add_line(std::ostream &, const char[]);
template void add_line(std::ostream &, std::string);

template void add_line(std::ostream &, const char[], const double);
template void add_line(std::ostream &, const char[], const unsigned int);
template void add_line(std::ostream &, const char[], const int);
template void add_line(std::ostream &, const char[], const std::string);
template void add_line(std::ostream &, const char[], const char[]);

template void add_line(std::ostream &, const std::string, const double);
template void add_line(std::ostream &, const std::string, const unsigned int);
template void add_line(std::ostream &, const std::string, const int);
template void add_line(std::ostream &, const std::string, const std::string);
template void add_line(std::ostream &, const std::string, const char[]);

template std::ostream & operator<<(std::ostream &, const RefinementParameters &);

}  // namespace Utility
