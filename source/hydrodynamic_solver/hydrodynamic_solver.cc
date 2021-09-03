/*
 * setup.cc
 *
 *  Created on: Aug 31, 2021
 *      Author: sg
 */

#include <deal.II/numerics/data_out.h>

#include <hydrodynamic_solver.h>
#include <hydrodynamic_postprocessor.h>

#include <filesystem>
#include <fstream>
#include <string>

namespace TopographyProblem {

HydrodynamicSolverParameters::HydrodynamicSolverParameters()
:
SolverBaseParameters(),
convective_term_weak_form(ConvectiveTermWeakForm::standard),
viscous_term_weak_form(ViscousTermWeakForm::laplacean)
{}



void HydrodynamicSolverParameters::declare_parameters(ParameterHandler &prm)
{
  SolverBaseParameters::declare_parameters(prm);

  prm.enter_subsection("Hydrodynamic solver parameters");
  {
    prm.declare_entry("Convective term weak form",
                      "standard",
                      Patterns::Selection("standard|skew-symmetric|divergence|rotational"));

    prm.declare_entry("Viscous term weak form",
                      "standard",
                      Patterns::Selection("standard|laplacean|stress"));
  }
  prm.leave_subsection();
}



void HydrodynamicSolverParameters::parse_parameters(ParameterHandler &prm)
{
  SolverBaseParameters::parse_parameters(prm);

  prm.enter_subsection("Hydrodynamic solver parameters");
  {
    const std::string str_convective_term_weak_form(prm.get("Convective term weak form"));

    if (str_convective_term_weak_form == std::string("standard"))
      convective_term_weak_form = ConvectiveTermWeakForm::standard;
    else if (str_convective_term_weak_form == std::string("skew-symmetric"))
      convective_term_weak_form = ConvectiveTermWeakForm::skewsymmetric;
    else if (str_convective_term_weak_form == std::string("divergence"))
      convective_term_weak_form = ConvectiveTermWeakForm::divergence;
    else if (str_convective_term_weak_form == std::string("rotational"))
      convective_term_weak_form = ConvectiveTermWeakForm::rotational;
    else
      AssertThrow(false,
                  ExcMessage("Unexpected identifier for the weak form "
                             "of the convective term."));

    const std::string str_viscous_term_weak_form(prm.get("Viscous term weak form"));

    if (str_convective_term_weak_form == std::string("standard"))
      viscous_term_weak_form = ViscousTermWeakForm::laplacean;
    else if (str_convective_term_weak_form == std::string("laplacean"))
      viscous_term_weak_form = ViscousTermWeakForm::laplacean;
    else if (str_convective_term_weak_form == std::string("divergence"))
      viscous_term_weak_form = ViscousTermWeakForm::stress;
    else
      AssertThrow(false,
                  ExcMessage("Unexpected identifier for the weak form "
                             "of the viscous term."));
  }
  prm.leave_subsection();
}



template<typename Stream>
Stream& operator<<(Stream &stream, const HydrodynamicSolverParameters &prm)
{
  stream << static_cast<const SolverBaseParameters &>(prm);

  internal::add_header(stream);
  internal::add_line(stream, "Hydrodynamic solver parameters");
  internal::add_header(stream);

  switch (prm.convective_term_weak_form)
  {
    case ConvectiveTermWeakForm::standard:
      internal::add_line(stream, "Convective term weak form", "standard");
      break;
    case ConvectiveTermWeakForm::rotational:
      internal::add_line(stream, "Convective term weak form", "rotational");
      break;
    case ConvectiveTermWeakForm::divergence:
      internal::add_line(stream, "Convective term weak form", "divergence");
      break;
    case ConvectiveTermWeakForm::skewsymmetric:
      internal::add_line(stream, "Convective term weak form", "skew-symmetric");
      break;
    default:
      AssertThrow(false, ExcMessage("Unexpected type identifier for the "
                                    "weak form of the convective term."));
      break;
  }

  switch (prm.viscous_term_weak_form)
  {
    case ViscousTermWeakForm::laplacean:
      internal::add_line(stream, "Viscous term weak form", "Laplacean");
      break;
    case ViscousTermWeakForm::stress:
      internal::add_line(stream, "Viscous term weak form", "stress");
      break;
    default:
      AssertThrow(false, ExcMessage("Unexpected type identifier for the "
                                    "weak form of the viscous term."));
      break;
  }

  return (stream);
}



template <int dim>
HydrodynamicSolver<dim>::HydrodynamicSolver
(Triangulation<dim>     &tria,
 Mapping<dim>           &mapping,
 const HydrodynamicSolverParameters &parameters,
 const double           reynolds,
 const double           froude)
:
SolverBase<dim>(tria, mapping, parameters),
velocity_boundary_conditions(this->triangulation),
pressure_boundary_conditions(this->triangulation),
body_force_ptr(nullptr),
convective_term_weak_form(parameters.convective_term_weak_form),
viscous_term_weak_form(parameters.viscous_term_weak_form),
velocity_fe_degree(2),
reynolds_number(reynolds),
froude_number(froude)
{}



template<int dim>
void HydrodynamicSolver<dim>::output_results(const unsigned int cycle) const
{
  if (this->verbose)
    std::cout << "    Output results..." << std::endl;

  HydrodynamicPostprocessor<dim>  postprocessor(0, dim);

  // prepare data out object
  DataOut<dim, DoFHandler<dim>>    data_out;
  data_out.attach_dof_handler(this->dof_handler);
  data_out.add_data_vector(this->present_solution, postprocessor);

  data_out.build_patches(velocity_fe_degree);

  // write output to disk
  const std::string filename = ("solution-" +
                                Utilities::int_to_string(cycle, 2) +
                                ".vtk");
  std::filesystem::path output_file(this->graphical_output_directory);
  output_file /= filename;

  std::ofstream fstream(output_file.c_str());
  data_out.write_vtk(fstream);
}

// explicit instantiation
template std::ostream & operator<<(std::ostream &, const HydrodynamicSolverParameters &);

template HydrodynamicSolver<2>::HydrodynamicSolver
(Triangulation<2>  &,
 Mapping<2>        &,
 const HydrodynamicSolverParameters &,
 const double       ,
 const double        );
template HydrodynamicSolver<3>::HydrodynamicSolver
(Triangulation<3>  &,
 Mapping<3>        &,
 const HydrodynamicSolverParameters &,
 const double       ,
 const double        );

template void HydrodynamicSolver<2>::output_results(const unsigned int ) const;
template void HydrodynamicSolver<3>::output_results(const unsigned int ) const;

template class HydrodynamicSolver<2>;
template class HydrodynamicSolver<3>;

}  // namespace TopographyProblem
