/*
 * hydrodynamic_solver.cc
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

namespace Hydrodynamic {

SolverParameters::SolverParameters()
:
SolverBase::Parameters(),
convective_term_weak_form(ConvectiveTermWeakForm::standard),
viscous_term_weak_form(ViscousTermWeakForm::laplacean),
stabilization(apply_none),
c(1.0),
mu(1.0)
{}



void SolverParameters::declare_parameters(ParameterHandler &prm)
{
  SolverBase::Parameters::declare_parameters(prm);

  prm.enter_subsection("Hydrodynamic solver parameters");
  {
    prm.declare_entry("Convective term weak form",
                      "standard",
                      Patterns::Selection("standard|skew-symmetric|divergence|rotational"));

    prm.declare_entry("Viscous term weak form",
                      "standard",
                      Patterns::Selection("standard|laplacean|stress"));

    prm.declare_entry("Stabilization type",
                      "none",
                      Patterns::Selection("none|SUPG|PSPG|GradDiv|"
                          "SUPG_GradDiv|GradDiv_SUPG|"
                          "PSPG_GradDiv|GradDiv_PSPG|"
                          "SUPG_PSPG_GradDiv|GradDiv_SUPG_PSPG|PSPG_GradDiv_SUPG|"
                          "SUPG_GradDiv_PSPG|GradDiv_PSPG_SUPG|PSPG_SUPG_GradDiv"));

    prm.declare_entry("SUPG/PSPG stabilization parameter",
                      "1.0",
                      Patterns::Double(std::numeric_limits<double>::epsilon()));

    prm.declare_entry("Grad-Div stabilization parameter",
                      "1.0",
                      Patterns::Double(std::numeric_limits<double>::epsilon()));

  }
  prm.leave_subsection();
}



void SolverParameters::parse_parameters(ParameterHandler &prm)
{
  SolverBase::Parameters::parse_parameters(prm);

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

    const std::string stabilization_str = prm.get("Stabilization type");

    stabilization = apply_none;
    if (stabilization_str.find("SUPG") != std::string::npos)
      stabilization |= apply_supg;
    if (stabilization_str.find("PSPG") != std::string::npos)
          stabilization |= apply_pspg;
    if (stabilization_str.find("GradDiv") != std::string::npos)
          stabilization |= apply_grad_div;

    c = prm.get_double("SUPG/PSPG stabilization parameter");
    AssertIsFinite(c);
    Assert(c > 0.0, ExcLowerRangeType<double>(c, 0.0));

    mu = prm.get_double("Grad-Div stabilization parameter");
    AssertIsFinite(mu);
    Assert(mu > 0.0, ExcLowerRangeType<double>(mu, 0.0));

  }
  prm.leave_subsection();
}



template<typename Stream>
Stream& operator<<(Stream &stream, const SolverParameters &prm)
{
  stream << static_cast<const SolverBase::Parameters &>(prm);

  Utility::add_header(stream);
  Utility::add_line(stream, "Hydrodynamic solver parameters");
  Utility::add_header(stream);

  switch (prm.convective_term_weak_form)
  {
    case ConvectiveTermWeakForm::standard:
      Utility::add_line(stream, "Convective term weak form", "standard");
      break;
    case ConvectiveTermWeakForm::rotational:
      Utility::add_line(stream, "Convective term weak form", "rotational");
      break;
    case ConvectiveTermWeakForm::divergence:
      Utility::add_line(stream, "Convective term weak form", "divergence");
      break;
    case ConvectiveTermWeakForm::skewsymmetric:
      Utility::add_line(stream, "Convective term weak form", "skew-symmetric");
      break;
    default:
      AssertThrow(false, ExcMessage("Unexpected type identifier for the "
                                    "weak form of the convective term."));
      break;
  }

  switch (prm.viscous_term_weak_form)
  {
    case ViscousTermWeakForm::laplacean:
      Utility::add_line(stream, "Viscous term weak form", "Laplacean");
      break;
    case ViscousTermWeakForm::stress:
      Utility::add_line(stream, "Viscous term weak form", "stress");
      break;
    default:
      AssertThrow(false, ExcMessage("Unexpected type identifier for the "
                                    "weak form of the viscous term."));
      break;
  }

  std::stringstream sstream;
  sstream << prm.stabilization;
  Utility::add_line(stream, "Stabilization type", sstream.str().c_str());

  Utility::add_line(stream, "SUPG/PSPG stab. parameter", prm.c);
  Utility::add_line(stream, "Grad-Div stab. parameter", prm.mu);

  return (stream);
}



template <int dim>
Solver<dim>::Solver
(Triangulation<dim>     &tria,
 Mapping<dim>           &mapping,
 const SolverParameters &parameters,
 const double           reynolds,
 const double           froude)
:
SolverBase::Solver<dim>(tria, mapping, parameters),
velocity_boundary_conditions(this->triangulation),
pressure_boundary_conditions(this->triangulation),
body_force_ptr(nullptr),
convective_term_weak_form(parameters.convective_term_weak_form),
viscous_term_weak_form(parameters.viscous_term_weak_form),
stabilization(parameters.stabilization),
velocity_fe_degree(2),
reynolds_number(reynolds),
froude_number(froude),
c(1.0),
mu(1.0)
{}



template<int dim>
void Solver<dim>::output_results(const unsigned int cycle) const
{
  if (this->verbose)
    std::cout << "    Output results..." << std::endl;

  Postprocessor<dim>  postprocessor(0, dim);

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
template std::ostream & operator<<(std::ostream &, const SolverParameters &);

template Solver<2>::Solver
(Triangulation<2>  &,
 Mapping<2>        &,
 const SolverParameters &,
 const double       ,
 const double        );
template Solver<3>::Solver
(Triangulation<3>  &,
 Mapping<3>        &,
 const SolverParameters &,
 const double       ,
 const double        );

template void Solver<2>::output_results(const unsigned int ) const;
template void Solver<3>::output_results(const unsigned int ) const;

template class Solver<2>;
template class Solver<3>;

}  // namespace Hydrodynamic
