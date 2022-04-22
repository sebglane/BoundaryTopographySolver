/*
 * buoyant_hydrodynamic_assembly_data.cc
 *
 *  Created on: Apr 12, 2022
 *      Author: sg
 */

#include <buoyant_hydrodynamic_assembly_data.h>

namespace BuoyantHydrodynamic {

namespace AssemblyData {

namespace Matrix {

template <int dim>
ScratchData<dim>::ScratchData
(const Mapping<dim>       &mapping,
 const FiniteElement<dim> &fe,
 const Quadrature<dim>    &quadrature,
 const UpdateFlags        &update_flags,
 const Quadrature<dim-1>  &face_quadrature,
 const UpdateFlags        &face_update_flags)
:
dealii::MeshWorker::ScratchData<dim>(mapping,
                                     fe,
                                     quadrature,
                                     update_flags,
                                     face_quadrature,
                                     face_update_flags),
Hydrodynamic::AssemblyData::Matrix::ScratchData<dim>(mapping,
                                                     fe,
                                                     quadrature,
                                                     update_flags,
                                                     face_quadrature,
                                                     face_update_flags),
Advection::AssemblyData::ScratchData<dim>(mapping,
                                          fe,
                                          quadrature,
                                          update_flags,
                                          face_quadrature,
                                          face_update_flags)
{}



template <int dim>
ScratchData<dim>::ScratchData
(const FiniteElement<dim> &fe,
 const Quadrature<dim>    &quadrature,
 const UpdateFlags        &update_flags,
 const Quadrature<dim-1>  &face_quadrature,
 const UpdateFlags        &face_update_flags)
:
dealii::MeshWorker::ScratchData<dim>(fe,
                                     quadrature,
                                     update_flags,
                                     face_quadrature,
                                     face_update_flags),
Hydrodynamic::AssemblyData::Matrix::ScratchData<dim>(fe,
                                                     quadrature,
                                                     update_flags,
                                                     face_quadrature,
                                                     face_update_flags),
Advection::AssemblyData::ScratchData<dim>(fe,
                                          quadrature,
                                          update_flags,
                                          face_quadrature,
                                          face_update_flags)
{}




template <int dim>
ScratchData<dim>::ScratchData(const ScratchData<dim> &other)
:
dealii::MeshWorker::ScratchData<dim>(other),
Hydrodynamic::AssemblyData::Matrix::ScratchData<dim>(other),
Advection::AssemblyData::ScratchData<dim>(other)
{}


template class ScratchData<2>;
template class ScratchData<3>;

}  // namespace Matrix

namespace RightHandSide {

template <int dim>
ScratchData<dim>::ScratchData
(const FiniteElement<dim> &fe,
 const Quadrature<dim>    &quadrature,
 const UpdateFlags        &update_flags,
 const Quadrature<dim-1>  &face_quadrature,
 const UpdateFlags        &face_update_flags)
:
dealii::MeshWorker::ScratchData<dim>(fe,
                                     quadrature,
                                     update_flags,
                                     face_quadrature,
                                     face_update_flags),
Hydrodynamic::AssemblyData::RightHandSide::ScratchData<dim>(fe,
                                                            quadrature,
                                                            update_flags,
                                                            face_quadrature,
                                                            face_update_flags),
Advection::AssemblyData::ScratchData<dim>(fe,
                                          quadrature,
                                          update_flags,
                                          face_quadrature,
                                          face_update_flags)
{}




template <int dim>
ScratchData<dim>::ScratchData(const ScratchData<dim> &other)
:
dealii::MeshWorker::ScratchData<dim>(other),
Hydrodynamic::AssemblyData::RightHandSide::ScratchData<dim>(other),
Advection::AssemblyData::ScratchData<dim>(other)
{}

template class ScratchData<2>;
template class ScratchData<3>;

}  // namespace RightHandSide

}  // namespace AssemblyData

}  // namespace Buoyanthydrodynamic


