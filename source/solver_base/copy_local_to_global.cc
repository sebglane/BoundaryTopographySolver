/*
 * copy_local_to_global.cc
 *
 *  Created on: Apr 11, 2022
 *      Author: sg
 */

#include <base.h>

namespace Base {

template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::
copy_local_to_global_system
(const MeshWorker::CopyData<1,1,1> &data,
 const bool                         use_homogeneous_constraints)
{
  const AffineConstraints<double> &constraints =
      (use_homogeneous_constraints ? this->zero_constraints: this->nonzero_constraints);

  constraints.distribute_local_to_global(data.matrices[0],
                                         data.vectors[0],
                                         data.local_dof_indices[0],
                                         this->system_matrix,
                                         this->system_rhs);
}



template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::copy_local_to_global_rhs
(const MeshWorker::CopyData<0,1,1>  &data,
 const bool use_homogeneous_constraints)
{
  const AffineConstraints<double> &constraints =
      (use_homogeneous_constraints ? this->zero_constraints: this->nonzero_constraints);

  constraints.distribute_local_to_global(data.vectors[0],
                                         data.local_dof_indices[0],
                                         this->system_rhs);
}


// explicit instantiation
template
void
Solver<2>::
copy_local_to_global_system
(const MeshWorker::CopyData<1,1,1> &, const bool);
template
void
Solver<3>::
copy_local_to_global_system
(const MeshWorker::CopyData<1,1,1> &, const bool);

template
void
Solver<2>::
copy_local_to_global_rhs
(const MeshWorker::CopyData<0,1,1> &, const bool);
template
void
Solver<3>::
copy_local_to_global_rhs
(const MeshWorker::CopyData<0,1,1> &, const bool);



}  // namespace Base


