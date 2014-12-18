/// \file vc/Vector3D.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BACKEND_VC_VECTOR3D_H_
#define VECGEOM_BACKEND_VC_VECTOR3D_H_

#include "base/Global.h"

#include "backend/vc/Backend.h"
#include "base/Vector3D.h"

namespace VECGEOM_NAMESPACE {

template <typename T1, typename T2>
void MaskedAssign(
    typename Vc::Vector<T1>::Mask const &condition,
    Vector3D<T2> const &value,
    Vector3D<Vc::Vector<T1> > *output) {
  ((*output)[0])(condition) = value[0];
  ((*output)[1])(condition) = value[1];
  ((*output)[2])(condition) = value[2];
}

}; // End global namespace

#endif // VECGEOM_BACKEND_VC_VECTOR3D_H_