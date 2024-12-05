// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// \brief A collection of template helpers to automize implementation
///        of UnplacedVolume interfaces.
/// \file volumes/UnplacedVolumeImplHelper.h
/// \author First version created by Sandro Wenzel

#ifndef VOLUMES_UNPLACEDVOLUMEIMPLHELPER_H_
#define VOLUMES_UNPLACEDVOLUMEIMPLHELPER_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/management/VolumeFactory.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

/*!
 * A template class with the aim to automatically implement
 * interfaces of UnplacedVolume by connecting them to the separately
 * available (reusable) kernels.
 *
 * This is an application of the CRT pattern in order to reduce
 * repeating code.
 */
template <class Implementation, class BaseUnplVol = VUnplacedVolume>
class UnplacedVolumeImplHelper : public BaseUnplVol {

public:
  using UnplacedStruct_t = typename Implementation::UnplacedStruct_t;
  using UnplacedVolume_t = typename Implementation::UnplacedVolume_t;

  // bring in constructors
  using BaseUnplVol::BaseUnplVol;
  // bring in some members from base (to avoid name hiding)
  using BaseUnplVol::DistanceToIn;
  using BaseUnplVol::DistanceToOut;
  using BaseUnplVol::SafetyToIn;
  using BaseUnplVol::SafetyToOut;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  virtual Precision DistanceToOut(Vector3D<Precision> const &p, Vector3D<Precision> const &d,
                                  Precision step_max = kInfLength) const override
  {
#ifndef VECCORE_CUDA
    assert(d.IsNormalized() && " direction not normalized in call to  DistanceToOut ");
#endif
    Precision output = kInfLength;
    Implementation::template DistanceToOut<>(((UnplacedVolume_t *)this)->UnplacedVolume_t::GetStruct(), p, d, step_max,
                                           output);

// detect -inf responses which are often an indication for a real bug
#ifndef VECCORE_CUDA
    assert(!((output < 0.) && std::isinf((Precision)output)));
#endif
    return output;
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  virtual bool Contains(Vector3D<Precision> const &p) const override
  {
    bool output(false);
    Implementation::Contains(((UnplacedVolume_t *)this)->UnplacedVolume_t::GetStruct(), p, output);
    return output;
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  virtual EnumInside Inside(Vector3D<Precision> const &p) const override
  {
    Inside_t output(0);
    Implementation::Inside(((UnplacedVolume_t *)this)->UnplacedVolume_t::GetStruct(), p, output);
    return (EnumInside)output;
  }

  VECCORE_ATT_HOST_DEVICE
  virtual Precision DistanceToIn(Vector3D<Precision> const &p, Vector3D<Precision> const &d,
                                 const Precision step_max = kInfLength) const override
  {
#ifndef VECCORE_CUDA
    assert(d.IsNormalized() && " direction not normalized in call to  DistanceToOut ");
#endif
    Precision output(kInfLength);
    Implementation::DistanceToIn(((UnplacedVolume_t *)this)->UnplacedVolume_t::GetStruct(), p, d, step_max, output);
    return output;
  }

  VECCORE_ATT_HOST_DEVICE
  virtual Precision SafetyToOut(Vector3D<Precision> const &p) const override
  {
    Precision output(kInfLength);
    Implementation::SafetyToOut(((UnplacedVolume_t *)this)->UnplacedVolume_t::GetStruct(), p, output);
    return output;
  }

  VECCORE_ATT_HOST_DEVICE
  virtual Precision SafetyToIn(Vector3D<Precision> const &p) const override
  {
    Precision output(kInfLength);
    Implementation::SafetyToIn(((UnplacedVolume_t *)this)->UnplacedVolume_t::GetStruct(), p, output);
    return output;
  }

  virtual int MemorySize() const override { return sizeof(*this); }
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif /* VOLUMES_UnplacedVolumeImplHelper_H_ */
