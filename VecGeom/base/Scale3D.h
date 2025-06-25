/// \file Scale3D.h
/// \author Mihaela Gheata (mihaela.gheata@cern.ch)

#ifndef VECGEOM_BASE_SCALE3D_H_
#define VECGEOM_BASE_SCALE3D_H_

#include "VecGeom/base/Assert.h"
#include "VecGeom/base/Cuda.h"
#include "VecGeom/base/Global.h"

#include "VecGeom/base/Vector3D.h"
#ifdef VECGEOM_CUDA_INTERFACE
#include "VecGeom/backend/cuda/Interface.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstring>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class Scale3D;);

inline namespace VECGEOM_IMPL_NAMESPACE {

#ifndef VECCORE_CUDA
}
namespace cuda {
class Scale3D;
}
inline namespace VECGEOM_IMPL_NAMESPACE {
// class vecgeom::cuda::Scale3D;
#endif

class Scale3D {

private:
  Vector3D<Precision> fScale;    /// scale transformation
  Vector3D<Precision> fInvScale; /// inverse scale (avoid divisions)
  Precision fSclLocal;           /// factor to apply to safety to convert to local frame
  Precision fSclMaster;          /// factor to apply to safety to convert to master frame

public:
  /**
   * Default constructor
   */
  VECCORE_ATT_HOST_DEVICE
  Scale3D() : fScale(1., 1., 1.), fInvScale(1., 1., 1.), fSclLocal(1.), fSclMaster(1.) {}

  /**
   * Constructor with scale parameters on each axis
   * @param sx Scale value on x
   * @param sy Scale value on y
   * @param sz Scale value on z
   */
  VECCORE_ATT_HOST_DEVICE
  Scale3D(Precision sx, Precision sy, Precision sz) : fScale(sx, sy, sz), fInvScale(), fSclLocal(1.), fSclMaster(1.)
  {
    Update();
  }

  /**
   * Constructor with scale parameters in a Vector3D
   * @param scale Scale as Vector3D
   */
  VECCORE_ATT_HOST_DEVICE
  Scale3D(Vector3D<Precision> const &scale) : fScale(scale), fInvScale(), fSclLocal(1.), fSclMaster(1.) { Update(); }

  /**
   * Copy constructor.
   */
  VECCORE_ATT_HOST_DEVICE
  Scale3D(Scale3D const &other)
      : fScale(other.fScale), fInvScale(other.fInvScale), fSclLocal(other.fSclLocal), fSclMaster(other.fSclMaster)
  {
  }

  /**
   * Assignment operator
   */
  VECCORE_ATT_HOST_DEVICE
  Scale3D &operator=(Scale3D const &other)
  {
    fScale     = other.fScale;
    fInvScale  = other.fInvScale;
    fSclLocal  = other.fSclLocal;
    fSclMaster = other.fSclMaster;
    return *this;
  }

  /**
   * Update the backed-up inverse scale and special conversion factors based
   * on the values of the scale. Needed whenever the scale has changed value.
   */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void Update()
  {
    VECGEOM_ASSERT(((fScale[0] != 0) && (fScale[1] != 0) && (fScale[2] != 0)));
    fInvScale.Set(1. / fScale[0], 1. / fScale[1], 1. / fScale[2]);
    // Keep into account that scale components may be negative for reflections
    fSclLocal  = Min(Abs(fInvScale[0]), Abs(fInvScale[1]));
    fSclLocal  = Min(fSclLocal, Abs(fInvScale[2]));
    fSclMaster = Min(Abs(fScale[0]), Abs(fScale[1]));
    fSclMaster = Min(fSclMaster, Abs(fScale[2]));
  }

  /**
   * Get reference to the scale vector.
   */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  const Vector3D<Precision> &Scale() const { return fScale; }

  /**
   * Get reference to the inverse scale vector.
   */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  const Vector3D<Precision> &InvScale() const { return fInvScale; }

  /**
   * Set scale based on vector.
   */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetScale(Vector3D<Precision> const &scale)
  {
    fScale = scale;
    Update();
  }

  /**
   * Set scale based on values.
   */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetScale(Precision sx, Precision sy, Precision sz)
  {
    fScale.Set(sx, sy, sz);
    Update();
  }

  /**
   * Transform point from master to local frame
   */
  template <typename InputType>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void Transform(Vector3D<InputType> const &master,
                                                              Vector3D<InputType> &local) const
  {
    local.Set(master[0] * fInvScale[0], master[1] * fInvScale[1], master[2] * fInvScale[2]);
  }

  template <typename InputType>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector3D<InputType> Transform(Vector3D<InputType> const &master) const
  {
    Vector3D<InputType> local(master[0] * fInvScale[0], master[1] * fInvScale[1], master[2] * fInvScale[2]);
    return local;
  }

  /**
   * Transform point from local to master frame
   */
  template <typename InputType>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void InverseTransform(Vector3D<InputType> const &local,
                                                                     Vector3D<InputType> &master) const
  {
    master.Set(local[0] * fScale[0], local[1] * fScale[1], local[2] * fScale[2]);
  }

  template <typename InputType>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector3D<InputType> InverseTransform(
      Vector3D<InputType> const &local) const
  {
    Vector3D<InputType> master(local[0] * fScale[0], local[1] * fScale[1], local[2] * fScale[2]);
    return master;
  }

  /**
   * Transform normal from master to local frame
   */
  template <typename InputType>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void TransformNormal(Vector3D<InputType> const &master,
                                                                    Vector3D<InputType> &local) const
  {
    local.Set(master[0] * fInvScale[0], master[1] * fInvScale[1], master[2] * fInvScale[2]);
    local.Normalize();
  }

  template <typename InputType>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector3D<InputType> TransformNormal(
      Vector3D<InputType> const &master) const
  {
    Vector3D<InputType> local(master[0] * fInvScale[0], master[1] * fInvScale[1], master[2] * fInvScale[2]);
    local.Normalize();
    return local;
  }

  /**
   * Transform normal from local to master frame
   */
  template <typename InputType>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void InverseTransformNormal(Vector3D<InputType> const &local,
                                                                           Vector3D<InputType> &master) const
  {
    master.Set(local[0] * fScale[0], local[1] * fScale[1], local[2] * fScale[2]);
    master.Normalize();
  }

  template <typename InputType>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector3D<InputType> InverseTransformNormal(
      Vector3D<InputType> const &local) const
  {
    Vector3D<InputType> master(local[0] * fScale[0], local[1] * fScale[1], local[2] * fScale[2]);
    master.Normalize();
    return master;
  }

  /**
   * Transform distance along given direction from master to local frame
   */
  template <typename InputType>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE InputType TransformDistance(InputType const &dist,
                                                                           Vector3D<InputType> const &dir) const
  {
    Vector3D<InputType> v = dir * fInvScale;
    InputType scale       = Sqrt(Vector3D<InputType>::Dot(v, v));
    return (scale * dist);
  }

  /**
   * Transform safe distance from master to local frame (conservative)
   */
  template <typename InputType>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE InputType TransformSafety(InputType safety) const
  {
    return (safety * fSclLocal);
  }

  /**
   * Transform distance along given direction from local to master frame
   */
  template <typename InputType>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE InputType InverseTransformDistance(InputType const &dist,
                                                                                  Vector3D<InputType> const &dir) const
  {
    Vector3D<InputType> v = dir * fScale;
    InputType scale       = Sqrt(Vector3D<InputType>::Dot(v, v));
    return (scale * dist);
  }

  /**
   * Transform safe distance from local to master frame (conservative)
   */
  template <typename InputType>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE InputType InverseTransformSafety(InputType safety) const
  {
    return (safety * fSclMaster);
  }

public:
  static const Scale3D kIdentity;

}; // End class Scale3D

std::ostream &operator<<(std::ostream &os, Scale3D const &scale);
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_BASE_SCALE3D_H_
