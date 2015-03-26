/// @file UnplacedTrd.h
/// @author Georgios Bitzes (georgios.bitzes@cern.ch)

#ifndef VECGEOM_VOLUMES_UNPLACEDTRD_H_
#define VECGEOM_VOLUMES_UNPLACEDTRD_H_

#include "base/Global.h"
#include "base/AlignedBase.h"
#include "volumes/UnplacedVolume.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( class UnplacedTrd; )
VECGEOM_DEVICE_DECLARE_CONV( UnplacedTrd );

inline namespace VECGEOM_IMPL_NAMESPACE {

class UnplacedTrd : public VUnplacedVolume, public AlignedBase {
private:
  // trd defining parameters
  Precision fDX1;   //Half-length along x at the surface positioned at -dz
  Precision fDX2;   //Half-length along x at the surface positioned at +dz
  Precision fDY1;   //Half-length along y at the surface positioned at -dz
  Precision fDY2;   //Half-length along y at the surface positioned at +dz
  Precision fDZ;    //Half-length along z axis

  // cached values
  Precision fX2minusX1;
  Precision fY2minusY1;
  Precision fDZtimes2;
  Precision fHalfX1plusX2;
  Precision fHalfY1plusY2;
  Precision fCalfX, fCalfY;

  Precision fFx, fFy;

  VECGEOM_CUDA_HEADER_BOTH
  void calculateCached() {
    fX2minusX1 = fDX2 - fDX1;
    fY2minusY1 = fDY2 - fDY1;
    fHalfX1plusX2 = 0.5 * (fDX1 + fDX2);
    fHalfY1plusY2 = 0.5 * (fDY1 + fDY2);

    fDZtimes2 = fDZ * 2;

    fFx = 0.5*(fDX1 - fDX2)/fDZ;
    fFy = 0.5*(fDY1 - fDY2)/fDZ;

    fCalfX = 1./Sqrt(1.0+fFx*fFx);
    fCalfY = 1./Sqrt(1.0+fFy*fFy);
  }

public:
  // special case Trd1 when dY1 == dY2
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedTrd(const Precision dx1, const Precision dx2, const Precision dy1, const Precision dz) :
  fDX1(dx1), fDX2(dx2), fDY1(dy1), fDY2(dy1), fDZ(dz),
fX2minusX1(0),
fY2minusY1(0),
fDZtimes2(0),
fHalfX1plusX2(0),
fHalfY1plusY2(0),
fCalfX(0),
fCalfY(0),
fFx(0),
fFy(0)
{
    calculateCached();
  }

  // general case
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedTrd(const Precision dx1, const Precision dx2, const Precision dy1, const Precision dy2, const Precision dz) :
  fDX1(dx1), fDX2(dx2), fDY1(dy1), fDY2(dy2), fDZ(dz),
fX2minusX1(0),
fY2minusY1(0),
fDZtimes2(0),
fHalfX1plusX2(0),
fHalfY1plusY2(0),
fCalfX(0),
fCalfY(0),
fFx(0),
fFy(0)
 {
    calculateCached();
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision dx1() const { return fDX1; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision dx2() const { return fDX2; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision dy1() const { return fDY1; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision dy2() const { return fDY2; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision dz() const { return fDZ; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision x2minusx1() const { return fX2minusX1; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision y2minusy1() const { return fY2minusY1; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision halfx1plusx2() const { return fHalfX1plusX2; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision halfy1plusy2() const { return fHalfY1plusY2; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision dztimes2() const { return fDZtimes2; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision fx() const { return fFx; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision fy() const { return fFy; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision calfx() const { return fCalfX; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision calfy() const { return fCalfY; }

  virtual int memory_size() const { return sizeof(*this); }

#ifndef VECGEOM_NVCC
  // Computes capacity of the shape in [length^3]
  Precision Capacity() const;
#endif

  VECGEOM_CUDA_HEADER_BOTH
  virtual void Print() const;

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  VECGEOM_CUDA_HEADER_DEVICE
  static VPlacedVolume* Create(LogicalVolume const *const logical_volume,
                               Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
                               const int id,
#endif
                               VPlacedVolume *const placement = NULL);

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const { return DevicePtr<cuda::UnplacedTrd>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const;
#endif

private:

  virtual void Print(std::ostream &os) const;

  VECGEOM_CUDA_HEADER_DEVICE
  virtual VPlacedVolume* SpecializedVolume(
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
      const int id,
#endif
      VPlacedVolume *const placement = NULL) const;
};

} } // end global namespace

#endif // VECGEOM_VOLUMES_UNPLACEDTRD_H_
