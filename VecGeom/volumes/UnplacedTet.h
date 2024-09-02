// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Declaration of the unplaced tetrahedron shape
/// @file volumes/UnplaceTet.h
/// @author Raman Sehgal, Evgueni Tcherniaev

#ifndef VECGEOM_VOLUMES_UNPLACEDTET_H_
#define VECGEOM_VOLUMES_UNPLACEDTET_H_

#include "VecGeom/base/Cuda.h"
#include "VecGeom/base/Global.h"
#include "VecGeom/base/AlignedBase.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/TetStruct.h" // the pure Tet struct
#include "VecGeom/volumes/kernel/TetImplementation.h"
#include "VecGeom/volumes/UnplacedVolumeImplHelper.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedTet;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedTet);

inline namespace VECGEOM_IMPL_NAMESPACE {

/// Class for tetrahedron shape primitive
///
/// Tetrahedron, also known as a triangular pyramid, is a polyhedron composed
/// by four triangular faces. It is completely defined by its four vertices.
class UnplacedTet : public UnplacedVolumeImplHelper<TetImplementation>, public AlignedBase {

private:
  TetStruct<Precision> fTet; ///< The structure with the tetrahedron data members

public:
  /// Default constructor
  VECCORE_ATT_HOST_DEVICE
  UnplacedTet();

  /// Constructor from four points
  /// @param [in] p0 Point given as 3D vector
  /// @param [in] p1 Point given as 3D vector
  /// @param [in] p2 Point given as 3D vector
  /// @param [in] p3 Point given as 3D vector
  VECCORE_ATT_HOST_DEVICE
  UnplacedTet(const Vector3D<Precision> &p0, const Vector3D<Precision> &p1, const Vector3D<Precision> &p2,
              const Vector3D<Precision> &p3);

  /// Constructor from four points
  /// @param [in] p0 Point given as array
  /// @param [in] p1 Point given as array
  /// @param [in] p2 Point given as array
  /// @param [in] p3 Point given as array
  VECCORE_ATT_HOST_DEVICE
  UnplacedTet(const Precision p0[], const Precision p1[], const Precision p2[], const Precision p3[])
      : fTet(p0, p1, p2, p3)
  {
    fGlobalConvexity = true;
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  virtual ESolidType GetType() const override { return ESolidType::tetrahedron; }

  /// Getter for the structure storing the tetrahedron data
  VECCORE_ATT_HOST_DEVICE
  TetStruct<Precision> const &GetStruct() const { return fTet; }

  /// Getter for the tetrahedron vertices
  /// @param [out] p0 Point given as 3D vector
  /// @param [out] p1 Point given as 3D vector
  /// @param [out] p2 Point given as 3D vector
  /// @param [out] p3 Point given as 3D vector
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void GetVertices(Vector3D<Precision> &p0, Vector3D<Precision> &p1, Vector3D<Precision> &p2,
                   Vector3D<Precision> &p3) const
  {
    p0 = fTet.fVertex[0];
    p1 = fTet.fVertex[1];
    p2 = fTet.fVertex[2];
    p3 = fTet.fVertex[3];
  }

  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &, Vector3D<Precision> &) const override;

  Precision Capacity() const override { return fTet.fCubicVolume; }

  Precision SurfaceArea() const override { return fTet.fSurfaceArea; }

  Vector3D<Precision> SamplePointOnSurface() const override;

  VECCORE_ATT_HOST_DEVICE
  virtual bool Normal(Vector3D<Precision> const &p, Vector3D<Precision> &normal) const override
  {
    bool valid;
    normal = TetImplementation::NormalKernel(fTet, p, valid);
    return valid;
  }

  /// Get the solid type as string
  /// @return Name of the solid type
  std::string GetEntityType() const;

  VECCORE_ATT_HOST_DEVICE
  void GetParametersList(int aNumber, Precision *aArray) const;

  std::ostream &StreamInfo(std::ostream &os) const;

public:
  virtual int MemorySize() const final { return sizeof(*this); }

  VECCORE_ATT_HOST_DEVICE
  virtual void Print() const override;

  virtual void Print(std::ostream &os) const override;

#ifndef VECCORE_CUDA
  virtual SolidMesh *CreateMesh3D(Transformation3D const &trans, size_t nSegments) const override;
#endif

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const override { return DevicePtr<cuda::UnplacedTet>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const override;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const override;
#endif

  /// Templated factory for creating a placed volume
#ifndef VECCORE_CUDA
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
                               VPlacedVolume *const placement = NULL);

  VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume, Transformation3D const *const transformation,
                                   VPlacedVolume *const placement) const override;
#else
  VECCORE_ATT_DEVICE
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
                               const int id, const int copy_no, const int child_id,
                               VPlacedVolume *const placement = NULL);
  VECCORE_ATT_DEVICE VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                                      Transformation3D const *const transformation, const int id,
                                                      const int copy_no, const int child_id,
                                                      VPlacedVolume *const placement) const override;

#endif
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
