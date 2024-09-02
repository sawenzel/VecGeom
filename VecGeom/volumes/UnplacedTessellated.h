// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// \brief Declaration of the data structure for the tessellated shape.
/// \file volumes/UnplacedTessellated.h
/// \author First version created by Mihaela Gheata (CERN/ISS)

#ifndef VECGEOM_VOLUMES_UNPLACEDTESSELLATED_H_
#define VECGEOM_VOLUMES_UNPLACEDTESSELLATED_H_

#include "VecGeom/base/Cuda.h"
#include "VecGeom/base/Global.h"
#include "VecGeom/base/AlignedBase.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "TessellatedStruct.h"
#include "VecGeom/volumes/kernel/TessellatedImplementation.h"
#include "VecGeom/volumes/UnplacedVolumeImplHelper.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedTessellated;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedTessellated);

inline namespace VECGEOM_IMPL_NAMESPACE {

/** Class for tessellated shape primitive.

  The tessellated solid is defined by a closed mesh of triangular facets. To construct facets, two
  methods are provided: AddTriangularFacet and AddQuadrilateralFacet, taking as input three, respectively
  four vertices. Note that quadrilaterals are internally represented as two triangles. The triangles should not
  be malformed (zero inner area), otherwise they will be ignored. It is the user responsibility to create a
  closed mesh, there is no runtime check for this.

  The class uses scalar navigation interfaces, but has internal vectorization on the triangle components and on
  the navigation optimizer.
*/
class UnplacedTessellated : public UnplacedVolumeImplHelper<TessellatedImplementation>, public AlignedBase {
protected:
  mutable TessellatedStruct<3, Precision> fTessellated; ///< Structure with Tessellated parameters

public:
  /// Default constructor for the unplaced tessellated shape class.
  VECCORE_ATT_HOST_DEVICE
  UnplacedTessellated() : fTessellated()
  {
    fGlobalConvexity = false;
    ComputeBBox();
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  virtual ESolidType GetType() const override { return ESolidType::tessellated; }

  /// Getter for the TessellatedStruct object containing the actual data (facets, vertices, clusters of facets)
  /** @return The tessellatedStruct object */
  VECCORE_ATT_HOST_DEVICE
  TessellatedStruct<3, Precision> const &GetStruct() const { return fTessellated; }

  /// Method for adding a new triangular facet, delegated to TessellatedStruct
  /** @param vt0      First vertex
      @param vt1      Second vertex
      @param vt2      Third vertex
      @param absolute If true then vt0, vt1 and vt2 are the vertices to be added in
        anti-clockwise order looking from the outsider. If false the vertices are relative
        to the first: vt0, vt0+vt1, vt0+vt2, in anti-clockwise order when looking from the outsider.
  */
  VECCORE_ATT_HOST_DEVICE
  bool AddTriangularFacet(Vector3D<Precision> const &vt0, Vector3D<Precision> const &vt1,
                          Vector3D<Precision> const &vt2, bool absolute = true)
  {
    bool result = fTessellated.AddTriangularFacet(vt0, vt1, vt2, absolute);
    ComputeBBox();
    return result;
  }

  /// Method for adding a new quadrilateral facet, delegated to TessellatedStruct
  /** @param vt0      First vertex
      @param vt1      Second vertex
      @param vt2      Third vertex
      @param vt3      Fourth vertex
      @param absolute If true then vt0, vt1, vt2 and vt3 are the vertices to be added in
        anti-clockwise order looking from the outsider. If false the vertices are relative
        to the first: vt0, vt0+vt1, vt0+vt2, vt0+vt3 in anti-clockwise order when looking from the
        outsider.
  */
  VECCORE_ATT_HOST_DEVICE
  bool AddQuadrilateralFacet(Vector3D<Precision> const &vt0, Vector3D<Precision> const &vt1,
                             Vector3D<Precision> const &vt2, Vector3D<Precision> const &vt3, bool absolute = true)
  {
    bool result = fTessellated.AddQuadrilateralFacet(vt0, vt1, vt2, vt3, absolute);
    ComputeBBox();
    return result;
  }

  /// Getter for the number of facets.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  size_t GetNFacets() const { return fTessellated.fFacets.size(); }

  /// Getter for a facet with a given index.
  /** @param ifacet Index of the facet
      @return Triangle facet pointer at the given index
  */
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  TriangleFacet<Precision> *GetFacet(int ifacet) const { return fTessellated.fFacets[ifacet]; }

  /// Closing method to be called mandatory by the user once all facets are defined.
  VECCORE_ATT_HOST_DEVICE
  void Close() { fTessellated.Close(); }

  /// Check if the tessellated solid is closed.
  VECCORE_ATT_HOST_DEVICE
  bool IsClosed() const { return fTessellated.fSolidClosed; }

  virtual int memory_size() const { return sizeof(*this); }

  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const override { fTessellated.Extent(aMin, aMax); }

  // Computes capacity of the shape in [length^3]
  // VECCORE_ATT_HOST_DEVICE
  Precision Capacity() const override;

  // VECCORE_ATT_HOST_DEVICE
  Precision SurfaceArea() const override;

  /// Randomly chose a facet with a probability proportional to its surface area. Scales like O(N).
  /** @return Facet index */
  VECCORE_ATT_HOST_DEVICE
  int ChooseSurface() const;

  Vector3D<Precision> SamplePointOnSurface() const override;

  VECCORE_ATT_HOST_DEVICE
  bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const override;

  VECCORE_ATT_HOST_DEVICE
  virtual void Print() const override;

  /// Get the solid type as string.
  /** @return Name of the solid type as string*/
  std::string GetEntityType() const { return "Tessellated"; }

  VECCORE_ATT_DEVICE
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                               const int id,
#endif
                               VPlacedVolume *const placement = NULL);

#ifdef VECGEOM_CUDA_INTERFACE
#ifdef HYBRID_NAVIGATOR_PORTED_TO_CUDA
  virtual size_t DeviceSizeOf() const override { return DevicePtr<cuda::UnplacedTessellated>::SizeOf(); }
#else
  virtual size_t DeviceSizeOf() const override { return 0; }
#endif
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const override;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const override;
#endif

  /// Stream the UnplacedTessellated to an ostream.
  std::ostream &StreamInfo(std::ostream &os) const;

  virtual void Print(std::ostream &os) const override;

private:
  VECCORE_ATT_DEVICE
  virtual VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                           Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                                           const int id,
#endif
                                           VPlacedVolume *const placement = NULL) const override;
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_UNPLACEDTESSELLATED_H_
