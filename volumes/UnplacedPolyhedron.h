/// \file UnplacedPolyhedron.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_UNPLACEDPOLYHEDRON_H_
#define VECGEOM_VOLUMES_UNPLACEDPOLYHEDRON_H_

#include "base/Global.h"

#include "base/AlignedBase.h"
#include "base/Array.h"
#include "base/SOA3D.h"
#include "volumes/Quadrilaterals.h"
#include "volumes/UnplacedVolume.h"

#include <ostream>

namespace VECGEOM_NAMESPACE {

class UnplacedPolyhedron : public VUnplacedVolume, public AlignedBase {

public:

  struct Segment {
    bool hasInnerRadius;
    Precision zMax;
    Quadrilaterals<0> inner, outer;
  };

private:

  int fSideCount;
  bool fHasInnerRadii;
  Precision fZBounds[2];
  Planes<2> fEndCaps;
  Precision fEndCapsOuterRadii[2];
  Array<Segment> fSegments;
  Array<Precision> fZPlanes;
  SOA3D<Precision> fPhiSections;

public:

#ifndef VECGEOM_NVCC
  UnplacedPolyhedron(
      const int sideCount,
      const int zPlaneCount,
      Precision zPlanes[],
      Precision rMin[],
      Precision rMax[]);
#endif

  virtual ~UnplacedPolyhedron() {}

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  int GetSideCount() const { return fSideCount; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  int GetSegmentCount() const { return fSegments.size(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  bool HasInnerRadii() const { return fHasInnerRadii; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Segment const& GetSegment(int index) const { return fSegments[index]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Array<Segment> const& GetSegments() const { return fSegments; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Array<Precision> const& GetZPlanes() const { return fZPlanes; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetTolerantZMin() const { return fZBounds[0]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetTolerantZMax() const { return fZBounds[1]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Planes<2> const& GetEndCaps() const { return fEndCaps; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetStartCapOuterRadius() const { return fEndCapsOuterRadii[0]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetEndCapOuterRadius() const { return fEndCapsOuterRadii[1]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  SOA3D<Precision> const& GetPhiSections() const { return fPhiSections; }

  /// Extract values for the position of Z-planes, and the radius from the
  /// center to the inner and outer shell. These are part of the parameters
  /// included in the original constructor, but which are not stored.
  ///
  /// \param z Output of Z-coordinates of Z-planes. Must be of size >= number of
  ///          segments as returned by GetSegments().
  /// \param rMin Output of R_min-coordinates of Z-planes. Must be of size
  ///             >= number of segments as returned by GetSegments().
  /// \param rMax Output of R_max-coordinates of Z-planes. Must be of size
  ///             >= number of segments as returned by GetSegments().
  void ExtractZPlanes(Precision *z, Precision *rMin, Precision *rMax) const;

  VECGEOM_INLINE
  virtual int memory_size() const { return sizeof(*this); }

  VECGEOM_CUDA_HEADER_BOTH
  virtual void Print() const;

  virtual void Print(std::ostream &os) const;

  VECGEOM_CUDA_HEADER_DEVICE
  VPlacedVolume* SpecializedVolume(
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
      const int id,
#endif
      VPlacedVolume *const placement) const;

#ifdef VECGEOM_CUDA_INTERFACE
  virtual VUnplacedVolume* CopyToGpu() const {
    Assert(0, "NYI");
    return NULL;
  }
  virtual VUnplacedVolume* CopyToGpu(VUnplacedVolume *const gpu_ptr) const {
    Assert(0, "NYI");
    return NULL;
  }
#endif

};

} // End global namespace

#endif // VECGEOM_VOLUMES_UNPLACEDPOLYHEDRON_H_
