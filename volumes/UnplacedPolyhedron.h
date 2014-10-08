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
#include "volumes/UnplacedTube.h"

#include <ostream>

namespace VECGEOM_NAMESPACE {

class UnplacedPolyhedron : public VUnplacedVolume, public AlignedBase {

public:

  struct Segment {
    bool hasInnerRadius;
    Quadrilaterals inner, outer;
  };

private:

  // ---- Cross section of single Z segment ----
  //
  // R/Phi--->    -o- Z
  // |        ________________
  // v       /        ^       \ fPhiSections
  //        /    rMax |     /  \.
  //       /          |    o    \.
  //      /       ____|___/      \. 
  //     /       /    ^   \       \.
  //    /       /     |rMin\       \. 
  //   /       /      |     \--o--o-\ fPhiSections
  //   \       \            /       /
  //    \       \          /       /
  //     \       \________/       /
  //      \                      /
  //       \    [Segment.inner] /
  //        \                  / Segment.outer
  //         \________________/
  //
  //
  // ---- Segments along Z ----
  //
  //
  //                            fZBounds[1]/fZPlanes[size-1]
  //  rMax[1]_____rMax[2]  __       |
  //       /|     |\     /|  \___   v
  //      / |     | \___/ |  |   |\.
  //     |  |     | |   | |  |   | \.
  //     |  |     | |   | |  |   |  | fEndCaps
  //     |  |     | |___| |  |   | / 
  //      \ |     | /   \ |  |___|/    ^
  //     ^ \|_____|/     \|__/         | R/Phi
  //     |                         Z   |
  //     zBounds[0]/fZPlanes[0]    <---

  int fSideCount;
  bool fHasInnerRadii;
  Precision fZBounds[2];
  Planes fEndCaps;
  Precision fEndCapsOuterRadii[2];
  Array<Segment> fSegments;
  Array<Precision> fZPlanes;
  Array<Precision> fRMin;
  Array<Precision> fRMax;
  SOA3D<Precision> fPhiSections;
  UnplacedTube fBoundingTube;
  Precision fBoundingTubeOffset;

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
  int GetZSegmentCount() const { return fSegments.size(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  bool HasInnerRadii() const { return fHasInnerRadii; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Segment const& GetZSegment(int i) const { return fSegments[i]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Array<Segment> const& GetZSegments() const { return fSegments; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetZPlane(int i) const { return fZPlanes[i]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Array<Precision> const& GetZPlanes() const { return fZPlanes; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Array<Precision> const& GetRMin() const { return fRMin; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Array<Precision> const& GetRMax() const { return fRMax; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetTolerantZMin() const { return fZBounds[0]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetTolerantZMax() const { return fZBounds[1]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Planes const& GetEndCaps() const { return fEndCaps; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetStartCapOuterRadius() const { return fEndCapsOuterRadii[0]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetEndCapOuterRadius() const { return fEndCapsOuterRadii[1]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  SOA3D<Precision> const& GetPhiSections() const { return fPhiSections; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  UnplacedTube const &GetBoundingTube() const { return fBoundingTube; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetBoundingTubeOffset() const { return fBoundingTubeOffset; }

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
