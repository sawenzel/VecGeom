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

  struct ZSegment {
    Quadrilaterals outer;
    Quadrilaterals phi;
    Quadrilaterals inner;
    bool hasInnerRadius;
#ifdef VECGEOM_CUDA_INTERFACE
    void CopyToGpu(void *gpuptr) const;
#endif
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
  //      \       ZSegment.inner /
  //       \                    /
  //        \                  / ZSegment.outer
  //         \________________/
  //
  //
  // ---- Segments along Z ----
  //
  //
  //                          fZPlanes[size-1]
  // fRMax[1]_____fRMax[2] __       |
  //       /|     |\     /|  \___   v
  //      / |     | \___/ |  |   |\.
  //     |  |     | |   | |  |   | \.
  //     |  |     | |   | |  |   |  |
  //     |  |     | |___| |  |   | /
  //      \ |     | /   \ |  |___|/    ^
  //     ^ \|_____|/     \|__/         | R/Phi
  //     |                         Z   |
  //     fZPlanes[0]               <---

  int fSideCount;
  bool fHasInnerRadii, fHasPhiCutout, fHasLargePhiCutout;
  Array<ZSegment> fZSegments;
  Array<Precision> fZPlanes, fRMin, fRMax;
  SOA3D<Precision> fPhiSections;
  UnplacedTube fBoundingTube;
  Precision fBoundingTubeOffset;

public:

#ifdef VECGEOM_STD_CXX11
  UnplacedPolyhedron(
      const int sideCount,
      const int zPlaneCount,
      Precision zPlanes[],
      Precision rMin[],
      Precision rMax[]);
  UnplacedPolyhedron(
      Precision phiStart,
      Precision phiDelta,
      const int sideCount,
      const int zPlaneCount,
      Precision zPlanes[],
      Precision rMin[],
      Precision rMax[]);
#endif

#ifdef VECGEOM_NVCC
  __device__
  UnplacedPolyhedron(
      int sideCount, bool hasInnerRadii, bool hasPhiCutout,
      bool hasLargePhiCutout, ZSegment *segmentData, Precision *zPlaneData,
      int zPlaneCount, Precision *phiSectionX, Precision *phiSectionY,
      Precision *phiSectionZ, UnplacedTube const &boundingTube,
      Precision boundingTubeOffset);
#endif

  virtual ~UnplacedPolyhedron() {}

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  int GetSideCount() const { return fSideCount; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  int GetZSegmentCount() const { return fZSegments.size(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  bool HasInnerRadii() const { return fHasInnerRadii; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  bool HasPhiCutout() const { return fHasPhiCutout; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  bool HasLargePhiCutout() const { return fHasLargePhiCutout; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  ZSegment const& GetZSegment(int i) const { return fZSegments[i]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Array<ZSegment> const& GetZSegments() const { return fZSegments; }

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
  Vector3D<Precision> GetPhiSection(int i) const { return fPhiSections[i]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  SOA3D<Precision> const& GetPhiSections() const { return fPhiSections; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  UnplacedTube const &GetBoundingTube() const { return fBoundingTube; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetBoundingTubeOffset() const { return fBoundingTubeOffset; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetPhiStart() const;

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetPhiEnd() const;

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetPhiDelta() const;

  VECGEOM_INLINE
  virtual int memory_size() const { return sizeof(*this); }

  VECGEOM_CUDA_HEADER_BOTH
  virtual void Print() const;

  VECGEOM_CUDA_HEADER_BOTH
  void PrintSegments() const;

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
  virtual VUnplacedVolume* CopyToGpu() const;
  virtual VUnplacedVolume* CopyToGpu(VUnplacedVolume *const gpu_ptr) const;
#endif

};

} // End global namespace

#endif // VECGEOM_VOLUMES_UNPLACEDPOLYHEDRON_H_
