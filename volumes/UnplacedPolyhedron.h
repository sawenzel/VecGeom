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

/// \class UnplacedPolyhedron
/// \brief A series of regular n-sided segments along the Z-axis with varying
///        radii and mutual distance in Z.
///
///
/// ---- Cross section of single Z segment ----
///
/// R/Phi--->    -o- Z
/// |        ________________
/// v       /        ^      .\,
///        /    rMax |     .  \,
///       /          |    . <------ fPhiSections[1]
///      /       ____|___.      \,
///     /       /    ^   \       \,
///    /       /     |rMin\       \,
///   /       /      |     \_______\ phiStart/fPhiSections[0]
///   \       \                ^
///    \       \               |
///     \       \________      |
///      \           ^   \<---fZSegments.phi
///      fZSegments.inner \,
///        \               \,
///         \_______________\,
///           ^              phiStart+phiDelta/fPhiSections[n-1]
/// zSegment.outer
///
///
/// ---- Segments along Z ----
///
///                          fZPlanes[size-1]
/// fRMax[1]_____fRMax[2] __       |
///       /|     |\     /|  \___   v
///      / |     | \___/ |  |   |\.
///     |  |     | |   | |  |   | \.
///     |  |     | |   | |  |   |  |
///     |  |     | |___| |  |   | /
///      \ |     | /   \ |  |___|/    ^ R/Phi
///     ^ \|_____|/     \|__/         |
///     |                             |     Z
///     fZPlanes[0]/fRMax[0]           ----->

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

  int fSideCount; ///< Number of segments along phi.
  bool fHasInnerRadii; ///< Has any Z-segments with an inner radius != 0.
  bool fHasPhiCutout; ///< Has a cutout angle along phi.
  bool fHasLargePhiCutout; ///< Phi cutout is larger than pi.
  Array<ZSegment> fZSegments; ///< AOS'esque collections of quadrilaterals
  Array<Precision> fZPlanes; ///< Z-coordinate of each plane separating segments
  // TODO: find a way to re-compute R-min and R-max when converting to another
  //       library's representation to avoid having to store them here.
  Array<Precision> fRMin; ///< Inner radii as specified in constructor.
  Array<Precision> fRMax; ///< Outer radii as specified in constructor.
  SOA3D<Precision> fPhiSections; ///< Unit vectors representing the angle
                                 ///  separating each phi segment, allowing for
                                 ///  quick determination of correct segment
                                 ///  without using trigonometric functions.
  UnplacedTube fBoundingTube; ///< Tube enclosing the outer bounds of the
                              ///  polyhedron. Used in Contains, Inside and
                              ///  DistanceToIn.
  Precision fBoundingTubeOffset; ///< Offset in Z of the center of the bounding
                                 ///  tube. Used as a quick substitution for
                                 ///  running a full transformation.

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

  VECGEOM_CUDA_HEADER_BOTH
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

  /// Not a stored value, and should not be called from performance critical
  /// code.
  /// \return The angle along phi where the first corner is placed, specified in
  ///         degrees.
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetPhiStart() const;

  /// Not a stored value, and should not be called from performance critical
  /// code.
  /// \return The angle along phi where the last corner is placed, specified in
  ///         degrees.
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetPhiEnd() const;

  /// Not a stored value, and should not be called from performance critical
  /// code.
  /// \return The difference in angle along phi between the last corner and the
  ///         first corner.
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
