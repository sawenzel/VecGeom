/// \file UnplacedPolyhedron.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "volumes/UnplacedPolyhedron.h"

#include "volumes/kernel/GenericKernels.h"
#include "volumes/Polygon.h"
#include "volumes/PlacedPolyhedron.h"
#include "volumes/SpecializedPolyhedron.h"

#include <cmath>

namespace VECGEOM_NAMESPACE {

#ifndef VECGEOM_NVCC

UnplacedPolyhedron::PolyhedronEdges::PolyhedronEdges(int sideCount)
    : normal(sideCount), corner{sideCount, sideCount},
      cornerNormal{sideCount, sideCount} {}

UnplacedPolyhedron::PolyhedronEdges::PolyhedronEdges() : PolyhedronEdges(0) {}

void UnplacedPolyhedron::PolyhedronEdges::Allocate(int sideCount) {
  normal.reserve(sideCount);
  normal.resize(sideCount);
  corner[0].reserve(sideCount);
  corner[0].resize(sideCount);
  corner[1].reserve(sideCount);
  corner[1].resize(sideCount);
  cornerNormal[0].reserve(sideCount);
  cornerNormal[0].resize(sideCount);
  cornerNormal[1].reserve(sideCount);
  cornerNormal[1].resize(sideCount);
}

UnplacedPolyhedron::PolyhedronSegment::PolyhedronSegment(int sideCount)
    : center(sideCount), normal(sideCount), surfPhi(sideCount),
      surfRZ(sideCount), edgeNormal{sideCount, sideCount},
      edge{0, 0}, rZLength(0), phiLength{0, 0}, rZPhiNormal(0) {}

UnplacedPolyhedron::PolyhedronSegment::PolyhedronSegment()
    : PolyhedronSegment(0) {}

void UnplacedPolyhedron::PolyhedronSegment::Allocate(int sideCount) {
  center.reserve(sideCount);
  center.resize(sideCount);
  normal.reserve(sideCount);
  normal.resize(sideCount);
  surfPhi.reserve(sideCount);
  surfPhi.resize(sideCount);
  surfRZ.reserve(sideCount);
  surfRZ.resize(sideCount);
  edgeNormal[0].reserve(sideCount);
  edgeNormal[0].resize(sideCount);
  edgeNormal[1].reserve(sideCount);
  edgeNormal[1].resize(sideCount);
  edge[0].Allocate(sideCount);
  edge[1].Allocate(sideCount);
}

UnplacedPolyhedron::UnplacedPolyhedron(
    const int sideCount, const Precision phiStart, Precision phiTotal,
    const int zPlaneCount, const Precision zPlane[], const Precision rInner[],
    const Precision rOuter[])
    : fSideCount(sideCount) {

  Assert(fSideCount > 0, "Polyhedron requires at least one side.\n");

  // Fix phi parameters to:
  //
  // 0 <= fPhiStart <= 2*pi
  // 0 <  fPhiTotal <= 2*pi
  // fPhiStart > fPhiEnd < 4*pi

  fPhiStart = GenericKernels<kScalar>::NormalizeAngle(phiStart);

  if ((phiTotal <= 0.) || (phiTotal >= kTwoPi * (1. - kEpsilon))) {
    phiTotal = kTwoPi;
    fHasPhi = false;
  } else {
    fHasPhi = true;
  }
  fPhiEnd = fPhiStart + phiTotal;
  fPhiEnd += kTwoPi * (fPhiEnd < fPhiStart);
  fPhiDelta = phiTotal / fSideCount;
  Precision convertRad = 1. / cos(.5 * phiTotal / fSideCount);

  // Check contiguity in segments along Z

  for (int i = 0; i < zPlaneCount-1; ++i) {
    if (zPlane[i] == zPlane[i+1]) {
      Assert(rInner[i] <= rOuter[i+1] && rInner[i+1] <= rOuter[i]);
    }
  }

  // Create corner polygon from segments and verify the outcome

  Polygon corners = Polygon(rInner, rOuter, zPlane, zPlaneCount);
  corners.Scale(convertRad, 1.);

  Assert(corners.GetXMin() >= 0.);

  if (corners.SurfaceArea() < -kTolerance) corners.ReverseOrder();
  Assert(corners.SurfaceArea() >= -kTolerance);

  // Construct segments

  fSegments.Allocate(corners.GetVertixCount());

  Array<PolyhedronSegment>::iterator segment = fSegments.begin();
  for (Polygon::const_iterator corner = corners.cbegin(),
       cornerEnd = corners.cend(); corner != cornerEnd; ++corner, ++segment) {
    ConstructSegment(corner, segment);
  }

  if (fHasPhi) {
    // Not yet implemented. Phi faces need to be created.
  }

}

void UnplacedPolyhedron::ConstructSegment(
    Polygon::const_iterator corner,
    Array<PolyhedronSegment>::iterator segment) {

  segment->Allocate(fSideCount);

  // Segments are constructed as a SOA to allow for internal vectorization.

  Vector2D<Precision> start = *corner;
  Vector2D<Precision> end = *(corner + 1);

  Vector2D<Precision> cornerPrevious = *(corner - 1);
  Vector2D<Precision> cornerNext = *(corner + 2);

  Precision phi = fPhiStart;
  Vector3D<Precision> a1, b1, c1, d1, a2, b2, c2, d2;

  a1.Set(start[0]*cos(phi), start[0]*sin(phi), start[1]);
  b1.Set(end[0]*cos(phi),   end[0]*sin(phi),   end[1]);
  c1.Set(cornerPrevious[0]*cos(phi), cornerPrevious[0]*sin(phi),
         cornerPrevious[1]);
  d1.Set(cornerNext[0]*cos(phi), cornerNext[0]*sin(phi), cornerNext[1]);

  for (int s = 0; s < fSideCount; ++s) {

    phi += fPhiDelta;

    Vector3D<Precision> temp, adj;

    a2.Set(start[0]*cos(phi), start[0]*sin(phi), start[1]);
    b2.Set(end[0]*cos(phi),   end[0]*sin(phi),   end[1]);
    c2.Set(cornerPrevious[0]*cos(phi), cornerPrevious[0]*sin(phi),
           cornerPrevious[1]);
    d2.Set(cornerNext[0]*cos(phi), cornerNext[0]*sin(phi), cornerNext[1]);

    temp = b2 + b1 - a2 - a1;
    segment->center[s].Set(0.25 * (a1 + a2 + b1 + b2));
    segment->surfRZ[s] = temp.Unit();
    if (s == 0) segment->rZLength = 0.25 * temp.Mag();

    temp = b2 - b1 + a2 - a1;
    segment->surfPhi[s] = temp.Unit();
    if (s == 0) {
      segment->phiLength[0] = 0.25 * temp.Mag();
      temp = b2 - b1;
      segment->phiLength[1] = (0.5 * temp.Mag() - segment->phiLength[0])
                            / segment->rZLength;
    }

    temp = segment->surfPhi[s].Cross(segment->surfRZ[s]);
    segment->normal[s] = temp.Unit();

    temp = a2 - a1;
    adj = 0.5 * (c1 + c2 - a1 - a2);
    adj = adj.Cross(temp);
    adj = adj.Unit() + segment->normal[s];
    segment->edgeNormal[0][s] = adj.Unit();

    temp = b1 - b2;
    adj = 0.5 * (d1 + d2 - b1 - b2);
    adj = adj.Cross(temp);
    adj = adj.Unit() + segment->normal[s];
    segment->edgeNormal[1][s] = adj.Unit();

    segment->edge[0].corner[0][s].Set(a1);
    segment->edge[0].corner[1][s].Set(b1);
    segment->edge[1].corner[0][s].Set(a2);
    segment->edge[1].corner[1][s].Set(b2);

    a1 = a2;
    b1 = b2;
    c1 = c2;
    d1 = d2;
  }

  // Last edge
  if (!fHasPhi) {
    segment->edge[0].corner[0][fSideCount-1]
        = segment->edge[0].corner[0][0];
    segment->edge[1].corner[0][fSideCount-1]
        = segment->edge[1].corner[0][0];
  }

  for (int s = 0; s < fSideCount; ++s) {

    int sPrev = (s == 0) ? fSideCount-1 : s-1;

    segment->edge[0].normal[sPrev] = (segment->normal[s]
                                     + segment->normal[sPrev]).Unit();

    segment->edge[0].corner[0][s] = (segment->edgeNormal[0][s]
                                    + segment->edgeNormal[0][sPrev]).Unit();

    segment->edge[1].corner[1][s] = (segment->edgeNormal[1][s]
                                    + segment->edgeNormal[1][sPrev]).Unit();

  }

  if (fHasPhi) {

    Vector3D<Precision> normal;
    normal = segment->edge[0].corner[0][0]
           - segment->edge[0].corner[1][0];
    normal = normal.Cross(segment->normal[0]);
    if (normal.Dot(segment->surfPhi[0]) > 0) normal = -normal;

    segment->edge[0].normal[0] = normal.Unit();
    segment->edge[0].cornerNormal[0][0] =
        (segment->edge[0].corner[0][0] - segment->center[0]).Unit();
    segment->edge[0].cornerNormal[1][0] =
        (segment->edge[0].corner[1][0] - segment->center[0]).Unit();

    int sEnd = fSideCount-1;

    normal = segment->edge[1].corner[0][sEnd] 
           - segment->edge[1].corner[1][sEnd];
    normal = normal.Cross(segment->normal[sEnd]);
    if (normal.Dot(segment->surfPhi[sEnd]) < 0) normal = -normal;

    segment->edge[1].normal[sEnd] = normal.Unit();
    segment->edge[1].cornerNormal[0][sEnd] =
        (segment->edge[1].corner[0][sEnd] - segment->center[sEnd]).Unit();
    segment->edge[1].cornerNormal[1][sEnd] =
        (segment->edge[1].corner[1][sEnd] - segment->center[sEnd]).Unit();

  }

  segment->rZPhiNormal =
      1. / Sqrt(1. + segment->phiLength[1]*segment->phiLength[1]);
}

#endif // VECGEOM_NVCC

UnplacedPolyhedron::~UnplacedPolyhedron() {}

VECGEOM_CUDA_HEADER_BOTH
void UnplacedPolyhedron::Print() const {
  printf("UnplacedPolyhedron {%i sides, %.2f phi start, %.2f phi end}",
         fSideCount, fPhiStart, fPhiEnd);
}

void UnplacedPolyhedron::Print(std::ostream &os) const {
  os << "UnplacedPolyhedron {" << fSideCount << " sides, " << fPhiStart
     << " phi start, " << fPhiEnd << " phi end}";
}

VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedPolyhedron::SpecializedVolume(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation,
    const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) const {

#ifdef VECGEOM_NVCC
  #define UNPLACEDPOLYHEDRON_SPECIALIZE(PHI) \
  if (placement) { \
    return new(placement) \
           SpecializedPolyhedron<PolyhedronSpecialization<PHI> >( \
               volume, transformation, id); \
  } else { \
    return new SpecializedPolyhedron<PolyhedronSpecialization<PHI> >( \
               volume, transformation, id); \
  }
#else
  #define UNPLACEDPOLYHEDRON_SPECIALIZE(PHI) \
  if (placement) { \
    return new(placement) \
           SpecializedPolyhedron<PolyhedronSpecialization<PHI> >( \
               volume, transformation); \
  } else { \
    return new SpecializedPolyhedron<PolyhedronSpecialization<PHI> >( \
               volume, transformation); \
  }
#endif
  if (fHasPhi) {
    UNPLACEDPOLYHEDRON_SPECIALIZE(true)
  } else {
    UNPLACEDPOLYHEDRON_SPECIALIZE(false)
  }
  #undef UNPLACEDPOLYHEDRON_CREATE
}

} // End global namespace