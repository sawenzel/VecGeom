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

  segment->sides.Allocate(fSideCount);

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
    segment->sides[s].center.Set(0.25 * (a1 + a2 + b1 + b2));
    segment->sides[s].surfRZ = temp.Unit();
    if (s == 0) segment->rZLength = 0.25 * temp.Mag();

    temp = b2 - b1 + a2 - a1;
    segment->sides[s].surfPhi = temp.Unit();
    if (s == 0) {
      segment->phiLength[0] = 0.25 * temp.Mag();
      temp = b2 - b1;
      segment->phiLength[1] = (0.5 * temp.Mag() - segment->phiLength[0])
                            / segment->rZLength;
    }

    temp = segment->sides[s].surfPhi.Cross(segment->sides[s].surfRZ);
    segment->sides[s].normal = temp.Unit();

    temp = a2 - a1;
    adj = 0.5 * (c1 + c2 - a1 - a2);
    adj = adj.Cross(temp);
    adj = adj.Unit() + segment->sides[s].normal;
    segment->sides[s].edgeNormal[0] = adj.Unit();

    temp = b1 - b2;
    adj = 0.5 * (d1 + d2 - b1 - b2);
    adj = adj.Cross(temp);
    adj = adj.Unit() + segment->sides[s].normal;
    segment->sides[s].edgeNormal[1] = adj.Unit();

    segment->sides[s].edges[0].corner[0].Set(a1);
    segment->sides[s].edges[0].corner[1].Set(b1);
    segment->sides[s].edges[1].corner[0].Set(a2);
    segment->sides[s].edges[1].corner[1].Set(b2);

    a1 = a2;
    b1 = b2;
    c1 = c2;
    d1 = d2;
  }

  // Last edge
  if (!fHasPhi) {
    segment->sides[fSideCount-1].edges[0].corner[0]
        = segment->sides[0].edges[0].corner[0];
    segment->sides[fSideCount-1].edges[1].corner[0]
        = segment->sides[0].edges[1].corner[0];
  }

  for (int s = 0; s < fSideCount; ++s) {

    int sPrev = (s == 0) ? fSideCount-1 : s-1;

    segment->sides[sPrev].edges[0].normal =
        (segment->sides[s].normal +
         segment->sides[sPrev].normal).Unit();

    segment->sides[s].edges[0].corner[0] =
        (segment->sides[s].edgeNormal[0] +
         segment->sides[sPrev].edgeNormal[0]).Unit();

    segment->sides[s].edges[1].corner[1] =
        (segment->sides[s].edgeNormal[1] +
         segment->sides[sPrev].edgeNormal[1]).Unit();

  }

  if (fHasPhi) {

    Vector3D<Precision> normal;
    normal = segment->sides[0].edges[0].corner[0]
           - segment->sides[0].edges[0].corner[1];
    normal = normal.Cross(segment->sides[0].normal);
    if (normal.Dot(segment->sides[0].surfPhi) > 0) normal = -normal;

    segment->sides[0].edges[0].normal = normal.Unit();
    segment->sides[0].edges[0].cornerNormal[0] =
        (segment->sides[0].edges[0].corner[0] -
         segment->sides[0].center).Unit();
    segment->sides[0].edges[0].cornerNormal[1] =
        (segment->sides[0].edges[0].corner[1] -
         segment->sides[0].center).Unit();

    int sEnd = fSideCount-1;

    normal = segment->sides[sEnd].edges[1].corner[0] 
           - segment->sides[sEnd].edges[1].corner[1];
    normal = normal.Cross(segment->sides[sEnd].normal);
    if (normal.Dot(segment->sides[sEnd].surfPhi) < 0) normal = -normal;

    segment->sides[sEnd].edges[1].normal = normal.Unit();
    segment->sides[sEnd].edges[1].cornerNormal[0] =
        (segment->sides[sEnd].edges[1].corner[0] -
         segment->sides[sEnd].center).Unit();
    segment->sides[sEnd].edges[1].cornerNormal[1] =
        (segment->sides[sEnd].edges[1].corner[1] -
         segment->sides[sEnd].center).Unit();

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