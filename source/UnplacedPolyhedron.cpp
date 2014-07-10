/// \file UnplacedPolyhedron.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "volumes/UnplacedPolyhedron.h"

#include "volumes/kernel/GenericKernels.h"
#include "volumes/Polygon.h"

#include <cmath>

namespace VECGEOM_NAMESPACE {

#ifndef VECGEOM_NVCC

UnplacedPolyhedron::PolyhedronEdges::PolyhedronEdges(int edgeCount)
    : normal(edgeCount), corner{edgeCount, edgeCount},
      cornerNormal{edgeCount, edgeCount} {}

UnplacedPolyhedron::PolyhedronEdges::PolyhedronEdges() : PolyhedronEdges(0) {}

UnplacedPolyhedron::PolyhedronSides::PolyhedronSides(int sideCount)
    : center(sideCount), normal(sideCount), surfPhi(sideCount),
      surfRZ(sideCount), edgesNormal{sideCount, sideCount},
      edges{0, 0}, rZLength(0), phiLength{0, 0}, edgeNormal(0) {}

UnplacedPolyhedron::PolyhedronSides::PolyhedronSides() : PolyhedronSides(0) {}

UnplacedPolyhedron::UnplacedPolyhedron(
    const int sideCount, const Precision phiStart, Precision phiTotal,
    const int zPlaneCount, const Precision zPlane[], const Precision rInner[],
    const Precision rOuter[])
    : fSideCount(sideCount), fPhiStart(phiStart) {

  Assert(fSideCount > 0, "Polyhedron requires at least one side.\n");

  // Fix phi parameters to:
  //
  // 0 <= fPhiStart <= 2*pi
  // 0 <  fPhiTotal <= 2*pi
  // fPhiStart > fPhiEnd < 4*pi

  fPhiStart = GenericKernels<kScalar>::NormalizeAngle(fPhiStart);

  if ((phiTotal <= 0.) || (phiTotal >= kTwoPi * (1. - kEpsilon))) {
    phiTotal = kTwoPi;
    fHasPhi = false;
  } else {
    fHasPhi = true;
  }
  fPhiEnd = fPhiStart + phiTotal;
  fPhiEnd += kTwoPi * (fPhiEnd < fPhiStart);
  Precision convertRad = 1. / cos(.5 * phiTotal / fSideCount);

  fEdgeCount = fSideCount + fHasPhi;

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

  Array<PolyhedronSides>::iterator segment = fSegments.begin();
  for (Polygon::const_iterator c = corners.cbegin(), cEnd = corners.cend();
       c != cEnd; ++c, ++segment) {
    ConstructSegment(c, segment);
  }

  if (fHasPhi) {
    // Not yet implemented. Phi faces need to be created.
  }

}

void UnplacedPolyhedron::ConstructSegment(
    Polygon::const_iterator corner,
    Array<PolyhedronSides>::iterator segment) {

  // Segments are constructed as a SOA to allow for internal vectorization.

  segment->edgesNormal[0] = SOA3D<Precision>(fSideCount);
  segment->edgesNormal[1] = SOA3D<Precision>(fSideCount);
  segment->edges[0] = PolyhedronEdges(fEdgeCount);
  segment->edges[1] = PolyhedronEdges(fEdgeCount);

  Precision phiTotal = fPhiEnd - fPhiStart;
  Precision phiDelta = phiTotal / fSideCount;
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

    phi += phiDelta;

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
    segment->edgesNormal[0][s] = adj.Unit();

    temp = b1 - b2;
    adj = 0.5 * (d1 + d2 - b1 - b2);
    adj = adj.Cross(temp);
    adj = adj.Unit() + segment->normal[s];
    segment->edgesNormal[1][s] = adj.Unit();

    segment->edges[0].corner[0][s].Set(a1);
    segment->edges[0].corner[1][s].Set(b1);
    segment->edges[1].corner[0][s].Set(a2);
    segment->edges[1].corner[1][s].Set(b2);

    a1 = a2;
    b1 = b2;
    c1 = c2;
    d1 = d2;
  }

  // Last edge
  if (!fHasPhi) {
    segment->edges[0].corner[0][fSideCount-1] = segment->edges[0].corner[0][0];
    segment->edges[1].corner[0][fSideCount-1] = segment->edges[1].corner[0][0];
  }

  for (int s = 0; s < fSideCount; ++s) {

    int sPrev = (s == 0) ? fSideCount-1 : s-1;

    segment->edges[0].normal[sPrev] = (segment->normal[s]
                                     + segment->normal[sPrev]).Unit();

    segment->edges[0].corner[0][s] = (segment->edgesNormal[0][s]
                                    + segment->edgesNormal[0][sPrev]).Unit();

    segment->edges[1].corner[1][s] = (segment->edgesNormal[1][s]
                                    + segment->edgesNormal[1][sPrev]).Unit();

  }

  if (fHasPhi) {

    Vector3D<Precision> normal;
    normal = segment->edges[0].corner[0][0]
           - segment->edges[0].corner[1][0];
    normal = normal.Cross(segment->normal[0]);
    if (normal.Dot(segment->surfPhi[0]) > 0) normal = -normal;

    segment->edges[0].normal[0] = normal.Unit();
    segment->edges[0].cornerNormal[0][0] =
        (segment->edges[0].corner[0][0] - segment->center[0]).Unit();
    segment->edges[0].cornerNormal[1][0] =
        (segment->edges[0].corner[1][0] - segment->center[0]).Unit();

    int sEnd = fSideCount-1;

    normal = segment->edges[1].corner[0][sEnd] 
           - segment->edges[1].corner[1][sEnd];
    normal = normal.Cross(segment->normal[sEnd]);
    if (normal.Dot(segment->surfPhi[sEnd]) < 0) normal = -normal;

    segment->edges[1].normal[sEnd] = normal.Unit();
    segment->edges[1].cornerNormal[0][sEnd] =
        (segment->edges[1].corner[0][sEnd] - segment->center[sEnd]).Unit();
    segment->edges[1].cornerNormal[1][sEnd] =
        (segment->edges[1].corner[1][sEnd] - segment->center[sEnd]).Unit();

  }

  segment->edgeNormal =
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
  os << "UnplacedPolyhedron {" << fSideCount << ", " << fPhiStart
     << " phi start, " << fPhiEnd << " phi end}";
}

} // End global namespace