/// \file UnplacedPolyhedron.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "volumes/UnplacedPolyhedron.h"

#include "volumes/Polygon.h"

#include <cmath>

namespace VECGEOM_NAMESPACE {

UnplacedPolyhedron::~UnplacedPolyhedron() {}

#ifndef VECGEOM_NVCC // CPU only constructor

UnplacedPolyhedron::UnplacedPolyhedron(
    const int sideCount, const Precision phiStart, const Precision phiTotal,
    const int zPlaneCount, const Precision zPlane[], const Precision rInner[],
    const Precision rOuter[])
    : fSideCount(sideCount), fPhiStart(phiStart), fPhiTotal(phiTotal) {

  Assert(fSideCount > 0, "Polyhedron requires at least one side.\n");

  // Fix phi parameters

  if (fPhiStart < 0.) {
    fPhiStart += kTwoPi*(1 - static_cast<int>(fPhiStart / kTwoPi));
  }
  if (fPhiStart > kTwoPi) {
    fPhiStart -= kTwoPi*static_cast<int>(fPhiStart / kTwoPi);
  }

  if ((fPhiTotal <= 0.) || (fPhiTotal >= kTwoPi * (1. - kEpsilon))) {
    fPhiTotal = kTwoPi;
    fHasPhi = false;
  } else {
    fPhiTotal = fPhiTotal;
    fHasPhi = true;
  }
  Precision convertRad = 1. / cos(.5 * fPhiTotal / fSideCount);

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

  // Create faces

  Array<PolyhedronSegment*> segments(corners.GetVertixCount());
  PolyhedronSegment **segment = segments.begin();
  for (Polygon::const_iterator c = corners.begin(), cEnd = corners.end();
       c != cEnd; ++c, ++segment) {
    *segment = new PolyhedronSegment(c, fSideCount, fPhiStart, fPhiTotal);
  }
}

UnplacedPolyhedron::PolyhedronSegment::PolyhedronSegment(
    const Polygon::const_iterator corner, const int sideCount,
    const Precision phiStart, const Precision phiTotal)
    : fSideCount(sideCount), fPhiStart(phiStart), fPhiTotal(phiTotal) {

  fHasPhi = phiTotal != kTwoPi;
  fPhiDelta = fPhiTotal / fSideCount;
  fPhiEnd = fPhiStart + fPhiDelta;
  fEdgeCount = (fHasPhi) ? fSideCount+1 : fSideCount;
  fStart = *corner;
  fEnd = *(corner + 1);
  fSides.Allocate(fSideCount);
  fEdges.Allocate(fEdgeCount);

  Vector2D<Precision> cornerPrevious = *(corner - 1);
  Vector2D<Precision> cornerNext = *(corner + 2);

  Precision phi = fPhiStart;
  Vector3D<Precision> a1, b1, c1, d1, a2, b2, c2, d2;

  auto calcCorners = [&] (Vector3D<Precision> &a, Vector3D<Precision> &b,
                          Vector3D<Precision> &c, Vector3D<Precision> &d) {
    a.Set(fStart[0]*cos(phi), fStart[0]*sin(phi), fStart[1]);
    b.Set(fEnd[0]*cos(phi),   fEnd[0]*sin(phi),   fEnd[1]);
    c.Set(cornerPrevious[0]*cos(phi), cornerPrevious[0]*sin(phi),
          cornerPrevious[1]);
    d.Set(cornerNext[0]*cos(phi), cornerNext[0]*sin(phi), cornerNext[1]);
  };

  calcCorners(a1, b1, c1, d1);

  CyclicIterator<PolyhedronEdge, false> edge =
      CyclicIterator<PolyhedronEdge, false>(fEdges.begin(), fEdges.end(),
                                            fEdges.begin());
  for (PolyhedronSide *side = fSides.begin(), *sideEnd = fSides.end();
       side != sideEnd; ++side, ++edge) {

    phi += fPhiDelta;

    Vector3D<Precision> temp, adj;
    calcCorners(a2, b2, c2, d2);

    temp = b2 + b1 - a2 - a1;
    side->center = 0.25 * (a1 + a2 + b1 + b2);
    side->surfRZ = temp.Unit();
    if (side == fSides.begin()) fRZLength = 0.25 * temp.Mag();

    temp = b2 - b1 + a2 - a1;
    side->surfPhi = temp.Unit();
    if (side == fSides.begin()) {
      fPhiLength[0] = 0.25 * temp.Mag();
      temp = b2 - b1;
      fPhiLength[1] = (0.5 * temp.Mag() - fPhiLength[0]) / fRZLength;
    }

    temp = side->surfPhi.Cross(side->surfRZ);
    side->normal = temp.Unit();

    temp = a2 - a1;
    adj = 0.5 * (c1 + c2 - a1 - a2);
    adj = adj.Cross(temp);
    adj = adj.Unit() + side->normal;
    side->edgeNormal[0] = adj.Unit();

    temp = b1 - b2;
    adj = 0.5 * (d1 + d2 - b1 - b2);
    adj = adj.Cross(temp);
    adj = adj.Unit() + side->normal;
    side->edgeNormal[1] = adj.Unit();

    side->edges[0] = &(*edge);
    side->edges[1] = &(*(edge+1));
    edge->corner[0] = a1;
    edge->corner[1] = b1;

    a1 = a2;
    b1 = b2;
    c1 = c2;
    d1 = d2;
  }

  // Last edge
  if (fHasPhi) {
    edge->corner[0] = a2;
    edge->corner[1] = b2;
  } else {
    (fSides.end()-1)->edges[1] = fEdges.begin();
  }

  for (PolyhedronSide *side = fSides.begin(), *prev = side-1,
       *sideEnd = fSides.end(); side != sideEnd; ++side, prev = side-1) {

    PolyhedronEdge *prevEdge = side->edges[0];

    prevEdge->normal = (side->normal + prev->normal).Unit();

    prevEdge->cornerNormal[0] =
        (side->edgeNormal[0] + prev->edgeNormal[0]).Unit();
    prevEdge->cornerNormal[1] = 
        (side->edgeNormal[1] + prev->edgeNormal[1]).Unit();

  }

  if (fHasPhi) {

    PolyhedronSide *side = fSides.begin();

    Vector3D<Precision> normal = side->edges[0]->corner[0]
                                 - side->edges[0]->corner[1];
    normal = normal.Cross(side->normal);
    if (normal.Dot(side->surfPhi) > 0) normal = -normal;

    side->edges[0]->normal = normal.Unit();
    side->edges[0]->cornerNormal[0] = (side->edges[0]->corner[0]
                                       - side->center).Unit();
    side->edges[0]->cornerNormal[1] = (side->edges[0]->corner[1]
                                       - side->center).Unit();

    side = fSides.end();

    normal = side->edges[1]->corner[0] - side->edges[1]->corner[1];
    normal = normal.Cross(side->normal);
    if (normal.Dot(side->surfPhi) < 0) normal = -normal;

    side->edges[1]->normal = normal.Unit();
    side->edges[1]->cornerNormal[0] = (side->edges[1]->corner[0]
                                       - side->center).Unit();
    side->edges[1]->cornerNormal[1] = (side->edges[1]->corner[1]
                                       - side->center).Unit();

  }

  fEdgeNormal = 1. / Sqrt(1. + fPhiLength[1]*fPhiLength[1]);
}

#endif // VECGEOM_NVCC

} // End global namespace