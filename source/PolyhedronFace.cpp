/// \file PolyhedronFace.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "volumes/PolyhedronFace.h"

namespace VECGEOM_NAMESPACE {

PolyhedronFace::PolyhedronFace(Polygon::PolygonIterator cornerTail)
    : fSurfaceArea(0.), fPhiCached(0.), fPhiCachedPoint(0.) {

  Polygon::PolygonIterator cornerPrevious = cornerTail - 1;
  Polygon::PolygonIterator cornerHead = cornerTail + 1;
  Polygon::PolygonIterator cornerNext = cornerHead + 1;

  fR.Set(cornerTail->x(), cornerHead->x());
  fZ.Set(cornerTail->y(), cornerHead->y());

}

}; // End global namespace