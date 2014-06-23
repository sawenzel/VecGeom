/// \file Polygon.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "volumes/Polygon.h"

#include "base/CyclicIterator.h"

#include <algorithm>
#include <ostream>
#include <sstream>

namespace VECGEOM_NAMESPACE {

Polygon::Polygon(Precision const x[], Precision const y[], const int size)
    : fVertices(size), fSurfaceArea(0.) {
  for (int i = 0; i < size; ++i) fVertices[i].Set(x[i], y[i]);
  Initialize();
}

Polygon::Polygon(Vector2D<Precision> const points[], const int size)
    : fVertices(size), fSurfaceArea(0.) {
  copy(points, points+size, &fVertices[0]);
  Initialize();
}

Polygon::Polygon(const Precision rMin[], const Precision rMax[],
                 const Precision z[], const int size)
    : fVertices(size<<1), fSurfaceArea(0.) {

  for (int i = 0, iRev = size-1, iOffset = i+size; i < size;
       ++i, --iRev, ++iOffset) {
    fVertices[i].Set(rMin[iRev], z[iRev]);
    fVertices[iOffset].Set(rMax[i], z[i]);
  }

  Initialize();
}

Polygon::Polygon(const Polygon &other)
    : fVertices(other.fVertices), fXLim(other.fXLim), fYLim(other.fYLim),
      fSurfaceArea(other.fSurfaceArea) {}

void Polygon::Initialize() {
  Assert(fVertices.size() > 2,
         "Polygon requires at least three distinct vertices.\n");
  RemoveRedundantVertices();
  CrossesItself();
  FindLimits();
}

void Polygon::FindLimits() {
  fXLim[0] = fYLim[0] = kInfinity;
  fXLim[1] = fYLim[1] = -kInfinity;
  for (Vector2D<Precision> *i = fVertices.begin(), *iEnd = fVertices.end();
       i != iEnd; ++i) {
    fXLim[0] = (i->x() < fXLim[0]) ? i->x() : fXLim[0];
    fXLim[1] = (i->x() > fXLim[1]) ? i->x() : fXLim[1];
    fYLim[0] = (i->y() < fXLim[0]) ? i->y() : fYLim[0];
    fYLim[1] = (i->y() > fXLim[1]) ? i->y() : fYLim[1];
  }
}

void Polygon::Scale(const double x, const double y) {
  const Vector2D<Precision> scale = Vector2D<Precision>(x, y);
  for (Vector2D<Precision> *i = fVertices.begin(), *iEnd = fVertices.end();
       i != iEnd; ++i) {
    (*i) *= scale;
  }
  fSurfaceArea = 0;
}

Precision Polygon::SurfaceArea() {
  if (!fSurfaceArea) {
    const int iMax = fVertices.size()-1;
    for (int i = 0, iNext = 1; i < iMax; ++i, ++iNext) {
      fSurfaceArea += fVertices[i].Cross(fVertices[iNext]);
    }
    fSurfaceArea = 0.5 * (fSurfaceArea + fVertices[iMax].Cross(fVertices[0]));
  }
  return fSurfaceArea;
}

void Polygon::ReverseOrder() {
  reverse(&fVertices[0], &fVertices[fVertices.size()]);
  fSurfaceArea = 0;
}

std::string Polygon::ToString() const {
  std::stringstream ss;
  ss << fVertices[0];
  for (int i = 1; i < fVertices.size(); ++i) {
    ss << " -- " << fVertices[i];
  }
  return ss.str();
}

std::ostream& Polygon::operator<<(std::ostream &os) const {
  os << ToString();
  return os;
}

void Polygon::Print() const {
  printf("%s\n", ToString().c_str());
}

void Polygon::RemoveVertex(const int index) {
  Array<Vector2D<Precision> > old(fVertices);
  fVertices.Allocate(old.size()-1);
  copy(&old[0], &old[index], &fVertices[0]);
  copy(&old[index+1], &old[old.size()], &fVertices[index]);
  fSurfaceArea = 0;
  Assert(fVertices.size() > 2,
         "Insufficient significant vertices in Polygon.\n");
}

void Polygon::RemoveRedundantVertices() {
  // Will also remove duplicate vertices as lines will be trivially parallel
  int i = 0, iMax = fVertices.size()-2;
  while (i < iMax) {
    // If subsequent lines are parallel, the middle one will be removed
    if (Abs((fVertices[i+2].x() - fVertices[i].x())
            *(fVertices[i+1].y() - fVertices[i].y()) -
            (fVertices[i+1].x() - fVertices[i].x())
            *(fVertices[i+2].y() - fVertices[i].y())) < kTolerance) {
      RemoveVertex(i+1);
      --iMax;
    } else {
      ++i;
    }
  }
}

bool LinesIntersect(
    Vector2D<Precision> const &p, Vector2D<Precision> const &p1,
    Vector2D<Precision> const &q, Vector2D<Precision> const &q1) {
  Vector2D<Precision> r = p1 - p;
  Vector2D<Precision> s = q1 - q;
  Vector2D<Precision> diff = q - p;
  Precision cross = r.Cross(s);
  if (Abs(cross) < kTolerance) return false;
  Precision u = diff.Cross(r) / cross;
  Precision t = diff.Cross(s) / cross;
  return u >= 0 && u <= 1 && t >= 0 && t <= 1;
}

void Polygon::CrossesItself() const {
  static char const *const errorMessage = "Polygon crosses itself.\n";
  Vector2D<Precision> p, p1;
  // Adjacent segments don't need to be checked and segments consist of two
  // points, so loop until end minus three.
  for (Array<Vector2D<Precision> >::const_iterator i = fVertices.cbegin(),
       iEnd = fVertices.cend()-3; i != iEnd; ++i) {
    p = *i;
    p1 = *(i+1);
    // Start from first non-adjacent segment and end at last non-adjacent
    // segment.
    for (Array<Vector2D<Precision> >::const_iterator j = i+2,
         jEnd = fVertices.cend()-1; j != jEnd; ++j) {
      Assert(!LinesIntersect(p, p1, *j, *(j+1)), errorMessage);
    }
  }
  // Segment between start and end
  p = *(fVertices.cend()-1);
  p1 = *(fVertices.cbegin());
  for (Array<Vector2D<Precision> >::const_iterator i = fVertices.cbegin()+1,
       iEnd = fVertices.cend()-2; i != iEnd; ++i) {
    Assert(!LinesIntersect(p, p1, *i, *(i+1)), errorMessage);
  }
}

} // End global namespace