/// \file Polygon.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "volumes/Polygon.h"

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
         "Polygon requires at least three input vertices.\n");
  RemoveRedundantVertices();
  CrossesItself();
  FindLimits();
}

void Polygon::FindLimits() {
  fXLim[0] = fYLim[0] = kInfinity;
  fXLim[1] = fYLim[1] = -kInfinity;
  for (iterator i = begin(), iEnd = end(); i != iEnd; ++i) {
    fXLim[0] = (i->x() < fXLim[0]) ? i->x() : fXLim[0];
    fXLim[1] = (i->x() > fXLim[1]) ? i->x() : fXLim[1];
    fYLim[0] = (i->y() < fXLim[0]) ? i->y() : fYLim[0];
    fYLim[1] = (i->y() > fXLim[1]) ? i->y() : fYLim[1];
  }
}

void Polygon::Scale(const double x, const double y) {
  const Vector2D<Precision> scale = Vector2D<Precision>(x, y);
  for (iterator i = begin(), iEnd = end(); i != iEnd; ++i) {
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
    if (Abs((fVertices[i+1].x() - fVertices[i].x())
            *(fVertices[i+2].y() - fVertices[i].y()) -
            (fVertices[i+2].x() - fVertices[i].x())
            *(fVertices[i+1].y() - fVertices[i].y()))) {
      RemoveVertex(i+1);
      --iMax;
    } else {
      ++i;
    }
  }
}

void Polygon::CrossesItself() const {
  const Precision toleranceOne = 1. - kTolerance;
  for (int i = 0, iMax = fVertices.size()-2; i < iMax; ++i) {
    Vector2D<Precision> p0 = fVertices[i];
    Vector2D<Precision> dP0 = fVertices[i+1] - p0;
    // Outer loop over each line segment
    for (int j = i+1, jMax = fVertices.size()-1; j < jMax; ++j) {
      Vector2D<Precision> p1 = fVertices[j];
      Vector2D<Precision> dP1 = fVertices[j+1] - p1;
      Precision cross = dP1.Cross(dP0);
      Vector2D<Precision> diff = p1 - p0;
      Precision slope0 = Abs(diff.Cross(dP1 / cross));
      Precision slope1 = Abs(diff.Cross(dP0 / cross));
      Assert(!(Abs(cross) > kTolerance &&
               (slope0 > kTolerance && slope0 < toleranceOne) &&
               (slope1 > kTolerance && slope1 < toleranceOne)),
             "Polygon crosses itself.\n");
    }
  }
}

} // End global namespace