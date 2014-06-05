/// \file Polygon.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "volumes/Polygon.h"

#include <algorithm>
#include <ostream>
#include <sstream>

namespace VECGEOM_NAMESPACE {

Polygon::Polygon(Precision const x[], Precision const y[], const int size)
    : fSurfaceArea(0.) {

  fVertixCount = size;
  fX = new Precision[size];
  fY = new Precision[size];

  std::copy(x, x+size, fX);
  std::copy(y, y+size, fY);

  Initialize();
}

Polygon::Polygon(Vector2D<Precision> const points[], const int size)
    : fSurfaceArea(0.) {

  fVertixCount = size;
  fX = new Precision[fVertixCount];
  fY = new Precision[fVertixCount];

  for (int i = 0; i < size; ++i) {
    fX[i] = points[i].x();
    fY[i] = points[i].y();
  }

  Initialize();
}

Polygon::Polygon(const Precision rMin[], const Precision rMax[],
                 const Precision z[], const int size) : fSurfaceArea(0.) {

  fVertixCount = size<<1;

  fX = new Precision[fVertixCount];
  fY = new Precision[fVertixCount];

  std::reverse_copy(rMin, rMin+fVertixCount, fX);
  std::copy(rMax, rMax+fVertixCount, fX+fVertixCount);
  std::reverse_copy(z, z+fVertixCount, fY);
  std::copy(z, z+fVertixCount, fY+fVertixCount);

  Initialize();
}

Polygon::Polygon(const Polygon &other)
    : fVertixCount(other.fVertixCount), fXLim(other.fXLim), fYLim(other.fYLim),
      fSurfaceArea(other.fSurfaceArea) {
  fX = new Precision[fVertixCount];
  fY = new Precision[fVertixCount];
  std::copy(other.fX, other.fX+fVertixCount, fX);
  std::copy(other.fY, other.fY+fVertixCount, fY);
}

void Polygon::Initialize() {
  assert(fVertixCount > 2 &&
         "Polygon requires at least three input vertices.\n");
  RemoveRedundantVertices();
  CrossesItself();
  FindLimits();
}

void Polygon::FindLimits() {
  fXLim[0] = *std::min_element(fX, fY+fVertixCount);
  fXLim[1] = *std::max_element(fX, fY+fVertixCount);
  fYLim[0] = *std::min_element(fY, fY+fVertixCount);
  fYLim[1] = *std::max_element(fY, fY+fVertixCount);
}

void Polygon::Scale(const double x, const double y) {
  for (int i = 0; i < fVertixCount; ++i) {
    fX[i] *= x;
    fY[i] *= y;
  }
}

Precision Polygon::SurfaceArea() {
  if (!fSurfaceArea) {
    const int iMax = fVertixCount-1;
    for (int i = 0, iNext = 1; i < iMax; ++i, ++iNext) {
      fSurfaceArea += fX[i] * fY[iNext] - fX[iNext] * fY[i];
    }
    fSurfaceArea = 0.5 * (fSurfaceArea + fX[iMax] * fY[0] - fX[0] * fY[iMax]);
  }
  return fSurfaceArea;
}

void Polygon::ReverseOrder() {
  std::reverse(fX, fX+fVertixCount);
  std::reverse(fY, fY+fVertixCount);
  fSurfaceArea = 0;
}

std::string Polygon::ToString() const {
  std::stringstream ss;
  ss << "(" << fX[0] << ", " << fY[0] << ")";
  for (int i = 1; i < fVertixCount; ++i) {
    ss << " -- (" << fX[i] << ", " << fY[i] << ")";
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
  Precision *const xNew = new Precision[fVertixCount-1];
  Precision *const yNew = new Precision[fVertixCount-1];
  std::copy(fX, fX+index, xNew);
  std::copy(fX+index+1, fX+fVertixCount, xNew+index);
  std::copy(fY, fY+index, yNew);
  std::copy(fY+index+1, fY+fVertixCount, yNew+index);
  delete fX;
  delete fY;
  fX = xNew;
  fY = yNew;
  --fVertixCount;
  fSurfaceArea = 0;
  assert(fVertixCount > 2 && "Insufficient significant vertices in Polygon.\n");
}

void Polygon::RemoveRedundantVertices() {
  // Will also remove duplicate vertices as lines will be trivially parallel
  int i = 0, iMax = fVertixCount-2;
  while (i < iMax) {
    if (Abs((fX[i+1] - fX[i])*(fY[i+2] - fY[i]) -
            (fX[i+2] - fX[i])*(fY[i+1] - fY[i])) < kTolerance) {
      RemoveVertex(i+1);
      --iMax;
    } else {
      ++i;
    }
  }
}

void Polygon::CrossesItself() const {
  const Precision toleranceOne = 1. - kTolerance;
  for (int i = 0; i < fVertixCount-2; ++i) {
    Vector2D<Precision> p0 = GetPoint(i);
    Vector2D<Precision> dP0 = GetPoint(i+1) - p0;
    // Outer loop over each line segment
    for (int j = i+1; j < fVertixCount-1; ++j) {
      Vector2D<Precision> p1 = GetPoint(j);
      Vector2D<Precision> dP1 = GetPoint(j+1) - p1;
      Precision cross = dP1.Cross(dP0);
      Vector2D<Precision> diff = p1 - p0;
      Precision slope0 = Abs(diff.Cross(dP1 / cross));
      Precision slope1 = Abs(diff.Cross(dP0 / cross));
      assert(!(Abs(cross) > kTolerance &&
               (slope0 > kTolerance && slope0 < toleranceOne) &&
               (slope1 > kTolerance && slope1 < toleranceOne))
             && "Polygon crosses itself.\n");
    }
  }
}

} // End global namespace