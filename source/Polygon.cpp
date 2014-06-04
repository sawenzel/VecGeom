/// \file Polygon.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "volumes/Polygon.h"

#include <algorithm>
#include <cstring>

namespace VECGEOM_NAMESPACE {

Polygon::Polygon(Precision const x[], Precision const y[], const int size)
    : fSurfaceArea(0.) {

  assert(size > 2 && "Polygon requires at least three vertices.\n");

  fVertixCount = size;
  fX = new Precision[size];
  fY = new Precision[size];

  std::copy(x, x+size, fX);
  std::copy(y, y+size, fY);

  FindLimits();
}

Polygon::Polygon(const Precision rMin[], const Precision rMax[],
                 const Precision z[], const int size) : fSurfaceArea(0.) {

  Assert(fVertixCount > 2, "Polygon requires at least three input vertices.\n");

  fVertixCount = size<<1;

  fX = new Precision[fVertixCount];
  fY = new Precision[fVertixCount];

  std::reverse_copy(rMin, rMin+fVertixCount, fX);
  std::copy(rMax, rMax+fVertixCount, fX+fVertixCount);
  std::reverse_copy(z, z+fVertixCount, fY);
  std::copy(z, z+fVertixCount, fY+fVertixCount);

  RemoveDuplicateVertices();
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

void Polygon::RemoveVertex(const int index) {
  Precision *const xNew = new Precision[fVertixCount+1];
  Precision *const yNew = new Precision[fVertixCount+1];
  std::copy(fX, fX+index+1, xNew);
  std::copy(fX+index, fX+fVertixCount, xNew+index);
  std::copy(fY, fY+index+1, yNew);
  std::copy(fY+index, fY+fVertixCount, yNew+index);
  --fVertixCount;
  assert(fVertixCount > 2 && "Insufficient remaining vertices remaining "
                             "in Polygon.\n");
}

void Polygon::RemoveDuplicateVertices() {
  int i = 0, iNext = 1, iMax = fVertixCount-1;
  while (i < iMax) {
    if (Abs(fX[i] - fX[iNext]) < kTolerance &&
        Abs(fY[i] - fY[iNext]) < kTolerance) {
      RemoveVertex(i);
      --iMax;
    } else {
      ++i;
      ++iNext;
    }
  }
}

void Polygon::RemoveRedundantVertices() {
  // int i = 0, iNext = 1, iMax = fVertixCount-1;
  // while (i < iMax) {
  //   int j = i+2, jNext = j+1, jMax = iMax;
  //   while (j < jMax) {

  //   }
  // }
}

} // End global namespace