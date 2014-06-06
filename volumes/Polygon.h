/// \file Polygon.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_POLYGON_H_
#define VECGEOM_VOLUMES_POLYGON_H_

#include "base/Global.h"

#include "base/AlignedBase.h"
#include "base/Array.h"
#include "base/Vector2D.h"

#include <string>

namespace VECGEOM_NAMESPACE {

class Polygon : public AlignedBase {

private:

  Array<Vector2D<Precision> > fVertices;
  Vector2D<Precision> fXLim, fYLim;
  Precision fSurfaceArea;

public:

  Polygon(Precision const x[], Precision const y[], const int size);

  Polygon(Vector2D<Precision> const points[], const int size);

  Polygon(Precision const rMin[], Precision const rMax[], Precision const z[],
          const int size);

  Polygon(Polygon const &other);

  void Scale(const Precision x, const Precision y);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  int GetVertixCount() const { return fVertices.size(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetXMin() const { return fXLim[0]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetXMax() const { return fXLim[1]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetYMin() const { return fYLim[0]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetYMax() const { return fYLim[1]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector2D<Precision> const& operator[](const int i) const;

  Precision SurfaceArea();

  void ReverseOrder();

  void RemoveVertex(const int index);

  std::string ToString() const;

  void Print() const;

  std::ostream& operator<<(std::ostream &os) const;

  typedef ConstIterator<Vector2D<Precision> > const_iterator;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  const_iterator begin() const { return fVertices.begin(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  const_iterator end() const { return fVertices.end(); }

private:

  typedef Iterator<Vector2D<Precision> > iterator;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  iterator begin() { return fVertices.begin(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  iterator end() { return fVertices.end(); }

  void Initialize();

  void FindLimits();

  bool AreParallel() const;

  void RemoveRedundantVertices();

  void CrossesItself() const;

};

VECGEOM_CUDA_HEADER_BOTH
Vector2D<Precision> const& Polygon::operator[](const int i) const {
  return fVertices[i];
}

} // End global namespace

#endif // VECGEOM_VOLUMES_POLYGON_H_