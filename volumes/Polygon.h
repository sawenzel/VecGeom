/// \file Polygon.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_POLYGON_H_
#define VECGEOM_VOLUMES_POLYGON_H_

#include "base/Global.h"

#include "base/AlignedBase.h"
#include "base/Array.h"
#include "base/CyclicIterator.h"
#include "base/Vector2D.h"

#include <iterator>
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

  /// \return Area of surface covered by the polygon. Is cached after the first
  ///         computation.
  Precision SurfaceArea();

  void ReverseOrder();

  void RemoveVertex(const int index);

  /// \return String representation of the polygon showing nodes in connected
  ///         sequence.
  std::string ToString() const;

  /// Prints to standard output. Does not use streams.
  void Print() const;

  std::ostream& operator<<(std::ostream &os) const;

  typedef CyclicIterator<Vector2D<Precision>, true> const_iterator;

  VECGEOM_INLINE
  const_iterator cbegin() const;

  VECGEOM_INLINE
  const_iterator cend() const;

private:

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

VECGEOM_INLINE
Polygon::const_iterator Polygon::cbegin() const {
  return Polygon::const_iterator(fVertices.cbegin(), fVertices.cend(),
                                 fVertices.cbegin());
}

VECGEOM_INLINE
Polygon::const_iterator Polygon::cend() const {
  return Polygon::const_iterator(fVertices.cbegin(), fVertices.cend(),
                                 fVertices.cend());
}

} // End global namespace

#endif // VECGEOM_VOLUMES_POLYGON_H_