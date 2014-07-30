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
  bool fValid;

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

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision SurfaceArea() const { return fSurfaceArea; }

  /// A polygon is considered valid when it has >2 vertices.
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  bool IsValid() const { return fValid; }

  void ReverseOrder();

  /// \return Whether the resulting polygon is still valid.
  bool RemoveVertex(const int index);

  /// \return Whether the resulting polygon is still valid.
  bool RemoveRedundantVertices();

  /// \return String representation of the polygon showing nodes in connected
  ///         sequence.
  std::string ToString() const;

  /// Prints to standard output. Does not use streams.
  void Print() const;

  std::ostream& operator<<(std::ostream &os) const;

  typedef Array<Vector2D<Precision> >::const_iterator const_iterator;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  const_iterator cbegin() const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  const_iterator cend() const;

private:

  void Initialize();

  void ComputeLimits();

  void ComputeSurfaceArea();

  bool AreParallel() const;

  void CrossesItself() const;

};

VECGEOM_CUDA_HEADER_BOTH
Vector2D<Precision> const& Polygon::operator[](const int i) const {
  return fVertices[i];
}

VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
Polygon::const_iterator Polygon::cbegin() const {
  return fVertices.cbegin();
}

VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
Polygon::const_iterator Polygon::cend() const {
  return fVertices.cend();
}

} // End global namespace

#endif // VECGEOM_VOLUMES_POLYGON_H_