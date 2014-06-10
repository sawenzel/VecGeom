/// \file Polygon.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_POLYGON_H_
#define VECGEOM_VOLUMES_POLYGON_H_

#include "base/Global.h"

#include "base/AlignedBase.h"
#include "base/Array.h"
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

  /// \brief Iterator that is cyclic when plus and minus operators are applied.
  ///        This does not apply for incrementation operators, which will
  ///        work as usual. This is to allow proper iteration through the
  ///        seqience.
  class PolygonIterator
      : std::iterator<std::forward_iterator_tag, Vector2D<Precision> > {

  private:

    Array<Vector2D<Precision> > const &fCorners;
    Array<Vector2D<Precision> >::const_iterator fTarget;

  public:

    VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    PolygonIterator(Array<Vector2D<Precision> > const &corners,
                    const Array<Vector2D<Precision> >::const_iterator corner);

    VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    PolygonIterator(PolygonIterator const &other);

    VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    PolygonIterator& operator=(PolygonIterator const &rhs);

    VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    PolygonIterator& operator++();

    VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    PolygonIterator operator++(int);

    VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    PolygonIterator& operator--();

    VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    PolygonIterator operator--(int);

    VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    PolygonIterator operator+(int val);

    VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    PolygonIterator operator-(int val);

    VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    Vector2D<Precision> const& operator*() const;

    VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    Vector2D<Precision> const* operator->() const;

    VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    bool operator==(PolygonIterator const &other) const;

    VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    bool operator!=(PolygonIterator const &other) const;

  }; // End class PolygonIterator

  typedef PolygonIterator const_iterator;

  VECGEOM_INLINE
  PolygonIterator begin() const;

  VECGEOM_INLINE
  PolygonIterator end() const;

private:

  typedef Array<Vector2D<Precision> >::iterator iterator;

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

VECGEOM_INLINE
Polygon::PolygonIterator Polygon::begin() const {
  return Polygon::PolygonIterator(fVertices, fVertices.begin());
}

VECGEOM_INLINE
Polygon::PolygonIterator Polygon::end() const {
  return Polygon::PolygonIterator(fVertices, fVertices.end());
}

VECGEOM_CUDA_HEADER_BOTH
Polygon::PolygonIterator::PolygonIterator(
    Array<Vector2D<Precision> > const &corners,
    const Array<Vector2D<Precision> >::const_iterator corner)
    : fCorners(corners), fTarget(corner) {}

VECGEOM_CUDA_HEADER_BOTH
Polygon::PolygonIterator::PolygonIterator(
    Polygon::PolygonIterator const &other)
    : fCorners(other.fCorners), fTarget(other.fTarget) {}

VECGEOM_CUDA_HEADER_BOTH
Polygon::PolygonIterator& Polygon::PolygonIterator::operator=(
    Polygon::PolygonIterator const &rhs) {
  fTarget = rhs.fTarget;
  return *this;
}

Polygon::PolygonIterator& Polygon::PolygonIterator::operator++() {
  ++fTarget;
  return *this;
}
VECGEOM_CUDA_HEADER_BOTH
Polygon::PolygonIterator Polygon::PolygonIterator::operator++(int) {
  PolygonIterator temp(*this);
  ++(*this);
  return temp;
}

VECGEOM_CUDA_HEADER_BOTH
Polygon::PolygonIterator& Polygon::PolygonIterator::operator--() {
  --fTarget;
  return *this;
}

VECGEOM_CUDA_HEADER_BOTH
Polygon::PolygonIterator Polygon::PolygonIterator::operator--(int) {
  PolygonIterator temp(*this);
  --(*this);
  return temp;
}

VECGEOM_CUDA_HEADER_BOTH
Polygon::PolygonIterator Polygon::PolygonIterator::operator+(int val) {
  return Polygon::PolygonIterator(fCorners,
             fCorners.begin() +
             ((fCorners.end()-fTarget) + val) % fCorners.size());
}

VECGEOM_CUDA_HEADER_BOTH
Polygon::PolygonIterator Polygon::PolygonIterator::operator-(int val) {
  return Polygon::PolygonIterator(fCorners,
             fCorners.begin() +
             ((fCorners.end()-fTarget) - val) % fCorners.size());
}

VECGEOM_CUDA_HEADER_BOTH
Vector2D<Precision> const& Polygon::PolygonIterator::operator*() const {
  return *fTarget;
}

VECGEOM_CUDA_HEADER_BOTH
Vector2D<Precision> const* Polygon::PolygonIterator::operator->() const {
  return fTarget;
}

VECGEOM_CUDA_HEADER_BOTH
bool Polygon::PolygonIterator::operator==(
    Polygon::PolygonIterator const &other) const {
  return fTarget == other.fTarget;
}

VECGEOM_CUDA_HEADER_BOTH
bool Polygon::PolygonIterator::operator!=(
    Polygon::PolygonIterator const &other) const {
  return fTarget != other.fTarget;
}

} // End global namespace

#endif // VECGEOM_VOLUMES_POLYGON_H_