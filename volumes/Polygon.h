/// \file Polygon.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_POLYGON_H_
#define VECGEOM_VOLUMES_POLYGON_H_

#include "base/Global.h"

#include "base/AlignedBase.h"
#include "base/Vector2D.h"

#include <string>

namespace VECGEOM_NAMESPACE {

class Polygon : public AlignedBase {

private:

  int fVertixCount;
  Precision *fX, *fY;
  Vector2D<Precision> fXLim, fYLim;
  Precision fSurfaceArea;

public:

  Polygon(Precision const x[], Precision const y[], const int size);

  Polygon(Vector2D<Precision> const points[], const int size);

  Polygon(Precision const rMin[], Precision const rMax[], Precision const z[],
          const int size);

  Polygon(Polygon const &other);

  void Scale(const Precision x, const Precision y);

  int GetVertixCount() const { return fVertixCount; }

  Precision GetXMin() const { return fXLim[0]; }

  Precision GetXMax() const { return fXLim[1]; }

  Precision GetYMin() const { return fYLim[0]; }

  Precision GetYMax() const { return fYLim[1]; }

  VECGEOM_INLINE
  Vector2D<Precision> GetPoint(const int index) const;

  Precision SurfaceArea();

  void ReverseOrder();

  void RemoveVertex(const int index);

  std::string ToString() const;

  void Print() const;

  std::ostream& operator<<(std::ostream &os) const;

private:

  void Initialize();

  void FindLimits();

  bool AreParallel() const;

  void RemoveRedundantVertices();

  void CrossesItself() const;

};

Vector2D<Precision> Polygon::GetPoint(const int index) const {
  return Vector2D<Precision>(fX[index], fY[index]);
}

} // End global namespace

#endif // VECGEOM_VOLUMES_POLYGON_H_