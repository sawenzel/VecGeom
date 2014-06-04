/// \file Polygon.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_POLYGON_H_
#define VECGEOM_VOLUMES_POLYGON_H_

#include "base/Global.h"

#include "base/AlignedBase.h"
#include "base/Vector2D.h"

namespace VECGEOM_NAMESPACE {

class Polygon : public AlignedBase {

private:

  int fVertixCount;
  Precision *fX, *fY;
  Vector2D<Precision> fXLim, fYLim;
  Precision fSurfaceArea;

public:

  Polygon(Precision const x[], Precision const y[], const int size);

  Polygon(Precision const rMin[], Precision const rMax[], Precision const z[],
          const int size);

  void Scale(const Precision x, const Precision y);

  Precision GetXMin() const { return fXLim[0]; }

  Precision GetXMax() const { return fXLim[1]; }

  Precision GetYMin() const { return fYLim[0]; }

  Precision GetYMax() const { return fYLim[1]; }

  Precision SurfaceArea();

  void ReverseOrder();

private:

  void FindLimits();

  void RemoveVertex(const int index);

  void RemoveDuplicateVertices();

  void RemoveRedundantVertices();

};

} // End global namespace

#endif // VECGEOM_VOLUMES_POLYGON_H_