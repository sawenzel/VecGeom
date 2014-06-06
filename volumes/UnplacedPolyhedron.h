/// \file UnplacedPolyhedron.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_UNPLACEDPOLYHEDRON_H_
#define VECGEOM_VOLUMES_UNPLACEDPOLYHEDRON_H_

#include "base/Global.h"

#include "base/AlignedBase.h"
#include "base/Array.h"
#include "volumes/Face.h"
#include "volumes/Polygon.h"
#include "volumes/UnplacedVolume.h"

namespace VECGEOM_NAMESPACE {

class UnplacedPolyhedron : public VUnplacedVolume, public AlignedBase {

private:

  Array<Face*> fFaces;

public:

  UnplacedPolyhedron(
      Precision phiStart,
      Precision phiDelta,
      int sideCount,
      const int cornerCount,
      const Precision zPlanes[],
      const Precision rInner[],
      const Precision rOuter[]);

  ~UnplacedPolyhedron();

};

} // End global namespace

#endif // VECGEOM_VOLUMES_UNPLACEDPOLYHEDRON_H_