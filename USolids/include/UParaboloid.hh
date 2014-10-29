//
// ********************************************************************
// * This Software is part of the AIDA Unified Solids Library package *
// * See: https://aidasoft.web.cern.ch/USolids                        *
// ********************************************************************
//
// $Id:$
//
// --------------------------------------------------------------------
//
// UParaboloid
//
// Class description:
//
// 22.09.14 J. de Fine Licht
//          Including VecGeom paraboloid implementation by Marilena
//          Bandieramonte.
// --------------------------------------------------------------------

#ifndef USOLIDS_UParaboloid
#define USOLIDS_UParaboloid

#ifndef VECGEOM
#include "volumes/SpecializedParaboloid.h"
typedef vecgeom::SimpleParaboloid UParaboloid;
#else
#error \
"UParaboloid only exists when USolids is compiled standalone. \
Please use the VecGeom classes instead (e.g. SimpleParaboloid)."
#endif

#endif