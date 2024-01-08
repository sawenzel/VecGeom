#ifndef VECGEOM_SURFACE_FRAMEMASKS_H
#define VECGEOM_SURFACE_FRAMEMASKS_H

#ifndef SURF_ACCURATE_SAFETY
#define SURF_ACCURATE_SAFETY 0
#endif

#define WINDOW_ACCURATE_SAFETY SURF_ACCURATE_SAFETY
#define QUAD_ACCURATE_SAFETY SURF_ACCURATE_SAFETY

#include <VecGeom/surfaces/mask/WindowMask.h>
#include <VecGeom/surfaces/mask/RingMask.h>
#include <VecGeom/surfaces/mask/ZPhiMask.h>
#include <VecGeom/surfaces/mask/TriangleMask.h>
#include <VecGeom/surfaces/mask/QuadrilateralMask.h>

#endif
