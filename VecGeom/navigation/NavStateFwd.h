/// \file NavStateFwd.h
/// \author Andrei Gheata (andrei.gheata@cern.ch)
/// \date 12.03.2014

#ifndef VECGEOM_NAVIGATION_NAVSTATEFWD_H_
#define VECGEOM_NAVIGATION_NAVSTATEFWD_H_

#include "VecGeom/base/Cuda.h"
#include "VecGeom/base/Config.h"

#ifdef VECGEOM_USE_NAVTUPLE
#define NavigationStateImpl NavStateTuple
#else
#define NavigationStateImpl NavStateIndex
#endif
namespace vecgeom {

VECGEOM_HOST_FORWARD_DECLARE(class NavigationStateImpl;);
VECGEOM_HOST_FORWARD_DECLARE(using NavigationState = NavigationStateImpl;);

VECGEOM_DEVICE_FORWARD_DECLARE(class NavigationStateImpl;);
VECGEOM_DEVICE_FORWARD_DECLARE(using NavigationState = NavigationStateImpl;);

class NavStateTuple;
inline namespace VECGEOM_IMPL_NAMESPACE {

class NavStateIndex;
using NavigationState = NavigationStateImpl;

} // namespace VECGEOM_IMPL_NAMESPACE

} // namespace vecgeom

#undef NavigationStateImpl

#endif // VECGEOM_NAVIGATION_NAVSTATEFWD_H_
