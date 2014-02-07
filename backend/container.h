#ifndef VECGEOM_BACKEND_CONTAINER_H_
#define VECGEOM_BACKEND_CONTAINER_H_

#include "base/types.h"

namespace vecgeom {

/**
 * A container class functioning like a random access array, but which can be
 * implemented differently depending on the backend (CUDA doesn't support STL
 * container, for instance).
 */
template <typename Type>
class Container;

} // End namespace vecgeom

#endif // VECGEOM_BASE_CONTAINER_H_