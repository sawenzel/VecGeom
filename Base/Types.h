#ifndef VECGEOM_BASE_TYPES_H_
#define VECGEOM_BASE_TYPES_H_

enum ImplType { kVc, kCuda, kScalar, kCilk };

template <ImplType it>
struct Impl {};

template <typename Type>
struct Vector3D;

template <typename Type>
struct SOA3D;

#endif // VECGEOM_BASE_TYPES_H_