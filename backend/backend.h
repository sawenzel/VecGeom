#ifndef VECGEOM_BACKEND_BACKEND_H_
#define VECGEOM_BACKEND_BACKEND_H_

namespace vecgeom {

template <ImplType it, typename Type>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void CondAssign(typename Impl<it>::bool_v const &cond,
                Type const &thenval, Type const &elseval, Type *const output);

template <ImplType it, typename Type1, typename Type2>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void MaskedAssign(typename Impl<it>::bool_v const &cond,
                  Type1 const &thenval, Type2 *const output);

template <ImplType it, typename Type>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
Type Abs(const Type&);

template <ImplType it, typename Type>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
Type Sqrt(const Type&);

} // End namespace vecgeom

#endif // VECGEOM_BACKEND_BACKEND_H_