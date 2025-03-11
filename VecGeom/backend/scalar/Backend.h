/// \file scalar/Backend.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BACKEND_SCALARBACKEND_H_
#define VECGEOM_BACKEND_SCALARBACKEND_H_

#include "VecGeom/base/Global.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <type_traits>
#include <utility>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

struct kScalar {
  typedef int int_v;
  typedef Precision precision_v;
  typedef bool bool_v;
  typedef Inside_t inside_v;
  // alternative typedefs ( might supercede above typedefs )
  typedef int Int_t;
  typedef Precision Double_t;
  typedef bool Bool_t;
  typedef int Index_t; // the type of indices

  constexpr static precision_v kOne  = 1.0;
  constexpr static precision_v kZero = 0.0;
  const static bool_v kTrue          = true;
  const static bool_v kFalse         = false;

  template <class Backend>
  VECCORE_ATT_HOST_DEVICE static VECGEOM_CONSTEXPR_RETURN bool IsEqual()
  {
    return false;
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static Precision Convert(Precision const &input) { return input; }
};

template <>
VECCORE_ATT_HOST_DEVICE inline VECGEOM_CONSTEXPR_RETURN bool kScalar::IsEqual<kScalar>()
{
  return true;
}

typedef kScalar::int_v ScalarInt;
typedef kScalar::precision_v ScalarDouble;
typedef kScalar::bool_v ScalarBool;

#ifdef VECGEOM_SCALAR
constexpr size_t kVectorSize = 1;
#define VECGEOM_BACKEND_TYPE vecgeom::kScalar
#define VECGEOM_BACKEND_PRECISION_FROM_PTR(P) (*(P))
#define VECGEOM_BACKEND_PRECISION_TYPE vecgeom::Precision
#define VECGEOM_BACKEND_PRECISION_TYPE_SIZE 1
// #define VECGEOM_BACKEND_PRECISION_NOT_SCALAR
#define VECGEOM_BACKEND_BOOL vecgeom::ScalarBool
#define VECGEOM_BACKEND_INSIDE vecgeom::kScalar::inside_v
#endif

// template <typename Type>
// VECGEOM_FORCE_INLINE
// VECCORE_ATT_HOST_DEVICE
// void swap(Type &a, Type &b)
//{
//  std::swap(a, b);
//}

template <typename Type>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void copy(Type const *begin, Type const *const end, Type *const target)
{
#ifndef VECCORE_CUDA_DEVICE_COMPILATION
  std::copy(begin, end, target);
#else
  std::memcpy(target, begin, sizeof(Type) * (end - begin));
#endif
}

namespace sfinae_impl {

// SFINAE method to call the static method aligned_sizeof_data(args...) on a type if it provides it.
// Primary template: Assumes the static function does not exist
template <typename, typename, typename = void>
struct has_aligned_sizeof_data : std::false_type {};

// Specialization: Detects if the static function exists and returns a value
template <typename T, typename Ret, typename... Args>
struct has_aligned_sizeof_data<T, Ret(Args...), std::void_t<decltype(T::aligned_sizeof_data(std::declval<Args>()...))>>
    : std::is_same<decltype(T::aligned_sizeof_data(std::declval<Args>()...)), Ret> {};

// Utility variable template
template <typename T, typename Signature>
constexpr bool has_aligned_sizeof_data_v = has_aligned_sizeof_data<T, Signature>::value;

// Function template to call the static function if it exists and return its result
template <typename T, typename Ret, typename... Args>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE std::enable_if_t<has_aligned_sizeof_data_v<T, Ret(Args...)>, Ret>
call_aligned_sizeof_data_if_exists(Args &&...args)
{
  return T::aligned_sizeof_data(std::forward<Args>(args)...);
}

// Overload for when the function does not exist (fallback return value)
template <typename T, typename Ret, typename... Args>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE std::enable_if_t<!has_aligned_sizeof_data_v<T, Ret(Args...)>, Ret>
call_aligned_sizeof_data_if_exists(Args &&...)
{
  return Ret{}; // Returns default-constructed value of Ret (0 for int, "" for string, etc.)
}
} // namespace sfinae_impl

///< @brief Helper for managing aligned data in a buffer
///< @details Usage: Create a memory buffer with the size equal to sizeof(T) + kAlignmentBoundary
///<   AlignedAllocator a(buffer, buffer_size);
///<   T* aligned_data = a.aligned_alloc<T>(kAlignmentBoundary, constructor_args...);
///< Multiple aligned data can be allocated in the same buffer, given that the allocated buffer size fits them
///<   U* next_aligned_data = a.aligned_alloc<T>(kAlignmentBoundary, constructor_args...);
///< The aligned_aloc function returns nullptr if the type cannot be fitted aligned in the prealocated buffer
struct AlignedAllocator {
  using size_t = std::size_t;
  void *address;
  size_t sz;

  AlignedAllocator()                         = delete;
  AlignedAllocator(const AlignedAllocator &) = delete;

  VECCORE_ATT_HOST_DEVICE
  AlignedAllocator(void *buffer, size_t size) : address{buffer}, sz{size} {}

  /**
   *  @brief Fit aligned storage in buffer. Copied from bits/aligned.h
   *
   *  This function tries to fit @a size bytes of storage with alignment
   *  @a align into the buffer @a ptr of size @a space bytes.  If such
   *  a buffer fits then @a ptr is changed to point to the first byte of the
   *  aligned storage and @a space is reduced by the bytes used for alignment.
   *
   *  C++11 20.6.5 [ptr.align]
   *
   *  @param align   A fundamental or extended alignment value.
   *  @param size    Size of the aligned storage required.
   *  @param ptr     Pointer to a buffer of @a space bytes.
   *  @param space   Size of the buffer pointed to by @a ptr.
   *  @return the updated pointer if the aligned storage fits, otherwise nullptr.
   *
   */
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void *align(size_t align, size_t size, void *&ptr, size_t &space) noexcept
  {
    if (space < size) return nullptr;
    const auto intptr  = reinterpret_cast<uintptr_t>(ptr);
    const auto aligned = (intptr - 1u + align) & -align;
    const auto diff    = aligned - intptr;
    if (diff > (space - size))
      return nullptr;
    else {
      space -= diff;
      return ptr = reinterpret_cast<void *>(aligned);
    }
  }

  /// @brief Compute buffer size needed to align an array of elements of type T, based on constructor arguments
  /// @tparam T Type for which to count the aligned size
  /// @tparam ...Args Optional arguments pack, assuming T provides a function having the signature
  /// `T::aligned_sizeof_data(args...)` which calculates the size of its aligned data excluding self
  /// @param num_elements Number of elements in the array
  /// @param alignment Requested alignment of the data
  /// @param ...args Concrete arguments passed by value to `T::aligned_sizeof_data`
  /// @return Size to be allocated in bytes
  template <typename T, typename... Args>
  VECCORE_ATT_HOST_DEVICE static size_t aligned_sizeof(size_t num_elements, size_t alignment, const Args... args)
  {
    size_t aligned_size = num_elements * sizeof(T) + vecCore::math::Max(alignment, alignof(T));
    aligned_size += num_elements * sfinae_impl::call_aligned_sizeof_data_if_exists<T, size_t>(args...);
    return aligned_size;
  }

  /// @brief Allocate and construct an object at the next free aligned address in the buffer.
  /// @details The function returns nullptr in case there is not enough aligned space free.
  /// @tparam T Type to be allocated
  /// @tparam ...Args Argument pack matching a valid constructor
  /// @param num_elements Number of elements to be allocated
  /// @param alignment Requested alignment, minimum alignof(T), multiple of alignof(T)
  /// @param ...args Arguments passed by reference to the constructor
  /// @return Pointer to allocated object in the buffer
  template <typename T, typename... Args>
  VECCORE_ATT_HOST_DEVICE T *aligned_alloc(size_t num_elements, size_t alignment, Args &&...args)
  {
    if (alignment < alignof(T)) alignment = alignof(T);
    if (alignment % alignof(T) == 0 && align(alignment, num_elements * sizeof(T), address, sz)) {
      T *result        = nullptr;
      auto address_old = reinterpret_cast<T *>(address);
      // make room in the buffer for the base object class num_elements times
      address = (char *)address + num_elements * sizeof(T);
      if (sz < num_elements * sizeof(T)) return nullptr;
      sz -= num_elements * sizeof(T);
      for (size_t i = 0; i < num_elements; ++i) {
        // Call constructor for each element in the right place in the buffer
        T *new_obj = new (address_old + i) T(std::forward<Args>(args)...);
        if (i == 0) result = new_obj;
      }
      assert(((unsigned long)result % alignment == 0));
      return result;
    }
    assert(0 && "No space left to allocate in buffer");
    return nullptr;
  }
};

template <typename Type>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Type *AlignedAllocate(size_t size)
{
#ifndef VECCORE_CUDA
  return static_cast<Type *>(vecCore::AlignedAlloc(kAlignmentBoundary, sizeof(Type) * size));
#else
  Type *ptr = new Type[size];
  assert(ptr != nullptr && "Error: Memory allocation failed! If on GPU, consider increasing the heap size on GPU with "
                           "CudaDeviceSetHeapLimit(new_size)");
  return ptr;
#endif
}

template <typename Type>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void AlignedFree(Type *allocated)
{
#ifndef VECCORE_CUDA
  vecCore::AlignedFree(allocated);
#else
  delete[] allocated;
#endif
}

template <typename InputIterator1, typename InputIterator2>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool equal(InputIterator1 first, InputIterator1 last,
                                                        InputIterator2 target)
{
#ifndef VECCORE_CUDA_DEVICE_COMPILATION
  return std::equal(first, last, target);
#else
  while (first != last) {
    if (*first++ != *target++) return false;
  }
  return true;
#endif
}
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_BACKEND_SCALARBACKEND_H_
