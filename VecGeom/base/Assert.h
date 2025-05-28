//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see https://github.com/celeritas-project/celeritas?tab=License-1-ov-file
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
/*!
 * \file Assert.h
 * \brief Macros, exceptions, and helpers for assertions and error handling.
 *
 * This defines host- and device-compatible assertion macros that are
 * currenntly toggled on the \c NDEBUG configure macros.
 *
 * Derived from celeritas corecel: DeviceRuntimeApi.hh, Macros.hh, Assert.hh
 */
//---------------------------------------------------------------------------/

#ifndef VECGEOM_BASE_ASSERT_H_
#define VECGEOM_BASE_ASSERT_H_

#include "VecGeom/base/Config.h"
#include "VecGeom/base/Global.h"

#ifndef __CUDA_ARCH__
#include <stdexcept>
#include <sstream>
#else
#include <cassert>
#endif

// NOTE: if we want more debug granularity, these could be defined via CMake config in the future
#ifdef NDEBUG
#define VECGEOM_DEBUG 0
#define VECGEOM_DEVICE_DEBUG 0
#else
#define VECGEOM_DEBUG 1
#define VECGEOM_DEVICE_DEBUG 1
#endif

/*!
 * \def VECGEOM_DEVICE_PLATFORM
 *
 * API prefix token for the device offloading type.
 */
/*!
 * \def VECGEOM_DEVICE_API_SYMBOL
 *
 * Add a prefix "hip" or "cuda" to a code token.
 */
#if defined(VECGEOM_ENABLE_CUDA)
#define VECGEOM_DEVICE_PLATFORM cuda
#define VECGEOM_DEVICE_PLATFORM_UPPER_STR "CUDA"
#define VECGEOM_DEVICE_API_SYMBOL(TOK) cuda##TOK
#elif defined(VECGEOM_ENABLE_HIP)
// NOTE: not yet implemented
#define VECGEOM_DEVICE_PLATFORM hip
#define VECGEOM_DEVICE_PLATFORM_UPPER_STR "HIP"
#define VECGEOM_DEVICE_API_SYMBOL(TOK) hip##TOK
#else
#define VECGEOM_DEVICE_PLATFORM none
#define VECGEOM_DEVICE_PLATFORM_UPPER_STR ""
#define VECGEOM_DEVICE_API_SYMBOL(TOK) void
#endif

/*!
 * \def VECGEOM_DEVICE_SOURCE
 *
 * Defined and true if building a HIP or CUDA source file. This is a generic
 * replacement for \c __CUDACC__ .
 */
/*!
 * \def VECGEOM_DEVICE_COMPILE
 *
 * Defined and true if building device code in HIP or CUDA. This is a generic
 * replacement for \c __CUDA_ARCH__ .
 */
#if defined(__CUDACC__) || defined(__HIP__)
#define VECGEOM_DEVICE_SOURCE 1
#elif defined(__DOXYGEN__)
#define VECGEOM_DEVICE_SOURCE 0
#endif

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define VECGEOM_DEVICE_COMPILE 1
#elif defined(__DOXYGEN__)
#define VECGEOM_DEVICE_COMPILE 0
#endif

/*!
 * \def VECGEOM_UNLIKELY(condition)
 *
 * Mark the result of this condition to be "unlikely".
 *
 * This asks the compiler to move the section of code to a "cold" part of the
 * instructions, improving instruction locality. It should be used primarily
 * for error checking conditions.
 */
#if defined(__clang__) || defined(__GNUC__)
// GCC and Clang support the same builtin
#define VECGEOM_UNLIKELY(COND) __builtin_expect(!!(COND), 0)
#else
// No other compilers seem to have a similar builtin
#define VECGEOM_UNLIKELY(COND) (COND)
#endif

/*!
 * \def VECGEOM_ASSERT
 *
 * Internal debug assertion macro. This replaces standard \c assert usage.
 */
/*!
 * \def VECGEOM_NOT_CONFIGURED
 *
 * Assert if the code point is reached because an optional feature is disabled.
 * This generally should be used for the constructors of dummy class
 * definitions in, e.g., \c Foo.nocuda.cc:
 * \code
    Foo::Foo()
    {
        VECGEOM_NOT_CONFIGURED("CUDA");
    }
 * \endcode
 */
/*!
 * \def VECGEOM_DEBUG_FAIL
 *
 * Throw a debug assertion regardless of the \c VECGEOM_DEBUG setting. This
 * is used internally but is also useful for catching subtle programming errors
 * in downstream code.
 */
/*!
 * \def VECGEOM_ASSERT_UNREACHABLE
 *
 * Throw an assertion if the code point is reached. When debug assertions are
 * turned off, this changes to a compiler hint that improves optimization (and
 * may force the code to exit uncermoniously if the point is encountered,
 * rather than continuing on with undefined behavior).
 */
/*!
 * \def VECGEOM_VALIDATE
 *
 * Always-on runtime assertion macro. This can check user input and input data
 * consistency, and will raise std::runtime_error on failure with a descriptive error
 * message that is streamed as the second argument. If used
 * in \c __device__ -annotated code, the second argument *must* be a single C string.
 *
 * An always-on debug-type assertion without a detailed message can be
 * constructed by omitting the stream (but leaving the comma):
 * \code
    VECGEOM_VALIDATE(file_stream,);
 * \endcode
 */

#if !defined(__HIP__) && !defined(__CUDA_ARCH__)
// Throw in host code
#define VECGEOM_DEBUG_THROW_(MSG, WHICH) throw ::vecgeom::make_debug_error(#WHICH, MSG, __FILE__, __LINE__)
#elif defined(__CUDA_ARCH__) && !defined(NDEBUG)
// Use the assert macro for CUDA when supported
#define VECGEOM_DEBUG_THROW_(MSG, WHICH) assert(false && sizeof(#WHICH ": " MSG))
#else
// Use a special device function to emulate assertion failure if HIP
// (assertion from multiple threads simultaeously can cause unexpected device
// failures on AMD hardware) or if NDEBUG is in use with CUDA
#define VECGEOM_DEBUG_THROW_(MSG, WHICH) ::vecgeom::device_debug_error(#WHICH, MSG, __FILE__, __LINE__)
#endif

#define VECGEOM_DEBUG_FAIL(MSG, WHICH) \
  do {                                 \
    VECGEOM_DEBUG_THROW_(MSG, WHICH);  \
    ::vecgeom::unreachable();          \
  } while (0)
#define VECGEOM_DEBUG_ASSERT_(COND, WHICH) \
  do {                                     \
    if (VECGEOM_UNLIKELY(!(COND))) {       \
      VECGEOM_DEBUG_THROW_(#COND, WHICH);  \
    }                                      \
  } while (0)
#define VECGEOM_NOASSERT_(COND) \
  do {                          \
    if (false && (COND)) {      \
    }                           \
  } while (0)

#if (VECGEOM_DEBUG && !VECGEOM_DEVICE_COMPILE) || (VECGEOM_DEVICE_DEBUG && VECGEOM_DEVICE_COMPILE)
#define VECGEOM_ASSERT(COND) VECGEOM_DEBUG_ASSERT_(COND, internal)
#define VECGEOM_ASSERT_UNREACHABLE() VECGEOM_DEBUG_FAIL("unreachable code point encountered", unreachable)
#else
#define VECGEOM_ASSERT(COND) VECGEOM_NOASSERT_(COND)
#define VECGEOM_ASSERT_UNREACHABLE() ::vecgeom::unreachable()
#endif

#if !VECGEOM_DEVICE_COMPILE
#define VECGEOM_RUNTIME_THROW(WHICH, WHAT, COND) \
  throw ::vecgeom::make_runtime_error(WHICH, WHAT, COND, __FILE__, __LINE__)
#else
#define VECGEOM_RUNTIME_THROW(WHICH, WHAT, COND) \
  VECGEOM_DEBUG_FAIL("Runtime errors cannot be thrown from device code", unreachable);
#endif

#if !VECGEOM_DEVICE_COMPILE
#define VECGEOM_VALIDATE(COND, MSG)                                           \
  do {                                                                        \
    if (VECGEOM_UNLIKELY(!(COND))) {                                          \
      std::ostringstream vg_runtime_msg_;                                     \
      vg_runtime_msg_ MSG;                                                    \
      VECGEOM_RUNTIME_THROW("runtime", vg_runtime_msg_.str().c_str(), #COND); \
    }                                                                         \
  } while (0)
#else
#define VECGEOM_VALIDATE(COND, MSG)                                                            \
  do {                                                                                         \
    if (VECGEOM_UNLIKELY(!(COND))) {                                                           \
      VECGEOM_RUNTIME_THROW("runtime", (::vecgeom::detail::StreamlikeIdentity {} MSG), #COND); \
    }                                                                                          \
  } while (0)
#endif

#define VECGEOM_NOT_CONFIGURED(WHAT) VECGEOM_RUNTIME_THROW("not configured", WHAT, nullptr)

/*!
 * \def VECGEOM_DEVICE_API_CALL
 *
 * Safely and portably dispatch a CUDA/HIP API call.
 *
 * When CUDA or HIP support is enabled, execute the wrapped statement
 * prepend the argument with "cuda" or "hip" and throw a
 * std::runtime_error if it fails. If no device platform is enabled, throw an
 * unconfigured assertion.
 *
 * Example:
 *
 * \code
   VECGEOM_DEVICE_API_CALL(Malloc(&ptr_gpu, 100 * sizeof(float)));
   VECGEOM_DEVICE_API_CALL(DeviceSynchronize());
 * \endcode
 */
#if defined(VECGEOM_ENABLE_CUDA) || defined(VECGEOM_ENABLE_HIP)
#define VECGEOM_DEVICE_API_CALL(STMT)                                                                              \
  do {                                                                                                             \
    using ErrT_   = VECGEOM_DEVICE_API_SYMBOL(Error_t);                                                            \
    ErrT_ result_ = VECGEOM_DEVICE_API_SYMBOL(STMT);                                                               \
    if (VECGEOM_UNLIKELY(result_ != VECGEOM_DEVICE_API_SYMBOL(Success))) {                                         \
      result_ = VECGEOM_DEVICE_API_SYMBOL(GetLastError)();                                                         \
      VECGEOM_RUNTIME_THROW(VECGEOM_DEVICE_PLATFORM_UPPER_STR, VECGEOM_DEVICE_API_SYMBOL(GetErrorString)(result_), \
                            #STMT);                                                                                \
    }                                                                                                              \
  } while (0)
#else
#define VECGEOM_DEVICE_API_CALL(STMT)      \
  do {                                     \
    VECGEOM_NOT_CONFIGURED("CUDA or HIP"); \
  } while (0)
#endif

namespace vecgeom {
//---------------------------------------------------------------------------//
// FUNCTION DECLARATIONS
//---------------------------------------------------------------------------//

#ifndef __CUDA_ARCH__
[[nodiscard]] std::logic_error make_debug_error(char const *which, char const *condition, char const *file, int line);

[[nodiscard]] std::runtime_error make_runtime_error(char const *which, char const *what, char const *condition,
                                                    char const *file, int line);
#endif

//---------------------------------------------------------------------------//
// INLINE FUNCTION DEFINITIONS
//---------------------------------------------------------------------------//

//! Invoke undefined behavior
[[noreturn]] inline VECCORE_ATT_HOST_DEVICE void unreachable()
{
#if (!defined(__CUDA_ARCH__) && (defined(__clang__) || defined(__GNUC__))) || defined(__NVCOMPILER) || \
    (defined(__CUDA_ARCH__) && CUDART_VERSION >= 11030) || defined(__HIP_DEVICE_COMPILE__)
  __builtin_unreachable();
#elif defined(_MSC_VER)
  __assume(false);
#else
  VECGEOM_UNREACHABLE;
#endif
}

#if defined(__CUDA_ARCH__) && defined(NDEBUG)
//! Host+device definition for CUDA when \c assert is unavailable
inline __attribute__((noinline)) __host__ __device__ void device_debug_error(char const *, char const *condition,
                                                                             char const *file, int line)
{
  printf("%s:%u:\nvecgeom: internal assertion failed: %s\n", file, line, condition);
  __trap();
}
#elif defined(__HIP__)
//! Host-only HIP call (whether or not NDEBUG is in use)
inline __host__ void device_debug_error(char const *which, char const *condition, char const *file, int line)
{
  throw make_debug_error(which, condition, file, line);
}

//! Device-only call for HIP (must always be declared; only used if
//! NDEBUG)
inline __attribute__((noinline)) __device__ void device_debug_error(char const *, char const *condition,
                                                                    char const *file, int line)
{
  printf("%s:%u:\nvecgeom: internal assertion failed: %s\n", file, line, condition);
  abort();
}
#endif

namespace detail {
//! Allow passing a single string into a streamlike operator for device-compatible VECGEOM_VALIDATE messages
struct StreamlikeIdentity {
   VECCORE_ATT_HOST_DEVICE operator char const *() const { return ""; }
};
inline VECCORE_ATT_HOST_DEVICE char const *operator<<(StreamlikeIdentity const &, char const *s) { return s; }
} // namespace detail

} // namespace vecgeom

#endif
