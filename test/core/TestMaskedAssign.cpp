#include "VecGeom/base/Global.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include "VecGeom/base/Assert.h"

int gGlobalIntForSideEffect = -1;

__attribute__((noinline)) double foo(double x)
{
  gGlobalIntForSideEffect = 1;
  return x;
}

__attribute__((noinline)) void test1(double y)
{
  gGlobalIntForSideEffect = -1;
  double x                = 1;
  // this variant calls the foo function
  vecCore::MaskedAssign(x, y < 0., foo(x)); // NOLINT
  VECGEOM_ASSERT(gGlobalIntForSideEffect == 1);
}

__attribute__((noinline)) void test2(double y)
{
  gGlobalIntForSideEffect = -1;
  double x                = 1;
  // this variant should never call the foo function
  vecCore__MaskedAssignFunc(x, y < 0., foo(x)); // NOLINT
  VECGEOM_ASSERT(gGlobalIntForSideEffect == -1);
}

int main()
{
  test1(1.);
  test2(1.);
  return 0;
}
