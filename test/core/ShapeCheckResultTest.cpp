// Purpose: Unit tests for ShapeCheckResult reporting helpers.

#undef NDEBUG

#include "VecGeom/base/FpeEnable.h"
#include "VecGeomTest/ShapeCheckResult.h"
#include "VecGeom/base/Assert.h"

int main()
{
  vecgeom::test::ShapeCheckResult result;
  vecgeom::test::Vec_t point(1., 2., 3.);
  vecgeom::test::Vec_t dir(0., 1., 0.);
  vecgeom::test::ShapeCheckContext inside_context;
  inside_context.sample_index   = 4;
  inside_context.sample_group   = vecgeom::test::ShapeSampleCategory::kInside;
  inside_context.convention_bit = 6;

  VECGEOM_ASSERT(result.CountErrors() == 0);
  VECGEOM_ASSERT(result.CountViolationTypes() == 0);

  auto first = result.Record("error-a", point, dir, 1.5, 2, inside_context);
  VECGEOM_ASSERT(first.should_display);
  VECGEOM_ASSERT(!first.suppress_future);
  VECGEOM_ASSERT(first.occurrence_count == 1);
  VECGEOM_ASSERT(result.CountErrors() == 1);
  VECGEOM_ASSERT(result.CountViolationTypes() == 1);
  VECGEOM_ASSERT(result.Violations().front().displayed_occurrences.size() == 1);
  VECGEOM_ASSERT(result.Violations().front().convention_bit == 6);
  VECGEOM_ASSERT(result.Violations().front().displayed_occurrences.front().context.sample_index == 4);

  auto second = result.Record("error-a", point, dir, 2.5, 2, inside_context);
  VECGEOM_ASSERT(second.should_display);
  VECGEOM_ASSERT(second.suppress_future);
  VECGEOM_ASSERT(second.occurrence_count == 2);
  VECGEOM_ASSERT(result.CountErrors() == 2);
  VECGEOM_ASSERT(result.Violations().front().displayed_occurrences.size() == 2);

  auto third = result.Record("error-a", point, dir, 3.5, 2, inside_context);
  VECGEOM_ASSERT(!third.should_display);
  VECGEOM_ASSERT(!third.suppress_future);
  VECGEOM_ASSERT(third.occurrence_count == 3);
  VECGEOM_ASSERT(result.CountErrors() == 3);
  VECGEOM_ASSERT(result.Violations().front().displayed_occurrences.size() == 2);

  auto fourth = result.Record("error-b", point, dir, 4.5, 1);
  VECGEOM_ASSERT(fourth.should_display);
  VECGEOM_ASSERT(fourth.suppress_future);
  VECGEOM_ASSERT(result.CountErrors() == 4);
  VECGEOM_ASSERT(result.CountViolationTypes() == 2);

  result.Clear();
  VECGEOM_ASSERT(result.CountErrors() == 0);
  VECGEOM_ASSERT(result.CountViolationTypes() == 0);

  return 0;
}
