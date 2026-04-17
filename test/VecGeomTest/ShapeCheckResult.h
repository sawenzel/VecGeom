// Structured ShapeTester violation/result helpers.

#ifndef VECGEOM_TEST_VECGEOMTEST_SHAPECHECKRESULT_HH
#define VECGEOM_TEST_VECGEOMTEST_SHAPECHECKRESULT_HH

#include <sstream>
#include <string>
#include <vector>

#include "VecGeom/base/Vector3D.h"

namespace vecgeom {
namespace test {

using Precision = vecgeom::Precision;
using Vec_t     = vecgeom::Vector3D<Precision>;

enum class ShapeSampleCategory { kUnknown = 0, kInside, kSurface, kEdge, kOutside };

inline const char *ShapeSampleCategoryLabel(ShapeSampleCategory category)
{
  switch (category) {
  case ShapeSampleCategory::kInside:
    return "inside";
  case ShapeSampleCategory::kSurface:
    return "surface";
  case ShapeSampleCategory::kEdge:
    return "edge";
  case ShapeSampleCategory::kOutside:
    return "outside";
  default:
    return "unknown";
  }
}

struct ShapeCheckContext {
  int sample_index                 = -1;
  ShapeSampleCategory sample_group = ShapeSampleCategory::kUnknown;
  int convention_bit               = -1;
};

struct ShapeCheckOccurrence {
  ShapeCheckContext context;
  Vec_t point;
  Vec_t direction;
  Precision distance;
};

struct ShapeViolation {
  std::string message;
  int convention_bit = -1;
  int count          = 0;
  std::vector<ShapeCheckOccurrence> displayed_occurrences;
};

struct ShapeRecordDecision {
  int occurrence_count = 0;
  bool should_display  = false;
  bool suppress_future = false;
};

class ShapeCheckResult {
public:
  ShapeRecordDecision Record(const std::string &message, const Vec_t &point, const Vec_t &direction, Precision distance,
                             int max_display, const ShapeCheckContext &context = {})
  {
    auto &violation = FindOrAdd(message, context.convention_bit);
    violation.count++;

    ShapeRecordDecision decision;
    decision.occurrence_count = violation.count;
    decision.should_display   = violation.count <= max_display;
    decision.suppress_future  = violation.count == max_display;

    if (decision.should_display) {
      violation.displayed_occurrences.push_back({context, point, direction, distance});
    }
    return decision;
  }

  void Clear() { fViolations.clear(); }

  int CountErrors() const
  {
    int total = 0;
    for (auto const &violation : fViolations) {
      total += violation.count;
    }
    return total;
  }

  int CountViolationTypes() const { return static_cast<int>(fViolations.size()); }

  const std::vector<ShapeViolation> &Violations() const { return fViolations; }

private:
  ShapeViolation &FindOrAdd(const std::string &message, int convention_bit)
  {
    for (auto &violation : fViolations) {
      if (violation.message == message) {
        if (violation.convention_bit < 0 && convention_bit >= 0) violation.convention_bit = convention_bit;
        return violation;
      }
    }
    fViolations.push_back({message, convention_bit, 0, {}});
    return fViolations.back();
  }

  std::vector<ShapeViolation> fViolations;
};

inline std::string FormatShapeCheckContext(const ShapeCheckContext &context)
{
  std::ostringstream out;
  out << "sample_index=" << context.sample_index << " category=" << ShapeSampleCategoryLabel(context.sample_group);
  if (context.convention_bit >= 0) out << " convention_bit=" << context.convention_bit;
  return out.str();
}

} // namespace test
} // namespace vecgeom

#endif
