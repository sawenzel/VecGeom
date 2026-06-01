/// \file NavIndexTable.cpp
/// \author Andrei Gheata (andrei.gheata@cern.ch)

#include "VecGeom/management/NavIndexTable.h"
#include "VecGeom/management/Logger.h"
#include "VecGeom/management/NavIndexTableLayout.h"
#include "VecGeom/management/ReferenceNavState.h"
#include "VecGeom/navigation/NavigationState.h"

#include <vector>
#include <numeric>
#include <iomanip>
#include <limits>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

namespace {

/**
 * @brief Validates a single encoded navigation state against reference geometry truth.
 *
 * Table construction no longer performs inline per-node validation while the
 * visitor is filling records. Instead, `NavIndexTable::Validate()` performs a
 * dedicated second traversal of the geometry tree and calls this helper for
 * each encoded state reached during that traversal. The helper keeps the
 * low-level checks shared by the recursive validation code for `NavStateIndex`
 * and `NavStateTuple` in one place.
 */
template <typename EncodedNavState, typename EncodedState>
VECCORE_ATT_HOST_DEVICE bool ValidateEncodedStateWithLogging(ReferenceNavState const &reference,
                                                             EncodedState encoded_state, int &error)
{
  auto validation_error = ValidateEncodedState<EncodedNavState>(reference, encoded_state);
  if (validation_error == ReferenceNavValidationError::kNone) return true;

  error = static_cast<int>(validation_error);
  VECGEOM_LOG(critical) << "Validate: " << ToString(validation_error);
#ifndef VECCORE_CUDA
  if (!reference.IsOutside()) {
    VECGEOM_LOG(critical) << "\nreference top volume: " << reference.Top()->GetLabel();
  }
#endif
  VECGEOM_LOG(critical) << "\n";
  PrintValidationFailure<EncodedNavState>(validation_error, reference, encoded_state);

  if (validation_error == ReferenceNavValidationError::kTransformationMismatch) {
    Transformation3D reference_matrix;
    Transformation3D encoded_matrix;
    reference.TopMatrix(reference_matrix);
    EncodedNavState::TopMatrixImpl(encoded_state, encoded_matrix);
    VECGEOM_LOG(critical) << "Reference transformation: " << reference_matrix << "\n";
    VECGEOM_LOG(critical) << "Encoded transformation: " << encoded_matrix << "\n";
  }
  return false;
}

/**
 * @brief Recursively validates index-based encoded states against the geometry tree.
 */
#ifndef VECGEOM_USE_NAVTUPLE
int ValidateNavIndexRecursive(VPlacedVolume const *currentvolume, ReferenceNavState &reference, NavIndex_t nav_ind,
                              int &error)
{
  if (!ValidateEncodedStateWithLogging<NavStateIndex>(reference, nav_ind, error)) return error;

  for (auto daughter : currentvolume->GetDaughters()) {
    if (daughter->GetChildId() < 0) {
      VECGEOM_LOG(critical) << "Validate: " << ToString(ReferenceNavValidationError::kIncompatibleDaughter) << "\n";
      error = static_cast<int>(ReferenceNavValidationError::kIncompatibleDaughter);
      return error;
    }
    auto child_nav_ind = nav_ind;
    NavStateIndex::PushImpl(child_nav_ind, daughter);
    reference.Push(daughter);
    auto ierr = ValidateNavIndexRecursive(daughter, reference, child_nav_ind, error);
    reference.Pop();
    if (ierr) return ierr;
  }
  return 0;
}
#else
/**
 * @brief Recursively validates tuple-based encoded states against the geometry tree.
 */
int ValidateNavTupleRecursive(VPlacedVolume const *currentvolume, ReferenceNavState &reference, NavTuple_t nav_tuple,
                              int &error)
{
  if (!ValidateEncodedStateWithLogging<NavStateTuple>(reference, nav_tuple, error)) return error;

  for (auto daughter : currentvolume->GetDaughters()) {
    if (daughter->GetChildId() < 0) {
      VECGEOM_LOG(critical) << "Validate: " << ToString(ReferenceNavValidationError::kIncompatibleDaughter) << "\n";
      error = static_cast<int>(ReferenceNavValidationError::kIncompatibleDaughter);
      return error;
    }

    auto child_nav_tuple = nav_tuple;
    NavStateTuple::PushImpl(child_nav_tuple, daughter);

    auto scene_error = ValidateSceneTransition<NavStateTuple>(nav_tuple, child_nav_tuple);
    if (scene_error != ReferenceNavValidationError::kNone) {
      error = static_cast<int>(scene_error);
      VECGEOM_LOG(critical) << "Validate: " << ToString(scene_error) << "\n";
      PrintSceneTransitionFailure<NavStateTuple>(nav_tuple, child_nav_tuple, currentvolume, daughter);
      return error;
    }

    reference.Push(daughter);
    auto ierr = ValidateNavTupleRecursive(daughter, reference, child_nav_tuple, error);
    reference.Pop();
    if (ierr) return ierr;
  }
  return 0;
}
#endif

constexpr size_t kBytesPerMByte = 1024 * 1024;
constexpr const char *kMemoryBudgetConfigHelp =
    " This memory budget is configured with -DVECGEOM_NAVTABLE_WARN_MEMORY_MB=<MB>; set it to 0 to disable size "
    "guidance.";

size_t NavTableWarningLimitBytes()
{
  if (VECGEOM_NAVTABLE_WARN_MEMORY_MB <= 0) return 0;
  return size_t(VECGEOM_NAVTABLE_WARN_MEMORY_MB) * kBytesPerMByte;
}

double BytesToMBytes(size_t bytes) { return double(bytes) / double(kBytesPerMByte); }

size_t CeilBytesToMBytes(size_t bytes) { return bytes / kBytesPerMByte + (bytes % kBytesPerMByte != 0); }

#ifdef VECGEOM_USE_NAVTUPLE
size_t SaturatingAdd(size_t lhs, size_t rhs)
{
  const auto max_size = std::numeric_limits<size_t>::max();
  if (lhs > max_size - rhs) return max_size;
  return lhs + rhs;
}

size_t SaturatingMultiply(size_t lhs, size_t rhs)
{
  const auto max_size = std::numeric_limits<size_t>::max();
  if (lhs != 0 && rhs > max_size / lhs) return max_size;
  return lhs * rhs;
}

size_t EstimateIndexTableSizeUpperBound(size_t touchables)
{
  if (touchables == 0) return sizeof(NavIndex_t);

  // Every non-world touchable contributes once to the daughter-index arrays.
  const auto daughter_indices                  = touchables - 1;
  constexpr size_t kMaxRecordAlignmentIndices  = 1;
  constexpr size_t kMaxTransformPaddingIndices = (sizeof(Precision) > sizeof(NavIndex_t)) ? size_t{1} : size_t{0};
  constexpr size_t kFixedRecordIndices =
      NavIndexTableLayout::Index::kDaughters + kMaxRecordAlignmentIndices + kMaxTransformPaddingIndices;
  constexpr size_t kCachedTransformBytes =
      NavIndexTableLayout::Index::kStoredTransformPrecisionCount * sizeof(Precision);
  constexpr size_t kPerTouchableBytes = kFixedRecordIndices * sizeof(NavIndex_t) + kCachedTransformBytes;

  auto estimate = sizeof(NavIndex_t);
  estimate      = SaturatingAdd(estimate, SaturatingMultiply(touchables, kPerTouchableBytes));
  estimate      = SaturatingAdd(estimate, SaturatingMultiply(daughter_indices, sizeof(NavIndex_t)));
  return estimate;
}

struct NavIndexCompatibility {
  bool compatible          = true;
  size_t max_daughters     = 0;
  unsigned int logical_id  = 0;
  const char *logical_name = "";
};

NavIndexCompatibility CheckNavIndexCompatibility()
{
  NavIndexCompatibility result;
  for (auto const &entry : GeoManager::Instance().GetLogicalVolumesMap()) {
    auto const *logical = entry.second;
    if (logical == nullptr) continue;
    const auto daughters = logical->GetDaughters().size();
    if (daughters < std::numeric_limits<unsigned short>::max() || daughters <= result.max_daughters) continue;
    result.compatible    = false;
    result.max_daughters = daughters;
    result.logical_id    = logical->id();
    result.logical_name  = logical->GetName();
  }
  return result;
}

void LogNavIndexIncompatibilityInfo(NavIndexCompatibility const &compatibility)
{
  if (compatibility.compatible) return;
  VECGEOM_LOG(info) << "NavStateIndex is not recommended for this geometry: logical volume " << compatibility.logical_id
                    << " (" << compatibility.logical_name << ") has " << compatibility.max_daughters
                    << " daughters, while NavStateIndex records store the daughter count in an unsigned short.";
}

bool LogIndexPerformanceRecommendation(size_t current_table_size, size_t index_table_size, size_t limit,
                                       size_t touchables, int current_tuple_depth, bool conservative,
                                       NavIndexCompatibility const &compatibility)
{
  if (limit == 0) return false;
  if (index_table_size > limit) return false;
  if (!compatibility.compatible) return false;

  VECGEOM_LOG(info) << "Recommended VECGEOM_NAV=index for this geometry: the navigation index table is "
                    << (conservative ? "conservatively estimated from " : "estimated for ") << touchables
                    << (conservative ? " touchables at no more than " : " touchables at ") << std::setprecision(5)
                    << BytesToMBytes(index_table_size)
                    << " MBytes, below VECGEOM_NAVTABLE_WARN_MEMORY_MB=" << VECGEOM_NAVTABLE_WARN_MEMORY_MB
                    << " MBytes. Current VECGEOM_NAV=tuple with VECGEOM_NAVTUPLE_MAXDEPTH=" << current_tuple_depth
                    << " gives a " << std::setprecision(5) << BytesToMBytes(current_table_size) << " MBytes table. "
                    << "Index navigation is preferred when it fits because tuple navigation uses a larger per-track "
                       "state."
                    << kMemoryBudgetConfigHelp;
  return true;
}

#ifndef VECGEOM_NAVTABLE_RECOMMEND
void LogIndexGuidanceWithoutRecommendation(size_t index_table_size, size_t limit, size_t touchables,
                                           NavIndexCompatibility const &compatibility)
{
  if (!compatibility.compatible) {
    LogNavIndexIncompatibilityInfo(compatibility);
    return;
  }
  if (index_table_size <= limit) return;

  VECGEOM_LOG(info) << "NavStateIndex navigation table is conservatively estimated from " << touchables
                    << " touchables at no more than " << std::setprecision(5) << BytesToMBytes(index_table_size)
                    << " MBytes, above VECGEOM_NAVTABLE_WARN_MEMORY_MB=" << VECGEOM_NAVTABLE_WARN_MEMORY_MB
                    << " MBytes. Tuple navigation remains selected for this memory budget." << kMemoryBudgetConfigHelp;
}

void LogTupleDepthRecommendationInfoIfUseful(size_t current_table_size, size_t limit, int current_tuple_depth)
{
  constexpr size_t kWellBelowMemoryLimitFactor = 4;
  if (limit == 0 || current_tuple_depth <= 1 || current_table_size > limit / kWellBelowMemoryLimitFactor) return;

  VECGEOM_LOG(info) << "Current NavStateTuple table size is " << std::setprecision(5)
                    << BytesToMBytes(current_table_size)
                    << " MBytes, well below VECGEOM_NAVTABLE_WARN_MEMORY_MB=" << VECGEOM_NAVTABLE_WARN_MEMORY_MB
                    << " MBytes. Smaller tuple depths reduce per-track navigation-state size and may still fit this "
                       "memory budget."
                    << kMemoryBudgetConfigHelp
                    << " Configure -DVECGEOM_NAVTABLE_RECOMMEND=ON once to estimate the smallest "
                       "VECGEOM_NAVTUPLE_MAXDEPTH for this geometry, then rebuild with the selected setting for "
                       "production.";
}
#endif
#endif

void WarnLargeNavTable(const char *mode, size_t table_size, size_t limit, size_t touchables, const char *recommendation)
{
  if (limit == 0 || table_size <= limit) return;
  VECGEOM_LOG(warning) << mode << " navigation table is estimated at " << std::setprecision(5)
                       << BytesToMBytes(table_size) << " MBytes for " << touchables
                       << " touchables, exceeding VECGEOM_NAVTABLE_WARN_MEMORY_MB=" << VECGEOM_NAVTABLE_WARN_MEMORY_MB
                       << " MBytes. Allocation, filling, validation, and host/device transfer may be slow. "
                       << recommendation << kMemoryBudgetConfigHelp << " If this table size is expected, configure "
                       << "-DVECGEOM_NAVTABLE_WARN_MEMORY_MB=" << CeilBytesToMBytes(table_size)
                       << " or set it to 0 to disable this warning.";
}

#ifdef VECGEOM_USE_NAVTUPLE
#ifdef VECGEOM_NAVTABLE_RECOMMEND
size_t EstimateIndexTableSize(VPlacedVolume const *top, int depth_limit)
{
  ReferenceNavState state;
  BuildNavIndexVisitor visitor(depth_limit, true);
  NavIndex_t id = 1;
  NavIndexTable::visitAllPlacedVolumesNavIndex(top, &visitor, &state, id);
  return visitor.GetTableSize();
}

size_t EstimateTupleTableSize(VPlacedVolume const *top, int depth_limit, int min_per_scene, int tuple_depth)
{
  ReferenceNavState state;
  BuildNavIndexVisitor visitor(depth_limit, true);
  NavIndex_t id = 1;
  int scene_id  = 0;
  visitor.NodeReduction(min_per_scene, tuple_depth - 1, false);
  NavIndexTable::visitAllPlacedVolumesNavTuple(top, &visitor, &state, id, scene_id, scene_id);
  return visitor.GetTableSize();
}
#endif

void ReportTupleNavTableGuidance(VPlacedVolume const *top, int depth_limit, int min_per_scene, int current_tuple_depth,
                                 size_t current_table_size, size_t limit)
{
  if (limit == 0) return;

#ifndef VECGEOM_NAVTABLE_RECOMMEND
  const auto touchables             = GeoManager::Instance().GetTotalNodeCount();
  const auto index_compatibility    = CheckNavIndexCompatibility();
  const auto cheap_index_table_size = EstimateIndexTableSizeUpperBound(touchables);
  const auto index_fits = LogIndexPerformanceRecommendation(current_table_size, cheap_index_table_size, limit,
                                                            touchables, current_tuple_depth, true, index_compatibility);
  if (!index_fits)
    LogIndexGuidanceWithoutRecommendation(cheap_index_table_size, limit, touchables, index_compatibility);
  if (!index_fits) LogTupleDepthRecommendationInfoIfUseful(current_table_size, limit, current_tuple_depth);
  if (current_table_size <= limit) return;
  WarnLargeNavTable("NavStateTuple", current_table_size, limit, touchables,
                    index_fits ? "Consider -DVECGEOM_NAV=index, or configure -DVECGEOM_NAVTABLE_RECOMMEND=ON once to "
                                 "check tuple-depth alternatives before rebuilding with the selected setting for "
                                 "production."
                               : "Extra VECGEOM_NAVTUPLE_MAXDEPTH estimates are disabled by default to avoid "
                                 "increasing geometry initialization time. Configure -DVECGEOM_NAVTABLE_RECOMMEND=ON "
                                 "once to get recommendations, then rebuild with the selected setting for production.");
#else
  const auto touchables             = GeoManager::Instance().GetTotalNodeCount();
  const auto index_compatibility    = CheckNavIndexCompatibility();
  const auto cheap_index_table_size = EstimateIndexTableSizeUpperBound(touchables);
  // Fast-path the preferred index recommendation without extra table scans when the conservative estimate fits.
  const auto cheap_index_fits_budget = LogIndexPerformanceRecommendation(
      current_table_size, cheap_index_table_size, limit, touchables, current_tuple_depth, true, index_compatibility);
  if (cheap_index_fits_budget) {
    WarnLargeNavTable("NavStateTuple", current_table_size, limit, touchables,
                      "Configure -DVECGEOM_NAV=index to use the faster navigation representation that fits the "
                      "configured memory budget.");
    return;
  }

  auto index_table_size = cheap_index_table_size;
  if (index_compatibility.compatible) index_table_size = EstimateIndexTableSize(top, depth_limit);
  if (index_compatibility.compatible && index_table_size <= limit) {
    LogIndexPerformanceRecommendation(current_table_size, index_table_size, limit, touchables, current_tuple_depth,
                                      false, index_compatibility);
    WarnLargeNavTable("NavStateTuple", current_table_size, limit, touchables,
                      "Configure -DVECGEOM_NAV=index to use the faster navigation representation that fits the "
                      "configured memory budget.");
    return;
  }

  if (current_table_size <= limit) {
    int smallest_fitting_depth = current_tuple_depth;
    size_t smallest_size       = current_table_size;
    for (int depth = 1; depth < current_tuple_depth; ++depth) {
      auto estimated_size = EstimateTupleTableSize(top, depth_limit, min_per_scene, depth);
      if (estimated_size <= limit) {
        smallest_fitting_depth = depth;
        smallest_size          = estimated_size;
        break;
      }
    }
    if (smallest_fitting_depth < current_tuple_depth) {
      VECGEOM_LOG(info) << "Recommended VECGEOM_NAVTUPLE_MAXDEPTH=" << smallest_fitting_depth
                        << " for this geometry: it is the smallest checked tuple depth that keeps the navigation "
                           "tuple table below VECGEOM_NAVTABLE_WARN_MEMORY_MB="
                        << VECGEOM_NAVTABLE_WARN_MEMORY_MB << " MBytes (estimated " << std::setprecision(5)
                        << BytesToMBytes(smallest_size) << " MBytes). Current depth " << current_tuple_depth
                        << " gives " << std::setprecision(5) << BytesToMBytes(current_table_size) << " MBytes."
                        << kMemoryBudgetConfigHelp;
      if (index_compatibility.compatible) {
        VECGEOM_LOG(info) << "The navigation index table is estimated at " << std::setprecision(5)
                          << BytesToMBytes(index_table_size)
                          << " MBytes, above the configured limit; larger tuple depths increase per-track state.";
      } else {
        LogNavIndexIncompatibilityInfo(index_compatibility);
      }
    } else {
      VECGEOM_LOG(info) << "Current VECGEOM_NAVTUPLE_MAXDEPTH=" << current_tuple_depth
                        << " is the smallest checked tuple depth that keeps the navigation tuple table below "
                           "VECGEOM_NAVTABLE_WARN_MEMORY_MB="
                        << VECGEOM_NAVTABLE_WARN_MEMORY_MB << " MBytes." << kMemoryBudgetConfigHelp;
      if (index_compatibility.compatible) {
        VECGEOM_LOG(info) << "The navigation index table is estimated at " << std::setprecision(5)
                          << BytesToMBytes(index_table_size)
                          << " MBytes, above the configured limit; larger tuple depths increase per-track state.";
      } else {
        LogNavIndexIncompatibilityInfo(index_compatibility);
      }
    }
    return;
  }

  constexpr int kMaxTupleDepthEstimate = 16;
  for (int depth = current_tuple_depth + 1; depth <= kMaxTupleDepthEstimate; ++depth) {
    auto estimated_size = EstimateTupleTableSize(top, depth_limit, min_per_scene, depth);
    if (estimated_size <= limit) {
      VECGEOM_LOG(warning) << "NavStateTuple navigation table is estimated at " << std::setprecision(5)
                           << BytesToMBytes(current_table_size) << " MBytes for "
                           << GeoManager::Instance().GetTotalNodeCount()
                           << " touchables, exceeding VECGEOM_NAVTABLE_WARN_MEMORY_MB="
                           << VECGEOM_NAVTABLE_WARN_MEMORY_MB
                           << " MBytes. Allocation, filling, validation, and host/device transfer may be slow. The "
                              "navigation index table is estimated at "
                           << std::setprecision(5) << BytesToMBytes(index_table_size)
                           << " MBytes, also above the configured limit. Recommended VECGEOM_NAVTUPLE_MAXDEPTH="
                           << depth << " gives an estimated " << std::setprecision(5) << BytesToMBytes(estimated_size)
                           << " MBytes table, at the cost of a larger per-track tuple state. If the current table size "
                              "is expected, configure -DVECGEOM_NAVTABLE_WARN_MEMORY_MB="
                           << CeilBytesToMBytes(current_table_size) << " or set it to 0 to disable this warning.";
      if (!index_compatibility.compatible) LogNavIndexIncompatibilityInfo(index_compatibility);
      return;
    }
  }

  VECGEOM_LOG(warning) << "NavStateTuple navigation table is estimated at " << std::setprecision(5)
                       << BytesToMBytes(current_table_size) << " MBytes for "
                       << GeoManager::Instance().GetTotalNodeCount()
                       << " touchables, exceeding VECGEOM_NAVTABLE_WARN_MEMORY_MB=" << VECGEOM_NAVTABLE_WARN_MEMORY_MB
                       << " MBytes. Allocation, filling, validation, and host/device transfer may be slow. The "
                          "navigation index table is estimated at "
                       << std::setprecision(5) << BytesToMBytes(index_table_size)
                       << " MBytes, also above the configured limit, and estimates up to VECGEOM_NAVTUPLE_MAXDEPTH="
                       << kMaxTupleDepthEstimate
                       << " did not fit the configured limit. If this table size is expected, configure "
                       << "-DVECGEOM_NAVTABLE_WARN_MEMORY_MB=" << CeilBytesToMBytes(current_table_size)
                       << " or set it to 0 to disable this warning.";
  if (!index_compatibility.compatible) LogNavIndexIncompatibilityInfo(index_compatibility);
#endif
}
#endif

} // namespace

/// @brief Filling one touchable record (TR) to use with NavStateTuple.
/// @details The records are delimited at unsigned int (4 bytes) size:
///
///  |    parent_addr    |  placed_vol_index |  index_in_parent |  touchable_index  |  logical_vol_addr |
///  |   scene |new_scene|lev |hasm|otr |orot|//// padding /////|
///                                          |            translation[0]            |
///  |            translation[1]             |            translation[2]            |
///  |            rotation[0]                |            rotation[1]               |
///  |              ...                      |               ...                    |
///  |            rotation[7]                |            rotation[8]               |
///  | logical_vol_index |       nd          |     d0_addr      |     d1_addr       |
///  |        ...        |    dN_addr        |     d0_record    |       ...         |
///  |      d1_record    |       ...         |     dN_record    |       ...         |
///
/// parent_addr         = index of the parent TR in the table
/// placed_vol_index    = index of the placed volume in the table of placed volumes
/// index_in_parent     = placed volume index in the parent logical volume
/// touchable index     = sequential index of the touchable in the scene. In case there
///                       is a single world scene, this index is unique
/// logical_vol_addr    = index of the logical volume record in the table
/// lev                 = depth of the current state
/// hasm                = offset of the first transformation element in the record
///                       (0 = no transformation)
/// otr                 = has translation flag
/// orot                = has rotation flag
/// nd                  = number of children of the logical volume
/// scene               = scene id
/// new_scene           = new scene id
/// d0_addr ... dn_addr = index of the daughter TR's
/// @param state Navigation state pointing to the current touchable
/// @param level Touchable depth
/// @param mother Address of the parent touchable record
/// @param dind Index of this touchable in the parent list of daughters
/// @return index of TR
NavIndex_t BuildNavIndexVisitor::apply_tuple(ReferenceNavState *state, int level, NavIndex_t mother, int dind,
                                             NavIndex_t &id, int scene_id, int new_scene_id)
{
  bool cacheTrans   = true;
  NavIndex_t record = fCurrent;
  auto pv           = state->Top();
  auto ichild       = pv->GetChildId();
  if (ichild < 0 && pv != GeoManager::Instance().GetWorld())
    VECGEOM_LOG(critical) << "Found placed volume having undefined child id";
  auto lv         = pv->GetLogicalVolume();
  auto ivol       = lv->id();
  bool selected   = IsSelected(ivol);
  bool visited    = IsVisited(ivol);
  bool new_scene  = level == 1 && scene_id > 0;
  size_t raw_size = lv->GetDaughters().size();
  if (raw_size > std::numeric_limits<NavIndex_t>::max()) {
    throw std::runtime_error("Volume has too many daughters for NavStateIndex");
  }
  NavIndex_t nd = static_cast<NavIndex_t>(raw_size);
  if (lv->GetDaughters().size() >= std::numeric_limits<unsigned int>::max()) {
    VECGEOM_LOG(critical) << "Navigation table does not support volumes having more than "
                          << std::numeric_limits<unsigned int>::max() << " daughters";
    fError = 1;
    return 0;
  }
  if (ivol > std::numeric_limits<int>::max()) {
    VECGEOM_LOG(critical) << "not supporting more than " << std::numeric_limits<int>::max() << " logical volumes";
    fError = 1;
    return 0;
  }
  if (new_scene_id >= std::numeric_limits<unsigned short>::max()) {
    VECGEOM_LOG(critical) << "not supporting more than " << std::numeric_limits<unsigned short>::max() << " scenes";
    fError = 1;
    return 0;
  }

  // All transformations are currently cached for the NavStateTuple case, so not using fLimitDepth

  size_t size_trans = 0;
  Transformation3D mat;
  bool has_trans = false;
  bool has_rot   = false;
  if (cacheTrans) {
    state->TopMatrix(mat);
    has_trans  = mat.HasTranslation();
    has_rot    = mat.HasRotation();
    size_trans = (int{has_trans} * 3 + int{has_rot} * 9) * sizeof(Precision);
  }

  // To keep the transformation data aligned, we may insert padding. Precision is either the same size or
  // twice the size as NavIndex_t, so in case we need to pad, we only need to pad one NavIndex_t.
  static_assert(sizeof(Precision) == sizeof(NavIndex_t) || sizeof(Precision) == 2 * sizeof(NavIndex_t));
  const auto indicesBefore = fDoCount ? fTableSize / sizeof(NavIndex_t) : fCurrent;
  // Count of touchable-record fields before the optional transformation.
  constexpr unsigned record_count_before_trans = NavIndexTableLayout::Tuple::kRecordFieldsBeforeTransform;
  // Count of volume data fields (volume index, number of daughters plus index of each daughter)
  NavIndex_t index_scene                = GetSceneIndex(ivol);
  const unsigned record_count_daughters = visited ? 0 : nd + 2;

  const auto record_count_notrans = record_count_before_trans + record_count_daughters;
  const bool padTransformationData =
      NavIndexTableLayout::NeedsPrecisionPadding(indicesBefore + record_count_before_trans);

  // Size in bytes of the current touchable data record
  const size_t current_size =
      (record_count_notrans + unsigned{padTransformationData}) * sizeof(NavIndex_t) + size_trans;
  // current_size does not need to be a multiple of sizeof(Precision), because the start of the node could be
  // misaligned.

  if (selected) SetVisited(ivol);
  if (fDoCount) {
    fTableSize += current_size;
    return 0;
  }

  // Add data for the current element.
  NavIndex_t index_mother    = fCurrent + NavIndexTableLayout::Tuple::kParent;
  NavIndex_t index_placed    = fCurrent + NavIndexTableLayout::Tuple::kPlacedVolume;
  NavIndex_t index_child     = fCurrent + NavIndexTableLayout::Tuple::kChildId;
  NavIndex_t index_touchable = fCurrent + NavIndexTableLayout::Tuple::kTouchableId;
  NavIndex_t index_logical   = fCurrent + NavIndexTableLayout::Tuple::kLogicalRecord;
  NavIndex_t index_scenes    = fCurrent + NavIndexTableLayout::Tuple::kScenes;
  NavIndex_t index_lhtr      = fCurrent + NavIndexTableLayout::Tuple::kPacked;
  auto content_ichild        = reinterpret_cast<int *>(fNavInd + index_child);
  auto content_scene         = reinterpret_cast<unsigned short *>(fNavInd + index_scenes);
  auto content_newscene      = content_scene + NavIndexTableLayout::Tuple::kCurrentSceneHalf;
  auto content_level         = reinterpret_cast<unsigned char *>(fNavInd + index_lhtr);
  auto content_hasm          = content_level + NavIndexTableLayout::Tuple::kTransformOffsetByte;
  auto content_hast          = content_level + NavIndexTableLayout::Tuple::kHasTranslationByte;
  auto content_hasr          = content_level + NavIndexTableLayout::Tuple::kHasRotationByte;

  NavIndex_t index_trans = fCurrent + record_count_before_trans + unsigned{padTransformationData};
  VECGEOM_ASSERT((index_trans * sizeof(NavIndex_t)) % sizeof(Precision) == 0);
  auto content_translation = (Precision *)(&fNavInd[index_trans]);
  auto content_rotation    = content_translation + 3 * unsigned{has_trans};

  NavIndex_t index_lvol_id = (index_scene > 2) ? index_scene : index_trans + size_trans / sizeof(NavIndex_t);
  if (selected) SetSceneIndex(ivol, index_lvol_id);

  NavIndex_t index_nd = index_lvol_id + NavIndexTableLayout::Tuple::kDaughterCount;
  // NavIndex_t index_daughters = index_lvol_id + NavIndexTableLayout::Tuple::kDaughters; // filled by daughters.
  // Don't remove this comment
  auto content_lvol_id = reinterpret_cast<int *>(fNavInd + index_lvol_id);

  // Fill the mother index for the current node
  fNavInd[index_mother] = new_scene ? 0 : mother;

  // Fill the node index in the mother list of daughters
  if (mother > 0) {
    NavIndex_t index_inmother =
        fNavInd[mother + NavIndexTableLayout::Tuple::kLogicalRecord] + NavIndexTableLayout::Tuple::kDaughters + dind;
    fNavInd[index_inmother] = fCurrent;
  }

  // Placed volume index
  fNavInd[index_placed] = (level >= 0) ? pv->id() : 0;

  // Child index in mother
  *content_ichild = ichild;

  // Touchable index
  fNavInd[index_touchable] = id++;

  // Logical volume address
  fNavInd[index_logical] = index_lvol_id;

  // Parent scene index
  *content_scene = scene_id;

  // Current scene index (different than parent if this is a new scene)
  *content_newscene = new_scene_id;

  // Reset lhtr
  fNavInd[index_lhtr] = 0;

  // Current touchable depth
  *content_level = level;

  // Store transformation offset
  if (has_trans || has_rot) *content_hasm = record_count_before_trans + (unsigned char){padTransformationData};

  // Write the translation of the touchable if any
  if (has_trans) {
    *content_hast = 1;
    for (auto i = 0; i < 3; ++i)
      content_translation[i] = mat.Translation(i);
  }
  if (has_rot) {
    *content_hasr = 1;
    for (auto i = 0; i < 9; ++i)
      content_rotation[i] = mat.Rotation(i);
  }

  // Increment table counter
  fCurrent += current_size / sizeof(NavIndex_t);
  VECGEOM_ASSERT((fCurrent - record) * sizeof(NavIndex_t) == current_size);

  // Return if this is a scene and the volume info is already processed
  if (index_scene > 0) return record;

  // Logical volume index
  *content_lvol_id = ivol;

  // Number of daughters
  fNavInd[index_nd] = nd;
  // Daughter addresses are at next nd fields, filled while creating the daughter records

  return record;
}

NavIndex_t BuildNavIndexVisitor::apply(ReferenceNavState *state, int level, NavIndex_t mother, int dind, NavIndex_t &id)
{
  bool cacheTrans       = true;
  NavIndex_t new_mother = fCurrent;
  auto pv               = state->Top();
  auto ichild           = pv->GetChildId();
  if (ichild < 0 && pv != GeoManager::Instance().GetWorld())
    VECGEOM_LOG(critical) << "Found placed volume having undefined child id";
  auto lv           = pv->GetLogicalVolume();
  unsigned short nd = (unsigned short)lv->GetDaughters().size();
  if (lv->GetDaughters().size() >= std::numeric_limits<unsigned short>::max()) {
    VECGEOM_LOG(critical) << "Navigation table does not support volumes having more than "
                          << std::numeric_limits<unsigned short>::max() << " daughters";
    fError = 1;
    return 0;
  }
  // Check if matrix has to be cached for this node
  if (fLimitDepth > 0 && level > fLimitDepth && !lv->IsReqCaching()) cacheTrans = false;

  // To keep the transformation data sufficiently aligned, we may insert padding. Precision is either the same size or
  // twice the size as NavIndex_t, so in case we need to pad, we only need to pad one NavIndex_t.
  static_assert(sizeof(Precision) == sizeof(NavIndex_t) || sizeof(Precision) == 2 * sizeof(NavIndex_t));
  const auto indicesBefore         = fDoCount ? fTableSize / sizeof(NavIndex_t) : fCurrent;
  const auto daughterIndices       = NavIndexTableLayout::Index::kDaughters + nd + ((nd + 1) & 1);
  const bool padTransformationData = NavIndexTableLayout::NeedsPrecisionPadding(indicesBefore + daughterIndices);

  // Size in bytes of the current node data
  const size_t current_size =
      (daughterIndices + int{padTransformationData}) * sizeof(NavIndex_t) +
      int{cacheTrans} * NavIndexTableLayout::Index::kStoredTransformPrecisionCount * sizeof(Precision);
  // current_size does not need to be a multiple of sizeof(Precision), because the start of the node could be
  // misaligned.

  if (fDoCount) {
    fTableSize += current_size;
    VECGEOM_ASSERT(fTableSize % sizeof(Precision) == 0 &&
                   "NavigationIndexTable size until now must be a multiple of sizeof(Precision)");
    return 0;
  }

  // Add data for the current element.
  // Fill the mother index for the current node
  fNavInd[fCurrent + NavIndexTableLayout::Index::kParent] = mother;

  // Fill the incremental id
  fNavInd[fCurrent + NavIndexTableLayout::Index::kTouchableId] = id++;

  // Fill the node index in the mother list of daughters
  if (mother > 0) fNavInd[mother + NavIndexTableLayout::Index::kDaughters + dind] = fCurrent;

  // Placed volume index
  fNavInd[fCurrent + NavIndexTableLayout::Index::kPlacedVolume] = (level >= 0) ? pv->id() : 0;

  // Child index in mother
  auto content_ichild = reinterpret_cast<int *>(fNavInd + fCurrent + NavIndexTableLayout::Index::kChildId);
  *content_ichild     = ichild;

  // Logical volume id
  fNavInd[fCurrent + NavIndexTableLayout::Index::kLogicalVolume] = lv->id();

  // Write current level in next byte
  auto content_ddt = (unsigned char *)(&fNavInd[fCurrent + NavIndexTableLayout::Index::kPacked]);
  VECGEOM_VALIDATE(level < std::numeric_limits<unsigned char>::max(),
                   << "unsupported geometry depth: " << level << " > 255");
  *content_ddt = (unsigned char)level;

  // Write number of daughters in next 2 bytes
  auto content_nd = (unsigned short *)(content_ddt + NavIndexTableLayout::Index::kDaughterCountByte);
  *content_nd     = nd;

  // Write the flag if matrix is stored in the next byte
  auto content_hasm = (unsigned char *)(content_ddt + NavIndexTableLayout::Index::kMatrixFlagsByte);
  *content_hasm     = 0;

  // Prepare the space for the daughter indices
  auto content_dind = &fNavInd[fCurrent + NavIndexTableLayout::Index::kDaughters];
  for (size_t i = 0; i < nd; ++i)
    content_dind[i] = 0;

  fCurrent += daughterIndices;

  if (!cacheTrans) return new_mother;

  Transformation3D mat;
  // encode has_trans, translation and rotation flags in the content_hasm byte
  state->TopMatrix(mat);
  *content_hasm = NavIndexTableLayout::Index::kHasStoredMatrixFlag +
                  NavIndexTableLayout::Index::kHasTranslationFlag * (unsigned short)mat.HasTranslation() +
                  NavIndexTableLayout::Index::kHasRotationFlag * (unsigned short)mat.HasRotation();

  // insert padding before transformation elements to align them
  if (padTransformationData) fCurrent++;
  VECGEOM_ASSERT((fCurrent * sizeof(NavIndex_t)) % sizeof(Precision) == 0);

  // Write the transformation elements
  auto content_mat = (Precision *)(&fNavInd[fCurrent]);
  VECGEOM_ASSERT(reinterpret_cast<uintptr_t>(content_mat) % sizeof(Precision) == 0);
  for (auto i = 0; i < 3; ++i)
    content_mat[i] = mat.Translation(i);
  for (auto i = 0; i < 9; ++i)
    content_mat[i + 3] = mat.Rotation(i);

  // Set new value for fCurrent
  fCurrent += NavIndexTableLayout::Index::kStoredTransformPrecisionCount * sizeof(Precision) / sizeof(NavIndex_t);
  VECGEOM_ASSERT((fCurrent - new_mother) * sizeof(NavIndex_t) == current_size);
  return new_mother;
}

void BuildNavIndexVisitor::NodeReduction(int min_per_scene, int max_scene_depth, bool log_summary)
{
  // Maximum allowed scene depth and minimum number of touchables per scene are exposed as cmake options
  // #define SCENEMAKE_DEBUG 1
  int max_depth      = max_scene_depth;
  using VolPtr_t     = LogicalVolume const *;
  auto &vol_selected = fSelectedVolumes;
  // This is the total number of registered volume, larger than the number of volumes in the hierarchy
  int ntot = GeoManager::Instance().GetRegisteredVolumesCount();
  VECGEOM_ASSERT(vol_selected.size() >= size_t(ntot) &&
                 "the selected volume container size must not be less than the registered volume count");
  std::fill(vol_selected.begin(), vol_selected.end(), 0);
  std::vector<int> nrep(ntot, 0), nleaves(ntot, 0), levels(ntot, 0), score(ntot, 0), scenes(ntot, 0);
  std::vector<bool> vol_visited(ntot, false);
  std::vector<VolPtr_t> volumes(ntot, nullptr);

  // repeated placement of a volume in a parent: nrep_parent<std::pair<iparent, nrep>>
  std::vector<std::vector<std::pair<int, int>>> nrep_parent(ntot);
  int nnodes          = 0;
  int measured_depth  = 0;
  int measured_nnodes = 0;
  int count_selected  = 0;
  int found_level     = 0;

  /// This lambda increments the number of repetitions of volume ivol in parent iparent and returns its value
  auto getNrep = [&](int ivol, int iparent, int increment = 0) {
    auto &vec_parents = nrep_parent[ivol];
    for (auto &reps : vec_parents) {
      if (reps.first == iparent) {
        reps.second += increment;
        return reps.second;
      }
    }
    // not found, if asked create an entry for this parent
    if (!increment) return 0;
    vec_parents.push_back({iparent, increment});
    return increment;
  };

  /// Recursively visit parents and increase scene level if a parent is selected
  typedef std::function<void(int, int &, int &)> funcGetMotherSceneLevel_t;
  funcGetMotherSceneLevel_t getMotherSceneLevel = [&](int ivol, int &level, int &max_level) {
    if (vol_selected[ivol]) {
      level++;
      max_level = std::max(max_level, level);
    }
    auto &vec_parents = nrep_parent[ivol];
    for (auto &reps : vec_parents) {
      int iparent = reps.first;
      getMotherSceneLevel(iparent, level, max_level);
    }
    if (vol_selected[ivol]) level--;
  };

  /// This visitor counts the daughter scenes for a given volume
  typedef std::function<void(int, int &, int &)> funcCountDaughterScenes_t;
  funcCountDaughterScenes_t getDaughterSceneLevel = [&](int ivol, int &level, int &max_level) {
    bool selected = vol_selected[ivol] > 0;
    if (selected) {
      level++;
      max_level = std::max(max_level, level);
    }
    if (!vol_visited[ivol]) {
      // count once the content of all non-visited volumes
      auto numparents = nrep_parent[ivol].size();
      if (numparents <= 1) vol_visited[ivol] = true;
      int nchildren = volumes[ivol]->GetDaughters().size();
      for (int i = 0; i < nchildren; ++i) {
        auto pvol      = volumes[ivol]->GetDaughters().operator[](i);
        auto idaughter = pvol->GetLogicalVolume()->id();
        getDaughterSceneLevel(idaughter, level, max_level);
      }
    }
    if (selected) level--;
  };

#ifdef SCENEMAKE_DEBUG
  auto checkConsistency = [&](int ivol) {
    int count_nodes = 0;
    auto vol        = volumes[ivol];
    int nchildren   = vol->GetDaughters().size();
    // first reset visited flag for all daughters
    for (int i = 0; i < nchildren; ++i) {
      auto pvol              = vol->GetDaughters().operator[](i);
      int idaughter          = pvol->GetLogicalVolume()->id();
      vol_visited[idaughter] = false;
    }

    for (int i = 0; i < nchildren; ++i) {
      auto pvol     = vol->GetDaughters().operator[](i);
      int idaughter = pvol->GetLogicalVolume()->id();
      if (!vol_visited[idaughter]) {
        vol_visited[idaughter] = true;
        auto &vec_parents      = nrep_parent[idaughter];
        for (auto &reps : vec_parents) {
          int iparent = reps.first;
          if (iparent == ivol) {
            count_nodes += reps.second * (nleaves[idaughter] + 1);
            break;
          }
        }
      }
    }
    VECGEOM_ASSERT(count_nodes == nrep[ivol] * nleaves[ivol]);
  };
#endif

  /// Reduce daughter scores
  typedef std::function<void(int, int)> funcRecomputeDaughterScores_t;
  funcRecomputeDaughterScores_t recomputeDaughterScores = [&](int ivol, int nrep_old) {
    // reset vol_visited before calling first time
    auto vol      = volumes[ivol];
    int nchildren = vol->GetDaughters().size();
    // first reset visited flag for all daughters
    for (int i = 0; i < nchildren; ++i) {
      auto pvol              = vol->GetDaughters().operator[](i);
      int idaughter          = pvol->GetLogicalVolume()->id();
      vol_visited[idaughter] = false;
    }
    // visit the children sub-tree once per child
    for (int i = 0; i < nchildren; ++i) {
      auto pvol     = vol->GetDaughters().operator[](i);
      int idaughter = pvol->GetLogicalVolume()->id();
      if (!vol_visited[idaughter] && nrep[idaughter] > 1) {
        vol_visited[idaughter] = true;
        // Recompute the number of repetitions for this daughter
        int nrep_daughter_old = nrep[idaughter];
        nrep[idaughter]       = 0;
        int nrep_ref          = 0;
        auto &vec_parents     = nrep_parent[idaughter];
        for (auto &reps : vec_parents) {
          int iparent = reps.first;
          nrep_ref += reps.second;
          if (iparent == ivol) {
            // Avoid integer overflow
            double newvalue = double(reps.second) * nrep[ivol] / nrep_old;
            VECGEOM_ASSERT(std::floor(newvalue) == newvalue);
            reps.second = int(newvalue);
            // reps.second = reps.second * nrep[ivol] / nrep_old;
          }
          nrep[idaughter] += reps.second;
        }
        VECGEOM_ASSERT(nrep_daughter_old == nrep_ref);
        // recompute daughter score
        score[idaughter] = nleaves[idaughter] * (nrep[idaughter] - 1);
        // propagate to children
        recomputeDaughterScores(idaughter, nrep_daughter_old);
      }
    }
    // reset again visited flag for all daughters because they may appear in separate branches
    for (int i = 0; i < nchildren; ++i) {
      auto pvol              = vol->GetDaughters().operator[](i);
      int idaughter          = pvol->GetLogicalVolume()->id();
      vol_visited[idaughter] = false;
    }
  };

  /// Recomputes the score for parents after compressing a volume
  typedef std::function<void(int, int)> funcCompressAndRecomputeParentScores_t;
  funcCompressAndRecomputeParentScores_t compressAndRecomputeParentScores = [&](int ivol, int new_nnodes) {
    // reduce the score of volume ivol
    int old_nnodes = nleaves[ivol];
    int reduction  = old_nnodes - new_nnodes;
    VECGEOM_ASSERT(reduction > 0);
    nleaves[ivol]     = new_nnodes;
    score[ivol]       = (nrep[ivol] - 1) * new_nnodes;
    auto &vec_parents = nrep_parent[ivol];
    for (auto &reps : vec_parents) {
      int iparent = reps.first;
      if (vol_selected[iparent]) continue;
      int nreduced = reduction * (reps.second / nrep[iparent]);
      int nparent  = nleaves[iparent] - nreduced;
      VECGEOM_ASSERT(nparent >= 0);
      // Propagate to parents
      compressAndRecomputeParentScores(iparent, nparent);
    }
  };

  /// This visitor fills volumes[], nrep[], nleaves[], levels[] and parents[] and should be called once for the top
  typedef std::function<void(VPlacedVolume const *, int &, int &, int &, int &)> funcCountReducedSize_t;
  funcCountReducedSize_t visitAndCountReducedSize = [&](VPlacedVolume const *pvol, int &count, int &count_selected,
                                                        int &level, int &scene_sum) {
    // reset vol_visited before calling first time
    count++;
    auto lvol = pvol->GetLogicalVolume();
    int ivol  = lvol->id();

    if (vol_selected[ivol]) {
      level++;
      measured_depth = std::max(measured_depth, level);
    }
    scene_sum += level;

    if (!vol_visited[ivol]) {
      // count once the content of all non-visited volumes
      vol_visited[ivol]     = true;
      int count_daughters   = 0;
      int count_scene_level = 0;
      int nchildren         = pvol->GetDaughters().size();
      for (int i = 0; i < nchildren; ++i) {
        visitAndCountReducedSize(pvol->GetDaughters().operator[](i), count_daughters, count_selected, level,
                                 count_scene_level);
      }
      scenes[ivol] = count_scene_level;
      scene_sum += count_scene_level;
      if (!vol_selected[ivol]) {
        VECGEOM_ASSERT(nleaves[ivol] == count_daughters);
        // nleaves[ivol] = count_daughters;
        count += count_daughters;
      } else {
        count_selected += count_daughters;
      }
    } else {
      // Count content of the volume only if not selected as scene
      if (!vol_selected[ivol]) count += nleaves[ivol];
      scene_sum += scenes[ivol];
    }
    if (vol_selected[ivol]) level--;
  };

  /// This visitor fills volumes[], nrep[], nleaves[], levels[] and parents[] and should be called once for the top
  typedef std::function<void(VPlacedVolume const *, ReferenceNavState *, int &)> funcFillRepetitions_t;
  funcFillRepetitions_t visitAndFillRepetitions = [&](VPlacedVolume const *pvol, ReferenceNavState *state, int &count) {
    // reset vol_visited before calling first time
    count++;
    auto parent = state->Top();
    state->Push(pvol);
    auto lvol = pvol->GetLogicalVolume();
    int ivol  = lvol->id();
    // count repetitions of the logical volume
    nrep[ivol]++;
    if (parent) getNrep(ivol, parent->GetLogicalVolume()->id(), 1);
    levels[ivol] += state->GetLevel();
    int countold  = count;
    int nchildren = pvol->GetDaughters().size();
    for (int i = 0; i < nchildren; ++i) {
      visitAndFillRepetitions(pvol->GetDaughters().operator[](i), state, count);
    }
    if (!vol_visited[ivol]) {
      volumes[ivol]     = lvol;
      vol_visited[ivol] = true;
      nleaves[ivol]     = count - countold;
    }
    state->Pop();
  };

  ReferenceNavState state;
  visitAndFillRepetitions(GeoManager::Instance().GetWorld(), &state, nnodes);
  if (nnodes < min_per_scene) return;
  // now fill score for each volume
  for (auto lvol : volumes) {
    if (!lvol) continue;
    int ivol = lvol->id();
    // if (levels[ivol] % nrep[ivol] != 0) printf(" *** %d: different levels\n", ivol);
    levels[ivol] /= nrep[ivol];
    score[ivol] = nleaves[ivol] * (nrep[ivol] - 1);
    // check consistency
#ifdef SCENEMAKE_DEBUG
    checkConsistency(ivol);
#endif
  }

  int estimated_depth = 0;
  int min_score       = (min_per_scene > 0) ? min_per_scene : 1000;
  int total_score     = 0;
  int scene_sum       = 0;
  while (1) {
    // Find and select volume with maximum score
    int score_max = 0;
    int imax      = -1;
    int lvl       = 0;
    int level_max = 0;
    // sorting volumes by their score would mitigate the linear complexity, but the scores of parents change
    // after selecting a volume, so we would need sorting after each selection. A smarter algorithm allowing to
    // reinsert scores in the sorted array?
    for (int i = 0; i < ntot; ++i) {
      if (vol_selected[i] || score[i] < score_max || score[i] < min_score ||
          volumes[i]->GetUnplacedVolume()->GetType() == ESolidType::boolean)
        continue;

      lvl         = 0;
      found_level = 0;
      getMotherSceneLevel(i, lvl, found_level);
      found_level++;
      lvl = found_level;
      if (found_level > max_depth) continue;
      std::fill(vol_visited.begin(), vol_visited.end(), false);
      getDaughterSceneLevel(i, lvl, found_level);
      if (found_level > max_depth) continue;
      score_max       = score[i];
      level_max       = found_level;
      imax            = i;
      estimated_depth = std::max(estimated_depth, level_max);
    }
    // Check if level done
    if (imax < 0) break;

    int nrep_max       = nrep[imax];
    nrep[imax]         = 1;
    vol_selected[imax] = 1;

    int last_score = score[imax];
    std::fill(vol_visited.begin(), vol_visited.end(), false);
    compressAndRecomputeParentScores(imax, 0);
    recomputeDaughterScores(imax, nrep_max);
    total_score += last_score;
#ifdef SCENEMAKE_DEBUG
    VECGEOM_LOG(debug) << imax << ": " << volumes[imax]->GetName() << "  nrep = " << nrep_max
                       << "  parents = " << nrep_parent[imax].size() << " level = " << levels[imax]
                       << " scene_level = " << estimated_depth << "  score = " << last_score
                       << " total_score = " << total_score << std::endl;

    int depth       = 0;
    scene_sum       = 0;
    measured_nnodes = 0;
    measured_depth  = 0;
    count_selected  = 0;
    std::fill(vol_visited.begin(), vol_visited.end(), false);
    visitAndCountReducedSize(GeoManager::Instance().GetWorld(), measured_nnodes, count_selected, depth, scene_sum);
    VECGEOM_LOG(debug) << "Measured: depth = " << measured_depth << "  nnodes = " << measured_nnodes
                       << "   selected_count = " << count_selected
                       << " score = " << nnodes - measured_nnodes - count_selected
                       << "  avg_scene_level = " << float(scene_sum) / nnodes << "\n";
#endif
    if (nnodes - total_score <= min_per_scene) break;
  }

  int depth       = 0;
  scene_sum       = 0;
  measured_nnodes = 0;
  measured_depth  = 0;
  count_selected  = 0;
  std::fill(vol_visited.begin(), vol_visited.end(), false);
  visitAndCountReducedSize(GeoManager::Instance().GetWorld(), measured_nnodes, count_selected, depth, scene_sum);
  if (log_summary) {
    VECGEOM_LOG(info) << "selected_volumes = " << std::accumulate(vol_selected.begin(), vol_selected.end(), int(0))
                      << "  node_count = " << nnodes - total_score << " (" << nnodes
                      << " initial)  max_scenes = " << estimated_depth;
  }
  // << "  * measured: max_scenes = " << measured_depth
  // << "  node_count = " << measured_nnodes + count_selected
  // << "  avg_scene_level = " << float(scene_sum) / nnodes << "\n";
}

bool NavIndexTable::AllocateTable(size_t bytes)
{
  size_t nelem       = bytes / sizeof(NavIndex_t) + 1;
  NavIndex_t *buffer = new (std::nothrow) NavIndex_t[nelem];
  if (buffer) {
    fNavInd    = buffer;
    fNavInd[0] = 0;
    fTableSize = bytes;
  } else {
    std::cerr << "=== EEE === AlocateTable bad_alloc intercepted while trying to allocate " << bytes << " bytes\n";
  }
  return buffer != nullptr;
}

bool NavIndexTable::CreateTable(VPlacedVolume const *top, int maxdepth, int depth_limit, int min_per_scene)
{
  fDepthLimit = depth_limit;
  (void)maxdepth;
  ReferenceNavState state;
  BuildNavIndexVisitor visitor(depth_limit, true); // just count table size
  NavIndex_t id = 1;
#ifdef VECGEOM_USE_NAVTUPLE
  visitor.NodeReduction(min_per_scene);
  int scene_id = 0;
  VECGEOM_LOG(info) << "=== creating navigation table with min_per_scene " << min_per_scene;
  visitAllPlacedVolumesNavTuple(top, &visitor, &state, id, scene_id, scene_id);
  visitor.ResetVisited();
#else
  visitAllPlacedVolumesNavIndex(top, &visitor, &state, id);
#endif
  const auto table_size = visitor.GetTableSize();
  const auto warn_limit = NavTableWarningLimitBytes();
  VECGEOM_LOG(info) << "navigation table size is " << std::setprecision(5) << BytesToMBytes(table_size) << " MBytes";
#ifdef VECGEOM_USE_NAVTUPLE
  ReportTupleNavTableGuidance(top, depth_limit, min_per_scene, VECGEOM_NAVTUPLE_MAXDEPTH, table_size, warn_limit);
#else
  WarnLargeNavTable("NavStateIndex", table_size, warn_limit, GeoManager::Instance().GetTotalNodeCount(),
                    "Consider -DVECGEOM_NAV=tuple for large repeated geometries.");
#endif

  bool has_table = AllocateTable(table_size);
  if (!has_table) return false;

  visitor.SetTable(fNavInd);
  visitor.SetDoCount(false);

  id = 1;
#ifdef VECGEOM_USE_NAVTUPLE
  scene_id = 0;
  visitAllPlacedVolumesNavTuple(top, &visitor, &state, id, scene_id, scene_id);
  // printf("scene_id_last = %d\n", scene_id);
#else
  visitAllPlacedVolumesNavIndex(top, &visitor, &state, id);
#endif
  return true;
}

bool NavIndexTable::Validate(VPlacedVolume const *top, int maxdepth) const
{
  (void)maxdepth;
  int error = 0;
  // Validation is a dedicated post-build pass: start from a reference state
  // containing the world volume and walk the geometry tree in lockstep with
  // the encoded navigation representation stored in the table.
  auto reference = ReferenceNavState::MakeWorld(top);
#ifdef VECGEOM_USE_NAVTUPLE
  int ierr = ValidateNavTupleRecursive(top, reference, NavTuple_t{fWorld}, error);
#else
  int ierr = ValidateNavIndexRecursive(top, reference, fWorld, error);
#endif
  if (ierr > 0) return false;
  return true;
}

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom
