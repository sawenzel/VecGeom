// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// \file management/NavIndexTableLayout.h
/// \brief Field offsets and helper routines for encoded navigation table records.

#ifndef VECGEOM_MANAGEMENT_NAVINDEXTABLELAYOUT_H_
#define VECGEOM_MANAGEMENT_NAVINDEXTABLELAYOUT_H_

#include "VecGeom/base/Global.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {
namespace NavIndexTableLayout {

/// \brief Byte-level alignment helper for `Precision` data stored in a `NavIndex_t` table.
VECCORE_ATT_HOST_DEVICE
VECGEOM_FORCE_INLINE
bool NeedsPrecisionPadding(NavIndex_t element) { return (element * sizeof(NavIndex_t)) % sizeof(Precision) != 0; }

/**
 * @brief Layout of the expanded navigation table used by `NavStateIndex`.
 *
 * @details Each touchable has one record in the table, addressed by a
 * `NavIndex_t`. The record starts with parent, touchable id, placed-volume id,
 * child id, logical-volume id, and one packed metadata word. The daughter block
 * follows immediately after these fixed fields and stores the navigation index
 * of each daughter touchable. Optional padding aligns the following cached
 * transformation block to `Precision` alignment.
 *
 * The packed metadata word stores the global level and transformation flags in
 * byte-sized fields, followed by the daughter count as an unsigned short. The
 * daughter count is therefore limited by the range of that field.
 *
 * For the rationale behind this representation and its tradeoffs with the
 * tuple table, see @ref navigation_state_tables.
 */
namespace Index {

/// \brief `NavStateIndex` record fields, expressed as `NavIndex_t` offsets from the record start.
static constexpr NavIndex_t kParent        = 0;
static constexpr NavIndex_t kTouchableId   = 1;
static constexpr NavIndex_t kPlacedVolume  = 2;
static constexpr NavIndex_t kChildId       = 3;
static constexpr NavIndex_t kLogicalVolume = 4;
static constexpr NavIndex_t kPacked        = 5;
static constexpr NavIndex_t kDaughters     = 6;

/// \brief Byte offsets inside the packed `NavStateIndex` field at `kPacked`.
static constexpr unsigned int kLevelByte         = 0;
static constexpr unsigned int kMatrixFlagsByte   = 1;
static constexpr unsigned int kDaughterCountByte = 2;

/// \brief Bits stored in the `kMatrixFlagsByte` byte.
static constexpr unsigned char kHasRotationFlag     = 0x01;
static constexpr unsigned char kHasTranslationFlag  = 0x02;
static constexpr unsigned char kHasStoredMatrixFlag = 0x04;

/// \brief Number of `Precision` values stored for a cached `NavStateIndex` transform.
static constexpr unsigned int kStoredTransformPrecisionCount = 12;

/// \brief Returns the first table element after the daughter-index block, before alignment padding.
VECCORE_ATT_HOST_DEVICE
VECGEOM_FORCE_INLINE
NavIndex_t TransformBase(NavIndex_t record, unsigned int ndaughters)
{
  return record + kDaughters + ndaughters + ((ndaughters + 1) & 1);
}

/// \brief Returns the aligned table element where cached transform data starts.
VECCORE_ATT_HOST_DEVICE
VECGEOM_FORCE_INLINE
NavIndex_t TransformStart(NavIndex_t record, unsigned int ndaughters)
{
  auto start = TransformBase(record, ndaughters);
  start += unsigned{NeedsPrecisionPadding(start)};
  return start;
}

} // namespace Index

/**
 * @brief Layout of the scene-compressed navigation table used by `NavStateTuple`.
 *
 * @details The tuple representation separates touchable records from shared
 * logical-volume records. A touchable record stores the parent within the
 * current scene, placed-volume id, child id, touchable id, logical-record
 * address, packed scene ids, and packed metadata. Optional transform data can
 * follow these fixed fields. The logical-volume record stores the compact
 * logical-volume id, daughter count, and daughter touchable-record addresses.
 *
 * A `NavStateTuple` stores one scene-local touchable index per active scene.
 * When traversal enters a new scene, a new tuple component is pushed instead of
 * expanding the repeated subtree for every parent context. The scene field
 * stores the parent scene id and the current scene id as two unsigned-short
 * halves.
 *
 * For the rationale behind this representation and its tradeoffs with the
 * expanded index table, see @ref navigation_state_tables.
 */
namespace Tuple {

/// \brief `NavStateTuple` touchable-record fields, expressed as `NavIndex_t` offsets from the record start.
static constexpr NavIndex_t kParent                      = 0;
static constexpr NavIndex_t kPlacedVolume                = 1;
static constexpr NavIndex_t kChildId                     = 2;
static constexpr NavIndex_t kTouchableId                 = 3;
static constexpr NavIndex_t kLogicalRecord               = 4;
static constexpr NavIndex_t kScenes                      = 5;
static constexpr NavIndex_t kPacked                      = 6;
static constexpr NavIndex_t kRecordFieldsBeforeTransform = 7;

/// \brief Half-word offsets inside the packed scene field at `kScenes`.
static constexpr unsigned int kParentSceneHalf  = 0;
static constexpr unsigned int kCurrentSceneHalf = 1;

/// \brief Byte offsets inside the packed tuple field at `kPacked`.
static constexpr unsigned int kLevelByte           = 0;
static constexpr unsigned int kTransformOffsetByte = 1;
static constexpr unsigned int kHasTranslationByte  = 2;
static constexpr unsigned int kHasRotationByte     = 3;

/// \brief Logical-volume record fields, expressed as `NavIndex_t` offsets from that logical record start.
static constexpr NavIndex_t kLogicalVolumeId = 0;
static constexpr NavIndex_t kDaughterCount   = 1;
static constexpr NavIndex_t kDaughters       = 2;

} // namespace Tuple

} // namespace NavIndexTableLayout
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_MANAGEMENT_NAVINDEXTABLELAYOUT_H_
