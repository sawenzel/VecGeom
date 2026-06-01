// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// \brief Lightweight reference state for validating encoded navigation data.
/// \file management/ReferenceNavState.h
/// \author OpenAI Codex

#ifndef VECGEOM_MANAGEMENT_REFERENCENAVSTATE_H_
#define VECGEOM_MANAGEMENT_REFERENCENAVSTATE_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/base/Transformation3D.h"
#include "VecGeom/volumes/PlacedVolume.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

/**
 * @brief Error codes returned when validating encoded navigation data against
 *        geometry truth.
 *
 * The numeric values intentionally match the historical validation diagnostics
 * used by the navigation-table checks so that existing callers can continue to
 * treat the values as stable integer error codes
 */
enum class ReferenceNavValidationError : int {
  kNone                   = 0, ///< Validation succeeded.
  kIncompatibleDaughter   = 1, ///< The geometry daughter pointer cannot be used for a valid descent.
  kIncompatibleScene      = 2, ///< The encoded child state stayed in the wrong scene.
  kTopVolumeMismatch      = 3, ///< The encoded top volume does not match the geometry traversal.
  kChildIdMismatch        = 4, ///< The encoded child id does not match the placed volume.
  kLogicalIdMismatch      = 5, ///< The encoded logical-volume id does not match the placed volume.
  kLevelMismatch          = 6, ///< The encoded depth does not match the traversal depth.
  kPushPopMismatch        = 7, ///< A Pop followed by Push does not reconstruct the encoded state.
  kDaughterCountMismatch  = 8, ///< The encoded number of daughters does not match the logical volume.
  kTransformationMismatch = 9  ///< The encoded top transformation does not match the reference transform.
};

/**
 * @brief Returns a short textual description for a validation error code.
 * @param error Validation error code.
 * @return Human-readable description suitable for host logs and test diagnostics.
 */
inline char const *ToString(ReferenceNavValidationError error)
{
  switch (error) {
  case ReferenceNavValidationError::kNone:
    return "success";
  case ReferenceNavValidationError::kIncompatibleDaughter:
    return "incompatible daughter pointer";
  case ReferenceNavValidationError::kIncompatibleScene:
    return "incompatible scene index";
  case ReferenceNavValidationError::kTopVolumeMismatch:
    return "top placed volume pointer mismatch";
  case ReferenceNavValidationError::kChildIdMismatch:
    return "top placed volume child id mismatch";
  case ReferenceNavValidationError::kLogicalIdMismatch:
    return "logical volume id mismatch";
  case ReferenceNavValidationError::kLevelMismatch:
    return "level mismatch";
  case ReferenceNavValidationError::kPushPopMismatch:
    return "navigation index inconsistency for Push/Pop";
  case ReferenceNavValidationError::kDaughterCountMismatch:
    return "number of daughters mismatch";
  case ReferenceNavValidationError::kTransformationMismatch:
    return "transformation matrix mismatch";
  }
  return "unknown validation error";
}

/**
 * @brief Compact host/device reference state derived directly from the geometry tree.
 *
 * The helper stores only the explicit touchable path:
 * - `fPath[0]` is the world placed volume when the state is non-empty,
 * - `fPath[fCurrentLevel - 1]` is the current top placed volume,
 * - `fCurrentLevel` is the number of valid entries in `fPath`, not the
 *   zero-based depth.
 *
 * Consequently:
 * - an empty state has `fCurrentLevel == 0`,
 * - the world-only state has `fCurrentLevel == 1` and `GetLevel() == 0`,
 * - a daughter of the world has `fCurrentLevel == 2` and `GetLevel() == 1`.
 *
 * The helper does not cache or incrementally update any derived transform.
 * Instead, `TopMatrix()` reconstructs the full global-to-top transform from the
 * stored placed-volume sequence each time it is requested. This keeps the
 * validation oracle tied to the explicit path data and avoids a second,
 * independent transform-accumulation implementation.
 */
class ReferenceNavState {
private:
  static constexpr unsigned int kMaxPathEntries = 256; ///< Supports geometry depths from 0 to 255.

  VPlacedVolume const *fPath[kMaxPathEntries] = {}; ///< Touchable path, with world at index 0 when non-empty.
  unsigned short fCurrentLevel                = 0;  ///< Count of valid entries currently stored in @ref fPath.

public:
  /// @brief Constructs an empty reference state.
  VECCORE_ATT_HOST_DEVICE
  ReferenceNavState() = default;

  /**
   * @brief Creates the reference state for the world volume.
   * @param world Top placed volume of the geometry tree.
   * @return Reference state whose path contains exactly the world volume.
   *
   * If @p world is non-null, the returned state satisfies:
   * - `Top() == world`
   * - `GetLevel() == 0`
   * - `fCurrentLevel == 1`
   *
   * This mirrors the standard VecGeom interpretation that the world touchable
   * lives at depth 0.
   */
  VECCORE_ATT_HOST_DEVICE
  static ReferenceNavState MakeWorld(VPlacedVolume const *world)
  {
    ReferenceNavState state;
    if (world) {
      state.fPath[0]      = world;
      state.fCurrentLevel = 1;
    }
    return state;
  }

  /**
   * @brief Pushes a placed volume onto the reference stack.
   * @param volume Placed volume to append.
   *
   * Callers are expected to push volumes in traversal order, meaning the
   * parent is already the current top and @p volume is its direct daughter.
   * The helper does not verify family relationships; it simply records the
   * explicit path presented by the caller.
   */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void Push(VPlacedVolume const *volume)
  {
    VECGEOM_VALIDATE(volume != nullptr, << "ReferenceNavState cannot push a null placed volume");
    VECGEOM_VALIDATE(fCurrentLevel < kMaxPathEntries,
                     << "ReferenceNavState does not support geometry depths beyond 255");
    fPath[fCurrentLevel++] = volume;
  }

  /**
   * @brief Pops the current top placed volume from the reference stack.
   *
   * This removes the last path entry and restores the previous top volume.
   * Popping the world entry returns the helper to the empty-state representation.
   */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void Pop()
  {
    VECGEOM_VALIDATE(fCurrentLevel > 0, << "ReferenceNavState cannot pop an empty path");
    fPath[--fCurrentLevel] = nullptr;
  }

  /// @brief Returns the placed volume at the end of the current reference path.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  VPlacedVolume const *Top() const { return (fCurrentLevel > 0) ? fPath[fCurrentLevel - 1] : nullptr; }

  /// @brief Returns the placed volume stored at a given path index.
  ///
  /// The index is a path position, not a daughter id:
  /// - `At(0)` is the world when the state is non-empty
  /// - `At(GetLevel())` is the same volume as `Top()`
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  VPlacedVolume const *At(unsigned char level) const { return (level < fCurrentLevel) ? fPath[level] : nullptr; }

  /// @brief Returns the zero-based depth of the current top volume.
  ///
  /// This is `fCurrentLevel - 1` for any non-empty state, so the world has
  /// level 0, its daughters level 1, and so on.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  unsigned char GetLevel() const { return (fCurrentLevel > 0) ? static_cast<unsigned char>(fCurrentLevel - 1) : 0; }

  /// @brief Returns whether the helper is not currently anchored on a placed volume.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsOutside() const { return fCurrentLevel == 0; }

  /**
   * @brief Reconstructs the global-to-top transform from the stored volume stack.
   * @param matrix Destination transform receiving the global-to-top transform.
   *
   * The reconstruction rule is intentionally explicit:
   * - start from the identity transform,
   * - ignore `fPath[0]` because the world entry does not contribute a local
   *   placement transform,
   * - then multiply the daughter transforms in reverse path order:
   *   `fPath[fCurrentLevel - 1]`, `fPath[fCurrentLevel - 2]`, ..., `fPath[1]`.
   *
   * In other words, for a path `[world, a, b, c]`, the resulting transform is
   * reconstructed as:
   * `T_c * T_b * T_a`
   *
   * This is the exact convention required by the encoded navigation-state
   * validation in this file and matches the way VecGeom reconstructs the
   * global-to-local transform from an explicit touchable path.
   */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void TopMatrix(Transformation3D &matrix) const
  {
    matrix.Clear();
    for (int i = fCurrentLevel - 1; i > 0; --i)
      matrix *= *(fPath[i]->GetTransformation());
  }

  /// @brief Prints the current reference path for diagnostics.
  VECCORE_ATT_HOST_DEVICE
  void Print() const
  {
#ifndef VECCORE_CUDA
    printf("ReferenceNavState: level=%u, path=<", unsigned(GetLevel()));
    for (unsigned int i = 0; i < fCurrentLevel; ++i)
      printf("/%s", fPath[i] ? fPath[i]->GetLabel().c_str() : "NULL");
    printf(">\n");
#else
    printf("ReferenceNavState: level=%u, topVol=<%p>\n", unsigned(GetLevel()), Top());
#endif
  }
};

/**
 * @brief Validates an encoded navigation state against the current reference traversal point.
 *
 * @tparam EncodedNavState Navigation-state class providing the static `...Impl`
 *         validation interface, for example `NavStateIndex` or `NavStateTuple`.
 * @tparam EncodedState Encoded state handle type used by @p EncodedNavState.
 * @param reference Geometry-truth reference state for the current touchable.
 * @param encoded_state Encoded state to be validated.
 * @return Validation result code.
 */
template <typename EncodedNavState, typename EncodedState>
VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE ReferenceNavValidationError
ValidateEncodedState(ReferenceNavState const &reference, EncodedState encoded_state)
{
  if (reference.IsOutside()) return ReferenceNavValidationError::kNone;

  if (EncodedNavState::TopImpl(encoded_state) != reference.Top())
    return ReferenceNavValidationError::kTopVolumeMismatch;

  if (EncodedNavState::GetChildIdImpl(encoded_state) != reference.Top()->GetChildId())
    return ReferenceNavValidationError::kChildIdMismatch;

  if (EncodedNavState::GetLogicalIdImpl(encoded_state) != reference.Top()->GetLogicalVolume()->id())
    return ReferenceNavValidationError::kLogicalIdMismatch;

  if (EncodedNavState::GetLevelImpl(encoded_state) != reference.GetLevel())
    return ReferenceNavValidationError::kLevelMismatch;

  auto roundtrip_state = encoded_state;
  if (reference.GetLevel() > 0) {
    EncodedNavState::PopImpl(roundtrip_state);
    EncodedNavState::PushImpl(roundtrip_state, reference.Top());
    if (roundtrip_state != encoded_state) return ReferenceNavValidationError::kPushPopMismatch;
  }

  if (EncodedNavState::GetNdaughtersImpl(encoded_state) != reference.Top()->GetDaughters().size())
    return ReferenceNavValidationError::kDaughterCountMismatch;

  Transformation3D reference_matrix;
  Transformation3D encoded_matrix;
  reference.TopMatrix(reference_matrix);
  EncodedNavState::TopMatrixImpl(encoded_state, encoded_matrix);
  if (!reference_matrix.ApproxEqual(encoded_matrix)) return ReferenceNavValidationError::kTransformationMismatch;

  return ReferenceNavValidationError::kNone;
}

/**
 * @brief Validates that a child encoded state moved into the expected scene.
 *
 * The check is meaningful for scene-aware encoded states such as
 * `NavStateTuple`. For scene-less states such as `NavStateIndex`, the static
 * `GetSceneIdImpl` interface simply reports no active scene and the validation
 * succeeds.
 *
 * @tparam EncodedNavState Navigation-state class providing the static scene
 *         accessors.
 * @tparam EncodedState Encoded state handle type used by @p EncodedNavState.
 * @param parent_state Encoded state before descending to the child.
 * @param child_state Encoded state after descending to the child.
 * @return Validation result code.
 */
template <typename EncodedNavState, typename EncodedState>
VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE ReferenceNavValidationError
ValidateSceneTransition(EncodedState parent_state, EncodedState child_state)
{
  unsigned short parent_scene = 0, parent_new_scene = 0;
  bool inside_scene          = EncodedNavState::GetSceneIdImpl(parent_state, parent_scene, parent_new_scene);
  unsigned short child_scene = 0, child_new_scene = 0;
  EncodedNavState::GetSceneIdImpl(child_state, child_scene, child_new_scene);
  if (inside_scene && child_scene == parent_scene) return ReferenceNavValidationError::kIncompatibleScene;
  return ReferenceNavValidationError::kNone;
}

/**
 * @brief Prints detailed diagnostics for a failed encoded-state validation.
 *
 * The output includes both the observed encoded value and the geometry-truth
 * value carried by the reference state so that callers can immediately see what
 * the validator expected.
 *
 * @tparam EncodedNavState Navigation-state class providing the static `...Impl`
 *         validation interface.
 * @tparam EncodedState Encoded state handle type used by @p EncodedNavState.
 * @param error Validation error code to describe.
 * @param reference Geometry-truth reference state for the current touchable.
 * @param encoded_state Encoded state that failed validation.
 */
template <typename EncodedNavState, typename EncodedState>
VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE void PrintValidationFailure(ReferenceNavValidationError error,
                                                                         ReferenceNavState const &reference,
                                                                         EncodedState encoded_state)
{
  printf("=== EEE === Validation detail: error code %d\n", static_cast<int>(error));
  switch (error) {
  case ReferenceNavValidationError::kTopVolumeMismatch: {
    auto const expected_top_id = reference.Top() ? static_cast<long long>(reference.Top()->id()) : -1LL;
    printf("    expected top placed volume id %lld, got %d\n", expected_top_id,
           EncodedNavState::TopIdImpl(encoded_state));
    break;
  }
  case ReferenceNavValidationError::kChildIdMismatch:
    printf("    expected child id %d, got %d\n", reference.Top()->GetChildId(),
           EncodedNavState::GetChildIdImpl(encoded_state));
    break;
  case ReferenceNavValidationError::kLogicalIdMismatch:
    printf("    expected logical volume id %d, got %u\n", reference.Top()->GetLogicalVolume()->id(),
           EncodedNavState::GetLogicalIdImpl(encoded_state));
    break;
  case ReferenceNavValidationError::kLevelMismatch:
    printf("    expected level %u, got %u\n", unsigned(reference.GetLevel()),
           unsigned(EncodedNavState::GetLevelImpl(encoded_state)));
    break;
  case ReferenceNavValidationError::kPushPopMismatch: {
    auto roundtrip_state = encoded_state;
    EncodedNavState::PopImpl(roundtrip_state);
    EncodedNavState::PushImpl(roundtrip_state, reference.Top());
    auto const expected_top_id = reference.Top() ? static_cast<long long>(reference.Top()->id()) : -1LL;
    printf("    expected Pop/Push to reconstruct the same encoded state for top id %lld\n", expected_top_id);
    printf("    original top id %d, roundtrip top id %d\n", EncodedNavState::TopIdImpl(encoded_state),
           EncodedNavState::TopIdImpl(roundtrip_state));
    break;
  }
  case ReferenceNavValidationError::kDaughterCountMismatch:
    printf("    expected daughter count %zu, got %u\n", reference.Top()->GetDaughters().size(),
           EncodedNavState::GetNdaughtersImpl(encoded_state));
    break;
  case ReferenceNavValidationError::kTransformationMismatch: {
    Transformation3D reference_matrix;
    Transformation3D encoded_matrix;
    reference.TopMatrix(reference_matrix);
    EncodedNavState::TopMatrixImpl(encoded_state, encoded_matrix);
    printf("    expected translation (%.8g, %.8g, %.8g), got (%.8g, %.8g, %.8g)\n", reference_matrix.Translation(0),
           reference_matrix.Translation(1), reference_matrix.Translation(2), encoded_matrix.Translation(0),
           encoded_matrix.Translation(1), encoded_matrix.Translation(2));
    break;
  }
  default:
    break;
  }
}

/**
 * @brief Prints detailed diagnostics for a failed scene transition.
 *
 * @tparam EncodedNavState Navigation-state class providing the static scene
 *         accessors.
 * @tparam EncodedState Encoded state handle type used by @p EncodedNavState.
 * @param parent_state Encoded state before descending to the child.
 * @param child_state Encoded state after descending to the child.
 * @param parent_volume Geometry parent used for the attempted descent.
 * @param child_volume Geometry child used for the attempted descent.
 */
template <typename EncodedNavState, typename EncodedState>
VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE void PrintSceneTransitionFailure(EncodedState parent_state,
                                                                              EncodedState child_state,
                                                                              VPlacedVolume const *parent_volume,
                                                                              VPlacedVolume const *child_volume)
{
  unsigned short parent_scene = 0, parent_new_scene = 0;
  unsigned short child_scene = 0, child_new_scene = 0;
  EncodedNavState::GetSceneIdImpl(parent_state, parent_scene, parent_new_scene);
  EncodedNavState::GetSceneIdImpl(child_state, child_scene, child_new_scene);
  auto const parent_id = parent_volume ? static_cast<long long>(parent_volume->id()) : -1LL;
  auto const child_id  = child_volume ? static_cast<long long>(child_volume->id()) : -1LL;
  printf("=== EEE === Validation detail: error code %d\n",
         static_cast<int>(ReferenceNavValidationError::kIncompatibleScene));
  printf("    expected child scene different from parent scene %u, got %u for %lld/%lld\n", parent_scene, child_scene,
         parent_id, child_id);
}

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_MANAGEMENT_REFERENCENAVSTATE_H_
