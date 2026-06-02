#include "VecGeom/management/GeoManager.h"
#include "VecGeom/base/Transformation3D.h"
#include "VecGeom/navigation/NavigationState.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/UnplacedBox.h"

#include <iostream>
#include <list>
#include <string>
#include <vector>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include "VecGeom/base/Assert.h"

using namespace vecgeom;

namespace {

VPlacedVolume const *BuildTestGeometry()
{
  auto &gm                          = GeoManager::Instance();
  static VPlacedVolume const *world = nullptr;
  if (world) return world;

  static std::vector<UnplacedBox *> solids;
  static std::vector<LogicalVolume *> volumes;

  constexpr int kDepth       = 6;
  constexpr int kNumChildren = 6;
  solids.reserve(kDepth);
  volumes.reserve(kDepth);

  for (int i = 0; i < kDepth; ++i) {
    solids.push_back(new UnplacedBox(10. - i, 10. - i, 10. - i));
    volumes.push_back(new LogicalVolume(("nav_test_" + std::to_string(i)).c_str(), solids.back()));
  }

  static std::vector<Transformation3D> placements;
  placements.reserve(kNumChildren);
  for (int i = 0; i < kNumChildren; ++i) {
    placements.emplace_back(Precision(i + 1), Precision(0), Precision(0));
  }

  for (int level = 0; level < kDepth - 1; ++level) {
    for (int child = 0; child < kNumChildren; ++child) {
      volumes[level]->PlaceDaughter(volumes[level + 1], &placements[child]);
    }
  }

  world = volumes.front()->Place();
  gm.SetWorld(const_cast<VPlacedVolume *>(world));
  gm.CloseGeometry();
  world = gm.GetWorld();
  return world;
}

std::string DownString(NavigationState const &state, int level)
{
  return "/down/" + std::to_string(state.ValueAt(level));
}

void ReleaseStates(NavigationState *state1, NavigationState *state2)
{
  delete state1;
  delete state2;
}

#ifdef VECGEOM_USE_NAVTUPLE
bool FindSceneState(VPlacedVolume const *world, NavigationState &state, std::vector<uint> &indices, int max_depth)
{
  std::list<uint> path(indices.begin(), indices.end());
  state.ResetPathFromListOfIndices(world, path);

  unsigned short scene_id = 0, new_scene_id = 0;
  if (state.GetSceneId(scene_id, new_scene_id) && new_scene_id != scene_id) return true;

  if (int(indices.size()) > max_depth) return false;
  auto *top = state.Top();
  if (!top) return false;

  auto const &daughters = top->GetDaughters();
  for (int id = 0; id < int(daughters.size()); ++id) {
    indices.push_back(id);
    if (FindSceneState(world, state, indices, max_depth)) return true;
    indices.pop_back();
  }
  return false;
}
#endif

} // namespace

void NavStateUnitTest1()
{
  auto *world = BuildTestGeometry();

  NavigationState *state1 = new NavigationState();
  NavigationState *state2 = new NavigationState();

  // test - 0
  // The current navigation states are always rooted in the geometry world, so
  // the closest equivalent to the historical "empty path versus one element"
  // check is "world versus one daughter below world".
  state1->ResetPathFromListOfIndices(world, std::list<uint>{0});
  state2->ResetPathFromListOfIndices(world, std::list<uint>{0, 1});
  VECGEOM_ASSERT(state1->Distance(*state2) == 1);

  // test - 1 ( equal paths )
  state1->ResetPathFromListOfIndices(world, std::list<uint>{0, 1});
  state2->ResetPathFromListOfIndices(world, std::list<uint>{0, 1});
  VECGEOM_ASSERT(state1->RelativePath(*state2).compare("") == 0);
  VECGEOM_ASSERT(state1->Distance(*state2) == 0);
  std::cerr << state1->RelativePath(*state2) << "\n";

  // test - 2
  state1->ResetPathFromListOfIndices(world, std::list<uint>{0, 1});
  state2->ResetPathFromListOfIndices(world, std::list<uint>{0, 1, 2});
  VECGEOM_ASSERT(state1->RelativePath(*state2).compare(DownString(*state2, 2)) == 0);
  VECGEOM_ASSERT(state1->Distance(*state2) == 1);
  std::cerr << state1->RelativePath(*state2) << "\n";

  // test - 3
  state1->ResetPathFromListOfIndices(world, std::list<uint>{0, 1});
  state2->ResetPathFromListOfIndices(world, std::list<uint>{0, 1, 2, 4});
  std::cerr << state1->RelativePath(*state2) << "\n";
  std::cerr << state1->Distance(*state2) << "\n";
  VECGEOM_ASSERT(state1->RelativePath(*state2).compare(DownString(*state2, 2) + DownString(*state2, 3)) == 0);
  VECGEOM_ASSERT(state1->Distance(*state2) == 2);

  // test - 4
  state1->ResetPathFromListOfIndices(world, std::list<uint>{0, 2, 2});
  state2->ResetPathFromListOfIndices(world, std::list<uint>{0});
  std::cerr << "HUHU " << state1->Distance(*state2) << "\n";
  VECGEOM_ASSERT(state1->Distance(*state2) == 2);
  VECGEOM_ASSERT(state1->RelativePath(*state2).compare("/up/up") == 0);
  std::cerr << state1->RelativePath(*state2) << "\n";

  // test - 5
  state1->ResetPathFromListOfIndices(world, std::list<uint>{0, 1, 1, 2, 2});
  state2->ResetPathFromListOfIndices(world, std::list<uint>{0, 1, 1, 5, 1});
  std::cerr << state1->RelativePath(*state2) << "\n";
  VECGEOM_ASSERT(state1->RelativePath(*state2).compare(
                     "/up/horiz/" + std::to_string(int(state2->ValueAt(3)) - int(state1->ValueAt(3))) +
                     DownString(*state2, 4)) == 0);
  VECGEOM_ASSERT(state1->Distance(*state2) == 4);

  // test - 6
  state1->ResetPathFromListOfIndices(world, std::list<uint>{0, 1, 1, 2, 2, 3});
  state2->ResetPathFromListOfIndices(world, std::list<uint>{0, 1, 1, 5, 1, 1});
  std::cerr << state1->RelativePath(*state2) << "\n";
  VECGEOM_ASSERT(state1->RelativePath(*state2).compare(
                     "/up/up/horiz/" + std::to_string(int(state2->ValueAt(3)) - int(state1->ValueAt(3))) +
                     DownString(*state2, 4) + DownString(*state2, 5)) == 0);
  VECGEOM_ASSERT(state1->Distance(*state2) == 6);

  ReleaseStates(state1, state2);
}

#ifdef VECGEOM_USE_NAVTUPLE
void NavStateTupleSceneSentinelTest()
{
  auto *world            = BuildTestGeometry();
  NavigationState *state = new NavigationState();

  std::vector<uint> indices{0};
  bool found_scene = FindSceneState(world, *state, indices, 6);
  VECGEOM_ASSERT(found_scene);

  unsigned short scene_id = 0, new_scene_id = 0;
  VECGEOM_ASSERT(state->GetSceneId(scene_id, new_scene_id));
  VECGEOM_ASSERT(new_scene_id != scene_id);

  NavigationState scene_top(*state);
  scene_top.PushScene(0);
  unsigned short sentinel_scene_id = 0, sentinel_new_scene_id = 0;
  VECGEOM_ASSERT(!scene_top.IsOutside());
  VECGEOM_ASSERT(scene_top.GetSceneId(sentinel_scene_id, sentinel_new_scene_id));
  VECGEOM_ASSERT(sentinel_scene_id == new_scene_id);
  VECGEOM_ASSERT(sentinel_new_scene_id == new_scene_id);
  VECGEOM_ASSERT(scene_top.GetLevel() == state->GetLevel());

  Transformation3D scene_volume_matrix;
  Transformation3D sentinel_top_matrix;
  Transformation3D sentinel_scene_matrix;
  Transformation3D sentinel_in_scene_matrix;
  state->TopMatrix(scene_volume_matrix);
  scene_top.TopMatrix(sentinel_top_matrix);
  scene_top.SceneMatrix(sentinel_scene_matrix);
  scene_top.TopInSceneMatrix(sentinel_in_scene_matrix);
  VECGEOM_ASSERT(sentinel_top_matrix.ApproxEqual(scene_volume_matrix));
  VECGEOM_ASSERT(sentinel_scene_matrix.ApproxEqual(scene_volume_matrix));
  VECGEOM_ASSERT(sentinel_in_scene_matrix.IsIdentity());

  delete state;
}
#endif

int main()
{
  NavStateUnitTest1();
#ifdef VECGEOM_USE_NAVTUPLE
  NavStateTupleSceneSentinelTest();
#endif
  return 0;
}
