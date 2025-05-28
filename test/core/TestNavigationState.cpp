#include "VecGeom/navigation/NavStatePath.h"
#include "VecGeom/base/Global.h"
#include <iostream>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "VecGeom/base/Assert.h"
using namespace vecgeom;

void NavStateUnitTest1()
{
  using NavigationState   = NavStatePath;
  NavigationState *state1 = NavigationState::MakeInstance(10);
  NavigationState *state2 = NavigationState::MakeInstance(10);
  // test - 0 ( one empty path )
  state1->Clear();
  state2->Clear();
  state2->PushIndexType(1);
  VECGEOM_ASSERT(state1->Distance(*state2) == 1);

  // test - 1 ( equal paths )
  state1->Clear();
  state2->Clear();
  state1->PushIndexType(1);
  state2->PushIndexType(1);
  VECGEOM_ASSERT(state1->RelativePath(*state2).compare("") == 0);
  VECGEOM_ASSERT(state1->Distance(*state2) == 0);
  std::cerr << state1->RelativePath(*state2) << "\n";

  // test - 2
  state1->Clear();
  state2->Clear();
  state1->PushIndexType(1);
  state2->PushIndexType(1);
  state2->PushIndexType(2);
  VECGEOM_ASSERT(state1->RelativePath(*state2).compare("/down/2") == 0);
  VECGEOM_ASSERT(state1->Distance(*state2) == 1);
  std::cerr << state1->RelativePath(*state2) << "\n";

  // test - 3
  state1->Clear();
  state2->Clear();
  state1->PushIndexType(1);
  state2->PushIndexType(1);
  state2->PushIndexType(2);
  state2->PushIndexType(4);
  std::cerr << state1->RelativePath(*state2) << "\n";
  std::cerr << state1->Distance(*state2) << "\n";
  VECGEOM_ASSERT(state1->RelativePath(*state2).compare("/down/2/down/4") == 0);
  VECGEOM_ASSERT(state1->Distance(*state2) == 2);

  // test - 4
  state1->Clear();
  state2->Clear();
  state1->PushIndexType(1);
  state1->PushIndexType(2);
  state1->PushIndexType(2);
  state2->PushIndexType(1);
  std::cerr << "HUHU " << state1->Distance(*state2) << "\n";
  VECGEOM_ASSERT(state1->Distance(*state2) == 2);
  VECGEOM_ASSERT(state1->RelativePath(*state2).compare("/up/up") == 0);
  std::cerr << state1->RelativePath(*state2) << "\n";

  // test - 5
  state1->Clear();
  state2->Clear();
  state1->PushIndexType(1);
  state1->PushIndexType(1);
  state1->PushIndexType(2);
  state1->PushIndexType(2);
  state2->PushIndexType(1);
  state2->PushIndexType(1);
  state2->PushIndexType(5);
  state2->PushIndexType(1);
  std::cerr << state1->RelativePath(*state2) << "\n";
  VECGEOM_ASSERT(state1->RelativePath(*state2).compare("/up/horiz/3/down/1") == 0);
  VECGEOM_ASSERT(state1->Distance(*state2) == 4);

  // test - 6
  state1->Clear();
  state2->Clear();
  state1->PushIndexType(1);
  state1->PushIndexType(1);
  state1->PushIndexType(2);
  state1->PushIndexType(2);
  state1->PushIndexType(3);

  state2->PushIndexType(1);
  state2->PushIndexType(1);
  state2->PushIndexType(5);
  state2->PushIndexType(1);
  state2->PushIndexType(1);
  std::cerr << state1->RelativePath(*state2) << "\n";
  VECGEOM_ASSERT(state1->RelativePath(*state2).compare("/up/up/horiz/3/down/1/down/1") == 0);
  VECGEOM_ASSERT(state1->Distance(*state2) == 6);
}

int main()
{
  NavStateUnitTest1();
  return 0;
}
