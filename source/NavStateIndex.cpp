/// \file NavStateIndex.cpp
/// \author Andrei Gheata (andrei.gheata@cern.ch)
/// \date 14.05.2020

#include "VecGeom/navigation/NavStateIndex.h"

#include <iostream>
#include <list>
#include <sstream>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

// returning a "delta" transformation that can transform
// coordinates given in reference frame of this->Top() to the reference frame of other->Top()
// simply with otherlocalcoordinate = delta.Transform( thislocalcoordinate )
VECCORE_ATT_HOST_DEVICE
void NavStateIndex::DeltaTransformation(NavStateIndex const &other, Transformation3D &delta) const
{
  Transformation3D g2;
  Transformation3D g1;
  other.TopMatrix(g2);
  this->TopMatrix(g1);
  delta = g1.Inverse();
  // Trans/rot properties already correctly set
  // g2.SetProperties();
  // delta.SetProperties();
  delta.FixZeroes();
  delta.MultiplyFromRight(g2);
  delta.FixZeroes();
}

template <typename Real_t>
VECCORE_ATT_HOST_DEVICE void NavStateIndex::DeltaTransformation(NavStateIndex const &other,
                                                                Transformation3DMP<Real_t> &delta) const
{
  Transformation3DMP<Real_t> g2;
  Transformation3DMP<Real_t> g1;
  other.TopMatrix(g2);
  this->TopMatrix(g1);
  delta = g1.Inverse();
  // Trans/rot properties already correctly set
  // g2.SetProperties();
  // delta.SetProperties();
  delta.FixZeroes();
  delta.MultiplyFromRight(g2);
  delta.FixZeroes();
}

void NavStateIndex::GetPathAsListOfIndices(std::list<uint> &indices) const
{
  indices.clear();
  if (IsOutside()) return;

  auto nav_ind = fNavInd;
  while (nav_ind > 1) {
    auto pvol = TopImpl(nav_ind);
    indices.push_front(pvol->GetChildId());
    PopImpl(nav_ind);
  }
  // Paths start always with 0
  indices.push_front(0);
}

VECCORE_ATT_HOST_DEVICE
void NavStateIndex::PrintRecord(NavIndex_t nav_ind)
{
  if (nav_ind == 0) return;
  auto parent     = NavInd(nav_ind + NavIndexTableLayout::Index::kParent);
  auto placed_id  = NavInd(nav_ind + NavIndexTableLayout::Index::kPlacedVolume);
  auto child_id   = NavInd(nav_ind + NavIndexTableLayout::Index::kChildId);
  auto id         = NavInd(nav_ind + NavIndexTableLayout::Index::kTouchableId);
  auto logical_id = NavInd(nav_ind + NavIndexTableLayout::Index::kLogicalVolume);
  auto level      = GetLevelImpl(nav_ind);
  auto nd         = GetNdaughtersImpl(nav_ind);
  printf("| navind %u |+0| parent %u |+1| id %u |+2| placed_id %u |+3| child_id %u |+4| logical_id %u "
         "| level %u | nd %u ",
         nav_ind, parent, id, placed_id, child_id, logical_id, level, nd);
  printf("\n");
}

VECCORE_ATT_HOST_DEVICE
void NavStateIndex::Print(bool print_names) const
{
  if (IsOutside()) {
    printf("navInd=%u, id=%u, path=outside\n", fNavInd, GetId());
    return;
  }
  auto level = GetLevel();
  printf("navInd=%u, id=%u, level=%u/%u,  onBoundary=%s, path=<", fNavInd, GetId(), level, GetMaxLevel(),
         (fOnBoundary ? "true" : "false"));
  for (int i = 0; i < level + 1; ++i) {
#ifndef VECCORE_CUDA
    if (print_names) {
      auto vol = At(i);
      printf("/%s", vol ? vol->GetLabel().c_str() : "NULL");
    } else {
      auto nav_ind = GetNavIndexImpl(fNavInd, i);
      printf("/%u", nav_ind);
    }
#else
    auto nav_ind = GetNavIndexImpl(fNavInd, i);
    printf("/%u", fNavInd);
#endif
  }
  printf(">\n");
}

void NavStateIndex::ResetPathFromListOfIndices(VPlacedVolume const *world, std::list<uint> const &indices)
{
  // clear current nav state
  Clear();
  auto vol    = world;
  int counter = 0;
  for (auto id : indices) {
    if (counter > 0) vol = vol->GetDaughters().operator[](id);
    Push(vol);
    counter++;
  }
}

std::string NavStateIndex::RelativePath(NavStateIndex const &other) const
{
  int lastcommonlevel = -1;
  int thislevel       = GetLevel();
  int otherlevel      = other.GetLevel();
  int maxlevel        = Min(thislevel, otherlevel);
  std::stringstream str;
  //  algorithm: start on top and go down until paths split
  for (int i = 0; i < maxlevel + 1; i++) {
    if (this->At(i) == other.At(i)) {
      lastcommonlevel = i;
    } else {
      break;
    }
  }

  // paths are the same
  if (thislevel == lastcommonlevel && otherlevel == lastcommonlevel) {
    return std::string("");
  }

  // emit only ups
  if (thislevel > lastcommonlevel && otherlevel == lastcommonlevel) {
    for (int i = 0; i < thislevel - lastcommonlevel; ++i) {
      str << "/up";
    }
    return str.str();
  }

  // emit only downs
  if (thislevel == lastcommonlevel && otherlevel > lastcommonlevel) {
    for (int i = lastcommonlevel + 1; i <= otherlevel; ++i) {
      str << "/down";
      str << "/" << other.ValueAt(i);
    }
    return str.str();
  }

  // mixed case: first up; then down
  if (thislevel > lastcommonlevel && otherlevel > lastcommonlevel) {
    // emit ups
    int level = thislevel;
    for (; level > lastcommonlevel + 1; --level) {
      str << "/up";
    }

    level = lastcommonlevel + 1;
    // emit horiz ( exists when there is a turning point )
    int delta = other.ValueAt(level) - this->ValueAt(level);
    if (delta != 0) str << "/horiz/" << delta;

    level++;
    // emit downs with index
    for (; level <= otherlevel; ++level) {
      str << "/down/" << other.ValueAt(level);
    }
  }
  return str.str();
}

VECCORE_ATT_HOST_DEVICE
int NavStateIndex::Distance(NavStateIndex const &other) const
{
  int lastcommonlevel = -1;
  int thislevel       = GetLevel();
  int otherlevel      = other.GetLevel();
  int maxlevel        = Min(thislevel, otherlevel);

  //  algorithm: start on top and go down until paths split
  for (int i = 0; i < maxlevel + 1; i++) {
    if (this->At(i) == other.At(i)) {
      lastcommonlevel = i;
    } else {
      break;
    }
  }

  return (thislevel - lastcommonlevel) + (otherlevel - lastcommonlevel);
}

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom
