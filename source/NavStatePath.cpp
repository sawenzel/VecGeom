/// \file NavStatePath.cpp
/// \author Sandro Wenzel (sandro.wenzel@cern.ch)
/// \date 17.04.2014

#include "VecGeom/navigation/NavStatePath.h"

#include <iostream>
#include <list>
#include <sstream>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
Vector3D<Precision> NavStatePath::GlobalToLocal(Vector3D<Precision> const &globalpoint, int tolevel) const
{
  Vector3D<Precision> tmp = globalpoint;
  Vector3D<Precision> current;
  for (int level = 0; level < tolevel; ++level) {
    Transformation3D const *m = At(level)->GetTransformation();
    current                   = m->Transform(tmp);
    tmp                       = current;
  }
  return tmp;
}

VECCORE_ATT_HOST_DEVICE
void NavStatePath::TopMatrix(int tolevel, Transformation3D &global_matrix) const
{
  for (int i = tolevel - 1; i > 0; --i) {
    global_matrix *= *(ToPlacedVolume(fPath[i])->GetTransformation());
  }
}

// returning a "delta" transformation that can transform
// coordinates given in reference frame of this->Top() to the reference frame of other->Top()
// simply with otherlocalcoordinate = delta.Transform( thislocalcoordinate )
VECCORE_ATT_HOST_DEVICE
void NavStatePath::DeltaTransformation(NavStatePath const &other, Transformation3D &delta) const
{
  Transformation3D g2;
  Transformation3D g1;
  other.TopMatrix(g2);
  this->TopMatrix(g1);
  delta = g1.Inverse();
  g2.SetProperties();
  delta.SetProperties();
  delta.FixZeroes();
  delta.MultiplyFromRight(g2);
  delta.FixZeroes();
}

/**
 * function that transforms a global point to local point in reference frame of deepest volume in current navigation
 * state
 * ( equivalent to using a global matrix )
 */
VECCORE_ATT_HOST_DEVICE
Vector3D<Precision> NavStatePath::GlobalToLocal(Vector3D<Precision> const &globalpoint) const
{
  Vector3D<Precision> tmp = globalpoint;
  Vector3D<Precision> current;
  for (int level = 0; level < fCurrentLevel; ++level) {
    Transformation3D const *m = At(level)->GetTransformation();
    current                   = m->Transform(tmp);
    tmp                       = current;
  }
  return tmp;
}

size_t FindIndexWithinMother(VPlacedVolume const *mother, VPlacedVolume const *daughter)
{
  for (size_t d = 0; d < mother->GetDaughters().size(); ++d) {
    if (mother->GetDaughters()[d] == daughter) return d;
  }
  assert(false && "did not find index of a daughter volume within mother");
  return static_cast<uint>(-1);
}

VPlacedVolume const *GetDaughterWithinMother(VPlacedVolume const *mother, uint index)
{
  if (index < (uint)mother->GetDaughters().size()) return mother->GetDaughters()[index];

  return NULL;
}

void NavStatePath::GetPathAsListOfIndices(std::list<uint> &indices) const
{
  indices.clear();
  if (IsOutside()) return;
  for (uint level = fCurrentLevel - 1; level > 0; --level) {
    indices.push_front(FindIndexWithinMother(At(level - 1), At(level)));
  }
  indices.push_front(0);
}

VECCORE_ATT_HOST_DEVICE
void NavStatePath::Print() const
{
// printf("VariableSizeObj: fPath=%p (%l bytes)\n", fPath, sizeof(fPath));
#ifndef VECCORE_CUDA
  printf("NavStatePath: level=%i/%i,  onBoundary=%s, path=<", fCurrentLevel - 1, GetMaxLevel(),
         (fOnBoundary ? "true" : "false"));
  for (int i = 0; i < fCurrentLevel; ++i)
    printf("/%s", ToPlacedVolume(fPath[i]) ? ToPlacedVolume(fPath[i])->GetLabel().c_str() : "NULL");
  printf(">\n");
#else
  printf("NavStatePath: level=%i/%i,  onBoundary=%s, topVol=<%p>, this=%p\n", fCurrentLevel - 1, GetMaxLevel(),
         (fOnBoundary ? "true" : "false"), Top(), (const void *)this);
#endif
}

void NavStatePath::ResetPathFromListOfIndices(VPlacedVolume const *world, std::list<uint> const &indices)
{
  // clear current nav state
  fCurrentLevel = indices.size();
  if (indices.size() > 0) {
    fPath[0] = ToIndex(world);
    // have to disregard first one;
    // then iterate through list
    int counter = 0;
    for (auto x : indices) {
      if (counter > 0) fPath[counter] = ToIndex(GetDaughterWithinMother(At(counter - 1), x));
      counter++;
    }
  }
}

std::string NavStatePath::RelativePath(NavStatePath const &other) const
{
  int lastcommonlevel = -1;
  int maxlevel        = Min(GetCurrentLevel(), other.GetCurrentLevel());
  std::stringstream str;
  //  algorithm: start on top and go down until paths split
  for (int i = 0; i < maxlevel; i++) {
    if (this->At(i) == other.At(i)) {
      lastcommonlevel = i;
    } else {
      break;
    }
  }

  auto filledlevel1 = GetCurrentLevel() - 1;
  auto filledlevel2 = other.GetCurrentLevel() - 1;

  // paths are the same
  if (filledlevel1 == lastcommonlevel && filledlevel2 == lastcommonlevel) {
    return std::string("");
  }

  // emit only ups
  if (filledlevel1 > lastcommonlevel && filledlevel2 == lastcommonlevel) {
    for (int i = 0; i < filledlevel1 - lastcommonlevel; ++i) {
      str << "/up";
    }
    return str.str();
  }

  // emit only downs
  if (filledlevel1 == lastcommonlevel && filledlevel2 > lastcommonlevel) {
    for (int i = lastcommonlevel + 1; i <= filledlevel2; ++i) {
      str << "/down";
      str << "/" << other.ValueAt(i);
    }
    return str.str();
  }

  // mixed case: first up; then down
  if (filledlevel1 > lastcommonlevel && filledlevel2 > lastcommonlevel) {
    // emit ups
    int level = filledlevel1;
    for (; level > lastcommonlevel + 1; --level) {
      str << "/up";
    }

    level = lastcommonlevel + 1;
    // emit horiz ( exists when there is a turning point )
    int delta = other.ValueAt(level) - this->ValueAt(level);
    if (delta != 0) str << "/horiz/" << delta;

    level++;
    // emit downs with index
    for (; level <= filledlevel2; ++level) {
      str << "/down/" << other.ValueAt(level);
    }
  }
  return str.str();
}
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom
