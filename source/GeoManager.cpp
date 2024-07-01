// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// \file GeoManager.cpp

#include "VecGeom/management/GeoManager.h"
#include "VecGeom/management/NavIndexTable.h"
#include "VecGeom/management/Logger.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/navigation/NavigationState.h"
#include "VecGeom/management/ABBoxManager.h"
#include "VecGeom/volumes/UnplacedBooleanVolume.h"
#include "VecGeom/volumes/UnplacedScaledShape.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/management/GeoVisitor.h"
#include "VecGeom/management/Logger.h"

#include <dlfcn.h>
#include <stdio.h>
#include <list>
#include <vector>
#include <set>
#include <functional>

namespace vecgeom {

VPlacedVolume *GeoManager::gCompactPlacedVolBuffer = nullptr;
NavIndex_t *GeoManager::gNavIndex                  = nullptr;
Precision GeoManager::gMillimeterUnit              = 0.1; // i.e default unit for length is centimeter

void GeoManager::RegisterLogicalVolume(LogicalVolume *const logical_volume)
{
  if (!fIsClosed)
    fLogicalVolumesMap[logical_volume->id()] = logical_volume;
  else {
    std::cerr << "Logical Volume created after geometry is closed --> will not be registered\n";
  }
}

void GeoManager::RegisterPlacedVolume(VPlacedVolume *const placed_volume)
{
  if (!fIsClosed)
    fPlacedVolumesMap[placed_volume->id()] = placed_volume;
  else {
    // std::cerr << "PlacedVolume " // << placed_volume->GetName()
    //          << " created after geometry is closed --> will not be registered\n";
  }
}

void GeoManager::DeregisterLogicalVolume(const int id)
{
  if (fLogicalVolumesMap.find(id) != fLogicalVolumesMap.end()) {
    if (fIsClosed) {
      std::cerr << "deregistering an object from GeoManager while geometry is closed\n";
    }
    fLogicalVolumesMap.erase(id);
  }
}

void GeoManager::DeregisterPlacedVolume(const int id)
{
  if (fPlacedVolumesMap.find(id) != fPlacedVolumesMap.end()) {
    if (fIsClosed) {
      std::cerr << "deregistering an object from GeoManager while geometry is closed\n";
    }
    fPlacedVolumesMap.erase(id);
  }
}

void GeoManager::CompactifyMemory()
{
  using UnplacedScaledShape = cxx::UnplacedScaledShape;
  // this function will compactify the memory a-posteriori
  // it might be worth investigating other methods that do this directly
  // ( for instance via specialized allocators )

  // ---------------------------------
  // start with just the placedvolumes

  // do a check on a fundamental hypothesis :
  // all placed volume objects have the same size ( so that we can compactify them in an array )
  for (auto v : fPlacedVolumesMap) {
    if (v.second->MemorySize() != fWorld->MemorySize())
      std::cerr << "Fatal Warning : placed volume instances have non-uniform size \n";
  }

  unsigned int pvolumecount = fPlacedVolumesMap.size();

  //  This piece of code was just to cross check something:

  //  std::vector<VPlacedVolume const *> pvolumes;
  //  getAllPlacedVolumes(pvolumes);
  //  // make it a set ( to get rid of potential duplicates )
  //  std::set<VPlacedVolume const *> pvolumeset(pvolumes.begin(), pvolumes.end());

  //  std::vector<LogicalVolume const *> lvolumes;
  //  GetAllLogicalVolumes(lvolumes);
  //  std::set<LogicalVolume const *> lvolumeset(lvolumes.begin(), lvolumes.end());

  //  std::cerr << pvolumecount << " vs " << pvolumeset.size() << "\n";
  //  std::cerr << fLogicalVolumesMap.size() << " vs " << lvolumeset.size() << "\n";

  // conversion map to repair pointers from old to new
  std::map<VPlacedVolume const *, VPlacedVolume const *> conversionmap;

  // allocate the buffer ( consider alignment issues later )
  // BIG NOTE HERE: we cannot call new VPlacedVolume[pvolumecount] as it is a pure virtual class
  // this also means: our mechanism will only work if none of the derived classes of VPlacedVolumes
  // adds a data member and we have to find a way to check or forbid this
  // ( a runtime check is done above )
  gCompactPlacedVolBuffer = (VPlacedVolume *)malloc(pvolumecount * sizeof(VPlacedVolume));

  //    // the first element in the buffer has to be the world
  //    buffer[0] = *fWorld; // copy assignment of PlacedVolumes
  //    // fix the index to pointer map
  //    fPlacedVolumesMap[fWorld->id()] = &buffer[0];
  //    conversionmap[ fWorld ] = &buffer[0];
  //    // free memory ( we should really be doing this with smart pointers --> check CUDA ! )
  //    // delete fWorld;
  //    // fix the global world pointer
  //    fWorld = &buffer[0];

  // go through rest of volumes
  // TODO: we could take an influence on the order here ( to place certain volumes next to each other )
  for (auto v : fPlacedVolumesMap) {
    unsigned int volumeindex             = v.first;
    gCompactPlacedVolBuffer[volumeindex] = *v.second;
    fPlacedVolumesMap[volumeindex]       = &gCompactPlacedVolBuffer[volumeindex];
    conversionmap[v.second]              = &gCompactPlacedVolBuffer[volumeindex];
    //   delete v.second;
  }

  // a little reusable lambda for the pointer conversion
  std::function<VPlacedVolume const *(VPlacedVolume const *)> ConvertOldToNew = [&](VPlacedVolume const *old) {
    if (conversionmap.find(old) == conversionmap.cend()) {
      // std::cerr << "CANNOT CONVERT ... probably already done" << std::endl;
      return old;
    }
    return conversionmap[old];
  };

  // fix pointers to placed volumes referenced in all logical volumes
  for (auto v : fLogicalVolumesMap) {
    LogicalVolume *lvol = v.second;
    auto ndaughter      = lvol->GetDaughtersp()->size();
    for (decltype(ndaughter) i = 0; i < ndaughter; ++i) {
      lvol->GetDaughtersp()->operator[](i) = ConvertOldToNew(lvol->GetDaughtersp()->operator[](i));
    }
  }

  for (auto v : fLogicalVolumesMap) {

    // check if this is a boolean type
    // FIXME: make this shorter!
    {
      using BoolT = UnplacedBooleanVolume<kSubtraction>;
      BoolT *bvol;
      if ((bvol = const_cast<BoolT *>(dynamic_cast<BoolT const *>(v.second->GetUnplacedVolume())))) {
        bvol->SetLeft(ConvertOldToNew(bvol->GetLeft()));
        bvol->SetRight(ConvertOldToNew(bvol->GetRight()));
      }
    }
    {
      using BoolT = UnplacedBooleanVolume<kUnion>;
      BoolT *bvol;
      if ((bvol = const_cast<BoolT *>(dynamic_cast<BoolT const *>(v.second->GetUnplacedVolume())))) {
        bvol->SetLeft(ConvertOldToNew(bvol->GetLeft()));
        bvol->SetRight(ConvertOldToNew(bvol->GetRight()));
      }
    }
    {
      using BoolT = UnplacedBooleanVolume<kIntersection>;
      BoolT *bvol;
      if ((bvol = const_cast<BoolT *>(dynamic_cast<BoolT const *>(v.second->GetUnplacedVolume())))) {
        bvol->SetLeft(ConvertOldToNew(bvol->GetLeft()));
        bvol->SetRight(ConvertOldToNew(bvol->GetRight()));
      }
    }

    // same for scaled shape volume
    UnplacedScaledShape *svol;
    if ((svol = const_cast<UnplacedScaledShape *>(
             dynamic_cast<UnplacedScaledShape const *>(v.second->GetUnplacedVolume())))) {
      svol->SetPlaced(ConvertOldToNew(svol->GetPlaced()));
    }
  }
  // cleanup conversion map ... automatically done

  // fix reference to World in GeoManager ( and everywhere else )
  fWorld = ConvertOldToNew(fWorld);
}

void GeoManager::CloseGeometry()
{
  assert(GetWorld() != nullptr);
  if (fIsClosed) {
    std::cerr << "geometry is already closed; I cannot close it again (very likely this message signifies a "
                 "substational error !!!\n";
  }
  // cache some important variables of this geometry
  GetMaxDepthVisitor depthvisitor;
  visitAllPlacedVolumes(GetWorld(), &depthvisitor, 1);
  fMaxDepth = depthvisitor.getMaxDepth();

  GetTotalNodeCountVisitor totalcountvisitor;
  visitAllPlacedVolumes(GetWorld(), &totalcountvisitor, 1);
  fTotalNodeCount = totalcountvisitor.GetTotalNodeCount();

  // get a consistent state for index - placed volumes lookups
  for (auto element : fPlacedVolumesMap) {
    fVolumeToIndexMap[element.second] = element.first;
  }

  // Initialize the Logical Volumes array for efficient access by index
  fLogicalVolumesArray.resize(GetRegisteredVolumesCount());
  for (uint i = 0; i < GetRegisteredVolumesCount(); i++) {
    fLogicalVolumesArray[i] = fLogicalVolumesMap[i];
  }

  CompactifyMemory();
  vecgeom::ABBoxManager<Precision>::Instance().InitABBoxesForCompleteGeometry();
  fIsClosed = true;

#if defined(VECGEOM_USE_NAVINDEX) || defined(VECGEOM_USE_NAVTUPLE)
  if (fCacheDepth == 0 || fCacheDepth > fMaxDepth) fCacheDepth = fMaxDepth;
  bool success = MakeNavIndexTable(fCacheDepth, fMinPerScene);
  if (!success) {
    VECGEOM_LOG(critical) << "GeoManager::CloseGeometry: The navigation table has errors";
  }
#endif
}

void GeoManager::LoadGeometryFromSharedLib(std::string libname, bool close)
{
  void *handle;
  handle = dlopen(libname.c_str(), RTLD_NOW);
  if (!handle) {
    VECGEOM_LOG(error) << "Failed to load geometry shared lib: " << dlerror() << "\n";
  }

  // the create detector "function type":
  typedef VPlacedVolume const *(*CreateFunc_t)();

  // find entry symbol to geometry creation
  // TODO: get rid of hard coded name
  CreateFunc_t create = (CreateFunc_t)dlsym(handle, "_Z16generateDetectorv");

  if (create != nullptr) {
    // call the create function and set the geometry world
    VPlacedVolume const *world = create();
    world->PrintType();
    SetWorld(world);

    // close the geometry
    // TODO: This step often necessitates extensive computation and could be done
    // as part of the shared lib load itself
    if (close)
      CloseGeometry();
    else {
      std::cerr << "Geometry left open for further manipulation; Please close later\n";
    }
  } else {
    std::cerr << "Loading geometry from shared lib failed\n";
  }
  //    dlclose(handle);
}

VPlacedVolume *GeoManager::FindPlacedVolume(const int id)
{
  auto iterator = fPlacedVolumesMap.find(id);
  return (iterator != fPlacedVolumesMap.end()) ? iterator->second : NULL;
}

VPlacedVolume *GeoManager::FindPlacedVolume(char const *const label)
{
  VPlacedVolume *output = NULL;
  bool multiple         = false;
  for (auto v = fPlacedVolumesMap.begin(), v_end = fPlacedVolumesMap.end(); v != v_end; ++v) {
    if (v->second->GetLabel() == label) {
      if (!output) {
        output = v->second;
      } else {
        if (!multiple) {
          multiple = true;
          std::cerr << "GeoManager::FindPlacedVolume: Multiple logical volumes with identifier \"" << label
                    << "\" found: [" << output->id() << "], ";
        } else {
          std::cerr << ", ";
        }
        std::cerr << "[" << (v->second)->id() << "]";
      }
    }
  }
  if (multiple) std::cerr << ". Returning first occurrence.\n";

  return output;
}

LogicalVolume *GeoManager::FindLogicalVolume(const int id)
{
  auto iterator = fLogicalVolumesMap.find(id);
  return (iterator != fLogicalVolumesMap.end()) ? iterator->second : NULL;
}

LogicalVolume *GeoManager::FindLogicalVolume(char const *const label)
{
  LogicalVolume *output = nullptr;
  bool multiple         = false;

  for (const auto &v : fLogicalVolumesMap) {
    const std::string &fullname = (v.second)->GetLabel();

    if (fullname.compare(label) == 0) {
      if (output == nullptr) {
        output = v.second;
      } else {
        if (!multiple) {
          multiple = true;
          std::cerr << "GeoManager::FindLogicalVolume: Multiple logical volumes with identifier \"" << label
                    << "\" found: [" << output->id() << "], ";
        } else {
          std::cerr << ", ";
        }
        std::cerr << "[" << (v.second)->id() << "]";
      }
    }
  }
  if (multiple) std::cerr << ". Returning first occurrence.\n";
  return output;
}

int GeoManager::GetLogicalVolumeId(const std::string &label)
{
  const LogicalVolume *lv = this->FindLogicalVolume(label.c_str());
  return (lv == nullptr) ? -1 : lv->id();
}

std::string GeoManager::GetLogicalVolumeLabel(int id)
{
  const LogicalVolume *lv = this->FindLogicalVolume(id);
  return (lv == nullptr) ? "" : lv->GetLabel();
}

void GeoManager::Clear()
{
  fVolumeCount    = 0;
  fTotalNodeCount = 0;
  fWorld          = nullptr;
  fPlacedVolumesMap.clear();
  fLogicalVolumesMap.clear();
  fVolumeToIndexMap.clear();
  fMaxDepth = -1;
  fIsClosed = false;
  // should we also reset the global static id counts?
  LogicalVolume::gIdCount   = 0;
  VPlacedVolume::g_id_count = 0;
  // delete compact buffer for placed volumes
  if (GeoManager::gCompactPlacedVolBuffer != nullptr) {
    free(gCompactPlacedVolBuffer);
    gCompactPlacedVolBuffer = nullptr;
  }

  if (GeoManager::gNavIndex != nullptr) {
    NavIndexTable::Instance()->CleanTable();
    gNavIndex = nullptr;
  }
}

#if defined(VECGEOM_USE_NAVINDEX) || defined(VECGEOM_USE_NAVTUPLE)
bool GeoManager::MakeNavIndexTable(int depth_limit, int min_per_scene, bool validate) const
{
  if (gNavIndex) {
    std::cerr << "=== GeoManager::MakeNavIndexTable: navigation table already created\n";
    return false;
  }
  bool success = NavIndexTable::Instance()->CreateTable(GetWorld(), getMaxDepth(), depth_limit, min_per_scene);
  if (success) {
    gNavIndex = NavIndexTable::Instance()->GetTable();
    NavIndexTable::Instance()->SetVolumeBuffer(gCompactPlacedVolBuffer);
  }
  if (validate) success = NavIndexTable::Instance()->Validate(GetWorld(), getMaxDepth());
  return success;
}
#endif

template <typename Container>
class GetPathsForLogicalVolumeVisitor : public GeoVisitorWithAccessToPath<Container> {
private:
  LogicalVolume const *fReferenceLogicalVolume;
  int fMaxDepth;

public:
  GetPathsForLogicalVolumeVisitor(Container &c, LogicalVolume const *lv, int maxd)
      : GeoVisitorWithAccessToPath<Container>(c), fReferenceLogicalVolume(lv), fMaxDepth(maxd)
  {
  }

  void apply(NavigationState *state, int /* level */)
  {
    if (state->Top()->GetLogicalVolume() == fReferenceLogicalVolume) {
      // the current state is a good one;

      // make a copy and store it in the container for this visitor
      NavigationState *copy = NavigationState::MakeCopy(*state);

      this->c_.push_back(copy);
    }
  }
};

template <typename Visitor>
void GeoManager::visitAllPlacedVolumesWithContext(VPlacedVolume const *currentvolume, Visitor *visitor,
                                                  NavigationState *state, int level) const
{
  if (currentvolume != NULL) {
    state->Push(currentvolume);
    visitor->apply(state, level);
    int size = currentvolume->GetDaughters().size();
    for (int i = 0; i < size; ++i) {
      visitAllPlacedVolumesWithContext(currentvolume->GetDaughters().operator[](i), visitor, state, level + 1);
    }
    state->Pop();
  }
}

template <typename Container>
__attribute__((noinline)) void GeoManager::getAllPathForLogicalVolume(LogicalVolume const *lvol, Container &c) const
{
  NavigationState *state = NavigationState::MakeInstance(getMaxDepth());
  c.clear();
  state->Clear();

  // instantiate the visitor
  GetPathsForLogicalVolumeVisitor<Container> pv(c, lvol, getMaxDepth());

  // now walk the placed volume hierarchy
  visitAllPlacedVolumesWithContext(GetWorld(), &pv, state);
  NavigationState::ReleaseInstance(state);
}

// explicitely init some symbols
template void GeoManager::getAllPathForLogicalVolume(LogicalVolume const *lvol, std::list<NavigationState *> &c) const;
template void GeoManager::getAllPathForLogicalVolume(LogicalVolume const *lvol,
                                                     std::vector<NavigationState *> &c) const;
} // namespace vecgeom
