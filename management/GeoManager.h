/// \file GeoManager.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_MANAGEMENT_GEOMANAGER_H_
#define VECGEOM_MANAGEMENT_GEOMANAGER_H_

#include "base/Global.h"

#include "volumes/PlacedVolume.h"
#include "volumes/LogicalVolume.h"
#include "navigation/NavigationState.h"


#include <map>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

// probably don't need apply to be virtual
template<typename Container>
class GeoVisitor
{
protected:
   Container & c_;
public:
   GeoVisitor( Container & c ) : c_(c) {};

   virtual void apply( VPlacedVolume *, int level=0 ) = 0;
   virtual ~GeoVisitor(){}
};



template<typename Container>
class GeoVisitorWithAccessToPath
{
protected:
   Container & c_;
public:
   GeoVisitorWithAccessToPath( Container & c ) : c_(c) {};

   virtual void apply( NavigationState * state, int level=0 ) = 0;
   virtual ~GeoVisitorWithAccessToPath( ) { }
};


template<typename Container>
class SimpleLogicalVolumeVisitor : public GeoVisitor<Container>
{
public:
   SimpleLogicalVolumeVisitor( Container & c ) : GeoVisitor<Container>(c) {}
   virtual void apply( VPlacedVolume * vol, int /*level*/ ){
      LogicalVolume const *lvol = vol->logical_volume();
      if( std::find( this->c_.begin(), this->c_.end(), lvol ) == this->c_.end() )
      {
         this->c_.push_back( const_cast<LogicalVolume *>(lvol) );
      }
   }
   virtual ~SimpleLogicalVolumeVisitor(){}
};

template<typename Container>
class SimplePlacedVolumeVisitor : public GeoVisitor<Container>
{
public:
   SimplePlacedVolumeVisitor( Container & c) : GeoVisitor<Container>(c) {}
   virtual void apply( VPlacedVolume * vol, int /* level */ ){
      this->c_.push_back( vol );
   }
   virtual ~SimplePlacedVolumeVisitor(){}
};

class GetMaxDepthVisitor
{
private:
   int maxdepth_;
public:
   GetMaxDepthVisitor() : maxdepth_(0) {}
   void apply( VPlacedVolume * /* vol */, int level )
   {
      maxdepth_ = (level>maxdepth_) ? level : maxdepth_;
   }
   int getMaxDepth( ) const {return maxdepth_;}
};

class GetTotalNodeCountVisitor
{
private:
    int fTotalNodeCount;
public:
    GetTotalNodeCountVisitor() : fTotalNodeCount(0) {}
    void apply( VPlacedVolume *, int /* level */ )
    {
        fTotalNodeCount++;
    }
    int GetTotalNodeCount() const {return fTotalNodeCount;}
};

template<typename Container>
class GetPathsForLogicalVolumeVisitor : public GeoVisitorWithAccessToPath<Container>
{
private:
    LogicalVolume const * fReferenceLogicalVolume;
    int fMaxDepth;
public:
    GetPathsForLogicalVolumeVisitor(
            Container &c, LogicalVolume const * lv, int maxd) : fReferenceLogicalVolume(lv), fMaxDepth(maxd),
            GeoVisitorWithAccessToPath<Container>(c) {}
    void apply( NavigationState * state, int /* level */ )
    {
        if( state->Top()->logical_volume() == fReferenceLogicalVolume ){
            // the current state is a good one;

            // make a copy and store it in the container for this visitor
            NavigationState * copy = NavigationState::MakeCopy( *state );

            this->c_.push_back( copy );
        }
    }
};



/**
 * @brief Knows about the current world volume.
 */
class GeoManager {

private:

  int fVolumeCount;
  int fTotalNodeCount; // total number of nodes in the geometry tree
  VPlacedVolume const *fWorld;

  std::map<int, VPlacedVolume*> fPlacedVolumesMap;
  std::map<int, LogicalVolume*> fLogicalVolumesMap;
  int fMaxDepth;

  // traverses the geometry tree of placed volumes and applies injected Visitor
  template<typename Visitor>
  void visitAllPlacedVolumes(VPlacedVolume const *, Visitor * visitor, int level=1 ) const;

  // traverse the geometry tree keeping track of the state context ( volume path or navigation state )
  // apply the injected Visitor
  template<typename Visitor>
  void visitAllPlacedVolumesWithContext(VPlacedVolume const *, Visitor * visitor, NavigationState * state, int level=1 ) const;


public:

  static GeoManager& Instance() {
    static GeoManager instance;
    return instance;
  }

  /**
   * mark the current detector geometry as finished and initialize
   * important cached variables such as the maximum tree depth etc.
   */
  void CloseGeometry();

  void SetWorld(VPlacedVolume const *const w) { fWorld = w; }

  VPlacedVolume const* GetWorld() const { return fWorld; }

  /**
   *  give back container containing all logical volumes in detector
   *  Container is supposed to be any Container that can store pointers to
   */
  template<typename Container>
  void getAllLogicalVolumes( Container & c ) const;

  /**
   *  give back container containing all logical volumes in detector
   */
  template<typename Container>
  void getAllPlacedVolumes( Container & c ) const;

  /**
   *  give back container containing all logical volumes in detector
   *  Container has to be an (stl::)container keeping pointer to NavigationStates
   */
  template<typename Container>
  void getAllPathForLogicalVolume( LogicalVolume const * lvol, Container & c ) const;

  /**
   *  return max depth of volume hierarchy
   */
  int getMaxDepth() const {
      Assert( fMaxDepth > 0, "geometry not closed" );
      return fMaxDepth;
  }

  void RegisterPlacedVolume(VPlacedVolume *const placed_volume);

  void RegisterLogicalVolume(LogicalVolume *const logical_volume);

  void DeregisterPlacedVolume(const int id);

  void DeregisterLogicalVolume(const int id);

  /**
   * \return Volume with passed id, or NULL is the id wasn't found.
   */
  VPlacedVolume* FindPlacedVolume(const int id);

  /**
   * \return First occurrence of volume with passed label. If multiple volumes
   *         are found, their id will be printed to standard output.
   */
  VPlacedVolume* FindPlacedVolume(char const *const label);


  /**
   * \return Volume with passed id, or NULL is the id wasn't found.
   */
  LogicalVolume* FindLogicalVolume(const int id);


  /**
   * \return First occurrence of volume with passed label. If multiple volumes
   *         are found, their id will be printed to standard output.
   */
  LogicalVolume* FindLogicalVolume(char const *const label);

  /**
   * Clear/reset the manager
   *
   */
   void Clear();


   int GetPlacedVolumesCount() const {return fPlacedVolumesMap.size();}
   int GetLogicalVolumesCount() const {return fLogicalVolumesMap.size();}
   int GetTotalNodeCount() const {return fTotalNodeCount;}

protected:

private:
 GeoManager() : fVolumeCount(0), fTotalNodeCount(0), fWorld(NULL), fPlacedVolumesMap(),
 fLogicalVolumesMap(), fMaxDepth(-1)
 {}

  GeoManager(GeoManager const&);
  GeoManager& operator=(GeoManager const&);
};

template<typename Visitor>
void
GeoManager::visitAllPlacedVolumes( VPlacedVolume const * currentvolume, Visitor * visitor, int level ) const
{
   if( currentvolume != NULL )
   {
      visitor->apply( const_cast<VPlacedVolume *>(currentvolume), level );
      int size = currentvolume->daughters().size();
      for( int i=0; i<size; ++i )
      {
         visitAllPlacedVolumes( currentvolume->daughters().operator[](i), visitor, level+1 );
      }
   }
}

template<typename Visitor>
void
GeoManager::visitAllPlacedVolumesWithContext( VPlacedVolume const * currentvolume, Visitor * visitor, NavigationState * state, int level ) const
{
   if( currentvolume != NULL )
   {
      state->Push( currentvolume );
      visitor->apply( state, level );
      int size = currentvolume->daughters().size();
      for( int i=0; i<size; ++i )
      {
         visitAllPlacedVolumesWithContext( currentvolume->daughters().operator[](i), visitor, state, level+1 );
      }
      state->Pop();
   }
}


template<typename Container>
void GeoManager::getAllLogicalVolumes( Container & c ) const
{
   c.clear();
   // walk all the volume hierarchy and insert
   // logical volume if not already in the container
   SimpleLogicalVolumeVisitor<Container> lv(c);
   visitAllPlacedVolumes( GetWorld(), &lv );
}


template<typename Container>
void GeoManager::getAllPlacedVolumes( Container & c ) const
{
   c.clear();
   // walk all the volume hierarchy and insert
   // placed volumes if not already in the container
   SimplePlacedVolumeVisitor<Container> pv(c);
   visitAllPlacedVolumes( GetWorld(), &pv );
}


template<typename Container>
void GeoManager::getAllPathForLogicalVolume( LogicalVolume const * lvol, Container & c ) const
{
   NavigationState * state = NavigationState::MakeInstance(getMaxDepth());
   c.clear();
   state->Clear();

   // instantiate the visitor
   GetPathsForLogicalVolumeVisitor<Container> pv(c, lvol, getMaxDepth());

   // now walk the placed volume hierarchy
   visitAllPlacedVolumesWithContext( GetWorld(), &pv, state );
   NavigationState::ReleaseInstance( state );
}





} } // End global namespace

#endif // VECGEOM_MANAGEMENT_GEOMANAGER_H_
