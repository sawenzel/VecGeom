#ifndef USOLIDS_UMultiUnion
#define USOLIDS_UMultiUnion

////////////////////////////////////////////////////////////////////////////////
//
//  An instance of "UMultiUnion" constitutes a grouping of several solids
//  deriving from the "VUSolid" mother class. The subsolids are stored with
//  their respective location in an instance of "UNode". An instance of
//  "UMultiUnion" is subsequently composed of one or several nodes.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef USOLIDS_VUSolid
#include "VUSolid.hh"
#endif

#ifndef USOLIDS_UUtils
#include "UUtils.hh"
#endif

#ifndef USOLIDS_UTransform3D
#include "UTransform3D.hh"
#endif

#include <vector>

class UVoxelFinder;

class UMultiUnion : public VUSolid
{
   // Internal structure for storing the association solid-placement
public:   
   class UNode {
   public:
      VUSolid          *fSolid;
      UTransform3D     *fTransform;   // Permits the placement of "fSolid"
      
      UNode() : fSolid(0), fTransform(0) {}
      UNode(VUSolid *solid, UTransform3D *trans)
      {
         fSolid = solid;
         fTransform = trans;
      }
      ~UNode();
};
   
public:
   UMultiUnion() : VUSolid(), fNodes(0), fVoxels(0) {}
   UMultiUnion(const char *name); 
   ~UMultiUnion();
   
   // Build the multiple union by adding nodes
   void                         AddNode(VUSolid *solid, UTransform3D *trans);
   
   // Finalize and prepare for use. User MUST call it once before navigation use.
   void                         CloseSolid();
   
   // Navigation methods
   EnumInside                   Inside (const UVector3 &aPoint) const;

   double                       SafetyFromInside (const UVector3 aPoint,
                                                  bool aAccurate=false) const;
                                      
   double                       SafetyFromOutside(const UVector3 aPoint,
                                                  bool aAccurate=false) const;
                                      
   double                       DistanceToIn     (const UVector3 &aPoint,
                                                  const UVector3 &aDirection,
                                                  // UVector3       &aNormalVector,
                                                  double aPstep = UUtils::kInfinity) const;
 
   double                       DistanceToOut    (const UVector3 &aPoint,
                                                  const UVector3 &aDirection,
                                                  UVector3       &aNormalVector,
                                                  bool           &aConvex,
                                                  double aPstep = UUtils::kInfinity) const;
                                      
   bool                         Normal (const UVector3 &aPoint, UVector3 &aNormal);
   void                         Extent (EAxisType aAxis, double &aMin, double &aMax);
   void                         Extent (double aMin[3], double aMax[3]);
   double                       Capacity();
   double                       SurfaceArea();
   VUSolid*                     Clone() const {return 0;}
   UGeometryType                GetEntityType() const {return "MultipleUnion";}
   void                         ComputeBBox(UBBox *aBox, bool aStore = false);
        
   // Other methods
   int                          GetNumNodes() const;
   /*const*/ VUSolid*           GetSolid(int index) /*const*/;
   const UTransform3D*          GetTransform(int index) const;                             

private:
   std::vector<UNode*>         *fNodes;   // Container of nodes
   UVoxelFinder                *fVoxels;
};
#endif   
