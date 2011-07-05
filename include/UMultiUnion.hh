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

#include <vector>
using namespace std;

class UMultiUnion : public VUSolid
{
   // Internal structure for storing the association solid-placement
public:   
   class UNode {
   public:
      VUSolid          *fSolid;
      double           *fTransform;   // Permits the placement of "fSolid"
      
      UNode() : fSolid(0), fTransform(0) {}
      UNode(VUSolid *solid, double *trans)
      {
         fSolid = solid;
         fTransform = trans;
      }
      ~UNode();
};
   
public:
   UMultiUnion() : VUSolid(), fNodes(0) {}
// UMultiUnion() : VUSolid(), fNodes(0), fVoxels(0) {}
   UMultiUnion(const char *name); 
   UMultiUnion(const char *name, vector<UNode*> *nodes);   
// UMultiUnion(const char *name, vector<UNode> nodes, UVoxelFinder *voxels);
// UMultiUnion(const char *name, vector<UNode> nodes, UNode *newnode, UVoxelFinder *voxels);
   virtual ~UMultiUnion() {}
   
   // Navigation methods
   EnumInside        Inside (const UVector3 &aPoint) const;

   virtual double    SafetyFromInside (const UVector3 aPoint,
                                       bool aAccurate=false) const;
                                      
   virtual double    SafetyFromOutside(const UVector3 aPoint,
                                       bool aAccurate=false) const;
                                      
   virtual double    DistanceToIn     (const UVector3 &aPoint,
                                       const UVector3 &aDirection,
                                       // UVector3       &aNormalVector,
                                       double aPstep = UUtils::kInfinity) const;
 
   virtual double    DistanceToOut    (const UVector3 &aPoint,
                                       const UVector3 &aDirection,
                                       UVector3       &aNormalVector,
                                       bool           &aConvex,
                                       double aPstep = UUtils::kInfinity) const;
                                      
   virtual bool      Normal (const UVector3 &aPoint, UVector3 &aNormal);
   virtual void      Extent (EAxisType aAxis, double &aMin, double &aMax);
   virtual void      Extent (double aMin[3], double aMax[3]);
   virtual double    Capacity();
   virtual double    SurfaceArea();
   virtual           VUSolid* Clone() const {return 0;}
   virtual           UGeometryType GetEntityType() const {return "MultipleUnion";}
   virtual void      ComputeBBox(UBBox *aBox, bool aStore = false);
        
   // Other methods
   void              AddNode(UNode *node);
   void              Voxelize();   // Builds the voxels for the considered solid

   protected:
//    UVoxelFinder     *fVoxels;
   
   private:
      vector<UNode*>   *fNodes;   // Container of nodes
};
#endif   
