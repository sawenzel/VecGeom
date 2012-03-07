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

#include "VUSolid.hh"

#include "UUtils.hh"

#include "UTransform3D.hh"

#include <vector>
#include <time.h>
#include <stdlib.h>

#include "UBits.h"

#include <UVoxelFinder.hh>

class UMultiUnion : public VUSolid
{
	friend class UVoxelFinder;
	friend class UVoxelCandidatesIterator;

   // Internal structure for storing the association solid-placement
public:   
   
public:
   UMultiUnion() : VUSolid() {}
   UMultiUnion(const char *name); 
   ~UMultiUnion();
      
   // Navigation methods
   EnumInside                   Inside (const UVector3 &aPoint) const;

   EnumInside                   InsideIterator (const UVector3 &aPoint) const;

   EnumInside                   InsideBits (const UVector3 &aPoint) const;

   double                       SafetyFromInside (const UVector3 &aPoint,
                                                  bool aAccurate=false) const;
                                      
   double                       SafetyFromOutside(const UVector3 &aPoint,
                                                  bool aAccurate=false) const;
                                     
   double                       DistanceToInDummy     (const UVector3 &aPoint,
                                                  const UVector3 &aDirection,
                                                  // UVector3       &aNormalVector,
                                                  double aPstep = UUtils::kInfinity) const;
 
   double                       DistanceToOut    (const UVector3 &aPoint,
                                                  const UVector3 &aDirection,
                                                  UVector3       &aNormalVector,
                                                  bool           &aConvex,
                                                  double aPstep = UUtils::kInfinity) const;

   double                       DistanceToOutVoxels    (const UVector3 &aPoint,
                                                  const UVector3 &aDirection,
                                                  UVector3       &aNormalVector,
                                                  bool           &aConvex,
                                                  double aPstep = UUtils::kInfinity) const;

   double                       DistanceToOutVoxelsCore    (const UVector3 &aPoint,
                                                  const UVector3 &aDirection,
                                                  UVector3       &aNormalVector,
                                                  bool           &aConvex,
                                                  std::vector<int> &candidates) const;

   double                       DistanceToOutVoxelsCoreNew    (const UVector3 &aPoint,
                                                  const UVector3 &aDirection,
                                                  UVector3       &aNormalVector,
                                                  bool           &aConvex,
                                                  std::vector<int> &candidates) const;

   double                       DistanceToOutDummy    (const UVector3 &aPoint,
                                                  const UVector3 &aDirection,
                                                  UVector3       &aNormalVector,
                                                  bool           &aConvex,
                                                  double aPstep = UUtils::kInfinity) 
												  const;

   bool                         Normal (const UVector3 &aPoint, UVector3 &aNormal) const;
   void                         Extent (EAxisType aAxis, double &aMin, double &aMax) const;
   void							Extent (UVector3 &aMin, UVector3 &aMax) const;
   double                       Capacity();
   double                       SurfaceArea();
   VUSolid*                     Clone() const {return 0;}
   UGeometryType                GetEntityType() const {return "MultipleUnion";}
   void                         ComputeBBox(UBBox *aBox, bool aStore = false);

   virtual void GetParametersList(int aNumber,double *aArray) const {}
   virtual UPolyhedron* GetPolyhedron() const {return 0;}   
   
     // Build the multiple union by adding nodes
   void                         AddNode(VUSolid &solid, UTransform3D &trans);

   void                         Voxelize();
   EnumInside                   InsideDummy(const UVector3 &aPoint) const;

   inline UVoxelFinder &GetVoxels() const
   {
	   return (UVoxelFinder &)voxels;
   }

   static UMultiUnion *CreateTestMultiUnion(int numNodes); // Number of nodes to implement

private:

   void                         SetVoxelFinder(const UVoxelFinder& finder);
   
   EnumInside                   InsideWithExclusion(const UVector3 &aPoint, UBits *bits=NULL) const;

   int                          SafetyFromOutsideNumberNode(const UVector3 &aPoint, bool aAccurate, double &safety) const;
   double                       DistanceToIn(const UVector3 &aPoint, 
                                                   const UVector3 &aDirection, 
                                                // UVector3 &aNormal, 
                                                   double aPstep) const;  

   double                       DistanceToInCandidates(const UVector3 &aPoint, const UVector3 &aDirection, double aPstep, std::vector<int> &candidates, UBits &bits) const;  

	  std::vector<VUSolid *> solids;
	  std::vector<UTransform3D *> transforms;
	  UVoxelFinder        voxels;  // Pointer to the vozelized solid

  
   // Finalize and prepare for use. User MUST call it once before navigation use.
   void                         CloseSolid();

};
#endif   
