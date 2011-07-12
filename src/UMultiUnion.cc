#include "UMultiUnion.hh"

#include <iostream>
#include "UUtils.hh"
#include "UVoxelFinder.hh"

using namespace std;

//______________________________________________________________________________       
UMultiUnion::UMultiUnion(const char *name)
{
   SetName(name);
   fNodes  = 0; 
   fVoxels = 0;  
}

//______________________________________________________________________________       
void UMultiUnion::AddNode(VUSolid *solid, UTransform3D *trans)
{
   UNode* node = new UNode(solid,trans);   

   if (!fNodes)
   {
      fNodes = new vector<UNode*>;
   }
   
   fNodes->push_back(node);
}

//______________________________________________________________________________       
double UMultiUnion::Capacity()
{
// Computes analytically the capacity of the solid.
   cout << "Capacity - Not implemented" << endl;
   return 0.;
}     

//______________________________________________________________________________
void UMultiUnion::ComputeBBox (UBBox */*aBox*/, bool /*aStore*/)
{
// Computes bounding box.
   cout << "ComputeBBox - Not implemented" << endl;
}   

//______________________________________________________________________________
double UMultiUnion::DistanceToIn(const UVector3 &/*aPoint*/, 
                          const UVector3 &/*aDirection*/, 
//                          UVector3 &aNormal, 
                          double /*aPstep*/) const
{
// Computes distance from a point presumably outside the solid to the solid 
// surface. Ignores first surface if the point is actually inside. Early return
// infinity in case the safety to any surface is found greater than the proposed
// step aPstep.
// The normal vector to the crossed surface is filled only in case the box is 
// crossed, otherwise aNormal.IsNull() is true.
   cout << "DistanceToIn - Not implemented" << endl;
   return 0.;
}     

//______________________________________________________________________________
double UMultiUnion::DistanceToOut( const UVector3  &aPoint, const UVector3 &aDirection,
			       UVector3 &aNormal,
			       bool    &convex,
                double /*aPstep*/) const
{
// Computes distance from a point presumably intside the solid to the solid 
// surface. Ignores first surface along each axis systematically (for points
// inside or outside. Early returns zero in case the second surface is behind
// the starting point.
// o The proposed step is ignored.
// o The normal vector to the crossed surface is always filled.
   cout << "DistanceToOut - Not implemented" << endl;
   return 0.;
}

//______________________________________________________________________________
VUSolid::EnumInside UMultiUnion::Inside(const UVector3 &aPoint) const
{
// Classify point location with respect to solid:
//  o eInside       - inside the solid
//  o eSurface      - close to surface within tolerance
//  o eOutside      - outside the solid

// Hitherto, it is considered that:
//        - only parallelepipedic nodes can be added to the container
//        - the transformation matrix involves only three translation components

   int CarNodes = fNodes->size(); // Total number of nodes contained in "fNodes".
   int iIndex = 0;
   
   VUSolid *TempSolid = NULL;
   UTransform3D *TempTransform = NULL;
   UVector3 TempPoint;
   VUSolid::EnumInside TempInside = eOutside;

   // Loop on the nodes:
   for(iIndex = 0 ; iIndex < CarNodes ; iIndex++)
   {
      TempSolid = ((*fNodes)[iIndex])->fSolid;
      TempTransform = ((*fNodes)[iIndex])->fTransform;

      // The coordinates of the point are modified so as to fit the intrinsic solid local frame:
      TempPoint.x = aPoint.x - TempTransform->fTr[0];
      TempPoint.y = aPoint.y - TempTransform->fTr[1];
      TempPoint.z = aPoint.z - TempTransform->fTr[2];            

      TempInside = TempSolid->Inside(TempPoint);
      
      if((TempInside == eInside) || (TempInside == eSurface))
      {
         return TempInside;
      }      
   }
   return eOutside;
}

//______________________________________________________________________________ 
void UMultiUnion::Extent(EAxisType aAxis,double &aMin,double &aMax)
{
// Determines the bounding box for the considered instance of "UMultipleUnion"
   int CarNodes = fNodes->size();
   double *vertices = new double[24];
   UVector3 TempPointConv,TempPoint;   
   int iIndex,jIndex,kIndex = 0;
   double mini,maxi = 0;
   double current = 0;

   VUSolid *TempSolid = NULL;
   UTransform3D *TempTransform = NULL; 
   UVector3 min, max;
    
   // Loop on the nodes:
   for(iIndex = 0 ; iIndex < CarNodes ; iIndex++)
   {
      TempSolid = ((*fNodes)[iIndex])->fSolid;
      TempTransform = ((*fNodes)[iIndex])->fTransform;
      
      
     // TempSolid->SetVertices(vertices);
     // TODO: Define a method: TransformVertices(UVector3 &min, UVector3 &max, const UTransform3D *transform)
     //  that should take the min/max limits in the local frame and transform them
//     UVector3 min, max;
//     TempSolid->Extent(min, max);
//       UUtils::TransformVertices(min, max, TempTransform)
      
      // Code below to be used to implement TransformVertices
      for(jIndex = 0 ; jIndex < 8 ; jIndex++)
      {
         kIndex = 3*jIndex;
         TempPoint.Set(vertices[kIndex],vertices[kIndex+1],vertices[kIndex+2]);
         TempPointConv = TempTransform->GlobalPoint(TempPoint);
         
         // Current position on he considered axis (global frame):
         if(aAxis == eXaxis)         
            current = TempPointConv.x;
         else if(aAxis == eYaxis)
            current = TempPointConv.y;
         else
            current = TempPointConv.z;
         
         // Initialization of extrema:
         if(iIndex == 0 && jIndex == 0)
         {
            mini = maxi = current;
         }
         
         // If need be, replacement of the min & max values:
         if(current > maxi)
            maxi = current;
         if(current < mini)
            mini = current;         
      }
   }
   aMin = mini;
   aMax = maxi;
}

//______________________________________________________________________________ 
void UMultiUnion::Extent(double aMin[3],double aMax[3])
{
   double min,max = 0;
   this->Extent(eXaxis,min,max);
   aMin[0] = min; aMax[0] = max;
   this->Extent(eYaxis,min,max);
   aMin[1] = min; aMax[1] = max;
   this->Extent(eZaxis,min,max);
   aMin[2] = min; aMax[2] = max;      
}

//______________________________________________________________________________
bool UMultiUnion::Normal( const UVector3& aPoint, UVector3 &aNormal)
{
// Computes the normal on a surface and returns it as a unit vector
//   In case a point is further than tolerance_normal from a surface, set validNormal=false
//   Must return a valid vector. (even if the point is not on the surface.)
//
//   On an edge or corner, provide an average normal of all facets within tolerance
// NOTE: the tolerance value used in here is not yet the global surface
//     tolerance - we will have to revise this value - TODO
   cout << "Normal - Not implemented" << endl;
   return 0.;   
}

//______________________________________________________________________________ 
double UMultiUnion::SafetyFromInside(const UVector3 aPoint,bool aAccurate) const
{
   cout << "SafetyFromInside - Not implemented" << endl;
   return 0.;
}

//______________________________________________________________________________
double UMultiUnion::SafetyFromOutside ( const UVector3 aPoint, bool aAccurate) const
{
// Estimates the isotropic safety from a point outside the current solid to any 
// of its surfaces. The algorithm may be accurate or should provide a fast 
// underestimate.
   cout << "SafetyFromOutside - Not implemented" << endl;
   return 0.;
}

//______________________________________________________________________________       
double UMultiUnion::SurfaceArea()
{
// Computes analytically the surface area.
   cout << "SurfaceArea - Not implemented" << endl;
   return 0.;
}     

//______________________________________________________________________________       
int UMultiUnion::GetNumNodes() const
{
   return(fNodes->size());
}

//______________________________________________________________________________       
const VUSolid* UMultiUnion::GetSolid(int index) const
{
   return(((*fNodes)[index])->fSolid);
}

//______________________________________________________________________________       

const UTransform3D* UMultiUnion::GetTransform(int index) const
{
   return(((*fNodes)[index])->fTransform);

}
