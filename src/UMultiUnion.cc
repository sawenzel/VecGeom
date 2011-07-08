#include "UMultiUnion.hh"
#include "UBox.hh"
#include <iostream>
#include "UUtils.hh"

//______________________________________________________________________________       
UMultiUnion::UMultiUnion(const char *name)
{
   SetName(name);
   fNodes = NULL;   
}

//______________________________________________________________________________       
UMultiUnion::UMultiUnion(const char *name, vector<UNode*> *nodes) 
{
   SetName(name);
   fNodes = nodes;   
}

//______________________________________________________________________________       
void UMultiUnion::AddNode(VUSolid *solid, UTransform3D *trans)
{
   UNode* node = new UNode(solid,trans);   

   if(fNodes == NULL)
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
    
   // Loop on the nodes:
   for(iIndex = 0 ; iIndex < CarNodes ; iIndex++)
   {
      TempSolid = ((*fNodes)[iIndex])->fSolid;
      TempTransform = ((*fNodes)[iIndex])->fTransform;
      
      TempSolid->SetVertices(vertices);
      
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
int UMultiUnion::GetNNodes()
{
   return(fNodes->size());
}

//______________________________________________________________________________       
VUSolid* UMultiUnion::GetSolid(int index)
{
   return(((*fNodes)[index])->fSolid);
}

//______________________________________________________________________________       

UTransform3D* UMultiUnion::GetTransform(int index)
{
   return(((*fNodes)[index])->fTransform);

}
