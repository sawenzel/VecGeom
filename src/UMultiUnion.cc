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
void UMultiUnion::AddNode(UNode *node)
{
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
   double *TempTransform = NULL;
   UVector3 TempPoint;
   VUSolid::EnumInside TempInside = eOutside;

   // Loop on the nodes:
   for(iIndex = 0 ; iIndex < CarNodes ; iIndex++)
   {
      TempSolid = ((*fNodes)[iIndex])->fSolid;
      TempTransform = ((*fNodes)[iIndex])->fTransform;

      // The coordinates of the point are modified so as to fit the intrinsic solid local frame:
      TempPoint.x = aPoint.x - TempTransform[0];
      TempPoint.y = aPoint.y - TempTransform[1];
      TempPoint.z = aPoint.z - TempTransform[2];            

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
   int CarNodes = fNodes->size(); // Total number of nodes contained in "fNodes".
   int iIndex = 0;
   double mini_int = 0;
   double maxi_int = 0;
   double mini = 0;      
   double maxi = 0;
   double mini_loc = 0;      
   double maxi_loc = 0;
   VUSolid *TempSolid = NULL;   
   double *TempTransform = NULL;
   int axis;  
   
   // Considered axis (conversion):
   if(aAxis == eXaxis)
   {
      axis = 0;
   }
   else if(aAxis == eYaxis)
   {
      axis = 1;
   }
   else
   {
      axis = 2;
   }
   
   // Loop on the nodes:
   for(iIndex = 0 ; iIndex < CarNodes ; iIndex++)
   {
      TempSolid = ((*fNodes)[iIndex])->fSolid;   
      TempTransform = ((*fNodes)[iIndex])->fTransform;   
      
      // Intrinsic extrema of the considered nodes and considering the axis "aAxis":
      TempSolid->Extent(aAxis,mini_int,maxi_int);
      mini_loc = mini_int + TempTransform[axis];
      maxi_loc = maxi_int + TempTransform[axis];
      
      if(iIndex == 0)
      {
         mini = mini_loc;
         maxi = maxi_loc;
      }
      
      if(mini_loc < mini)
      {
         mini = mini_loc;
      }
      if(maxi_loc > maxi)
      {
         maxi = maxi_loc;
      }
   }
   
   // Using refrences :
   aMin = mini;
   aMax = maxi;      
}

//______________________________________________________________________________ 
void UMultiUnion::Extent(double aMin[3],double aMax[3])
{
   int CarNodes = fNodes->size();
   int iIndex = 0;
   int jIndex = 0;
   double mini_int;
   double maxi_int;
   double mini;      
   double maxi;
   double mini_loc;      
   double maxi_loc;   
   VUSolid *TempSolid = NULL;
   double *TempTransform = NULL;
   VUSolid::EAxisType axis = VUSolid::eXaxis;
  
   // Loop on the axis:
   for(jIndex = 0 ; jIndex < 3 ; jIndex++)
   {
      mini_int = 0;
      maxi_int = 0;
      mini = 0;      
      maxi = 0;
      mini_loc = 0;      
      maxi_loc = 0;    
      
      if(jIndex == 0)
      {
         axis = VUSolid::eXaxis;
      }
      else if(jIndex == 1)
      {
         axis = VUSolid::eYaxis;
      }
      else
      {
         axis = VUSolid::eZaxis;      
      }
   
      // Loop on the axis:   
      for(iIndex = 0 ; iIndex < CarNodes ; iIndex++)
      {
         TempSolid = ((*fNodes)[iIndex])->fSolid;   
         TempTransform = ((*fNodes)[iIndex])->fTransform;   
         
         // Intrinsic extrema of the considered nodes and considering the axis "aAxis":
         TempSolid->Extent(axis,mini_int,maxi_int);
         mini_loc = mini_int + TempTransform[axis];
         maxi_loc = maxi_int + TempTransform[axis];
      
         if(iIndex == 0)
         {
            mini = mini_loc;
            maxi = maxi_loc;
         }
      
         if(mini_loc < mini)
         {
            mini = mini_loc;
         }
         if(maxi_loc > maxi)
         {
            maxi = maxi_loc;
         }
      }

      aMin[jIndex] = mini;
      aMax[jIndex] = maxi;   
   }
}

//______________________________________________________________________________
bool UMultiUnion::Normal( const UVector3& /*aPoint*/, UVector3 &/*aNormal*/ )
{
// Computes the normal on a surface and returns it as a unit vector
//   In case a point is further than tolerance_normal from a surface, set validNormal=false
//   Must return a valid vector. (even if the point is not on the surface.)
//
//   On an edge or corner, provide an average normal of all facets within tolerance
// NOTE: the tolerance value used in here is not yet the global surface
//     tolerance - we will have to revise this value - TODO
   cout << "Normal - Not implemented" << endl;
   return false;
}

//______________________________________________________________________________ 
double UMultiUnion::SafetyFromInside(const UVector3 aPoint,bool aAccurate) const
{
   int CarNodes = fNodes->size();
   int iIndex = 0;
   int jIndex = 0;
   double current_safety = 0;
   double safety = 0;
   VUSolid *TempSolid = NULL;
   double *TempTransform = NULL;
   UVector3 TempPoint;   
     
   for(iIndex = 0 ; iIndex < CarNodes ; iIndex++)   
   {   
      TempSolid = ((*fNodes)[iIndex])->fSolid;   
      TempTransform = ((*fNodes)[iIndex])->fTransform; 

      // Check if considered point is inside current slid:
      TempPoint.x = aPoint.x - TempTransform[0];
      TempPoint.y = aPoint.y - TempTransform[1];
      TempPoint.z = aPoint.z - TempTransform[2];   

      if(TempSolid->Inside(TempPoint) == eInside)
      {
         current_safety = TempSolid->SafetyFromInside(TempPoint,aAccurate);
      
         if(jIndex == 0)
         {
            safety = current_safety;
         }
      
         if(current_safety < safety)
         {
            safety = current_safety;
         }
         
         jIndex++;
         
      }
   }
   return safety;
}

//______________________________________________________________________________
double UMultiUnion::SafetyFromOutside ( const UVector3 aPoint, 
                bool aAccurate) const
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
void UMultiUnion::Voxelize()
{
// Build voxels structure.
   cout << "Voxelize - Not implemented" << endl;
}
