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
UMultiUnion::~UMultiUnion()
{
   if(fNodes) delete fNodes;
   if(fVoxels) delete fVoxels;
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

   int carNodes = fNodes->size(); // Total number of nodes contained in "fNodes".
   int iIndex = 0;
   
   VUSolid *tempSolid = NULL;
   UTransform3D *tempTransform = NULL;
   UVector3 tempPoint;
   VUSolid::EnumInside tempInside = eOutside;

   // Loop on the nodes:
   for(iIndex = 0 ; iIndex < carNodes ; iIndex++)
   {
      tempSolid = ((*fNodes)[iIndex])->fSolid;
      tempTransform = ((*fNodes)[iIndex])->fTransform;

      // The coordinates of the point are modified so as to fit the intrinsic solid local frame:
      tempPoint.x = aPoint.x - tempTransform->fTr[0];
      tempPoint.y = aPoint.y - tempTransform->fTr[1];
      tempPoint.z = aPoint.z - tempTransform->fTr[2];            

      tempInside = tempSolid->Inside(tempPoint);
      
      if((tempInside == eInside) || (tempInside == eSurface))
      {
         return tempInside;
      }      
   }
   return eOutside;
}

//______________________________________________________________________________ 
void UMultiUnion::Extent(EAxisType aAxis,double &aMin,double &aMax)
{
// Determines the bounding box for the considered instance of "UMultipleUnion"
   int carNodes = fNodes->size();  
   int iIndex = 0;
   double mini, maxi = 0;  

   VUSolid *tempSolid = NULL;
   UTransform3D *tempTransform = NULL;
  
   double* min = new double[3];
   double* max = new double[3];   
    
   // Loop on the nodes:
   for(iIndex = 0 ; iIndex < carNodes ; iIndex++)
   {
      tempSolid = ((*fNodes)[iIndex])->fSolid;
      tempTransform = ((*fNodes)[iIndex])->fTransform;
      
      tempSolid->Extent(min, max);      
      /*UUtils::*/TransformLimits(min, max, tempTransform);
     
      if(iIndex == 0)
      {
         switch(aAxis)
         {
            case eXaxis:
            {
               mini = min[0];
               maxi = max[0];            
               break;
            }
            case eYaxis:
            {
               mini = min[1];
               maxi = max[1];            
               break;
            }
            case eZaxis:
            {
               mini = min[2];
               maxi = max[2];            
               break;
            }                       
         }
         continue;
      }            
     
      // Deternine the min/max on the considered axis:
      switch(aAxis)
      {
         case eXaxis:
         {
            if(min[0] < mini)
               mini = min[0];
            if(max[0] > maxi)
               maxi = max[0];
            break;
         }
         case eYaxis:
         {
            if(min[1] < mini)
               mini = min[1];
            if(max[1] > maxi)
               maxi = max[1];
            break;
         }
         case eZaxis:
         {
            if(min[2] < mini)
               mini = min[2];
            if(max[2] > maxi)
               maxi = max[2];
            break;
         }                        
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
/*const*/ VUSolid* UMultiUnion::GetSolid(int index) /*const*/
{
   return(((*fNodes)[index])->fSolid);
}

//______________________________________________________________________________       

/*const*/ UTransform3D* UMultiUnion::GetTransform(int index) /*const*/
{
   return(((*fNodes)[index])->fTransform);
}

//______________________________________________________________________________
void UMultiUnion::TransformLimits(double *min, double *max, UTransform3D *transformation)
{
   int jIndex, kIndex = 0;
   double *vertices = new double[24]; 
   UVector3 tempPointConv,tempPoint;
   double currentX, currentY, currentZ = 0;
   double miniX, miniY, miniZ, maxiX, maxiY, maxiZ;          

   // Detemination of the vertices thanks to the extension of each solid:
      // 1st vertice:
   vertices[ 0] = min[0]; vertices[ 1] = min[1]; vertices[ 2] = min[2];
      // 2nd vertice:
   vertices[ 3] = min[0]; vertices[ 4] = max[1]; vertices[ 5] = min[2];   
      // etc.:
   vertices[ 6] = max[0]; vertices[ 7] = max[1]; vertices[ 8] = min[2];
   vertices[ 9] = max[0]; vertices[10] = min[1]; vertices[11] = min[2];
   vertices[12] = min[0]; vertices[13] = min[1]; vertices[14] = max[2];
   vertices[15] = min[0]; vertices[16] = max[1]; vertices[17] = max[2];
   vertices[18] = max[0]; vertices[19] = max[1]; vertices[20] = max[2];
   vertices[21] = max[0]; vertices[22] = min[1]; vertices[23] = max[2];   
   
   // Loop on th vertices
   for(jIndex = 0 ; jIndex < 8 ; jIndex++)
   {
      kIndex = 3*jIndex;
      tempPoint.Set(vertices[kIndex],vertices[kIndex+1],vertices[kIndex+2]);
      // From local frame to the gobal one:
      tempPointConv = transformation->GlobalPoint(tempPoint);
      
      // Current positions on the three axis:         
      currentX = tempPointConv.x;
      currentY = tempPointConv.y;
      currentZ = tempPointConv.z;
      
      // Initialization of extrema:
      if(jIndex == 0)
      {
         miniX = maxiX = currentX;
         miniY = maxiY = currentY;
         miniZ = maxiZ = currentZ;
         continue;                  
      }
         
      // If need be, replacement of the min & max values:
      if(currentX > maxiX)
         maxiX = currentX;
      if(currentX < miniX)
         miniX = currentX;

      if(currentY > maxiY)
         maxiY = currentY;
      if(currentY < miniY)
         miniY = currentY;  

      if(currentZ > maxiZ)
         maxiZ = currentZ;
      if(currentZ < miniZ)
         miniZ = currentZ;                             
   }
   // Recopy of the extrema in the passed pointers:
   min[0] = miniX;
   min[1] = miniY;
   min[2] = miniZ;
   max[0] = maxiX;
   max[1] = maxiY;
   max[2] = maxiZ; 
} 
