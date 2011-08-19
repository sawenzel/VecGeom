#include "UMultiUnion.hh"

#include <iostream>
#include "UUtils.hh"
#include "UVoxelFinder.hh"
#include <sstream>

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
   // Capacity computes the cubic volume of the "UMultiUnion" structure using random points

   // Random initialization:
   srand(time(NULL));

   double* extentMin = new double[3];
   double* extentMax = new double[3];
   UVector3 tempPoint;
   double dX, dY, dZ, oX, oY, oZ;
   int iGenerated = 0, iInside = 0;
   
   this->Extent(extentMin,extentMax);

   dX = (extentMax[0] - extentMin[0])/2;
   dY = (extentMax[1] - extentMin[1])/2;
   dZ = (extentMax[2] - extentMin[2])/2;
   
   oX = (extentMax[0] + extentMin[0])/2;
   oY = (extentMax[1] + extentMin[1])/2;
   oZ = (extentMax[2] + extentMin[2])/2;     
   
   double vbox = (2*dX)*(2*dY)*(2*dZ);
   
   while(iInside < 100000)
   {
      tempPoint.Set(oX - dX + 2*dX*(rand()/(double)RAND_MAX),
                    oY - dY + 2*dY*(rand()/(double)RAND_MAX),
                    oZ - dZ + 2*dZ*(rand()/(double)RAND_MAX));
      iGenerated++;
      if(this->Inside(tempPoint) != eOutside) iInside++;
   }
   double capacity = iInside*vbox/iGenerated;
   return capacity;      
}

//______________________________________________________________________________
void UMultiUnion::ComputeBBox (UBBox */*aBox*/, bool /*aStore*/)
{
// Computes bounding box.
   cout << "ComputeBBox - Not implemented" << endl;
}   

//______________________________________________________________________________
double UMultiUnion::DistanceToIn(const UVector3 &aPoint, 
                          const UVector3 &aDirection, 
                       // UVector3 &aNormal, 
                          double aPstep) const
{
// Computes distance from a point presumably outside the solid to the solid 
// surface. Ignores first surface if the point is actually inside. Early return
// infinity in case the safety to any surface is found greater than the proposed
// step aPstep.
// The normal vector to the crossed surface is filled only in case the box is 
// crossed, otherwise aNormal.IsNull() is true.

   if(this->Inside(aPoint) != eOutside)
   {
      cout << "Point is not outside UMultiUnion... ERROR" << endl;
      return -1;
   }
   // When the considered point is outside, the nearest surface is looked after
   else
   {
      int iIndex;
      int carNodes = fNodes->size();
   
      VUSolid *tempSolid = 0;
      UTransform3D *tempTransform = 0;     
   
      double resultDistToIn = UUtils::kInfinity;
      double temp;     
   
      for(iIndex = 0 ; iIndex < carNodes ; iIndex++)
      {
         UVector3 tempPointConv, tempDirConv;      
      
         tempSolid = ((*fNodes)[iIndex])->fSolid;
         tempTransform = ((*fNodes)[iIndex])->fTransform;
      
         tempPointConv = tempTransform->LocalPoint(aPoint);
         tempDirConv = tempTransform->LocalVector(aDirection);                 
         
         temp = tempSolid->DistanceToIn(tempPointConv, tempDirConv, aPstep);         
         if(temp < resultDistToIn) resultDistToIn = temp;
      }
      return resultDistToIn;
   }
}     

//______________________________________________________________________________
double UMultiUnion::DistanceToOut(const UVector3 &aPoint, const UVector3 &aDirection,
			       UVector3 &aNormal,
			       bool     &convex,
                double   aPstep) const
{
// Computes distance from a point presumably intside the solid to the solid 
// surface. Ignores first surface along each axis systematically (for points
// inside or outside. Early returns zero in case the second surface is behind
// the starting point.
// o The proposed step is ignored.
// o The normal vector to the crossed surface is always filled.

   if(this->Inside(aPoint) != eInside)
   {
      cout << "Point is not inside UMultiUnion... ERROR" << endl;
      return -1;
   }
   // In the case the considered point is located inside the UMultiUnion structure,
   // the treatments are as follows:
   //      - investigation of the candidates for the passed point
   //      - progressive moving of the point towards the surface, along the passe direction
   //      - processing of the normal
   else
   {     
      int iIndex;
      vector<int> vectorOutcome;   
 
      VUSolid *tempSolid = 0;
      UTransform3D *tempTransform = 0;   
   
      double dist = 0, tempDist = 0;
   
      vectorOutcome = fVoxels -> GetCandidatesVoxelArray(aPoint); 
      UVector3 tempGlobal, tempRot;

      do
      {      
         for(iIndex = 0 ; iIndex < (int)vectorOutcome.size() ; iIndex++)
         {
            UVector3 tempPointConv, tempDirConv;
            UVector3 tempNormal;
            bool dtobool = true;
         
            tempSolid = ((*fNodes)[vectorOutcome[iIndex]])->fSolid;
            tempTransform = ((*fNodes)[vectorOutcome[iIndex]])->fTransform;             

            // The coordinates of the point are modified so as to fit the intrinsic solid local frame:
            tempPointConv = tempTransform->LocalPoint(aPoint);
            tempDirConv = tempTransform->LocalVector(aDirection);            
                       
            if(tempSolid->Inside(tempPointConv+dist*tempDirConv) != eOutside)
            {                       
               tempDist = tempSolid->DistanceToOut(tempPointConv+dist*tempDirConv,tempDirConv,tempNormal,dtobool,0.); 
               dist += tempDist;
               tempGlobal = tempTransform->GlobalPoint(tempPointConv+dist*tempDirConv);

               // Treatment of Normal - To Do
               tempRot = tempTransform->GlobalVector(tempNormal);
            }
         }
         vectorOutcome.clear();   
         vectorOutcome = fVoxels -> GetCandidatesVoxelArray(tempGlobal);
      }
      while(this->Inside(tempGlobal) == eInside);
      aNormal = tempRot;
      return dist;
   }
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

   // Implementation using voxelisation techniques:
   // ---------------------------------------------
//   return InsideDummy(aPoint);
   
   int iIndex;
   vector<int> vectorOutcome;   
 
   VUSolid *tempSolid = 0;
   UTransform3D *tempTransform = 0;
   
   UVector3 tempPointConv;
   VUSolid::EnumInside tempInside = eOutside;
   bool boolSurface = false;
         
   vectorOutcome = fVoxels -> GetCandidatesVoxelArray(aPoint); 

   for(iIndex = 0 ; iIndex < (int)vectorOutcome.size() ; iIndex++)
   {
      tempSolid = ((*fNodes)[vectorOutcome[iIndex]])->fSolid;
      tempTransform = ((*fNodes)[vectorOutcome[iIndex]])->fTransform;  
            
      // The coordinates of the point are modified so as to fit the intrinsic solid local frame:
      tempPointConv = tempTransform->LocalPoint(aPoint);
     
      tempInside = tempSolid->Inside(tempPointConv);        
      
      if(tempInside == eSurface) boolSurface = true; 
      
      if(tempInside == eInside) return eInside;      
   }          
///////////////////////////////////////////////////////////////////////////
// Important comment: When two solids touch each other along a flat
// surface, the surface points will be considered as eSurface, while points 
// located around will correspond to eInside (cf. G4UnionSolid in GEANT4)
   if(boolSurface == true) return eSurface;
   return eOutside;
}

//______________________________________________________________________________
VUSolid::EnumInside UMultiUnion::InsideDummy(const UVector3 &aPoint) const
{
   int iIndex;
   int carNodes = fNodes->size();
     
   VUSolid *tempSolid = 0;
   UTransform3D *tempTransform = 0;
   
   UVector3 tempPointConv;
   VUSolid::EnumInside tempInside = eOutside;
   int countSurface = 0;
         
   for(iIndex = 0 ; iIndex < carNodes ; iIndex++)
   {
      tempSolid = ((*fNodes)[iIndex])->fSolid;
      tempTransform = ((*fNodes)[iIndex])->fTransform;  
            
      // The coordinates of the point are modified so as to fit the intrinsic solid local frame:
      tempPointConv = tempTransform->LocalPoint(aPoint);
     
      tempInside = tempSolid->Inside(tempPointConv);        
      
      if(tempInside == eSurface) countSurface++; 
      
      if(tempInside == eInside) return eInside;      
   }       
   if(countSurface != 0) return eSurface;
   return eOutside;
}

//______________________________________________________________________________ 
void UMultiUnion::Extent(EAxisType aAxis,double &aMin,double &aMax)
{
// Determines the bounding box for the considered instance of "UMultipleUnion"
   int carNodes = fNodes->size();  
   int iIndex = 0;
   double mini = 0, maxi = 0;  

   VUSolid *tempSolid = 0;
   UTransform3D *tempTransform = 0;
  
   double* arrMin = new double[3];
   double* arrMax = new double[3];
   
   UVector3 min, max;
    
   // Loop on the nodes:
   for(iIndex = 0 ; iIndex < carNodes ; iIndex++)
   {
      tempSolid = ((*fNodes)[iIndex])->fSolid;
      tempTransform = ((*fNodes)[iIndex])->fTransform;
      
      tempSolid->Extent(arrMin, arrMax);
      min.Set(arrMin[0],arrMin[1],arrMin[2]);      
      max.Set(arrMax[0],arrMax[1],arrMax[2]);           
      UUtils::TransformLimits(min, max, tempTransform);
     
      if(iIndex == 0)
      {
         switch(aAxis)
         {
            case eXaxis:
            {
               mini = min.x;
               maxi = max.x;            
               break;
            }
            case eYaxis:
            {
               mini = min.y;
               maxi = max.y;            
               break;
            }
            case eZaxis:
            {
               mini = min.z;
               maxi = max.z;            
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
            if(min.x < mini)
               mini = min.x;
            if(max.x > maxi)
               maxi = max.x;
            break;
         }
         case eYaxis:
         {
            if(min.y < mini)
               mini = min.y;
            if(max.y > maxi)
               maxi = max.y;
            break;
         }
         case eZaxis:
         {
            if(min.z < mini)
               mini = min.z;
            if(max.z > maxi)
               maxi = max.z;
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
   double min = 0,max = 0;
   this->Extent(eXaxis,min,max);
   aMin[0] = min; aMax[0] = max;
   this->Extent(eYaxis,min,max);
   aMin[1] = min; aMax[1] = max;
   this->Extent(eZaxis,min,max);
   aMin[2] = min; aMax[2] = max;      
}

//______________________________________________________________________________
bool UMultiUnion::Normal(const UVector3& aPoint, UVector3 &aNormal)
{
// Computes the normal on a surface and returns it as a unit vector
//   In case a point is further than tolerance_normal from a surface, set validNormal=false
//   Must return a valid vector. (even if the point is not on the surface.)
//
//   On an edge or corner, provide an average normal of all facets within tolerance
// NOTE: the tolerance value used in here is not yet the global surface
//     tolerance - we will have to revise this value - TODO

   int iIndex;
   vector<int> vectorOutcome;   
   VUSolid *tempSolid = 0;
   UTransform3D *tempTransform = 0;
   UVector3 tempPointConv, tempRot, outcomeNormal;
   double tempSafety = UUtils::kInfinity;
   int tempNodeSafety = 0;
   
   double normalTolerance = 1E-6;

///////////////////////////////////////////////////////////////////////////
// Important comment: Cases for which the point is located on an edge or
// on a vertice remain to be treated  
       
   vectorOutcome = fVoxels -> GetCandidatesVoxelArray(aPoint); 

   for(iIndex = 0 ; iIndex < (int)vectorOutcome.size() ; iIndex++)
   {
      tempSolid = ((*fNodes)[vectorOutcome[iIndex]])->fSolid;
      tempTransform = ((*fNodes)[vectorOutcome[iIndex]])->fTransform;  
               
      // The coordinates of the point are modified so as to fit the intrinsic solid local frame:
      tempPointConv = tempTransform->LocalPoint(aPoint);
         
      if(tempSolid->Inside(tempPointConv) == eSurface)
      {
         tempSolid->Normal(tempPointConv,outcomeNormal);
      
         tempRot = tempTransform->GlobalVector(outcomeNormal);
         
         aNormal = tempRot.Unit();           
         return true;  
      }
      else
      {
         if(tempSolid->Inside(tempPointConv) == eInside)
         {
            if(tempSolid->SafetyFromInside(tempPointConv) < tempSafety)
            {
               tempSafety = tempSolid->SafetyFromInside(tempPointConv);
               tempNodeSafety = vectorOutcome[iIndex];
            }
         }
         else // case eOutside
         {
            if(tempSolid->SafetyFromOutside(tempPointConv) < tempSafety)
            {
               tempSafety = tempSolid->SafetyFromOutside(tempPointConv);
               tempNodeSafety = vectorOutcome[iIndex];
            }    
         }
      }      
   }   
   tempSolid = ((*fNodes)[tempNodeSafety])->fSolid;
   tempTransform = ((*fNodes)[tempNodeSafety])->fTransform;            
   tempPointConv = tempTransform->LocalPoint(aPoint);

   tempSolid->Normal(tempPointConv,outcomeNormal);
   
   tempRot = tempTransform->GlobalVector(outcomeNormal);
   
   aNormal = tempRot.Unit();
   if(tempSafety > normalTolerance) return false;
   return true;     
}

//______________________________________________________________________________ 
double UMultiUnion::SafetyFromInside(const UVector3 aPoint, bool aAccurate) const
{
   // Estimates isotropic distance to the surface of the solid. This must
   // be either accurate or an underestimate. 
   //  Two modes: - default/fast mode, sacrificing accuracy for speed
   //             - "precise" mode,  requests accurate value if available.   

   int iIndex;
   vector<int> vectorOutcome;
     
   VUSolid *tempSolid = 0;
   UTransform3D *tempTransform = 0;
   
   UVector3 tempPointConv;
   double safetyTemp;
   double safetyMax = 0;
   
   if(this->Inside(aPoint) != eInside)
   {
      cout << "Point is not Inside UMultiUnion... ERROR" << endl;
      return -1;
   }
   // In general, the value return by SafetyFromInside will not be the exact
   // but only an undervalue (cf. overlaps)
   else
   {         
      vectorOutcome = fVoxels -> GetCandidatesVoxelArray(aPoint); 
   
      for(iIndex = 0 ; iIndex < (int)vectorOutcome.size() ; iIndex++)
      {      
         tempSolid = ((*fNodes)[vectorOutcome[iIndex]])->fSolid;
         tempTransform = ((*fNodes)[vectorOutcome[iIndex]])->fTransform;  
               
         // The coordinates of the point are modified so as to fit the intrinsic solid local frame:
         tempPointConv = tempTransform->LocalPoint(aPoint);
         
         if(tempSolid->Inside(tempPointConv) == eInside)
         {
            safetyTemp = tempSolid->SafetyFromInside(tempPointConv,aAccurate);
            if(safetyTemp > safetyMax) safetyMax = safetyTemp;         
         }   
      }
      return safetyMax;
   }   
}

//______________________________________________________________________________
double UMultiUnion::SafetyFromOutside(const UVector3 aPoint, bool aAccurate) const
{
   // Estimates the isotropic safety from a point outside the current solid to any 
   // of its surfaces. The algorithm may be accurate or should provide a fast 
   // underestimate.
   
   int iIndex;
   int carNodes = fNodes->size();
   
   VUSolid *tempSolid = 0;
   UTransform3D *tempTransform = 0;     
      
   double *boxes = fVoxels->GetBoxes();
   double safety = UUtils::kInfinity;
   double safetyTemp;
   
   UVector3 tempPointConv;    

   if(this->Inside(aPoint) != eOutside)
   {
      cout << "Point is not Outside UMultiUnion... ERROR" << endl;
      return -1;
   }
   else
   {
      for(iIndex = 0 ; iIndex < carNodes ; iIndex++)
      {
         tempSolid = ((*fNodes)[iIndex])->fSolid;
         tempTransform = ((*fNodes)[iIndex])->fTransform;
      
         tempPointConv = tempTransform->LocalPoint(aPoint);      
    
         int ist = 6*iIndex;
         double d2xyz = 0.;
      
         double dxyz0 = UUtils::Abs(aPoint.x-boxes[ist+3])-boxes[ist];
         if (dxyz0 > safety) continue;
         double dxyz1 = UUtils::Abs(aPoint.y-boxes[ist+4])-boxes[ist+1];
         if (dxyz1 > safety) continue;
         double dxyz2 = UUtils::Abs(aPoint.z-boxes[ist+5])-boxes[ist+2];      
         if (dxyz2 > safety) continue;
      
         if(dxyz0>0) d2xyz+=dxyz0*dxyz0;
         if(dxyz1>0) d2xyz+=dxyz1*dxyz1;
         if(dxyz2>0) d2xyz+=dxyz2*dxyz2;
         if(d2xyz >= safety*safety) continue;
      
         safetyTemp = tempSolid->SafetyFromOutside(tempPointConv,true);
      
         if(safetyTemp < safety) safety = safetyTemp;
      }
      return safety;
   }
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
const UTransform3D* UMultiUnion::GetTransform(int index) const
{
   return(((*fNodes)[index])->fTransform);
}

//______________________________________________________________________________       
void UMultiUnion::SetVoxelFinder(UVoxelFinder* finder)
{
   fVoxels = finder;
}

//______________________________________________________________________________       
void UMultiUnion::Voxelize()
{
   fVoxels = new UVoxelFinder(this);
   fVoxels -> Voxelize();
}
