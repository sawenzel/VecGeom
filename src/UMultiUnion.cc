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
   srand((unsigned int)time(NULL));

   double extentMin[3], extentMax[3];
   UVector3 tempPoint;
   double dX, dY, dZ, oX, oY, oZ;
   int iGenerated = 0, iInside = 0;
   
   Extent(extentMin,extentMax);

   dX = (extentMax[0] - extentMin[0])/2;
   dY = (extentMax[1] - extentMin[1])/2;
   dZ = (extentMax[2] - extentMin[2])/2;
   
   oX = (extentMax[0] + extentMin[0])/2;
   oY = (extentMax[1] + extentMin[1])/2;
   oZ = (extentMax[2] + extentMin[2])/2;     
   
   double vbox = (2*dX)*(2*dY)*(2*dZ);
   
   while (iInside < 100000)
   {
      tempPoint.Set(oX - dX + 2*dX*(rand()/(double)RAND_MAX),
                    oY - dY + 2*dY*(rand()/(double)RAND_MAX),
                    oZ - dZ + 2*dZ*(rand()/(double)RAND_MAX));
      iGenerated++;
      if(Inside(tempPoint) != eOutside) iInside++;
   }
   double capacity = iInside*vbox/iGenerated;
   return capacity;      
}

//______________________________________________________________________________
void UMultiUnion::ComputeBBox (UBBox * /*aBox*/, bool /*aStore*/)
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

   UVector3 direction = aDirection.Unit();   
   int carNodes = fNodes->size();
   
   VUSolid *tempSolid = 0;
   UTransform3D *tempTransform = 0;     
 
   double resultDistToIn = UUtils::kInfinity;   
   for(int i = 0 ; i < carNodes ; i++)
   {
      UVector3 tempPointConv, tempDirConv;      
   
      tempSolid = ((*fNodes)[i])->fSolid;
      tempTransform = ((*fNodes)[i])->fTransform;
      
      tempPointConv = tempTransform->LocalPoint(aPoint);
      tempDirConv = tempTransform->LocalVector(direction);                 
         
      double temp = tempSolid->DistanceToIn(tempPointConv, tempDirConv, aPstep);         
      if(temp < resultDistToIn) resultDistToIn = temp;
   }
   return resultDistToIn;
}

//______________________________________________________________________________
double UMultiUnion::DistanceToInVoxels(const UVector3 &aPoint, 
                                       const UVector3 &aDirection, 
                                    // UVector3 &aNormal, 
                                       double aPstep) const
{
   // This method should be the optimized method of "DistanceToIn". It uses
   // voxelization techniques. Some tests have shown it works for many cases,
   // but this method is not validated by ROOT bridge tests
   // TO BE REVIEWED 

   UVector3 direction = aDirection.Unit();       
   vector<int> vectorOutcome;     
  
   fVoxels->GetCandidatesVoxelArray(aPoint, vectorOutcome);
   UVector3 tempGlobal, tempRot;
   
   int incDir[3], outcomeBinarySearch[3];
   double invDir[3], distance[3];
   double maxDistance = 0;
   double minDistance = UUtils::kInfinity;
   double distanceTemp = 0;
   UVector3 newPoint, currentPoint;
        
   incDir[0] = incDir[1] = incDir[2] = 0;      
   int carX = fVoxels->GetNumSlices(eXaxis);
   int carY = fVoxels->GetNumSlices(eYaxis);
   int carZ = fVoxels->GetNumSlices(eZaxis);   
    
   double* xBound = fVoxels->GetXSortedBoundaries();
   double* yBound = fVoxels->GetYSortedBoundaries();
   double* zBound = fVoxels->GetZSortedBoundaries();

   currentPoint = aPoint;
	while(vectorOutcome.size() == 0)
   {  
      if(currentPoint.x < xBound[0] || currentPoint.y < yBound[0] || currentPoint.z < zBound[0] ||
         currentPoint.x > xBound[carX - 1] || currentPoint.y > yBound[carY - 1] || currentPoint.z > zBound[carZ - 1])
      {           
         distance[0] = distance[1] = distance[2] = 0;
             
         // X axis   
         invDir[0] = 1e30;
         if(std::abs(direction.x) >= 1e-10)
         {
            incDir[0] = (direction.x > 0) ? 1 : -1;
            invDir[0] = 1/direction.x;
         }         
         outcomeBinarySearch[0] = fVoxels->OutcomeBinarySearch(currentPoint.x,eXaxis);
        
         if( ((outcomeBinarySearch[0] <= 0) && (incDir[0] < 0)) ||
             ((outcomeBinarySearch[0] == (carX - 1)) && (incDir[0] > 0)))
         {
            return UUtils::kInfinity;
         }

         // Y axis
         invDir[1] = 1e30;
         if(std::abs(direction.y) >= 1e-10)
         {
            incDir[1] = (direction.y > 0)?1:-1;
            invDir[1] = 1/direction.y;
         }
         outcomeBinarySearch[1] = fVoxels->OutcomeBinarySearch(currentPoint.y,eYaxis);
   
         if( ((outcomeBinarySearch[1] <= 0) && (incDir[1] < 0)) ||
             ((outcomeBinarySearch[1] == (carY - 1)) && (incDir[1] > 0)) )
         {
            return UUtils::kInfinity;
         }

         // Z axis      
         invDir[2] = 1e30;
         if(std::abs(direction.z) >= 1e-10)
         {
            incDir[2] = (direction.z > 0)?1:-1;
            invDir[2] = 1/direction.z;
         }      
         outcomeBinarySearch[2] = fVoxels->OutcomeBinarySearch(currentPoint.z,eZaxis);            

         if( ((outcomeBinarySearch[2] <= 0) && (incDir[2] < 0)) ||
             ((outcomeBinarySearch[2] == (carZ - 1)) && (incDir[2] > 0)) )
         {
            return UUtils::kInfinity;
         }
        
         // Looking for the first voxel on the considered direction   
         if((currentPoint.x < xBound[0]) && incDir[0] == 1)
         {
            distance[0] = (xBound[0] - currentPoint.x)*invDir[0];
         }
         else if((currentPoint.x > xBound[carX - 1]) && incDir[0] == -1)
         {
            distance[0] = (xBound[carX - 1] - currentPoint.x)*invDir[0];         
         }
            
         if((currentPoint.y < yBound[0]) && incDir[1] == 1)
         {
            distance[1] = (yBound[0] - currentPoint.y)*invDir[1];
         }
         else if((currentPoint.y > yBound[carY - 1]) && incDir[1] == -1)
         {
            distance[1] = (yBound[carY - 1] - currentPoint.y)*invDir[1];         
         }      
      
         if((currentPoint.z < zBound[0]) && incDir[2] == 1)
         {
            distance[2] = (zBound[0] - currentPoint.z)*invDir[2];
         }
         else if((currentPoint.z > zBound[carZ - 1]) && incDir[2] == -1)
         {
            distance[2] = (zBound[carZ - 1] - currentPoint.z)*invDir[2];         
         }                  
   
         // Computing the max
         for(int i = 0 ; i < 3 ; i++)
         {
            if(distance[i] > maxDistance) maxDistance = distance[i];
         }                        
            
         newPoint.Set(currentPoint.x+direction.x*maxDistance,
                      currentPoint.y+direction.y*maxDistance,
                      currentPoint.z+direction.z*maxDistance);
                       
         cout << "newPoint: [" << newPoint.x << " , " << newPoint.y << " , " << newPoint.z << "]" << endl;
         
         currentPoint = newPoint;
         distanceTemp += maxDistance;
         fVoxels->GetCandidatesVoxelArray(currentPoint, vectorOutcome);
      }
      else
      {            
         outcomeBinarySearch[0] = fVoxels->OutcomeBinarySearch(currentPoint.x,eXaxis);    
         outcomeBinarySearch[1] = fVoxels->OutcomeBinarySearch(currentPoint.y,eYaxis);    
         outcomeBinarySearch[2] = fVoxels->OutcomeBinarySearch(currentPoint.z,eZaxis);

         distance[0] = distance[1] = distance[2] = UUtils::kInfinity;
         
         if(incDir[0] == 1 && outcomeBinarySearch[0] != carX - 1)
         {
            distance[0] = (xBound[outcomeBinarySearch[0]+1] - currentPoint.x)*invDir[0];
         }
         else if(incDir[0] == -1 && outcomeBinarySearch[0] != 0)
         {
            distance[0] = (xBound[outcomeBinarySearch[0]-1] - currentPoint.x)*invDir[0];         
         }
     
         if(incDir[1] == 1 && outcomeBinarySearch[1] != carY - 1)
         {
            distance[1] = (yBound[outcomeBinarySearch[1]+1] - currentPoint.y)*invDir[1];
         }
         else if(incDir[1] == -1 && outcomeBinarySearch[1] != 0)
         {
            distance[1] = (yBound[outcomeBinarySearch[1]-1] - currentPoint.y)*invDir[1];         
         }     
         
         if(incDir[2] == 1 && outcomeBinarySearch[2] != carZ - 1)
         {
            distance[2] = (yBound[outcomeBinarySearch[2]+1] - currentPoint.z)*invDir[2];
         }
         else if(incDir[2] == -1 && outcomeBinarySearch[2] != 0)
         {
            distance[2] = (xBound[outcomeBinarySearch[2]-1] - currentPoint.z)*invDir[2];         
         }
   
         // Computing the min
         for(int i = 0 ; i < 3 ; i++)
         {
            if(distance[i] < minDistance) minDistance = distance[i];
         }
    
         newPoint.Set(currentPoint.x+direction.x*minDistance,
                      currentPoint.y+direction.y*minDistance,
                      currentPoint.z+direction.z*minDistance);
                      
         cout << "newPoint2: [" << newPoint.x << " , " << newPoint.y << " , " << newPoint.z << "]" << endl; 
         
         currentPoint = newPoint;
         distanceTemp += minDistance;         
         fVoxels->GetCandidatesVoxelArray(currentPoint, vectorOutcome);
      }                           
   }
   while(vectorOutcome.size() == 0);   
   return distanceTemp;      

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

   // In the case the considered point is located inside the UMultiUnion structure,
   // the treatments are as follows:
   //      - investigation of the candidates for the passed point
   //      - progressive moving of the point towards the surface, along the passed direction
   //      - processing of the normal
   
   UVector3 direction = aDirection.Unit();       
   vector<int> vectorOutcome;
   double localTolerance = 1E-5;       
 
   VUSolid *tempSolid = 0;
   UTransform3D *tempTransform = 0;   
   
   double dist = 0, tempDist = 0;
   
   UVector3 tempPointConv, tempDirConv, tempNormal;
   int forbiddenNode = -1;
   bool dtobool = true;   
  
   fVoxels->GetCandidatesVoxelArray(aPoint, vectorOutcome); 
   UVector3 tempGlobal, tempRot, pointTemp;

   // For the normal case for which the point is inside, after "else if" statement
   if(vectorOutcome.size() == 0)
   {
      for(int i = -1 ; i <= 1 ; i +=2)
      {
         for(int j = -1 ; j <= 1 ; j +=2)
         {
            for(int k = -1 ; k <= 1 ; k +=2)
            {
               tempGlobal.Set(aPoint.x+i*localTolerance, aPoint.y+j*localTolerance,
                              aPoint.z+k*localTolerance);
                             
               fVoxels->GetCandidatesVoxelArray(tempGlobal, vectorOutcome);
               
               if(vectorOutcome.size() != 0)
               {
                  do
                  {                     
					  int limit = vectorOutcome.size();
                     for(i = 0 ; i < limit ; i++)
                     {     
                        if(vectorOutcome[i] == forbiddenNode) continue;         
                                
                        tempSolid = ((*fNodes)[vectorOutcome[i]])->fSolid;
                        tempTransform = ((*fNodes)[vectorOutcome[i]])->fTransform;             
            
                        // The coordinates of the point are modified so as to fit the intrinsic solid local frame:
                        tempPointConv = tempTransform->LocalPoint(tempGlobal);
                        tempDirConv = tempTransform->LocalVector(direction);            
            
                        if(tempSolid->Inside(tempPointConv) != eOutside)
                        {
                           forbiddenNode = vectorOutcome[i];            
                        
                           tempDist = tempSolid->DistanceToOut(tempPointConv,tempDirConv,tempNormal,dtobool,0.);
                           dist += tempDist;
                           tempGlobal = tempTransform->GlobalPoint(tempPointConv+tempDist*tempDirConv);                                          
            
                           // Treatment of Normal
                           tempRot = tempTransform->GlobalVector(tempNormal);     
               
                           if(Inside(tempGlobal) != eInside)
                           {
                              aNormal = tempRot;
                              return dist;               
                           }
                        
                           vectorOutcome.clear();   
                           fVoxels->GetCandidatesVoxelArray(tempGlobal, vectorOutcome);                   
                           break;
                        }
                     }
                  }
                  while(SafetyFromInside(tempGlobal,true) > 1E-5);     
                  aNormal = tempRot;
                  return dist;                                           
               }      
            }
         }
      }
      return 0;     
   }
   else
   { 
      tempGlobal = aPoint;
           
      do
      {
		  int limit = vectorOutcome.size();
         for(int i = 0 ; i < limit ; i++)
         {     
            if(vectorOutcome[i] == forbiddenNode) continue;         
                    
            tempSolid = ((*fNodes)[vectorOutcome[i]])->fSolid;
            tempTransform = ((*fNodes)[vectorOutcome[i]])->fTransform;             

            // The coordinates of the point are modified so as to fit the intrinsic solid local frame:
            tempPointConv = tempTransform->LocalPoint(tempGlobal);
            tempDirConv = tempTransform->LocalVector(direction);            

            if(tempSolid->Inside(tempPointConv) != eOutside)
            {
               forbiddenNode = vectorOutcome[i];            
            
               tempDist = tempSolid->DistanceToOut(tempPointConv,tempDirConv,tempNormal,dtobool,0.);
               dist += tempDist;
               tempGlobal = tempTransform->GlobalPoint(tempPointConv+tempDist*tempDirConv);                                          

               // Treatment of Normal
               tempRot = tempTransform->GlobalVector(tempNormal);
               
               if(Inside(tempGlobal) != eInside)
               {
                  aNormal = tempRot;
                  return dist;               
               }
               vectorOutcome.clear();   
			   fVoxels->GetCandidatesVoxelArray(tempGlobal, vectorOutcome);                       
               break;
            }
         }
      }
      while(SafetyFromInside(tempGlobal,true) > 1E-5);     
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

//	return InsideDummy(aPoint);
	
	#ifdef DEBUG
		VUSolid::EnumInside insideDummy = InsideDummy(aPoint);
	#endif
      
   UVector3 tempPointConv;
   VUSolid::EnumInside tempInside = eOutside;
   bool boolSurface = false;
   
   vector<int> vectorOutcome;
   vectorOutcome = fVoxels->GetCandidatesVoxelArrayOld(aPoint); 
   fVoxels->GetCandidatesVoxelArray(aPoint, vectorOutcome);

   int limit = vectorOutcome.size();
   for(int i = 0 ; i < limit ; i++)
   {
      VUSolid *tempSolid = ((*fNodes)[vectorOutcome[i]])->fSolid;
      UTransform3D *tempTransform = ((*fNodes)[vectorOutcome[i]])->fTransform;  
            
      // The coordinates of the point are modified so as to fit the intrinsic solid local frame:
      tempPointConv = tempTransform->LocalPoint(aPoint);

      tempInside = tempSolid->Inside(tempPointConv);        
      
      if(tempInside == eSurface) boolSurface = true; 
      
      if(tempInside == eInside) 
	  {
		   #ifdef DEBUG 
			if (insideDummy != tempInside)
				insideDummy = tempInside; // you can place a breakpoint here
			#endif

		  return eInside;      
	  }
   }          
///////////////////////////////////////////////////////////////////////////
// Important comment: When two solids touch each other along a flat
// surface, the surface points will be considered as eSurface, while points 
// located around will correspond to eInside (cf. G4UnionSolid in GEANT4)
   tempInside = boolSurface ? eSurface : eOutside;

#ifdef DEBUG
   if (insideDummy != tempInside)
	   insideDummy = tempInside; // you can place a breakpoint here
#endif
   return tempInside;
}

//______________________________________________________________________________
VUSolid::EnumInside UMultiUnion::InsideDummy(const UVector3 &aPoint) const
{
   int carNodes = fNodes->size();
     
   VUSolid *tempSolid = 0;
   UTransform3D *tempTransform = 0;
   
   UVector3 tempPointConv;
   VUSolid::EnumInside tempInside = eOutside;
   int countSurface = 0;
         
   for(int i = 0 ; i < carNodes ; i++)
   {
      tempSolid = ((*fNodes)[i])->fSolid;
      tempTransform = ((*fNodes)[i])->fTransform;  
            
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
   double mini = 0, maxi = 0;  

   VUSolid *tempSolid = 0;
   UTransform3D *tempTransform = 0;
  
   double arrMin[3], arrMax[3];
   UVector3 min, max;
    
   // Loop on the nodes:
   for(int i = 0 ; i < carNodes ; i++)
   {
      tempSolid = ((*fNodes)[i])->fSolid;
      tempTransform = ((*fNodes)[i])->fTransform;
      
      tempSolid->Extent(arrMin, arrMax);
      min.Set(arrMin[0],arrMin[1],arrMin[2]);      
      max.Set(arrMax[0],arrMax[1],arrMax[2]);           
      UUtils::TransformLimits(min, max, tempTransform);
     
      if(i == 0)
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
   Extent(eXaxis,min,max);
   aMin[0] = min; aMax[0] = max;
   Extent(eYaxis,min,max);
   aMin[1] = min; aMax[1] = max;
   Extent(eZaxis,min,max);
   aMin[2] = min; aMax[2] = max;      
}

//______________________________________________________________________________
bool UMultiUnion::Normal(const UVector3& aPoint, UVector3 &aNormal)
{
// Computes the normal on a surface and returns it as a unit vector
//   In case a point is further than toleranceNormal from a surface, set validNormal=false
//   Must return a valid vector. (even if the point is not on the surface.)
//
//   On an edge or corner, provide an average normal of all facets within tolerance
// NOTE: the tolerance value used in here is not yet the global surface
//     tolerance - we will have to revise this value - TODO

   vector<int> vectorOutcome;   
   VUSolid *tempSolid = 0;
   UTransform3D *tempTransform = 0;
   UVector3 tempPointConv, tempRot, outcomeNormal;
   double tempSafety = UUtils::kInfinity;
   int tempNodeSafety = -1;
   int searchNode;
   
   double normalTolerance = 1E-5;

///////////////////////////////////////////////////////////////////////////
// Important comment: Cases for which the point is located on an edge or
// on a vertice remain to be treated  
       
   fVoxels->GetCandidatesVoxelArray(aPoint, vectorOutcome); 

   if(vectorOutcome.size() > 0)
   {
	   int limit = vectorOutcome.size();
      for(int i = 0 ; i < limit ; i++)
      {
         tempSolid = ((*fNodes)[vectorOutcome[i]])->fSolid;
         tempTransform = ((*fNodes)[vectorOutcome[i]])->fTransform;  
                  
         // The coordinates of the point are modified so as to fit the intrinsic solid local frame:
         tempPointConv = tempTransform->LocalPoint(aPoint);
   
		 VUSolid::EnumInside location = tempSolid->Inside(tempPointConv);
         if(location == eSurface)
         {
            tempSolid->Normal(tempPointConv,outcomeNormal);
         
            tempRot = tempTransform->GlobalVector(outcomeNormal);
            
            aNormal = tempRot.Unit();           
            return true;  
         }
         else
         {
            if(location == eInside)
            {
               if(tempSolid->SafetyFromInside(tempPointConv) < tempSafety)
               {
                  tempSafety = tempSolid->SafetyFromInside(tempPointConv);
                  tempNodeSafety = vectorOutcome[i];
               }
            }
            else // case eOutside
            {
               if(tempSolid->SafetyFromOutside(tempPointConv) < tempSafety)
               {
                  tempSafety = tempSolid->SafetyFromOutside(tempPointConv);
                  tempNodeSafety = vectorOutcome[i];
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
   else
   {
      searchNode = SafetyFromOutsideNumberNode(aPoint,true);
      tempSolid = ((*fNodes)[searchNode])->fSolid;
      tempTransform = ((*fNodes)[searchNode])->fTransform;            
      tempPointConv = tempTransform->LocalPoint(aPoint);
   
      tempSolid->Normal(tempPointConv,outcomeNormal);
      
      tempRot = tempTransform->GlobalVector(outcomeNormal);
      
      aNormal = tempRot.Unit();
      if (tempSafety > normalTolerance) return false;
      return true;     
   }
}

//______________________________________________________________________________ 
double UMultiUnion::SafetyFromInside(const UVector3 &aPoint, bool aAccurate) const
{
   // Estimates isotropic distance to the surface of the solid. This must
   // be either accurate or an underestimate. 
   //  Two modes: - default/fast mode, sacrificing accuracy for speed
   //             - "precise" mode,  requests accurate value if available.   

   vector<int> vectorOutcome;     
   VUSolid *tempSolid = 0;
   UTransform3D *tempTransform = 0;
   
   UVector3 tempPointConv;
   double safetyMax = 0.;
   
   // In general, the value return by SafetyFromInside will not be the exact
   // but only an undervalue (cf. overlaps)    
   fVoxels->GetCandidatesVoxelArray(aPoint, vectorOutcome);
   
   int limit = vectorOutcome.size();
   for(int i = 0 ; i < limit ; i++)
   {      
      tempSolid = ((*fNodes)[vectorOutcome[i]])->fSolid;
      tempTransform = ((*fNodes)[vectorOutcome[i]])->fTransform;  
            
      // The coordinates of the point are modified so as to fit the intrinsic solid local frame:
      tempPointConv = tempTransform->LocalPoint(aPoint);
         
      if(tempSolid->Inside(tempPointConv) == eInside)
      {
         double safetyTemp = tempSolid->SafetyFromInside(tempPointConv,aAccurate);
         if(safetyTemp > safetyMax) safetyMax = safetyTemp;         
      }   
   }
   return safetyMax;   
}

//______________________________________________________________________________
double UMultiUnion::SafetyFromOutside(const UVector3 &aPoint, bool aAccurate) const
{
   // Estimates the isotropic safety from a point outside the current solid to any 
   // of its surfaces. The algorithm may be accurate or should provide a fast 
   // underestimate.
   
   int carNodes = fNodes->size();
   VUSolid *tempSolid = 0;
   UTransform3D *tempTransform = 0;     
      
   double *boxes = fVoxels->GetBoxes();
   double safety = UUtils::kInfinity;   
   UVector3 tempPointConv;
   for(int i = 0 ; i < carNodes ; i++)
   {
      tempSolid = ((*fNodes)[i])->fSolid;
      tempTransform = ((*fNodes)[i])->fTransform;
      tempPointConv = tempTransform->LocalPoint(aPoint);      
  
      int ist = 6*i;
      double dxyz0 = std::abs(aPoint.x-boxes[ist+3])-boxes[ist];
      if (dxyz0 > safety) continue;
      double dxyz1 = std::abs(aPoint.y-boxes[ist+4])-boxes[ist+1];
      if (dxyz1 > safety) continue;
      double dxyz2 = std::abs(aPoint.z-boxes[ist+5])-boxes[ist+2];      
      if (dxyz2 > safety) continue;

      double d2xyz = 0.;
      if(dxyz0>0) d2xyz+=dxyz0*dxyz0;
      if(dxyz1>0) d2xyz+=dxyz1*dxyz1;
      if(dxyz2>0) d2xyz+=dxyz2*dxyz2;
      if(d2xyz >= safety*safety) continue;
      
      double safetyTemp = tempSolid->SafetyFromOutside(tempPointConv,true);
    
      if (safetyTemp < safety) safety = safetyTemp;
   }
   return safety;
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
   fVoxels->Voxelize();
}

//______________________________________________________________________________
int UMultiUnion::SafetyFromOutsideNumberNode(const UVector3 &aPoint, bool aAccurate) const
{
   // Method returning the closest node from a point located outside a UMultiUnion.
   // This is used to compute the normal in the case no candidate has been found.
   
   int carNodes = fNodes->size();
   
   VUSolid *tempSolid = 0;
   UTransform3D *tempTransform = 0;     
      
   double *boxes = fVoxels->GetBoxes();
   double safety = UUtils::kInfinity;
   int safetyNode = -1;
   UVector3 tempPointConv;    

   for(int i = 0; i < carNodes; i++)
   {
      tempSolid = ((*fNodes)[i])->fSolid;
      tempTransform = ((*fNodes)[i])->fTransform;
      
      tempPointConv = tempTransform->LocalPoint(aPoint);      
  
      int ist = 6*i;
      double d2xyz = 0.;
      
      double dxyz0 = std::abs(aPoint.x-boxes[ist+3])-boxes[ist];
      if (dxyz0 > safety) continue;
      double dxyz1 = std::abs(aPoint.y-boxes[ist+4])-boxes[ist+1];
      if (dxyz1 > safety) continue;
      double dxyz2 = std::abs(aPoint.z-boxes[ist+5])-boxes[ist+2];      
      if (dxyz2 > safety) continue;
      
      if(dxyz0>0) d2xyz+=dxyz0*dxyz0;
      if(dxyz1>0) d2xyz+=dxyz1*dxyz1;
      if(dxyz2>0) d2xyz+=dxyz2*dxyz2;
      if(d2xyz >= safety*safety) continue;
      
      double safetyTemp = tempSolid->SafetyFromOutside(tempPointConv,true);
    
      if(safetyTemp < safety)
      {
         safety = safetyTemp;
         safetyNode = i;
      }
   }
   return safetyNode;
}
