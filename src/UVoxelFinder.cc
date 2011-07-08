#include "UVoxelFinder.hh"
#include "UMultiUnion.hh"
#include "UBox.hh"
#include <iostream>
#include "UUtils.hh"

//______________________________________________________________________________   
UVoxelFinder::UVoxelFinder()
{
   fMultiUnion = NULL;
   fBoxes = NULL;
}

//______________________________________________________________________________   
UVoxelFinder::UVoxelFinder(UMultiUnion* multi_union)
{
   fMultiUnion = multi_union;
   fBoxes = NULL;
}

//______________________________________________________________________________   
void UVoxelFinder::BuildVoxelLimits()
{
// "BuildVoxelLimits"'s aim is to store the coordinates of the origin as well as
// the half lengths related to the bounding box of each node
   int iIndex,jIndex,kIndex = 0;
   int CarNodes = fMultiUnion->GetNNodes(); // Number of nodes in "fMultiUnion"
   UVector3 TempPointConv,TempPoint;                                                  
   if(!CarNodes) return; // If no node, no need to carry on
   int fNBoxes = 6*CarNodes; // 6 different quantities are stored per node
   if(fBoxes) delete fBoxes;
   fBoxes = new double[fNBoxes]; // Array which will store the half lengths
                                 // related to a particular node, but also
                                 // the coordinates of its origin                                 
      
   VUSolid *TempSolid = NULL;
   UTransform3D *TempTransform = NULL;
   double vertices[24];
   double minX,maxX,minY,maxY,minZ,maxZ;
   
   for(iIndex = 0 ; iIndex < CarNodes ; iIndex++)   
   {
      TempSolid = fMultiUnion->GetSolid(iIndex);
      TempTransform = fMultiUnion->GetTransform(iIndex);
      
      TempSolid->SetVertices(vertices); // Fills in "vertices" in the local
                                        // frame of "TempSolid"
      
      for(jIndex = 0 ; jIndex < 8 ; jIndex++)
      {
         kIndex = 3*jIndex;
         // Store the coordinate of each vertice in "TempPoint"
         TempPoint.Set(vertices[kIndex],vertices[kIndex+1],vertices[kIndex+2]);
         // Conversion of the local frame "TempPoint" to the global frame
         // "TempPointConv"
         TempPointConv = TempTransform->GlobalPoint(TempPoint);
         
         // Initialization of extrema:
         if(jIndex == 0)
         {
            minX = maxX = TempPointConv.x;
            minY = maxY = TempPointConv.y;
            minZ = maxZ = TempPointConv.z;                        
         }
         
         // Extrema research:
         if(TempPointConv.x > maxX)
            maxX = TempPointConv.x;
         if(TempPointConv.x < minX)
            minY = TempPointConv.x;
            
         if(TempPointConv.y > maxY)
            maxY = TempPointConv.y;
         if(TempPointConv.y < minY)
            minY = TempPointConv.y;
            
         if(TempPointConv.z > maxZ)
            maxZ = TempPointConv.z;
         if(TempPointConv.z < minZ)
            minZ = TempPointConv.z;                       

      }
      fBoxes[6*iIndex]   = 0.5*(maxX-minX); // dX
      fBoxes[6*iIndex+1] = 0.5*(maxY-minY); // dY
      fBoxes[6*iIndex+2] = 0.5*(maxZ-minZ); // dZ
      fBoxes[6*iIndex+3] = TempTransform->fTr[0]; // Ox
      fBoxes[6*iIndex+4] = TempTransform->fTr[1]; // Oy
      fBoxes[6*iIndex+5] = TempTransform->fTr[2]; // Oz
   }   
}

//______________________________________________________________________________   
void UVoxelFinder::DisplayVoxelLimits()
{
   int CarNodes = fMultiUnion->GetNNodes();
   int iIndex = 0;
   
   for(iIndex = 0 ; iIndex < CarNodes ; iIndex++)
   {
      printf("    -> Node %d:\n       * dX = %f ; * dY = %f ; * dZ = %f\n       * oX = %f ; * oY = %f ; * oZ = %f\n",iIndex+1,fBoxes[6*iIndex],fBoxes[6*iIndex+1],fBoxes[6*iIndex+2],fBoxes[6*iIndex+3],fBoxes[6*iIndex+4],fBoxes[6*iIndex+5]);
   }
}

//______________________________________________________________________________   
void UVoxelFinder::CreateBoundaries()
{
// "SortAll"'s aim is to determine the slices induced by the bounding boxes,
// along each axis
   int iIndex = 0;
   int CarNodes = fMultiUnion->GetNNodes(); // Number of nodes in structure of
                                            // "UMultiUnion" type
   
   // Determination of the 3D extent of the "UMultIUnion" structure:
   double minima_multi[3], maxima_multi[3];
   fMultiUnion->Extent(minima_multi,maxima_multi);
   
   // Determination of the boundries along x, y and z axis:
   fBoundaries = new double[6*CarNodes];
   
   for(iIndex = 0 ; iIndex < CarNodes ; iIndex++)   
   {
      // For each node, the boundaries are created by using the array "fBoxes"
      // built in method "BuildVoxelLimits":

      // x boundaries
      fBoundaries[2*iIndex] = fBoxes[6*iIndex+3]-fBoxes[6*iIndex];
      fBoundaries[2*iIndex+1] = fBoxes[6*iIndex+3]+fBoxes[6*iIndex];
      // y boundaries
      fBoundaries[2*iIndex+2*CarNodes] = fBoxes[6*iIndex+4]-fBoxes[6*iIndex+1];
      fBoundaries[2*iIndex+2*CarNodes+1] = fBoxes[6*iIndex+4]+fBoxes[6*iIndex+1];
      // z boundaries
      fBoundaries[2*iIndex+4*CarNodes] = fBoxes[6*iIndex+5]-fBoxes[6*iIndex+2];
      fBoundaries[2*iIndex+4*CarNodes+1] = fBoxes[6*iIndex+5]+fBoxes[6*iIndex+2];
   }
}

//______________________________________________________________________________   
void UVoxelFinder::SortBoundaries()
{
// "SortBoundaries" orders the boundaries along each axis
   int iIndex = 0;
   int CarNodes = fMultiUnion->GetNNodes();
   int *indexSortedBound = new int[2*CarNodes];
   double *tempBoundaries = new double[2*CarNodes];
   
   // x axis:
   int number_boundaries = 0; // Total number of boundaries after treatment:
   UUtils::Sort(2*CarNodes,&fBoundaries[0],&indexSortedBound[0],false);
   
   for(iIndex = 0 ; iIndex < 2*CarNodes ; iIndex++)
   {
      if(number_boundaries == 0)
      {
         tempBoundaries[number_boundaries] = fBoundaries[indexSortedBound[iIndex]];
         number_boundaries++;
         continue;
      }
      
//      if(UUtils::Abs(tempBoundaries[number_boundaries-1]-fBoundaries[indexSortedBound[iIndex]] > 1E-10))
      {
         tempBoundaries[number_boundaries] = fBoundaries[indexSortedBound[iIndex]];
         number_boundaries++;         
      }
   }
   
//   if(fXBoundaries) delete fXBoundaries;
   fXBoundaries = new double[number_boundaries];
   memcpy(fXBoundaries,&tempBoundaries[0],number_boundaries*sizeof(double));
   fXNumBound = number_boundaries;
   
   // y axis:
   number_boundaries = 0; // Total number of boundaries after treatment:
   UUtils::Sort(2*CarNodes,&fBoundaries[2*CarNodes],&indexSortedBound[0],false);
   
   for(iIndex = 0 ; iIndex < 2*CarNodes ; iIndex++)
   {
      if(number_boundaries == 0)
      {
         tempBoundaries[number_boundaries] = fBoundaries[2*CarNodes+indexSortedBound[iIndex]];
         number_boundaries++;
         continue;
      }
      
//      if(UUtils::Abs(tempBoundaries[number_boundaries-1]-fBoundaries[indexSortedBound[iIndex]] > 1E-10))
      {
         tempBoundaries[number_boundaries] = fBoundaries[2*CarNodes+indexSortedBound[iIndex]];
         number_boundaries++;         
      }
   }
   
//   if(fYBoundaries) delete fYBoundaries;
   fYBoundaries = new double[number_boundaries];
   memcpy(fYBoundaries,&tempBoundaries[0],number_boundaries*sizeof(double));
   fYNumBound = number_boundaries;
   
   // z axis:
   number_boundaries = 0; // Total number of boundaries after treatment:
   UUtils::Sort(2*CarNodes,&fBoundaries[4*CarNodes],&indexSortedBound[0],false);
   
   for(iIndex = 0 ; iIndex < 2*CarNodes ; iIndex++)
   {
      if(number_boundaries == 0)
      {
         tempBoundaries[number_boundaries] = fBoundaries[4*CarNodes+indexSortedBound[iIndex]];
         number_boundaries++;
         continue;
      }
      
//      if(UUtils::Abs(tempBoundaries[number_boundaries-1]-fBoundaries[indexSortedBound[iIndex]] > 1E-10))
      {
         tempBoundaries[number_boundaries] = fBoundaries[4*CarNodes+indexSortedBound[iIndex]];
         number_boundaries++;         
      }
   }
   
//   if(fZBoundaries) delete fZBoundaries;
   fZBoundaries = new double[number_boundaries];
   memcpy(fZBoundaries,&tempBoundaries[0],number_boundaries*sizeof(double));
   fZNumBound = number_boundaries;         
}

//______________________________________________________________________________   
void UVoxelFinder::DisplayBoundaries()
{
   int iIndex = 0;
   
   printf(" * X axis:\n    ");
   for(iIndex = 0 ; iIndex < fXNumBound ; iIndex++)
   {
      printf("| %f ",fXBoundaries[iIndex]);
   }
   printf("|\n");   
   
   printf(" * Y axis:\n    ");
   for(iIndex = 0 ; iIndex < fYNumBound ; iIndex++)
   {
      printf("| %f ",fYBoundaries[iIndex]);
   }
   printf("|\n");   
   
   printf(" * Z axis:\n    ");
   for(iIndex = 0 ; iIndex < fZNumBound ; iIndex++)
   {
      printf("| %f ",fZBoundaries[iIndex]);
   }      
   printf("|\n");
}
