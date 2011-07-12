#include "UVoxelFinder.hh"

#include <iostream>
#include <stdio.h>
#include <string.h>
#include "UUtils.hh"
#include "UMultiUnion.hh"

//using namespace std;

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
// the half lengths related to the bounding box of each node.
// These quantities are stored in the array "fBoxes" (6 different values per
// node.
   int iIndex,jIndex,kIndex = 0;
   int CarNodes = fMultiUnion->GetNumNodes(); // Number of nodes in "fMultiUnion"
   UVector3 TempPointConv,TempPoint;                                                  
   if(!CarNodes) return; // If no node, no need to carry on
   int fNBoxes = 6*CarNodes; // 6 different quantities are stored per node
   if(fBoxes) delete fBoxes;
   fBoxes = new double[fNBoxes]; // Array which will store the half lengths
                                 // related to a particular node, but also
                                 // the coordinates of its origin                                 
      
   const VUSolid *TempSolid = NULL;
   const UTransform3D *TempTransform = NULL;
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
// "DisplayVoxelLimits" displays the dX, dY, dZ, oX, oY and oZ for each node
   int CarNodes = fMultiUnion->GetNumNodes();
   int iIndex = 0;
   
   for(iIndex = 0 ; iIndex < CarNodes ; iIndex++)
   {
      printf("    -> Node %d:\n       * dX = %f ; * dY = %f ; * dZ = %f\n       * oX = %f ; * oY = %f ; * oZ = %f\n",iIndex+1,fBoxes[6*iIndex],fBoxes[6*iIndex+1],fBoxes[6*iIndex+2],fBoxes[6*iIndex+3],fBoxes[6*iIndex+4],fBoxes[6*iIndex+5]);
   }
}

//______________________________________________________________________________   
void UVoxelFinder::CreateBoundaries()
{
// "CreateBoundaries"'s aim is to determine the slices induced by the bounding boxes,
// along each axis. The created boundaries are stored in the array "fBoundaries"
   int iIndex = 0;
   int CarNodes = fMultiUnion->GetNumNodes(); // Number of nodes in structure of
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
// "SortBoundaries" orders the boundaries along each axis (increasing order)
// and also does not take into account redundant boundaries, ie if two boundaries
// are separated by a distance strictly inferior to "DIST_INTER_BOUND".
// The sorted boundaries are respectively stored in:
//              * fXBoundaries
//              * fYBoundaries
//              * fZBoundaries
// In addition, the number of elements contained in the three latter arrays are
// precised thanks to variables: fXNumBound, fYNumBound and fZNumBound.
   int iIndex = 0;
   int CarNodes = fMultiUnion->GetNumNodes();
   int *indexSortedBound = new int[2*CarNodes];
   double *tempBoundaries = new double[2*CarNodes];
   
   // x axis:
   // -------
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

      // If two successive boundaries are too close from each other, only the first one is considered      
      if(UUtils::Abs(tempBoundaries[number_boundaries-1]-fBoundaries[indexSortedBound[iIndex]]) > DIST_INTER_BOUND)
      {
         tempBoundaries[number_boundaries] = fBoundaries[indexSortedBound[iIndex]];
         number_boundaries++;         
      }
   }
   
//   if(fXBoundaries) delete [] fXBoundaries;
   fXBoundaries = new double[number_boundaries];
   memcpy(fXBoundaries,&tempBoundaries[0],number_boundaries*sizeof(double));
   fXNumBound = number_boundaries;
   
   // y axis:
   // -------
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
      
      if(UUtils::Abs(tempBoundaries[number_boundaries-1]-fBoundaries[indexSortedBound[iIndex]]) > DIST_INTER_BOUND)
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
   // -------
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
      
      if(UUtils::Abs(tempBoundaries[number_boundaries-1]-fBoundaries[indexSortedBound[iIndex]]) > DIST_INTER_BOUND)
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
// Prints the positions of the boundaries of the slices on the three axis:
   int iIndex = 0;
   
   printf(" * X axis:\n    ");
   for(iIndex = 0 ; iIndex < fXNumBound ; iIndex++)
   {
      printf("| %f ",fXBoundaries[iIndex]);
   }
   printf("|\n    Number of boundaries: %d\n",fXNumBound);  
   
   printf(" * Y axis:\n    ");
   for(iIndex = 0 ; iIndex < fYNumBound ; iIndex++)
   {
      printf("| %f ",fYBoundaries[iIndex]);
   }
   printf("|\n    Number of boundaries: %d\n",fYNumBound);  
      
   printf(" * Z axis:\n    ");
   for(iIndex = 0 ; iIndex < fZNumBound ; iIndex++)
   {
      printf("| %f ",fZBoundaries[iIndex]);
   }      
   printf("|\n    Number of boundaries: %d\n",fZNumBound);  
}

//______________________________________________________________________________   
void UVoxelFinder::BuildListNodes()
{
// "BuildListNodes" stores in the arrays "fIndcX", "fIndcY" and "fIndcZ" the
// solids present in each slice aong the three axis.
// In array "fNsliceX", "fNsliceY" and "fNsliceZ" are stored the number of
// solids in the considered slice.
   int iIndex, jIndex = 0;
   int CarNodes = fMultiUnion->GetNumNodes();   
   int nmaxslices = 2*CarNodes+1;
   int nperslice = 1+(CarNodes-1)/(8*sizeof(char));
   int current = 0;
   
   // Number of slices per axis:
   int xNumslices = fXNumBound-1;
   int yNumslices = fYNumBound-1;
   int zNumslices = fZNumBound-1;

   double xbmin, xbmax, ybmin, ybmax, zbmin, zbmax = 0;
   
   // Memory:
   char *storage = new char[nmaxslices*nperslice];
   memset(storage,0,(nmaxslices*nperslice)*sizeof(double));
   char *bits;
   int number_byte;
   char position;
   
   // Loop on x slices:
   fNsliceX = new int[xNumslices];
   
   for(iIndex = 0 ; iIndex < xNumslices ; iIndex++)
   {
      bits = &storage[current];
   
      // Loop on the nodes:
      for(jIndex = 0 ; jIndex < CarNodes ; jIndex++)
      {
         // Determination of the minimum and maximum position along x
         // of the bounding boxe of each node:
         xbmin = fBoxes[6*jIndex+3]-fBoxes[6*jIndex];
         xbmax = fBoxes[6*jIndex+3]+fBoxes[6*jIndex];
         
         // Is the considered node inside slice?
         if((xbmin - fXBoundaries[iIndex+1]) > -DIST_INTER_BOUND) continue;
         if((xbmax - fXBoundaries[iIndex]) < DIST_INTER_BOUND) continue;
         
         // The considered node is contained in the current slice:
         fNsliceX[iIndex]++;
         
         // Storage of the number of the node in the array:
         number_byte = jIndex/8;
         position = (char)(jIndex%8);
         bits[number_byte] |= (1 << position);         
      }
      if(fNsliceX[iIndex]>0) current += nperslice;
   }
   fNx = current;
   fIndcX = new char[fNx];
   memcpy(fIndcX,storage,current*sizeof(char));
   
   // Loop on y slices:
   fNsliceY = new int[yNumslices];
   current = 0;
   memset(storage,0,(nmaxslices*nperslice)*sizeof(double));
      
   for(iIndex = 0 ; iIndex < xNumslices ; iIndex++)
   {
      bits = &storage[current];
   
      // Loop on the nodes:
      for(jIndex = 0 ; jIndex < CarNodes ; jIndex++)
      {
         // Determination of the minimum and maximum position along x
         // of the bounding boxe of each node:
         ybmin = fBoxes[6*jIndex+4]-fBoxes[6*jIndex+1];
         ybmax = fBoxes[6*jIndex+4]+fBoxes[6*jIndex+1];
         
         // Is the considered node inside slice?
         if((ybmin - fYBoundaries[iIndex+1]) > -DIST_INTER_BOUND) continue;
         if((ybmax - fYBoundaries[iIndex]) < DIST_INTER_BOUND) continue;
         
         // The considered node is contained in the current slice:
         fNsliceY[iIndex]++;
         
         // Storage of the number of the node in the array:
         number_byte = jIndex/8;
         position = (char)(jIndex%8);
         bits[number_byte] |= (1 << position);         
      }
      if(fNsliceY[iIndex]>0) current += nperslice;
   }
   fNy = current;
   fIndcY = new char[fNy];
   memcpy(fIndcY,storage,current*sizeof(char));
   
   // Loop on z slices:
   fNsliceZ = new int[zNumslices];
   current = 0;
   memset(storage,0,(nmaxslices*nperslice)*sizeof(double));
   
   for(iIndex = 0 ; iIndex < zNumslices ; iIndex++)
   {
      bits = &storage[current];
   
      // Loop on the nodes:
      for(jIndex = 0 ; jIndex < CarNodes ; jIndex++)
      {
         // Determination of the minimum and maximum position along x
         // of the bounding boxe of each node:
         zbmin = fBoxes[6*jIndex+5]-fBoxes[6*jIndex+2];
         zbmax = fBoxes[6*jIndex+5]+fBoxes[6*jIndex+2];
         
         // Is the considered node inside slice?
         if((zbmin - fZBoundaries[iIndex+1]) > -DIST_INTER_BOUND) continue;
         if((zbmax - fZBoundaries[iIndex]) < DIST_INTER_BOUND) continue;
         
         // The considered node is contained in the current slice:
         fNsliceZ[iIndex]++;
         
         // Storage of the number of the node in the array:
         number_byte = jIndex/8;
         position = (char)(jIndex%8);
         bits[number_byte] |= (1 << position);         
      }
      if(fNsliceZ[iIndex]>0) current += nperslice;
   }
   fNz = current;
   fIndcZ = new char[fNz];
   memcpy(fIndcZ,storage,current*sizeof(char));      
}

//______________________________________________________________________________   
void UVoxelFinder::DisplayListNodes()
{
// Prints which solids are present in the slices previously elaborated.
   int iIndex, jIndex = 0;
   int CarNodes = fMultiUnion->GetNumNodes();   
//   int nmaxslices = 2*CarNodes+1;
   int nperslice = 1+(CarNodes-1)/(8*sizeof(char));
   
   printf(" * X axis:\n");
   for(iIndex = 0 ; iIndex < fXNumBound-1 ; iIndex++)
   {
      printf("    Slice #%d: ",iIndex+1);
   
      for(jIndex = 0 ; jIndex < nperslice ; jIndex++)
      {
         printf("%d ",(int)fIndcX[iIndex+jIndex]);
      }
      printf("\n");
   }
   
   printf(" * Y axis:\n");
   for(iIndex = 0 ; iIndex < fYNumBound-1 ; iIndex++)
   {
      printf("    Slice #%d: ",iIndex+1);
   
      for(jIndex = 0 ; jIndex < nperslice ; jIndex++)
      {
         printf("%d ",(int)fIndcY[iIndex+jIndex]);
      }
      printf("\n");
   }
   
   printf(" * Z axis:\n");
   for(iIndex = 0 ; iIndex < fZNumBound-1 ; iIndex++)
   {
      printf("    Slice #%d: ",iIndex+1);
   
      for(jIndex = 0 ; jIndex < nperslice ; jIndex++)
      {
         printf("%d ",(int)fIndcZ[iIndex+jIndex]);
      }
      printf("\n");
   }       
}
