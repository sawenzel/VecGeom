#include "UVoxelFinder.hh"

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <string.h>
#include "UUtils.hh"
#include "UMultiUnion.hh"

using namespace std;

//______________________________________________________________________________   
UVoxelFinder::UVoxelFinder() : fMultiUnion(0),
                               fBoxes(0),
                               fBoundaries(0),
                               fXSortedBoundaries(0),
                               fXNumBound(0),
                               fYSortedBoundaries(0),
                               fYNumBound(0),
                               fZSortedBoundaries(0),
                               fZNumBound(0),
                               fNumNodesSliceX(0),
                               fNumNodesSliceY(0),
                               fNumNodesSliceZ(0),
                               fMemoryX(0),
                               fMemoryY(0),
                               fMemoryZ(0),      
                               fNx(0),
                               fNy(0),
                               fNz(0)
                               {}

//______________________________________________________________________________   
UVoxelFinder::UVoxelFinder(UMultiUnion* multiUnion) : fMultiUnion(multiUnion),
                                                       fBoxes(0),
                                                       fBoundaries(0),
                                                       fXSortedBoundaries(0),
                                                       fXNumBound(0),
                                                       fYSortedBoundaries(0),
                                                       fYNumBound(0),
                                                       fZSortedBoundaries(0),
                                                       fZNumBound(0),
                                                       fNumNodesSliceX(0),
                                                       fNumNodesSliceY(0),
                                                       fNumNodesSliceZ(0),
                                                       fMemoryX(0),
                                                       fMemoryY(0),
                                                       fMemoryZ(0),      
                                                       fNx(0),
                                                       fNy(0),
                                                       fNz(0)
                                                       {}

//______________________________________________________________________________   
UVoxelFinder::~UVoxelFinder()
{
   if(fMultiUnion) delete fMultiUnion;
   if(fBoxes) delete fBoxes;
   if(fBoundaries) delete fBoundaries;
   if(fXSortedBoundaries) delete fXSortedBoundaries;
   if(fYSortedBoundaries) delete fYSortedBoundaries;
   if(fZSortedBoundaries) delete fZSortedBoundaries;
   if(fNumNodesSliceX) delete fNumNodesSliceX;
   if(fNumNodesSliceY) delete fNumNodesSliceY;
   if(fNumNodesSliceZ) delete fNumNodesSliceZ;   
   if(fMemoryX) delete fMemoryX;
   if(fMemoryY) delete fMemoryY;
   if(fMemoryZ) delete fMemoryZ;   
}

//______________________________________________________________________________   
void UVoxelFinder::BuildVoxelLimits()
{
// "BuildVoxelLimits"'s aim is to store the coordinates of the origin as well as
// the half lengths related to the bounding box of each node.
// These quantities are stored in the array "fBoxes" (6 different values per
// node.
   int iIndex = 0;
   int carNodes = fMultiUnion->GetNumNodes(); // Number of nodes in "fMultiUnion"
   UVector3 tempPointConv,tempPoint;                                                  
   if(!carNodes) return; // If no node, no need to carry on
   int fNBoxes = 6*carNodes; // 6 different quantities are stored per node
   if(fBoxes) delete fBoxes;
   fBoxes = new double[fNBoxes]; // Array which will store the half lengths
                                 // related to a particular node, but also
                                 // the coordinates of its origin                                 
      
   /*const*/ VUSolid *tempSolid;
   const UTransform3D *tempTransform;
   
   double* arrMin = new double[3];
   double* arrMax = new double[3];
   UVector3 min, max;
   
   for(iIndex = 0 ; iIndex < carNodes ; iIndex++)   
   {
      tempSolid = fMultiUnion->GetSolid(iIndex);
      tempTransform = fMultiUnion->GetTransform(iIndex);

      tempSolid->Extent(arrMin, arrMax);
      min.Set(arrMin[0],arrMin[1],arrMin[2]);      
      max.Set(arrMax[0],arrMax[1],arrMax[2]);           
      UUtils::TransformLimits(min, max, tempTransform);                        
      
      // Storage of the positions:
      fBoxes[6*iIndex]   = 0.5*(max.x-min.x); // dX
      fBoxes[6*iIndex+1] = 0.5*(max.y-min.y); // dY
      fBoxes[6*iIndex+2] = 0.5*(max.z-min.z); // dZ
      fBoxes[6*iIndex+3] = tempTransform->fTr[0]; // Ox
      fBoxes[6*iIndex+4] = tempTransform->fTr[1]; // Oy
      fBoxes[6*iIndex+5] = tempTransform->fTr[2]; // Oz
   }   
}

//______________________________________________________________________________   
void UVoxelFinder::DisplayVoxelLimits()
{
// "DisplayVoxelLimits" displays the dX, dY, dZ, oX, oY and oZ for each node
   int carNodes = fMultiUnion->GetNumNodes();
   int iIndex = 0;
   
   for(iIndex = 0 ; iIndex < carNodes ; iIndex++)
   {
//      cout << "    -> Node " << iIndex+1 << ":\n       * dX = " << fBoxes[6*iIndex] << " ; * dY = " << fBoxes[6*iIndex+1] << " ; * dZ = " << fBoxes[6*iIndex+2] << "\n       * oX = " << fBoxes[6*iIndex+3] << " ; * oY = " << fBoxes[6*iIndex+4] << " ; * oZ = " << fBoxes[6*iIndex+5] << "\n";
      cout << fixed;
      cout << "    -> Node " << iIndex+1 << ":\n       * dX = " << setprecision (7) << fBoxes[6*iIndex] << " ; * dY = " << setprecision (7) << fBoxes[6*iIndex+1] << " ; * dZ = " << setprecision (7) << fBoxes[6*iIndex+2] << "\n       * oX = " << setprecision (7) << fBoxes[6*iIndex+3] << " ; * oY = " << setprecision (7) << fBoxes[6*iIndex+4] << " ; * oZ = " << setprecision (7) << fBoxes[6*iIndex+5] << "\n";      
   }
}

//______________________________________________________________________________   
void UVoxelFinder::CreateBoundaries()
{
// "CreateBoundaries"'s aim is to determine the slices induced by the bounding boxes,
// along each axis. The created boundaries are stored in the array "fBoundaries"
   int iIndex = 0;
   int carNodes = fMultiUnion->GetNumNodes(); // Number of nodes in structure of
                                            // "UMultiUnion" type
   
   // Determination of the 3D extent of the "UMultIUnion" structure:
   double minima_multi[3], maxima_multi[3];
   fMultiUnion->Extent(minima_multi,maxima_multi);
   
   // Determination of the boundries along x, y and z axis:
   fBoundaries = new double[6*carNodes];
   
   for(iIndex = 0 ; iIndex < carNodes ; iIndex++)   
   {
      // For each node, the boundaries are created by using the array "fBoxes"
      // built in method "BuildVoxelLimits":

      // x boundaries
      fBoundaries[2*iIndex] = fBoxes[6*iIndex+3]-fBoxes[6*iIndex];
      fBoundaries[2*iIndex+1] = fBoxes[6*iIndex+3]+fBoxes[6*iIndex];
      // y boundaries
      fBoundaries[2*iIndex+2*carNodes] = fBoxes[6*iIndex+4]-fBoxes[6*iIndex+1];
      fBoundaries[2*iIndex+2*carNodes+1] = fBoxes[6*iIndex+4]+fBoxes[6*iIndex+1];
      // z boundaries
      fBoundaries[2*iIndex+4*carNodes] = fBoxes[6*iIndex+5]-fBoxes[6*iIndex+2];
      fBoundaries[2*iIndex+4*carNodes+1] = fBoxes[6*iIndex+5]+fBoxes[6*iIndex+2];
   }
}

//______________________________________________________________________________   
void UVoxelFinder::SortBoundaries()
{
// "SortBoundaries" orders the boundaries along each axis (increasing order)
// and also does not take into account redundant boundaries, ie if two boundaries
// are separated by a distance strictly inferior to "tolerance".
// The sorted boundaries are respectively stored in:
//              * fXSortedBoundaries
//              * fYSortedBoundaries
//              * fZSortedBoundaries
// In addition, the number of elements contained in the three latter arrays are
// precised thanks to variables: fXNumBound, fYNumBound and fZNumBound.
   static const double tolerance = 10E-10; // Minimal distance to discrminate two boundaries.
   int iIndex = 0;
   int carNodes = fMultiUnion->GetNumNodes();
   int *indexSortedBound = new int[2*carNodes];
   double *tempBoundaries = new double[2*carNodes];
   
   // x axis:
   // -------
   int number_boundaries = 0; // Total number of boundaries after treatment:
   UUtils::Sort(2*carNodes,&fBoundaries[0],&indexSortedBound[0],false);
   
   for(iIndex = 0 ; iIndex < 2*carNodes ; iIndex++)
   {
      if(number_boundaries == 0)
      {
         tempBoundaries[number_boundaries] = fBoundaries[indexSortedBound[iIndex]];
         number_boundaries++;
         continue;
      }

      // If two successive boundaries are too close from each other, only the first one is considered      
      if(UUtils::Abs(tempBoundaries[number_boundaries-1]-fBoundaries[indexSortedBound[iIndex]]) > tolerance)
      {
         tempBoundaries[number_boundaries] = fBoundaries[indexSortedBound[iIndex]];
         number_boundaries++;         
      }
   }
   
   if(fXSortedBoundaries) delete [] fXSortedBoundaries;
   fXSortedBoundaries = new double[number_boundaries];
   memcpy(fXSortedBoundaries,&tempBoundaries[0],number_boundaries*sizeof(double));
   fXNumBound = number_boundaries;
   
   // y axis:
   // -------
   number_boundaries = 0; // Total number of boundaries after treatment:
   UUtils::Sort(2*carNodes,&fBoundaries[2*carNodes],&indexSortedBound[0],false);
   
   for(iIndex = 0 ; iIndex < 2*carNodes ; iIndex++)
   {
      if(number_boundaries == 0)
      {
         tempBoundaries[number_boundaries] = fBoundaries[2*carNodes+indexSortedBound[iIndex]];
         number_boundaries++;
         continue;
      }
      
      if(UUtils::Abs(tempBoundaries[number_boundaries-1]-fBoundaries[2*carNodes+indexSortedBound[iIndex]]) > tolerance)
      {
         tempBoundaries[number_boundaries] = fBoundaries[2*carNodes+indexSortedBound[iIndex]];
         number_boundaries++;         
      }
   }
   
   if(fYSortedBoundaries) delete fYSortedBoundaries;
   fYSortedBoundaries = new double[number_boundaries];
   memcpy(fYSortedBoundaries,&tempBoundaries[0],number_boundaries*sizeof(double));
   fYNumBound = number_boundaries;
   
   // z axis:
   // -------
   number_boundaries = 0; // Total number of boundaries after treatment:
   UUtils::Sort(2*carNodes,&fBoundaries[4*carNodes],&indexSortedBound[0],false);
   
   for(iIndex = 0 ; iIndex < 2*carNodes ; iIndex++)
   {
      if(number_boundaries == 0)
      {
         tempBoundaries[number_boundaries] = fBoundaries[4*carNodes+indexSortedBound[iIndex]];
         number_boundaries++;
         continue;
      }
      
      if(UUtils::Abs(tempBoundaries[number_boundaries-1]-fBoundaries[4*carNodes+indexSortedBound[iIndex]]) > tolerance)
      {
         tempBoundaries[number_boundaries] = fBoundaries[4*carNodes+indexSortedBound[iIndex]];
         number_boundaries++;         
      }
   }
   
   if(fZSortedBoundaries) delete fZSortedBoundaries;
   fZSortedBoundaries = new double[number_boundaries];
   memcpy(fZSortedBoundaries,&tempBoundaries[0],number_boundaries*sizeof(double));
   fZNumBound = number_boundaries;         
}

//______________________________________________________________________________   
void UVoxelFinder::DisplayBoundaries()
{
// Prints the positions of the boundaries of the slices on the three axis:
   int iIndex = 0;
   
   cout << " * X axis:" << endl << "    | ";
   for(iIndex = 0 ; iIndex < fXNumBound ; iIndex++)
   {
      cout << fXSortedBoundaries[iIndex] << " ";
      if(iIndex != fXNumBound-1) cout << "-> ";
   }
   cout << "|" << endl << "Number of boundaries: " << fXNumBound << endl;
   
   cout << " * Y axis:" << endl << "    | ";
   for(iIndex = 0 ; iIndex < fYNumBound ; iIndex++)
   {
      cout << fYSortedBoundaries[iIndex] << " ";
      if(iIndex != fYNumBound-1) cout << "-> ";
   }
   cout << "|" << endl << "Number of boundaries: " << fYNumBound << endl;
      
   cout << " * Z axis:" << endl << "    | ";
   for(iIndex = 0 ; iIndex < fZNumBound ; iIndex++)
   {
      cout << fZSortedBoundaries[iIndex] << " ";
      if(iIndex != fZNumBound-1) printf("-> ");
   }      
   cout << "|" << endl << "Number of boundaries: " << fZNumBound << endl;
}

//______________________________________________________________________________   
void UVoxelFinder::BuildListNodes()
{
// "BuildListNodes" stores in the arrays "fMemoryX", "fMemoryY" and "fMemoryZ" the
// solids present in each slice aong the three axis.
// In array "fNumNodesSliceX", "fNumNodesSliceY" and "fNumNodesSliceZ" are stored the number of
// solids in the considered slice.
   const double localTolerance = 10E-10;
   int iIndex = 0, jIndex = 0;
   int carNodes = fMultiUnion->GetNumNodes();   
   int nmaxslices = 2*carNodes+1;
   int nperslice = 1+(carNodes-1)/(8*sizeof(char));
   int current = 0;
   
   // Number of slices per axis:
   int xNumslices = fXNumBound-1;
   int yNumslices = fYNumBound-1;
   int zNumslices = fZNumBound-1;

   double xbmin, xbmax, ybmin, ybmax, zbmin, zbmax = 0;
   
   // Memory:
   char *storage = new char[nmaxslices*nperslice];
   memset(storage,0,(nmaxslices*nperslice)*sizeof(char));
   char *bits;
   int number_byte;
   int position;
   
   // Loop on x slices:
   fNumNodesSliceX = new int[xNumslices];
   
   for(iIndex = 0 ; iIndex < xNumslices ; iIndex++)
   {
      bits = &storage[current];
   
      // Loop on the nodes:
      for(jIndex = 0 ; jIndex < carNodes ; jIndex++)
      {
         // Determination of the minimum and maximum position along x
         // of the bounding boxe of each node:
         xbmin = fBoxes[6*jIndex+3]-fBoxes[6*jIndex];
         xbmax = fBoxes[6*jIndex+3]+fBoxes[6*jIndex];
         
         // Is the considered node inside slice?
         if((xbmin - fXSortedBoundaries[iIndex+1]) > -localTolerance) continue;
         if((xbmax - fXSortedBoundaries[iIndex]) < localTolerance) continue;
         
         // The considered node is contained in the current slice:
         fNumNodesSliceX[iIndex]++;
         
         // Storage of the number of the node in the array:
         number_byte = jIndex/8;
         position = jIndex%8;
         bits[number_byte] |= (1 << position);                  
      }
      current += nperslice;
   }
   fNx = current;
   fMemoryX = new char[fNx];
   memcpy(fMemoryX,storage,current*sizeof(char));
   
   // Loop on y slices:
   fNumNodesSliceY = new int[yNumslices];
   current = 0;
   memset(storage,0,(nmaxslices*nperslice)*sizeof(char));
   
   for(iIndex = 0 ; iIndex < yNumslices ; iIndex++)
   {
      bits = &storage[current];
   
      // Loop on the nodes:
      for(jIndex = 0 ; jIndex < carNodes ; jIndex++)
      {
         // Determination of the minimum and maximum position along x
         // of the bounding boxe of each node:
         ybmin = fBoxes[6*jIndex+4]-fBoxes[6*jIndex+1];
         ybmax = fBoxes[6*jIndex+4]+fBoxes[6*jIndex+1];
         
         // Is the considered node inside slice?
         if((ybmin - fYSortedBoundaries[iIndex+1]) > -localTolerance) continue;
         if((ybmax - fYSortedBoundaries[iIndex]) < localTolerance) continue;
         
         // The considered node is contained in the current slice:
         fNumNodesSliceY[iIndex]++;
         
         // Storage of the number of the node in the array:
         number_byte = jIndex/8;
         position = jIndex%8;
         bits[number_byte] |= (1 << position);                  
      }
      current += nperslice;
   }
   fNy = current;
   fMemoryY = new char[fNy];
   memcpy(fMemoryY,storage,current*sizeof(char));    
   
   // Loop on z slices:
   fNumNodesSliceZ = new int[zNumslices];
   current = 0;
   memset(storage,0,(nmaxslices*nperslice)*sizeof(char));
   
   for(iIndex = 0 ; iIndex < zNumslices ; iIndex++)
   {
      bits = &storage[current];
   
      // Loop on the nodes:
      for(jIndex = 0 ; jIndex < carNodes ; jIndex++)
      {
         // Determination of the minimum and maximum position along x
         // of the bounding boxe of each node:
         zbmin = fBoxes[6*jIndex+5]-fBoxes[6*jIndex+2];
         zbmax = fBoxes[6*jIndex+5]+fBoxes[6*jIndex+2];
         
         // Is the considered node inside slice?
         if((zbmin - fZSortedBoundaries[iIndex+1]) > -localTolerance) continue;
         if((zbmax - fZSortedBoundaries[iIndex]) < localTolerance) continue;
         
         // The considered node is contained in the current slice:
         fNumNodesSliceZ[iIndex]++;
         
         // Storage of the number of the node in the array:
         number_byte = jIndex/8;
         position = jIndex%8;
         bits[number_byte] |= (1 << position);                  
      }
      current += nperslice;
   }
   fNz = current;
   fMemoryZ = new char[fNz];
   memcpy(fMemoryZ,storage,current*sizeof(char));      
}

//______________________________________________________________________________   
void UVoxelFinder::GetCandidatesAsString(const char* mask, std::string &result)
{
   // Decodes the candidates in mask as string.
   static char buffer[30];
   int carNodes = fMultiUnion->GetNumNodes();   
   char mask1 = 1;
   result = "";
   
   for(int icand=0; icand<carNodes; icand++)
   {
      int byte = icand/8;
      int bit = icand - 8*byte;
   
      if (mask1<<bit & mask[byte])
      {
         sprintf(buffer, "%d ", icand+1); 
         result += buffer;
      }   
   }
}

//______________________________________________________________________________   
vector<int> UVoxelFinder::GetCandidatesAsVector(const char* mask)
{
   int iIndex;
   int carNodes = fMultiUnion->GetNumNodes();   
   char maskMove = 1;
   vector<int> candidatesVoxel;
   
   for(iIndex = 0 ; iIndex < carNodes ; iIndex++)
   {
      int byte = iIndex/8;
      int bit = iIndex - 8*byte;
   
      if(maskMove<<bit & mask[byte])
      {
         candidatesVoxel.push_back(iIndex);
      }   
   }
   return candidatesVoxel;
}

//______________________________________________________________________________   
vector<int> UVoxelFinder::GetCandidatesAsVector2(const char* mask)
{
   int iIndex, jIndex;
   int carNodes = fMultiUnion->GetNumNodes();
   int nperslices = 1+(carNodes-1)/(8*sizeof(char));   
   char maskFixed = 1;
   vector<int> candidatesVoxel;
   char tempMask, temp;
   
   for(iIndex = 0 ; iIndex < nperslices ; iIndex++)
   {
      tempMask = mask[iIndex];
   
      for(jIndex = 0 ; jIndex < 8 ; jIndex++)
      {
         if((jIndex + 8*iIndex) >= carNodes)
         {
            return candidatesVoxel;
         }
      
         if(tempMask & maskFixed)
         {
            candidatesVoxel.push_back(jIndex + 8*iIndex);            
         }
         temp = tempMask >> 1;
         tempMask = temp;
      }
   }
   return candidatesVoxel;
}

//______________________________________________________________________________   
vector<int> UVoxelFinder::GetCandidatesAsVector3(const char* mask)
{
   int iIndex;
   int carNodes = fMultiUnion->GetNumNodes();
   int nperslices = 1+(carNodes-1)/(8*sizeof(char));   
   vector<int> candidatesVoxel;
   
   for(iIndex = 0 ; iIndex < nperslices ; iIndex++)
   {
      switch(mask[iIndex])
      {
         case(0):
            break;
         case(1):
            candidatesVoxel.push_back(0 + 8*iIndex);
            break;
         case(2):
            candidatesVoxel.push_back(1 + 8*iIndex);
            break;
         case(3):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            break;
         case(4):
            candidatesVoxel.push_back(2 + 8*iIndex);
            break;
         case(5):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            break;
         case(6):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            break;
         case(7):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            break;
         case(8):
            candidatesVoxel.push_back(3 + 8*iIndex);
            break;
         case(9):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            break;
         case(10):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            break;
         case(11):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            break;
         case(12):
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            break;
         case(13):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            break;
         case(14):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            break;
         case(15):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            break;
         case(16):
            candidatesVoxel.push_back(4 + 8*iIndex);
            break;
         case(17):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            break;
         case(18):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            break;
         case(19):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            break;
         case(20):
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            break;
         case(21):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            break;
         case(22):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            break;
         case(23):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            break;
         case(24):
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            break;
         case(25):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            break;
         case(26):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            break;
         case(27):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            break;
         case(28):
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            break;
         case(29):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            break;
         case(30):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            break;
         case(31):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            break;
         case(32):
            candidatesVoxel.push_back(5 + 8*iIndex);
            break;
         case(33):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            break;
         case(34):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            break;
         case(35):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            break;
         case(36):
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            break;
         case(37):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            break;
         case(38):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            break;
         case(39):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            break;
         case(40):
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            break;
         case(41):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            break;
         case(42):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            break;
         case(43):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            break;
         case(44):
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            break;
         case(45):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            break;
         case(46):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            break;
         case(47):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            break;
         case(48):
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            break;
         case(49):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            break;
         case(50):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            break;
         case(51):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            break;
         case(52):
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            break;
         case(53):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            break;
         case(54):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            break;
         case(55):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            break;
         case(56):
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            break;
         case(57):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            break;
         case(58):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            break;
         case(59):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            break;
         case(60):
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            break;
         case(61):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            break;
         case(62):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            break;
         case(63):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            break;
         case(64):
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(65):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(66):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(67):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(68):
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(69):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(70):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(71):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(72):
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(73):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(74):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(75):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(76):
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(77):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(78):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(79):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(80):
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(81):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(82):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(83):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(84):
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(85):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(86):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(87):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(88):
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(89):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(90):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(91):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(92):
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(93):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(94):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(95):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(96):
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(97):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(98):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(99):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(100):
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(101):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(102):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(103):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(104):
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(105):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(106):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(107):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(108):
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(109):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(110):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(111):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(112):
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(113):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(114):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(115):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(116):
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(117):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(118):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(119):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(120):
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(121):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(122):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(123):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(124):
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(125):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(126):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(127):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            break;
         case(128):
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(129):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(130):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(131):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(132):
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(133):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(134):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(135):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(136):
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(137):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(138):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(139):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(140):
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(141):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(142):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(143):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(144):
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(145):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(146):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(147):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(148):
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(149):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(150):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(151):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(152):
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(153):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(154):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(155):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(156):
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(157):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(158):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(159):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(160):
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(161):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(162):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(163):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(164):
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(165):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(166):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(167):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(168):
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(169):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(170):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(171):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(172):
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(173):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(174):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(175):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(176):
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(177):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(178):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(179):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(180):
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(181):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(182):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(183):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(184):
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(185):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(186):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(187):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(188):
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(189):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(190):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(191):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(192):
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(193):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(194):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(195):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(196):
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(197):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(198):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(199):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(200):
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(201):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(202):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(203):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(204):
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(205):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(206):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(207):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(208):
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(209):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(210):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(211):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(212):
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(213):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(214):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(215):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(216):
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(217):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(218):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(219):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(220):
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(221):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(222):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(223):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(224):
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(225):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(226):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(227):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(228):
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(229):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(230):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(231):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(232):
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(233):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(234):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(235):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(236):
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(237):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(238):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(239):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(240):
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(241):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(242):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(243):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(244):
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(245):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(246):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(247):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(248):
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(249):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(250):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(251):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(252):
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(253):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(254):
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;
         case(255):
            candidatesVoxel.push_back(0 + 8*iIndex);
            candidatesVoxel.push_back(1 + 8*iIndex);
            candidatesVoxel.push_back(2 + 8*iIndex);
            candidatesVoxel.push_back(3 + 8*iIndex);
            candidatesVoxel.push_back(4 + 8*iIndex);
            candidatesVoxel.push_back(5 + 8*iIndex);
            candidatesVoxel.push_back(6 + 8*iIndex);
            candidatesVoxel.push_back(7 + 8*iIndex);
            break;                                                                                   
      }
   }
   return candidatesVoxel;
}

//______________________________________________________________________________   
void UVoxelFinder::DisplayListNodes()
{
// Prints which solids are present in the slices previously elaborated.
   int iIndex = 0, jIndex = 0;
   int carNodes = fMultiUnion->GetNumNodes();   
   int nperslice = 1+(carNodes-1)/(8*sizeof(char));
   string result = "";
   
   cout << " * X axis:" << endl;
   for(iIndex = 0 , jIndex = 0 ; iIndex < fXNumBound-1 ; iIndex++ , jIndex += nperslice)
   {
      cout << "    Slice #" << iIndex+1 << ": [" << fXSortedBoundaries[iIndex] << " ; " << fXSortedBoundaries[iIndex+1] << "] -> ";
      
      GetCandidatesAsString(&fMemoryX[jIndex], result);
      cout << "[ " << result.c_str() << "]  " << endl;
   }
   
   cout << " * Y axis:" << endl;
   for(iIndex = 0 , jIndex = 0 ; iIndex < fYNumBound-1 ; iIndex++ , jIndex += nperslice)
   {
      cout << "    Slice #" << iIndex+1 << ": [" << fYSortedBoundaries[iIndex] << " ; " << fYSortedBoundaries[iIndex+1] << "] -> ";
      
      GetCandidatesAsString(&fMemoryY[jIndex], result);
      cout << "[ " << result.c_str() << "]  " << endl;
   }
   
   cout << " * Z axis:" << endl;
   for(iIndex = 0 , jIndex = 0 ; iIndex < fZNumBound-1 ; iIndex++ , jIndex += nperslice)
   {
      cout << "    Slice #" << iIndex+1 << ": [" << fZSortedBoundaries[iIndex] << " ; " << fZSortedBoundaries[iIndex+1] << "] -> ";
      
      GetCandidatesAsString(&fMemoryZ[jIndex], result);
      cout << "[ " << result.c_str() << "]  " << endl;
   }    
}

//______________________________________________________________________________   
void UVoxelFinder::Voxelize()
{
   BuildVoxelLimits();
   CreateBoundaries();    
   SortBoundaries(); 
   BuildListNodes();  
}

//______________________________________________________________________________       
void UVoxelFinder::GetCandidatesVoxel(int indexX, int indexY, int indexZ)
{
   // "GetCandidates" should compute which solids are possibly contained in
   // the voxel defined by the three slices characterized by the passed indexes.
   int iIndex;
   string result = "";  
   int carNodes = fMultiUnion->GetNumNodes();
   int nperslices = 1+(carNodes-1)/(8*sizeof(char));
   
   // Voxelized structure:      
   char *maskResult = new char[nperslices];
   
   for(iIndex = 0 ; iIndex < nperslices ; iIndex++)
   {
      maskResult[iIndex] = fMemoryX[nperslices*(indexX-1)+iIndex]
                         & fMemoryY[nperslices*(indexY-1)+iIndex]
                         & fMemoryZ[nperslices*(indexZ-1)+iIndex];
   } 
   GetCandidatesAsString(&(maskResult[0]),result);
   cout << "   Candidates in voxel [" << indexX << " ; " << indexY << " ; " << indexZ << "]: ";
   cout << "[ " << result.c_str() << "]  " << endl;   
}

//______________________________________________________________________________       
vector<int> UVoxelFinder::GetCandidatesVoxelArray(UVector3 point)
{
// Method returning the candidates corresponding to the passed point
   int carNodes = fMultiUnion->GetNumNodes();
   int nperslices = 1+(carNodes-1)/(8*sizeof(char));   
   vector<int> voidList, checkList;

   if(carNodes == 1)
   {
      if(fXSortedBoundaries)
      {
         if(point.x < fXSortedBoundaries[0] || point.x > fXSortedBoundaries[1]) return voidList;
      }
      if(fYSortedBoundaries)
      {
         if(point.y < fYSortedBoundaries[0] || point.y > fYSortedBoundaries[1]) return voidList;
      }
      if(fZSortedBoundaries)
      {
         if(point.z < fZSortedBoundaries[0] || point.z > fZSortedBoundaries[1]) return voidList;
      }
      checkList.push_back(0);
      return checkList;
   }
   else
   {
      int resultBinSearchX, resultBinSearchY, resultBinSearchZ;
      int numberNodes[3] = {0,0,0};
      char* maskX;
      char* maskY;
      char* maskZ;            
      
      // Along x axis:
      resultBinSearchX = UUtils::BinarySearch(fXNumBound, fXSortedBoundaries, point.x);
      
      if((resultBinSearchX == -1) || (resultBinSearchX == fXNumBound-1)) return voidList;
      
      numberNodes[0] = fNumNodesSliceX[resultBinSearchX];
      if(!numberNodes[0]) return voidList;
      
      maskX = &fMemoryX[nperslices*resultBinSearchX];
      
      // Along y axis:
      resultBinSearchY = UUtils::BinarySearch(fYNumBound, fYSortedBoundaries, point.y);    
      
      if((resultBinSearchY == -1) || (resultBinSearchY == fYNumBound-1)) return voidList;
      
      numberNodes[1] = fNumNodesSliceY[resultBinSearchY];
      if(!numberNodes[1]) return voidList;
      
      maskY = &fMemoryY[nperslices*resultBinSearchY];
      
      // Along z axis:
      resultBinSearchZ = UUtils::BinarySearch(fZNumBound, fZSortedBoundaries, point.z);
      
      if((resultBinSearchZ == -1) || (resultBinSearchZ == fZNumBound-1)) return voidList;
      
      numberNodes[2] = fNumNodesSliceZ[resultBinSearchZ];
      if(!numberNodes[2]) return voidList;
      
      maskZ = &fMemoryZ[nperslices*resultBinSearchZ];  
      
      // Using Intersect method:   
      return Intersect(numberNodes[0],maskX,numberNodes[1],maskY,numberNodes[2],maskZ);      
   }
}

//______________________________________________________________________________       
vector<int> UVoxelFinder::Intersect(int numNodesX, char* maskX, int numNodesY, char* maskY, int numNodesZ, char* maskZ)
{
// Returns the list of nodes corresponding to the intersection of three arrays of bits
   int carNodes = fMultiUnion->GetNumNodes();
   int nperslices = 1+(carNodes-1)/(8*sizeof(char)); 

   int currentByte;
   int currentBit;
   
   char byte;
   vector<int> checkList;
   
   for(currentByte = 0 ; currentByte < nperslices ; currentByte++)
   {
      byte = maskX[currentByte] & maskY[currentByte] & maskZ[currentByte];
   
      if (!byte) continue;
   
      for (currentBit = 0 ; currentBit < 8 ; currentBit++)
      {
         if (byte & (1<<currentBit))
         {
            checkList.push_back((currentByte<<3) + currentBit);
         
            if (((int)checkList.size() == numNodesX) || ((int)checkList.size() == numNodesY) || ((int)checkList.size() == numNodesZ)) break;   
         }
      }
   }
   return checkList;
}
