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

	int carNodes = fMultiUnion->GetNumNodes(); // Number of nodes in "fMultiUnion"
	if(!carNodes) return; // If no node, no need to carry on
	UVector3 tempPointConv,tempPoint;                                                  
   int fNBoxes = 6*carNodes; // 6 different quantities are stored per node
   if(fBoxes) delete fBoxes;
   fBoxes = new double[fNBoxes]; // Array which will store the half lengths
                                 // related to a particular node, but also
                                 // the coordinates of its origin                                   
   /*const*/ VUSolid *tempSolid;
   const UTransform3D *tempTransform;
   
   double arrMin[3], arrMax[3];
   UVector3 min, max;
   
   for(int i = 0 ; i < carNodes ; i++)   
   {
      tempSolid = fMultiUnion->GetSolid(i);
      tempTransform = fMultiUnion->GetTransform(i);

      tempSolid->Extent(arrMin, arrMax);
      min.Set(arrMin[0],arrMin[1],arrMin[2]);      
      max.Set(arrMax[0],arrMax[1],arrMax[2]);           
      UUtils::TransformLimits(min, max, tempTransform);                        
      
      // Storage of the positions:
      fBoxes[6*i]   = 0.5*(max.x-min.x); // dX
      fBoxes[6*i+1] = 0.5*(max.y-min.y); // dY
      fBoxes[6*i+2] = 0.5*(max.z-min.z); // dZ
      fBoxes[6*i+3] = tempTransform->fTr[0]; // Ox
      fBoxes[6*i+4] = tempTransform->fTr[1]; // Oy
      fBoxes[6*i+5] = tempTransform->fTr[2]; // Oz
   }
}

//______________________________________________________________________________   
void UVoxelFinder::DisplayVoxelLimits()
{
// "DisplayVoxelLimits" displays the dX, dY, dZ, oX, oY and oZ for each node
   int carNodes = fMultiUnion->GetNumNodes();
   
   for(int i = 0 ; i < carNodes ; i++)
   {
      cout << setw(10) << setiosflags(ios::fixed) << setprecision(16) <<"    -> Node " << i+1 << ":\n       * dX = " << fBoxes[6*i] << " ; * dY = " << fBoxes[6*i+1] << " ; * dZ = " << fBoxes[6*i+2] << "\n       * oX = " << fBoxes[6*i+3] << " ; * oY = " << fBoxes[6*i+4] << " ; * oZ = " << fBoxes[6*i+5] << "\n";
   }
}

//______________________________________________________________________________   
void UVoxelFinder::CreateBoundaries()
{
// "CreateBoundaries"'s aim is to determine the slices induced by the bounding boxes,
// along each axis. The created boundaries are stored in the array "fBoundaries"
   int carNodes = fMultiUnion->GetNumNodes(); // Number of nodes in structure of
                                            // "UMultiUnion" type
   
   // Determination of the 3D extent of the "UMultIUnion" structure:
   double minimaMulti[3], maximaMulti[3];
   fMultiUnion->Extent(minimaMulti,maximaMulti);
   
   // Determination of the boundries along x, y and z axis:
   fBoundaries = new double[6*carNodes];
   
   for(int i = 0 ; i < carNodes ; i++)   
   {
      // For each node, the boundaries are created by using the array "fBoxes"
      // built in method "BuildVoxelLimits":

      // x boundaries
      fBoundaries[2*i] = fBoxes[6*i+3]-fBoxes[6*i];
      fBoundaries[2*i+1] = fBoxes[6*i+3]+fBoxes[6*i];
      // y boundaries
      fBoundaries[2*i+2*carNodes] = fBoxes[6*i+4]-fBoxes[6*i+1];
      fBoundaries[2*i+2*carNodes+1] = fBoxes[6*i+4]+fBoxes[6*i+1];
      // z boundaries
      fBoundaries[2*i+4*carNodes] = fBoxes[6*i+5]-fBoxes[6*i+2];
      fBoundaries[2*i+4*carNodes+1] = fBoxes[6*i+5]+fBoxes[6*i+2];
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
   const double tolerance = 10E-10; // Minimal distance to discrminate two boundaries.
   int carNodes = fMultiUnion->GetNumNodes();
   int *indexSortedBound = new int[2*carNodes];
   double *tempBoundaries = new double[2*carNodes];
   
   // x axis:
   // -------
   int numberBoundaries = 0; // Total number of boundaries after treatment:
   UUtils::Sort(2*carNodes,&fBoundaries[0],&indexSortedBound[0],false);

   int carNodes2x = 2*carNodes;
   for(int i = 0 ; i < carNodes2x; i++)
   {
      if(numberBoundaries == 0)
      {
         tempBoundaries[numberBoundaries] = fBoundaries[indexSortedBound[i]];
         numberBoundaries++;
         continue;
      }

      // If two successive boundaries are too close from each other, only the first one is considered      
      if(std::abs(tempBoundaries[numberBoundaries-1]-fBoundaries[indexSortedBound[i]]) > tolerance)
      {
         tempBoundaries[numberBoundaries] = fBoundaries[indexSortedBound[i]];
         numberBoundaries++;         
      }
   }
   
   if(fXSortedBoundaries) delete [] fXSortedBoundaries;
   fXSortedBoundaries = new double[numberBoundaries];
   memcpy(fXSortedBoundaries,&tempBoundaries[0],numberBoundaries*sizeof(double));
   fXNumBound = numberBoundaries;
   
   // y axis:
   // -------
   numberBoundaries = 0; // Total number of boundaries after treatment:
   UUtils::Sort(2*carNodes,&fBoundaries[2*carNodes],&indexSortedBound[0],false);
   
   for(int i = 0 ; i < carNodes2x; i++)
   {
      if(numberBoundaries == 0)
      {
         tempBoundaries[numberBoundaries] = fBoundaries[2*carNodes+indexSortedBound[i]];
         numberBoundaries++;
         continue;
      }
      
      if(std::abs(tempBoundaries[numberBoundaries-1]-fBoundaries[2*carNodes+indexSortedBound[i]]) > tolerance)
      {
         tempBoundaries[numberBoundaries] = fBoundaries[2*carNodes+indexSortedBound[i]];
         numberBoundaries++;         
      }
   }
   
   if(fYSortedBoundaries) delete fYSortedBoundaries;
   fYSortedBoundaries = new double[numberBoundaries];
   memcpy(fYSortedBoundaries,&tempBoundaries[0],numberBoundaries*sizeof(double));
   fYNumBound = numberBoundaries;
   
   // z axis:
   // -------
   numberBoundaries = 0; // Total number of boundaries after treatment:
   UUtils::Sort(2*carNodes,&fBoundaries[4*carNodes],&indexSortedBound[0],false);
   
   for(int i = 0 ; i < carNodes2x; i++)
   {
      if(numberBoundaries == 0)
      {
         tempBoundaries[numberBoundaries] = fBoundaries[4*carNodes+indexSortedBound[i]];
         numberBoundaries++;
         continue;
      }
      
      if(std::abs(tempBoundaries[numberBoundaries-1]-fBoundaries[4*carNodes+indexSortedBound[i]]) > tolerance)
      {
         tempBoundaries[numberBoundaries] = fBoundaries[4*carNodes+indexSortedBound[i]];
         numberBoundaries++;         
      }
   }
   
   if(fZSortedBoundaries) delete fZSortedBoundaries;
   fZSortedBoundaries = new double[numberBoundaries];
   memcpy(fZSortedBoundaries,&tempBoundaries[0],numberBoundaries*sizeof(double));
   fZNumBound = numberBoundaries;

   delete indexSortedBound;
   delete tempBoundaries;

}

//______________________________________________________________________________   
void UVoxelFinder::DisplayBoundaries()
{
// Prints the positions of the boundaries of the slices on the three axis:
   
   cout << " * X axis:" << endl << "    | ";
   for(int i = 0 ; i < fXNumBound ; i++)
   {
      cout << setw(10) << setiosflags(ios::fixed) << setprecision(16) << fXSortedBoundaries[i];
//      printf("%19.15f ",fXSortedBoundaries[i]);      
//      cout << fXSortedBoundaries[i] << " ";
      if(i != fXNumBound-1) cout << "-> ";
   }
   cout << "|" << endl << "Number of boundaries: " << fXNumBound << endl;
   
   cout << " * Y axis:" << endl << "    | ";
   for(int i = 0 ; i < fYNumBound ; i++)
   {
      cout << setw(10) << setiosflags(ios::fixed) << setprecision(16) << fYSortedBoundaries[i];   
//      printf("%19.15f ",fYSortedBoundaries[i]);   
//      cout << fYSortedBoundaries[i] << " ";
      if(i != fYNumBound-1) cout << "-> ";
   }
   cout << "|" << endl << "Number of boundaries: " << fYNumBound << endl;
      
   cout << " * Z axis:" << endl << "    | ";
   for(int i = 0 ; i < fZNumBound ; i++)
   {
      cout << setw(10) << setiosflags(ios::fixed) << setprecision(16) << fZSortedBoundaries[i];   
//      printf("%19.15f ",fXSortedBoundaries[i]);
//      cout << fZSortedBoundaries[i] << " ";
      if(i != fZNumBound-1) printf("-> ");
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
   const double localTolerance = 10e-4;
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
   int numberByte;
   int position;
   
   // Loop on x slices:
   fNumNodesSliceX = new int[xNumslices];
   
   for(int i = 0 ; i < xNumslices ; i++)
   {
      bits = &storage[current];
   
      // Loop on the nodes:
      for(int j = 0 ; j < carNodes ; j++)
      {
         // Determination of the minimum and maximum position along x
         // of the bounding boxe of each node:
         xbmin = fBoxes[6*j+3]-fBoxes[6*j];
         xbmax = fBoxes[6*j+3]+fBoxes[6*j];
         
         // Is the considered node inside slice?
         if((xbmin - fXSortedBoundaries[i+1]) > -localTolerance) continue;
         if((xbmax - fXSortedBoundaries[i]) < localTolerance) continue;
         
         // The considered node is contained in the current slice:
         fNumNodesSliceX[i]++;
         
         // Storage of the number of the node in the array:
         numberByte = j/8;
         position = j%8;
         bits[numberByte] |= (1 << position);                  
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
   
   for(int i = 0 ; i < yNumslices ; i++)
   {
      bits = &storage[current];
   
      // Loop on the nodes:
      for(int j = 0 ; j < carNodes ; j++)
      {
         // Determination of the minimum and maximum position along x
         // of the bounding boxe of each node:
         ybmin = fBoxes[6*j+4]-fBoxes[6*j+1];
         ybmax = fBoxes[6*j+4]+fBoxes[6*j+1];
         
         // Is the considered node inside slice?
         if((ybmin - fYSortedBoundaries[i+1]) > -localTolerance) continue;
         if((ybmax - fYSortedBoundaries[i]) < localTolerance) continue;
         
         // The considered node is contained in the current slice:
         fNumNodesSliceY[i]++;
         
         // Storage of the number of the node in the array:
         numberByte = j/8;
         position = j%8;
         bits[numberByte] |= (1 << position);                  
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
   
   for(int i = 0 ; i < zNumslices ; i++)
   {
      bits = &storage[current];
   
      // Loop on the nodes:
      for(int j = 0 ; j < carNodes ; j++)
      {
         // Determination of the minimum and maximum position along x
         // of the bounding boxe of each node:
         zbmin = fBoxes[6*j+5]-fBoxes[6*j+2];
         zbmax = fBoxes[6*j+5]+fBoxes[6*j+2];
         
         // Is the considered node inside slice?
         if((zbmin - fZSortedBoundaries[i+1]) > -localTolerance) continue;
         if((zbmax - fZSortedBoundaries[i]) < localTolerance) continue;
         
         // The considered node is contained in the current slice:
         fNumNodesSliceZ[i]++;
         
         // Storage of the number of the node in the array:
         numberByte = j/8;
         position = j%8;
         bits[numberByte] |= (1 << position);                  
      }
      current += nperslice;
   }
   fNz = current;
   fMemoryZ = new char[fNz];
   memcpy(fMemoryZ,storage,current*sizeof(char));      

   delete storage;
}

//______________________________________________________________________________   
void UVoxelFinder::GetCandidatesAsString(const char* mask, std::string &result)
{
   // Decodes the candidates in mask as string.
   char buffer[30] = {0};
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
void UVoxelFinder::DisplayListNodes()
{
// Prints which solids are present in the slices previously elaborated.
   int carNodes = fMultiUnion->GetNumNodes();   
   int nperslice = 1+(carNodes-1)/(8*sizeof(char));
   string result = "";
   
   cout << " * X axis:" << endl;
   for(int i = 0 , j = 0 ; i < fXNumBound-1 ; i++ , j += nperslice)
   {
      cout << "    Slice #" << i+1 << ": [" << fXSortedBoundaries[i] << " ; " << fXSortedBoundaries[i+1] << "] -> ";
      
      GetCandidatesAsString(&fMemoryX[j], result);
      cout << "[ " << result.c_str() << "]  " << endl;
   }
   
   cout << " * Y axis:" << endl;
   for(int i = 0 , j = 0 ; i < fYNumBound-1 ; i++ , j += nperslice)
   {
      cout << "    Slice #" << i+1 << ": [" << fYSortedBoundaries[i] << " ; " << fYSortedBoundaries[i+1] << "] -> ";
      
      GetCandidatesAsString(&fMemoryY[j], result);
      cout << "[ " << result.c_str() << "]  " << endl;
   }
   
   cout << " * Z axis:" << endl;
   for(int i = 0 , j = 0 ; i < fZNumBound-1 ; i++ , j += nperslice)
   {
      cout << "    Slice #" << i+1 << ": [" << fZSortedBoundaries[i] << " ; " << fZSortedBoundaries[i+1] << "] -> ";
      
      GetCandidatesAsString(&fMemoryZ[j], result);
      cout << "[ " << result.c_str() << "]  " << endl;
   }    
}

//______________________________________________________________________________   
void UVoxelFinder::Voxelize()
{
   // Method creating the voxelized structure
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

   string result = "";  
   int carNodes = fMultiUnion->GetNumNodes();
   int nperslices = 1+(carNodes-1)/(8*sizeof(char));
   
   // Voxelized structure:      
   char *maskResult = new char[nperslices];
   
   for(int i = 0 ; i < nperslices ; i++)
   {
      maskResult[i] = fMemoryX[nperslices*(indexX-1)+i]
                         & fMemoryY[nperslices*(indexY-1)+i]
                         & fMemoryZ[nperslices*(indexZ-1)+i];
   } 
   GetCandidatesAsString(&(maskResult[0]),result);
   cout << "   Candidates in voxel [" << indexX << " ; " << indexY << " ; " << indexZ << "]: ";
   cout << "[ " << result.c_str() << "]  " << endl;   

   delete maskResult;
}


//______________________________________________________________________________       
// Method returning the candidates corresponding to the passed point
void UVoxelFinder::GetCandidatesVoxelArray(const UVector3 &point, vector<int> &list)
{
	list.clear();
   int carNodes = fMultiUnion->GetNumNodes();
   if(carNodes == 1)
   {
      if(fXSortedBoundaries)
         if(point.x < fXSortedBoundaries[0] || point.x > fXSortedBoundaries[1]) return;

      if(fYSortedBoundaries)
         if(point.y < fYSortedBoundaries[0] || point.y > fYSortedBoundaries[1]) return;

      if(fZSortedBoundaries)
         if(point.z < fZSortedBoundaries[0] || point.z > fZSortedBoundaries[1]) return;
      list.push_back(0);
   }
   else
   {
	  int nperslices = 1+(carNodes-1)/(8*sizeof(char));   

      int bytesCorrected = nperslices + (sizeof(unsigned int) - nperslices%sizeof(unsigned int));

      int resultBinSearchX = UUtils::BinarySearch(fXNumBound, fXSortedBoundaries, point.x); 
      if((resultBinSearchX == -1) || (resultBinSearchX == fXNumBound-1)) return;

      int resultBinSearchY = UUtils::BinarySearch(fYNumBound, fYSortedBoundaries, point.y);         
      if((resultBinSearchY == -1) || (resultBinSearchY == fYNumBound-1)) return;

      int resultBinSearchZ = UUtils::BinarySearch(fZNumBound, fZSortedBoundaries, point.z);      
      if((resultBinSearchZ == -1) || (resultBinSearchZ == fZNumBound-1)) return;
  
      bool bitwiseOrX = (resultBinSearchX && (point.x == fXSortedBoundaries[resultBinSearchX]));

      bool bitwiseOrY = (resultBinSearchY && (point.y == fYSortedBoundaries[resultBinSearchY]));

	  bool bitwiseOrZ = (resultBinSearchZ && (point.z == fZSortedBoundaries[resultBinSearchZ]));

	  unsigned int mask, maskAxis; 
	  int limit = (bytesCorrected+1)/sizeof(unsigned int);
      for(int i = 0 ; i < limit; i++)
      { 
         // Logic "and" of the masks along the 3 axes x, y, z:
		 mask = ((unsigned int *)(fMemoryX+nperslices*resultBinSearchX))[i];
         if (bitwiseOrX) mask |= ((unsigned int *)(fMemoryX+nperslices*(resultBinSearchX-1)))[i];
		 if (!mask) continue; // try to remove it.. is faster?

		 maskAxis = ((unsigned int *)(fMemoryY+nperslices*resultBinSearchY))[i];
         if (bitwiseOrY) maskAxis |= ((unsigned int *)(fMemoryY+nperslices*(resultBinSearchY-1)))[i];
		 if (!(mask &= maskAxis)) continue; // try to remove it.. is faster?
         
		 maskAxis = ((unsigned int *)(fMemoryZ+nperslices*resultBinSearchZ))[i];
         if (bitwiseOrZ) maskAxis |= ((unsigned int *)(fMemoryZ+nperslices*(resultBinSearchZ-1)))[i];
		  if (!(mask &= maskAxis)) continue; // try to remove it.. is faster?

		  int currentBit = 8*sizeof(unsigned int)*i;
		  for(int bit = 0; bit < (int) (8*sizeof(unsigned int)); bit++)
		  {
			 if(currentBit >= carNodes) return;
			 if (mask & (1<<bit)) list.push_back(currentBit);
			 currentBit++;
		  }
      }   
   }
   return;
}


//______________________________________________________________________________       
vector<int> UVoxelFinder::GetCandidatesVoxelArrayOld(UVector3 point)
{
// Method returning the candidates corresponding to the passed point
   int carNodes = fMultiUnion->GetNumNodes();
   int nperslices = 1+(carNodes-1)/(8*sizeof(char));   
   vector<int> voidList, checkList;
   bool bitwiseOrX = false, bitwiseOrY = false, bitwiseOrZ = false;

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
      
      int bytesCorrected = nperslices + (sizeof(unsigned int) - nperslices%sizeof(unsigned int));

      resultBinSearchX = UUtils::BinarySearch(fXNumBound, fXSortedBoundaries, point.x); 
      if((resultBinSearchX == -1) || (resultBinSearchX == fXNumBound-1)) return voidList;

      resultBinSearchY = UUtils::BinarySearch(fYNumBound, fYSortedBoundaries, point.y);         
      if((resultBinSearchY == -1) || (resultBinSearchY == fYNumBound-1)) return voidList;

      resultBinSearchZ = UUtils::BinarySearch(fZNumBound, fZSortedBoundaries, point.z);      
      if((resultBinSearchZ == -1) || (resultBinSearchZ == fZNumBound-1)) return voidList;
  
      if((point.x == fXSortedBoundaries[resultBinSearchX]) && (resultBinSearchX != 0))
      {
         bitwiseOrX = true;         
      }      
      if((point.y == fYSortedBoundaries[resultBinSearchY]) && (resultBinSearchY != 0))
      {
         bitwiseOrY = true;         
      } 
      if((point.z == fZSortedBoundaries[resultBinSearchZ]) && (resultBinSearchZ != 0))
      {
         bitwiseOrZ = true;         
      }   

      char *maskResult = new char[bytesCorrected];             
      char *maskX = new char[bytesCorrected];
      memset(maskX,0, bytesCorrected*sizeof(char));      
      char *maskY = new char[bytesCorrected];
      memset(maskY,0, bytesCorrected*sizeof(char));       
      char *maskZ = new char[bytesCorrected];      
      memset(maskZ,0, bytesCorrected*sizeof(char));             

      for(int i = 0 ; i < (int)((bytesCorrected+1)/sizeof(unsigned int)) ; i++)
      {
         // Along X axis:
         if(bitwiseOrX == true)
         {
            ((unsigned int *)(maskX))[i] = ((unsigned int *)(fMemoryX+nperslices*resultBinSearchX))[i] | ((unsigned int *)(fMemoryX+nperslices*(resultBinSearchX-1)))[i];            
         }
         else
         {
            ((unsigned int *)(maskX))[i] = ((unsigned int *)(fMemoryX+nperslices*resultBinSearchX))[i];              
         }
         
         // Along Y axis:
         if(bitwiseOrY == true)
         {
            ((unsigned int *)(maskY))[i] = ((unsigned int *)(fMemoryY+nperslices*resultBinSearchY))[i] | ((unsigned int *)(fMemoryY+nperslices*(resultBinSearchY-1)))[i];            
         }
         else
         {
            ((unsigned int *)(maskY))[i] = ((unsigned int *)(fMemoryY+nperslices*resultBinSearchY))[i];              
         }
         
         // Along Z axis:
         if(bitwiseOrZ == true)
         {
            ((unsigned int *)(maskZ))[i] = ((unsigned int *)(fMemoryZ+nperslices*resultBinSearchZ))[i] | ((unsigned int *)(fMemoryZ+nperslices*(resultBinSearchZ-1)))[i];            
         }
         else
         {
            ((unsigned int *)(maskZ))[i] = ((unsigned int *)(fMemoryZ+nperslices*resultBinSearchZ))[i];              
         }                 
         
         // Logic "and" of the masks along the 3 axes:
         ((unsigned int *)(maskResult))[i] = ((unsigned int *)(maskX))[i] & ((unsigned int *)(maskY))[i] & ((unsigned int *)(maskZ))[i];          
      }

//      return GetCandidatesAsVector3(maskResult); 
      checkList = Intersect(maskResult);

      delete maskResult;
      delete maskX;
      delete maskY;
      delete maskZ;

	  return checkList;
   }
}

//______________________________________________________________________________       
vector<int> UVoxelFinder::Intersect(char* mask)
{
// Return the list of nodes corresponding to one array of bits

   int carNodes = fMultiUnion->GetNumNodes();
   int nperslices = 1+(carNodes-1)/(8*sizeof(char));
   int bytesCorrected = nperslices + (sizeof(unsigned int) - nperslices%sizeof(unsigned int));   
  
   vector<int> outcomeIntersect;
   int temp;   
   
   for(int currentByte = 0 ; currentByte < (int)((bytesCorrected+1)/sizeof(unsigned int)) ; currentByte++)
   {
      unsigned int byte = ((unsigned int *)(mask))[currentByte];
      if (!byte) continue;
 
      for(int currentBit = 0 ; currentBit < (int)(8*sizeof(unsigned int)) ; currentBit++)
      {
         temp = (int)(8*sizeof(unsigned int)*currentByte+currentBit);
         if(temp >= carNodes) return outcomeIntersect; 
      
         if (byte & (1<<currentBit))
         {  
            outcomeIntersect.push_back(temp);
         }
      }
   }
   return outcomeIntersect;   
}

//______________________________________________________________________________       
double* UVoxelFinder::GetBoxes()
{
   return fBoxes;
}

//______________________________________________________________________________       
int UVoxelFinder::OutcomeBinarySearch(double position, VUSolid::EAxisType axis)
{
   if(axis == VUSolid::eXaxis) return UUtils::BinarySearch(fXNumBound, fXSortedBoundaries, position);
   else if(axis == VUSolid::eYaxis) return UUtils::BinarySearch(fYNumBound, fYSortedBoundaries, position);   
   else return UUtils::BinarySearch(fZNumBound, fZSortedBoundaries, position);            
}

//______________________________________________________________________________       
int UVoxelFinder::GetNumSlices(VUSolid::EAxisType axis)
{
   if(axis == VUSolid::eXaxis) return fXNumBound;
   else if(axis == VUSolid::eYaxis) return fYNumBound;   
   else return fZNumBound;      
}

//______________________________________________________________________________       
double* UVoxelFinder::GetXSortedBoundaries()
{
   return fXSortedBoundaries;
}

//______________________________________________________________________________       
double* UVoxelFinder::GetYSortedBoundaries()
{
   return fYSortedBoundaries;
}

//______________________________________________________________________________       
double* UVoxelFinder::GetZSortedBoundaries()
{
   return fZSortedBoundaries;
}
