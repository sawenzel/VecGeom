#include "UUtils.hh"
///////////////////////////////////////////////////////////////////////////////
//
//  UUtils - Utility namespace providing common constants and mathematical
//      utilities.
//
////////////////////////////////////////////////////////////////////////////////
#include <algorithm>
#include "UVector3.hh"
#include "UTransform3D.hh"

//______________________________________________________________________________
long UUtils::LocMin(long n, const double *a) {
   // Return index of array with the minimum element.
   // If more than one element is minimum returns first found.

   // Implement here since this one is found to be faster (mainly on 64 bit machines)
   // than stl generic implementation.
   // When performing the comparison,  the STL implementation needs to de-reference both the array iterator
   // and the iterator pointing to the resulting minimum location

   if  (n <= 0 || !a) return -1;
   double xmin = a[0];
   long loc = 0;
   for  (long i = 1; i < n; i++) {
      if (xmin > a[i])  {
         xmin = a[i];
         loc = i;
      }
   }
   return loc;
}
    
//______________________________________________________________________________
long UUtils::LocMax(long n, const double *a) {
   // Return index of array with the maximum element.
   // If more than one element is maximum returns first found.

   // Implement here since it is faster (see comment in LocMin function)

   if  (n <= 0 || !a) return -1;
   double xmax = a[0];
   long loc = 0;
   for  (long i = 1; i < n; i++) {
      if (xmax < a[i])  {
         xmax = a[i];
         loc = i;
      }
   }
   return loc;
}

//______________________________________________________________________________
template <typename Iterator, typename IndexIterator>
void UUtils::SortItr(Iterator first, Iterator last, IndexIterator index, bool down)
{
   // Sort the n1 elements of the Short_t array defined by its
   // iterators.  In output the array index contains the indices of
   // the sorted array.  If down is false sort in increasing order
   // (default is decreasing order).

   // NOTE that the array index must be created with a length bigger
   // or equal than the main array before calling this function.

   int i = 0;

   IndexIterator cindex = index;
   for ( Iterator cfirst = first; cfirst != last; ++cfirst )
   {
      *cindex = i++;
      ++cindex;
   }

   if ( down )
      std::sort(index, cindex, CompareDesc<Iterator>(first) );
   else
      std::sort(index, cindex, CompareAsc<Iterator>(first) );
}

//______________________________________________________________________________
template <typename Element, typename Index> void UUtils::Sort(Index n, const Element* a, Index* index, bool down)
{
   // Sort the n elements of the  array a of generic templated type Element.
   // In output the array index of type Index contains the indices of the sorted array.
   // If down is false sort in increasing order (default is decreasing order).

   // NOTE that the array index must be created with a length >= n
   // before calling this function.
   // NOTE also that the size type for n must be the same type used for the index array
   // (templated type Index)

   for(Index i = 0; i < n; i++) { index[i] = i; }
   if ( down )
      std::sort(index, index + n, CompareDesc<const Element*>(a) );
   else
      std::sort(index, index + n, CompareAsc<const Element*>(a) );
}

//______________________________________________________________________________
void UUtils::Sort(int n, const double* a, int* index, bool down)
{
   // Sort the n elements of the  array a of generic templated type Element.
   // In output the array index of type Index contains the indices of the sorted array.
   // If down is false sort in increasing order (default is decreasing order).

   // NOTE that the array index must be created with a length >= n
   // before calling this function.
   // NOTE also that the size type for n must be the same type used for the index array
   // (templated type Index)

   for(int i = 0; i < n; i++) { index[i] = i; }
   if ( down )
      std::sort(index, index + n, CompareDesc<const double*>(a) );
   else
      std::sort(index, index + n, CompareAsc<const double*>(a) );
}

//______________________________________________________________________________
void UUtils::TransformLimits(UVector3 &min, UVector3 &max, const UTransform3D *transformation)
{
   // The goal of this method is to convert the quantities min and max (representing the
   // bounding box of a given solid in its local frame) to the main frame, using
   // "transformation"
   int kIndex;
   double vertices[24];
   UVector3 tempPointConv,tempPoint;
   double currentX, currentY, currentZ;
   double miniX = kInfinity;
   double miniY = kInfinity;
   double miniZ = kInfinity;
   double maxiX = -kInfinity;
   double maxiY = -kInfinity;
   double maxiZ = -kInfinity;

   // Detemination of the vertices thanks to the extension of each solid:
      // 1st vertice:
   vertices[ 0] = min.x; vertices[ 1] = min.y; vertices[ 2] = min.z;
      // 2nd vertice:
   vertices[ 3] = min.x; vertices[ 4] = max.y; vertices[ 5] = min.z;   
      // etc.:
   vertices[ 6] = max.x; vertices[ 7] = max.y; vertices[ 8] = min.z;
   vertices[ 9] = max.x; vertices[10] = min.y; vertices[11] = min.z;
   vertices[12] = min.x; vertices[13] = min.y; vertices[14] = max.z;
   vertices[15] = min.x; vertices[16] = max.y; vertices[17] = max.z;
   vertices[18] = max.x; vertices[19] = max.y; vertices[20] = max.z;
   vertices[21] = max.x; vertices[22] = min.y; vertices[23] = max.z;   
   
   // Loop on th vertices
   for(int jIndex = 0 ; jIndex < 8 ; jIndex++)
   {
      kIndex = 3*jIndex;
      tempPoint.Set(vertices[kIndex],vertices[kIndex+1],vertices[kIndex+2]);
      // From local frame to the gobal one:
      tempPointConv = transformation->GlobalPoint(tempPoint);
     
      // Current positions on the three axis:         
      currentX = tempPointConv.x;
      currentY = tempPointConv.y;
      currentZ = tempPointConv.z;
      
      // If need be, replacement of the min & max values:
      if(currentX > maxiX) maxiX = currentX;
      if(currentX < miniX) miniX = currentX;

      if(currentY > maxiY) maxiY = currentY;
      if(currentY < miniY) miniY = currentY;  

      if(currentZ > maxiZ) maxiZ = currentZ;
      if(currentZ < miniZ) miniZ = currentZ;                             
   }
   // Recopy of the extrema in the passed pointers:
   min.Set(miniX, miniY, miniZ);
   max.Set(maxiX, maxiY, maxiZ);
}
