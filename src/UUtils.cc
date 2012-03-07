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

#include "VUSolid.hh"

using namespace std;

//______________________________________________________________________________
int UUtils::BinarySearch(int n, const double *array, double value)
{
// Binary search in an array of doubles. If match is found, function returns
   // position of element.  If no match found, function gives nearest
   // element smaller than value.

//	if (array[n-1] == value) return n - 2; // patch, let us discuss it
   
   int nabove, nbelow, middle;
   nabove = n+1;
   nbelow = 0;
   while(nabove-nbelow > 1) {
      middle = (nabove+nbelow)/2;
      if(value == array[middle-1]) 
      {
          nbelow = middle;
          break;
      }
      if (value  < array[middle-1]) nabove = middle;
      else                          nbelow = middle;
   }    
   return nbelow-1;
}   

//______________________________________________________________________________
int UUtils::BinarySearch(const std::vector<double> &vec, double value)
{
// Binary search in an array of doubles. If match is found, function returns
   // position of element.  If no match found, function gives nearest
   // element smaller than value.

//	if (array[n-1] == value) return n - 2; // patch, let us discuss it

	/*
	double myints[] = {1,2,3,4};
  vector<double> v(myints,myints+4);           // 10 20 30 30 20 10 10 20
  vector<double>::iterator low,up;

  sort (v.begin(), v.end());                // 10 10 10 20 20 20 30 30

  double val = 1;
  low=lower_bound (v.begin(), v.end(), val); //          ^
  up= upper_bound (v.begin(), v.end(), val); //                   ^

  cout << "lower_bound at position " << int(low- v.begin()) << endl;
  cout << "upper_bound at position " << int(up - v.begin()) << endl;
  */

//	return UUtils::BinarySearch(vec.size(), &vec[0], value);

	vector<double>::const_iterator begin=vec.begin(), end=vec.end();
    int res = upper_bound(begin, end, value) - begin - 1;

#ifdef DEBUG
	int resold = UUtils::BinarySearch(vec.size(), &vec[0], value);
   if (res != resold)
	   res = resold;
#endif
   
   return res;
}


//______________________________________________________________________________
int UUtils::BinarySearchLower(const std::vector<double> &vec, double value)
{
// Binary search in an array of doubles. If match is found, function returns
   // position of element.  If no match found, function gives nearest
   // element smaller than value.

//	if (array[n-1] == value) return n - 2; // patch, let us discuss it

	/*
	double myints[] = {1,2,3,4};
  vector<double> v(myints,myints+4);           // 10 20 30 30 20 10 10 20
  vector<double>::iterator low,up;

  sort (v.begin(), v.end());                // 10 10 10 20 20 20 30 30

  double val = 1;
  low=lower_bound (v.begin(), v.end(), val); //          ^
  up= upper_bound (v.begin(), v.end(), val); //                   ^

  cout << "lower_bound at position " << int(low- v.begin()) << endl;
  cout << "upper_bound at position " << int(up - v.begin()) << endl;
  */

//	return UUtils::BinarySearch(vec.size(), &vec[0], value);

	vector<double>::const_iterator begin=vec.begin(), end=vec.end();
    int res = lower_bound(begin, end, value) - begin - 1;  
   return res;
}   

   
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

/*
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
*/

//______________________________________________________________________________
void UUtils::TransformLimits(UVector3 &min, UVector3 &max, const UTransform3D &transformation)
{
   // The goal of this method is to convert the quantities min and max (representing the
   // bounding box of a given solid in its local frame) to the main frame, using
   // "transformation"
	UVector3 vertices[8] = {   // Detemination of the vertices thanks to the extension of each solid:
    UVector3(min.x, min.y, min.z), // 1st vertice:
    UVector3(min.x, max.y, min.z), // 2nd vertice:
    UVector3(max.x, max.y, min.z),
    UVector3(max.x, min.y, min.z),
    UVector3(min.x, min.y, max.z),
    UVector3(min.x, max.y, max.z),
    UVector3(max.x, max.y, max.z),
    UVector3(max.x, min.y, max.z)};

   min.Set (kInfinity); max.Set (-kInfinity);

   // Loop on th vertices
   for(int i = 0 ; i < sizeof(vertices) / sizeof(UVector3) ; i++)
   {
      // From local frame to the gobal one:
      // Current positions on the three axis:         
      UVector3 current = transformation.GlobalPoint(vertices[i]);
           
      // If need be, replacement of the min & max values:
	  if (current.x > max.x) max.x = current.x;
      if (current.x < min.x) min.x = current.x;

      if (current.y > max.y) max.y = current.y;
      if (current.y < min.y) min.y = current.y;  

      if (current.z > max.z) max.z = current.z;
      if (current.z < min.z) min.z = current.z;                             
   }
}

/*
//______________________________________________________________________________
void UUtils::TransformLimits(UVector3 &min, UVector3 &max, const UTransform3D &transformation)
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
      tempPointConv = transformation.GlobalPoint(tempPoint);
     
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
*/


double UUtils::RandomUniform(double min, double max)
{
	// srand((unsigned)time(NULL));
    double number = (double) rand() / RAND_MAX;
    double res = min + number * (max - min);
    return res;
}
