#include "UUtils.hh"
///////////////////////////////////////////////////////////////////////////////
//
//  UUtils - Utility namespace providing common constants and mathematical
//      utilities.
//
////////////////////////////////////////////////////////////////////////////////

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
