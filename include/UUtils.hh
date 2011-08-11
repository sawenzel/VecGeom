#ifndef USOLIDS_UUtils
#define USOLIDS_UUtils
////////////////////////////////////////////////////////////////////////////////
//
//  UUtils - Utility namespace providing common constants and mathematical
//      utilities.
//
////////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <limits>
#include <float.h>
#include <math.h>

struct UTransform3D;
struct UVector3;

namespace UUtils {
  
  // Mathematical constants
   // Abs
  inline short  Abs(short d);
  inline int    Abs(int d);
  inline long   Abs(long d);
  inline float  Abs(float d);
  inline double Abs(double d);
  // Sign
  inline short  Sign(short a, short b);
  inline int    Sign(int a, int b);
  inline long   Sign(long a, long b);
  inline float  Sign(float a, float b);
  inline double Sign(double a, double b);

   // Min, Max of two scalars
  inline short  Min(short a, short b);
  inline int    Min(int a, int b);
  inline unsigned int Min(unsigned int a, unsigned int b);
  inline long   Min(long a, long b);
  inline unsigned long   Min(unsigned long a, unsigned long b);
  inline float  Min(float a, float b);
  inline double Min(double a, double b);

  inline short  Max(short a, short b);
  inline int    Max(int a, int b);
  inline unsigned int Max(unsigned int a, unsigned int b);
  inline long   Max(long a, long b);
  inline unsigned long   Max(unsigned long a, unsigned long b);
  inline float  Max(float a, float b);
  inline double Max(double a, double b);

  // Trigonometric
  static const double kPi       = 3.14159265358979323846;
  static const double kTwoPi    = 2.0 * kPi;
  static const double kRadToDeg = 180.0 / kPi;
  static const double kDegToRad = kPi / 180.0;
  static const double kSqrt2    = 1.4142135623730950488016887242097;
  static const double kInfinity = DBL_MAX;

  inline double Infinity();
  inline double Sin(double);
  inline double Cos(double);
  inline double Tan(double);
  inline double ASin(double);
  inline double ACos(double);
  inline double ATan(double);
  inline double ATan2(double, double);
  inline double Sqrt(double x);

  // Comparing floating points
  inline bool AreEqualAbs(double af, double bf, double epsilon) {
    //return true if absolute difference between af and bf is less than epsilon
    return UUtils::Abs(af-bf) < epsilon;
  }
  inline bool AreEqualRel(double af, double bf, double relPrec) {
    //return true if relative difference between af and bf is less than relPrec
    return UUtils::Abs(af-bf) <= 0.5*relPrec*(UUtils::Abs(af)+UUtils::Abs(bf));
  }   

  // Locate Min, Max element number in an array
  long  LocMin(long n, const double *a);
  long  LocMax(long n, const double *a);

  // Sorting
  void Sort(int n, const double* a, int* index, bool down = true);
  template <typename Element, typename Index>
  void Sort(Index n, const Element* a, Index* index, bool down=true);
  template <typename Iterator, typename IndexIterator>
  void SortItr(Iterator first, Iterator last, IndexIterator index, bool down=true);

  // TransformLimits: Use the transformation to convert the local limits defined
  // by min/max vectors to the master frame. Returns modified limits.
  void TransformLimits(UVector3 &min, UVector3 &max, const UTransform3D *transformation);    

  // Templates:
  template<typename T>
  struct CompareDesc {

     CompareDesc(T d) : fData(d) {}

     template<typename Index>
     bool operator()(Index i1, Index i2) {
        return *(fData + i1) > *(fData + i2);
     }

     T fData;
  };

  template<typename T>
  struct CompareAsc {
  
     CompareAsc(T d) : fData(d) {}
  
     template<typename Index>
     bool operator()(Index i1, Index i2) {
        return *(fData + i1) < *(fData + i2);
     }

     T fData;
  };

  // Binary search
  int BinarySearch(int n, const double *array, double value);

  // Equations
//  bool         RootsQuadratic(const double coef[3], double xmin, double xmax);
//  bool         RootsCubic(const double coef[4],double &x1, double &x2, double &x3);   
}
  
    
//____________________________________________________________________________
inline double UUtils::Infinity() { 
   // returns an infinity as defined by the IEEE standard
   return std::numeric_limits<double>::infinity();
}
//---- Abs ---------------------------------------------------------------------
inline short UUtils::Abs(short d)
   { return (d >= 0) ? d : -d; }

inline int UUtils::Abs(int d)
   { return (d >= 0) ? d : -d; }

inline long UUtils::Abs(long d)
   { return (d >= 0) ? d : -d; }

inline float UUtils::Abs(float d)
   { return (d >= 0) ? d : -d; }

inline double UUtils::Abs(double d)
   { return (d >= 0) ? d : -d; }
  
//---- Sign --------------------------------------------------------------------
inline short UUtils::Sign(short a, short b)
   { return (b >= 0) ? Abs(a) : -Abs(a); }

inline int UUtils::Sign(int a, int b)
   { return (b >= 0) ? Abs(a) : -Abs(a); }

inline long UUtils::Sign(long a, long b)
   { return (b >= 0) ? Abs(a) : -Abs(a); }

inline float UUtils::Sign(float a, float b)
   { return (b >= 0) ? Abs(a) : -Abs(a); }

inline double UUtils::Sign(double a, double b)
   { return (b >= 0) ? Abs(a) : -Abs(a); }

//---- Min ---------------------------------------------------------------------
inline short UUtils::Min(short a, short b)
   { return a <= b ? a : b; }

inline int UUtils::Min(int a, int b)
   { return a <= b ? a : b; }

inline unsigned int UUtils::Min(unsigned int a, unsigned int b)
   { return a <= b ? a : b; }

inline long UUtils::Min(long a, long b)
   { return a <= b ? a : b; }

inline unsigned long UUtils::Min(unsigned long a, unsigned long b)
   { return a <= b ? a : b; }

inline float UUtils::Min(float a, float b)
   { return a <= b ? a : b; }

inline double UUtils::Min(double a, double b)
   { return a <= b ? a : b; }

//---- Max ---------------------------------------------------------------------
inline short UUtils::Max(short a, short b)
   { return a >= b ? a : b; }

inline int UUtils::Max(int a, int b)
   { return a >= b ? a : b; }

inline unsigned int UUtils::Max(unsigned int a, unsigned int b)
   { return a >= b ? a : b; }

inline long UUtils::Max(long a, long b)
   { return a >= b ? a : b; }

inline unsigned long UUtils::Max(unsigned long a, unsigned long b)
   { return a >= b ? a : b; }

inline float UUtils::Max(float a, float b)
   { return a >= b ? a : b; }

inline double UUtils::Max(double a, double b)
   { return a >= b ? a : b; }

//---- Trigonometric------------------------------------------------------------
inline double UUtils::Sin(double x)
   { return sin(x); }

inline double UUtils::Cos(double x)
   { return cos(x); }

inline double UUtils::Tan(double x)
   { return tan(x); }

inline double UUtils::ASin(double x)
   { if (x < -1.) return -kPi/2;
     if (x >  1.) return  kPi/2;
     return asin(x);
   }

inline double UUtils::ACos(double x)
   { if (x < -1.) return kPi;
     if (x >  1.) return 0;
     return acos(x);
   }

inline double UUtils::ATan(double x)
   { return atan(x); }

inline double UUtils::ATan2(double y, double x)
   { if (x != 0) return  atan2(y, x);
     if (y == 0) return  0;
     if (y >  0) return  kPi/2;
     else        return -kPi/2;
   }

inline double UUtils::Sqrt(double x)
   { return sqrt(x); }

#endif
