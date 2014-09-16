/// @file TorusImplementation.h

#ifndef VECGEOM_VOLUMES_KERNEL_TORUSIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_TORUSIMPLEMENTATION_H_


#include "base/Global.h"
#include "base/Transformation3D.h"
#include "volumes/kernel/GenericKernels.h"
#include "volumes/UnplacedTorus.h"
#include <math.h>

#include <Vc/Vc>

namespace VECGEOM_NAMESPACE {



using namespace Vc;

inline
double_v Vccbrt( double_v x )
{
  Vc::double_v xorig=x;
  x=Vc::abs(x);
  Vc::double_v tmp= Vc::exp(0.33333333333333333333*Vc::log(x) );
  return tmp.copySign(xorig);
}



template<class T>
class Complex
{
 private:
  T fR, fI;

 public:

 Complex() : fR(T(0.)), fI(T(0.)) {}
 Complex(T x, T y) : fR(x), fI(y) {}
  ~Complex(){}

  // copy constructor
  Complex( const Complex & other )
    {
      fR=other.fR;
      fI=other.fI;
    }

 Complex& operator=( const Complex<T>& x)
   {
     fR=x.fR;
     fI=x.fI;
     return *this;
   }

 template <typename S>
 Complex& operator=( const S& x)
   {
     fR=x;
     fI=0;
     return *this;
   }

 T real() const {return fR;}
 T imag() const {return fI;}

 // problem: how can we vary the math functions here
 // for Vc:: argument dependent lookup might help
 T carg() const {return atan2(fI,fR);}
 T cabs() const {return sqrt(fR*fR+fI*fI);}

 Complex conj() const { return Complex(fR,-fI);}

 Complex operator+=( Complex const & x ) const
 {
   return Complex( fR + x.real(), fI + x.imag() );
 }

 Complex operator-() const
 {
   return Complex(-fR,-fI);
 }

}; // end class complex

template <typename T>
inline
Complex<T> operator+( Complex<T> const & x, Complex<T> const & y )
{
  return Complex<T>( x.real() + y.real(), x.imag() + y.imag() );
}

template <typename T>
inline
Complex<T> operator+( Complex<T> const & x, T const & y )
{
  return Complex<T>( x.real()+y , x.imag()  );
}

template <typename T>
inline
Complex<T> operator+( T const & y, Complex<T> const & x )
{
  return Complex<T>( x.real()+y , x.imag()  );
}


template <typename T>
inline
Complex<T> operator-( Complex<T> const & x, Complex<T> const & y )
{
  return Complex<T>( x.real() - y.real(), x.imag() - y.imag() );
}

template <typename T>
inline
Complex<T> operator-( Complex<T> const & x, T const & y )
{
  return Complex<T>( x.real() - y, x.imag() );
}

template <typename T>
inline
Complex<T> operator-(  T const & y,  Complex<T> const & x)
{
  return Complex<T>( y - x.real() , -x.imag() );
}


template <typename T>
inline
Complex<T> operator*( Complex<T> const & x, Complex<T> const & y )
{
  return Complex<T>( x.real()*y.real() - x.imag()*y.imag(), x.real()*y.imag() + x.imag()*y.real() );
}

template <typename T, typename Other>
inline
Complex<T> operator*( Complex<T> const & x, Other const & y )
{
  return Complex<T>( x.real()*y , x.imag()*y );
}

template <typename T, typename Other>
inline
Complex<T> operator*( Other const & y, Complex<T> const & x )
{
  return Complex<T>( x.real()*y , x.imag()*y );
}


// division by T
template <typename T>
inline
Complex<T> operator/( Complex<T> const & x, T const & y )
{
  // could use some fast math here
  // division by zero is not treated
  T invy = 1./y;
  return Complex<T>( x.real()*invy, x.imag()*invy );
}


template <typename T>
inline
Complex<T> operator/( T const & x, Complex<T> const & y )
{
  // multiply by conjugate
  Complex<T> tmp = y*y.conj();
  return Complex<T>( x*y.conj()) / tmp.real();
}

template <typename T>
inline
Complex<T> operator/( Complex<T> const & x, Complex<T> const & y )
{
  // multiply by conjugate
  Complex<T> tmp = y*y.conj();
  return Complex<T>( x*y.conj()) / tmp.real();
}


// standalone function for sqrt
template <typename T>
inline
Complex<T> csqrt( const Complex<T>& x )
{
  T r = x.real();
  T i = x.imag();
  T l = sqrt( r*r + i*i );
  return Complex<T>( sqrt( 0.5*(l+r) ), sqrt( 0.5*(l-r) ));
}


// standalone function for sqrt
template <typename T>
inline
Complex<T> csqrtrealargument( const T & x )
{
  T imcoef   = (x>=0.) ? 0. : 1.;
  T realcoef = (x>=0.) ? 1. : 0.;
  T l = sqrt( fabs(x) );
  return Complex<T>( realcoef * l , imcoef * l );
}

// template specialization for Vc
typedef Vc::double_v VCT;
template <>
inline
Complex<VCT> csqrtrealargument( const VCT & x )
{
  VCT::Mask real = (x>=0.);
  VCT l = Vc::sqrt( Vc::abs(x) );
  VCT realpart(Vc::Zero);
  VCT impart(Vc::Zero);
  realpart(real) = l;
  impart(!real) = l;
  return Complex<VCT>( realpart , impart );
}


// we need standalone function for cubic root
template <typename T>
inline
Complex<T> cbrt( const Complex<T>& x )
{
  // use the sin/cosine formula??
  T r;

  T sinnewangle, cosnewangle;
  if( x.imag() != 0.0 )
    {
      r = x.cabs();
      T angle = x.carg();

      T newangle = angle/3.;
      sinnewangle=sin(newangle);
      cosnewangle=cos(newangle);
      //sincos(newangle, &sinnewangle, &cosnewangle);
    }
  else
    {
      r = x.real();
      sinnewangle=0.;
      cosnewangle=1.;
    }
  // use ordinary cubic root function ( is it in Vc ?? )
  T rcbrt = pow(r,1./3);//cbrt( r );
  return Complex<T>(  rcbrt*cosnewangle, rcbrt*sinnewangle );
}

// template specialization for Vc
// we need standalone function for cubic root
template <>
inline
Complex<Vc::double_v> cbrt( const Complex<Vc::double_v>& x )
{
  // use the sin/cosine formula??
  Vc::double_v r;
  Vc::double_v sinnewangle, cosnewangle;
  Vc::double_m mask = x.imag() != 0.0;
  if( !mask.isEmpty() )
    {
      r = x.cabs();
      Vc::double_v angle = x.carg();
      Vc::double_v newangle = 0.33333333333333333*angle;
      sincos(newangle, &sinnewangle, &cosnewangle);
    }
  else
    {
      r = x.real();
      sinnewangle=Vc::Zero;
      cosnewangle=Vc::One;
    }
  // use cubic root function defined above
  Vc::double_v rcbrt = Vccbrt( r );
  return Complex<Vc::double_v>(  rcbrt*cosnewangle, rcbrt*sinnewangle );
}



template <typename T, typename stream>
stream& operator<<(stream& s, const Complex<T> & x)
{
  s << "(" << x.real() << " , " << x.imag() << ")";
  return s;
}


template <typename CT>
void solveQuartic(double a, double b, double c, double d, double e, CT * roots)
{
  // Uses Ferrari's Method; this will vectorize trivially ( at least when we treat the real and imaginary parts separately )
  double aa = a*a, aaa=aa*a, bb=b*b, bbb=bb*b;
  double alpha = -3.0*bb/(8.0*aa)   + c/a, alpha2 = alpha * alpha;
  double beta  =    bbb/(8.0*aaa) + b*c/(-2.0*aa) + d/a;
  double gamma = -3.0*bbb*b/(256.0*aaa*a) + c*bb/(16.0*aaa) + b*d/(-4.0*aa) + e/a;


  CT P = -alpha2/12.0 - gamma;
  CT Q = -alpha2*alpha/108.0 + alpha*gamma/3.0 - beta*beta/8.0;
  CT R = Q*0.5 + sqrt(Q*Q*0.25 + P*P*P/27.0);
  CT U = pow(R,1./3);
  CT y = -5.0*alpha/6.0 - U;
  if(U != 0.0) y += P/(3.0*U);
  CT W = sqrt(alpha + y + y);

  CT aRoot;
   double firstPart = b/(-4.0*a);
  CT secondPart = -3.0*alpha - 2.0*y;
  CT thirdPart = 2.0*beta/(W);

  aRoot = firstPart + 0.5 * (-W - sqrt(secondPart + thirdPart));
  roots[0] = aRoot;

  aRoot = firstPart + 0.5 * (-W + sqrt(secondPart + thirdPart));
  roots[1] = aRoot;

  aRoot = firstPart + 0.5 * (W - sqrt(secondPart - thirdPart));
  roots[2] = aRoot;

  aRoot = firstPart + 0.5 * (W + sqrt(secondPart - thirdPart));
  roots[3] = aRoot;

}

// finding all real roots
// this is Ferrari's method which is not really numerically stable but very elegant
// CT == complextype
//template <typename CT>
typedef Complex<double> CT;
inline
void solveQuartic2(double a, double b, double c, double d, double e, CT * roots)
{

  // Uses Ferrari's Method; this will vectorize trivially ( at least when we treat the real and imaginary parts separately )
  double inva=1./a;
  double invaa = inva*inva;
  double invaaa=invaa*inva;
  double bb=b*b;
  double bbb=bb*b;
  double alpha = -3.0*0.125*bb*invaa   + c*inva, alpha2 = alpha * alpha;
  double beta  =    0.125*bbb*invaaa - 0.5*b*c*invaa + d*inva;
  double gamma = -3.0*bbb*b*invaaa*inva/(256.0) + c*bb*invaaa/(16.0) - 0.25*b*d*invaa + e*inva;

  std::cerr << alpha << "\n";
  std::cerr << alpha2 << "\n";
  std::cerr << beta << "\n";
  std::cerr << gamma << "\n";

  double P = -alpha2/12.0 - gamma;
  double Q = -alpha2*alpha/108.0 + alpha*gamma/3.0 - beta*beta/8.0;
  std::cerr << "P " << P << "\n";
  std::cerr << "Q " << Q << "\n";

  double tmp = 0.25*Q*Q + P*P*P/27.;
  std::cerr << "tmp " << tmp << "\n";
  CT R = Q*0.5 + csqrtrealargument(tmp);
  std::cerr << "R " << R << "\n";
  CT U = cbrt(R);
  std::cerr << "U " << U << "\n";
  std::cerr << "U*U*U " << U*U*U << "\n";

  CT y = -5.0*alpha/6.0 - U;
  y = y + P/(3.*U);
  std::cerr << "y " << y << "\n";
  CT W = csqrt((alpha + y) + y);
  std::cerr << "W " << W << "\n";

  CT aRoot;
  double firstPart = -0.25*b*inva;
  CT secondPart = -3.0*alpha - 2.0*y;
  CT thirdPart = (2.0*beta)/W;

  aRoot = firstPart + 0.5 * (-W - csqrt(secondPart + thirdPart));
  roots[0] = aRoot;
  aRoot = firstPart + 0.5 * (-W + csqrt(secondPart + thirdPart));
  roots[1] = aRoot;
  aRoot = firstPart + 0.5 * (W - csqrt(secondPart - thirdPart));
  roots[2] = aRoot;
  aRoot = firstPart + 0.5 * (W + csqrt(secondPart - thirdPart));
  roots[3] = aRoot;
}

// finding all real roots
// this is Ferrari's method which is not really numerically stable but very elegant
// CT == complextype
//typedef Vc::double_v VCT2;
typedef Complex<VCT> CVCT;
inline
void solveQuartic2(VCT a, VCT b, VCT c, VCT d, VCT e, CVCT * roots)
{
  VCT inva=1./a;
  VCT invaa = inva*inva;
  VCT invaaa=invaa*inva;
  VCT bb=b*b;
  VCT bbb=bb*b;
  VCT alpha = -3.0*0.125*bb*invaa   + c*inva, alpha2 = alpha * alpha;
  VCT beta  = 0.125*bbb*invaaa - 0.5*b*c*invaa + d*inva;
  VCT gamma = -3.0*bbb*b*invaaa*inva/(256.0) + c*bb*invaaa/(16.0) - 0.25*b*d*invaa + e*inva;

  VCT P = -alpha2/12.0 - gamma;
  VCT Q = -alpha2*alpha/108.0 + alpha*gamma/3.0 - beta*beta/8.0;
  VCT tmp = 0.25*Q*Q + P*P*P/27.;
  CVCT R = Q*0.5 + csqrtrealargument(tmp);
  CVCT U = cbrt(R);
  CVCT y = -5.0*alpha/6.0 - U;
  y = y + P/(3.*U );
  CVCT W = csqrt((alpha + y) + y);

  CVCT aRoot;

  VCT firstPart = -0.25*b*inva;
  CVCT secondPart = -3.0*alpha - 2.0*y;
  CVCT thirdPart = (2.0*beta)/(W);

  aRoot = firstPart + 0.5 * (-W - csqrt(secondPart + thirdPart));
  roots[0] = aRoot;

  aRoot = firstPart + 0.5 * (-W + csqrt(secondPart + thirdPart));
  roots[1] = aRoot;

  aRoot = firstPart + 0.5 * (W - csqrt(secondPart - thirdPart));
  roots[2] = aRoot;

  aRoot = firstPart + 0.5 * (W + csqrt(secondPart - thirdPart));
  roots[3] = aRoot;
}





template <TranslationCode transCodeT, RotationCode rotCodeT>
struct TorusImplementation {

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void UnplacedContains( UnplacedTorus const &torus,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::bool_v &inside) {

    typedef typename Backend::precision_v Float_t;

    // TODO: do this generically WITH a generic contains/inside kernel
    // forget about sector for the moment

    Float_t rxy = Sqrt(point[0]*point[0] + point[1]*point[1]);
    Float_t radsq = ( rxy - torus.rtor() ) * (rxy - torus.rtor() ) + point[2]*point[2];
    inside = radsq > torus.rmin2() && radsq < torus.rmax2();
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void Contains(
      UnplacedTorus const &torus,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> &localPoint,
      typename Backend::bool_v &inside) {

    localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
    UnplacedContains<Backend>(torus, localPoint, inside);
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void Inside(UnplacedTorus const &torus,
                     Transformation3D const &transformation,
                     Vector3D<typename Backend::precision_v> const &point,
                     typename Backend::inside_v &inside) {

    // TODO
  }

    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    static void DistanceToIn(
        UnplacedTorus const &torus,
        Transformation3D const &transformation,
        Vector3D<typename Backend::precision_v> const &point,
        Vector3D<typename Backend::precision_v> const &direction,
        typename Backend::precision_v const &stepMax,
        typename Backend::precision_v &distance) {
      
    // TODO
  }

#include <iostream>
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static
  typename Backend::precision_v ToBoundary(
          UnplacedTorus const& torus,
          Vector3D<typename Backend::precision_v> const &point,
          Vector3D<typename Backend::precision_v> const &dir,
          Precision radius )
  {
      // Returns distance to the surface or the torus (fR,r) from a point, along
      // a direction. Point is close enough to the boundary so that the distance
      // to the torus is decreasing while moving along the given direction.

     // Compute coefficients of the quartic polynomial
     typedef typename Backend::precision_v Float_t;
     typedef typename Backend::bool_v Bool_t;

     Float_t tol = kTolerance;

     // actually a scalar product
     Float_t r0sq  = point[0]*point[0]+point[1]*point[1]+point[2]*point[2];

     // actually a scalar product
     Float_t rdotn = point[0]*dir[0]+point[1]*dir[1]+point[2]*dir[2];

     // can be precomputed
     Float_t rsumsq = torus.rtor2() + radius*radius;
     Float_t a = 4.*rdotn;
     Float_t b = 2.*(r0sq+2.*rdotn*rdotn-rsumsq+2.*torus.rtor2()*dir[2]*dir[2]);
     Float_t c = 4.*(r0sq*rdotn-rsumsq*rdotn+2.*torus.rtor2()*point[2]*dir[2]);
     Float_t d = r0sq*r0sq-2.*r0sq*rsumsq+4.*torus.rtor2()*point[2]*point[2]+(torus.rtor2()-radius*radius)*(torus.rtor2()-radius*radius);

     std::cerr << "#a " << a << "\n";
     std::cerr << "#b " << b<< "\n";
     std::cerr << "#c " << c << "\n";
     std::cerr << "#d " << d << "\n";

     // the 4 complex roots
     Complex<Float_t> roots[4];
     // get the roots
     solveQuartic2(Float_t(1.),a,b,c,d,roots);

     std::cerr << "#ROOTS " << roots[0] << "\n";
     std::cerr << "#ROOTS " << roots[1] << "\n";
     std::cerr << "#ROOTS " << roots[2] << "\n";
     std::cerr << "#ROOTS " << roots[3] << "\n";

     Float_t validdistance = kInfinity;
     Bool_t havevalidsolution = Abs(roots[0].imag()) < 1E-6 && roots[0].real() > 0.;
     MaskedAssign( havevalidsolution, roots[0].real(), &validdistance );

     havevalidsolution = Abs(roots[1].imag()) < 1E-6 && roots[1].real() > 0.;
     MaskedAssign( havevalidsolution, Min(roots[1].real(), validdistance), &validdistance );

     havevalidsolution = Abs(roots[2].imag()) < 1E-6 && roots[2].real() > 0.;
     MaskedAssign( havevalidsolution, Min(roots[2].real(), validdistance), &validdistance );

     havevalidsolution = Abs(roots[3].imag()) < 1E-6 && roots[3].real() > 0.;
     MaskedAssign( havevalidsolution, Min(roots[3].real(), validdistance), &validdistance );

     return validdistance;
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToOut(
      UnplacedTorus const &torus,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &dir,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance) {

    typedef typename Backend::precision_v Float_t;

    distance = kInfinity;

    // TODO
    // Compute distance from inside point to surface of the torus.
    bool hasphi = (torus.dphi()<kTwoPi);
    bool hasrmin = (torus.rmin()>0);

    Float_t dout = ToBoundary<Backend>(torus,point,dir,torus.rmax());
    Float_t din(kInfinity);
    if( hasrmin )
    {
       din = ToBoundary<Backend>(torus,point,dir,torus.rmin());
    }
    distance = Min(dout,din);

    if( hasphi )
    {
    // TODO
    }
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToIn(UnplacedTorus const &torus,
                         Transformation3D const &transformation,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety) {

    // TODO
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToOut(UnplacedTorus const &torus,
                          Vector3D<typename Backend::precision_v> const &point,
                          typename Backend::precision_v &safety) {

    // TODO

}

}; // end struct

} // end namespace


#endif // VECGEOM_VOLUMES_KERNEL_TORUSIMPLEMENTATION_H_
