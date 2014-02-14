#include "Vector3DFast.h"
#include "Vector3D.h"
#include <iostream>
#include "Vc/vector.h"

bool containsold( Vector3D const & point, Vector3D const & p ) 
{
  if ( std::abs(point.x) > p.x ) return false;
  if ( std::abs(point.y) > p.y ) return false;
  if ( std::abs(point.z) > p.z ) return false;
  return true; 
}


void test1( Vector3DFast const & point, Vector3DFast & x ) 
{
  x+=point;
}


void test2( Vector3DFast const & point, Vector3DFast & x ) 
{
  x=point + point;
}

bool containsnew( Vector3DFast const & point, Vector3DFast const & p ) 
{
  Vector3DFast tmp = point.Abs();
  return ! tmp.IsAnyLargerThan( p ); 
}


bool transform( FastTransformationMatrix const & m, Vector3DFast const & master, Vector3DFast & local )
{
  m.MasterToLocal( master,local );
}


bool transform2( FastTransformationMatrix const & m, Vector3DFast const & master, Vector3DFast & local )
{
  m.LocalToMaster( master,local );
}


void abs0( Vc::double_v const & a, Vc::double_v  &b )
{
  b=Vc::abs( a );
}

void abs1( Vector3DFast const & a, Vector3DFast &b )
{
  b=a.Abs();
}

void abs2( Vector3D const & a, Vector3D &b )
{
  b.x=std::abs(a.x);
  b.y=std::abs(a.y);
  b.z=std::abs(a.z);
}

void ass1( Vector3DFast const & a, Vector3DFast &b )
{
  b=a;
}


void foo( Vector3D const & a, Vector3D &b )
{
  b+=a;
  b*=a;
  b+=a;
  b/=a;
}

void bar( Vector3DFast const & a, Vector3DFast &b )
{
  b+=a;
  b*=a;
  b+=a;
  b/=a;
}

double baz( Vector3DFast const & a, Vector3DFast const &b )
{
  return a.ScalarProduct(b);
}

double xxx( Vector3D const & a, Vector3D const &b )
{
  return Vector3D::scalarProduct(a,b);
}


int main()
{
  Vector3DFast x(1,2,3);
  Vector3DFast y(1,2,3);
  Vector3D x2(1,2,3);
  Vector3D y2(1,2,3);

  foo(x2,y2);
  bar(x,y);

  double z = baz(x,y);

  std::cerr << y2 << std::endl;
  std::cerr << y << std::endl;
  std::cerr << y << std::endl;

  // test new contains functionality
  Vector3DFast p1( -20,9,9 );
  Vector3DFast p2( 9,9,9 );
  Vector3D p1o( -20,9,9 );
  Vector3D p2o( 9,9,9 );

  Vector3DFast para(10,10,10);
  Vector3D parao(10,10,10);

  std::cerr << containsold(p1o, parao)  << " " << containsnew(p1, para) << std::endl;
  std::cerr << containsold(p2o, parao)  << " " << containsnew(p2, para) << std::endl;

  FastTransformationMatrix m(10,0,-10,34,0,45);
  m.print();
  Vector3DFast a(1,1,1);
  Vector3DFast b(1,0,0);
  Vector3DFast tmp;

  m.LocalToMaster(a, b);
  std::cerr << b << std::endl;  

  m.MasterToLocal(b, tmp);
  std::cerr << tmp << std::endl;  

  para.print();

  test2( a, b );
  std::cerr << a << std::endl;
  std::cerr << b << std::endl;
  Vector3DFast v=a;
  std::cerr << v << std::endl;
 

  return 1;
}
