#include "Vector3DFast.h"
#include "Vector3D.h"
#include <iostream>
#include "Vc/vector.h"
#include "TransformationMatrix.h"

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


void test3( Vector3DFast const & point, Vector3DFast & x, double d )
{
  x=point + (d + 0.2) * point;
}


bool containsnew( Vector3DFast const & point, Vector3DFast const & p ) 
{
  Vector3DFast tmp = point.Abs();
  return ! tmp.IsAnyLargerThan( p ); 
}

// these are some box things
double safetyold( TransformationMatrix const * m, Vector3D const & boxp, Vector3D const & point )
{
	 double safe, safy, safz;
	 Vector3D localPoint;
	 m->MasterToLocal<1,-1>(point, localPoint);
	 safe = boxp.x - std::fabs(localPoint.x);
	 safy = boxp.y - std::fabs(localPoint.y);
	 safz = boxp.z - std::fabs(localPoint.z);
	 if (safy < safe) safe = safy;
	 if (safz < safe) safe = safz;
	 return safe;
}


double safetynew( FastTransformationMatrix const *m, Vector3DFast const & boxp, Vector3DFast const & point )
{
	 Vector3DFast localPoint;
	 m->MasterToLocal(point, localPoint);
	 Vector3DFast safe;
	 safe = boxp - localPoint;
	 return safe.Min();
}

double distancetooutold( Vector3D const & boxp, Vector3D const & point, Vector3D const & dir )
{
	 double s,smin,saf[6];
	 double newpt[3];
	 int i;
	 saf[0] = boxp.x+newpt[0];
	 saf[1] = boxp.x-newpt[0];
	 saf[2] = boxp.y+newpt[1];
	 saf[3] = boxp.y-newpt[1];
	 saf[4] = boxp.z+newpt[2];
	 saf[5] = boxp.z-newpt[2];

	 // compute distance to surface
	 smin=1E30;
	 // loop over directions
	 if (dir.x!=0)
	 {
		 // calculating with correct face of box
		 s = (dir.x>0)? (saf[1]/dir.x) : (-saf[0]/dir.x);
		 if (s < 0) return 0.0; // this is detecting pint outside?
		 if (s < smin) smin = s;
	 }

	 if (dir.y!=0)
	 {
		 // calculating with correct face of box
		 s = (dir.y>0)? (saf[3]/dir.y) : (-saf[2]/dir.y);
		 if (s < 0) return 0.0; // this is detecting pint outside?
		 if (s < smin) smin = s;
	 }

	 if (dir.y!=0)
	 {
		 // calculating with correct face of box
		 s = (dir.z>0)? (saf[5]/dir.z) : (-saf[4]/dir.z);
		 if (s < 0) return 0.0; // this is detecting pint outside?
		 if (s < smin) smin = s;
	 }
	return smin;
}


double distancetooutnew( Vector3DFast const & boxp, Vector3DFast const & point, Vector3DFast const & dir )
{
	Vector3DFast safetyPlus = boxp + point;
	Vector3DFast safetyMinus = boxp - point;
	// gather right safeties
	Vector3DFast rightSafeties = Vector3DFast::ChooseComponentsBasedOnCondition( safetyPlus, safetyMinus, dir );

	Vector3DFast distances = rightSafeties / dir;
	return distances.MinButNotNegative();
}

double distancetooutnew2( Vector3DFast const & boxp, Vector3DFast const & point, Vector3DFast const & dir )
{
	Vector3DFast safetyPlus = boxp + point;
	Vector3DFast safetyMinus = boxp - point;
	// gather right safeties
	Vector3DFast rightSafeties = Vector3DFast::ChooseComponentsBasedOnConditionFast( safetyPlus, safetyMinus, dir );

	Vector3DFast distances = rightSafeties / dir;
	return distances.MinButNotNegative();
}


double distancetooutnew2WithSafety( Vector3DFast const & boxp, Vector3DFast const & point, Vector3DFast const & dir, double & safety )
{
	Vector3DFast safetyPlus = boxp + point;
	Vector3DFast safetyMinus = boxp - point;
	// gather right safeties
	Vector3DFast rightSafeties = Vector3DFast::ChooseComponentsBasedOnConditionFast( safetyPlus, safetyMinus, dir );

	Vector3DFast distances = rightSafeties / dir;
	Vector3DFast safe = Vector3DFast::Min( safetyMinus, safetyPlus );
	safety=safe.Min();
	return distances.MinButNotNegative();
}


double distancetoinnew( Vector3DFast const & boxp, Vector3DFast const & point, Vector3DFast const & dir, double cPstep )
{
	const double delta = 1E-9;
	// here we do the point transformation
	//	Vector3D aPoint;
    //	matrix->MasterToLocal<tid,rid>(x, aPoint);

	//   aNormal.SetNull();
	Vector3DFast safety = point.Abs() - boxp;

	// check this::
	if( safety.IsAnyLargerThan(cPstep) )
		return 1E30;

	// only here we do the directional transformation
	//Vector3D aDirection;
	//matrix->MasterToLocalVec<rid>(y, aDirection);

	// Check if numerical inside

	// not yet vectorized
	bool outside = safety.IsAnyLargerThan( 0 );

	if ( !outside )
	{
/*
		// this check has to be done again

		// If point close to this surface, check against the normal
			if ( safx > -delta ) {
				return ( aPoint.x * aDirection.x > 0 ) ? Utils::kInfinity : 0.0;
			}
			if ( safy > -delta ) {
				return ( aPoint.y * aDirection.y > 0 ) ? Utils::kInfinity : 0.0;
			}
			if ( safz > -delta ) {
				return ( aPoint.z * aDirection.z > 0 ) ? Utils::kInfinity : 0.0;
			}
			// Point actually "deep" inside, return zero distance, normal un-defined
			return 0.0;
 */
	}


	// check any early return stuff ( because going away ) here:
	Vector3DFast pointtimesdirection = point*dir;
	if( Vector3DFast::ExistsIndexWhereBothComponentsPositive(safety, pointtimesdirection)) return Utils::kInfinity;

	// compute distance to surfaces
	Vector3DFast distv = safety/dir.Abs();

	// compute target points ( needs some reshuffling )
	// might be suboptimal for SSE or really scalar
	Vector3DFast hitxyplane = point + distv.GetZ()*dir;

	// the following could be made faster ( maybe ) by calculating the abs on the whole vector
	if(    std::abs(hitxyplane.GetX()) < boxp.GetX()
		&& std::abs(hitxyplane.GetY()) < boxp.GetY())
		return distv.GetZ();

	Vector3DFast hitxzplane = point + distv.GetY()*dir;
	if(    std::abs(hitxzplane.GetX()) < boxp.GetX()
		&& std::abs(hitxzplane.GetZ()) < boxp.GetZ())
		return distv.GetY();

	Vector3DFast hityzplane = point + distv.GetX()*dir;
	if(	   std::abs(hityzplane.GetY()) < boxp.GetY()
		&& std::abs(hityzplane.GetZ()) < boxp.GetZ())
		return distv.GetX();

	return Utils::kInfinity;
}


void transform( FastTransformationMatrix const & m, Vector3DFast const & master, Vector3DFast & local )
{
  m.MasterToLocal( master,local );

}

void transform2( FastTransformationMatrix const & m, Vector3DFast const & master, Vector3DFast & local )
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
  Vector3DFast a(10,2,1);
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
 
  std::cerr << "min test:" << std::endl;
  std::cerr << a.Min() << std::endl;

  return 1;
}
