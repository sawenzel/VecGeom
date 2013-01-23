
#include "VUSolid.hh"

////////////////////////////////////////////////////////////////////////////////
//  "Universal" Solid Interface
//  Authors: J. Apostolakis, G. Cosmo, M. Gayer, A. Gheata, A. Munnich, T. Nikitina (CERN)
//
//  Created: 25 May 2011
//
////////////////////////////////////////////////////////////////////////////////

double VUSolid::fgTolerance = 1.0E-9;  // cartesian tolerance; to be changed (for U was 1e-8, but we keep Geant4)
double VUSolid::frTolerance = 1.0E-9;  // radial tolerance; to be changed

double VUSolid::faTolerance = 1.0E-9;  // angular tolerance; to be changed

//______________________________________________________________________________
VUSolid::VUSolid() : fName(), fBBox(0)
{

}

//______________________________________________________________________________
VUSolid::VUSolid(const std::string &name) :
fName(name),
	fBBox(0)
{
	// Named constructor
	SetName(name);
}
 
//______________________________________________________________________________
VUSolid::~VUSolid()
{

}

/*
int UIntersectingCone::LineHitsCone2( const UVector3 &p,
	const UVector3 &v,
	double &s1, double &s2 )
{
	double x0 = p.x, y0 = p.y, z0 = p.z;
	double tx = v.x, ty = v.y, tz = v.z;

	// Special case which might not be so rare: B = 0 (precisely)
	//
	if (B==0)
	{
		if (std::fabs(tz) < 1/UUtils::Infinity())	{ return 0; }

		s1 = (A-z0)/tz;
		return 1;
	}

	double B2 = B*B;

	double a = tz*tz - B2*(tx*tx + ty*ty);
	double b = 2*( (z0-A)*tz - B2*(x0*tx + y0*ty) );
	double c = UUtils::sqr(z0-A) - B2*( x0*x0 + y0*y0 );

	double radical = b*b - 4*a*c;

	if (radical < -1E-6*std::fabs(b)) { return 0; }	 // No solution

	if (radical < 1E-6*std::fabs(b))
	{
		//
		// The radical is roughly zero: check for special, very rare, cases
		//
		if (std::fabs(a) > 1/UUtils::Infinity())
		{
			if ( std::fabs(x0*ty - y0*tx) < std::fabs(1E-6/B) )
			{
				s1 = -0.5*b/a;
				return 1;
			}
			return 0;
		}
	}
	else
	{
		radical = std::sqrt(radical);
	}

	if (a < -1/UUtils::Infinity())
	{
		double sa, sb, q = -0.5*( b + (b < 0 ? -radical : +radical) );
		sa = q/a;
		sb = c/q;
		if (sa < sb) { s1 = sa; s2 = sb; } else { s1 = sb; s2 = sa; }
		if ((z0 + (s1)*tz - A)/B < 0)	{ return 0; }
		return 2;
	}
	else if (a > 1/UUtils::Infinity())
	{
		double sa, sb, q = -0.5*( b + (b < 0 ? -radical : +radical) );
		sa = q/a;
		sb = c/q;
		s1 = (tz*B > 0)^(sa > sb) ? sb : sa;
		return 1;
	}
	else if (std::fabs(b) < 1/UUtils::Infinity())
	{
		return 0;
	}
	else
	{
		s1 = -c/b;
		if ((z0 + (s1)*tz - A)/B < 0)	{ return 0; }
		return 1;
	}
}
int UIntersectingCone::LineHitsCone2( const UVector3 &p,
	const UVector3 &v,
	double &s1, double &s2 )
{
	double x0 = p.x, y0 = p.y, z0 = p.z;
	double tx = v.x, ty = v.y, tz = v.z;

	// Special case which might not be so rare: B = 0 (precisely)
	//
	if (B==0)
	{
		if (std::fabs(tz) < 1/UUtils::Infinity())	{ return 0; }

		s1 = (A-z0)/tz;
		return 1;
	}

	double B2 = B*B;

	double a = tz*tz - B2*(tx*tx + ty*ty);
	double b = 2*( (z0-A)*tz - B2*(x0*tx + y0*ty) );
	double c = UUtils::sqr(z0-A) - B2*( x0*x0 + y0*y0 );

	double radical = b*b - 4*a*c;

	if (radical < -1E-6*std::fabs(b)) { return 0; }	 // No solution

	if (radical < 1E-6*std::fabs(b))
	{
		//
		// The radical is roughly zero: check for special, very rare, cases
		//
		if (std::fabs(a) > 1/UUtils::Infinity())
		{
			if ( std::fabs(x0*ty - y0*tx) < std::fabs(1E-6/B) )
			{
				s1 = -0.5*b/a;
				return 1;
			}
			return 0;
		}
	}
	else
	{
		radical = std::sqrt(radical);
	}

	if (a < -1/UUtils::Infinity())
	{
		double sa, sb, q = -0.5*( b + (b < 0 ? -radical : +radical) );
		sa = q/a;
		sb = c/q;
		if (sa < sb) { s1 = sa; s2 = sb; } else { s1 = sb; s2 = sa; }
		if ((z0 + (s1)*tz - A)/B < 0)	{ return 0; }
		return 2;
	}
	else if (a > 1/UUtils::Infinity())
	{
		double sa, sb, q = -0.5*( b + (b < 0 ? -radical : +radical) );
		sa = q/a;
		sb = c/q;
		s1 = (tz*B > 0)^(sa > sb) ? sb : sa;
		return 1;
	}
	else if (std::fabs(b) < 1/UUtils::Infinity())
	{
		return 0;
	}
	else
	{
		s1 = -c/b;
		if ((z0 + (s1)*tz - A)/B < 0)	{ return 0; }
		return 1;
	}
}
*/
