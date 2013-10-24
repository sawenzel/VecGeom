//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the  intellectual property  of the *
// * Vanderbilt University Free Electron Laser Center                 *
// * Vanderbilt University, Nashville, TN, USA                        *
// * Development supported by:                                        *
// * United States MFEL program  under grant FA9550-04-1-0045         *
// * and NASA under contract number NNG04CT05P.                       *
// * Written by Marcus H. Mendenhall and Robert A. Weller.            *
// *                                                                  *
// * Contributed to the Geant4 Core, January, 2005.                   *
// *                                                                  *
// ********************************************************************
//
//
// $Id: UTet.hh 66356 2012-12-18 09:02:32Z gcosmo $
//
//
// --------------------------------------------------------------------
// USolids class header file
//
// UTet
//
// Class description:
//
//   A UTet is a tetrahedrasolid.
//

// History:
// -------
// 03.09.2004 - M.H.Mendenhall & R.A.Weller (Vanderbilt University, USA)
// 10.02.2005 - D.Anninos (CERN) - Added GetPointOnSurface() method.
// 12.11.2006 - M.H.Mendenhall - Added GetSurfaceArea() concrete implementation.
// 20.09.2010 - G.Cosmo (CERN) - Added copy-ctor and operator=().
// 08.03.2012 - T.Nikitina(Geant4 col)- Changed G4Tet to UTet
// --------------------------------------------------------------------
#ifndef UTet_hh
#define UTet_hh

#include "VUSolid.hh"
#include "UUtils.hh"

class UTet : public VUSolid
{

  public:  // with description

  UTet(const std::string & name, 
                UVector3 &anchor,
                UVector3 &p2,
                UVector3 &p3,
                UVector3 &p4, 
                bool *degeneracyFlag=0);

    virtual ~UTet();

      
    // Methods for solid

    EnumInside Inside(const UVector3  &p) const;

    bool Normal( const UVector3& aPoint, UVector3 &aNormal ) const; 

    	double  SafetyFromInside ( const UVector3 &aPoint, 
		bool aAccurate=false) const;
	double  SafetyFromOutside( const UVector3 &aPoint, 
		bool aAccurate=false) const;
	double  DistanceToIn     ( const UVector3 &aPoint, 
		const UVector3 &aDirection,
		// UVector3       &aNormalVector,
		double aPstep = UUtils::kInfinity) const;                               

	double DistanceToOut     ( const UVector3 &aPoint,
		const UVector3 &aDirection,
		UVector3       &aNormalVector, 
		bool           &aConvex,
		double aPstep = UUtils::kInfinity) const;

    void Extent (UVector3 &aMin, UVector3 &aMax) const; 
    double Capacity(){return fCubicVolume;}
    double SurfaceArea(){return fSurfaceArea;}
	UGeometryType GetEntityType() const { return "Tet";}
	void ComputeBBox(UBBox * /*aBox*/, bool /*aStore = false*/) {}

	//G4Visualisation
	void GetParametersList(int /*aNumber*/, double * /*aArray*/) const{} 
	UPolyhedron* CreatePolyhedron () const;
     UPolyhedron* GetPolyhedron() const{return CreatePolyhedron();}

        VUSolid* Clone() const
	{
	  std::vector<UVector3> v=GetVertices();
	  return new UTet(GetName(),v[0],v[1],v[2],v[3]);
	}

   

    UVector3 GetPointOnSurface() const;
    std::ostream& StreamInfo(std::ostream& os) const;
   
  public:   // without description

  //UTet(__void__&);
      // Fake default constructor for usage restricted to direct object
      // persistency for clients requiring preallocation of memory for
      // persistifiable objects.

    UTet(const UTet& rhs);
    UTet& operator=(const UTet& rhs); 
      // Copy constructor and assignment operator.

    const char* CVSHeaderVers()
      { return "$Id: G4Tet.hh 66356 2012-12-18 09:02:32Z gcosmo $"; }
    const char* CVSFileVers()
      { return CVSVers; }
    void PrintWarnings(bool flag)
      { warningFlag=flag; }
    static bool CheckDegeneracy(UVector3 &anchor,
                                  UVector3 &p2,
                                  UVector3 &p3,
                                  UVector3 &p4);
    std::vector<UVector3> GetVertices() const;
      // Return the four vertices of the shape.

  protected:  // with description

  /*  UVectorList*
    CreateRotatedVertices(const G4AffineTransform& pTransform) const;
      // Create the List of transformed vertices in the format required
      // for G4VSolid:: ClipCrossSection and ClipBetweenSections.
      */
  private:

    double fCubicVolume, fSurfaceArea;

    mutable UPolyhedron* fpPolyhedron;

    UVector3 GetPointOnFace(UVector3 p1, UVector3 p2, 
                                 UVector3 p3, double& area) const;
    static const char CVSVers[];

  private:

    UVector3 fAnchor, fP2, fP3, fP4, fMiddle;
    UVector3 fNormal123, fNormal142, fNormal134, fNormal234;

    bool warningFlag;

    double fCdotN123, fCdotN142, fCdotN134, fCdotN234;
    double fXMin, fXMax, fYMin, fYMax, fZMin, fZMax;
    double fDx, fDy, fDz, fTol, fMaxSize;
};

#endif
