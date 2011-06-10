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
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
//
// $Id:$
// GEANT4 tag $Name:$
//
// 
// G4USolid implementation
//
// --------------------------------------------------------------------

G4USolid::G4USolid(VUSolid* s)
  : fShape(s)
{
}

G4USolid::~G4USolid()
{
}

G4bool G4USolid::operator==( const G4USolid& s) const
{
  return (this==&s) ? true : false;
}

G4bool G4USolid::CalculateExtent(const EAxis pAxis,
				  const G4VoxelLimits& pVoxelLimit,
				  const G4AffineTransform& pTransform,
				  G4double& pMin, G4double& pMax) const;
{
  if (!pTransform.IsRotated())
  {
    fShape->Extent(pAxis,pMin,pMax);
   
    if (pVoxelLimit.IsXLimited())
    {
      switch (pAxis)
      {
        case kXAxis:
          if ((pMin > pVoxelLimit.GetMaxXExtent()+kCarTolerance) || 
              (pMax < pVoxelLimit.GetMinXExtent()-kCarTolerance))
          {
            return false;
          }
          else
          {
            pMin = std::max(pMin, pVoxelLimit.GetMinXExtent());
            pMax = std::min(pMax, pVoxelLimit.GetMaxXExtent());
          }
          break; 
        case kYAxis:
          if ((pMin > pVoxelLimit.GetMaxYExtent()+kCarTolerance) ||
              (pMax < pVoxelLimit.GetMinYExtent()-kCarTolerance))
          {
            return false;
          }
          else
          {
            pMin = std::max(pMin, pVoxelLimit.GetMinYExtent());
            pMax = std::min(pMax, pVoxelLimit.GetMaxYExtent());
          }
          break;
        case kZAxis:
          if ((pMin > pVoxelLimit.GetMaxZExtent()+kCarTolerance) ||
              (pMax < pVoxelLimit.GetMinZExtent()-kCarTolerance))
          {
            return false;
          }
          else
          {
            pMin = std::max(pMin, pVoxelLimit.GetMinZExtent());
            pMax = std::min(pMax, pVoxelLimit.GetMaxZExtent());
          }
          break;
        default:
          break;
      }  
      pMin -= kCarTolerance ;
      pMax += kCarTolerance ;
    }
    return true;
  }   
  else  // General rotated case - create and clip mesh to boundaries
  {
    // Rotate BoundingBox and Calculate Extent as for BREPS

    G4bool existsAfterClip=false;
    G4ThreeVectorList *vertices;

    pMin=+kInfinity;
    pMax=-kInfinity;

    // Calculate rotated vertex coordinates
    //
    vertices=CreateRotatedVertices(pTransform);
    ClipCrossSection(vertices,0,pVoxelLimit,pAxis,pMin,pMax);
    ClipCrossSection(vertices,4,pVoxelLimit,pAxis,pMin,pMax);
    ClipBetweenSections(vertices,0,pVoxelLimit,pAxis,pMin,pMax);

    if ( (pMin!=kInfinity) || (pMax!=-kInfinity) )
    {
      existsAfterClip=true;
    
      // Add 2*tolerance to avoid precision troubles
      //
      pMin-=kCarTolerance;
      pMax+=kCarTolerance;
    }
    else
    {
      // Check for case where completely enveloping clipping volume.
      // If point inside then we are confident that the solid completely
      // envelopes the clipping volume. Hence set min/max extents according
      // to clipping volume extents along the specified axis.
      //
      G4ThreeVector clipCentre(
                (pVoxelLimit.GetMinXExtent()+pVoxelLimit.GetMaxXExtent())*0.5,
                (pVoxelLimit.GetMinYExtent()+pVoxelLimit.GetMaxYExtent())*0.5,
                (pVoxelLimit.GetMinZExtent()+pVoxelLimit.GetMaxZExtent())*0.5);

      if (Inside(pTransform.Inverse().TransformPoint(clipCentre))!=kOutside)
      {
        existsAfterClip=true;
        pMin=pVoxelLimit.GetMinExtent(pAxis);
        pMax=pVoxelLimit.GetMaxExtent(pAxis);
      }
    }
    delete vertices;
    return existsAfterClip;
  }                           // end rotation
}

EInside G4USolid::Inside(const G4ThreeVector& p) const
{
  UVector3 p; p.x=pt.x(); p.y=pt.y(); p.z=pt.z();
  return fShape->Inside(p);
}

G4ThreeVector G4USolid::SurfaceNormal(const G4ThreeVector& pt) const
{
  UVector3 p; p.x=pt.x(); p.y=pt.y(); p.z=pt.z();
  UVector3 n;
  fShape->Normal(p, n);
  return G4ThreeVector(n.x, n.y, n.z);
}

G4double G4USolid::DistanceToIn(const G4ThreeVector& pt,
                                 const G4ThreeVector& d)
{
  UVector3 p; p.x=pt.x(); p.y=pt.y(); p.z=pt.z();
  UVector3 v; v.x=d.x(); v.y=d.y(); v.z=d.z();
  return fShape->DistanceToIn(p,v);
}

G4double G4USolid::DistanceToIn(const G4ThreeVector& pt) const
{
  UVector3 p; p.x=pt.x(); p.y=pt.y(); p.z=pt.z();
  return fShape->SafetyOut(p);
}

G4double G4USolid::DistanceToOut(const G4ThreeVector& pt,
				  const G4ThreeVector& d,
				  const G4bool calcNorm=false,
				  G4bool *validNorm,
				  G4ThreeVector *norm) const
{
  UVector3 p; p.x=pt.x(); p.y=pt.y(); p.z=pt.z();
  UVector3 v; v.x=d.x(); v.y=d.y(); v.z=d.z();
  UVector3 n;
  G4double dist = fShape->DistanceToOut(p, v, n, *validNorm);
  norm->SetX(n.x); norm->SetY(n.y); norm->SetZ(n.z);
  return dist;
}

G4double G4USolid::DistanceToOut(const G4ThreeVector& pt) const
{
  UVector3 p; p.x=pt.x(); p.y=pt.y(); p.z=pt.z();
  return fShape->SafetyIn(p);
}

G4double G4USolid::GetCubicVolume()
{
  return fShape->Capacity();
}

G4double G4USolid::GetSurfaceArea()
{
  return fShape->SurfaceArea();
}

G4GeometryType G4USolid::GetEntityType() const
{
  return fShape->GetEntityTyoe();
}

G4ThreeVector G4USolid::GetPointOnSurface() const
{
  UVector3 p = fShape->SamplePointOnSurface();
  return G4ThreeVector(p.x, p.y, p.z);
}

G4VSolid* G4USolid::Clone() const
{
  return new G4USolid(fShape->Clone());
}

std::ostream& G4USolid::StreamInfo(std::ostream& os) const
{
  os << "-----------------------------------------------------------\n"
     << "    *** Dump for solid - " << fShape->GetName() << " ***\n"
     << "    ===================================================\n"
     << " Solid type: " << fShape->GetEntityType() << "\n"
     << "-----------------------------------------------------------\n";

  return fShape->StreamInfo(os);
}

void G4USolid::DescribeYourselfTo (G4VGraphicsScene& scene) const
{
  scene.AddSolid (*this);
}

G4USolid::G4USolid(__void__& a)
  : G4VSolid(a), fShape(0)
{
}

G4USolid::G4USolid(const G4USolid& rhs)
  : G4VSolid(rhs), fShape(rhs.fShape)
{
}

G4USolid& G4USolid::operator=(const G4USolid& rhs)
{
  // Check assignment to self
  //
  if (this == &rhs)  { return *this; }

  // Copy base class data
  //
  G4VSolid::operator=(rhs);

  // Copy data
  //
  fShape = rhs.fShape;

  return *this;
}

G4ThreeVectorList*
G4USolid::CreateRotatedVertices(const G4AffineTransform& pTransform) const
{
  G4double xMin,xMax,yMin,yMax,zMin,zMax;
  G4double xoffset,yoffset,zoffset;
  fShape->Extent(kXAxis,xMin,xMax);  
  fShape->Extent(kYAxis,yMin,yMax); 
  fShape->Extent(kZAxis,zMin,zMax); 

  // Or new method Extend3D
  xoffset = pTransform.NetTranslation().x() ;
  yoffset = pTransform.NetTranslation().y() ;
  zoffset = pTransform.NetTranslation().z() ;

  xMin-=xoffset;xMax-=xoffset;
  yMin-=yoffset;yMax-=yoffset;
  zMin-=zoffset;zMax-=zoffset;

  G4ThreeVectorList *vertices;
  vertices=new G4ThreeVectorList();
    
  if (vertices)
  {
    vertices->reserve(8);
    G4ThreeVector vertex0(xMin,yMin,zMin);
    G4ThreeVector vertex1(xMax,yMin,zMin);
    G4ThreeVector vertex2(xMax,yMax,zMin);
    G4ThreeVector vertex3(xMin,yMax,zMin);
    G4ThreeVector vertex4(xMin,yMin,zMax);
    G4ThreeVector vertex5(xMax,yMin,zMax);
    G4ThreeVector vertex6(xMax,yMax,zMax));
    G4ThreeVector vertex7(xMin,yMax,zMax));

    vertices->push_back(pTransform.TransformPoint(vertex0));
    vertices->push_back(pTransform.TransformPoint(vertex1));
    vertices->push_back(pTransform.TransformPoint(vertex2));
    vertices->push_back(pTransform.TransformPoint(vertex3));
    vertices->push_back(pTransform.TransformPoint(vertex4));
    vertices->push_back(pTransform.TransformPoint(vertex5));
    vertices->push_back(pTransform.TransformPoint(vertex6));
    vertices->push_back(pTransform.TransformPoint(vertex7));
  }
  else
  {
    G4Exception("G4VUSolid::CreateRotatedVertices()", "FatalError",
                FatalException, "Out of memory - Cannot allocate vertices!");
  }
  return vertices;
}
