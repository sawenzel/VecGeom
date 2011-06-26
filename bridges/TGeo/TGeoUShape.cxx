#include "TGeoUShape.h"
#include "VUSolid.hh"
////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TGeoUShape - bridge class for using a USolid as TGeoShape.             //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#include "UBox.hh"
#include "TGeoManager.h"
#include "TGeoMaterial.h"
#include "TGeoMedium.h"
#include "TGeoVolume.h"

ClassImp(TGeoUShape)

//_____________________________________________________________________________
TGeoUShape::TGeoUShape(const char *name, VUSolid *solid)
           :TGeoBBox(name, 0.,0.,0.),
            fUSolid(solid)
{
// Named constructor
   ComputeBBox();
}   

//_____________________________________________________________________________
TGeoUShape::~TGeoUShape()
{
// Destructor
   delete fUSolid;
}   

//_____________________________________________________________________________
void TGeoUShape::ComputeBBox()
{
// Compute bounding box - nothing to do in this case.
   Double_t min[3], max[3];
   fUSolid->Extent(min,max);
   fDX = 0.5*(max[0]-min[0]);
   fDY = 0.5*(max[1]-min[1]);
   fDZ = 0.5*(max[2]-min[2]);
   fOrigin[0] = 0.5*(min[0]+max[0]);
   fOrigin[1] = 0.5*(min[1]+max[1]);
   fOrigin[2] = 0.5*(min[2]+max[2]);
}   

//_____________________________________________________________________________
Double_t TGeoUShape::Capacity() const
{
// Returns analytic capacity of the solid
   return fUSolid->Capacity();
}
   
//_____________________________________________________________________________
void TGeoUShape::ComputeNormal(Double_t *point, Double_t *dir, Double_t *norm)
{
// Normal computation.
   UVector3 normal;
   Bool_t valid = fUSolid->Normal(point, normal);
   if (!valid) return;
   norm[0] = normal.x;
  // Decision: provide or not the Boolean 'validNormal' ar
   norm[1] = normal.y;
   norm[2] = normal.z;
   if (norm[0]*dir[0]+norm[1]*dir[1]+norm[2]*dir[2] < 0) {
      norm[0] = -norm[0];
      norm[1] = -norm[1];
      norm[2] = -norm[2];
   }   
}
   
//_____________________________________________________________________________
Bool_t TGeoUShape::Contains(Double_t *point) const
{
// Test if point is inside this shape.
   VUSolid::EnumInside inside = fUSolid->Inside(point);
   if (inside == VUSolid::eOutside) return kFALSE;
   return kTRUE;
}

//_____________________________________________________________________________
Double_t TGeoUShape::DistFromInside(Double_t *point, Double_t *dir, Int_t /*iact*/, 
                                   Double_t step, Double_t */*safe*/) const
{
   UVector3 normal;
   Bool_t convex;   
   return fUSolid->DistanceToOut(point, dir, normal, convex, step);
}

//_____________________________________________________________________________
Double_t TGeoUShape::DistFromOutside(Double_t *point, Double_t *dir, Int_t /*iact*/, 
                                   Double_t step, Double_t */*safe*/) const
{
   return fUSolid->DistanceToIn(point, dir, step);
}

//_____________________________________________________________________________
Double_t TGeoUShape::Safety(Double_t *point, Bool_t in) const
{
   if (in) return fUSolid->SafetyFromInside(point, kTRUE);
   return fUSolid->SafetyFromOutside(point, kTRUE);
}

//_____________________________________________________________________________
void  TGeoUShape::TestBox()
{
// Make a simple geometry containing a UBox and check it
   TGeoManager *geom = new TGeoManager("UBox", "test of a UBox");
   TGeoMaterial *matVacuum = new TGeoMaterial("Vacuum", 0,0,0);
   TGeoMaterial *matAl = new TGeoMaterial("Al", 26.98,13,2.7);
   TGeoMedium *Vacuum = new TGeoMedium("Vacuum",1, matVacuum);
   TGeoMedium *Al = new TGeoMedium("Root Material",2, matAl);

   TGeoVolume *top = geom->MakeBox("TOP", Vacuum, 200., 200., 200.);
   geom->SetTopVolume(top);
   
   UBox *box = new UBox("UBox", 50,50,50);
   TGeoUShape *shape = new TGeoUShape(box->GetName(), box);
   
   TGeoVolume *vol = new TGeoVolume("UBox", shape, Al);
   top->AddNode(vol,1);
   geom->CloseGeometry();
   shape->CheckShape(1);
}

