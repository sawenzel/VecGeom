#include "TGeoUShape.h"
#include "VUSolid.hh"
////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TGeoUShape - bridge class for using a USolid as TGeoShape.             //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

ClassImp(TGeoUShape)

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

