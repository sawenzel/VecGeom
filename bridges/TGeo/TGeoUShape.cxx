#include "TGeoUShape.h"

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
   UBBox * box = fUSolid->GetBBox();
   fDX = box->GetX();
   fDY = box->GetY();
   fDZ = box->GetZ();
   fOrigin[0] = box->GetOx();
   fOrigin[1] = box->GetOy();
   fOrigin[2] = box->GetOz();
}   

   
//_____________________________________________________________________________
Bool_t TGeoBBox::Contains(Double_t *point) const
{
// Test if point is inside this shape.
   VUSolid::EInside inside = fUSolid->Inside(point);
   if (inside == VUSolid::kOutside) return kFALSE;
   return kTRUE;
}

