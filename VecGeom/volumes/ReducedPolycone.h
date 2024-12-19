/*
 * ReducedPolycone.h
 *
 *  Created on: Sep 20, 2018
 *      Author: rasehgal
 */

#ifndef VOLUMES_REDUCEDPOLYCONE_H_
#define VOLUMES_REDUCEDPOLYCONE_H_

#include "VecGeom/base/Vector2D.h"
#include "VecGeom/base/Vector.h"
#include <iostream>
#include "VecGeom/volumes/CoaxialCones.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct Line2D;);
VECGEOM_DEVICE_DECLARE_CONV(struct, Line2D);
VECGEOM_DEVICE_FORWARD_DECLARE(struct Section;);
VECGEOM_DEVICE_DECLARE_CONV(struct, Section);
VECGEOM_DEVICE_FORWARD_DECLARE(class ReducedPolycone;);
VECGEOM_DEVICE_DECLARE_CONV(class, ReducedPolycone);
VECGEOM_DEVICE_FORWARD_DECLARE(struct IndexStruct;);
VECGEOM_DEVICE_DECLARE_CONV(struct, IndexStruct);

inline namespace VECGEOM_IMPL_NAMESPACE {
// A Container to store index and distance
struct IndexStruct {
  unsigned int index;
  double distance;
  VECCORE_ATT_HOST_DEVICE
  IndexStruct(unsigned int ind, double dist) : index(ind), distance(dist) {}
};

// Just a Container for 2D Line
struct Line2D {
  Vector2D<Precision> p1;
  Vector2D<Precision> p2;

  VECCORE_ATT_HOST_DEVICE
  Line2D() {}

  VECCORE_ATT_HOST_DEVICE
  ~Line2D() {}

  VECCORE_ATT_HOST_DEVICE
  Line2D(Vector2D<Precision> p1v, Vector2D<Precision> p2v)
  {
    p1 = p1v;
    p2 = p2v;
  }
#ifndef VECCORE_CUDA
  void Print()
  {
    std::cerr << p1 << " : ";
    std::cerr << p2 << std::endl;
  }
#endif
  VECCORE_ATT_HOST_DEVICE
  Line2D(Precision rmax, Precision zval)
  {
    p1 = Vector2D<Precision>(0, zval);
    p2 = Vector2D<Precision>(rmax, zval);
  }
  VECCORE_ATT_HOST_DEVICE
  Precision GetHorizontalDistance(double yVal)
  {
    if (p1.x() == p2.x()) {
      return p1.x();
    } else {
      return p1.x() + ((p2.x() - p1.x()) / (p2.y() - p1.y())) * (yVal - p1.y());
    }
  }
};

// Just a container for a Section
struct Section {
  Precision rMin1;
  Precision rMax1;
  Precision rMin2;
  Precision rMax2;
  Precision z1;
  Precision z2;

  VECCORE_ATT_HOST_DEVICE
  Section() {}

  VECCORE_ATT_HOST_DEVICE
  ~Section() {}

  VECCORE_ATT_HOST_DEVICE
  Section(Precision rmin1, Precision rmax1, Precision Z1, Precision rmin2, Precision rmax2, Precision Z2)
      : rMin1(rmin1), rMax1(rmax1), rMin2(rmin2), rMax2(rmax2), z1(Z1), z2(Z2)
  {
  }
#ifndef VECCORE_CUDA
  void Print()
  {

    std::cerr << "Rmin1 : " << rMin1 << " :: Rmax1 : " << rMax1 << " :: Z1 : " << z1 << std::endl
              << "Rmin2 : " << rMin2 << " :: Rmax2 : " << rMax2 << " :: Z2 : " << z2 << std::endl;
  }
#endif
};

// struct ConeParam;

class ReducedPolycone {
  Vector<Vector2D<Precision>> fRZVect;
  Precision fRMax;
  // For SimplePolycone
  Vector<Section> fSectionVect;

  // For GenericPolycone
  // Vector<Section> fCoaxialCones;
  Vector<Vector<Section>> fCoaxialConesSectionVect;

public:
  VECCORE_ATT_HOST_DEVICE
  ReducedPolycone(Vector<Vector2D<Precision>> rzVect);
  VECCORE_ATT_HOST_DEVICE
  ~ReducedPolycone();
  VECCORE_ATT_HOST_DEVICE
  void SetRZ(Vector<Vector2D<Precision>>);
  VECCORE_ATT_HOST_DEVICE
  void SetRMax();
  VECCORE_ATT_HOST_DEVICE
  void ConvertToUniqueVector(Vector<Precision> &vect);
  VECCORE_ATT_HOST_DEVICE
  Vector<Precision> GetUniqueZVector();
  VECCORE_ATT_HOST_DEVICE
  void PrintVector(Vector<Precision> vect);
  VECCORE_ATT_HOST_DEVICE
  Vector<Line2D> GetLineVector();
  VECCORE_ATT_HOST_DEVICE
  void CalcPoIVectorFor2DPolygon(Vector<Vector2D<Precision>> &poiVect, Vector<Precision> z);
  VECCORE_ATT_HOST_DEVICE
  bool ContourCheck(Vector<Precision> z);
  VECCORE_ATT_HOST_DEVICE
  bool Contour(Vector<Precision> z);
  VECCORE_ATT_HOST_DEVICE
  bool ContourGeneric(Vector<Precision> z);
  VECCORE_ATT_HOST_DEVICE
  bool PointExist(Vector2D<Precision> pt);
  VECCORE_ATT_HOST_DEVICE
  void CreateNewContour();
  VECCORE_ATT_HOST_DEVICE
  void ProcessContour(Vector<Precision> z);
  VECCORE_ATT_HOST_DEVICE
  void ProcessGenericContour(Vector<Precision> z);
  VECCORE_ATT_HOST_DEVICE
  Vector<Vector<Precision>> GetRandZVectorAtDiffZ(Vector<Vector2D<Precision>> poiVect, Vector<Precision> &dz);
  VECCORE_ATT_HOST_DEVICE
  Section CreateSectionFromTwoLines(Line2D l1, Line2D l2);
  VECCORE_ATT_HOST_DEVICE
  void Swap(Precision &a, Precision &b);
  VECCORE_ATT_HOST_DEVICE
  bool Check();
  VECCORE_ATT_HOST_DEVICE
  bool CheckGeneric();
  VECCORE_ATT_HOST_DEVICE
  bool GetPolyconeParameters(Vector<Precision> &rmin, Vector<Precision> &rmax, Vector<Precision> &z);

  VECCORE_ATT_HOST_DEVICE
  bool GetLineIntersection(Precision p0_x, Precision p0_y, Precision p1_x, Precision p1_y, Precision p2_x,
                           Precision p2_y, Precision p3_x, Precision p3_y, Precision *i_x, Precision *i_y);

  VECCORE_ATT_HOST_DEVICE
  bool GetLineIntersection(Line2D l1, Line2D l2, Vector2D<Precision> &poi);

  // Returns poi at line l1 where y=yVal, and true if poi is between l1 points
  // or false otherwise
  VECCORE_ATT_HOST_DEVICE
  bool GetLineIntersection(Line2D l1, Precision zVal, Vector2D<Precision> &poi)
  {
    // Line l1: (r1,z1) -> (r2,z2)
    Precision r1 = l1.p1.x();
    Precision z1 = l1.p1.y();
    Precision r2 = l1.p2.x();
    Precision z2 = l1.p2.y();

    if (z1 == z2) {
        // vertical line or degenerate point - no solution
        poi.x() = poi.y() = 0.;
        return false;
    }

    // solution: poi(r, zVal)
    poi.y() = zVal;
    poi.x() = r1 + (r2 - r1) * (zVal - z1) / (z2 - z1);

    // Validation: is poi contained between (r1,z1)<->(r2,z2)?
    bool ok = (poi.x() >= std::min(r1, r2)) && (poi.x() <= std::max(r1, r2))
           && (poi.y() >= std::min(z1, z2)) && (poi.y() <= std::max(z1, z2));

    return ok;
  }

  VECCORE_ATT_HOST_DEVICE
  Vector<Line2D> FindLinesInASection(unsigned int secIndex);

  VECCORE_ATT_HOST_DEVICE
  Vector<Line2D> GetVectorOfSortedLinesByHorizontalDistance(unsigned int secIndex);

  /*
    VECCORE_ATT_HOST_DEVICE
    void GetPolyconeParameters(Vector<Vector<ConeParam>> &sectionsParamVector, Vector<Precision> &zS,
                               Vector3D<Precision> &aMin, Vector3D<Precision> &aMax);
  */

  VECCORE_ATT_HOST_DEVICE
  void GetPolyconeParameters(Vector<Vector<Precision>> &vectOfRmin1Vect, Vector<Vector<Precision>> &vectOfRmax1Vect,
                             Vector<Vector<Precision>> &vectOfRmin2Vect, Vector<Vector<Precision>> &vectOfRmax2Vect,
                             Vector<Precision> &zS, Vector3D<Precision> &aMin, Vector3D<Precision> &aMax);
};

} // namespace VECGEOM_IMPL_NAMESPACE
} /* namespace vecgeom */

#endif /* VOLUMES_REDUCEDPOLYCONE_H_ */
