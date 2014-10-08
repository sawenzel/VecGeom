#include "volumes/Planes.h"

namespace VECGEOM_NAMESPACE {

Planes::Planes(int size) : fNormals(size), fDistances(size) {}

Planes::~Planes() {}

Planes& Planes::operator=(Planes const &other) {
  fNormals = other.fNormals;
  fDistances = other.fDistances;
  return *this;
}

void Planes::Set(
    int index,
    Vector3D<Precision> const &normal,
    Vector3D<Precision> const &x0) {
  Precision inverseLength = 1. / normal.Mag();
  fNormals.set(index, inverseLength*normal);
  fDistances[index] = inverseLength * -normal.Dot(x0);;
}

void Planes::Set(
    int index,
    Vector3D<Precision> const &normal,
    Precision distance) {
  fNormals.set(index, normal);
  fDistances[index] = distance;
}

std::ostream& operator<<(std::ostream &os, Planes const &planes) {
  for (int i = 0, iMax = planes.size(); i < iMax; ++i) {
    os << "{" << planes.GetNormal(i) << ", " << planes.GetDistance(i) << "}\n";
  }
  return os;
}

}