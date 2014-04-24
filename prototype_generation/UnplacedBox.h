#ifndef VECGEOM_UNPLACEDBOX_H_
#define VECGEOM_UNPLACEDBOX_H_

#include "Global.h"
#include "UnplacedVolume.h"

class UnplacedBox : public UnplacedVolume {

private:

  double dimensions_[3];

public:

  UnplacedBox(const double x, const double y, const double z)
      : dimensions_{x, y, z} {}

  ~UnplacedBox() {}

  double const* dimensions() const { return dimensions_; }

  double x() const { return dimensions_[0]; }

  double y() const { return dimensions_[1]; }

  double z() const { return dimensions_[2]; }

};

#endif // VECGEOM_UNPLACEDBOX_H_