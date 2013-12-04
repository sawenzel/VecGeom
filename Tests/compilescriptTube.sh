g++ -DSHOWDIFFERENCES -O2 -finline-limit=10000000 -mavx -fpermissive -ffast-math -fabi-version=6 -std=c++11 -I ../ -I /home/swenzel/local/vc/include/ -I $ROOTSYS/include ../PhysicalBox.cpp ../GeoManager.cpp ../PhysicalVolume.cpp ../TransformationMatrix.cpp TestVectorizedPlacedTube.cpp -o foo.x -L /home/swenzel/local/vc/lib/ -lVc -L${TBBROOT}/lib -ltbb -L $ROOTSYS/lib/ -lGeantVGeom


