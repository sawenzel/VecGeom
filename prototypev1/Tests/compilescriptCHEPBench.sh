#g++ -DVC_IMPL=Scalar -g -O0 -DAVOIDSPECIALIZATION -DSHOWDIFFERENCES  -ffast-math  -finline-limit=10000000 -mavx -fpermissive -fabi-version=6 -std=c++11 -I ../ -I /home/swenzel/local/vc/include/ -I $ROOTSYS/include ../PhysicalBox.cpp ../GeoManager.cpp ../PhysicalVolume.cpp ../TransformationMatrix.cpp TestVectorizedPlacedTube.cpp -o foo.x -L /home/swenzel/local/vc/lib/ -lVc -L${TBBROOT}/lib -ltbb -L $ROOTSYS/lib/ -lGeantVGeom
g++ -O2 -DSHOWDIFFERENCES -ffast-math  -finline-limit=10000000 -mavx -fpermissive -fabi-version=6 -std=c++11 -I ../ -I /home/swenzel/local/vc/include/ -I $ROOTSYS/include -I ${USOLIDSROOT}/include ../PhysicalBox.cpp ../GeoManager_MakeBox.cpp ../GeoManager_MakeTube.cpp ../GeoManager_MakeCone.cpp ../PhysicalVolume.cpp ../TransformationMatrix.cpp ../SimpleVecNavigator.cpp CHEP13Benchmark.cpp -o bench.x -L /home/swenzel/local/vc/lib/ -lVc -L${TBBROOT}/lib -ltbb -L $ROOTSYS/lib/ -lGeantVGeom -L ${USOLIDSROOT}/build -lusolids



