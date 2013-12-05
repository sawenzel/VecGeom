#clang++ -O2 -std=c++11 -I /home/swenzel/local/root_geantv/include -I ../ -I /home/swenzel/local/vc/include/ ../PhysicalBox.cpp ../GeoManager.cpp ../TransformationMatrix.cpp TransformOfVectorsTest.cpp -o foo.x -L /home/swenzel/local/vc/lib/ -lVc  -L /home/swenzel/local/root_geantv/lib/ -lGeom
#g++ -fpermissive -mavx -O2 -ffast-math -fabi-version=6 -std=c++11 -I /home/swenzel/local/root_geantv/include -I ../ -I /home/swenzel/local/vc/include/ ../PhysicalBox.cpp ../GeoManager.cpp ../PhysicalVolume.cpp ../TransformationMatrix.cpp TestVectorizedPlacedBox.cpp -o foo.x -L /home/swenzel/local/vc/lib/ -lVc  -L /home/swenzel/local/root_geantv/lib/ -lGeom -L${TBBROOT}/lib -ltbb
g++ -g  -O2 -finline-limit=10000000 -mavx -fpermissive -ffast-math -fabi-version=6 -std=c++11 -I ../ -I /home/swenzel/local/vc/include/ -I $ROOTSYS/include ../PhysicalBox.cpp ../GeoManager.cpp ../PhysicalVolume.cpp ../TransformationMatrix.cpp TestVectorizedPlacedBox.cpp -o foo.x -L /home/swenzel/local/vc/lib/ -lVc -L${TBBROOT}/lib -ltbb -L $ROOTSYS/lib/ -lGeantVGeom
#g++ -finline-limit=1000 -mavx -O3 -fpermissive -ffast-math -fabi-version=6 -std=c++11 -I ../ -I /home/swenzel/local/vc/include/ -I $ROOTSYS/include ../PhysicalBox.cpp ../GeoManager.cpp ../PhysicalVolume.cpp ../TransformationMatrix.cpp TestVectorizedPlacedBox.cpp -o foo.x -L /home/swenzel/local/vc/lib/ -lVc -L${TBBROOT}/lib -ltbb -L $ROOTSYS/lib/ -lGeantVGeom
