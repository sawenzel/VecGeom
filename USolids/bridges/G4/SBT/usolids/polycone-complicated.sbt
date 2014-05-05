
/performance/maxPoints 10000
/performance/repeat 100

# Build a much more complicated polycone
/solid/G4Polycone 0 360 17 (0.0,0.2,0.3,0.32,0.32,0.4,0.4,0.5,0.5,0.8,0.8,0.9,0.9,0.8,0.8,0.3,0.0) (-0.5,-0.5,-1.1,-1.1,-0.4,-0.4,-1.0,-1.0,-0.4,-1.0,0.0,0.0,0.2,0.2,1.0,0.0,1.0) 
/performance/errorFileName log/polycone-complicated-p10k/polyconep.a1.log
/control/execute usolids/performance.sbt

# Build a much more complicated polycone, now with a slice
/solid/G4Polycone 0 90 17 (0.0,0.2,0.3,0.32,0.32,0.4,0.4,0.5,0.5,0.8,0.8,0.9,0.9,0.8,0.8,0.3,0.0) (-0.5,-0.5,-1.1,-1.1,-0.4,-0.4,-1.0,-1.0,-0.4,-1.0,0.0,0.0,0.2,0.2,1.0,0.0,1.0)
/performance/errorFileName log/polycone-complicated-slice-p10k/polyconep.a1.log
/control/execute usolids/performance.sbt

# Build a much more complicated polycone, now with a thin slice
/solid/G4Polycone -1 2 17 (0.0,0.2,0.3,0.32,0.32,0.4,0.4,0.5,0.5,0.8,0.8,0.9,0.9,0.8,0.8,0.3,0.0) (-0.5,-0.5,-1.1,-1.1,-0.4,-0.4,-1.0,-1.0,-0.4,-1.0,0.0,0.0,0.2,0.2,1.0,0.0,1.0) 
/performance/errorFileName log/polycone-complicated-thin-slice-p10k/polyconep.a1.log
/control/execute usolids/performance.sbt
