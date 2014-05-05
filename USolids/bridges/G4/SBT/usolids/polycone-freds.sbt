
/performance/maxPoints 100000
/performance/repeat 10

# Build a polycone similiar to fred's PCON4 (and testG4Polycone.cc)
/solid/G4Polycone2 -10 355 8 (-0.2,-0.1,-0.1,0,0.1,0.2,0.3,0.4) (0.3,0.3,0,0,0,0,0.4,0.4) (0.7,0.7,0.7,0.4,0.4,0.8,0.8,0.8)
/performance/errorFileName log/polycone-pcon4-p10k/polyconep.a1.log
/control/execute usolids/performance.sbt

# Build a polycone similiar to fred's PCON2
/solid/G4Polycone 10 250 10 (0.6,0.6,1.0,1.0,1.1,0.9,0.0,0.0,0.4,0.5) (-1.0,0.0,0.0,0.8,1.0,1.0,0.8,0.0,0.0,-1.0)
/performance/errorFileName log/polycone-pcon2-p10k/polyconep.a1.log
/control/execute usolids/performance.sbt

# Build a polycone similiar to fred's PCON3
/solid/G4Polycone -10 355 16 (0.7,0.7,0.8,0.9,1.0,1.0,0.5,0.5,0.0,0.4,0.4,0.4,0.5,0.5,0.6,0.6) (-1.0,-0.5,-0.5,-1.0,-1.0,0.7,0.7,1.0,1.0,0.7,0.7,-1.0,-1.0,-0.5,-0.5,-1.0)
/performance/errorFileName log/polycone-pcon3-p10k/polyconep.a1.log
/control/execute usolids/performance.sbt

