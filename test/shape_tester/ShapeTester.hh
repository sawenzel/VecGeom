//
// Definition of the batch solid test
//

#ifndef ShapeTester_hh
#define ShapeTester_hh

#include <iostream>

#include "VUSolid.hh"
#include "UUtils.hh"

class ShapeTester{

public:
	ShapeTester();
	~ShapeTester();

  void Run(VUSolid *testVolume);//std::ofstream &logger);
 void RunMethod(VUSolid *testVolume, std::string method1 );//std::ofstream &logger);

	inline void SetFilename( const std::string &newFilename ) { filename = newFilename; }
	inline void SetMaxPoints( const int newMaxPoints ) { maxPoints = newMaxPoints; }
	inline void SetRepeat( const int newRepeat ) { repeat = newRepeat; }
	inline void SetMethod( const std::string &newMethod ) { method = newMethod; }
	inline void SetInsidePercent( const double percent ) { insidePercent = percent; }
	inline void SetOutsidePercent( const double percent ) { outsidePercent = percent; }
        inline void SetEdgePercent( const double percent ) { edgePercent = percent; }

	inline void SetOutsideMaxRadiusMultiple( const double percent ) { outsideMaxRadiusMultiple = percent; }
	inline void SetOutsideRandomDirectionPercent( const double percent ) { outsideRandomDirectionPercent = percent; }
	inline void SetDifferenceTolerance( const double tolerance ) { differenceTolerance = tolerance; }
        inline void SetNewSaveValue( const double tolerance ) { minDifference = tolerance; }
        inline void SetSaveAllData( const bool safe ) { ifSaveAllData = safe; }
        inline void SetRunAllTests( const bool safe ) { ifMoreTests = safe; }
	void SetFolder( const std::string &newFolder );
  void SetVerbose(int verbose){ fVerbose = verbose; }
  inline int GetMaxPoints() const { return maxPoints; }
  inline int GetRepeat() const { return repeat; }

private:
	void SetDefaults();

	int SaveVectorToExternalFile(const std::vector<double> &vector, const std::string &filename);
	int SaveVectorToExternalFile(const std::vector<UVector3> &vector, const std::string &filename);
	int SaveLegend(const std::string &filename);
        int SaveDifLegend(const std::string &filename);
	int SaveDoubleResults(const std::string &filename);
        int SaveDifDoubleResults(const std::string &filename);
	int SaveVectorResults(const std::string &filename);
        int SaveDifVectorResults(const std::string &filename);
         int SaveDifVectorResults1(const std::string &filename);

	std::string PrintCoordinates (const UVector3 &vec, const std::string &delimiter, int precision=4);
	std::string PrintCoordinates (const UVector3 &vec, const char *delimiter, int precision=4);
	void PrintCoordinates (std::stringstream &ss, const UVector3 &vec, const std::string &delimiter, int precision=4);
	void PrintCoordinates (std::stringstream &ss, const UVector3 &vec, const char *delimiter, int precision=4);

	template <class T> void VectorDifference(const std::vector<T> &first, const std::vector<T> &second, std::vector<T> &result);
	
	void VectorToDouble(const std::vector<UVector3> &vectorUVector, std::vector<double> &vectorDouble);

  void BoolToDouble(const std::vector<bool> &vectorBool, std::vector<double> &vectorDouble);
	
	int CountDoubleDifferences(const std::vector<double> &differences);
	int CountDoubleDifferences(const std::vector<double> &differences, const std::vector<double> &values1, const std::vector<double> &values2);

//	int CompareVectorDifference(std::string filename);

protected:
	UVector3	GetRandomPoint() const;
	double	GaussianRandom(const double cutoff) const;

	void	ReportError( int *nError, UVector3 &p, 
		UVector3 &v, double distance,
			     std::string comment);//, std::ostream &logger );
	void 	ClearErrors();		
	int 	CountErrors() const;		

protected:

	int maxPoints, repeat;
        int fVerbose;   
  double	insidePercent, outsidePercent,edgePercent, outsideMaxRadiusMultiple, outsideRandomDirectionPercent, differenceTolerance;
	std::string method;


	typedef struct sShapeTesterErrorList {
	  std::string	message;
	  int		nUsed;
		struct sShapeTesterErrorList *next;
	} ShapeTesterErrorList;

	ShapeTesterErrorList *errorList;

private:
	int numCheckPoints;

	int compositeCounter;

	void FlushSS(std::stringstream &ss);
	void Flush(const std::string &s);
       
	VUSolid *volumeUSolids;
        std::stringstream volumeSS;
	std::string volumeString;
   

   
 	std::vector<UVector3> points, directions;
	std::vector<UVector3> resultVectorGeant4;
	std::vector<UVector3> resultVectorRoot;
        std::vector<UVector3> resultVectorUSolids,resultVectorDifference;
        std::vector<double> resultDoubleGeant4, resultDoubleRoot, resultDoubleUSolids, resultDoubleDifference;
        std::vector<bool> resultBoolGeant4, resultBoolUSolids, resultBoolDifference;
       
  int offsetSurface, offsetInside, offsetOutside, offsetEdge;
  int maxPointsInside, maxPointsOutside, maxPointsSurface,maxPointsEdge;
	std::ostream *log, *perftab, *perflabels;
	std::string folder;
	std::string filename;
        //Save only differences
        bool ifSaveAllData;//save alldata, big files 
        bool ifMoreTests;//do all additional tests, 
                         //take more time, but not affect performance measures
        bool ifDifUSolids;//save differences of Geant4 with Usolids or with ROOT
        double minDifference;//save data, when difference is bigger that min
        bool definedNormal, ifException; 
        std::vector<UVector3> difPoints;
        std::vector<UVector3> difDirections;
        std::vector<UVector3> difVectorGeant4,difVectorRoot,difVectorUSolids;
        std::vector<double> difGeant4,difRoot,difUSolids;
        int difPointsInside,difPointsSurface,difPointsOutside;
        int maxErrorBreak;

	UVector3 GetPointOnOrb(double r);
	UVector3 GetRandomDirection();


	void TestConsistencySolids();
        void TestInsidePoint();
        void TestOutsidePoint();
        void TestSurfacePoint();

	void TestNormalSolids();


	void TestSafetyFromInsideSolids();
        void TestSafetyFromOutsideSolids();
        void ShapeSafetyFromInside(int max);
	void ShapeSafetyFromOutside(int max);

	void PropagatedNormal(const UVector3 &point, const UVector3 &direction, double distance, UVector3 &normal);
        void PropagatedNormalU(const UVector3 &point, const UVector3 &direction, double distance, UVector3 &normal);

	void TestDistanceToInSolids();
        void TestAccuracyDistanceToIn(double dist);
        void ShapeDistances();
        void TestFarAwayPoint();

	void TestDistanceToOutSolids();             
        void ShapeNormal();

	void CreatePointsAndDirections();
	void CreatePointsAndDirectionsSurface();
        void CreatePointsAndDirectionsEdge();
	void CreatePointsAndDirectionsInside();
	void CreatePointsAndDirectionsOutside();

	void CompareAndSaveResults(const std::string &method, double resG, double resR, double resU);

	int SaveResultsToFile(const std::string &method);

	void SavePolyhedra(const std::string &method);

	double MeasureTest (void (ShapeTester::*funcPtr)(int), const std::string &method);

	double NormalizeToNanoseconds(double time);

	void TestMethod(void (ShapeTester::*funcPtr)());
	void TestMethodAll();

     inline double RandomRange(double min, double max)
  {
    double rand = min + (max - min) * UUtils::Random();
    return rand;
  }
    inline void GetVectorUSolids(UVector3 &point, const std::vector<UVector3> &apoints, int index)
  {
    const UVector3 &p = apoints[index];
    point.Set(p.x(), p.y(), p.z());
  }
 
  inline void SetVectorUSolids(const UVector3 &point, std::vector<UVector3> &apoints, int index)
  {
    UVector3 &p = apoints[index];
    p.Set(point.x(), point.y(), point.z());
  }
  
  inline double RandomIncrease()
  {
    double tolerance = VUSolid::Tolerance();
    double rand = -1 + 2 * UUtils::Random();
    double dif = tolerance * 0.1 * rand;
    return dif;
  }
  

};

#endif
