#include <algorithm>


double getDoubleOpt(char** begin, char** end, const std::string& option, double defaultval) {
    char** itr = std::find(begin, end, option);
    if(itr != end && ++itr != end) {
        double ret;
        sscanf(*itr, "%lf", &ret);
        return ret;
    }
    std::cout << "INFO: using default " << defaultval << " for option " << option << "\n";
    return defaultval;
}

double getIntOpt(char** begin, char** end, const std::string& option, int defaultval) {
    char** itr = std::find(begin, end, option);
    if(itr != end && ++itr != end) {
        int ret;
        sscanf(*itr, "%d", &ret);
        return ret;
    }
    std::cout << "INFO: using default " << defaultval << " for option " << option << "\n";
    return defaultval;
}

#define OPTION_INT(name,defaultval) int name = getIntOpt(argv, argc+argv, "-" #name, defaultval)
#define OPTION_DOUBLE(name,defaultval) double name = getDoubleOpt(argv, argc+argv, "-" #name, defaultval)

