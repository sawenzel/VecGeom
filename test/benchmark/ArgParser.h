#include <algorithm>

double getDoubleOpt(char** begin, char** end, const std::string& option) {
    char** itr = std::find(begin, end, option);
    if(itr != end && ++itr != end) {
        double ret;
        sscanf(*itr, "%lf", &ret);
        return ret;
    }
    std::cout << "ERROR: Could not read argument " << option << std::endl;
    exit(1);
}

double getIntOpt(char** begin, char** end, const std::string& option) {
    char** itr = std::find(begin, end, option);
    if(itr != end && ++itr != end) {
        int ret;
        sscanf(*itr, "%d", &ret);
        return ret;
    }
    std::cout << "ERROR: Could not read argument " << option << std::endl;
    exit(1);
}

#define OPTION_INT(name) int name = getIntOpt(argv, argc+argv, "-" #name)
#define OPTION_DOUBLE(name) double name = getDoubleOpt(argv, argc+argv, "-" #name)

