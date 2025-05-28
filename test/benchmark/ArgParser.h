#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector>
#include <cstdint>

double getDoubleOpt(char **begin, char **end, const std::string &option, double defaultval)
{
  char **itr = std::find(begin, end, option);
  if (itr != end && ++itr != end) {
    double ret;
    sscanf(*itr, "%lf", &ret);
    return ret;
  }
  std::cout << "INFO: using default " << defaultval << " for option " << option << "\n";
  return defaultval;
}

int getIntOpt(char **begin, char **end, const std::string &option, int defaultval)
{
  char **itr = std::find(begin, end, option);
  if (itr != end && ++itr != end) {
    int ret;
    sscanf(*itr, "%d", &ret);
    return ret;
  }
  std::cout << "INFO: using default " << defaultval << " for option " << option << "\n";
  return defaultval;
}

uint64_t getULongOpt(char **begin, char **end, const std::string &option, uint64_t defaultval)
{
  char **itr = std::find(begin, end, option);
  if (itr != end && ++itr != end) {
    uint64_t ret;
    sscanf(*itr, "%ld", &ret);
    return ret;
  }
  std::cout << "INFO: using default " << defaultval << " for option " << option << "\n";
  return defaultval;
}

bool getBoolOpt(char **begin, char **end, const std::string &option, bool defaultval)
{
  char **itr = std::find(begin, end, option);
  if (itr != end && ++itr != end) {
    bool ret =
        !((*(itr[0]) == 'n') || (*(itr[0]) == 'N') || (*(itr[0]) == 'f') || (*(itr[0]) == 'F') || (*(itr[0]) == '0'));
    return ret;
  }
  std::cout << "INFO: using default " << defaultval << " for option " << option << "\n";
  return defaultval;
}

std::string getStringOpt(char **begin, char **end, const std::string &option, const std::string &defaultval)
{
  char **itr = std::find(begin, end, option);
  if (itr != end && ++itr != end) {
    std::string ret(*itr);
    std::cout << " From getStringOpt: ret=<" << ret << ">\n";
    return ret;
  }
  std::cout << "INFO: using default " << defaultval << " for option " << option << "\n";
  return defaultval;
}

bool isNumeric(const char *value)
{
  std::string stringValue(value);
  // Remove trailing comma if it exists
  if (!stringValue.empty() && stringValue.back() == ',') stringValue.pop_back();
  std::istringstream stream(stringValue);
  double temp;

  // Attempt to read a value from the string to a double and check if the entire string was consumed
  return (stream >> temp) && stream.eof();
}

std::vector<double> getVectorOpt(char **begin, char **end, const std::string &option,
                                 const std::vector<double> &defaultval)
{
  char **itr = std::find(begin, end, option);
  if (itr != end && itr + 1 != end) {
    std::vector<double> components;
    ++itr; // Move to the next element after the option

    while (itr != end && *itr != nullptr && isNumeric(*itr)) {
      double value;
      std::istringstream(*itr) >> value;
      components.push_back(value);
      ++itr;
    }

    return components;
  }

  std::cout << "INFO: using default for option " << option << "\n";
  return defaultval;
}

#define OPTION_INT(name, defaultval) auto name = getIntOpt(argv, argc + argv, "-" #name, defaultval)
#define OPTION_ULONG(name, defaultval) auto name = getULongOpt(argv, argc + argv, "-" #name, defaultval)
#define OPTION_DOUBLE(name, defaultval) auto name = getDoubleOpt(argv, argc + argv, "-" #name, defaultval)
#define OPTION_BOOL(name, defaultval) auto name = getBoolOpt(argv, argc + argv, "-" #name, defaultval)
#define OPTION_STRING(name, defaultval) auto name = getStringOpt(argv, argc + argv, "-" #name, defaultval)
#define OPTION_VECTOR(name, defaultval) auto name = getVectorOpt(argv, argc + argv, "-" #name, defaultval)