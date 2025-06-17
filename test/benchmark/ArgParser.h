#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector>

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

unsigned long getULongOpt(char **begin, char **end, const std::string &option, unsigned long defaultval)
{
  char **itr = std::find(begin, end, option);
  if (itr != end && ++itr != end) {
    unsigned long ret;
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

template <typename Scalar>
std::vector<Scalar> getVectorOpt(char **begin, char **end, const std::string &option,
                                 const std::vector<Scalar> &defaultval)
{
  char **itr = std::find(begin, end, option);
  if (itr != end && itr + 1 != end) {
    std::vector<Scalar> components;
    ++itr; // Move to the next element after the option

    while (itr != end && *itr != nullptr && isNumeric(*itr)) {
      Scalar value;
      std::istringstream(*itr) >> value;
      components.push_back(value);
      ++itr;
    }

    return components;
  }

  std::cout << "INFO: using default for option " << option << "\n";
  return defaultval;
}

template <typename Scalar>
Scalar getScalarOpt(char **begin, char **end, const std::string &option, Scalar defaultval)
{
  char **itr = std::find(begin, end, option);
  if (itr != end && itr + 1 != end) {
    Scalar value;
    ++itr; // Move to the next element after the option
    if (itr != end && *itr != nullptr && isNumeric(*itr)) {
      std::istringstream(*itr) >> value;
      return value;
    }
  }
  std::cout << "INFO: using default for option " << option << "\n";
  return defaultval;
}

#define OPTION_INT(name, defaultval) auto name = getIntOpt(argv, argc + argv, "-" #name, defaultval)
#define OPTION_ULONG(name, defaultval) auto name = getULongOpt(argv, argc + argv, "-" #name, defaultval)
#define OPTION_DOUBLE(name, defaultval) auto name = getDoubleOpt(argv, argc + argv, "-" #name, defaultval)
#define OPTION_BOOL(name, defaultval) auto name = getBoolOpt(argv, argc + argv, "-" #name, defaultval)
#define OPTION_STRING(name, defaultval) auto name = getStringOpt(argv, argc + argv, "-" #name, defaultval)
#define OPTION_VECTOR(Scalar_t, name, defaultval) \
  auto name = getVectorOpt<Scalar_t>(argv, argc + argv, "-" #name, defaultval)
#define OPTION_SCALAR(Scalar_t, name, defaultval) \
  auto name = getScalarOpt<Scalar_t>(argv, argc + argv, "-" #name, defaultval)
