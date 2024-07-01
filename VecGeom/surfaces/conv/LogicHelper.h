#ifndef VECGEOM_SURFACE_LOGICHELPER_H_
#define VECGEOM_SURFACE_LOGICHELPER_H_

#include <VecGeom/surfaces/Model.h>
#include <VecGeom/volumes/BooleanVolume.h>
#include <VecGeom/management/Logger.h>

namespace vgbrep {

using LogicExpressionCPU = std::vector<logic_int>;

namespace logichelper {

bool string_to_logic(std::string s, LogicExpressionCPU &logic);
char logic_to_char(logic_int item);
void print_item(logic_int item);
void print_logic(LogicExpressionCPU const &logic, int istart = 0, int iend = -1, bool jumps = true);
void print_logic(LogicExpression const &logic, int istart = 0, int iend = -1, bool jumps = true);
void remove_range(LogicExpressionCPU &logic, size_t istart, size_t iend);
void remove_parentheses(LogicExpressionCPU &logic, size_t start, size_t end);
void insert_jumps(LogicExpressionCPU &logic);
bool is_negated(int isurf, LogicExpressionCPU &logic);
size_t find_matching_parenthesis(LogicExpressionCPU &logic, size_t start);

/// @brief Logic expression decomposed in a vector of operands (also logic expressions) connected by and/or operators
struct LogicExpressionConstruct {
  bool fNegated{false};                            /// is negated logic
  bool fHasScope{false};                           /// has a scope
  int fComplexity{0};                              /// expression complexity
  LogicExpressionCPU fLogic;                       /// Logic expression
  LogicExpressionCPU fOperators;                   /// Vector of operators
  std::vector<LogicExpressionConstruct> fOperands; /// Vector of operands

  LogicExpressionConstruct(LogicExpressionCPU const &logic);
  LogicExpressionConstruct &operator&=(LogicExpressionConstruct const &other);
  LogicExpressionConstruct &operator|=(LogicExpressionConstruct const &other);

  void Concatenate(LogicExpressionConstruct const &other, logic_int op_concat);
  void PushNegation(bool neg = false);
  bool RemoveScope(logic_int logic_op);
  void RemoveChildrenScopes(bool top = false);
  void Print() const;
  void GivePrecedenceToAnd();
  void SwapByComplexity();
  void GetLogicExpression(LogicExpressionCPU &logic) const;
  int GetNumOperands() const;
  void Simplify(LogicExpressionCPU &logic);
};

int max_operand(LogicExpressionCPU const &logic);
void substitute_logic(LogicExpressionCPU const &logic, LogicExpressionCPU &substutute_logic, const bool *values);
bool evaluate_logic(LogicExpressionCPU const &logic, const bool *values, int &num_eval);

} // namespace logichelper
} // namespace vgbrep
#endif
