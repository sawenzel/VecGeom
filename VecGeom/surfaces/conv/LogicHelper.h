#ifndef VECGEOM_SURFACE_LOGICHELPER_H_
#define VECGEOM_SURFACE_LOGICHELPER_H_

#include <VecGeom/surfaces/Model.h>
#include <VecGeom/volumes/BooleanVolume.h>
#include <VecGeom/management/Logger.h>

// #define SURF_DEBUG_LOGIC

namespace vgbrep {

using LogicExpressionCPU = std::vector<logic_int>;

namespace logichelper {

bool string_to_logic(std::string s, LogicExpressionCPU &logic)
{
  // remove white spaces
  logic.clear();
  s.erase(std::remove_if(s.begin(), s.end(), [](unsigned char x) { return std::isspace(x); }), s.end());
  std::string buff;
  logic_int next_operand  = 0;
  logic_int next_operator = 0;
  size_t pos              = 0;
  int num_lplus           = 0;
  int num_lminus          = 0;
  while (pos < s.length()) {
    next_operator = 0;
    switch (s[pos]) {
    case '(':
      next_operator = lplus;
      num_lplus++;
      break;
    case ')':
      next_operator = lminus;
      num_lminus++;
      break;
    case '|':
      next_operator = lor;
      break;
    case '&':
      next_operator = land;
      break;
    case '!':
      next_operator = lnot;
      break;
    default:
      buff += s[pos];
    };
    pos++;
    if (next_operator > 0 || pos == s.length()) {
      // check if the buffer is not empty
      if (buff.length() > 0) {
        try {
          next_operand = std::stoi(buff);
        } catch (std::invalid_argument const &ex) {
          std::cout << "std::invalid_argument::what(): " << ex.what() << " \"" << buff << "\"\n ";
          return false;
        }
        logic.push_back(next_operand);
        buff.clear();
      }
      if (next_operator > 0) logic.push_back(next_operator);
    }
  }
  if (num_lplus != num_lminus) {
    std::cout << "logic parantheses mismatch\n";
    return false;
  }
  return true;
}

char logic_to_char(logic_int item)
{
  switch (item) {
  case ltrue:
    return '1';
  case lfalse:
    return '0';
  case lplus:
    return '(';
  case lminus:
    return ')';
  case lor:
    return '|';
  case land:
    return '&';
  case lnot:
    return '!';
  };
  return '0' + item;
}

void print_item(logic_int item)
{
  if (LogicExpression::is_operator_token(item)) {
    switch (item) {
    case ltrue:
      printf("ltrue ");
      break;
    case lfalse:
      printf("lfalse ");
      break;
    case lplus:
      printf("( ");
      break;
    case lminus:
      printf(") ");
      break;
    case lor:
      printf("| ");
      break;
    case land:
      printf("& ");
      break;
    case lnot:
      printf("!");
    };
  } else {
    printf("%d ", item);
  }
}

void print_logic(LogicExpressionCPU const &logic, int istart = 0, int iend = -1, bool jumps = true)
{
  size_t ilast = (iend >= 0) ? size_t(iend) : logic.size() - 1;
  for (size_t i = istart; i <= ilast; ++i) {
    print_item(logic[i]);
    if (jumps && (logic[i] == land || logic[i] == lor)) i++;
  }
  printf("\n");
}

void print_logic(LogicExpression const &logic, int istart = 0, int iend = -1, bool jumps = true)
{
  size_t ilast = (iend >= 0) ? size_t(iend) : logic.size() - 1;
  for (size_t i = istart; i <= ilast; ++i) {
    print_item(logic[i]);
    if (jumps && (logic[i] == land || logic[i] == lor)) i++;
  }
  printf("\n");
}

void remove_range(LogicExpressionCPU &logic, size_t istart, size_t iend)
{
  for (size_t i = iend + 1; i < logic.size(); ++i)
    logic[i - iend + istart - 1] = logic[i]; // shift left
  for (size_t i = 0; i < iend - istart + 1; ++i)
    logic.pop_back();
}

void remove_parentheses(LogicExpressionCPU &logic, size_t start, size_t end)
{
  assert(logic[start] == lplus && logic[end] == lminus);
  for (size_t i = start + 1; i < end; ++i)
    logic[i - 1] = logic[i]; // shift left
  for (size_t i = end + 1; i < logic.size(); ++i)
    logic[i - 2] = logic[i];
  logic.pop_back();
  logic.pop_back();
}

void insert_jumps(LogicExpressionCPU &logic)
{
  size_t i = 0;
  while (i < logic.size()) {
    if (logic[i] == land || logic[i] == lor) {
      // insert jump placeholder just after the operation
      logic.insert(logic.begin() + i + 1, 0);
    }
    i++;
  }
  int depth = 0;
  for (i = 0; i < logic.size(); ++i) {
    if (logic[i] == lplus)
      depth++;
    else if (logic[i] == lminus)
      depth--;
    else if (logic[i] == land || logic[i] == lor) {
      auto d = depth;
      auto j = i + 1;
      assert(j + 1 < logic.size() && "cannot end expression with an operator");
      // scan for next operator at the same depth or end parenthesis
      while (++j < logic.size()) {
        if (logic[j] == lplus) {
          d++;
        } else if (d == depth && (logic[j] == lminus || logic[j] == land || logic[j] == lor))
          break;
        else if (logic[j] == lminus)
          d--;
      }
      logic[++i] = j;
    }
  }
}

void swap_expressions(LogicExpressionCPU &logic, size_t istart1, size_t iend1, size_t istart2, size_t iend2)
{
  LogicExpressionCPU temp;
  for (size_t i = 0; i < istart1; ++i)
    temp.push_back(logic[i]);
  for (size_t i = istart2; i <= iend2; ++i)
    temp.push_back(logic[i]);
  for (size_t i = iend1 + 1; i < istart2; ++i)
    temp.push_back(logic[i]);
  for (size_t i = istart1; i <= iend1; ++i)
    temp.push_back(logic[i]);
  for (size_t i = iend2 + 1; i < logic.size(); ++i)
    temp.push_back(logic[i]);
  logic = temp;
}

void negate_logic(LogicExpressionCPU &logic, size_t istart, size_t iend, int &inserts)
{
  // negate logic between istart and iend
#ifdef SURF_DEBUG_LOGIC
  printf("negating logic: ");
  print_logic(logic, istart, iend, false);
#endif
  int depth     = 0;
  int mindepth  = std::numeric_limits<int>::max();
  logic_int op  = 0;
  size_t op_pos = 0;
  // search for the first operator
  for (size_t i = istart; i <= iend; ++i) {
    if (logic[i] == lplus) depth++;
    if (logic[i] == lminus) depth--;
    if (depth < mindepth && (logic[i] == lor || logic[i] == land)) {
      op       = logic[i];
      op_pos   = i;
      mindepth = depth;
      if (mindepth == 0) break;
    }
  }

  if (op) {
    // negate left operand
    auto inserts_left = 0;
    negate_logic(logic, istart, op_pos - 1, inserts_left);
    op_pos += inserts_left;
    if (op == lor)
      logic[op_pos] = land;
    else if (op == land)
      logic[op_pos] = lor;
    // negate right operand
    size_t iend1       = iend + inserts_left;
    auto inserts_right = 0;
    inserts            = 0;
    negate_logic(logic, op_pos + 1, iend1, inserts_right);
    inserts = inserts_left + inserts_right;
  } else {
    // this is an operand, find if already negated
    assert(logic[istart] != lplus && "Cannot have paranthesis before operand");
    if (logic[istart] == lnot) {
      logic.erase(logic.begin() + istart);
      inserts--;
    } else {
      logic.insert(logic.begin() + istart, lnot);
      inserts++;
    }
  }
#ifdef SURF_DEBUG_LOGIC
  printf("after negation: ");
  print_logic(logic, istart, iend + inserts, false);
#endif
}

bool is_negated(int isurf, LogicExpressionCPU &logic)
{
  // Find if a given surface is negated in the logic
  for (size_t i = 0; i < logic.size(); ++i) {
    if (logic[i] == isurf) return false;
    if (logic[i] == lnot && logic[++i] == isurf) return true;
  }
  // Should never reach this point
  assert(0 && "Surface not found");
  return false;
}

void negate_closure(LogicExpressionCPU &logic, size_t start)
{
  int ldepth = 1;
  assert(logic[start] == lplus);
  size_t i;
  for (i = start + 1; i < logic.size(); ++i) {
    if (logic[i] == lplus) ldepth++;
    if (logic[i] == lminus) ldepth--;
    if (ldepth == 0) break;
  }
  assert(ldepth == 0 && "parenthesis closure not found");
  int inserts = 0;
  negate_logic(logic, start + 1, i - 1, inserts);
}

size_t find_matching_parenthesis(LogicExpressionCPU &logic, size_t start)
{
  // logic[start] must be lplus
  assert(logic[start] == lplus);
  int ldepth = 0;
  for (size_t i = start; i < logic.size(); ++i) {
    switch (logic[i]) {
    case lminus:
      ldepth--;
      assert(ldepth >= 0);
      if (ldepth == 0) return i;
      break;
    case lplus:
      ldepth++;
      break;
    }
  }
  return logic.size();
}

size_t find_closure_and_op(LogicExpressionCPU &logic, size_t start, logic_int &op, int &complexity)
{
  // Returns the paranthesis closure index matching to the one opened at `start`
  // Fills the operation done at the same parantheses depth, and complexity (number of ops)
  int ldepth = 1;
  complexity = 0;
  op         = 0;
  assert(logic[start] == lplus);
  size_t i;
  for (i = start + 1; i < logic.size(); ++i) {
    if (logic[i] == lplus) ldepth++;
    if (logic[i] == lminus) ldepth--;
    bool is_op = (logic[i] == lor) || (logic[i] == land);
    complexity += int(is_op);
    if (ldepth == 1 && is_op) {
      // Only allow identical operators to exist at the same depth
      assert(op == 0 || op == logic[i]);
      op = logic[i];
    }
    if (ldepth == 0) break;
  }

  return i;
}

/// @brief Logic expression decomposed in a vector of operands (also logic expressions) connected by and/or operators
struct LogicExpressionConstruct {
  bool fNegated{false};                            /// is negated logic
  bool fHasScope{false};                           /// has a scope
  int fComplexity{0};                              /// expression complexity
  LogicExpressionCPU fLogic;                       /// Logic expression
  LogicExpressionCPU fOperators;                   /// Vector of operators
  std::vector<LogicExpressionConstruct> fOperands; /// Vector of operands

  LogicExpressionConstruct(LogicExpressionCPU const &logic)
  try : fLogic(logic) {
    // Decompose expression in operands
    //  scan expression
    int num_operands     = 0;
    size_t start_operand = 0;
    size_t end_operand   = 0;
    int ldepth           = 0;

    auto add_operand = [&]() {
      LogicExpressionCPU current_operand;
      for (size_t i = start_operand; i <= end_operand; ++i)
        current_operand.push_back(fLogic[i]);
      num_operands++;
      // Construct the expression for the current operand
      fOperands.push_back({current_operand});
    };

    auto throw_error = [&](const char *what, size_t index) {
      Print();
      VECGEOM_LOG(critical) << what << " at index " << index;
      throw std::runtime_error(what);
    };

    // Perform some basic checks
    logic_int previous = -1;
    for (size_t i = 0; i < fLogic.size(); ++i) {
      auto item        = fLogic[i];
      bool is_operator = (item == land) || (item == lor);
      if (i == 0) {
        if (is_operator || item == lminus) throw_error("illegal first item", i);
        previous = item;
        continue;
      }
      bool last_operator = (previous == land) || (previous == lor);
      bool last_operand  = !LogicExpression::is_operator_token(previous);
      if (i == fLogic.size() - 1) {
        if (is_operator || item == lnot || item == lplus) throw_error("illegal last item", i);
      }

      switch (item) {
      case lminus:
        // cannot come after a lplus, operator, or lnot
        if (previous == lplus || previous == lnot || last_operator) throw_error("illegal follow-up item ", i);
        break;
      case land:
      case lor:
        // cannot come after an operator an lplus or a lnot
        if (previous == lplus || previous == lnot || last_operator) throw_error("illegal follow-up item ", i);
        break;
      default:
        // cannot come after lminus or operand
        if (previous == lminus || last_operand) throw_error("illegal follow-up item ", i);
      }
      previous = item;
    }

    // Determine the expression complexity
    for (auto item : fLogic) {
      if (item == land || item == lor) {
        fComplexity++;
      }
    }
    if (fComplexity == 0) fHasScope = false;

#ifdef SURF_DEBUG_LOGIC
    std::cout << "constructing logic with complexity " << fComplexity << ": ";
    print_logic(fLogic, 0, -1, false);
#endif

    // Count the operands
    int noperands   = 1;
    bool simplified = true;
    while (simplified && noperands == 1) {
      simplified = false;
      for (size_t i = 0; i < fLogic.size(); ++i) {
        switch (fLogic[i]) {
        case lminus:
          ldepth--;
          break;
        case lplus:
          ldepth++;
          break;
        case land:
        case lor:
          if (ldepth == 0) noperands++;
        }
      }

      if (noperands == 1) {
        // If there is a single operand, extract negation and/or parentheses

        simplified = false;
        while (fLogic[0] == lnot) {
          fNegated   = !fNegated;
          simplified = true;
          remove_range(fLogic, 0, 0);
        }
        // Remove external parentheses if any left
        while (*fLogic.begin() == lplus && find_matching_parenthesis(fLogic, 0) == fLogic.size() - 1) {
          remove_parentheses(fLogic, 0, fLogic.size() - 1);
          simplified = true;
          fHasScope  = true;
        }
        if (simplified) {
#ifdef SURF_DEBUG_LOGIC
          printf("simplified operand: ");
          print_logic(fLogic, 0, -1, false);
#endif
        }
      }
    }

    // Decompose the logic expression
    for (size_t i = 0; i < fLogic.size(); ++i) {
      // scan for next operator
      switch (fLogic[i]) {
      case lminus:
        ldepth--;
        if (ldepth < 0) {
          VECGEOM_LOG(critical) << "LogicExpressionConstruct: parentheses mismatch at index " << i;
          throw std::runtime_error("parentheses mismatch");
        }
        break;
      case lplus:
        ldepth++;
        break;
      case land:
      case lor:
        if (ldepth == 0) {
          fOperators.push_back(fLogic[i]);
          if (i > start_operand) {
            end_operand = i - 1;
            add_operand();
            start_operand = i + 1;
          }
        }
        break;
      };
    }
    if (ldepth != 0) {
      throw std::runtime_error("parentheses mismatch");
    }
    if (fComplexity > 0 && start_operand < fLogic.size()) {
      end_operand = fLogic.size() - 1;
      add_operand();
    }
  } catch (const std::exception &e) {
    VECGEOM_LOG(critical) << "exception " << e.what();
  }

  LogicExpressionConstruct &operator&=(LogicExpressionConstruct const &other)
  {
    Concatenate(other, land);
    return *this;
  }

  LogicExpressionConstruct &operator|=(LogicExpressionConstruct const &other)
  {
    Concatenate(other, lor);
    return *this;
  }

  void Concatenate(LogicExpressionConstruct const &other, logic_int op_concat)
  {
    // Create operand if none exist
#ifdef SURF_DEBUG_LOGIC
    printf("Concatenating : ");
    Print();
    printf("         with : ");
    other.Print();
#endif
    if (!fOperands.size()) {
      if (fNegated) {
        fLogic.insert(fLogic.begin(), lnot);
        fNegated = false;
      }
      fOperands.push_back({fLogic});
    }
    fOperands.push_back(other);
    fOperators.push_back(op_concat);
    fComplexity++;
#ifdef SURF_DEBUG_LOGIC
    printf("       result : ");
    Print();
#endif
  }

  void PushNegation(bool neg = false)
  {
    fNegated ^= neg;
    if (!fComplexity) {
      if (fNegated) fLogic.insert(fLogic.begin(), lnot);
      fNegated = false;
    }
    for (auto &operand : fOperands)
      operand.PushNegation(fNegated);
    for (size_t i = 0; i < fOperators.size(); ++i) {
      if (fNegated) {
        if (fOperators[i] == land)
          fOperators[i] = lor;
        else if (fOperators[i] == lor)
          fOperators[i] = land;
      }
    }
    fNegated = false;
  }

  bool RemoveScope(logic_int logic_op)
  {
    // Remove scope if all operators in the expression match the input one
    bool removed_scope = false;
    if (fHasScope) {
#ifdef SURF_DEBUG_LOGIC
      printf("removing scope for operation ");
      print_item(logic_op);
      printf("in operand: ");
      Print();
#endif
      removed_scope = true;
      for (auto op : fOperators) {
        if (op != logic_op) {
          removed_scope = false;
          break;
        }
      }
      if (removed_scope) fHasScope = false;
#ifdef SURF_DEBUG_LOGIC
      printf("after scope removal: ");
      Print();
#endif
    }
    RemoveChildrenScopes();
    return removed_scope;
  }

  void RemoveChildrenScopes(bool top = false)
  {
    // Remove scope for children operands for surrounding operators
#ifdef SURF_DEBUG_LOGIC
    printf("removing children scopes for: ");
    Print();
#endif
    if (top) fHasScope = false;
    if (fOperators.size() == 0) return;

    for (size_t i = 0; i < fOperands.size(); ++i) {
      size_t sjump        = 0;
      auto &operand       = fOperands[i];
      logic_int op_before = (i > 0) ? fOperators[i - 1] : fOperators[i];
      logic_int op_after  = (i < fOperands.size() - 1) ? fOperators[i] : fOperators[i - 1];
      bool match_op       = op_before == op_after;
      if (match_op) {
        bool removed_scope = operand.RemoveScope(op_before);
        if (removed_scope) {
          // pull the operand expression up
          fComplexity += operand.fComplexity;
          sjump = operand.fOperands.size() - 1;
          fOperators.insert(fOperators.begin() + i, operand.fOperators.begin(), operand.fOperators.end());
          fOperands.insert(fOperands.begin() + i + 1, operand.fOperands.begin(), operand.fOperands.end());
          fOperands.erase(fOperands.begin() + i);
        }
      } else {
        operand.RemoveChildrenScopes();
      }
#ifdef SURF_DEBUG_LOGIC
      printf("  after operand %ld : ", i);
      Print();
#endif
      i += sjump;
    }
#ifdef SURF_DEBUG_LOGIC
    printf("after removing children scopes: ");
    Print();
#endif
  }

  void Print() const
  {
    LogicExpressionCPU logic;
    GetLogicExpression(logic);
    print_logic(logic, 0, -1, false);
  }

  void GivePrecedenceToAnd()
  {
    // If the expression contains both AND and OR operators, swap the expression so that AND has precedence
    bool has_and = std::find(fOperators.begin(), fOperators.end(), land) != fOperators.end();
    bool has_or  = std::find(fOperators.begin(), fOperators.end(), lor) != fOperators.end();
    if (!has_and || !has_or) return;

    while (fOperators[0] == lor) {
      // if the first operator is an OR, move the first operand to the end
      std::rotate(fOperators.begin(), fOperators.begin() + 1, fOperators.end());
      std::rotate(fOperands.begin(), fOperands.begin() + 1, fOperands.end());
    }

    // Add parentheses to AND groups if they are not at the beginning
    // find position of first AND after an OR
    size_t ior  = std::distance(fOperators.begin(), std::find(fOperators.begin(), fOperators.end(), lor));
    size_t iand = std::distance(fOperators.begin(), std::find(fOperators.begin() + ior + 1, fOperators.end(), land));
    while (iand < fOperators.size()) {
      size_t iand_last = iand;
      for (size_t i = iand + 1; i < fOperators.size() && fOperators[i] == land; ++i)
        iand_last = i;
      // Add scope for the AND group
      for (size_t i = iand; i <= iand_last; ++i) {
        fOperands[iand] &= fOperands[i + 1];
      }
      assert(!fOperands[iand].fNegated); // this should be the case, if not we need to push negation to logic expression
      fOperands[iand].fHasScope = true;
      fOperands.erase(fOperands.begin() + iand + 1, fOperands.begin() + iand_last + 2);
      fOperators.erase(fOperators.begin() + iand, fOperators.begin() + iand_last + 1);
      ior  = std::distance(fOperators.begin(), std::find(fOperators.begin() + iand + 1, fOperators.end(), lor));
      iand = std::distance(fOperators.begin(), std::find(fOperators.begin() + ior + 1, fOperators.end(), land));
    }
  }

  void SwapByComplexity()
  {
    // Find range for which swapping is allowed
    if (fComplexity == 0) return;
#ifdef SURF_DEBUG_LOGIC
    printf("SwapByComplexity start: ");
    Print();
#endif
    size_t istart_op = 0;
    size_t iend_op   = 0;
    auto op          = fOperators[istart_op];
    while (istart_op < fOperators.size()) {
      for (size_t i = istart_op + 1; i < fOperators.size(); ++i) {
        if (fOperators[i] == op)
          iend_op = i;
        else
          break;
      }
      size_t istart = istart_op;
      size_t iend   = (op == land) ? iend_op + 1 : iend_op;
      iend          = std::min(iend, fOperands.size() - 1);
      if (iend > istart) {
        // swap by complexity
#ifdef SURF_DEBUG_LOGIC
        printf("sorting operands [%ld, %ld]\n", istart, iend);
#endif
        for (size_t iop = istart; iop < iend; ++iop) {
          for (size_t jop = iop + 1; jop <= iend; ++jop) {
            auto c1 = fOperands[iop].fComplexity;
            auto c2 = fOperands[jop].fComplexity;
            if (c1 > c2) {
              std::iter_swap(fOperands.begin() + iop, fOperands.begin() + jop);
#ifdef SURF_DEBUG_LOGIC
              printf("after swapping %ld (compl=%d) <-> %ld (compl=%d): ", iop, c1, jop, c2);
              Print();
#endif
            }
          }
        }
      }
      istart_op = iend_op + 1;
      if (istart_op < fOperators.size()) {
        op = fOperators[istart_op];
        if (op == lor) {
          istart_op++;
          if (istart_op < fOperators.size()) op = fOperators[istart_op];
        }
      }
      iend_op = istart_op;
    }
    // Now process children
    for (auto &operand : fOperands)
      operand.SwapByComplexity();
  }

  void GetLogicExpression(LogicExpressionCPU &logic) const
  {
    if (fNegated) logic.push_back(lnot);
    if (fHasScope) logic.push_back(lplus);
    if (fComplexity == 0) {
      for (auto &item : fLogic)
        logic.push_back(item);
    } else {
      for (size_t i = 0; i < fOperands.size(); ++i) {
        fOperands[i].GetLogicExpression(logic);
        if (i < fOperators.size()) logic.push_back(fOperators[i]);
      }
    }
    if (fHasScope) logic.push_back(lminus);
  }

  int GetNumOperands() const
  {
    int noperands = (fOperands.size() == 0) ? 1 : 0;
    for (auto const &operand : fOperands)
      noperands += operand.GetNumOperands();
    return noperands;
  }

  void Simplify(LogicExpressionCPU &logic)
  {
    PushNegation();
    RemoveChildrenScopes(true);
    GivePrecedenceToAnd();
    SwapByComplexity();
    logic.clear();
    GetLogicExpression(logic);
    insert_jumps(logic);
  }
};

int max_operand(LogicExpressionCPU const &logic)
{
  int max_op = -1;
  for (auto const &item : logic) {
    if (LogicExpression::is_operator_token(item)) continue;
    max_op = std::max(max_op, int(item));
  }
  return max_op;
}

void substitute_logic(LogicExpressionCPU const &logic, LogicExpressionCPU &substutute_logic, const bool *values)
{
  substutute_logic.clear();
  for (auto item : logic) {
    if (LogicExpression::is_operator_token(item))
      substutute_logic.push_back(item);
    else
      substutute_logic.push_back(logic_int(values[item]));
  }
}

bool evaluate_logic(LogicExpressionCPU const &logic, const bool *values, int &num_eval)
{
  auto test_bit   = [](unsigned bset, int bit) { return (bset & (unsigned(1) << bit)) > 0; };
  auto set_bit    = [](unsigned &bset, int bit) { bset |= unsigned(1) << bit; };
  auto reset_bit  = [](unsigned &bset, int bit) { bset &= ~(unsigned(1) << bit); };
  auto swap_bit   = [](unsigned &bset, int bit) { bset ^= unsigned(1) << bit; };
  unsigned stack  = 0;
  unsigned negate = 0;
  int depth       = 0;
  num_eval        = 0;
  bool result     = false;
  unsigned i;
  for (i = 0; i < logic.size(); ++i) {
    switch (logic[i]) {
    case lplus:
      depth++;
      break;
    case lminus:
      result = test_bit(stack, depth);
      reset_bit(negate, depth--);
      result ^= test_bit(negate, depth);
      if (result)
        set_bit(stack, depth);
      else
        reset_bit(stack, depth);
      break;
    case lnot:
      swap_bit(negate, depth);
      break;
    case lor:
      if (test_bit(stack, depth))
        i = logic[i + 1] - 1;
      else
        i++;
      break;
    case land:
      if (!test_bit(stack, depth))
        i = logic[i + 1] - 1;
      else
        i++;
      break;
    default:
      // This is an operand
      result = values[int(logic[i])];
      result ^= test_bit(negate, depth);
      reset_bit(negate, depth);
      num_eval++;
      // We can ignore the previous value because of short-circuiting
      if (result)
        set_bit(stack, depth);
      else
        reset_bit(stack, depth);
    }
  }
  assert(depth == 0);
  return (stack & 1) > 0;
}

struct Placed {
  vecgeom::Transformation3D fTrans;                   ///< Transformation
  int fVolId{-1};                                     ///< Volume Id
  vecgeom::VUnplacedVolume const *fUnplaced{nullptr}; ///< Unplaced volume

  virtual int GetComplexity() const { return 0; }

  Placed(vecgeom::Transformation3D const &trans, int volId, vecgeom::VUnplacedVolume const *unplaced = nullptr)
      : fTrans{trans}, fVolId{volId}, fUnplaced(unplaced)
  {
  }
  virtual ~Placed() {}
};

struct Bnode : public Placed {
  int depth_{0};
  bool neg_left_{false};
  bool neg_right_{false};
  Placed *left_{nullptr};
  Placed *right_{nullptr};
  vecgeom::BooleanOperation op_;

  virtual ~Bnode()
  {
    delete left_;
    delete right_;
  }

  Bnode(vecgeom::Transformation3D const &trans, int volId, vecgeom::BooleanStruct const &bstruct) : Placed(trans, volId)
  {
    // Transform subtractions in intersection with negation
    op_ = bstruct.fOp;
    if (op_ == vecgeom::kSubtraction) {
      op_        = vecgeom::kIntersection;
      neg_right_ = true;
    }

    // left node
    vecgeom::Transformation3D tr_left(trans);
    tr_left.MultiplyFromRight(*bstruct.fLeftVolume->GetTransformation());
    auto const unplaced_left = bstruct.fLeftVolume->GetUnplacedVolume();
    auto bstruct_left        = vecgeom::BooleanHelper::GetBooleanStruct(unplaced_left);
    if (bstruct_left)
      left_ = new Bnode(tr_left, volId, *bstruct_left);
    else
      left_ = new Placed(tr_left, volId, unplaced_left);

    // right node
    vecgeom::Transformation3D tr_right(trans);
    tr_right.MultiplyFromRight(*bstruct.fRightVolume->GetTransformation());
    auto const unplaced_right = bstruct.fRightVolume->GetUnplacedVolume();
    auto bstruct_right        = vecgeom::BooleanHelper::GetBooleanStruct(unplaced_right);
    if (bstruct_right)
      right_ = new Bnode(tr_right, volId, *bstruct_right);
    else
      right_ = new Placed(tr_right, volId, unplaced_right);
  }

  int GetComplexity() const override
  {
    int complexity = 1;
    complexity += left_->GetComplexity() + right_->GetComplexity();
    return complexity;
  }
};
} // namespace logichelper
} // namespace vgbrep
#endif
