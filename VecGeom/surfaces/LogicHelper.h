#ifndef VECGEOM_SURFACE_LOGICHELPER_H_
#define VECGEOM_SURFACE_LOGICHELPER_H_

#include <cassert>
#include <VecGeom/surfaces/Model.h>
#include <VecGeom/volumes/BooleanVolume.h>

namespace vgbrep {

using LogicExpressionCPU = std::vector<logic_int>;

namespace logichelper {

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

void print_logic(LogicExpressionCPU const &logic, size_t istart = 0, size_t iend = 0, bool jumps = true)
{
  size_t ilast = (iend > 0) ? iend : logic.size() - 1;
  for (size_t i = istart; i <= ilast; ++i) {
    print_item(logic[i]);
    if (jumps && (logic[i] == land || logic[i] == lor)) i++;
  }
  printf("\n");
}

void print_logic(LogicExpression const &logic, size_t istart = 0, size_t iend = 0, bool jumps = true)
{
  size_t ilast = (iend > 0) ? iend : logic.size() - 1;
  for (size_t i = istart; i <= ilast; ++i) {
    print_item(logic[i]);
    if (jumps && (logic[i] == land || logic[i] == lor)) i++;
  }
  printf("\n");
}

size_t find_closure_and_op(LogicExpressionCPU &logic, size_t start, logic_int &op, int &complexity)
{
  // Returns the paranthesis closure index matching to the one opened at `start`
  // Fills the operation done at the same paranthesis depth, and complexity (number of ops)
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

void remove_paranthesys(LogicExpressionCPU &logic, size_t start, size_t end)
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
      auto j = i;
      while (d >= depth) {
        j++;
        if (j == logic.size()) break;
        if (logic[j] == lplus)
          d++;
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
  //  printf("negating logic: ");
  //  print_logic(logic, istart, iend, false);
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
  //  printf("after negation: ");
  //  print_logic(logic, istart, iend + inserts, false);
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

void simplify_logic(LogicExpressionCPU &logic)
{
  // no need for outer paranthesys if any
  size_t start = 0, end = 0;
  logic_int op = 0;
  int complexity;
  while (logic[start] == lplus) {
    end = find_closure_and_op(logic, start, op, complexity);
    if (op == 0 || (end == logic.size() - 1))
      remove_paranthesys(logic, start, end);
    else
      break;
  }
  // print_logic(logic, 0, 0, false);
  //  scan expression
  logic_int last_op = 0;
  int ldepth        = 0;
  int last_depth    = 0;
  bool action       = true;
  while (action) {
    action = false;
    for (size_t i = 0; i < logic.size(); ++i) {
      if (logic[i] == lor || logic[i] == land) {
        last_op    = logic[i];
        last_depth = ldepth;
      }

      if (logic[i] == lplus) {
        size_t ilast = find_closure_and_op(logic, i, op, complexity);
        // printf("complexity %d for closure: ", complexity);
        // print_logic(logic, i, ilast, false);
        assert(op > 0);
        if (last_op) {
          if (op == last_op && ldepth == last_depth) {
            // printf("removing paranthesys for: ");
            // print_logic(logic, i, ilast, false);
            remove_paranthesys(logic, i--, ilast);
            // print_logic(logic, 0, 0, false);
            action = true;
            continue;
          }
        }
        logic_int next_op = 0;
        if (ilast < logic.size() - 1) {
          next_op = logic[ilast + 1];
          if (next_op != lor && next_op != land) next_op = 0;
        }

        if (op == next_op) {
          // printf("removing paranthesys for: ");
          // print_logic(logic, i, ilast, false);
          remove_paranthesys(logic, i--, ilast);
          // print_logic(logic, 0, 0, false);
          action = true;
          ldepth--;
          last_op = op;
        } else if (next_op > 0) {
          // we can swap (commutativity) if next expression is simpler
          int inext     = ilast + 2;
          int inextlast = inext;
          if (logic[inextlast] == lnot) inextlast++;
          int complexity_next = 0;
          if (logic[inext] == lplus) {
            // another expression to the right
            inextlast = find_closure_and_op(logic, inext, op, complexity_next);
            // printf("complexity %d for next closure: ", complexity_next);
            // print_logic(logic, inext, inextlast, false);
          }
          if (complexity_next < complexity) {
            // we can swap the expressions
            swap_expressions(logic, i--, ilast, inext, inextlast);
            // printf("after swap: ");
            // print_logic(logic, 0, 0, false);
            last_op = 0;
            action  = true;
          }
        }
        ldepth++;
      } else if (logic[i] == lminus)
        ldepth--;
    }
  }
  insert_jumps(logic);
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
