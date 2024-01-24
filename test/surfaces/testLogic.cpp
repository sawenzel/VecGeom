#include <iostream>
#include <string>

#include "test/benchmark/ArgParser.h"
#include <VecGeom/base/RNG.h>
#include <VecGeom/surfaces/conv/LogicHelper.h>
#include <VecGeom/surfaces/conv/shunt.h>

int test_logic(vgbrep::LogicExpressionCPU &logic_expr, const char *swhat, int nsamples)
{
  auto max_operand                            = vgbrep::logichelper::max_operand(logic_expr);
  vgbrep::LogicExpressionCPU logic_expr_start = logic_expr;

  std::cout << "testing " << swhat << ": ";
  vgbrep::logichelper::print_logic(logic_expr_start, 0, -1, false);
  vgbrep::logichelper::LogicExpressionConstruct lc(logic_expr);
  lc.Simplify(logic_expr);
  std::cout << "   simplified logic: ";
  vgbrep::logichelper::print_logic(logic_expr, 0, -1, true);
  if (max_operand > 1000) {
    std::cout << "To test the expression, introduce operands with values less than 1000\n";
    return 0;
  }
  bool *test_sample = new bool[max_operand + 1];

  auto fill_sample = [](bool *values, int n) {
    auto &rng = vecgeom::RNG::Instance();
    for (int i = 0; i <= n; ++i)
      values[i] = rng.uniform() < 0.5;
  };

  bool success     = true;
  int num_operands = lc.GetNumOperands();
  int num_eval_tot = 0;
  vgbrep::LogicExpressionCPU logic_expr_eval;
  for (int i = 0; i < nsamples; ++i) {
    fill_sample(test_sample, max_operand);
    vgbrep::logichelper::substitute_logic(logic_expr_start, logic_expr_eval, test_sample);
    std::string string_eval;
    for (auto item : logic_expr_eval)
      string_eval.push_back(vgbrep::logichelper::logic_to_char(item));
    auto v1      = shunt::logic_evaluate(string_eval);
    int num_eval = 0;
    auto v2      = vgbrep::logichelper::evaluate_logic(logic_expr, test_sample, num_eval);
    num_eval_tot += num_eval;
    if (v1 != v2) {
      std::cout << "Logic simplification error for sample " << string_eval << " should give: " << v1 << "\n";
      vgbrep::logichelper::substitute_logic(logic_expr_start, logic_expr_eval, test_sample);
      v2      = vgbrep::logichelper::evaluate_logic(logic_expr, test_sample, num_eval);
      success = false;
      break;
    }
  }
  delete[] test_sample;
  if (success) {
    std::cout << "   operands / number of evaluations: " << num_operands << " / " << float(num_eval_tot) / nsamples
              << " = " << float(num_operands) * nsamples / num_eval_tot << "\n";
    return 0;
  }
  return 1;
}

int run_tests(int nsamples)
{
  vgbrep::LogicExpressionCPU logic_expr;
  bool success;
  int result = 0;
  // Double negation
  success = vgbrep::logichelper::string_to_logic("!!1", logic_expr);
  if (!success) return 1;
  result += test_logic(logic_expr, "double negation", nsamples);
  success = vgbrep::logichelper::string_to_logic("!(1 & 2) & !(3 | 4)", logic_expr);
  if (!success) return 1;
  result += test_logic(logic_expr, "De Morgan", nsamples);
  success = vgbrep::logichelper::string_to_logic("((1 & 2))", logic_expr);
  if (!success) return 1;
  result += test_logic(logic_expr, "double scope", nsamples);
  success = vgbrep::logichelper::string_to_logic("(1 & 2) & !((3 | 4) | 5) & (6 & 7)", logic_expr);
  if (!success) return 1;
  result += test_logic(logic_expr, "scope removal", nsamples);
  success = vgbrep::logichelper::string_to_logic("(1 | 2) & (3 | 4 | 5) & 6 & 7", logic_expr);
  if (!success) return 1;
  result += test_logic(logic_expr, "swap complexity", nsamples);
  success = vgbrep::logichelper::string_to_logic("1 | 2 & 3 | 4 | 5 & 6 | 7 | 8 & 9 & 10", logic_expr);
  if (!success) return 1;
  result += test_logic(logic_expr, "operator precedence", nsamples);
  success = vgbrep::logichelper::string_to_logic(
      "(1 | !(2 & 3) | !((4 | 5) & !(6 | 7) | (8 & 9 & 10))) & ((11 | 12) & 13)", logic_expr);
  if (!success) return 1;
  result += test_logic(logic_expr, "complex", nsamples);
  success = vgbrep::logichelper::string_to_logic(
      "((6&7&8&9&(10|11))|((12&13&14&15&(16|17))|((18&19&20&21&(22|23))|((24&25&26&27&(28|29))|((30&31&32&33&(34|35))|("
      "(36&37&38&39&(40|41))|((42&43&44&45&(46|47))|((48&49&50&51&(52|53))|((54&55&56&57&(58|59))|(60&61&62&63&(64|65))"
      ")))))))))",
      logic_expr);
  if (!success) return 1;
  result += test_logic(logic_expr, "very complex", nsamples);
  return result;
}

int main(int argc, char *argv[])
{
  OPTION_STRING(logic, "");
  OPTION_INT(nsamples, 100000);

  int result = 0;
  if (!logic.empty()) {
    vgbrep::LogicExpressionCPU logic_expr;
    auto success = vgbrep::logichelper::string_to_logic(logic, logic_expr);
    if (!success) return 1;
    result = test_logic(logic_expr, "user logic", nsamples);
  } else {
    result = run_tests(nsamples);
  }
  return result;
}
