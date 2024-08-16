/* Copyright (c) 2012 the authors listed at the following URL, and/or
the authors of referenced articles or incorporated external code:
http://en.literateprograms.org/Shunting_yard_algorithm_(C)?action=history&offset=20080201043325

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Retrieved from: http://en.literateprograms.org/Shunting_yard_algorithm_(C)?oldid=12454
*/

#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>

#define SHUNT_MAXOPSTACK 256
#define SHUNT_MAXNUMSTACK 256

namespace shunt {

bool eval_not(bool a1, bool a2) { return !a1; }
bool eval_and(bool a1, bool a2) { return a1 && a2; }
bool eval_or(bool a1, bool a2) { return a1 || a2; }

enum { ASSOC_NONE = 0, ASSOC_LEFT, ASSOC_RIGHT };

struct op_s {
  char op;
  int prec;
  int assoc;
  int unary;
  bool (*eval)(bool a1, bool a2);
} ops[] = {{'!', 10, ASSOC_RIGHT, 1, eval_not},
           {'&', 8, ASSOC_LEFT, 0, eval_and},
           {'|', 5, ASSOC_LEFT, 0, eval_or},
           {'(', 0, ASSOC_NONE, 0, NULL},
           {')', 0, ASSOC_NONE, 0, NULL}};

struct op_s *getop(char ch)
{
  size_t i;
  for (i = 0; i < sizeof ops / sizeof ops[0]; ++i) {
    if (ops[i].op == ch) return ops + i;
  }
  return nullptr;
}

void push_opstack(struct op_s *op, struct op_s **opstack, int &nopstack)
{
  if (nopstack > SHUNT_MAXOPSTACK - 1) {
    fprintf(stderr, "ERROR: Operator stack overflow\n");
    exit(EXIT_FAILURE);
  }
  opstack[nopstack++] = op;
}

struct op_s *pop_opstack(struct op_s **opstack, int &nopstack)
{
  if (!nopstack) {
    fprintf(stderr, "ERROR: Operator stack empty\n");
    exit(EXIT_FAILURE);
  }
  return opstack[--nopstack];
}

void push_numstack(bool num, bool *numstack, int &nnumstack)
{
  if (nnumstack > SHUNT_MAXNUMSTACK - 1) {
    fprintf(stderr, "ERROR: Number stack overflow\n");
    exit(EXIT_FAILURE);
  }
  numstack[nnumstack++] = num;
}

bool pop_numstack(bool *numstack, int &nnumstack)
{
  if (!nnumstack) {
    fprintf(stderr, "ERROR: Number stack empty\n");
    exit(EXIT_FAILURE);
  }
  return numstack[--nnumstack];
}

void shunt_op(struct op_s *op, struct op_s **opstack, int &nopstack, bool *numstack, int &nnumstack)
{
  struct op_s *pop;
  bool n1, n2;
  if (op->op == '(') {
    push_opstack(op, opstack, nopstack);
    return;
  } else if (op->op == ')') {
    while (nopstack > 0 && opstack[nopstack - 1]->op != '(') {
      pop = pop_opstack(opstack, nopstack);
      n1  = pop_numstack(numstack, nnumstack);
      if (pop->unary)
        push_numstack(pop->eval(n1, false), numstack, nnumstack);
      else {
        n2 = pop_numstack(numstack, nnumstack);
        push_numstack(pop->eval(n2, n1), numstack, nnumstack);
      }
    }
    if (!(pop = pop_opstack(opstack, nopstack)) || pop->op != '(') {
      fprintf(stderr, "ERROR: Stack error. No matching \'(\'\n");
      exit(EXIT_FAILURE);
    }
    return;
  }

  if (op->assoc == ASSOC_RIGHT) {
    while (nopstack && op->prec < opstack[nopstack - 1]->prec) {
      pop = pop_opstack(opstack, nopstack);
      n1  = pop_numstack(numstack, nnumstack);
      if (pop->unary)
        push_numstack(pop->eval(n1, 0), numstack, nnumstack);
      else {
        n2 = pop_numstack(numstack, nnumstack);
        push_numstack(pop->eval(n2, n1), numstack, nnumstack);
      }
    }
  } else {
    while (nopstack && op->prec <= opstack[nopstack - 1]->prec) {
      pop = pop_opstack(opstack, nopstack);
      n1  = pop_numstack(numstack, nnumstack);
      if (pop->unary)
        push_numstack(pop->eval(n1, 0), numstack, nnumstack);
      else {
        n2 = pop_numstack(numstack, nnumstack);
        push_numstack(pop->eval(n2, n1), numstack, nnumstack);
      }
    }
  }
  push_opstack(op, opstack, nopstack);
}

bool logic_evaluate(std::string const &slogic)
{
  struct op_s *opstack[SHUNT_MAXOPSTACK];
  int nopstack = 0;

  bool numstack[SHUNT_MAXNUMSTACK];
  int nnumstack       = 0;
  auto tstart         = slogic.size();
  struct op_s startop = {'X', 0, ASSOC_NONE, 0, NULL}; /* Dummy operator to mark start */
  struct op_s *op     = NULL;
  int n1, n2;
  struct op_s *lastop = &startop;

  for (size_t iexpr = 0; iexpr < slogic.size(); ++iexpr) {
    if (tstart == slogic.size()) {

      if ((op = getop(slogic[iexpr]))) {
        if (lastop && (lastop == &startop || lastop->op != ')')) {
          if (op->op != '(' && op->op != '!') {
            fprintf(stderr, "ERROR: Illegal use of binary operator (%c)\n", op->op);
            exit(EXIT_FAILURE);
          }
        }
        shunt_op(op, opstack, nopstack, numstack, nnumstack);
        lastop = op;
      } else if (isdigit(slogic[iexpr]))
        tstart = iexpr;
      else if (!isspace(slogic[iexpr])) {
        fprintf(stderr, "ERROR: Syntax error\n");
        exit(EXIT_FAILURE);
      }
    } else {
      if (isspace(slogic[iexpr])) {
        push_numstack(slogic[tstart] - '0', numstack, nnumstack);
        tstart = slogic.size();
        lastop = nullptr;
      } else if ((op = getop(slogic[iexpr]))) {
        push_numstack(slogic[tstart] - '0', numstack, nnumstack);
        tstart = slogic.size();
        shunt_op(op, opstack, nopstack, numstack, nnumstack);
        lastop = op;
      } else if (!isdigit(slogic[iexpr])) {
        fprintf(stderr, "ERROR: Syntax error\n");
        exit(EXIT_FAILURE);
      }
    }
  }
  if (tstart < slogic.size()) push_numstack(slogic[tstart] - '0', numstack, nnumstack);

  while (nopstack) {
    op = pop_opstack(opstack, nopstack);
    n1 = pop_numstack(numstack, nnumstack);
    if (op->unary)
      push_numstack(op->eval(n1, 0), numstack, nnumstack);
    else {
      n2 = pop_numstack(numstack, nnumstack);
      push_numstack(op->eval(n2, n1), numstack, nnumstack);
    }
  }
  if (nnumstack != 1) {
    fprintf(stderr, "ERROR: Number stack has %d elements after evaluation. Should be 1.\n", nnumstack);
    exit(EXIT_FAILURE);
  }
  //printf("%d\n", numstack[0]);

  return numstack[0];
}
} // end namespace shunt
