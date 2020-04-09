# multiple fixes by buriy

# -*- coding: utf-8 -*-
"""
    codegen
    ~~~~~~~
    Extension to ast that allow ast -> python code generation.
    :copyright: Copyright 2008 by Armin Ronacher.
    :license: BSD.
"""
from .my_ast import *


def to_source(node, indent_with=' ' * 4, add_line_information=False):
    """This function can convert a node tree back into python sourcecode.
    This is useful for debugging purposes, especially if you're dealing with
    custom asts not generated by python itself.
    It could be that the sourcecode is evaluable when the AST itself is not
    compilable / evaluable.  The reason for this is that the AST contains some
    more data than regular sourcecode does, which is dropped during
    conversion.
    Each level of indentation is replaced with `indent_with`.  Per default this
    parameter is equal to four spaces as suggested by PEP 8, but it might be
    adjusted to match the application's styleguide.
    If `add_line_information` is set to `True` comments for the line numbers
    of the nodes are added to the output.  This can be used to spot wrong line
    number information of statement nodes.
    """
    generator = SourceGenerator(indent_with, add_line_information)
    generator.visit(node)
    return ''.join(map(str, generator.result))

class SourceGenerator(NodeVisitor):
    """This visitor is able to transform a well formed syntax tree into python
    sourcecode.  For more details have a look at the docstring of the
    `node_to_source` function.
    """

    def __init__(self, indent_with, add_line_information=False):
        self.result = []
        self.parents = []
        self.indent_with = indent_with
        self.add_line_information = add_line_information
        self.indentation = 0
        self.new_lines = 0
        self.first = True
        self.docstring = ''

    def write(self, x):
        if self.new_lines:
            if self.result:
                self.result.append('\n' * self.new_lines)
            if self.indentation!=0:
                self.result.append(self.indent_with * self.indentation)
            self.new_lines = 0
        self.result.append(x)

    def newline(self, node=None, extra=0):
        self.new_lines = max(self.new_lines, 1 + extra)
        if node is not None and self.add_line_information:
            # self.write('# line: %s' % node.lineno)
            self.parents.append(type(node).__name__)
            # self.new_lines = 1
            self.write('%s: ' % node.lineno)
            self.parents.append(type(node).__name__)

    def body(self, statements, parent=None):
        node = parent
        self.new_line = True
        self.indentation += 1
        for stmt in statements:
            self.visit(stmt, node)
        self.indentation -= 1

    def body_or_else(self, node):
        self.body(node.body, node)
        if node.orelse:
            self.newline()
            self.write('else:')
            self.parents.append(type(node).__name__)
            self.body(node.orelse, node)

    def signature(self, node):
        want_comma = []

        def write_comma():
            if want_comma:
                self.write(', ')
                self.parents.append(type(node).__name__)
            else:
                want_comma.append(True)

        padding = [None] * (len(node.args) - len(node.defaults))
        for arg, default in zip(node.args, padding + node.defaults):
            write_comma()
            self.write(arg.arg)
            self.parents.append(type(node).__name__)
            if default is not None:
                self.write('=')
                self.parents.append(type(node).__name__)
                self.visit(default, node)

        if node.vararg is not None:
            write_comma()
            self.write('*' + node.vararg.arg)
            self.parents.append(type(node).__name__)

        for arg, default in zip(node.kwonlyargs, node.kw_defaults):
            write_comma()
            self.write(arg.arg)
            self.parents.append(type(node).__name__)
            if default is not None:
                self.write('=')
                self.parents.append(type(node).__name__)
                self.visit(default, node)

        if node.kwarg is not None:
            write_comma()
            self.write('**' + node.kwarg.arg)
            self.parents.append(type(node).__name__)

    def decorators(self, node):
        if not node:
            print
            "No node"
            return
        if not hasattr(node, 'decorator_list'): return
        for decorator in node.decorator_list:
            self.newline(decorator)
            self.write('@')
            self.parents.append(type(node).__name__)
            self.visit(decorator, node)

    # Statements

    def visit_Assign(self, node):
        # self.newline(node)
        for idx, target in enumerate(node.targets):
            if idx:
                self.write(', ')
                self.parents.append(type(node).__name__)
            self.visit(target, node)
        # self.write(' = ')
        self.parents.append(type(node).__name__)
        self.visit(node.value, node)

    def visit_AugAssign(self, node):
        self.newline(node)
        self.visit(node.target, node)
        # self.write(BINOP_SYMBOLS[type(node.op)] + '=')
        self.parents.append(type(node).__name__)
        self.visit(node.value, node)

    def visit_ImportFrom(self, node):
        self.newline(node)
        self.write('from %s%s import ' % ('.' * node.level, node.module))
        self.parents.append(type(node).__name__)
        for idx, item in enumerate(node.names):
            if idx:
                self.write(', ')
                self.parents.append(type(node).__name__)
            self.visit(item, node)

    def visit_Import(self, node):
        self.newline(node)
        self.write('import ')
        self.parents.append(type(node).__name__)
        for idx, item in enumerate(node.names):
            if idx:
                self.write(', ')
                self.parents.append(type(node).__name__)
            self.visit(item, node)

    def visit_Expr(self, node):
        self.newline(node)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.docstring = get_docstring(node, clean=False)
        self.newline()
        self.decorators(node)
        self.newline(node)
        if node.decorator_list:
            for decorator in node.decorator_list:
                self.write('@')
                self.parents.append(type(node).__name__)
                self.visit(decorator, node)
        self.write('def ')
        self.parents.append(type(node).__name__)
        self.write('%s' % node.name)
        self.parents.append(type(node).__name__)
        self.write('(')
        self.parents.append(type(node).__name__)
        self.signature(node.args)
        self.write('):')
        self.parents.append(type(node).__name__)
        self.body(node.body, node)

    def visit_ClassDef(self, node):
        have_args = []

        def paren_or_comma():
            if have_args:
                self.write(', ')
                self.parents.append(type(node).__name__)
            else:
                have_args.append(True)
                self.write('(')
                self.parents.append(type(node).__name__)

        self.newline(extra=2)
        self.decorators(node)
        self.newline(node)
        self.write('class %s' % node.name)
        self.parents.append(type(node).__name__)
        for base in node.bases:
            paren_or_comma()
            self.visit(base, node)
        # XXX: the if here is used to keep this module compatible
        #      with python 2.6.
        if hasattr(node, 'keywords'):
            for keyword in node.keywords:
                paren_or_comma()
                self.write(keyword.arg + '=')
                self.parents.append(type(node).__name__)
                self.visit(keyword.value, node)
            if hasattr(node, 'starargs') and node.starargs is not None:
                paren_or_comma()
                self.write('*')
                self.parents.append(type(node).__name__)
                self.visit(node.starargs, node)
            if hasattr(node, 'kwargs') and node.kwargs is not None:
                paren_or_comma()
                self.write('**')
                self.parents.append(type(node).__name__)
                self.visit(node.kwargs, node)
        self.write(have_args and '):' or ':')
        self.parents.append(type(node).__name__)
        self.body(node.body, node)

    def visit_If(self, node):
        self.newline(node)
        self.write('if ')
        self.parents.append(type(node).__name__)
        self.visit(node.test, node)
        self.write(':')
        self.parents.append(type(node).__name__)
        self.body(node.body, node)
        while True:
            else_ = node.orelse
            if len(else_) == 1 and isinstance(else_[0], If):
                node = else_[0]
                self.newline()
                self.write('elif ')
                self.parents.append(type(node).__name__)
                self.visit(node.test, node)
                self.write(':')
                self.parents.append(type(node).__name__)
                self.body(node.body, node)
            elif else_:
                self.newline()
                self.write('else:')
                self.parents.append(type(node).__name__)
                self.body(else_, node)
                break
            else:
                break

    def visit_For(self, node):
        self.newline(node)
        self.write('for ')
        self.parents.append(type(node).__name__)
        self.visit(node.target, node)
        self.write(' in ')
        self.parents.append(type(node).__name__)
        self.visit(node.iter, node)
        self.write(':')
        self.parents.append(type(node).__name__)
        self.body_or_else(node, node)

    def visit_While(self, node):
        self.newline(node)
        self.write('while ')
        self.parents.append(type(node).__name__)
        self.visit(node.test, node)
        self.write(':')
        self.parents.append(type(node).__name__)
        self.body_or_else(node, node)

    def visit_With(self, node):
        self.newline(node)
        self.write('with ')
        self.parents.append(type(node).__name__)
        for item in node.items:
            self.visit(item.context_expr, node)
            if item.optional_vars is not None:
                self.write(' as ')
                self.parents.append(type(node).__name__)
                self.visit(item.optional_vars, node)
        self.write(':')
        self.parents.append(type(node).__name__)
        self.body(node.body, node)

    def visit_Pass(self, node):
        self.newline(node)
        self.write('pass')
        self.parents.append(type(node).__name__)

    def visit_Print(self, node):
        # XXX: python 2.6 only
        self.newline(node)
        self.write('print ')
        self.parents.append(type(node).__name__)
        want_comma = False
        if node.dest is not None:
            self.write(' >> ')
            self.parents.append(type(node).__name__)
            self.visit(node.dest, node)
            want_comma = True
        for value in node.values:
            if want_comma:
                self.write(', ')
                self.parents.append(type(node).__name__)
            self.visit(value, node)
            want_comma = True
        if not node.nl:
            self.write(',')
            self.parents.append(type(node).__name__)

    def visit_Delete(self, node):
        self.newline(node)
        self.write('del ')
        self.parents.append(type(node).__name__)
        for idx, target in enumerate(node.targets):
            if idx:
                self.write(', ')
                self.parents.append(type(node).__name__)
            self.visit(target, node)

    def visit_Try(self, node):
        self.newline(node)
        #try block
        self.write('try:')
        self.parents.append(type(node).__name__)
        self.body(node.body, node)
        self.newline(node)

        #except block
        for handler in node.handlers:
            self.visit(handler, node)

        #except else
        if len(node.orelse):
            self.write('else:')
            self.parents.append(type(node).__name__)
            self.body(node.orelse, node)

        #except finally
        if len(node.finalbody):
            self.write('finally:')
            self.parents.append(type(node).__name__)
            self.body(node.finalbody, node)



    def visit_TryExcept(self, node):
        self.newline(node)
        self.write('try:')
        self.parents.append(type(node).__name__)
        self.body(node.body, node)
        for handler in node.handlers:
            self.visit(handler, node)

    def visit_TryFinally(self, node):
        self.newline(node)
        self.write('try:')
        self.parents.append(type(node).__name__)
        self.body(node.body, node)
        self.newline(node)
        self.write('finally:')
        self.parents.append(type(node).__name__)
        self.body(node.finalbody, node)

    def visit_Global(self, node):
        self.newline(node)
        self.write('global ' + ', '.join(node.names))
        self.parents.append(type(node).__name__)

    def visit_Nonlocal(self, node):
        self.newline(node)
        self.write('nonlocal ' + ', '.join(node.names))
        self.parents.append(type(node).__name__)

    def visit_Return(self, node):
        self.newline(node)
        self.write('return ')
        self.parents.append(type(node).__name__)
        if node.value:
            self.visit(node.value, node)

    def visit_Break(self, node):
        self.newline(node)
        self.write('break')
        self.parents.append(type(node).__name__)

    def visit_Continue(self, node):
        self.newline(node)
        self.write('continue')
        self.parents.append(type(node).__name__)

    def visit_Raise(self, node):
        # XXX: Python 2.6 / 3.0 compatibility
        self.newline(node)
        self.write('raise')
        self.parents.append(type(node).__name__)
        if hasattr(node, 'exc') and node.exc is not None:
            self.write(' ')
            self.parents.append(type(node).__name__)
            self.visit(node.exc, node)
            if node.cause is not None:
                self.write(' from ')
                self.parents.append(type(node).__name__)
                self.visit(node.cause, node)
        elif hasattr(node, 'type') and node.type is not None:
            self.visit(node.type, node)
            if node.inst is not None:
                self.write(', ')
                self.parents.append(type(node).__name__)
                self.visit(node.inst, node)
            if node.tback is not None:
                self.write(', ')
                self.parents.append(type(node).__name__)
                self.visit(node.tback, node)

    # Expressions

    def visit_Attribute(self, node):
        self.visit(node.value, node)
        self.write('.')
        self.parents.append(type(node).__name__)
        self.write(node.attr)
        self.parents.append(type(node).__name__)

    def visit_Call(self, node):
        want_comma = []

        def write_comma():
            if want_comma:
                self.write(', ')
                self.parents.append(type(node).__name__)
            else:
                want_comma.append(True)

        self.visit(node.func, node)
        self.write('(')
        self.parents.append(type(node).__name__)
        for arg in node.args:
            write_comma()
            self.visit(arg, node)
        for keyword in node.keywords:
            write_comma()
            if keyword.arg:
                self.write(keyword.arg + '=')
                self.parents.append(type(node).__name__)
                self.visit(keyword.value, node)
            else:
                self.write('**')
                self.parents.append(type(node).__name__)
                self.visit(keyword.value, node)

        # if hasattr(node, 'starargs') and node.starargs is not None:
        #     write_comma()
        #     self.write('*')
        self.parents.append(type(node).__name__)
        #     self.visit(node.starargs, node)
        # if hasattr(node, 'kwargs') and node.kwargs is not None:
        #     write_comma()
        #     self.write('**')
        self.parents.append(type(node).__name__)
        #     self.visit(node.kwargs, node)
        self.write(')')
        self.parents.append(type(node).__name__)

    def visit_Name(self, node):
        self.write(node.id)
        self.parents.append(type(node).__name__)

    def visit_Str(self, node):
        if self.docstring != node.s:
            self.write(repr(node.s))
            self.parents.append(type(node).__name__)

    def visit_Bytes(self, node):
        self.write(repr(node.s))
        self.parents.append(type(node).__name__)

    def visit_Num(self, node):
        self.write(repr(node.n))
        self.parents.append(type(node).__name__)

    def visit_Tuple(self, node):
        self.write('(')
        self.parents.append(type(node).__name__)
        idx = -1
        for idx, item in enumerate(node.elts):
            if idx:
                self.write(',')
                self.parents.append(type(node).__name__)
            self.visit(item, node)
        self.write(idx and ')' or ',)')
        self.parents.append(type(node).__name__)

    def sequence_visit(left, right):
        def visit(self, node):
            self.write(left)
            self.parents.append(type(node).__name__)
            for idx, item in enumerate(node.elts):
                if idx:
                    self.write(', ')
                    self.parents.append(type(node).__name__)
                self.visit(item, node)
            self.write(right)
            self.parents.append(type(node).__name__)

        return visit

    visit_List = sequence_visit('[', ']')
    visit_Set = sequence_visit('{', '}')
    del sequence_visit

    def visit_Dict(self, node):
        self.write('{')
        self.parents.append(type(node).__name__)
        for idx, (key, value) in enumerate(zip(node.keys, node.values)):
            if idx:
                self.write(', ')
                self.parents.append(type(node).__name__)
            if key!=None:
                self.visit(key, node)
                self.write(': ')
                self.parents.append(type(node).__name__)
                self.visit(value, node)
            elif key==None:
                self.write('**')
                self.parents.append(type(node).__name__)
                self.visit(value, node)
        self.write('}')
        self.parents.append(type(node).__name__)

    def visit_BinOp(self, node):
        self.visit(node.left, node)
        # self.write('%s' % BINOP_SYMBOLS[type(node.op)])
        self.parents.append(type(node).__name__)
        self.visit(node.right, node)

    def visit_BoolOp(self, node):
        self.write('(')
        self.parents.append(type(node).__name__)
        for idx, value in enumerate(node.values):
            if idx:
                self.write(' %s ' % BOOLOP_SYMBOLS[type(node.op)])
                self.parents.append(type(node).__name__)
            self.visit(value, node)
        self.write(')')
        self.parents.append(type(node).__name__)

    def visit_Compare(self, node):
        self.write('(')
        self.parents.append(type(node).__name__)
        self.visit(node.left, node)
        for op, right in zip(node.ops, node.comparators):
            self.write(' %s ' % CMPOP_SYMBOLS[type(op)])
            self.parents.append(type(node).__name__)
            self.visit(right, node)
        self.write(')')
        self.parents.append(type(node).__name__)

    def visit_UnaryOp(self, node):
        self.write('(')
        self.parents.append(type(node).__name__)
        op = UNARYOP_SYMBOLS[type(node.op)]
        self.write(op)
        self.parents.append(type(node).__name__)
        if op == 'not':
            self.write(' ')
            self.parents.append(type(node).__name__)
        self.visit(node.operand, node)
        self.write(')')
        self.parents.append(type(node).__name__)

    def visit_Subscript(self, node):
        self.visit(node.value, node)
        self.write('[')
        self.parents.append(type(node).__name__)
        self.visit(node.slice, node)
        self.write(']')
        self.parents.append(type(node).__name__)

    def visit_Slice(self, node):
        if node.lower is not None:
            self.visit(node.lower, node)
        self.write(':')
        self.parents.append(type(node).__name__)
        if node.upper is not None:
            self.visit(node.upper, node)
        if node.step is not None:
            self.write(':')
            self.parents.append(type(node).__name__)
            if not (isinstance(node.step, Name) and node.step.id == 'None'):
                self.visit(node.step, node)

    def visit_ExtSlice(self, node):
        for idx, item in enumerate(node.dims):
            if idx:
                self.write(', ')
                self.parents.append(type(node).__name__)
            self.visit(item, node)

    def visit_Yield(self, node):
        self.write('yield ')
        self.parents.append(type(node).__name__)
        if node.value:
            self.visit(node.value, node)

    def visit_Lambda(self, node):
        self.write('lambda ')
        self.parents.append(type(node).__name__)
        self.signature(node.args)
        self.write(': ')
        self.parents.append(type(node).__name__)
        self.visit(node.body, node)

    def visit_Ellipsis(self, node):
        self.write('Ellipsis')
        self.parents.append(type(node).__name__)

    def generator_visit(left, right):
        def visit(self, node):
            self.write(left)
            self.parents.append(type(node).__name__)
            self.visit(node.elt, node)
            for comprehension in node.generators:
                self.visit(comprehension, node)
            self.write(right)
            self.parents.append(type(node).__name__)

        return visit

    visit_ListComp = generator_visit('[', ']')
    visit_GeneratorExp = generator_visit('(', ')')
    visit_SetComp = generator_visit('{', '}')
    del generator_visit

    def visit_DictComp(self, node):
        self.write('{')
        self.parents.append(type(node).__name__)
        self.visit(node.key, node)
        self.write(': ')
        self.parents.append(type(node).__name__)
        self.visit(node.value, node)
        for comprehension in node.generators:
            self.visit(comprehension, node)
        self.write('}')
        self.parents.append(type(node).__name__)

    def visit_IfExp(self, node):
        self.visit(node.body, node)
        self.write(' if ')
        self.parents.append(type(node).__name__)
        self.visit(node.test, node)
        self.write(' else ')
        self.parents.append(type(node).__name__)
        self.visit(node.orelse, node)

    def visit_Starred(self, node):
        self.write('*')
        self.parents.append(type(node).__name__)
        self.visit(node.value, node)

    def visit_Repr(self, node):
        # XXX: python 2.6 only
        self.write('`')
        self.parents.append(type(node).__name__)
        self.visit(node.value, node)
        self.write('`')
        self.parents.append(type(node).__name__)

    # Helper Nodes

    def visit_alias(self, node):
        self.write(node.name)
        self.parents.append(type(node).__name__)
        if node.asname is not None:
            self.write(' as ' + node.asname)
            self.parents.append(type(node).__name__)

    def visit_comprehension(self, node):
        self.write(' for ')
        self.parents.append(type(node).__name__)
        self.visit(node.target, node)
        self.write(' in ')
        self.parents.append(type(node).__name__)
        self.visit(node.iter, node)
        if node.ifs:
            for if_ in node.ifs:
                self.write(' if ')
                self.parents.append(type(node).__name__)
                self.visit(if_, node)

    def visit_ExceptHandler(self, node):
        self.newline(node)
        self.write('except')
        self.parents.append(type(node).__name__)
        if node.type is not None:
            self.write(' ')
            self.parents.append(type(node).__name__)
            self.visit(node.type, node)
            if node.name is not None:
                self.write(', ')
                self.parents.append(type(node).__name__)
                self.write(node.name)
                self.parents.append(type(node).__name__)
        self.write(':')
        self.parents.append(type(node).__name__)
        self.body(node.body, node)

    # def visit_exceptHandler(self, node):
    #     self.newline(node)
    #     self.write('except')
    # self.parents.append(type(node).__name__)
    #     if node.type is not None:
    #         self.write(' ')
    # self.parents.append(type(node).__name__)
    #         self.visit(node.type, node)
    #         if node.name is not None:
    # #             self.write(', ')
    # self.parents.append(type(node).__name__)
    #             self.visit(node.name, node)
    #     self.write(':')
    # self.parents.append(type(node).__name__)
    #     self.body(node.body, node)