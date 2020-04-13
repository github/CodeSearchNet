
unicode = lambda s: str(s)
import ast
from pprint import pprint
import pandas as pd

def create_tree_without_parents(code):
    global c, d
    tree = ast.parse(code)

    json_tree = []

    def gen_identifier(identifier, node_type='identifier'):
        pos = len(json_tree)
        json_node = {}
        json_tree.append(json_node)
        json_node['type'] = node_type
        json_node['value'] = identifier
        return pos

    def traverse_list(l, node_type='list'):
        pos = len(json_tree)
        json_node = {}
        json_tree.append(json_node)
        json_node['type'] = node_type
        children = []
        for item in l:
            children.append(traverse(item))
        if (len(children) != 0):
            json_node['children'] = children
        return pos

    def traverse(node):
        pos = len(json_tree)
        json_node = {}
        json_tree.append(json_node)
        json_node['type'] = type(node).__name__
        children = []
        if isinstance(node, ast.Name):
            json_node['value'] = node.id
        elif isinstance(node, ast.Num):
            json_node['value'] = unicode(node.n)
        elif isinstance(node, ast.Str):
            json_node['value'] = node.s
        elif isinstance(node, ast.alias):
            json_node['value'] = unicode(node.name)
            if node.asname:
                children.append(gen_identifier(node.asname))
        elif isinstance(node, ast.FunctionDef):
            json_node['value'] = unicode(node.name)
        elif isinstance(node, ast.ClassDef):
            json_node['value'] = unicode(node.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                json_node['value'] = unicode(node.module)
        elif isinstance(node, ast.Global):
            for n in node.names:
                children.append(gen_identifier(n))
        elif isinstance(node, ast.keyword):
            json_node['value'] = unicode(node.arg)

        # Process children.
        if isinstance(node, ast.For):
            children.append(traverse(node.target))
            children.append(traverse(node.iter))
            children.append(traverse_list(node.body, 'body'))
            if node.orelse:
                children.append(traverse_list(node.orelse, 'orelse'))
        elif isinstance(node, ast.If) or isinstance(node, ast.While):
            children.append(traverse(node.test))
            children.append(traverse_list(node.body, 'body'))
            if node.orelse:
                children.append(traverse_list(node.orelse, 'orelse'))
        elif isinstance(node, ast.With):
            children.append(traverse(node.context_expr))
            if node.optional_vars:
                children.append(traverse(node.optional_vars))
            children.append(traverse_list(node.body, 'body'))
        elif isinstance(node, ast.Try):
            children.append(traverse_list(node.body, 'body'))
            children.append(traverse_list(node.handlers, 'handlers'))
            if node.orelse:
                children.append(traverse_list(node.orelse, 'orelse'))
            if node.finalbody:
                children.append(traverse_list(node.finalbody, 'finalbody'))
        elif isinstance(node, ast.arguments):
            children.append(traverse_list(node.args, 'args'))
            children.append(traverse_list(node.defaults, 'defaults'))
            if node.vararg:
                children.append(gen_identifier(node.vararg, 'vararg'))
            if node.kwarg:
                children.append(gen_identifier(node.kwarg, 'kwarg'))
        elif isinstance(node, ast.ExceptHandler):
            if node.type:
                children.append(traverse_list([node.type], 'type'))
            if node.name:
                children.append(traverse_list([node.name], 'name'))
            children.append(traverse_list(node.body, 'body'))
        elif isinstance(node, ast.ClassDef):
            children.append(traverse_list(node.bases, 'bases'))
            children.append(traverse_list(node.body, 'body'))
            children.append(traverse_list(node.decorator_list, 'decorator_list'))
        elif isinstance(node, ast.FunctionDef):
            children.append(traverse(node.args))
            children.append(traverse_list(node.body, 'body'))
            children.append(traverse_list(node.decorator_list, 'decorator_list'))
        else:
            # Default handling: iterate over children.
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.expr_context) or isinstance(child, ast.operator) or isinstance(child,
                                                                                                        ast.boolop) or isinstance(
                        child, ast.unaryop) or isinstance(child, ast.cmpop):
                    # Directly include expr_context, and operators into the type instead of creating a child.
                    json_node['type'] = json_node['type'] + type(child).__name__
                else:
                    children.append(traverse(child))

        if isinstance(node, ast.Attribute):
            children.append(gen_identifier(node.attr, 'attr'))

        if (len(children) != 0):
            json_node['children'] = children
        return pos

    traverse(tree)
    return json_tree


def get_docstring(node, clean=True):
    """
    Return the docstring for the given node or None if no docstring can
    be found.  If the node provided does not have docstrings a TypeError
    will be raised.

    If *clean* is `True`, all tabs are expanded to spaces and any whitespace
    that can be uniformly removed from the second line onwards is removed.
    """
    if not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
        raise TypeError("%r can't have docstrings" % node.__class__.__name__)
    if not(node.body and isinstance(node.body[0], ast.Expr)):
        return None
    node = node.body[0].value
    if isinstance(node, ast.Str):
        text = node.s
    # elif isinstance(node, Constant) and isinstance(node.value, str):
    #     text = node.value
    else:
        return None
    if clean:
        import inspect
        text = inspect.cleandoc(text)
    return text


def dfs_traversal_with_parents(tree):
    global c, d

    docstring = ''
    json_tree = []

    def gen_identifier(identifier, node_type='identifier', parent=None):
        global docstring
        pos = len(json_tree)
        json_node = {}
        json_tree.append(json_node)
        json_node['type'] = node_type
        json_node['value'] = identifier

        if parent:
            if hasattr(parent, 'ctx'):
                json_node['parent'] = type(parent).__name__+ type(parent.ctx).__name__
            else:
                json_node['parent'] = type(parent).__name__
        else:
            json_node['parent'] = None
        return pos

    def traverse_list(l, node_type='list', parent=None):
        pos = len(json_tree)
        json_node = {}
        json_tree.append(json_node)
        json_node['type'] = node_type
        if parent:
            if hasattr(parent, 'ctx'):
                json_node['parent'] = type(parent).__name__ + type(parent.ctx).__name__
            else:
                json_node['parent'] = type(parent).__name__
        else:
            json_node['parent'] = None
        children = []
        for item in l:
            if item:
                children.append(traverse(item, node_type))
        if (len(children) != 0):
            json_node['children'] = children
        return pos

    def traverse(node, parent=None):
        global docstring
        pos = len(json_tree)
        if not (isinstance(node, ast.Str) and docstring == node.s):
            json_node = {}
            json_tree.append(json_node)
            json_node['type'] = type(node).__name__
            if parent:
                if type(parent) == str:
                    json_node['parent'] = parent
                elif hasattr(parent, 'ctx'):
                    json_node['parent'] = type(parent).__name__ + type(parent.ctx).__name__
                else:
                    json_node['parent'] = type(parent).__name__
            else:
                json_node['parent'] = None
        children = []
        if isinstance(node, ast.Name):
            json_node['value'] = node.id
        elif isinstance(node, ast.Num):
            json_node['value'] = unicode(node.n)
        elif isinstance(node, ast.Str):
            if docstring != node.s:
                json_node['value'] = node.s
        elif isinstance(node, ast.alias):
            json_node['value'] = unicode(node.name)
            if node.asname:
                json_node['value'] = unicode(node.name) + " as " + str(node.asname)
                # children.append(gen_identifier(node.asname, 'asname', node))
        elif isinstance(node, ast.FunctionDef):
            docstring = get_docstring(node, clean=False)
            json_node['value'] = unicode(node.name)
        elif isinstance(node, ast.ClassDef):
            json_node['value'] = unicode(node.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                json_node['value'] = unicode(node.module)
            # if node.names:
            #     children.append(traverse_list(node.names, 'imports', node))

        elif isinstance(node, ast.Global):
            for n in node.names:
                children.append(gen_identifier(n, 'name', node))
        elif isinstance(node, ast.keyword):
            json_node['value'] = unicode(node.arg)
        elif isinstance(node, ast.arg):
            json_node['value'] = unicode(node.arg)

        # Process children.
        if isinstance(node, ast.For):
            children.append(traverse(node.target, node))
            children.append(traverse(node.iter, node))
            children.append(traverse_list(node.body, 'body', node))
            if node.orelse:
                children.append(traverse_list(node.orelse, 'orelse', node))
        elif isinstance(node, ast.If) or isinstance(node, ast.While):
            children.append(traverse(node.test, node))
            children.append(traverse_list(node.body, 'body', node))
            if node.orelse:
                children.append(traverse_list(node.orelse, 'orelse', node))
        elif isinstance(node, ast.With):
            for item in node.items:
                children.append(traverse(item.context_expr, node))
                if item.optional_vars:
                    children.append(traverse(item.optional_vars, node))
            children.append(traverse_list(node.body, 'body', node))
        elif isinstance(node, ast.Try):
            children.append(traverse_list(node.body, 'body', node))
            children.append(traverse_list(node.handlers, 'handlers', node))
            if node.orelse:
                children.append(traverse_list(node.orelse, 'orelse', node))
            if node.finalbody:
                children.append(traverse_list(node.finalbody, 'finalbody', node))
        elif isinstance(node, ast.arguments):
            if node.args:
                children.append(traverse_list(node.args, 'args', node))
            if node.defaults:
                children.append(traverse_list(node.defaults, 'defaults', node))
            if node.vararg:
                children.append(gen_identifier(node.vararg.arg, 'vararg', node))
            if node.kwarg:
                children.append(gen_identifier(node.kwarg.arg, 'kwarg', node))
            if node.kwonlyargs:
                children.append(traverse_list(node.kwonlyargs, 'kwonlyargs', node))
            if node.kw_defaults:
                children.append(traverse_list(node.kw_defaults, 'kw_defaults', node))

        elif isinstance(node, ast.ExceptHandler):
            if node.type:
                children.append(traverse(node.type))
            # if node.name:
            #         children.append(traverse(node.name))
            children.append(traverse_list(node.body, 'body', node))
        elif isinstance(node, ast.ClassDef):
            children.append(traverse_list(node.bases, 'bases', node))
            children.append(traverse_list(node.body, 'body', node))
            children.append(traverse_list(node.decorator_list, 'decorator_list', node))
        elif isinstance(node, ast.FunctionDef):
            children.append(traverse(node.args, node))
            children.append(traverse_list(node.body, 'body', node))
            if node.decorator_list:
                children.append(traverse_list(node.decorator_list, 'decorator_list', node))
        else:
            # Default handling: iterate over children.
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.expr_context) or isinstance(child, ast.operator) or isinstance(child,
                                                                                                        ast.boolop) or isinstance(
                        child, ast.unaryop) or isinstance(child, ast.cmpop):
                    # Directly include expr_context, and operators into the type instead of creating a child.
                    json_node['type'] = json_node['type'] + type(child).__name__
                else:
                    children.append(traverse(child, node))

        if isinstance(node, ast.Attribute):
            children.append(gen_identifier(node.attr, 'attribute', node))

        if (len(children) != 0):
            json_node['children'] = children
        return pos

    traverse(tree)

    dfs_list = []
    parent_dfs = []
    for node in json_tree:
        parent_dfs.append(node['parent'])
        dfs_list.append(node['type'])
        value = node.get('value', None)
        if value:
            dfs_list.append(value)
            parent_dfs.append(node['type'])

    # df = pd.DataFrame([dfs_list, parent_dfs])
    # print(df.T)

    # pprint(json_tree)

    return dfs_list, parent_dfs
    # return json_tree

