
from dpu_utils.utils import RichPath
from src.utils import my_ast
from src.utils.codegen import *
import subprocess
import pandas as pd
import os

# path = 'resources/data/python/final/jsonl/valid_old/temp_train_10.jsonl.gz'
# # s_path = 'resources/data/python/final/jsonl/valid/temp_valid_10.jsonl.gz'
#
# a = RichPath.create(path)
# s = RichPath.create(s_path)
#
# print('started')
# b = list(a.read_as_jsonl())


count = 0
def convert_code_to_tokens(code):
    global count
    tree =''
    # tree = my_ast.parse(code)

    try:
        tree = my_ast.parse(code)
    except:
        try:
            f = open('temp.py', 'w+')
            f.write(code)
            f.close()
            subprocess.run(['2to3', '-w', 'temp.py'])
            f = open('temp.py', 'r')
            code = f.read()
            # print(code)
            tree = my_ast.parse(code)
            # os.rmdir('temp.py')
        except:
            pass
    if tree!='':
        an = SourceGenerator('    ')
        an.visit(tree)
        return an.result, an.parents
    else:
        return []
#

# templist = []
# for idx, sample in enumerate(b):
#     print("sample {} in progress".format(idx))
# #    print(sample['code'])
#     if idx==3282:
#         print(sample['code'])
#
#     tokenization = convert_code_to_tokens(sample['code'])
#     if tokenization == []:
#         templist.append(idx)
#     else:
#         b[idx]['code_tokens'] = tokenization
#     # tree = my_ast.parse(sample['code'])
#     # an = SourceGenerator('    ')
#     # an.visit(tree)
#     # b[idx]['code_tokens'] = an.result
#
# s.save_as_compressed_file(b)
# print('finished', templist, len(templist), tokenization)

import ast
import sys
import json
# def parse_file(code):
#     global c, d
#     tree = ast.parse(code)
#
#     json_tree = []
#
#     def gen_identifier(identifier, node_type='identifier'):
#         pos = len(json_tree)
#         json_node = {}
#         json_tree.append(json_node)
#         json_node['type'] = node_type
#         json_node['value'] = identifier
#         return pos
#
#     def traverse_list(l, node_type='list'):
#         pos = len(json_tree)
#         json_node = {}
#         json_tree.append(json_node)
#         json_node['type'] = node_type
#         children = []
#         for item in l:
#             children.append(traverse(item))
#         if (len(children) != 0):
#             json_node['children'] = children
#         return pos
#
#     def traverse(node):
#         pos = len(json_tree)
#         json_node = {}
#         json_tree.append(json_node)
#         json_node['type'] = type(node).__name__
#         children = []
#         if isinstance(node, ast.Name):
#             json_node['value'] = node.id
#         elif isinstance(node, ast.Num):
#             json_node['value'] = unicode(node.n)
#         elif isinstance(node, ast.Str):
#             json_node['value'] = node.s.decode('utf-8')
#         elif isinstance(node, ast.alias):
#             json_node['value'] = unicode(node.name)
#             if node.asname:
#                 children.append(gen_identifier(node.asname))
#         elif isinstance(node, ast.FunctionDef):
#             json_node['value'] = unicode(node.name)
#         elif isinstance(node, ast.ClassDef):
#             json_node['value'] = unicode(node.name)
#         elif isinstance(node, ast.ImportFrom):
#             if node.module:
#                 json_node['value'] = unicode(node.module)
#         elif isinstance(node, ast.Global):
#             for n in node.names:
#                 children.append(gen_identifier(n))
#         elif isinstance(node, ast.keyword):
#             json_node['value'] = unicode(node.arg)
#
#         # Process children.
#         if isinstance(node, ast.For):
#             children.append(traverse(node.target))
#             children.append(traverse(node.iter))
#             children.append(traverse_list(node.body, 'body'))
#             if node.orelse:
#                 children.append(traverse_list(node.orelse, 'orelse'))
#         elif isinstance(node, ast.If) or isinstance(node, ast.While):
#             children.append(traverse(node.test))
#             children.append(traverse_list(node.body, 'body'))
#             if node.orelse:
#                 children.append(traverse_list(node.orelse, 'orelse'))
#         elif isinstance(node, ast.With):
#             children.append(traverse(node.context_expr))
#             if node.optional_vars:
#                 children.append(traverse(node.optional_vars))
#             children.append(traverse_list(node.body, 'body'))
#         elif isinstance(node, ast.Try):
#             children.append(traverse_list(node.body, 'body'))
#             children.append(traverse_list(node.handlers, 'handlers'))
#             if node.orelse:
#                 children.append(traverse_list(node.orelse, 'orelse'))
#             if node.finalbody:
#                 children.append(traverse_list(node.finalbody, 'finalbody'))
#         elif isinstance(node, ast.arguments):
#             children.append(traverse_list(node.args, 'args'))
#             children.append(traverse_list(node.defaults, 'defaults'))
#             if node.vararg:
#                 children.append(gen_identifier(node.vararg, 'vararg'))
#             if node.kwarg:
#                 children.append(gen_identifier(node.kwarg, 'kwarg'))
#         elif isinstance(node, ast.ExceptHandler):
#             if node.type:
#                 children.append(traverse_list([node.type], 'type'))
#             if node.name:
#                 children.append(traverse_list([node.name], 'name'))
#             children.append(traverse_list(node.body, 'body'))
#         elif isinstance(node, ast.ClassDef):
#             children.append(traverse_list(node.bases, 'bases'))
#             children.append(traverse_list(node.body, 'body'))
#             children.append(traverse_list(node.decorator_list, 'decorator_list'))
#         elif isinstance(node, ast.FunctionDef):
#             children.append(traverse(node.args))
#             children.append(traverse_list(node.body, 'body'))
#             children.append(traverse_list(node.decorator_list, 'decorator_list'))
#         else:
#             # Default handling: iterate over children.
#             for child in ast.iter_child_nodes(node):
#                 if isinstance(child, ast.expr_context) or isinstance(child, ast.operator) or isinstance(child,
#                                                                                                         ast.boolop) or isinstance(
#                         child, ast.unaryop) or isinstance(child, ast.cmpop):
#                     # Directly include expr_context, and operators into the type instead of creating a child.
#                     json_node['type'] = json_node['type'] + type(child).__name__
#                 else:
#                     children.append(traverse(child))
#
#         if isinstance(node, ast.Attribute):
#             children.append(gen_identifier(node.attr, 'attr'))
#
#         if (len(children) != 0):
#             json_node['children'] = children
#         return pos
#
#     traverse(tree)
#     return json_tree

# def updated_parse_file(code):
#     global c, d
#     tree = ast.parse(code)
#
#     json_tree = []
#
#     def gen_identifier(identifier, node_type='identifier', parent=None):
#         pos = len(json_tree)
#         json_node = {}
#         json_tree.append(json_node)
#         # json_node['type'] = node_type
#         json_node[node_type] = identifier
#         if parent:
#             json_node['parent'] = type(parent).__name__
#         else:
#             json_node['parent'] = None
#         return pos
#
#     def traverse_list(l, node_type='list', parent=None):
#         pos = len(json_tree)
#         json_node = {}
#         json_tree.append(json_node)
#         json_node[node_type] = []
#         if parent:
#             json_node['parent'] = type(parent).__name__
#         else:
#             json_node['parent'] = None
#         children = []
#         for item in l:
#             children.append(traverse(item))
#         if (len(children) != 0):
#             json_node[node_type] = children
#         return pos
#
#     def traverse(node, parent=None):
#         pos = len(json_tree)
#         json_node = {}
#         json_tree.append(json_node)
#         json_node[type(node).__name__] = []
#         if parent:
#             json_node['parent'] = type(parent).__name__
#         else:
#             json_node['parent'] = None
#         children = []
#         if isinstance(node, ast.Name):
#             json_node[type(node).__name__] = node.id
#         elif isinstance(node, ast.Num):
#             json_node[type(node).__name__] = unicode(node.n)
#         elif isinstance(node, ast.Str):
#             json_node[type(node).__name__] = node.s.decode('utf-8')
#         elif isinstance(node, ast.alias):
#             json_node[type(node).__name__] = unicode(node.name)
#             if node.asname:
#                 children.append(gen_identifier(node.asname))
#         elif isinstance(node, ast.FunctionDef):
#             json_node[type(node).__name__] = unicode(node.name)
#         elif isinstance(node, ast.ClassDef):
#             json_node[type(node).__name__] = unicode(node.name)
#         elif isinstance(node, ast.ImportFrom):
#             if node.module:
#                 json_node[type(node).__name__] = unicode(node.module)
#         elif isinstance(node, ast.Global):
#             for n in node.names:
#                 children.append(gen_identifier(n))
#         elif isinstance(node, ast.keyword):
#             json_node[type(node).__name__] = unicode(node.arg)
#
#         # Process children.
#         if isinstance(node, ast.For):
#             children.append(traverse(node.target, node))
#             children.append(traverse(node.iter, node))
#             children.append(traverse_list(node.body, 'body', node))
#             if node.orelse:
#                 children.append(traverse_list(node.orelse, 'orelse', node))
#         elif isinstance(node, ast.If) or isinstance(node, ast.While):
#             children.append(traverse(node.test, node))
#             children.append(traverse_list(node.body, 'body', node))
#             if node.orelse:
#                 children.append(traverse_list(node.orelse, 'orelse', node))
#         elif isinstance(node, ast.With):
#             children.append(traverse(node.context_expr, node))
#             if node.optional_vars:
#                 children.append(traverse(node.optional_vars, node))
#             children.append(traverse_list(node.body, 'body', node))
#         elif isinstance(node, ast.Try):
#             children.append(traverse_list(node.body, 'body', node))
#             children.append(traverse_list(node.handlers, 'handlers', node))
#             if node.orelse:
#                 children.append(traverse_list(node.orelse, 'orelse', node))
#             if node.finalbody:
#                 children.append(traverse_list(node.finalbody, 'finalbody', node))
#         elif isinstance(node, ast.arguments):
#             children.append(traverse_list(node.args, 'args', node))
#             children.append(traverse_list(node.defaults, 'defaults', node))
#             if node.vararg:
#                 children.append(gen_identifier(node.vararg, 'vararg'))
#             if node.kwarg:
#                 children.append(gen_identifier(node.kwarg, 'kwarg'))
#         elif isinstance(node, ast.ExceptHandler):
#             if node.type:
#                 children.append(traverse_list([node.type], 'type', node))
#             if node.name:
#                 children.append(traverse_list([node.name], 'name', node))
#             children.append(traverse_list(node.body, 'body', node))
#         elif isinstance(node, ast.ClassDef):
#             children.append(traverse_list(node.bases, 'bases', node))
#             children.append(traverse_list(node.body, 'body', node))
#             children.append(traverse_list(node.decorator_list, 'decorator_list', node))
#         elif isinstance(node, ast.FunctionDef):
#             children.append(traverse(node.args, node))
#             children.append(traverse_list(node.body, 'body',node))
#             children.append(traverse_list(node.decorator_list, 'decorator_list',node))
#         else:
#             # Default handling: iterate over children.
#             for child in ast.iter_child_nodes(node):
#                 if isinstance(child, ast.expr_context) or isinstance(child, ast.operator) or isinstance(child,
#                                                                                                         ast.boolop) or isinstance(
#                     child, ast.unaryop) or isinstance(child, ast.cmpop):
#                     # Directly include expr_context, and operators into the type instead of creating a child.
#                     json_node[type(node).__name__ + type(child).__name__] = json_node[type(node).__name__]
#                     del json_node[type(node).__name__]
#                 else:
#                     children.append(traverse(child,node))
#
#         if isinstance(node, ast.Attribute):
#             children.append(gen_identifier(node.attr, 'Attr'))
#
#         if (len(children) != 0):
#             if type(node).__name__ not in json_node.keys():
#                 json_node[type(node).__name__ + type(child).__name__] = children
#             else:
#                 json_node[type(node).__name__] = children
#         return pos
#
#     traverse(tree)
#     return json_tree
#     # return json.dumps(json_tree, separators=(',', ':'), ensure_ascii=False)

def parse_file_with_parents(code):
    global c, d
    tree = ast.parse(code)

    json_tree = []

    def gen_identifier(identifier, node_type='identifier', parent=None):
        pos = len(json_tree)
        json_node = {}
        json_tree.append(json_node)
        json_node['type'] = node_type
        json_node['value'] = identifier
        if parent:
            json_node['parent'] = type(parent).__name__
        else:
            json_node['parent'] = None
        return pos

    def traverse_list(l, node_type='list'):
        pos = len(json_tree
        json_node = {}
        json_tree.append(json_node)
        json_node['type'] = node_type
        if parent:
            json_node['parent'] = type(parent).__name__
        else:
            json_node['parent'] = None
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
        if parent:
            json_node['parent'] = type(parent).__name__
        else:
            json_node['parent'] = None
        children = []
        if isinstance(node, ast.Name):
            json_node['value'] = node.id
        elif isinstance(node, ast.Num):
            json_node['value'] = unicode(node.n)
        elif isinstance(node, ast.Str):
            json_node['value'] = node.s.decode('utf-8')
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


# [{'children': [1], 'type': 'Module'},
# {'children': [2, 3], 'type': 'Assign'},
# {'type': 'NameStore', 'value': 'ip'},
# {'children': [4, 7], 'type': 'Call'},
# {'children': [5, 6], 'type': 'AttributeLoad'},
# {'type': 'NameLoad', 'value': 'socket'},
# {'type': 'attr', 'value': 'gethostbyname'},
# {'type': 'NameLoad', 'value': 'host'}]


from pprint import pprint
if __name__=='__main__':
    print('something')

#     code ='''print('something')
# try:
#     a+1
# except IOError:
#     return 1
# else:
#     a+2
# finally:
#     return 2'''

    # code= '''def f(a, b=1, c=2, *d, e, f=3, **g):
    #              pass'''

    code = '''ip = socket.gethostbyname(host)'''
    # code = '''func(a, b=c, *d, **e)'''
    # a, b = convert_code_to_tokens(code)
    # df = pd.DataFrame([a, b])
    # print(df.T)

    result_tree = updated_parse_file(code)

    # print(pd.read_json(result_tree))
    print(result_tree)