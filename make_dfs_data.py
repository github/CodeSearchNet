import ast
import json_lines
import json
import gzip
import shutil
import os
import shutil



def main():
    old_dfs_path = os.getcwd() + '/dfs'

    if os.path.exists(old_dfs_path):
        shutil.rmtree(old_dfs_path)
        print('Old dfs files removed. Creating new dfs files.')

    new_dirs = ['./dfs/jsonl/train/', './dfs/jsonl/valid/', './dfs/jsonl/test/']
    for d in new_dirs:
        if not os.path.exists(d):
            os.makedirs(d)

    for subdir, dirs, files in os.walk('.', topdown=True):
        current_path = os.getcwd()

        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith(".gz"):
                dfs_data = []
                with json_lines.open(filepath) as f:
                    for line in iter(f):
                        try:
                            tree = ast.parse(line["original_string"])
                            dfs_tokens = leaf_node_only().visit(tree)
                            line['code_tokens_original'] = line["code_tokens"]
                            line['code_tokens'] = dfs_tokens
                            dfs_data.append(line)
                        except:
                            pass

                with open('output.jsonl', 'w') as outfile:
                    for entry in dfs_data:
                        json.dump(entry, outfile)
                        outfile.write('\n')

                with open('output.jsonl', 'rb') as f_in, gzip.open(current_path + '/dfs' + subdir[1:] + os.sep + file, 'w') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                os.remove('output.jsonl')

    print('New dfs files created successfully.')
#    node_type_only().visit(tree)
#    node_type_and_leaf().visit(tree)
#    leaf_node_only().visit(tree)

class node_type_only(ast.NodeVisitor):
    def generic_visit(self, node):
        return(type(node).__name__)
        ast.NodeVisitor.generic_visit(self, node)

class node_type_and_leaf(ast.NodeVisitor):
    def visit_Name(self, node): return('Name:', node.id)

class leaf_node_only(ast.NodeVisitor):
    def visit_Module(self, node):
        self.names = set()
        self.generic_visit(node)
        return(sorted(self.names))
    def visit_Name(self, node):
        self.names.add(node.id)

from pprint import pprint

if __name__ == "__main__":
    # main()
    # assing,
    #     name 
    #     call
    #         attr
    #             leaf
    # print("something")
    code = '''ip = socket.gethostbyname(host)'''
    tree = ast.parse(code)
    # pprint(tree)
    # print(list(ast.walk(tree)))
    print(node_type_and_leaf().visit(tree))

