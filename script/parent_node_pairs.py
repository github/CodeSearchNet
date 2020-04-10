
from dpu_utils.utils import RichPath
from src.utils import my_ast
from src.utils.codegen import *
import subprocess

from parent_node_parse_helpers import dfs_traversal_with_parents
import pandas as pd
import os



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
    if tree!='' and tree != None:
        return dfs_traversal_with_parents(tree)
    else:
        return [], []
#


from pprint import pprint
if __name__=='__main__':
    print('something')

    #[26045, 28475]

    path = '../resources/data/python/final/jsonl/train_old/temp_train_10.jsonl.gz'
    s_path = '../resources/data/python/final/jsonl/train/temp_train_10_dfs_parent.jsonl.gz'

    a = RichPath.create(path)
    s = RichPath.create(s_path)

    print('started')
    b = list(a.read_as_jsonl())

    b = sorted(b, key=lambda v: len(v['code_tokens']))
    templist = []
    c = []
    for idx, sample in enumerate(b):
        print("sample {} in progress".format(idx))
        # print(sample['code'])

        if idx == 19 or sample['sha']=='618d6bff71073c8c93501ab7392c3cc579730f0b':
            print(sample['code'])

        dfs, parent_dfs = convert_code_to_tokens(sample['code'])
        if dfs == [] or parent_dfs==[]:
            templist.append(idx)
        else:
            b[idx]['code_tokens'] = dfs
            b[idx]['parent_dfs'] = parent_dfs
            c.append(b[idx])

    s.save_as_compressed_file(c)
    #     df = pd.DataFrame([dfs, parent_dfs])
    #     print(parent_dfs)
    print('finished', templist, len(templist), len(c))


    # code= '''def f(a, b=1, c=2, *d, e, f=3, **g):
    #              pass'''
    #
    # code = b[2]['code']
    # print(code)
    # code = '''ip = socket.gethostbyname(host)'''
    #
    # code = '''ip = socket.gethostbyname(host)\n[ port , request_size , num_requests , num_conns ] = map (
    # string .atoi , sys . argv [2:]
    # )\nchain = build_request_chain ( num_requests , host , request_size )'''

    # code = '''from foo import bar as b, car as c, dar as d'''
    # print(convert_code_to_tokens(code))

#     code ='''print('something')
# try:
#     a+1
# except IOError:
#     return 1
# else:
#     a+2
# finally:
#     return 2'''




#     # code = '''func(a, b=c, *d, **e)'''
#     # a, b = parse_file_with_parents(code)
#     # df = pd.DataFrame([a, b])
#     # print(df.T)
#
#     result_tree = parse_file_with_parents(code)
    # #
    # # # print(pd.read_json(result_tree))
    # pprint(result_tree)