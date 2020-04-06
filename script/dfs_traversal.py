
from dpu_utils.utils import RichPath
from utils.my_utils import DotDict
from utils import my_ast
from utils.codegen import *
import pandas as pd

path = '../resources/data/python/final/jsonl/train/temp_train_10.jsonl.gz'
s_path = '../resources/data/python/final/jsonl/train/temp_train_10_dfs_preorder.jsonl.gz'

a = RichPath.create(path)
s = RichPath.create(s_path)

b = list(a.read_as_jsonl())

for idx, sample in enumerate(b):
    tree = my_ast.parse(sample['code'])
    an = SourceGenerator('    ')
    an.visit(tree)
    b[idx]['code_tokens'] = an.result

s.save_as_compressed_file(b)

if __name__=='__main__':

    # b = list(map(DotDict, b))
    # b = sorted(b, key=lambda v: len(v.code_tokens))

    # ip = socket.gethostbyname(host)
    # chain = build_request_chain(num_requests, host, request_size)'

    # source = '''[port, request_size] = map(string.atoi, sys.argv[2:])'''
    # source = "d = b+(a-c)"
    # tree = my_ast.parse(b[0].code)
    #
    # an = SourceGenerator('    ')
    # an.visit(tree)
    # default_tokens = ['b', "+", "(", "a", "-", "c", ")"]
    # default_tokens = b[0].code_tokens
    # df = pd.DataFrame([c.code_tokens for c in b])
    # print(df.T[8])
    print(b[9]['code'])
    print(b[9]['code_tokens'])
    # print(' '.join(b[0].code_tokens))
    #
    # f = open('temp.py', 'w+')
    #
    # f.write(''.join(map(str, an.result)))
    # print(an.result)
    # print(b[0].code_tokens)