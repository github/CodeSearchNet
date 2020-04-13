# from src.dpu_utils.utils import RichPath
from src.utils import my_ast
from src.utils.codegen2 import *
import json_lines
import gzip
import codecs
import json
# import pandas as pd
# from path

def save_jsonl_gz(data, filename):
    with gzip.GzipFile(filename, 'wb') as out_file:
        writer = codecs.getwriter('utf-8')
        for element in data:
            writer(out_file).write(json.dumps(element))
            writer(out_file).write('\n')
count = 0
def convert_code_to_tokens(code):
    global count
    tree = ''
    try:
        tree = my_ast.parse(code)
    except:
        count+=1
    
    if tree=='':
        return []
    else:
        an = SourceGenerator('    ')
        an.visit(tree)
        return an.result


path = 'resources/data/python/final/jsonl/train_old/python_train_0.jsonl.gz'
s_path = 'resources/data/python/final/jsonl/train/python_train_0_updated.jsonl.gz'

# a = json_lines.open(path, 'r')
# s = json_lines.open(path, 'w')
with json_lines.open(path, 'r') as a:
    b = list(a)

# print(b[0]['code'])
# print('started')
# b = list(a.read_as_jsonl())

# #
for idx, sample in enumerate(b):
    global count
    print("sample {} in progress".format(idx))
#    print(sample['code'])
    # print(sample['code'])
    updated_tokens = convert_code_to_tokens(sample['code'])
    b[idx]['code_tokens'] = updated_tokens
    # tree = my_ast.parse(sample['code'])
    # an = SourceGenerator('    ')
    # an.visit(tree)
    # b[idx]['code_tokens'] = an.result

print(count)
# save_jsonl_gz(b, s_path)
print('finished')






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

    # code = '''ip = socket.gethostbyname(host)'''
    # code = '''func(a, b=c, *d, **e)'''
    # print(convert_code_to_tokens(code))

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
    # print(b[9]['code'])
    # print(b[9]['code_tokens'])
    # print(' '.join(b[0].code_tokens))
    #
    # f = open('temp.py', 'w+')
    #
    # f.write(''.join(map(str, an.result)))
    # print(an.result)
    # print(b[0].code_tokens)
