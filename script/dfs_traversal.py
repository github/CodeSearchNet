
from dpu_utils.utils import RichPath
from src.utils import my_ast
from src.utils.codegen import *
import subprocess
import os

path = 'resources/data/python/final/jsonl/valid_old/python_valid_0.jsonl.gz'
s_path = 'resources/data/python/final/jsonl/valid/python_valid_0_updated.jsonl.gz'

a = RichPath.create(path)
s = RichPath.create(s_path)

print('started')
b = list(a.read_as_jsonl())


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
        return an.result
    else:
        return []
#

templist = []
for idx, sample in enumerate(b):
    print("sample {} in progress".format(idx))
#    print(sample['code'])
    if idx==3282:
        print(sample['code'])

    tokenization = convert_code_to_tokens(sample['code'])
    if tokenization == []:
        templist.append(idx)
    else:
        b[idx]['code_tokens'] = tokenization
    # tree = my_ast.parse(sample['code'])
    # an = SourceGenerator('    ')
    # an.visit(tree)
    # b[idx]['code_tokens'] = an.result

s.save_as_compressed_file(b)
print('finished', templist, len(templist), tokenization)


# a = [3282, 10821, 15646, 15806, 15868, 15907, 15908, 15909, 15912, 15913, 15915, 15926, 16107, 16255, 16259, 16261, 16337, 16373, 16374, 16378, 16379, 16389, 16390, 16392, 16907, 16966, 16971, 17139, 17179, 21304, 21305]
#b  = [10821, 21304]


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
