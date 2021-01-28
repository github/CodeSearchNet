# Function Parser

> # A community-driven, standalone version of the function-parsing code in this folder, that can be installed as a PyPI package can be found [here](https://github.com/ncoop57/function_parser).
> The code in this codebase is maintained only for fixing issues related with the CodeSearchNet challenge data.

This repository contains various utils to parse GitHub repositories into function definition and docstring pairs. It is based on [tree-sitter](https://github.com/tree-sitter/) to parse code into [ASTs](https://en.wikipedia.org/wiki/Abstract_syntax_tree) and apply heuristics to parse metadata in more details. Currently, it supports 6 languages: Python, Java, Go, Php, Ruby, and Javascript.

It also parses function calls and links them with their definitions for Python.

## Examples

Input library `keras-team/keras` is parsed into list of functions including various metadata (e.g. identifier, docstring, sha, url, etc.). Below is an example output of `Activation` function from `keras` library.
```
{
    'nwo': 'keras-team/keras',
    'sha': '0fc33feb5f4efe3bb823c57a8390f52932a966ab',
    'path': 'keras/layers/core.py',
    'language': 'python',
    'identifier': 'Activation.__init__',
    'parameters': '(self, activation, **kwargs)',
    'argument_list': '',
    'return_statement': '',
    'docstring': '',
    'function': 'def __init__(self, activation, **kwargs):\n        super(Activation, self).__init__(**kwargs)\n        self.supports_masking = True\n        self.activation = activations.get(activation)',
    'url': 'https://github.com/keras-team/keras/blob/0fc33feb5f4efe3bb823c57a8390f52932a966ab/keras/layers/core.py#L294-L297'
}
```

One example of `Activation` in the call sites of `eriklindernoren/Keras-GAN` repository is shown below:
```
{
    'nwo': 'eriklindernoren/Keras-GAN',
    'sha': '44d3320e84ca00071de8a5c0fb4566d10486bb1d',
    'path': 'dcgan/dcgan.py',
    'language': 'python',
    'identifier': 'Activation',
    'argument_list': '("relu")',
    'url': 'https://github.com/eriklindernoren/Keras-GAN/blob/44d3320e84ca00071de8a5c0fb4566d10486bb1d/dcgan/dcgan.py#L61-L61'
}
```

With an edge linking the two urls
```
(
    'https://github.com/eriklindernoren/Keras-GAN/blob/44d3320e84ca00071de8a5c0fb4566d10486bb1d/dcgan/dcgan.py#L61-L61',
    'https://github.com/keras-team/keras/blob/0fc33feb5f4efe3bb823c57a8390f52932a966ab/keras/layers/core.py#L294-L297'
)
```

A [demo notebook](function_parser/demo.ipynb) is also provided for exploration.

## Usages
### To run the notebook on your own:
1. `script/bootstrap` to build docker container
2. `script/server` to run the jupyter notebook server and navigate to `function_parser/demo.ipynb`

### To run the script:
1. `script/bootstrap` to build docker container
2. `script/setup` to download libraries.io data
3. `script/console` to ssh into the container
4. Inside the container, run `python function_parser/process.py --language python --processes 16 '/src/function-parser/data/libraries-1.4.0-2018-12-22/' '/src/function-parser/data/'`
