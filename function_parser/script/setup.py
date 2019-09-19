import glob

from tree_sitter import Language

languages = [
    '/src/vendor/tree-sitter-python',
    '/src/vendor/tree-sitter-javascript',
    # '/src/vendor/tree-sitter-typescript/typescript',
    # '/src/vendor/tree-sitter-typescript/tsx',
    '/src/vendor/tree-sitter-go',
    '/src/vendor/tree-sitter-ruby',
    '/src/vendor/tree-sitter-java',
    '/src/vendor/tree-sitter-cpp',
    '/src/vendor/tree-sitter-php',
]

Language.build_library(
    # Store the library in the directory
    '/src/build/py-tree-sitter-languages.so',
    # Include one or more languages
    languages
)
