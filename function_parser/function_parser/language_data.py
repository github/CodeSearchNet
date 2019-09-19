from parsers.go_parser import GoParser
from parsers.java_parser import JavaParser
from parsers.javascript_parser import JavascriptParser
from parsers.php_parser import PhpParser
from parsers.python_parser import PythonParser
from parsers.ruby_parser import RubyParser


LANGUAGE_METADATA = {
    'python': {
        'platform': 'pypi',
        'ext': 'py',
        'language_parser': PythonParser
    },
    'java': {
        'platform': 'maven',
        'ext': 'java',
        'language_parser': JavaParser
    },
    'go': {
        'platform': 'go',
        'ext': 'go',
        'language_parser': GoParser
    },
    'javascript': {
        'platform': 'npm',
        'ext': 'js',
        'language_parser': JavascriptParser
    },
    'php': {
        'platform': 'packagist',
        'ext': 'php',
        'language_parser': PhpParser
    },
    'ruby': {
        'platform': 'rubygems',
        'ext': 'rb',
        'language_parser': RubyParser
    }
}
