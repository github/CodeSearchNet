from typing import List, Dict, Any

from parsers.language_parser import LanguageParser, match_from_span, tokenize_code, traverse_type
from parsers.commentutils import strip_c_style_comment_delimiters, get_docstring_summary


class PhpParser(LanguageParser):

    FILTER_PATHS = ('test', 'tests')

    BLACKLISTED_FUNCTION_NAMES = {'__construct', '__destruct', '__call', '__callStatic',
                                  '__get', '__set', '__isset', '__unset',
                                  '__sleep', '__wakeup', '__toString', '__invoke',
                                  '__set_state', '__clone', '__debugInfo', '__serialize',
                                  '__unserialize'}

    @staticmethod
    def get_docstring(trait_node, blob: str, idx: int) -> str:
        docstring = ''
        if idx - 1 >= 0 and trait_node.children[idx-1].type == 'comment':
            docstring = match_from_span(trait_node.children[idx-1], blob)
            docstring = strip_c_style_comment_delimiters(docstring)
        return docstring


    @staticmethod
    def get_declarations(declaration_node, blob: str, node_type: str) -> List[Dict[str, Any]]:
        declarations = []
        for idx, child in enumerate(declaration_node.children):
            if child.type == 'name':
                declaration_name = match_from_span(child, blob)
            elif child.type == 'method_declaration':
                docstring = PhpParser.get_docstring(declaration_node, blob, idx)
                docstring_summary = get_docstring_summary(docstring)
                function_nodes = []
                traverse_type(child, function_nodes, 'function_definition')
                if function_nodes:
                    function_node = function_nodes[0]
                    metadata = PhpParser.get_function_metadata(function_node, blob)

                    if metadata['identifier'] in PhpParser.BLACKLISTED_FUNCTION_NAMES:
                        continue

                    declarations.append({
                        'type': node_type,
                        'identifier': '{}.{}'.format(declaration_name, metadata['identifier']),
                        'parameters': metadata['parameters'],
                        'function': match_from_span(child, blob),
                        'function_tokens': tokenize_code(child, blob),
                        'docstring': docstring,
                        'docstring_summary': docstring_summary,
                        'start_point': function_node.start_point,
                        'end_point': function_node.end_point
                    })
        return declarations


    @staticmethod
    def get_definition(tree, blob: str) -> List[Dict[str, Any]]:
        trait_declarations = [child for child in tree.root_node.children if child.type == 'trait_declaration']
        class_declarations = [child for child in tree.root_node.children if child.type == 'class_declaration']
        definitions = []
        for trait_declaration in trait_declarations:
            definitions.extend(PhpParser.get_declarations(trait_declaration, blob, trait_declaration.type))
        for class_declaration in class_declarations:
            definitions.extend(PhpParser.get_declarations(class_declaration, blob, class_declaration.type))
        return definitions


    @staticmethod
    def get_function_metadata(function_node, blob: str) -> Dict[str, str]:
        metadata = {
            'identifier': '',
            'parameters': '',
        }
        metadata['identifier'] = match_from_span(function_node.children[1], blob)
        metadata['parameters'] = match_from_span(function_node.children[2], blob)
        return metadata
