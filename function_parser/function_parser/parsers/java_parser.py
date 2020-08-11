from typing import List, Dict, Any

from parsers.language_parser import LanguageParser, match_from_span, tokenize_code, traverse_type
from parsers.commentutils import strip_c_style_comment_delimiters, get_docstring_summary


class JavaParser(LanguageParser):

    FILTER_PATHS = ('test', 'tests')

    BLACKLISTED_FUNCTION_NAMES = {'toString', 'hashCode', 'equals', 'finalize', 'notify', 'notifyAll', 'clone'}

    @staticmethod
    def get_definition(tree, blob: str) -> List[Dict[str, Any]]:
        classes = (node for node in tree.root_node.children if node.type == 'class_declaration')

        definitions = []
        for _class in classes:
            class_identifier = match_from_span([child for child in _class.children if child.type == 'identifier'][0], blob).strip()
            for child in (child for child in _class.children if child.type == 'class_body'):
                for idx, node in enumerate(child.children):
                    if node.type == 'method_declaration':
                        if JavaParser.is_method_body_empty(node):
                            continue
                        docstring = ''
                        if idx - 1 >= 0 and child.children[idx-1].type == 'comment':
                            docstring = match_from_span(child.children[idx - 1], blob)
                            docstring = strip_c_style_comment_delimiters(docstring)
                        docstring_summary = get_docstring_summary(docstring)

                        metadata = JavaParser.get_function_metadata(node, blob)
                        if metadata['identifier'] in JavaParser.BLACKLISTED_FUNCTION_NAMES:
                            continue
                        definitions.append({
                            'type': node.type,
                            'identifier': '{}.{}'.format(class_identifier, metadata['identifier']),
                            'parameters': metadata['parameters'],
                            'function': match_from_span(node, blob),
                            'function_tokens': tokenize_code(node, blob),
                            'docstring': docstring,
                            'docstring_summary': docstring_summary,
                            'start_point': node.start_point,
                            'end_point': node.end_point
                        })
        return definitions

    @staticmethod
    def get_class_metadata(class_node, blob: str) -> Dict[str, str]:
        metadata = {
            'identifier': '',
            'argument_list': '',
        }
        is_header = False
        for n in class_node.children:
            if is_header:
                if n.type == 'identifier':
                    metadata['identifier'] = match_from_span(n, blob).strip('(:')
                elif n.type == 'argument_list':
                    metadata['argument_list'] = match_from_span(n, blob)
            if n.type == 'class':
                is_header = True
            elif n.type == ':':
                break
        return metadata

    @staticmethod
    def is_method_body_empty(node):
        for c in node.children:
            if c.type in {'method_body', 'constructor_body'}:
                if c.start_point[0] == c.end_point[0]:
                    return True

    @staticmethod
    def get_function_metadata(function_node, blob: str) -> Dict[str, str]:
        metadata = {
            'identifier': '',
            'parameters': '',
        }

        declarators = []
        traverse_type(function_node, declarators, '{}_declaration'.format(function_node.type.split('_')[0]))
        parameters = []
        for n in declarators[0].children:
            if n.type == 'identifier':
                metadata['identifier'] = match_from_span(n, blob).strip('(')
            elif n.type == 'formal_parameter':
                parameters.append(match_from_span(n, blob))
        metadata['parameters'] = ' '.join(parameters)
        return metadata
