from typing import Dict, Iterable, Optional, Iterator, Any, List

from parsers.language_parser import LanguageParser, match_from_span, tokenize_code, traverse_type
from parsers.commentutils import get_docstring_summary


class PythonParser(LanguageParser):

    FILTER_PATHS = ('test',)
    STOPWORDS = ()

    # Get function calls
    @staticmethod
    def get_context(tree, blob):
        def _get_import_from(import_from_statement, blob):
            context = {}
            mode = 'from'
            library = ''
            for n in import_from_statement.children:
                if n.type == 'from':
                    mode = 'from'
                elif n.type == 'import':
                    mode = 'import'
                elif n.type == 'dotted_name':
                    if mode == 'from':
                        library = match_from_span(n, blob).strip()
                    elif mode == 'import':
                        if library:
                            context[match_from_span(n, blob).strip().strip(',')] = library
            return context

        def _get_import(import_statement, blob):
            context = []
            for n in import_statement.children:
                if n.type == 'dotted_name':
                    context.append(match_from_span(n, blob).strip())
                if n.type == 'aliased_import':
                    for a in n.children:
                        if a.type == 'dotted_name':
                            context.append(match_from_span(a, blob).strip())
            return context

        import_from_statements = []
        traverse_type(tree.root_node, import_from_statements, 'import_from_statement')

        import_statements = []
        traverse_type(tree.root_node, import_statements, 'import_statement')

        context = []
        context.extend((_get_import_from(i, blob) for i in import_from_statements))
        context.extend((_get_import(i, blob) for i in import_statements))
        return context

    @staticmethod
    def get_calls(tree, blob):
        calls = []
        traverse_type(tree.root_node, calls, 'call')

        def _traverse_calls(node, identifiers):
            if node.type == 'identifier':
                identifiers.append(node)
            if not node.children or node.type == 'argument_list':
                return
            for n in node.children:
                _traverse_calls(n, identifiers)

        results = []
        for call in calls:
            identifiers = []
            _traverse_calls(call, identifiers)

            if identifiers:
                identifier = identifiers[-1]
                argument_lists = [n for n in call.children if n.type == 'argument_list']
                argument_list = ''
                if argument_lists:
                    argument_list = match_from_span(argument_lists[-1], blob)
                results.append({
                    'identifier': match_from_span(identifier, blob),
                    'argument_list': argument_list,
                    'start_point': identifier.start_point,
                    'end_point': identifier.end_point,
                })
        return results

    @staticmethod
    def __get_docstring_node(function_node):
        docstring_node = [node for node in function_node.children if
                          node.type == 'expression_statement' and node.children[0].type == 'string']
        if len(docstring_node) > 0:
            return docstring_node[0].children[0]
        return None

    @staticmethod
    def get_docstring(docstring_node, blob: str) -> str:
        docstring = ''
        if docstring_node is not None:
            docstring = match_from_span(docstring_node, blob)
            docstring = docstring.strip().strip('"').strip("'")
        return docstring

    @staticmethod
    def get_function_metadata(function_node, blob: str) -> Dict[str, str]:
        metadata = {
            'identifier': '',
            'parameters': '',
            'return_statement': ''
        }
        is_header = False
        for child in function_node.children:
            if is_header:
                if child.type == 'identifier':
                    metadata['identifier'] = match_from_span(child, blob)
                elif child.type == 'parameters':
                    metadata['parameters'] = match_from_span(child, blob)
            if child.type == 'def':
                is_header = True
            elif child.type == ':':
                is_header = False
            elif child.type == 'return_statement':
                metadata['return_statement'] = match_from_span(child, blob)
        return metadata

    @staticmethod
    def get_class_metadata(class_node, blob: str) -> Dict[str, str]:
        metadata = {
            'identifier': '',
            'argument_list': '',
        }
        is_header = False
        for child in class_node.children:
            if is_header:
                if child.type == 'identifier':
                    metadata['identifier'] = match_from_span(child, blob)
                elif child.type == 'argument_list':
                    metadata['argument_list'] = match_from_span(child, blob)
            if child.type == 'class':
                is_header = True
            elif child.type == ':':
                break
        return metadata

    @staticmethod
    def is_function_empty(function_node) -> bool:
        seen_header_end = False
        for child in function_node.children:
            if seen_header_end and (child.type=='pass_statement' or child.type=='raise_statement'):
                return True
            elif seen_header_end:
                return False

            if child.type == ':':
                seen_header_end = True
        return False

    @staticmethod
    def __process_functions(functions: Iterable, blob: str, func_identifier_scope: Optional[str]=None) -> Iterator[Dict[str, Any]]:
        for function_node in functions:
            if PythonParser.is_function_empty(function_node):
                continue
            function_metadata = PythonParser.get_function_metadata(function_node, blob)
            if func_identifier_scope is not None:
                function_metadata['identifier'] = '{}.{}'.format(func_identifier_scope,
                                                                 function_metadata['identifier'])
                if function_metadata['identifier'].startswith('__') and function_metadata['identifier'].endswith('__'):
                    continue  # Blacklist built-in functions
            docstring_node = PythonParser.__get_docstring_node(function_node)
            function_metadata['docstring'] = PythonParser.get_docstring(docstring_node, blob)
            function_metadata['docstring_summary'] = get_docstring_summary(function_metadata['docstring'])
            function_metadata['function'] = match_from_span(function_node, blob)
            function_metadata['function_tokens'] = tokenize_code(function_node, blob, {docstring_node})
            function_metadata['start_point'] = function_node.start_point
            function_metadata['end_point'] = function_node.end_point

            yield function_metadata

    @staticmethod
    def get_function_definitions(node):
        for child in node.children:
            if child.type == 'function_definition':
                yield child
            elif child.type == 'decorated_definition':
                for c in child.children:
                    if c.type == 'function_definition':
                        yield c

    @staticmethod
    def get_definition(tree, blob: str) -> List[Dict[str, Any]]:
        functions = PythonParser.get_function_definitions(tree.root_node)
        classes = (node for node in tree.root_node.children if node.type == 'class_definition')

        definitions = list(PythonParser.__process_functions(functions, blob))

        for _class in classes:
            class_metadata = PythonParser.get_class_metadata(_class, blob)
            docstring_node = PythonParser.__get_docstring_node(_class)
            class_metadata['docstring'] = PythonParser.get_docstring(docstring_node, blob)
            class_metadata['docstring_summary'] = get_docstring_summary(class_metadata['docstring'])
            class_metadata['function'] = ''
            class_metadata['function_tokens'] = []
            class_metadata['start_point'] = _class.start_point
            class_metadata['end_point'] = _class.end_point
            definitions.append(class_metadata)

            functions = PythonParser.get_function_definitions(_class)
            definitions.extend(PythonParser.__process_functions(functions, blob, class_metadata['identifier']))

        return definitions
