def strip_c_style_comment_delimiters(comment: str) -> str:
    comment_lines = comment.split('\n')
    cleaned_lines = []
    for l in comment_lines:
        l = l.strip()
        if l.endswith('*/'):
            l = l[:-2]
        if l.startswith('*'):
            l = l[1:]
        elif l.startswith('/**'):
            l = l[3:]
        elif l.startswith('//'):
            l = l[2:]
        cleaned_lines.append(l.strip())
    return '\n'.join(cleaned_lines)


def get_docstring_summary(docstring: str) -> str:
    """Get the first lines of the documentation comment up to the empty lines."""
    if '\n\n' in docstring:
        return docstring.split('\n\n')[0]
    elif '@' in docstring:
        return docstring[:docstring.find('@')]  # This usually is the start of a JavaDoc-style @param comment.
    return docstring