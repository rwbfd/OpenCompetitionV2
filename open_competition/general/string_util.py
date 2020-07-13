# coding = 'utf-8'


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def remove_continuous_discrete_prefix(text):
    result = text
    to_removes = ['continuous_', 'discrete_']
    for to_remove in to_removes:
        result = remove_prefix(result, to_remove)
    return result
