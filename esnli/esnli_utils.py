import re
import string

def extract_marked_token(sent):
    sent = sent.strip()
    # print(sent)
    token_list = []
    p_highlight = re.compile(r"[\*].*?[\*]")
    p_punct = re.compile("[" + re.escape(string.punctuation) + "]")
    findings = p_highlight.findall(sent)

    for token in findings:
        token = p_punct.sub("", token)
        token_list += [token]

    if len(token_list) == 0:
        return "None"
    else:
        return " ||| ".join(token_list)
