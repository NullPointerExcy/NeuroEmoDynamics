import re


# English 2nd person → 1st person
PRONOUN_MAP = {
    "you": "i",
    "your": "my",
    "yours": "mine",
    "yourself": "myself",
}

# Common English verb adjustments after 2nd→1st person
VERB_MAP = {
    "are": "am",
}

ALL_MAP = {**PRONOUN_MAP, **VERB_MAP}

SECOND_PERSON_TOKENS = set(PRONOUN_MAP.keys())


def detect_second_person(text):
    tokens = text.lower().split()
    return any(t in SECOND_PERSON_TOKENS for t in tokens)


def transform_perspective(text):
    tokens = text.lower().split()
    was_transformed = False
    result = []
    for token in tokens:
        clean = re.sub(r'[^a-z]', '', token)
        if clean in ALL_MAP:
            replacement = ALL_MAP[clean]
            # Preserve any trailing punctuation
            suffix = token[len(clean):] if len(token) > len(clean) else ""
            result.append(replacement + suffix)
            was_transformed = True
        else:
            result.append(token)
    return " ".join(result), was_transformed
