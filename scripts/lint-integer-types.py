#!/usr/bin/env python3
"""
Check that all C++ and CUDA code only uses integer types with an explicit size
(``int32_t``, ``uint64_t``, …).

Usage:

    ./scripts/lint-integer-types.py [<path> ...]
"""

import fnmatch
import os
import re
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# directories checked when no explicit path is given on the command line
DEFAULT_PATHS = [
    os.path.join(ROOT, "vesin"),
    os.path.join(ROOT, "vesin-torch"),
]

EXTENSIONS = [".c", ".h", ".cc", ".cpp", ".hpp", ".cu", ".cuh"]

# files excluded from the check, relative to the root of the repository. The
# signature of `main` is imposed by the C++ standard, and `integer.cuh` defines
# the explicit-size integers for CUDA code from the built-in types.
EXCLUDED_FILES = [
    os.path.join("vesin", "tests", "single-cpp", "main.cpp"),
    os.path.join("vesin", "src", "cuda", "integer.cuh"),
]

# built-in integer types, including multi-word ones such as `unsigned long long`
INTEGER_KEYWORDS = ["unsigned", "signed", "short", "int", "long", "char"]

# `char` is only allowed when it is used as text, i.e. through a `char*`
POINTER_RE = re.compile(r"[ \t]*\*")

BUILTIN_RE = re.compile(
    r"\b(?:" + "|".join(INTEGER_KEYWORDS) + r")"
    r"(?:[ \t]+(?:" + "|".join(INTEGER_KEYWORDS) + r"))*\b"
)

# integer typedefs from the standard library which don't have an explicit size
TYPEDEFS_RE = re.compile(
    r"\b(?:std::)?(?:"
    r"ssize_t|ptrdiff_t|intptr_t|uintptr_t|intmax_t|uintmax_t"
    r"|u?int_(?:fast|least)(?:8|16|32|64)_t"
    r")\b"
)

SUGGESTION = "use one of int8_t/int16_t/int32_t/int64_t or their unsigned counterparts"


def strip_comments_and_literals(source):
    """
    Replace all comments, string literals and character literals in ``source``
    with spaces, keeping the position of everything else unchanged.
    """
    output = []
    start = 0
    i = 0
    n = len(source)

    def blank(text):
        # keep the newlines to preserve line numbers
        return "".join("\n" if char == "\n" else " " for char in text)

    while i < n:
        char = source[i]

        if char == "/" and source.startswith("//", i):
            end = source.find("\n", i)
            end = n if end == -1 else end
        elif char == "/" and source.startswith("/*", i):
            end = source.find("*/", i + 2)
            end = n if end == -1 else end + 2
        elif char == "R" and source.startswith('R"', i):
            # raw string literal: R"delimiter(...)delimiter"
            open_paren = source.find("(", i + 2)
            if open_paren == -1:
                i += 1
                continue
            delimiter = source[i + 2 : open_paren]
            end = source.find(")" + delimiter + '"', open_paren)
            end = n if end == -1 else end + len(delimiter) + 2
        elif char == '"' or char == "'":
            end = i + 1
            while end < n:
                if source[end] == "\\":
                    end += 2
                elif source[end] == char:
                    end += 1
                    break
                elif source[end] == "\n":
                    # unterminated literal, don't swallow the rest of the file
                    break
                else:
                    end += 1
        else:
            i += 1
            continue

        output.append(source[start:i])
        output.append(blank(source[i:end]))
        start = end
        i = end

    output.append(source[start:])
    return "".join(output)


def check_file(path):
    """
    Look for forbidden integer types in the file at ``path``, printing an error
    for each of them and returning the number of errors found.
    """
    with open(path, encoding="utf8") as fd:
        source = fd.read()

    code = strip_comments_and_literals(source)

    relative = os.path.relpath(path, ROOT)
    errors = 0
    for i, line in enumerate(code.split("\n")):
        matches = list(BUILTIN_RE.finditer(line)) + list(TYPEDEFS_RE.finditer(line))
        for match in sorted(matches, key=lambda m: m.start()):
            if match.group() == "char" and POINTER_RE.match(line, match.end()):
                continue

            print(
                f"{relative}:{i + 1}:{match.start() + 1}: "
                f"`{match.group()}` does not have an explicit size, {SUGGESTION}"
            )
            errors += 1

    return errors


def ignored_directories():
    """
    Get the directory patterns to skip from the repository ``.gitignore``,
    adding ``.git`` itself since it is not listed in there.
    """
    patterns = [".git"]
    with open(os.path.join(ROOT, ".gitignore"), encoding="utf8") as fd:
        for line in fd:
            line = line.strip()
            if line == "" or line.startswith("#") or line.startswith("!"):
                continue
            patterns.append(line.rstrip("/"))

    return patterns


def collect_files(paths):
    ignored = ignored_directories()

    files = []
    for path in paths:
        if os.path.isfile(path):
            files.append(path)
            continue

        for root, directories, file_names in os.walk(path):
            directories[:] = [
                directory
                for directory in directories
                if not any(fnmatch.fnmatch(directory, p) for p in ignored)
            ]
            for file_name in file_names:
                if os.path.splitext(file_name)[1] in EXTENSIONS:
                    files.append(os.path.join(root, file_name))

    return sorted(f for f in files if os.path.relpath(f, ROOT) not in EXCLUDED_FILES)


if __name__ == "__main__":
    paths = sys.argv[1:] or DEFAULT_PATHS

    for path in paths:
        if not os.path.exists(path):
            print(f"error: no such file or directory: {path}", file=sys.stderr)
            sys.exit(2)

    files = collect_files(paths)

    errors = 0
    for path in files:
        errors += check_file(path)

    if errors != 0:
        print(
            f"\nfound {errors} integer types without an explicit size in "
            f"{len(files)} files"
        )
        sys.exit(1)
