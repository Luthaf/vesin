#!/usr/bin/env python
import os

HERE = os.path.dirname(os.path.realpath(__file__))
ALREADY_SEEN = set()


def find_file(path):
    candidates = [
        os.path.join(HERE, "vesin", "src", path),
        os.path.join(HERE, "vesin", "include", path),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise RuntimeError(f"unable to find file for {path}")


def include_path(line):
    assert "#include" in line

    _, path = line.split("#include")
    path = path.strip()
    if path.startswith('"'):
        return path[1:-1]
    else:
        return ""


def merge_files(path, output):
    path = find_file(path)

    if path in ALREADY_SEEN:
        return
    else:
        ALREADY_SEEN.add(path)

    if path.endswith("include/vesin.h"):
        output.write('#include "vesin.h"\n')
        return

    with open(path) as fd:
        for line in fd:
            if "#include" in line:
                new_path = include_path(line)
                if new_path != "":
                    merge_files(new_path, output)
                else:
                    output.write(line)
            else:
                output.write(line)


if __name__ == "__main__":
    with open("vesin-single-build.cpp", "w") as output:
        merge_files("cpu_cell_list.cpp", output)
        merge_files("vesin.cpp", output)

    print("created single build file 'vesin-single-build.cpp'")
