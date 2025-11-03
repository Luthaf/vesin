#!/usr/bin/env python
import os
import tarfile


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


def add_version(output):
    with open(os.path.join(HERE, "vesin", "VERSION")) as fd:
        version = fd.read().strip()

    output.write("// automatically generated \n")
    output.write(f"// vesin version: {version}\n\n")


if __name__ == "__main__":
    with open(os.path.join(HERE, "vesin", "VERSION")) as fd:
        version = fd.read().strip()

    with open("vesin-single-build.cpp", "w") as output:
        add_version(output)
        merge_files("cpu_cell_list.cpp", output)
        merge_files("cuda_stub.cpp", output)
        merge_files("vesin.cpp", output)

    with tarfile.open(f"vesin-single-build-v{version}.tar.gz", "w:gz") as tar:
        tar.add("vesin-single-build.cpp")
        tar.add(os.path.join(HERE, "vesin", "include", "vesin.h"), arcname="vesin.h")

    print(
        f"created 'vesin-single-build.cpp' and 'vesin-single-build-v{version}.tar.gz'"
    )
