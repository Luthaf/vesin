#!/usr/bin/env python
import os
import subprocess
import tarfile
import tempfile


HERE = os.path.dirname(os.path.realpath(__file__))
ALREADY_SEEN = set()


def find_file(path):
    candidates = [
        os.path.join(HERE, "vesin", "src", path),
        os.path.join(HERE, "vesin", "include", path),
        os.path.join(os.getcwd(), path),
        os.path.join(os.getcwd(), "_deps", "gpulite-src", path),
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
    elif path == "<gpulite/gpulite.hpp>":
        return "gpulite/gpulite.hpp"
    else:
        return ""


def merge_files(path, output):
    if path.startswith("generated/"):
        pass

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

    with tempfile.TemporaryDirectory() as tempdir:
        os.chdir(tempdir)
        subprocess.run(
            ["cmake", os.path.join(HERE, "vesin")],
            check=True,
        )

        with open(os.path.join(HERE, "vesin-single-build.cpp"), "w") as output:
            add_version(output)
            merge_files("cpu_cell_list.cpp", output)
            merge_files("vesin_cuda.cpp", output)
            merge_files("vesin.cpp", output)

        tarpath = os.path.join(HERE, f"vesin-single-build-v{version}.tar.gz")
        with tarfile.open(tarpath, "w:gz") as tar:
            tar.add(
                os.path.join(HERE, "vesin-single-build.cpp"),
                arcname="vesin-single-build.cpp",
            )
            tar.add(
                os.path.join(HERE, "vesin", "include", "vesin.h"),
                arcname="vesin.h",
            )

    print(
        f"created 'vesin-single-build.cpp' and 'vesin-single-build-v{version}.tar.gz'"
    )
