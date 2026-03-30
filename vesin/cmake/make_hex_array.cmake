# make_includeable.cmake
# Converts a source file into a C byte array for embedding in C++ source.

if (NOT CMAKE_SCRIPT_MODE_FILE OR CMAKE_SCRIPT_MODE_DIRECTORY)
    message(FATAL_ERROR "This script is intended to be used with 'cmake -P' and should not be included in a CMakeLists.txt")
endif()

if(NOT EXISTS "${INPUT_FILE}")
    message(FATAL_ERROR "Input file '${INPUT_FILE}' does not exist")
endif()

# Read file as text so we can optionally expand supported includes
file(READ "${INPUT_FILE}" content)

# Expand a local include of "vesin.h" by inlining the project's include/vesin.h
# This allows CUDA sources to use #include "vesin.h" and share definitions
# with the C++ code. We only support that single include per instructions.
string(FIND "${content}" "#include \"vesin.h\"" include_pos)
if(NOT include_pos EQUAL -1)
    # Compute project root: parent directory of the INPUT_FILE's directory
    get_filename_component(input_dir "${INPUT_FILE}" DIRECTORY)
    get_filename_component(project_root "${input_dir}" DIRECTORY)
    set(vesin_header_path "${project_root}/include/vesin.h")

    file(READ "${vesin_header_path}" vesin_header_content)
    string(REPLACE "#include <stddef.h>" "" vesin_header_content "${vesin_header_content}")
    string(REPLACE "#include <stdint.h>" "" vesin_header_content "${vesin_header_content}")

    # Replace all occurrences of the include directive with the header content
    string(REPLACE "#include \"vesin.h\"" "${vesin_header_content}" content "${content}")
endif()

# Write the (possibly preprocessed) content to a temporary file and read as hex
file(WRITE "${OUTPUT_FILE}.tmp" "${content}")
file(READ "${OUTPUT_FILE}.tmp" hex HEX)

# Get base name for variable name
get_filename_component(basename "${INPUT_FILE}" NAME)
string(REGEX REPLACE "[^a-zA-Z0-9]" "_" var_name "${basename}")

# Build list of "0xHH" byte values
string(LENGTH "${hex}" hex_len)
math(EXPR byte_count "${hex_len} / 2")
set(hex_list "")
if(byte_count GREATER 0)
    math(EXPR last_idx "${byte_count} - 1")
    foreach(idx RANGE 0 ${last_idx})
        math(EXPR pos "${idx} * 2")
        string(SUBSTRING "${hex}" ${pos} 2 byte)
        string(TOUPPER "${byte}" byte_upper)
        list(APPEND hex_list "0x${byte_upper}")
    endforeach()
endif()

# Format as C initializer, ~13 bytes per line
set(output "")
set(line "")
set(line_count 0)
foreach(byte IN LISTS hex_list)
    if(line STREQUAL "")
        set(line "${byte}")
    else()
        set(line "${line}, ${byte}")
    endif()
    math(EXPR line_count "${line_count} + 1")
    math(EXPR mod "${line_count} % 13")
    if(mod EQUAL 0)
        set(output "${output}${line},\n")
        set(line "")
    endif()
endforeach()

if(NOT line STREQUAL "")
    # There is a partial line (no trailing comma yet). Append the terminator
    set(output "${output}${line}, 0x00\n")
else()
    # Last flush already added a trailing comma and newline. Put the terminator
    # on its own indented line so we don't produce an empty element.
    set(output "${output}0x00\n")
endif()

# Write output
get_filename_component(base_name_we "${INPUT_FILE}" NAME_WE)
file(WRITE "${OUTPUT_FILE}" "/* Generated from ${basename} by make_includeable.cmake. Do not edit. */\n")
file(APPEND "${OUTPUT_FILE}" "${output}\n")
