# make_includeable.cmake
# Converts a source file into a C byte array for embedding in C++ source.
# Local includes (#include "...") are inlined, looking for files in the
# same directory as the input file and in vesin/include/.

if (NOT CMAKE_SCRIPT_MODE_FILE OR CMAKE_SCRIPT_MODE_DIRECTORY)
    message(FATAL_ERROR "This script is intended to be used with 'cmake -P' and should not be included in a CMakeLists.txt")
endif()

if(NOT EXISTS "${INPUT_FILE}")
    message(FATAL_ERROR "Input file '${INPUT_FILE}' does not exist")
endif()

# Determine include search paths from the script's own location
get_filename_component(script_dir "${CMAKE_SCRIPT_MODE_FILE}" DIRECTORY)
get_filename_component(project_root "${script_dir}" DIRECTORY)
set(vesin_include_dir "${project_root}/include")

get_filename_component(input_dir "${INPUT_FILE}" DIRECTORY)

# Read file as text so we can expand included files
file(READ "${INPUT_FILE}" content)

# Expand local includes (#include "...") iteratively, supporting
# transitive includes up to a reasonable depth. Look for included
# files first next to the input file, then in <project>/include/.
set(max_depth 10)
foreach(depth RANGE 1 ${max_depth})
    string(REGEX MATCH "#include \"([^\"]+)\"" matched_include "${content}")
    if(NOT matched_include)
        break()
    endif()

    set(include_file "${CMAKE_MATCH_1}")

    # Locate the included file
    set(found_path "")
    if(EXISTS "${input_dir}/${include_file}")
        set(found_path "${input_dir}/${include_file}")
    elseif(EXISTS "${vesin_include_dir}/${include_file}")
        set(found_path "${vesin_include_dir}/${include_file}")
    endif()

    if(NOT found_path)
        message(FATAL_ERROR "Cannot find included file '${include_file}' "
            "(looked in '${input_dir}' and '${vesin_include_dir}')"
        )
    endif()

    # Read included content, strip angle-bracket includes, then inline it
    file(READ "${found_path}" include_content)
    string(REGEX REPLACE "#include <[^>]+>\n?" "" include_content "${include_content}")
    string(REPLACE "#include \"${include_file}\"" "${include_content}" content "${content}")
endforeach()

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
