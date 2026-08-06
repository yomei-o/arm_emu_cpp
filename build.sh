#!/bin/sh
# Builds the emulator without needing CMake.
#
#   sh build.sh                     # g++ (or $CXX), the usual case
#   CXX=clang++ sh build.sh         # clang, same flags
#   CXX=cl sh build.sh              # MSVC: cl.exe's own flags, and armemu.exe
#
# The MSVC branch exists because the sources are free of GCC/Clang extensions and
# a second compiler is the best differential oracle available on a host without a
# cross-compiler: cl.exe and clang must agree, byte for byte, over a whole guest run.
set -e
cd "$(dirname "$0")"
CXX=${CXX:-g++}

case "$(basename "$CXX")" in
    cl|cl.exe)
        "$CXX" -nologo -std:c++17 -O2 -EHsc -utf-8 -W3 -DNOMINMAX \
               -D_CRT_SECURE_NO_WARNINGS -Isrc src/*.cpp -Fe:armemu.exe
        echo "built armemu.exe"
        ;;
    *)
        CXXFLAGS=${CXXFLAGS:--std=c++17 -O2 -Wall -Wextra}
        $CXX $CXXFLAGS -Isrc -o armemu src/*.cpp
        echo "built armemu"
        ;;
esac
