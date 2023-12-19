cmake_minimum_required(VERSION 3.25)
project(OSD)

set(CMAKE_CXX_STANDARD 17)

add_library(OSD SHARED DYNACON_SD.c)
add_compile_options(-O3)