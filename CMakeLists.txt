#==============================
# Requirements
#==============================

cmake_minimum_required(VERSION 3.10)

#==============================
# Project Settings
#==============================

if (NOT PROJECT_DECLARED)
    project("Gaia DNN Helper" LANGUAGES CXX)
    set(PROJECT_DECLARED)
endif()

#==============================
# Project Modules
#==============================

add_subdirectory("GaiaDNNHelper")