#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "pmc::pmc" for configuration "Debug"
set_property(TARGET pmc::pmc APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(pmc::pmc PROPERTIES
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libpmc.so"
  IMPORTED_SONAME_DEBUG "libpmc.so"
  )

list(APPEND _cmake_import_check_targets pmc::pmc )
list(APPEND _cmake_import_check_files_for_pmc::pmc "${_IMPORT_PREFIX}/lib/libpmc.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
