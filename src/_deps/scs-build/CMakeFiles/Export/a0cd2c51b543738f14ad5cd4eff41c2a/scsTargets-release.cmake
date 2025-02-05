#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "scs::scsdir" for configuration "Release"
set_property(TARGET scs::scsdir APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(scs::scsdir PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libscsdir.so.3.2.3"
  IMPORTED_SONAME_RELEASE "libscsdir.so.3.2.3"
  )

list(APPEND _cmake_import_check_targets scs::scsdir )
list(APPEND _cmake_import_check_files_for_scs::scsdir "${_IMPORT_PREFIX}/lib/libscsdir.so.3.2.3" )

# Import target "scs::scsindir" for configuration "Release"
set_property(TARGET scs::scsindir APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(scs::scsindir PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libscsindir.so.3.2.3"
  IMPORTED_SONAME_RELEASE "libscsindir.so.3.2.3"
  )

list(APPEND _cmake_import_check_targets scs::scsindir )
list(APPEND _cmake_import_check_files_for_scs::scsindir "${_IMPORT_PREFIX}/lib/libscsindir.so.3.2.3" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
