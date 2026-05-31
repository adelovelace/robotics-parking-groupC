#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "robomaster_msgs::robomaster_msgs__rosidl_generator_py" for configuration ""
set_property(TARGET robomaster_msgs::robomaster_msgs__rosidl_generator_py APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(robomaster_msgs::robomaster_msgs__rosidl_generator_py PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_NOCONFIG "robomaster_msgs::robomaster_msgs__rosidl_generator_c;robomaster_msgs::robomaster_msgs__rosidl_typesupport_c;std_msgs::std_msgs__rosidl_generator_py;builtin_interfaces::builtin_interfaces__rosidl_generator_py;action_msgs::action_msgs__rosidl_generator_py;unique_identifier_msgs::unique_identifier_msgs__rosidl_generator_py"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/librobomaster_msgs__rosidl_generator_py.dylib"
  IMPORTED_SONAME_NOCONFIG "@rpath/librobomaster_msgs__rosidl_generator_py.dylib"
  )

list(APPEND _cmake_import_check_targets robomaster_msgs::robomaster_msgs__rosidl_generator_py )
list(APPEND _cmake_import_check_files_for_robomaster_msgs::robomaster_msgs__rosidl_generator_py "${_IMPORT_PREFIX}/lib/librobomaster_msgs__rosidl_generator_py.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
