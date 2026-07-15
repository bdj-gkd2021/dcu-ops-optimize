find_package(HIP REQUIRED CONFIG PATHS ${HIP_TOOLKIT_ROOT_DIR})

################# RT #########################
find_library(
  GALAXY_HIP galaxyhip
  ${HIP_TOOLKIT_ROOT_DIR}/lib
  NO_DEFAULT_PATH
)

if(NOT TARGET hip::galaxyhip AND GALAXY_HIP)
  message(STATUS "Found galaxyhip: True")
  add_library(galaxyhip SHARED IMPORTED GLOBAL)
  add_library(hip::galaxyhip ALIAS hipgalaxy)
  set_property(
    TARGET galaxyhip
    PROPERTY IMPORTED_LOCATION
    ${GALAXY_HIP}
  )
elseif(TARGET hip::galaxyhip)
  message(STATUS "Found galaxyhip: True")
else()
  message(STATUS "Found galaxyhip: True")
endif()


find_library(
  HIPRTC_LIBRARY hiprtc
  PATHS
  ${HIP_TOOLKIT_ROOT_DIR}/lib
  NO_DEFAULT_PATH
  )

if(NOT TARGET hiprtc AND HIPRTC_LIBRARY)
  message(STATUS "Found hiprtc: True")
  add_library(hiprtc SHARED IMPORTED GLOBAL)
  add_library(hip::hiprtc ALIAS hiprtc)
  set_property(
    TARGET hiprtc
    PROPERTY IMPORTED_LOCATION
    ${HIPRTC_LIBRARY}
    )
elseif(TARGET hiprtc)
  message(STATUS "Found hiprtc: True")
else()
  message(STATUS "Found hiprtc: False")
endif()

include_directories(SYSTEM "${HIP_TOOLKIT_ROOT_DIR}/include")

# set hip property as *.cu
function(cutlass_correct_source_file_language_property)
  foreach(File ${ARGN})
    # add compile option -xhip while using clang++
    if(File MATCHES ".*\.cu$")
    #   set_source_files_properties(${File} PROPERTIES COMPILE_FLAGS "-x hip")
    set_source_files_properties(${File} PROPERTIES LANGUAGE CXX)
    endif()
  endforeach()
endfunction()

set(CUTLASS_UNITY_BUILD_ENABLED_INIT OFF)
set(CUTLASS_UNITY_BUILD_ENABLED ${CUTLASS_UNITY_BUILD_ENABLED_INIT} CACHE BOOL "Enable combined source compilation")

set(CUTLASS_UNITY_BUILD_BATCH_SIZE_INIT 16)
set(CUTLASS_UNITY_BUILD_BATCH_SIZE ${CUTLASS_UNITY_BUILD_BATCH_SIZE_INIT} CACHE STRING "Batch size for unified source files")

# set unify 
function(cutlass_unify_source_files TARGET_ARGS_VAR)

  set(options)
  set(oneValueArgs BATCH_SOURCES BATCH_SIZE)
  set(multiValueArgs)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT DEFINED TARGET_ARGS_VAR)
    message(FATAL_ERROR "TARGET_ARGS_VAR parameter is required")
  endif()

  if (__BATCH_SOURCES AND NOT DEFINED __BATCH_SIZE)
    set(__BATCH_SIZE ${CUTLASS_UNITY_BUILD_BATCH_SIZE})
  endif()

  if (CUTLASS_UNITY_BUILD_ENABLED AND DEFINED __BATCH_SIZE AND __BATCH_SIZE GREATER 1)
    set(CUDA_FILE_ARGS)
    set(TARGET_SOURCE_ARGS)
    
    foreach(ARG ${__UNPARSED_ARGUMENTS})
      if(${ARG} MATCHES ".*\.cu$")
        list(APPEND CUDA_FILE_ARGS ${ARG})
      else()
        list(APPEND TARGET_SOURCE_ARGS ${ARG})
      endif()
    endforeach()
    
    list(LENGTH CUDA_FILE_ARGS NUM_CUDA_FILE_ARGS)
    while(NUM_CUDA_FILE_ARGS GREATER 0)
      list(SUBLIST CUDA_FILE_ARGS 0 ${__BATCH_SIZE} CUDA_FILE_BATCH)
      string(SHA256 CUDA_FILE_BATCH_HASH "${CUDA_FILE_BATCH}")
      string(SUBSTRING ${CUDA_FILE_BATCH_HASH} 0 12 CUDA_FILE_BATCH_HASH)
      set(BATCH_FILE ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.unity.${CUDA_FILE_BATCH_HASH}.cu)
      message(STATUS "Generating ${BATCH_FILE}")
      file(WRITE ${BATCH_FILE} "// Unity File - Auto Generated!\n")
      foreach(CUDA_FILE ${CUDA_FILE_BATCH})
        get_filename_component(CUDA_FILE_ABS_PATH ${CUDA_FILE} ABSOLUTE)
        file(APPEND ${BATCH_FILE} "#include \"${CUDA_FILE_ABS_PATH}\"\n")
      endforeach()
      list(APPEND TARGET_SOURCE_ARGS ${BATCH_FILE})
      if (NUM_CUDA_FILE_ARGS LESS_EQUAL __BATCH_SIZE)
        break()
      endif()
      list(SUBLIST CUDA_FILE_ARGS ${__BATCH_SIZE} -1 CUDA_FILE_ARGS)
      list(LENGTH CUDA_FILE_ARGS NUM_CUDA_FILE_ARGS)
    endwhile()
  else()
    set(TARGET_SOURCE_ARGS ${__UNPARSED_ARGUMENTS})
  endif()
  set(${TARGET_ARGS_VAR} ${TARGET_SOURCE_ARGS} PARENT_SCOPE)
endfunction()


# unify -> set property -> add library
function(cutlass_add_library NAME)

  set(options SKIP_GENCODE_FLAGS)
  set(oneValueArgs EXPORT_NAME)
  set(multiValueArgs)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  cutlass_unify_source_files(TARGET_SOURCE_ARGS ${__UNPARSED_ARGUMENTS})

  cutlass_correct_source_file_language_property(${TARGET_SOURCE_ARGS})
  add_library(${NAME} ${TARGET_SOURCE_ARGS} "")


  cutlass_apply_standard_compile_options(${NAME})
  if (NOT __SKIP_GENCODE_FLAGS)
  cutlass_apply_cuda_gencode_flags(${NAME})
  endif()

  target_compile_features(
   ${NAME}
   INTERFACE
   cxx_std_11
   )

  if(__EXPORT_NAME)
    add_library(nvidia::cutlass::${__EXPORT_NAME} ALIAS ${NAME})
    set_target_properties(${NAME} PROPERTIES EXPORT_NAME ${__EXPORT_NAME})
  endif()

endfunction()

function(cutlass_add_executable NAME)

  set(options)
  set(oneValueArgs)
  set(multiValueArgs)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  cutlass_unify_source_files(TARGET_SOURCE_ARGS ${__UNPARSED_ARGUMENTS})

  cutlass_correct_source_file_language_property(${TARGET_SOURCE_ARGS})
  add_executable(${NAME} ${TARGET_SOURCE_ARGS})

  cutlass_apply_standard_compile_options(${NAME})
  cutlass_apply_cuda_gencode_flags(${NAME})

  target_compile_features(
   ${NAME}
   INTERFACE
   cxx_std_11
   )

endfunction()

function(cutlass_target_sources NAME)

  set(options)
  set(oneValueArgs)
  set(multiValueArgs)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  cutlass_unify_source_files(TARGET_SOURCE_ARGS ${__UNPARSED_ARGUMENTS})
  cutlass_correct_source_file_language_property(${TARGET_SOURCE_ARGS})
  target_sources(${NAME} ${TARGET_SOURCE_ARGS})

endfunction()
