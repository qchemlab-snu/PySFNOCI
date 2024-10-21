add_library(SFNOCI_contract SHARED SFNOCI_contract.c)
set_target_properties(np_helper PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})

target_link_libraries(SFNOCI_contract ${BLAS_LIBRARIES})
