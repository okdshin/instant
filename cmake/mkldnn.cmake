include(ExternalProject)
set(EXTERNAL_DIR ${CMAKE_SOURCE_DIR}/external)
set(MKLDNN_DIR ${EXTERNAL_DIR}/mkldnn)
include(ProcessorCount)
ProcessorCount(CORE_NUM)
ExternalProject_Add(MKLDNN
	SOURCE_DIR ${MKLDNN_DIR}/src
	CMAKE_ARGS
		-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
		-DCMAKE_INSTALL_PREFIX=${MKLDNN_DIR}/install
		-DMKLROOT=${MKL_ROOT_DIR}
	GIT_REPOSITORY https://github.com/01org/mkl-dnn.git
	BINARY_DIR ${MKLDNN_DIR}/build
	BUILD_COMMAND cmake ${MKLDNN_DIR}/src
	INSTALL_DIR ${MKLDNN_DIR}/install
	INSTALL_COMMAND make install -j${CORE_NUM}
	LOG_CONFIGURE 1
	LOG_BUILD 1
	LOG_INSTALL 1)
set(MKLDNN_INCLUDE_DIR ${MKLDNN_DIR}/install/include CACHE PATH "Include file path of MKLDNN")
set(MKLDNN_LIB_DIR ${MKLDNN_DIR}/install/lib CACHE PATH "Library path of MKLDNN")
