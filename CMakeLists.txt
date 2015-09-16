cmake_minimum_required(VERSION 3.2)
project(sgdtk_cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -funroll-loops -mfpmath=sse")

include_directories(include)

#find_package(Boost 1.57.0)
find_package(BLAS)
set(SOURCE_FILES
    include/sgdtk/Exception.h
    include/sgdtk/FeatureProvider.h
    include/sgdtk/FeatureVector.h
    include/sgdtk/HingeLoss.h
    include/sgdtk/Learner.h
    include/sgdtk/LinearModel.h
    include/sgdtk/LogLoss.h
    include/sgdtk/Loss.h
    include/sgdtk/Metrics.h
    include/sgdtk/Model.h
    include/sgdtk/Params.h
    include/sgdtk/SGDLearner.h
    include/sgdtk/sgdtk.h
    include/sgdtk/SquaredHingeLoss.h
    include/sgdtk/SquaredLoss.h
    include/sgdtk/SVMLightFileFeatureProvider.h
    include/sgdtk/Types.h
    src/Exception.cpp
    src/LinearModel.cpp
    src/Params.cpp
    src/SGDLearner.cpp
    src/SVMLightFileFeatureProvider.cpp
    src/DenseVectorN.cpp
    src/SparseVectorN.cpp)


add_library(sgdtk SHARED ${SOURCE_FILES} include/sgdtk/DenseVectorN.h include/sgdtk/ModelFactory.h include/sgdtk/RobbinsMonroUpdateSchedule.h include/sgdtk/FixedLearningRate.h include/sgdtk/ConsumerProducerQueue.h)
#target_link_libraries(sgdtk ${BLAS_blas_library})


include_directories(${BLAS_INCLUDE_DIRS})
target_link_libraries (sgdtk ${BLAS_LIBRARIES})

add_executable(Train Train.cpp)
add_dependencies(Train sgdtk ${BLAS_LIBRARIES})


message(STATUS "Found ${BLAS_LIBRARIES}")
target_link_libraries (Train sgdtk ${BLAS_LIBRARIES})


#target_link_libraries(Train -L. sgdtk ${BLAS_blas_library})

#add_executable(TestCP TestCP.cpp)
#add_dependencies(TestCP sgdtk)
#target_link_libraries(TestCP -L. sgdtk)
