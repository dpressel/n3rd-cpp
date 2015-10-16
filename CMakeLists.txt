cmake_minimum_required(VERSION 2.8.12.2)
project(sgdtk_cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -funroll-loops -mfpmath=sse")

include_directories(include)

#find_package(Boost 1.57.0)
#find_package(BLAS)
#find_package(LAPACK)
set(SOURCE_FILES
        include/sgdtk/DenseVectorN.h
        include/sgdtk/ModelFactory.h
        include/sgdtk/RobbinsMonroUpdateSchedule.h
        include/sgdtk/FixedLearningRate.h
        include/sgdtk/ConsumerProducerQueue.h
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
        include/sgdtk/Tensor.h
        include/n3rd/NeuralNetModel.h
        include/n3rd/Layer.h
        include/n3rd/AverageFoldingLayer.h
        src/Exception.cpp
        src/LinearModel.cpp
        src/Params.cpp
        src/SGDLearner.cpp
        src/SVMLightFileFeatureProvider.cpp
        src/DenseVectorN.cpp
        src/SparseVectorN.cpp
        src/Tensor.cpp
        src/NeuralNetModel.cpp
        src/AverageFoldingLayer.cpp
        include/n3rd/FilterOps.h
        src/FilterOps.cpp
        include/n3rd/TanhLayer.h
        src/TanhLayer.cpp
        include/n3rd/FullyConnectedLayer.h
        include/n3rd/FullyConnectedLayerBlas.h
        src/FullyConnectedLayer.cpp
        src/FullyConnectedLayerBlas.cpp
        include/n3rd/LogSoftMaxLayer.h
        src/LogSoftMaxLayer.cpp
        include/n3rd/SigmoidLayer.h
        src/SigmoidLayer.cpp
        include/n3rd/TemporalConvolutionalLayer.h
        src/TemporalConvolutionalLayer.cpp
        include/n3rd/KMaxPoolingLayer.h
        src/KMaxPoolingLayer.cpp
        include/n3rd/Word2VecModel.h
        include/n3rd/OrderedEmbeddedDatasetReader.h
        src/OrderedEmbeddedDatasetReader.cpp
        include/n3rd/SumEmbeddedDatasetReader.h
        src/SumEmbeddedDatasetReader.cpp
        src/Word2VecModel.cpp
        include/n3rd/NeuralNetModelFactory.h
        include/n3rd/WeightHacks.h
        src/WeightHacks.cpp)

add_definitions(-DUSE_BLAS=1)
#add_compile_options(-march=native -mfpmath=sse)
add_library(sgdtk SHARED ${SOURCE_FILES})
#STRING(REGEX MATCH "blas" "libopenblas" ${OPENBLAS} ${BLAS_LIBRARIES})
#STRING(REGEX MATCH "dll" "" ${LAPACK_STATIC} ${LAPACK_lapack_LIBRARY})
include_directories(/usr/openblas/include)
#target_link_libraries (sgdtk ${BLAS_LIBRARIES}")
target_link_libraries(sgdtk openblas)
add_executable(Train Train.cpp)
add_executable(TrainCNN TrainCNN.cpp)
add_executable(TrainNBoW TrainNBoW.cpp)
add_executable(TestFC TestFC.cpp)
add_executable(TestFCBlas TestFCBlas.cpp)
add_executable(TestConv TestConv.cpp)
#add_executable(TestConvBlas TestConvBlas.cpp)
#add_dependencies(Train sgdtk ${BLAS_LIBRARIES})
#add_dependencies(Train sgdtk ${BLAS_LIBRARIES})

target_link_libraries (Train sgdtk openblas)

target_link_libraries (TrainCNN sgdtk openblas)
target_link_libraries (TestFC sgdtk openblas)
target_link_libraries (TestFCBlas sgdtk openblas)
target_link_libraries (TestConv sgdtk openblas)
#target_link_libraries (TestConvBlas sgdtk libopenblas.dll)
target_link_libraries(TrainNBoW sgdtk openblas)
#target_link_libraries(Train -L. sgdtk ${BLAS_blas_library})

#add_executable(TestCP TestCP.cpp)
#add_dependencies(TestCP sgdtk)
#target_link_libraries(TestCP -L. sgdtk)
