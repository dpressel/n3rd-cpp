cmake_minimum_required(VERSION 2.8.12.2)
project(n3rd_cpp)


option(ENABLE_CUDA "Enable the cuda build" ON)



if(ENABLE_CUDA)
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    set(CUDA_EXTRA_LIBRARIES ${CUDA_INCLUDE_DIRS}/../lib64/libcublas.so)
    #set(CUDA_EXTRA_LIBRARIES ${CUDA_INCLUDE_DIRS}/../lib64/libcudnn.so ${CUDA_INCLUDE_DIRS}/../lib64/libcublas.so)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -funroll-loops -mfpmath=sse")

include_directories(include)
link_libraries(${CUDA_LIBRARIES})
#find_package(Boost 1.57.0)
#find_package(BLAS)
#find_package(LAPACK)
set(SOURCE_FILES
        include/sgdtk/VectorN.h
        include/sgdtk/WeightModel.h
        include/sgdtk/TensorI.h
        include/sgdtk/DenseVectorN.h
        include/sgdtk/ModelFactory.h
        include/sgdtk/RobbinsMonroUpdateSchedule.h
        include/sgdtk/FixedLearningRateSchedule.h
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
        include/n3rd/ClassNLLLoss.h
        include/n3rd/NeuralNetModel.h
        include/n3rd/Layer.h
        include/n3rd/AbstractLayer.h
        include/n3rd/AverageFoldingLayer.h
        include/sgdtk/GPU.h
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
        include/n3rd/FullyConnectedLayerCuBlas.h
        src/FullyConnectedLayer.cpp
        src/FullyConnectedLayerBlas.cpp
        src/FullyConnectedLayerCuBlas.cpp
        include/n3rd/LogSoftMaxLayer.h
        src/LogSoftMaxLayer.cpp
        include/n3rd/SigmoidLayer.h
        src/SigmoidLayer.cpp
        src/GPU.cpp
        include/n3rd/TemporalConvolutionalLayer.h
        src/TemporalConvolutionalLayer.cpp
        include/n3rd/TemporalConvolutionalLayerCuBlas.h
        src/TemporalConvolutionalLayerCuBlas.cpp
        include/n3rd/SpatialConvolutionalLayerCuBlas.h
        src/SpatialConvolutionalLayerCuBlas.cpp
        include/n3rd/TemporalConvolutionalLayerFFT.h
        src/TemporalConvolutionalLayerFFT.cpp
        include/n3rd/TemporalConvolutionalLayerBlas.h
        src/TemporalConvolutionalLayerBlas.cpp
        include/n3rd/SpatialConvolutionalLayer.h
        src/SpatialConvolutionalLayer.cpp
        include/n3rd/SpatialConvolutionalLayerBlas.h
        src/SpatialConvolutionalLayerBlas.cpp
        include/n3rd/KMaxPoolingLayer.h
        src/KMaxPoolingLayer.cpp
        include/n3rd/MaxPoolingLayer.h
        src/MaxPoolingLayer.cpp
        include/n3rd/Word2VecModel.h
        include/n3rd/ReLULayer.h
        include/n3rd/OrderedEmbeddedDatasetReader.h
        src/OrderedEmbeddedDatasetReader.cpp
        include/n3rd/MNISTReader.h
        src/MNISTReader.cpp
        include/n3rd/SumEmbeddedDatasetReader.h
        src/SumEmbeddedDatasetReader.cpp
        src/Word2VecModel.cpp
        include/n3rd/NeuralNetModelFactory.h
        include/n3rd/NeuralNetModelCuda.h
        src/NeuralNetModelCuda.cpp
        include/n3rd/WeightHacks.h
        include/n3rd/MaxOverTimePoolingLayer.h
        src/ReLULayer.cpp
        include/n3rd/FFTOps.h
        src/FFTOps.cpp
        include/n3rd/TanhLayerCuda.h
        include/n3rd/MaxOverTimePoolingLayerCuda.h
        src/MaxOverTimePoolingLayerCuda.cpp
        src/WeightHacks.cpp include/n3rd/DropoutLayer.h src/DropoutLayer.cpp
        src/MaxOverTimePoolingLayer.cpp)

add_definitions(-DUSE_BLAS=1)
#add_compile_options(-march=native -mfpmath=sse)
add_library(n3rd SHARED ${SOURCE_FILES})
cuda_add_library(n3rdg SHARED include/n3rd/GPUOps.h src/GPUOps.cu)


#STRING(REGEX MATCH "blas" "libopenblas" ${OPENBLAS} ${BLAS_LIBRARIES})
#STRING(REGEX MATCH "dll" "" ${LAPACK_STATIC} ${LAPACK_lapack_LIBRARY})
include_directories(/usr/openblas/include)
#target_link_libraries (sgdtk ${BLAS_LIBRARIES}")
target_link_libraries(n3rd openblas ${CUDA_EXTRA_LIBRARIES} fftw m n3rdg)
add_executable(Train Train.cpp)
add_executable(TrainCNN TrainCNN.cpp)
add_executable(TrainNBoW TrainNBoW.cpp)
add_executable(TestFFT1D TestFFT1D.cpp)
add_executable(TestFC TestFC.cpp)
add_executable(TestFCBlas TestFCBlas.cpp)
add_executable(TestFCCuBlas TestFCCuBlas.cpp)
add_executable(TestConv TestConv.cpp)
add_executable(TestMaxOverTime TestMaxOverTime.cpp)
add_executable(TestConvBlas TestConvBlas.cpp)
add_executable(TestGPU TestGPU.cpp)
add_executable(LeNetish LeNetish.cpp)
#add_dependencies(Train sgdtk ${BLAS_LIBRARIES})
#add_dependencies(Train sgdtk ${BLAS_LIBRARIES})

target_link_libraries (Train n3rd ${CUDA_EXTRA_LIBRARIES} openblas fftw3 m n3rdg)

target_link_libraries (TrainCNN n3rd ${CUDA_EXTRA_LIBRARIES} openblas fftw3 m n3rdg)
target_link_libraries (LeNetish n3rd ${CUDA_EXTRA_LIBRARIES} openblas fftw3 m n3rdg)
target_link_libraries (TestFFT1D n3rd ${CUDA_EXTRA_LIBRARIES} openblas fftw3 m n3rdg)
target_link_libraries (TestFC n3rd ${CUDA_EXTRA_LIBRARIES} openblas fftw3 m n3rdg)
target_link_libraries (TestGPU n3rd ${CUDA_EXTRA_LIBRARIES} openblas fftw3 m n3rdg)
target_link_libraries (TestFCBlas n3rd ${CUDA_EXTRA_LIBRARIES} openblas fftw3 m n3rdg)
target_link_libraries (TestFCCuBlas n3rd ${CUDA_EXTRA_LIBRARIES} openblas fftw3 n3rdg)
target_link_libraries (TestConv n3rd ${CUDA_EXTRA_LIBRARIES} openblas fftw3 m n3rdg)
target_link_libraries (TestConvBlas n3rd ${CUDA_EXTRA_LIBRARIES} openblas fftw3 m n3rdg)
target_link_libraries (TestMaxOverTime n3rd ${CUDA_EXTRA_LIBRARIES} openblas fftw3 m n3rdg)
#target_link_libraries (TestConvBlas sgdtk libopenblas.dll)
target_link_libraries(TrainNBoW n3rd ${CUDA_EXTRA_LIBRARIES} openblas fftw3 m n3rdg)
#target_link_libraries(Train -L. sgdtk ${BLAS_blas_library})

#add_executable(TestCP TestCP.cpp)
#add_dependencies(TestCP sgdtk)
#target_link_libraries(TestCP -L. sgdtk)
