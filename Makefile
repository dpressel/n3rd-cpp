# TODO: better make system, possibly switch to waf

I=./include
L=./src
CXX=g++
OPT=
OPTS=-O3 -funroll-loops -mfpmath=sse
CXXFLAGS= ${OPTS} ${OPT} -Wall -I$I 
LDFLAGS = -static-libstdc++ -static-libgcc
LIBS = -lz -lm

PROGRAMS = Train 
OBJS = Exception.o Params.o SVMLightFileFeatureProvider.o LinearModel.o SGDLearner.o

all: ${PROGRAMS}

clean:
	-rm ${PROGRAMS} 2>/dev/null
	-rm *.o 2>/dev/null

Exception.o: $L/Exception.cpp ${INCS}
	${CXX} ${CXXFLAGS} -c $L/Exception.cpp

Params.o: $L/Params.cpp ${INCS}
	${CXX} ${CXXFLAGS} -c $L/Params.cpp

SVMLightFileFeatureProvider.o: $L/SVMLightFileFeatureProvider.cpp ${INCS}
	${CXX} ${CXXFLAGS} -c $L/SVMLightFileFeatureProvider.cpp

LinearModel.o: $L/LinearModel.cpp ${INCS}
	${CXX} ${CXXFLAGS} -c $L/LinearModel.cpp

SGDLearner.o: $L/SGDLearner.cpp ${INCS}
	${CXX} ${CXXFLAGS} -c $L/SGDLearner.cpp

Train.o: Train.cpp ${INCS}
	${CXX} ${CXXFLAGS} -c ./Train.cpp

Train: Train.o ${OBJS}
	${CXX} ${CXXFLAGS} -o $@ Train.o ${OBJS} ${LDFLAGS} ${LIBS}




