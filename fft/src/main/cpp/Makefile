CC		= g++
INCLUDES	= -Iinclude -I${JAVA_HOME}/include \
		  -I${JAVA_HOME}/include/linux \
		  -I../../clFFT/build/install/include
CFLAGS		= -Wall -O -fPIC $(INCLUDES)
LFLAGS		= -Wall -L. -lJOCL_2_0_4-linux-x86_64 -lclFFT \
		  -shared -o $(SONAME)
SRC		= fft.c
OBJ		= $(SRC:.c=.o)
SONAME		= libfft.so

DEPFILE		= .depends
DEPFLAGS	= -Y -f$(DEPFILE) $(INCLUDES)

all:	$(SONAME)

$(SONAME): $(DEPFILE) $(OBJ)
	$(CC) $(LFLAGS) -o $@ $(OBJ) $(LIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c $<

$(DEPFILE):
	touch $(DEPFILE)
	makedepend $(DEPFLAGS) $(SRC) $(HDR)


depend:
	rm -rf $(DEPFILE)
	make $(DEPFILE)


clean: 
	rm -f $(SONAME) $(LIBNAME) *.o .depends .depends.bak

# DO NOT DELETE
