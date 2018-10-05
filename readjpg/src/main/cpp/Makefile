CC		= g++
INCLUDES	= -Iinclude -I${JAVA_HOME}/include \
	-I${JAVA_HOME}/include/linux \
	-I../../libjpeg-turbo/build/libjpeg-turbo-1.5.2
CFLAGS		= -Wall -O -fPIC $(INCLUDES)
LFLAGS		= -Wall -L. -ljpeg -shared -o $(SONAME)
SRC		= readjpg.c
OBJ		= $(SRC:.c=.o)
SONAME		= libreadjpg.so

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
