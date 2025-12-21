CC=gcc
CFLAGS=-std=c99 -pedantic -Wall -Wextra $(EFLAGS)
LDFLAGS=-lm

SRC_DIR=./src
BIN_DIR=./build

SRC=$(wildcard $(SRC_DIR)/*.c)
OBJECTS=$(patsubst %.c, %.o, $(SRC))

TEST_SRC=$(wildcard $(TEST_DIR)/*.c)
TEST_OBJS=$(patsubst %.c, %.o, $(TEST_SRC))

EXECUTABLE=$(BIN_DIR)/llnn

all: $(SOURCES) $(EXECUTABLE)

clean:
	rm -f $(SRC_DIR)/*.o
	rm -f $(BIN_DIR)/llnn

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $@ $(LDFLAGS)

%.o : %.c
	$(CC) $(CFLAGS) -c $< -o $@
