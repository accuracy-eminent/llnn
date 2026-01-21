CC=gcc
CFLAGS=-std=c99 -pedantic -Wall -Wextra -ggdb3 -O0 $(EFLAGS)
LDFLAGS=-lm

SRC_DIR=./src
BIN_DIR=./build
TEST_DIR=./test

SRC=$(wildcard $(SRC_DIR)/*.c)
OBJECTS=$(patsubst %.c, %.o, $(SRC))

TEST_SRC=$(wildcard $(TEST_DIR)/*.c)
TEST_OBJS=$(patsubst %.c, %.o, $(TEST_SRC))

EXECUTABLE=$(BIN_DIR)/llnn
TEST_EXE=$(BIN_DIR)/llnn_test

all: $(SOURCES) $(EXECUTABLE)  $(TEST_SRC) $(TEST_EXE)

clean:
	rm -f $(SRC_DIR)/*.o
	rm -f $(BIN_DIR)/llnn

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $@ $(LDFLAGS)

$(TEST_EXE): $(TEST_OBJS)
	$(CC) $(CFLAGS) $(filter-out $(SRC_DIR)/main.o, $(OBJECTS)) $(TEST_OBJS) -o $@ $(LDFLAGS)

%.o : %.c
	$(CC) $(CFLAGS) -c $< -o $@
