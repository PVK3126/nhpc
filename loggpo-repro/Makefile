# LogGPO reproduction -- build with the cluster's MPI compiler wrapper.
#   make                 (uses mpicc)
#   make MPICC=mpiicc    (Intel MPI)
MPICC  ?= mpicc
CFLAGS ?= -O3 -g
LDLIBS ?= -lm

APPS = pingpong cg_mini sweep_mini

all: bin/params_bench $(APPS:%=bin/%) $(APPS:%=bin/%_traced) lib/libloggpotrace.so

bin lib:
	mkdir -p $@

bin/%_traced: src/%.c src/tracer.c | bin
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDLIBS)

bin/%: src/%.c | bin
	$(MPICC) $(CFLAGS) -o $@ $< $(LDLIBS)

lib/libloggpotrace.so: src/tracer.c | lib
	$(MPICC) $(CFLAGS) -fPIC -shared -o $@ $<

test:
	python3 tests/test_sim.py

clean:
	rm -rf bin lib

.PHONY: all test clean
