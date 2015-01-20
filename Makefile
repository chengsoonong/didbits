# Makefile used to manage the git repository, not for the tutorial

TARGETS="KernelNorm"

all: debug render clean

debug:
	python ipynbhelper.py --debug $(TARGETS)

render:
	python ipynbhelper.py --render $(TARGETS)

clean:
	find . -name "*.pyc" | xargs rm -f
	python ipynbhelper.py --clean $(TARGETS)

clean-data:
	find . -name "*.pkl" | xargs rm -f
	find . -name "*.npy" | xargs rm -f
	find . -name "*.mmap" | xargs rm -f

