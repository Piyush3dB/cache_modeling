all : examples
	./examples

examples : $(wildcard *.cpp *.h) Makefile
	g++ -o $@ $(wildcard *.cpp) -I armadillo/ --std=c++11 -Ofast -DNDEBUG

	# # # PGO is slower??
	# g++ -o $@ $(wildcard *.cpp) -I armadillo/ --std=c++11 -Ofast -fprofile-generate -DNDEBUG
	# @echo PROFILING PASS
	# ./examples
	# @echo PROFILING DONE
	# g++ -o $@ $(wildcard *.cpp) -I armadillo/ --std=c++11 -Ofast -fprofile-use -DNDEBUG
