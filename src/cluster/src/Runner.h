/* -*- C++ -*-
 *
 * Runner.h
 *
 * Author: Benjamin T James
 *
 * Runner class, sets default params
 * and runs program
 */
#ifndef RUNNER_H
#define RUNNER_H

#include <iostream>
#include <map>
#include <set>
#include "Point.h"
using namespace std;

class Runner {
public:
	Runner(int argc, char** argv);
	~Runner() { indices.clear(); files.clear(); qfiles.clear(); };
	int run();
private:
	template<class T> int do_run(std::vector<std::pair<std::string,std::string*> > &sequences);
	template<class T> void print_output(const map<Point<T>*, vector<Point<T>*>*> &m) const;
	int k = -1;
        int bandwidth;
	double similarity = -1;
	long largest_count = 0;
	int iterations = 15;
	int delta = 5;
	bool align = false;
	bool recover = false;
	int sample_size = 0;
	int pivots = 40;
	uint8_t mode = 0;
	uint64_t feats = 0;
	uint64_t chunk_size = 10000;
	std::vector<std::string> files, qfiles;
	std::vector<size_t> indices;
	string output = "output.search";
	void get_opts(int argc, char** argv);
	pair<int,uint64_t> find_k();
};
#endif
