/* -*- C++ -*-
 *
 * Selector.h
 *
 * Author: Benjamin T James
 *
 * Old selection algorithm used to gather training data
 * from bulk sequences, e.g. balancing positive and negatives
 *
 * Contains pra<> data structure which contains a pair of
 * Points and an alignment value
 */
#ifndef SELECTOR_H
#define SELECTOR_H
#include <vector>
#include <unordered_map>
#include "Point.h"

template<class T>
struct pra {
	Point<T>* first;
	Point<T>* second;
	double val;
	pra() {}
	pra(const pra<T>&f) : first(f.first), second(f.second), val(f.val) {}
	pra(Point<T>* a, Point<T>* b, double c) : first(a), second(b), val(c) {}
};

template<class T>
class Selector {
public:
	Selector(std::vector<Point<T>*> v,
		 size_t sample_size_,
		 size_t max_pts_from_one_)
		: points(v),
		  sample_size(sample_size_),
		  max_pts_from_one(max_pts_from_one_) {

	}
	~Selector() { training.first.clear(); training.second.clear(); testing.first.clear(); training.second.clear(); }
	void select(double cutoff);
	static double align(Point<T>*a, Point<T>*b);
	pair<vector<pra<T> >,
	     vector<pra<T> > > get_training() const { return training; }
	pair<vector<pra<T> >,
	     vector<pra<T> > > get_testing() const { return testing; }
private:
	vector<std::pair<Point<T>*, Point<T>*> > split(double cutoff);

	vector<pra<T> > get_align(vector<std::pair<Point<T>*,Point<T>*> >&) const;

	pair<vector<pra<T> > ,
	     vector<pra<T> > > get_labels(vector<pra<T> > &, double cutoff) const;

	const size_t sample_size, max_pts_from_one;
	std::vector<Point<T>*> points;
	pair<vector<pra<T> >,
	     vector<pra<T> > > training, testing;
};
#endif
