/* -*- C++ -*-
 *
 * Selector.cpp
 *
 * Author: Benjamin T James
 *
 * Old selection algorithm used to gather training data
 * from bulk sequences, e.g. balancing positive and negatives
 */
#include "Selector.h"
#include <set>
#include <algorithm>
#include <cmath>
#include "Progress.h"
#include "Feature.h"
#include "../../utility/GlobAlignE.h"

template<class T>
void Selector<T>::select(double cutoff)
{
	auto splt = split(cutoff);
	auto mp = get_align(splt);
	splt.clear();
	auto both = get_labels(mp, cutoff);
	mp.clear();
	cout << "both: " << both.first.size() << " " << both.second.size() << endl;
	for (int i = 0; i < both.first.size(); i++) {
		if (i % 2 == 0) {
			training.first.push_back(both.first[i]);
		} else {
			testing.first.push_back(both.first[i]);
		}
	}
	both.first.clear();
	for (int i = 0; i < both.second.size(); i++) {
		if (i % 2 == 0) {
			training.second.push_back(both.second[i]);
		} else {
			testing.second.push_back(both.second[i]);
		}
	}
	both.second.clear();

}

template<class T>
vector<pair<Point<T>*, Point<T>*> > Selector<T>::split(double cutoff)
{
	// n_points total per side
	// max_pts_from_one on each side
	auto cmp = [](const pair<Point<T>*,Point<T>*> a, const pair<Point<T>*,Point<T>*> b) {
		return a.first->get_header().compare(b.first->get_header()) < 0
||
		(a.first->get_header() == b.first->get_header() && a.second->get_header().compare(b.second->get_header()) < 0);
	};
        set<pair<Point<T>*, Point<T>*>, decltype(cmp)> pairs(cmp);
//	vector<pair<Point<T>*, Point<T>*> > pairs;
	const size_t total_num_pairs = sample_size * 2;
	int aerr = 0;
	vector<Point<T>*> indices;
	std::sort(points.begin(), points.end(), [](const Point<T>* a,
						   const Point<T>* b) -> bool {
			  return a->get_length() < b->get_length();
			  });
	Point<T> *begin_pt = points[points.size()/2];

	std::sort(points.begin(), points.end(), [&](const Point<T>* a,
							    const Point<T>* b) -> bool {
				  return a->distance(*begin_pt) < b->distance(*begin_pt);
			  });
	int num_iterations = ceil(((double)sample_size) / max_pts_from_one) - 1;
	if (num_iterations == 0) {
		cerr << "num_iterations == 0" << endl;
		throw 10;
	}
	for (int i = 0; i <= num_iterations; i++) {
		int idx = i * (points.size()-1) / num_iterations;
		indices.push_back(points[idx]);
	}
	cout << "Point pairs: " << indices.size() << endl;
	size_t to_add_each = max_pts_from_one / 2;
	size_t total_align = 0;
	Progress prog(indices.size(), "Sorting data");
#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < indices.size(); i++) {
		vector<Point<T>*> pts = points;
		Point<T>* p = indices[i];
		std::sort(pts.begin(), pts.end(), [&](const Point<T>* a,
						      const Point<T>* b) {
				  return a->distance(*p) < b->distance(*p);
			  });
		// do binary search with alignment
		size_t offset = pts.size() / 4;
		size_t pivot = offset;
		double closest_algn = 20000;
		size_t best_pivot = 2 * offset;
		size_t p_i = 0, pivot_cutoff = floor(log(2*offset)/log(2))/2;
		size_t before = 0, after = pts.size()-1;
		for (before = 2 * offset, offset = pts.size()/4;
		     offset > 0;
		     offset /= 2) {
			double algn = Feature<T>::intersection(*p, *pts[before]);
			if (fabs(algn - (cutoff + 0.1)) < closest_algn) {
				closest_algn = fabs(algn - (cutoff + 0.1));
				best_pivot = before;
			}
			if (algn < cutoff + 0.1) {
				before -= offset;
			} else if (algn > cutoff + 0.1) {
				before += offset;
			} else {
				break;
			}
		}
		before = best_pivot;
		offset = pts.size()/4;
		closest_algn = 20000;
		best_pivot = 2 * offset;
		for (after = 2 * offset, offset = pts.size()/4;
		     offset > 0;
		     offset /= 2) {
			double algn = Feature<T>::intersection(*p, *pts[after]);
			if (fabs(algn - (cutoff - 0.1)) < closest_algn) {
				closest_algn = fabs(algn - (cutoff - 0.1));
				best_pivot = after;
			}
			if (algn < cutoff - 0.1) {
				after -= offset;
			} else if (algn > cutoff - 0.1) {
				after += offset;
			} else {
				break;
			}
		}
		after = best_pivot;
		offset = (after - before) / 4;
		closest_algn = 20000;
		best_pivot = 2 * offset;
		for (pivot = before + (after - before) / 2; offset > 0; offset /= 2) {
			double algn = 0;
			// if (p_i++ < pivot_cutoff) {
			// 	algn = Feature<T>::intersection(*p, *pts[pivot]);
			// } else
			{
				algn = align(p, pts[pivot]);
#pragma omp atomic
				total_align++;
			}

			// cout << "Pivot: " << pivot << " point: " << pts[pivot]->get_header() << " sim: " << align(p, pts[pivot]) << endl;
			if (fabs(algn - cutoff) < closest_algn) {
				closest_algn = fabs(algn - cutoff);
				best_pivot = pivot;
			}
			if (algn < cutoff) {
				pivot -= offset;
			} else if (algn > cutoff) {
				pivot += offset;
			} else {
				break;
			}
		}
//		cout << "Pivot: " << pivot << " point: " << pts[pivot]->get_header() << " sim: " << align(p, pts[pivot]) << endl;
		// before: [0, pivot) size: to_add_each
		// after: [pivot, size) size: to_add_each
		double before_inc = (double)pivot / to_add_each;
		double after_inc = ((double)(pts.size() - pivot)) / to_add_each;
#pragma omp critical
		{
			prog++;
			if (before_inc < 1) {
				aerr = 1;
			} else if (after_inc < 1) {
				aerr = -1;
			}
		}
		double before_start = 0;
		double after_start = pivot;
		double top_start = 0;
		size_t size_before = pairs.size();
		vector<pair<Point<T>*,Point<T>*> > buf;
		// Adds points above cutoff by adding before_inc
		for (int i = 0; i < to_add_each; i++) {
			int idx = round(before_start);
			int dist = pts.at(idx)->distance(*p);
			//	cout << p->get_header() << " " << pts[idx]->get_header() << " " << dist << endl;
			auto pr = p->get_header().compare(pts.at(idx)->get_header()) < 0 ? make_pair(p, pts.at(idx)) : make_pair(pts.at(idx), p);
			buf.push_back(pr);
			before_start += before_inc;
		}
		// Adds points before cutoff by adding after_inc
		for (int i = 0; i < to_add_each && round(after_start) < pts.size(); i++) {
			int idx = round(after_start);
			int dist = pts[idx]->distance(*p);
			//		cout << p->get_header() << " " << pts[idx]->get_header() << " " << dist << endl;
			auto pr = p->get_header().compare(pts[idx]->get_header()) < 0 ? make_pair(p, pts[idx]) : make_pair(pts[idx], p);
			buf.push_back(pr);
			after_start += after_inc;
		}
#pragma omp critical
		{
			// Adds buffer to total pairs
		// 	for (auto p : buf) {
// 				pairs.push_back(p);
// 			}
			pairs.insert(std::begin(buf), std::end(buf));
		}
//			cout << "added " << pairs.size() - size_before << " pairs" << endl;
	}
	prog.end();
	if (aerr < 0) {
		cerr << "Warning: Alignment may be too small for sampling" << endl;
	} else if (aerr > 0) {
		cerr << "Warning: Alignment may be too large for sampling" << endl;
	}
	cout << "Total alignments: " << total_align << endl;
	int i = 0;
	for (auto a : pairs) {
		cout << "Before Pair: " << a.first->get_header() << ", " << a.second->get_header() << endl;
		if (++i == 4) {
			break;
		}
	}
	return std::vector<std::pair<Point<T>*,Point<T>*> >(pairs.begin(), pairs.end());
}

template<class T>
double Selector<T>::align(Point<T> *a, Point<T>* b)
{
	auto sa = a->get_data_str();
	auto sb = b->get_data_str();
	int la = sa.length();
	int lb = sb.length();

	GlobAlignE galign(sa.c_str(), 0, la-1,
			  sb.c_str(), 0, lb-1,
			  1, -1, 2, 1);

	return galign.getIdentity();

}

template<class T>
vector<pra<T> > resize_vec(vector<pra<T> > &vec, size_t new_size)
{
	cout << "Vector size: " << vec.size() << " min size: " << new_size << endl;
	vector<pair<Point<T>*, Point<T>*> > data;
	if (vec.size() <= new_size) {
		for (int i = 0; i < vec.size(); i++) {
			data.push_back(vec[i].first);
		}
		return data;
	}
	using k = pair<pair<Point<T>*,Point<T>*>, double>;
	std::sort(vec.begin(), vec.end(), [](const k& a, const k& b) {
			return a.second < b.second;
		});
	double interval = (double)vec.size() / (vec.size() - new_size);
	std::set<int> indices;
	int i = 0;
	for (double index = 0; round(index) < vec.size() && i < (vec.size() - new_size);
	     i++, index += interval) {
		int j = round(index);
		indices.insert(j);
	}

	std::cout << "index size: " << indices.size() << std::endl;

	// for (double index = 0; round(index) < vec.size() && indices.size() < new_size;
	//      index += interval) {
	// 	int j = round(index);
	// 	indices.insert(vec[j]);
	// }
	// vec.erase(vec.begin(), std::remove_if(vec.begin(), vec.end(), [&](const k& a) {
	// 			return indices.find(a) == indices.end();
	// 		}));
	for (auto iter = indices.rbegin(); iter != indices.rend(); iter++) {
		int idx = *iter;
		vec.erase(vec.begin() + idx);
	}
	if (vec.size() != new_size) {
		cerr << "sizes are not the same: " << vec.size() << " " << new_size <<  endl;
		throw "Resize did not work";
	}
	for (auto a : vec) {
		data.push_back(a.first);
	}
	return data;
}

struct rng {
	rng() {
		srand(0);
	}
	int operator()(int n) const {
		return rand() % n;
	}
};


template<class T>
vector<pra<T> > Selector<T>::get_align(vector<pair<Point<T>*,Point<T>*> >&vec) const
{
	vector<pra<T> > hash;
	auto at = [](Point<T>* a, Point<T>* b) {
		return a->get_header() < b->get_header() ? make_pair(a, b) : make_pair(b, a);
	};
	Progress p(vec.size(), "Alignment");
	#pragma omp parallel for
	for (size_t i = 0; i < vec.size(); i++) {
		pra<T> pr;
		pr.first = vec[i].first;
		pr.second = vec[i].second;
		pr.val = align(pr.first, pr.second);
		#pragma omp critical
		{
			hash.push_back(pr);
			p++;
		}
	}
	p.end();
	return hash;
}

template<class T>
pair<vector<pra<T> >,
     vector<pra<T> > > Selector<T>::get_labels(vector<pra<T> > &vec, double cutoff) const
{

	auto scmp = [](const pra<T> a, const pra<T> b) {
		return a.first->get_header().compare(b.first->get_header()) < 0
		||
		(a.first->get_header() == b.first->get_header() && a.second->get_header().compare(b.second->get_header()) < 0);
	};

	// todo: convert to std::map
	std::set<pra<T>, decltype(scmp)> buf_pos(scmp), buf_neg(scmp);
	std::vector<pra<T> > buf_vpos, buf_vneg;
//	std::sort(vec.begin(), vec.end(), cmp);
	// cout << "Before Pair: " << vec[0].first->get_header() << ", " << vec[0].second->get_header() << endl;
	// cout << "Before Pair: " << vec[vec.size()-1].first->get_header() << ", " << vec[vec.size()-1].second->get_header() << endl;

	rng gen;
	random_shuffle(vec.begin(), vec.end(), gen);
	// cout << "Pair: " << vec[0].first->get_header() << ", " << vec[0].second->get_header() << endl;
	// cout << "Pair: " << vec[vec.size()-1].first->get_header() << ", " << vec[vec.size()-1].second->get_header() << endl;

	for (size_t i = 0; i < vec.size(); i++) {
		bool is_pos = vec[i].val >= cutoff;
// #pragma omp critical
		{
			if (is_pos) {
				buf_pos.insert(vec[i]);
				//cout << vec[i].first->get_header() << " " << vec[i].second->get_header() << " " << algn << endl;
			} else {
				buf_neg.insert(vec[i]);
			}
		}
	}
	std::cout << "positive=" << buf_pos.size() << " negative=" << buf_neg.size() << endl;
	if (buf_pos.empty() || buf_neg.empty()) {
		std::cout << "Identity value does not match sampled data: ";
		if (buf_pos.empty()) {
			std::cout << "Too many sequences below identity";
		} else {
			std::cout << "Too many sequences above identity";
		}
		std::cout << std::endl;
		exit(0);
	}
	size_t m_size = std::min(buf_pos.size(), buf_neg.size());

	std::cout << "resizing positive" << std::endl;
	size_t pos_size = 0;
	for (auto p : buf_pos) {
		if (pos_size++ < m_size) {
			buf_vpos.push_back(p);
		} else {
			break;
		}
	}
	size_t neg_size = 0;
	for (auto p : buf_neg) {
		if (neg_size++ < m_size) {
			buf_vneg.push_back(p);
		}
	}
        auto ret = make_pair(buf_vpos, buf_vneg);
	std::cout << "positive=" << ret.first.size() << " negative=" << ret.second.size() << endl;
	return ret;

}
template class Selector<uint8_t>;
template class Selector<uint16_t>;
template class Selector<uint32_t>;
template class Selector<uint64_t>;
template class Selector<int>;
template class Selector<double>;
