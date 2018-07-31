/* -*- C++ -*-
 *
 * Predictor.h
 *
 * Author: Benjamin T James
 *
 * Main class for training and prediction
 * Does bulk training, but can be adapted for on-line training
 */

#ifndef PREDICTOR_H
#define PREDICTOR_H

#include "../cluster/src/GLM.h"
#include "../cluster/src/Point.h"
#include "../cluster/src/Feature.h"
#include <set>
#include <omp.h>
#define PRED_MODE_CLASS 1
#define PRED_MODE_REGR  2

#define PRED_FEAT_FAST (FEAT_EUCLIDEAN | FEAT_MANHATTAN | FEAT_INTERSECTION | FEAT_KULCZYNSKI2 | FEAT_SIMRATIO | FEAT_NORMALIZED_VECTORS | FEAT_PEARSON_COEFF | FEAT_EMD | FEAT_LENGTHD )
#define PRED_FEAT_DIV (FEAT_JEFFEREY_DIV | FEAT_JENSEN_SHANNON)

template<class T>
class Predictor {
public:
	Predictor(int k_, double id_, uint8_t mode_, uint64_t feats, int threshold_=100, int max_num_feat_=4) : k(k_), id(id_), is_trained(false), is_training(false), mode(mode_), max_num_feat(max_num_feat_), threshold(threshold_) {
		add_feats(possible_feats, feats);
		feat_c = NULL;
		feat_r = NULL;
		omp_init_lock(&lock);
	};
	Predictor(const std::string filename);
	~Predictor() {
		possible_feats.clear();
		omp_destroy_lock(&lock);
		if (feat_c) {
			delete feat_c;
		}
		if (feat_r) {
			delete feat_r;
		}
		if (selector) {
			delete selector;
		}
		training.clear();
		testing.clear();
	}
	void train(const std::vector<Point<T>* >& vec, const std::vector<Point<T>* >& vecq, uintmax_t& _id, size_t num_sample);
	double similarity(Point<T>* a, Point<T>* b);
	bool close(Point<T>* a, Point<T>* b);
	void save(std::string file);
	void check();
	uint8_t get_mode() const { return mode; }
private:
	static void add_feats(std::vector<std::pair<uint64_t, Combo> >& vec, uint64_t flags);
	static pair<matrix::GLM, Feature<T>*> read_from(std::ifstream &in, int k_);
	static void write_to(std::ofstream &out, Feature<T>* f, matrix::GLM glm);
	void filter(std::vector<pra<T> > &s, std::string prefix="");
	void train();
	void train_class(Feature<T>* feat);
	void train_regr(Feature<T>* feat);
	void train_class_regr(Feature<T>* feat);
	double predict(Point<T>* a, Point<T>* b);
	bool p_close(Point<T>* a, Point<T>* b);
	double p_predict(Point<T>* a, Point<T>* b);
	void mutate_seqs(Point<T>* p, size_t num_seq, vector<pra<T> > &,vector<pra<T> > & , double id_begin, double id_end, uintmax_t& _id);
	Selector<T> *selector = NULL;
	Feature<T> *feat_c, *feat_r;
	matrix::GLM c_glm, r_glm;
        vector<pra<T> > training, testing;
	bool is_trained, is_training;
	int max_num_feat, k, threshold;
	uint8_t mode;
	double id;
	vector<std::pair<uint64_t, Combo> > possible_feats;
	omp_lock_t lock;
};
#endif
