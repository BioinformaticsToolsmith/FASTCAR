/* -*- C++ -*-
 *
 * Runner.cpp
 *
 * Author: Benjamin T James
 *
 * Runner class that parses options and controls
 * the process of the program.
 */
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/sysinfo.h>
#include <cstdlib>
#include "../../nonltr/ChromListMaker.h"
#include "DivergencePoint.h"
#include "../../utility/AffineId.h"
#include "Runner.h"
#include "../../train/Predictor.h"
#include "ClusterFactory.h"
#include "Loader.h"
#include "bvec.h"
#include "Progress.h"
#include "Selector.h"
#include <omp.h>


Runner::Runner(int argc, char **argv)
{
	get_opts(argc, argv);
	if (sample_size == 0) {
		sample_size = 300;
	}
	srand(10);
}

int parseLine(char* line) {
	int i = strlen(line);
	const char* p = line;
	while (*p < '0' || *p > '9') p++;
	line[i-3] = '\0';
	i = atoi(p);
	return i;
}

void mem_used(std::string prefix)
{
	struct sysinfo memInfo;
	sysinfo(&memInfo);
	FILE* file = fopen("/proc/self/status", "r");
	int result = -1;
	char line[128];
	while (fgets(line, 128, file)) {
		if (strncmp(line, "VmSize:", 7) == 0) {
			result = parseLine(line);
			break;
		}
	}
	fclose(file);
	cout << prefix << ": used memory: " << result << " KB" << endl;
}

int Runner::run()
{
	vector<std::pair<std::string,std::string*> > sequences;
	uintmax_t total_length = 0;
	largest_count = 0;
	Progress progress(files.size(), "Reading in sequences");
	for (auto i = 0; i < files.size(); i++) {
		auto f = files.at(i);
		SingleFileLoader maker(f);

		progress++;
		uint64_t local_largest_count = 0;
		std::pair<std::string,std::string*> pr;
		while ((pr = maker.next()).first != "" && sequences.size() <= 100000) {
			sequences.push_back(pr);
			total_length += pr.second->length();
		}
	}
	progress.end();

	double avg_length = (double)total_length / sequences.size();
	k = std::max((int)(ceil(log(avg_length) / log(4)) - 1), 2);
	cout << "K: " << k << endl;
#pragma omp parallel for reduction(max:largest_count)
	for (size_t i = 0; i < sequences.size(); i++) {
		std::vector<uint64_t> values;
		KmerHashTable<unsigned long, uint64_t> table(k, 1);
		ChromosomeOneDigit chrom(*sequences[i].second, sequences[i].first);
		fill_table<uint64_t>(table, &chrom, values);
		uint64_t l_count = 0;
		for (auto elt : values) {
			if (elt > l_count) {
				l_count = elt;
			}
		}
		if (l_count > largest_count) {
			largest_count = l_count;
		}
		values.clear();
	}
	largest_count *= 2;
	mem_used("sequences read in");
	if (largest_count <= std::numeric_limits<uint8_t>::max()) {
		cout << "Using 8 bit histograms" << endl;
		return do_run<uint8_t>(sequences);
	} else if (largest_count <= std::numeric_limits<uint16_t>::max()) {
		cout << "Using 16 bit histograms" << endl;
		return do_run<uint16_t>(sequences);
	} else if (largest_count <= std::numeric_limits<uint32_t>::max()){
	       	cout << "Using 32 bit histograms" << endl;
		return do_run<uint32_t>(sequences);
	} else if (largest_count <= std::numeric_limits<uint64_t>::max()) {
	       	cout << "Using 64 bit histograms" << endl;
		return do_run<uint64_t>(sequences);
	} else {
		throw "Too big sequence";
	}
}


void usage(std::string progname)
{
	std::cout << "Usage: " << progname << " *.fasta --query queryFile.fasta [--id 0.90] [--mode rc] [--feat fast|slow] [--kmer 3] [--chunk 10000] [--output output_first_string] [--sample 3000] [--threads 4]" << std::endl << std::endl;
	#ifndef VERSION
        #define VERSION "(undefined)"
        #endif
        std::cout << "Version " << VERSION << " compiled on " << __DATE__ << " " << __TIME__;
        #ifdef _OPENMP
        std::cout << " with OpenMP " << _OPENMP;
        #else
        std::cout << " without OpenMP";
        #endif
	std::cout << std::endl;
}

void Runner::get_opts(int argc, char **argv)
{
	for (int i = 1; i < argc; i++) {
		string arg = argv[i];
		if (arg == "--id" && i + 1 < argc) {
			try {
				std::string opt = argv[i+1];
				similarity = std::stod(opt);
				if (similarity <= 0 || similarity >= 1) {
					throw std::invalid_argument("");
				}
			} catch(std::exception e) {
				cerr << "Similarity must be between 0 and 1" << endl;
				exit(EXIT_FAILURE);
			}
			i++;
		} else if ((arg == "-c" || arg == "--chunk") && i + 1 < argc) {
			chunk_size = strtol(argv[i+1], NULL, 10);
			if (errno) {
				perror(argv[i+1]);
				exit(EXIT_FAILURE);
			} else if (chunk_size <= 0) {
				fprintf(stderr, "Chunk size must be greater than 0.\n");
				exit(EXIT_FAILURE);
			}
			i++;
		} else if ((arg == "-k" || arg == "--kmer") && i + 1 < argc) {
			k = strtol(argv[i+1], NULL, 10);
			if (errno) {
				perror(argv[i+1]);
				exit(EXIT_FAILURE);
			} else if (k <= 0) {
				fprintf(stderr, "K must be greater than 0.\n");
				exit(EXIT_FAILURE);
			}
			align = false;
			i++;
		} else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
			output = string(argv[i+1]);
			i++;
		} else if ((arg == "-q" || arg == "--query") && i + 1 < argc) {
			char* qfile = argv[++i];
			struct stat st;
			stat(qfile, &st);
			if (S_ISREG(st.st_mode)) {
				qfiles.emplace_back(qfile);
			} else {
				usage(*argv);
				exit(EXIT_FAILURE);
			}
		} else if (arg == "-r" || arg == "--recover") {
			recover = true;
		} else if ((arg == "-f" || arg == "--feat") && i + 1 < argc) {
			std::string val = argv[++i];
			if (val == "fast") {
				feats = PRED_FEAT_FAST;
			} else if (val == "slow") {
				feats = PRED_FEAT_FAST | PRED_FEAT_DIV;
			} else {
				cerr << "Features must be either \"fast\" or \"slow\"" << endl;
			}
		} else if ((arg == "-m" || arg == "--mode") && i + 1 < argc) {
			std::string val = argv[++i];
			if (val == "c") {
				mode |= PRED_MODE_CLASS;
			} else if (val == "r") {
				mode |= PRED_MODE_REGR;
			} else if (val == "cr" || val == "rc") {
				mode |= PRED_MODE_CLASS | PRED_MODE_REGR;
			} else {
				cerr << "Mode must be either c, r, or a combination" << endl;
				exit(EXIT_FAILURE);
			}
		} else if ((arg == "-s" || arg == "--sample") && i + 1 < argc) {
			sample_size = strtol(argv[i+1], NULL, 10);
			if (errno) {
				perror(argv[i+1]);
				exit(EXIT_FAILURE);
			} else if (sample_size <= 0) {
				fprintf(stderr, "Sample size must be greater than 0.\n");
				exit(EXIT_FAILURE);
			}
			i++;
		} else if ((arg == "-p" || arg == "--pivot") && i + 1 < argc) {
			pivots = strtol(argv[i+1], NULL, 10);
			if (errno) {
				perror(argv[i+1]);
				exit(EXIT_FAILURE);
			} else if (sample_size <= 0) {
				fprintf(stderr, "Points per pivot must be greater than 0.\n");
				exit(EXIT_FAILURE);
			}
			i++;
		} else if ((arg == "-t" || arg == "--threads") && i + 1 < argc) {
			try {
				std::string opt = argv[i+1];
				int threads = std::stoi(opt);
				if (threads <= 0) {
					throw std::invalid_argument("");
				}
				#ifdef _OPENMP
				omp_set_num_threads(threads);
				#endif
			} catch (std::exception e) {
				cerr << "Number of threads must be greater than 0." << endl;
				exit(1);
			}

			i++;

		} else if ((arg == "-d" || arg == "--delta") && i + 1 < argc) {
			delta = strtol(argv[i+1], NULL, 10);
			if (errno) {
				perror(argv[i+1]);
				exit(EXIT_FAILURE);
			} else if (delta <= 0) {
				fprintf(stderr, "Delta must be greater than 0.\n");
				exit(EXIT_FAILURE);
			}
			i++;
		} else if ((arg == "-i" || arg == "--iter" || arg == "--iterations") && i + 1 < argc) {
			iterations = strtol(argv[i+1], NULL, 10);
			if (errno) {
				perror(argv[i+1]);
				exit(EXIT_FAILURE);
			} else if (iterations <= 0) {
				fprintf(stderr, "Iterations must be greater than 0.\n");
				exit(EXIT_FAILURE);
			}
			i++;
		} else if ((arg == "-h") || (arg == "--help")) {
			usage(*argv);
			exit(EXIT_FAILURE);
		} else {
			struct stat st;
			stat(argv[i], &st);
			if (S_ISREG(st.st_mode)) {
				files.push_back(argv[i]);
			} else {
				usage(*argv);
				exit(EXIT_FAILURE);
			}
		}
	}
	if (files.empty()) {
		usage(*argv);
		exit(EXIT_FAILURE);
	}
}

pair<int,uint64_t> Runner::find_k()
{
	mem_used("before find_k");
	std::vector<pair<size_t, uint64_t> > lengths;
        uint64_t max_len = 0, min_len = std::numeric_limits<uint64_t>::max(), tot = 0;
	size_t idx = 0;
	clock_t out_begin = clock();
	for (auto f : files) {
	        SingleFileLoader maker(f);
		unsigned long long l = 0;
		std::pair<std::string,std::string*> chrom;
		clock_t w_begin = clock();
		while ((chrom = maker.next()).second != NULL && *chrom.second != "") {
			auto sz = chrom.second->length();
			lengths.emplace_back(idx++, sz);
			if (sz > max_len) {
				max_len = sz;
			}
			if (sz < min_len) {
				min_len = sz;
			}
			tot += sz;
			delete chrom.second;
		}
		clock_t w_diff = clock() - w_begin;
		cout << "inner loop time: " << (double)w_diff / CLOCKS_PER_SEC << endl;
	}
	clock_t diff = clock() - out_begin;
	cout << "Find_k() loop time: "  << (double)diff / CLOCKS_PER_SEC << endl;
	size_t total_num = 10000;
	std::sort(std::begin(lengths), std::end(lengths), [](pair<size_t, uint64_t> l,
							     pair<size_t, uint64_t> r) {
			  return l.second < r.second;
		  });
	double incr = (double)lengths.size() / total_num;
	size_t last = 10000000000;
	for (double i = 0; round(i) < lengths.size(); i += incr) {
		size_t cur = lengths[round(i)].first;
		if (cur != last) {
			indices.push_back(cur);
			last = cur;
		}
	}
	std::sort(indices.begin(), indices.end());
	double avg_length = (double)tot / (double)lengths.size();
	int newk = std::max((int)(ceil(log(avg_length) / log(4)) - 1), 2);
	cout << "avg length: " << avg_length << endl;
	cout << "Recommended K: " << newk << endl;
	cout << "indices.size(): " << indices.size() << endl;
	mem_used("after find_k");
	return make_pair(newk, max_len);
}


double global_mat[4][4] = {{1, -1, -1, -1},
			   {-1, 1, -1, -1},
			   {-1, -1, 1, -1},
			   {-1, -1, -1, 1}};
double global_sigma = -2;
double global_epsilon = -1;

template<class T>
long bin_search(const std::vector<Point<T>*> &points, size_t begin, size_t last, size_t length)
{
	if (last < begin) {
		return 0;
	}
	size_t idx = begin + (last - begin) / 2;
	if (points.at(idx)->get_length() == length) {
		while (idx > 0 && points[idx-1]->get_length() == length) {
			idx--;
		}
		return idx;
	} else if (points.at(idx)->get_length() > length) {
		if (begin == idx) { return idx; }
		return bin_search(points, begin, idx-1, length);
	} else {
		return bin_search(points, idx+1, last, length);
	}
}


template<class T>
void work(const std::vector<Point<T>*> &queries, const std::vector<Point<T>*> &pts, double similarity, Predictor<T>* pred, std::string delim, std::string output, uintmax_t filenum)
{
	if (pts.empty()) {
		return;
	}
	uint8_t mode = pred->get_mode();
	std::ostringstream oss;
	oss << output << filenum;
	std::ofstream out(oss.str());
	for (auto query : queries) {
		size_t q_len = query->get_length();
		size_t begin_length = q_len * similarity;
		size_t end_length = q_len / similarity;
		size_t start = bin_search(pts, 0, pts.size()-1,
					  begin_length);

		for (size_t i = start;
		     i < pts.size() && pts[i]->get_length() <= end_length;
		     i++) {
			double sim = 0.0;
			bool cls = false;
			if (mode & PRED_MODE_REGR) {
				sim = pred->similarity(pts[i], query);
			}
			if (mode & PRED_MODE_CLASS) {
				cls = pred->close(pts[i], query);
			}
			if (cls || sim > 0) {
				out << query->get_header() << delim << pts[i]->get_header() << delim << cls << delim << sim << endl;
			}
		}
	}
#pragma omp critical
	cout << "Wrote to " << oss.str() << endl;
}

template<class T>
int Runner::do_run(std::vector<std::pair<std::string,std::string*> > &seqs)
{
	using pvec = vector<Point<T> *>;
	using pmap = map<Point<T>*, pvec*>;
	srand(0xFF);
	mem_used("before do_run");
	size_t num_points = 0;
	uintmax_t _id = 0;

	std::vector<Point<T>*> trpoints;
	{


		std::sort(seqs.begin(), seqs.end(), [](std::pair<std::string,std::string*>& a, std::pair<std::string,std::string*>& b) {
				return a.second->length() < b.second->length();
			});
		double incr = (double)seqs.size() / 10000.0;
		size_t last = 10000000000;
		for (double i = 0; round(i) < seqs.size(); i += incr) {
			size_t cur = seqs[round(i)].second->length();
			if (cur != last) {
				indices.push_back(cur);
				trpoints.push_back(NULL);
				last = cur;
			}
		}
		std::sort(indices.begin(), indices.end());
		#pragma omp parallel for
		for (size_t i = 0; i < indices.size(); i++) {
			auto chrom = seqs[indices[i]];
			Point<T>* p = Loader<T>::get_point(chrom.first, *chrom.second, _id, k);
			trpoints[i] = p;
		}
		for (auto p : seqs) {
			delete p.second;
		}
		seqs.clear();
	}
	indices.clear();
	ClusterFactory<T> factory(k);
	mem_used("after selection");
	cout << "TRpoints.size(): " << trpoints.size() << endl;

	std::sort(trpoints.begin(), trpoints.end(), [](const Point<T>* a, const Point<T>* b) {
			return a->get_length() < b->get_length(); });

	int n_threads = omp_get_max_threads();
	Predictor<T> *pred = NULL;
	if (recover) {
		pred = new Predictor<T>("predictor.save");

	} else {
		if (mode == 0) {
			cout << "No mode specified, using regression and classification by default" << endl;
			mode = PRED_MODE_REGR | PRED_MODE_CLASS;
		}
		if (feats == 0) {
			cout << "No feature set specified, using fast features by default" << endl;
			feats = PRED_FEAT_FAST;
		}
		if ((mode & PRED_MODE_CLASS) == PRED_MODE_CLASS && similarity < 0) {
			cout << "Classification specified, but no identity score given. Please supply a cutoff with \"--id\"" << endl;
			exit(EXIT_FAILURE);
		} else if (similarity < 0) {
			similarity = 0.9;
		}

		pred = new Predictor<T>(k, similarity, mode, feats, 4);
		auto before = clock();
		mem_used("before predictor training");
		vector<Point<T>*> qpoints;
		pred->train(trpoints, qpoints, _id, sample_size);
		double elapsed = (clock() - before);
		elapsed /= CLOCKS_PER_SEC;
		cout << "Training time: " << elapsed << endl;
		for (auto p : trpoints) {
			delete p;
		}
		for (auto q : qpoints) {
			delete q;
		}
		trpoints.clear();
	}
	mem_used("after predictor training");


	string delim = "!";
	uint64_t query_id_start = num_points;
	int num_query = num_points;
	Loader<T> qloader(qfiles, n_threads * num_points, chunk_size, 1, k, query_id_start);
	mem_used("before loop");
	uintmax_t file_num = 0;
	while (!qloader.done()) {
		qloader.preload(0);
		auto queries = qloader.load_next(0);
		Loader<T> loader(files, 0, chunk_size, n_threads, k);


		while (!loader.done()) {
			int n_iter = n_threads;
			mem_used("during inner loop");
			for (int h = 0; h < n_iter; h++) {
				loader.preload(h);
			}
			#pragma omp parallel for
			for (int h = 0; h < n_iter; h++) {
				int tid = omp_get_thread_num();
				auto pts = loader.load_next(tid);
				std::sort(std::begin(pts), std::end(pts), [](Point<T>*a, Point<T>*b) {
						return a->get_length() < b->get_length();
					});
				work(queries, pts, similarity, pred, delim, output, file_num + h);
				for (auto p : pts) {
					delete p;
				}
			}
			file_num += n_iter;
		}

		for (auto q : queries) {
			delete q;
		}
		mem_used("mid loop");
	}
	mem_used("after loop");
	return 0;
}


template<class T>
void Runner::print_output(const map<Point<T>*, vector<Point<T>*>*> &partition) const
{
	cout << "Printing output" << endl;
	std::ofstream ofs;
	ofs.open(output, std::ofstream::out);
	int counter = 0;
	for (auto const& kv : partition) {
		if (kv.second->size() == 0) {
			continue;
		}
		ofs << ">Cluster " << counter << endl;
		int pt = 0;
		for (auto p : *kv.second) {
			string s = p->get_header();
			ofs << pt << "\t"  << p->get_length() << "nt, " << s << "... " << endl;
			pt++;
		}
		counter++;
	}
	ofs.close();
}
