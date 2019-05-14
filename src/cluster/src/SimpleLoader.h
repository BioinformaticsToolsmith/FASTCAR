/* -*- C++ -*-
 *
 * SimpleLoader.h
 *
 * Author: Benjamin T James
 *
 */
#ifndef SIMPLELOADER_H
#define SIMPLELOADER_H

#include "SingleFileLoader.h"
#include "Point.h"
#include "../../train/Clock.h"

template<class T>
class SimpleLoader {
public:
	SimpleLoader(std::vector<std::string> files_, int k_) : k(k_), file_list(files_) {
		if (file_list.size() == 0) {
			std::cerr << "No files were passed" << std::endl;
			throw std::exception();
		}
		maker = new SingleFileLoader(file_list.at(0));
		file_sizes.push_back(0);
		// for (size_t i = 0; i < start_index && !done(); i++) {
		// 	auto pr = next();
		// 	delete pr.second;
		// }
	}
	~SimpleLoader() {
		if (maker != NULL) {
			delete maker;
		}
		cout << "IO time: " << clockIO.total() << endl;
		cout << "Histogram time: " << clockHist.total() << endl;
	};
	SimpleLoader(const SimpleLoader<T>& c);
	/* Go to pos sequences after the start according to the file sizes passed */
	void seek(std::vector<size_t> file_sizes, size_t pos, bool is_done);
	void load_next(int len, std::vector<Point<T>*> & points, bool set_seq=true);
	bool done() const;

	size_t get_file_idx() const { return file_idx; };
	std::vector<size_t> get_file_sizes() const { return file_sizes; };
	std::vector<std::string> get_file_list() const { return file_list; };
	int get_k() const { return k; }

	SingleFileLoader* get_maker() const { return maker; }
private:
	Clock clockIO, clockHist;
	std::pair<std::string,std::string*> next();

	const std::vector<std::string> file_list;
	std::vector<size_t> file_sizes;

	size_t file_idx = 0;
	uintmax_t index = 0;
	SingleFileLoader *maker = NULL;
	const int k;


};
#endif
