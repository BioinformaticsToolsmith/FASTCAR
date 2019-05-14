/* -*- C++ -*-
 *
 * SingleFileLoader.h
 *
 * Author: Benjamin T James
 *
 * A way of reading in 1 sequence at a time
 * from FASTA, sequence is heap allocated
 */
#ifndef SINGLEFILELOADER_H
#define SINGLEFILELOADER_H

#include <fstream>
#include "../../nonltr/ChromosomeOneDigitDna.h"
class SingleFileLoader {
public:
	SingleFileLoader(std::string file);
	SingleFileLoader(std::string file, const SingleFileLoader& loader);
	~SingleFileLoader() {
		if (in != NULL) {
			delete in;
		}
	}
	std::pair<std::string,std::string*> next();
	ChromosomeOneDigitDna* nextChrom();

	intmax_t get_position() const { return in->tellg(); }
	std::string get_buffer() const { return buffer; }
	bool get_is_first() const { return is_first; };
private:
	std::ifstream *in;
	std::string buffer;
	bool is_first;
};
#endif
