/* -*- C++ -*-
 *
 * SimpleLoader.cpp
 *
 * Author: Benjamin T James
 *
 */
#include "SimpleLoader.h"
#include "Loader.h"

template<class T>
void SimpleLoader<T>::load_next(int len, std::vector<Point<T>*>& points, bool set_seq)
{
	std::vector<std::pair<std::string,std::string*> > buffer;

	clockIO.begin();
	buffer.reserve(len);
	for (int i = 0; i < len && !done(); i++) {
		std::pair<std::string, std::string*> val = next();
		if (val.first != "") {
			file_sizes.at(file_idx)++;
			buffer.push_back(val);
		}
	}
	clockIO.end();
	clockHist.begin();
	points.resize(buffer.size(), NULL);
	#pragma omp parallel for
	for (int i = 0; i < buffer.size(); i++) {

		points.at(i) = Loader<T>::get_point(buffer.at(i).first,
						    *buffer.at(i).second,
						    index,
						    k,
						    set_seq);
		delete buffer.at(i).second;
	}
	clockHist.end();
}

template<class T>
SimpleLoader<T>::SimpleLoader(const SimpleLoader<T>& loader) : file_list(loader.get_file_list()), k(loader.get_k())
{
	file_idx = loader.get_file_idx();
	file_sizes = loader.get_file_sizes();
	if (file_idx < file_list.size()) {
		SingleFileLoader* loader_maker = loader.get_maker();
		maker = new SingleFileLoader(file_list.at(file_idx), *loader_maker);
		// for (size_t i = 0; !done() && i < file_sizes.at(file_idx); i++) {
		// 	auto pr = next();
		// 	delete pr.second;
		// }
	}

}

template<class T>
void SimpleLoader<T>::seek(std::vector<size_t> p_file_sizes, size_t pos, bool is_done)
{
	size_t cur_pos = 0;
	for (size_t i = 0; i < p_file_sizes.size(); i++) {
		size_t f_size = p_file_sizes[i];
		if (cur_pos <= pos < cur_pos + f_size) {
			if (i != file_idx) {
				if (maker != NULL) {
					delete maker;
				}
				file_idx = i;
				maker = new SingleFileLoader(file_list.at(i));
				file_sizes = p_file_sizes;
			}
			break;
		}
		cur_pos += f_size;
	}
	if (is_done) {
		file_idx = file_list.size();
	}
	for (size_t i = cur_pos; i < pos && !done(); i++) {
		auto pr = next();
		delete pr.second;
	}

}
template<class T>
bool SimpleLoader<T>::done() const
{
	return file_idx == file_list.size();
}

template<class T>
std::pair<std::string,std::string*> SimpleLoader<T>::next()
{
	auto n = maker->next();
	if (n.first != "") {
		return n;
	}
	delete maker;
	maker = NULL;
	file_idx++;

	if (file_idx >= file_list.size()) {
		return n;
	}
	maker = new SingleFileLoader(file_list.at(file_idx));
	file_sizes.push_back(0);
	return maker->next();
}

template class SimpleLoader<uint8_t>;
template class SimpleLoader<uint16_t>;
template class SimpleLoader<uint32_t>;
template class SimpleLoader<uint64_t>;
template class SimpleLoader<int>;
template class SimpleLoader<double>;
