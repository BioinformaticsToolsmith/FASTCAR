/* -*- C++ -*-
 *
 * Clock.cpp
 *
 * Author: Benjamin T James
 */

#include "Clock.h"
#include <chrono>
#include <ctime>

static const auto _begin = std::chrono::system_clock::now();

void Clock::begin()
{
	time = std::chrono::system_clock::now();
}

void Clock::end()
{
	auto start = time;
	begin();
	std::chrono::duration<double> diff = time - start;
	accum += diff.count();
}

double Clock::total() const
{
	return accum;
}

void Clock::stamp(std::string desc)
{
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - _begin;
	std::cout << "timestamp " << desc << " " << diff.count() << std::endl;
}
