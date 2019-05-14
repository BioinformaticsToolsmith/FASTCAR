// -*- C++ -*-
/*
 * Clock.h
 *
 * Author: Benjamin T James
 */

#ifndef CLOCK_H
#define CLOCK_H
#include <iostream>
#include <chrono>

class Clock {
public:
	Clock(bool do_begin=true) { if (do_begin) { begin(); }};
	void begin();
	void end();
	double total() const;
	static void stamp(std::string desc);

private:
	std::chrono::time_point<std::chrono::system_clock> time;
	double accum = 0;


};
#endif
