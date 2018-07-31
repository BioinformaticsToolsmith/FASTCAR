/* -*- C++ -*-
 *
 * main.cpp
 *
 * Author: Benjamin T James
 */
#include "Runner.h"
#include <sys/resource.h>
int main(int argc, char **argv)
{
	Runner runner(argc, argv);
	return runner.run();
}
