#ifndef args_h
#define args_h
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <getopt.h>

namespace exafmm_t {
  static struct option long_options[] = {
    {"ncrit",        required_argument, 0, 'c'},
    {"numBodies",    required_argument, 0, 'n'},
    {"P",            required_argument, 0, 'P'},
    {"threads",      required_argument, 0, 'T'},
    {"distribution", required_argument, 0, 'd'},
    {"maxlevel",     required_argument, 0, 'l'},
    {"wavenumber",   required_argument, 0, 'k'},
    {0, 0, 0, 0}
  };

  class Args {
  public:
    int ncrit;
    int numBodies;
    int P;
    int threads;
    int maxlevel;
    double k;
    const char * distribution;

  private:
    void usage(char * name) {
      fprintf(stderr,
	      "Usage: %s [options]\n"
	      "Long option (short option)     : Description (Default value)\n"
	      " --ncrit (-c)                  : Number of bodies per leaf node (%d)\n"
              " --distribution (-d) [c/s/p]   : cube, sphere, plummer (%s)\n"
	      " --numBodies (-n)              : Number of bodies (%d)\n"
	      " --P (-P)                      : Order of expansion (%d)\n"
	      " --threads (-T)                : Number of threads (%d)\n"
              " --maxlevel (-l)               : Max level of tree (%d) (only applies to non-adaptive tree)\n"
              " --wavenumber (-k)             : Wavenumber of Helmholtz kernel (%f)\n",
	      name,
	      ncrit,
              distribution,
	      numBodies,
	      P,
	      threads,
              maxlevel,
              k);
    }

    const char * parseDistribution(const char * arg) {
      switch (arg[0]) {
        case 'c': return "cube";
        case 's': return "sphere";
        case 'p': return "plummer";
        default:
          fprintf(stderr, "invalid distribution %s\n", arg);
          abort();
      }
      return "";
    }

  public:
    Args(int argc=0, char ** argv=nullptr) :
      ncrit(64),
      numBodies(1000000),
      P(4),
      threads(16),
      maxlevel(5),
      distribution("cube"),
      k(20) {
      while (1) {
	int option_index;
	int c = getopt_long(argc, argv, "c:d:n:P:T:l:k:", long_options, &option_index);
	if (c == -1) break;
	switch (c) {
	case 'c':
	  ncrit = atoi(optarg);
	  break;
        case 'd':
          distribution = parseDistribution(optarg);
          break;
	case 'n':
	  numBodies = atoi(optarg);
	  break;
	case 'P':
	  P = atoi(optarg);
	  break;
	case 'T':
	  threads = atoi(optarg);
	  break;
        case 'l':
          maxlevel = atoi(optarg);
          break;
        case 'k':
          k = atof(optarg);
          break;
	default:
	  usage(argv[0]);
	  exit(0);
	}
      }
    }

    void print(int stringLength=20) {
      std::cout << std::setw(stringLength) << std::fixed << std::left
                << "ncrit" << " : " << ncrit << std::endl
                << std::setw(stringLength)
                << "distribution" << " : " << distribution << std::endl
                << std::setw(stringLength)
                << "numBodies" << " : " << numBodies << std::endl
                << std::setw(stringLength)
                << "P" << " : " << this->P << std::endl
                << std::setw(stringLength)
                << "maxlevel" << " : " << maxlevel << std::endl
                << std::setw(stringLength)
                << "threads" << " : " << threads << std::endl
                << std::setw(stringLength)
                << "wavenumber" << " : " << k << std::endl;
    }
  };
}
#endif
