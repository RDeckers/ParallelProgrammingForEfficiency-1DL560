#include <time.h>
#ifdef _MSC_VER
struct timespec {
	time_t   tv_sec;        /* seconds */
	long     tv_nsec;       /* nanoseconds */
};
#endif
//returns the time passed between start and end in ns.
double time_diff(struct timespec *start, struct timespec *end);
//returns the time in ns elapsed since T.
double elapsed_since(struct timespec *T);
//gettime wrapper
void tick(struct timespec *T);
//updates T and returns the elapsed ns.
double tock(struct timespec *T);