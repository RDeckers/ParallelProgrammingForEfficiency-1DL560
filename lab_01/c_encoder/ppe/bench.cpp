#include "bench.h"
#ifdef _MSC_VER
#include <Windows.h>
//http://stackoverflow.com/a/38212960
#define CLOCK_MONOTONIC 1

int clock_gettime(int dummy, struct timespec *ct) {
	static BOOL g_first_time = 1;
	static LARGE_INTEGER g_counts_per_sec;
	LARGE_INTEGER count;

	if (g_first_time) {
		g_first_time = 0;

		if (0 == QueryPerformanceFrequency(&g_counts_per_sec)) {
			g_counts_per_sec.QuadPart = 0;
		}
	}

	if ((NULL == ct) || (g_counts_per_sec.QuadPart <= 0) ||
		(0 == QueryPerformanceCounter(&count))) {
		return -1;
	}

	ct->tv_sec = count.QuadPart / g_counts_per_sec.QuadPart;
	ct->tv_nsec = ((count.QuadPart % g_counts_per_sec.QuadPart) * 1e9) / g_counts_per_sec.QuadPart;

	return 0;
}
#endif

double time_diff(struct timespec *start, struct timespec *end) {
	struct timespec tmp;
	if ((end->tv_nsec - start->tv_nsec)<0) {
		tmp.tv_sec = end->tv_sec - start->tv_sec - 1;
		tmp.tv_nsec = 1000000000 + end->tv_nsec - start->tv_nsec;
	} else {
		tmp.tv_sec = end->tv_sec - start->tv_sec;
		tmp.tv_nsec = end->tv_nsec - start->tv_nsec;
	}
	return (double)tmp.tv_sec*1e9 + (double)tmp.tv_nsec;
}

double elapsed_since(struct timespec *T) {
	struct timespec tmp;
	clock_gettime(CLOCK_MONOTONIC, &tmp);
	return time_diff(T, &tmp);
}

double tock(struct timespec *T) {
	struct timespec tmp;
	tick(&tmp);
	double elapsed = time_diff(T, &tmp);
	*T = tmp;
	return elapsed;
}

void tick(struct timespec *T) {
	clock_gettime(CLOCK_MONOTONIC, T);
}