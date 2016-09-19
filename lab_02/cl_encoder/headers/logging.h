#include <stdio.h>
#ifndef _UTIL_LOG_H
#define _UTIL_LOG_H
#ifdef __cplusplus
extern "C"{
#endif

	typedef enum {
		PASS,
		INFO,
		WARN,
		FAIL
	}report_level_t;

	extern FILE* DEFAULT_REPORT_STREAM;
	extern int REPORT_TIMESTAMP;
	extern int REPORT_W_COLORS;

#ifdef _MSC_VER 
	void report(report_level_t level, const char* format, ...);
	void reportf(report_level_t level, int timestamped, int colored, const char* format, ...);
	void freport(FILE* stream, report_level_t level, const char* format, ...);
	void freportf(FILE* stream, report_level_t level, int timestamped, int colored, const char* format, ...);
#else 
	void report(report_level_t level, const char* format, ...) __attribute__((format(printf, 2, 3)));
	void reportf(report_level_t level, int timestamped, int colored, const char* format, ...) __attribute__((format(printf, 4, 5)));
	void freport(FILE* stream, report_level_t level, const char* format, ...) __attribute__((format(printf, 3, 4)));
	void freportf(FILE* stream, report_level_t level, int timestamped, int colored, const char* format, ...) __attribute__((format(printf, 5, 6)));
#endif
#ifdef __cplusplus
}
#endif
#endif
