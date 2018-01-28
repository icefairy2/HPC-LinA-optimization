#ifndef STOPWATCH_H
#define STOPWATCH_H

#include <time.h>

class Stopwatch {
private:
  long long m_time;
  struct timespec m_start;

  long long difftime(struct timespec end) {
    return 1000000000L * (end.tv_sec - m_start.tv_sec) + end.tv_nsec - m_start.tv_nsec;
  }
  
  double seconds(long long time) {
    return 1.0e-9 * time;
  }
  
public:
  Stopwatch() : m_time(0) {}
  
  void reset() {
    m_time = 0;
  }

  void start() {
    clock_gettime(CLOCK_MONOTONIC, &m_start);
  }
  
  double split() {
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    return seconds(difftime(end));
  }
  
  double pause() {    
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    m_time += difftime(end);
    return seconds(m_time);
  }

  double stop() {
    double time = pause();
    reset();
    return time;
  }
};

#endif // STOPWATCH_H
