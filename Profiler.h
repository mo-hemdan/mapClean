#pragma once
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <sys/resource.h>
#include <sys/time.h>
#include <iomanip>

class ScopeProfiler {
public:
    ScopeProfiler(const std::string& name)
        : name_(name), startWall_(std::chrono::high_resolution_clock::now()) {
        startCPU_ = getCPUTime();
        startMemKB_ = getMemoryUsageKB();
    }

    ~ScopeProfiler() {
        double endCPU = getCPUTime();
        auto endWall = std::chrono::high_resolution_clock::now();
        size_t endMemKB = getMemoryUsageKB();

        double wallTime = std::chrono::duration<double>(endWall - startWall_).count();
        double cpuTime = endCPU - startCPU_;
        double cpuUtil = (wallTime > 0)
            ? (cpuTime / wallTime) * 100.0
            : 0.0;

        std::cout << "=== Profiling: " << name_ << " ===\n";
        std::cout << "Wall time: " << wallTime << " s\n";
        std::cout << "CPU time: " << cpuTime << " s\n";
        std::cout << "CPU utilization: " << cpuUtil << " %\n";
        std::cout << "Memory Before: " << formatMemory(startMemKB_) << "\n";
        std::cout << "Memory After: " << formatMemory(endMemKB) << "\n";
        std::cout << "Memory change: " << formatMemory(endMemKB - startMemKB_) << "\n";
        std::cout << "===================================\n";
    }

private:
    std::string name_;
    std::chrono::time_point<std::chrono::high_resolution_clock> startWall_;
    double startCPU_;
    size_t startMemKB_;

    double getCPUTime() {
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        return (usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1e6) +
               (usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1e6);
    }

    size_t getMemoryUsageKB() {
        std::ifstream file("/proc/self/status");
        std::string line;
        while (std::getline(file, line)) {
            if (line.rfind("VmRSS:", 0) == 0) { // Resident Set Size
                std::istringstream iss(line);
                std::string key, unit;
                size_t value;
                iss >> key >> value >> unit;
                return value; // in KB
            }
        }
        return 0;
    }

    std::string formatMemory(size_t kb) {
        static const char* units[] = {"KB", "MB", "GB", "TB"};
        double size = static_cast<double>(kb);
        int unitIndex = 0;
        while (size >= 1024.0 && unitIndex < 3) {
            size /= 1024.0;
            unitIndex++;
        }
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << size << " " << units[unitIndex];
        return oss.str();
    }
};
