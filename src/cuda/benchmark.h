#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <cmath>

// Benchmark configuration
struct BenchmarkConfig {
    int warmupFrames = 50;      // Frames to skip before measuring
    int measureFrames = 100;    // Frames to measure
    bool verbose = true;        // Print per-frame info
    std::string outputFile = "benchmark_results.csv";
};

// Per-frame timing data
struct FrameTiming {
    float cullMs;
    float transformMs;
    float duplicateMs;
    float sortMs;
    float rangeMs;
    float blendMs;
    float totalMs;
    int numGaussians;
    int numVisible;
    int numTileEntries;
};

// Statistics for a set of measurements
struct TimingStats {
    float mean;
    float stddev;
    float min;
    float max;
    float median;
    float p95;  // 95th percentile
};

// Compute statistics from a vector of values
inline TimingStats computeStats(std::vector<float>& values) {
    TimingStats stats;
    if (values.empty()) {
        stats = {0, 0, 0, 0, 0, 0};
        return stats;
    }
    
    // Sort for percentiles
    std::sort(values.begin(), values.end());
    
    // Mean
    float sum = std::accumulate(values.begin(), values.end(), 0.0f);
    stats.mean = sum / values.size();
    
    // Stddev
    float sqSum = 0;
    for (float v : values) {
        sqSum += (v - stats.mean) * (v - stats.mean);
    }
    stats.stddev = std::sqrt(sqSum / values.size());
    
    // Min/Max
    stats.min = values.front();
    stats.max = values.back();
    
    // Median
    size_t mid = values.size() / 2;
    stats.median = (values.size() % 2 == 0) 
        ? (values[mid-1] + values[mid]) / 2.0f 
        : values[mid];
    
    // 95th percentile
    size_t p95idx = static_cast<size_t>(values.size() * 0.95);
    stats.p95 = values[std::min(p95idx, values.size() - 1)];
    
    return stats;
}

// Main benchmark tracker
class BenchmarkTracker {
public:
    BenchmarkConfig config;
    std::vector<FrameTiming> timings;
    int frameCount = 0;
    bool measuring = false;
    
    // CUDA events for precise GPU timing
    cudaEvent_t startEvent, endEvent;
    cudaEvent_t cullStart, cullEnd;
    cudaEvent_t transformStart, transformEnd;
    cudaEvent_t duplicateStart, duplicateEnd;
    cudaEvent_t sortStart, sortEnd;
    cudaEvent_t rangeStart, rangeEnd;
    cudaEvent_t blendStart, blendEnd;
    
    void init() {
        cudaEventCreate(&startEvent);
        cudaEventCreate(&endEvent);
        cudaEventCreate(&cullStart);
        cudaEventCreate(&cullEnd);
        cudaEventCreate(&transformStart);
        cudaEventCreate(&transformEnd);
        cudaEventCreate(&duplicateStart);
        cudaEventCreate(&duplicateEnd);
        cudaEventCreate(&sortStart);
        cudaEventCreate(&sortEnd);
        cudaEventCreate(&rangeStart);
        cudaEventCreate(&rangeEnd);
        cudaEventCreate(&blendStart);
        cudaEventCreate(&blendEnd);
        
        timings.reserve(config.measureFrames);
    }
    
    void cleanup() {
        cudaEventDestroy(startEvent);
        cudaEventDestroy(endEvent);
        cudaEventDestroy(cullStart);
        cudaEventDestroy(cullEnd);
        cudaEventDestroy(transformStart);
        cudaEventDestroy(transformEnd);
        cudaEventDestroy(duplicateStart);
        cudaEventDestroy(duplicateEnd);
        cudaEventDestroy(sortStart);
        cudaEventDestroy(sortEnd);
        cudaEventDestroy(rangeStart);
        cudaEventDestroy(rangeEnd);
        cudaEventDestroy(blendStart);
        cudaEventDestroy(blendEnd);
    }
    
    void startFrame() {
        frameCount++;
        if (frameCount == config.warmupFrames) {
            measuring = true;
            std::cout << "\n=== Starting benchmark measurements ===" << std::endl;
        }
        if (measuring && timings.size() < config.measureFrames) {
            cudaEventRecord(startEvent);
        }
    }
    
    void recordCullStart() { if (measuring) cudaEventRecord(cullStart); }
    void recordCullEnd() { if (measuring) cudaEventRecord(cullEnd); }
    void recordTransformStart() { if (measuring) cudaEventRecord(transformStart); }
    void recordTransformEnd() { if (measuring) cudaEventRecord(transformEnd); }
    void recordDuplicateStart() { if (measuring) cudaEventRecord(duplicateStart); }
    void recordDuplicateEnd() { if (measuring) cudaEventRecord(duplicateEnd); }
    void recordSortStart() { if (measuring) cudaEventRecord(sortStart); }
    void recordSortEnd() { if (measuring) cudaEventRecord(sortEnd); }
    void recordRangeStart() { if (measuring) cudaEventRecord(rangeStart); }
    void recordRangeEnd() { if (measuring) cudaEventRecord(rangeEnd); }
    void recordBlendStart() { if (measuring) cudaEventRecord(blendStart); }
    void recordBlendEnd() { if (measuring) cudaEventRecord(blendEnd); }
    
    void endFrame(int numGaussians, int numVisible, int numTileEntries) {
        if (!measuring || timings.size() >= config.measureFrames) return;
        
        cudaEventRecord(endEvent);
        cudaEventSynchronize(endEvent);
        
        FrameTiming timing;
        cudaEventElapsedTime(&timing.cullMs, cullStart, cullEnd);
        cudaEventElapsedTime(&timing.transformMs, transformStart, transformEnd);
        cudaEventElapsedTime(&timing.duplicateMs, duplicateStart, duplicateEnd);
        cudaEventElapsedTime(&timing.sortMs, sortStart, sortEnd);
        cudaEventElapsedTime(&timing.rangeMs, rangeStart, rangeEnd);
        cudaEventElapsedTime(&timing.blendMs, blendStart, blendEnd);
        cudaEventElapsedTime(&timing.totalMs, startEvent, endEvent);
        
        timing.numGaussians = numGaussians;
        timing.numVisible = numVisible;
        timing.numTileEntries = numTileEntries;
        
        timings.push_back(timing);
        
        if (config.verbose && timings.size() % 10 == 0) {
            std::cout << "Benchmark frame " << timings.size() << "/" << config.measureFrames 
                      << " - " << std::fixed << std::setprecision(2) << timing.totalMs << " ms" << std::endl;
        }
        
        if (timings.size() >= config.measureFrames) {
            printResults();
            saveResults();
        }
    }
    
    void printResults() {
        std::cout << "\n========================================" << std::endl;
        std::cout << "       BENCHMARK RESULTS" << std::endl;
        std::cout << "========================================" << std::endl;
        
        // Collect per-stage timings
        std::vector<float> cullTimes, transformTimes, duplicateTimes, sortTimes, rangeTimes, blendTimes, totalTimes;
        for (const auto& t : timings) {
            cullTimes.push_back(t.cullMs);
            transformTimes.push_back(t.transformMs);
            duplicateTimes.push_back(t.duplicateMs);
            sortTimes.push_back(t.sortMs);
            rangeTimes.push_back(t.rangeMs);
            blendTimes.push_back(t.blendMs);
            totalTimes.push_back(t.totalMs);
        }
        
        auto cullStats = computeStats(cullTimes);
        auto transformStats = computeStats(transformTimes);
        auto duplicateStats = computeStats(duplicateTimes);
        auto sortStats = computeStats(sortTimes);
        auto rangeStats = computeStats(rangeTimes);
        auto blendStats = computeStats(blendTimes);
        auto totalStats = computeStats(totalTimes);
        
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "\nTiming Breakdown (ms):" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        std::cout << std::setw(15) << "Stage" << std::setw(10) << "Mean" << std::setw(10) << "Std" 
                  << std::setw(10) << "Min" << std::setw(10) << "Max" << std::setw(10) << "%" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        
        auto printRow = [&](const char* name, TimingStats& stats) {
            float pct = (stats.mean / totalStats.mean) * 100.0f;
            std::cout << std::setw(15) << name << std::setw(10) << stats.mean << std::setw(10) << stats.stddev
                      << std::setw(10) << stats.min << std::setw(10) << stats.max << std::setw(9) << pct << "%" << std::endl;
        };
        
        printRow("Cull", cullStats);
        printRow("Transform", transformStats);
        printRow("Duplicate", duplicateStats);
        printRow("Sort", sortStats);
        printRow("Tile Ranges", rangeStats);
        printRow("Blend", blendStats);
        std::cout << "----------------------------------------" << std::endl;
        printRow("TOTAL", totalStats);
        
        std::cout << "\nPerformance:" << std::endl;
        std::cout << "  Mean FPS: " << std::setprecision(1) << 1000.0f / totalStats.mean << std::endl;
        std::cout << "  Min FPS:  " << 1000.0f / totalStats.max << std::endl;
        std::cout << "  Max FPS:  " << 1000.0f / totalStats.min << std::endl;
        
        std::cout << "\nWorkload:" << std::endl;
        std::cout << "  Gaussians: " << timings[0].numGaussians << std::endl;
        std::cout << "  Visible: " << timings[0].numVisible << " (" 
                  << std::setprecision(1) << (100.0f * timings[0].numVisible / timings[0].numGaussians) << "%)" << std::endl;
        std::cout << "  Tile entries: " << timings[0].numTileEntries << " ("
                  << std::setprecision(1) << (float)timings[0].numTileEntries / timings[0].numVisible << " tiles/Gaussian avg)" << std::endl;
        
        std::cout << "========================================\n" << std::endl;
    }
    
    void saveResults() {
        std::ofstream file(config.outputFile);
        if (!file.is_open()) {
            std::cerr << "Failed to open " << config.outputFile << " for writing" << std::endl;
            return;
        }
        
        // Header
        file << "frame,cull_ms,transform_ms,duplicate_ms,sort_ms,range_ms,blend_ms,total_ms,num_gaussians,num_visible,num_tile_entries\n";
        
        // Data
        for (size_t i = 0; i < timings.size(); i++) {
            const auto& t = timings[i];
            file << i << "," << t.cullMs << "," << t.transformMs << "," << t.duplicateMs << ","
                 << t.sortMs << "," << t.rangeMs << "," << t.blendMs << "," << t.totalMs << ","
                 << t.numGaussians << "," << t.numVisible << "," << t.numTileEntries << "\n";
        }
        
        file.close();
        std::cout << "Results saved to " << config.outputFile << std::endl;
    }
    
    bool isComplete() const {
        return timings.size() >= config.measureFrames;
    }
};

// Global benchmark instance
extern BenchmarkTracker g_benchmark;

