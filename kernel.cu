#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <climits>
#include <chrono>
#include <iomanip>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__   \
                      << " - " << cudaGetErrorString(err) << std::endl;    \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

// ---------------- kernels ----------------
__global__ void rowTop2SumKernel(const int* matrix, int* rowSums, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    int max1 = INT_MIN, max2 = INT_MIN;
    for (int col = 0; col < n; ++col) {
        int val = matrix[row * n + col];
        if (val >= max1) {
            max2 = max1;
            max1 = val;
        }
        else if (val > max2) {
            max2 = val;
        }
    }
    rowSums[row] = max1 + max2;
}

__global__ void colTop2SumKernel(const int* matrix, int* colSums, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;

    int max1 = INT_MIN, max2 = INT_MIN;
    for (int row = 0; row < n; ++row) {
        int val = matrix[row * n + col];
        if (val >= max1) {
            max2 = max1;
            max1 = val;
        }
        else if (val > max2) {
            max2 = val;
        }
    }
    colSums[col] = max1 + max2;
}

// ---------------- CPU ----------------
void cpuRowTop2Sum(const std::vector<int>& matrix, std::vector<int>& out, int n) {
    for (int row = 0; row < n; ++row) {
        int m1 = INT_MIN, m2 = INT_MIN;
        for (int col = 0; col < n; ++col) {
            int v = matrix[row * n + col];
            if (v >= m1) { m2 = m1; m1 = v; }
            else if (v > m2) { m2 = v; }
        }
        out[row] = m1 + m2;
    }
}

void cpuColTop2Sum(const std::vector<int>& matrix, std::vector<int>& out, int n) {
    for (int col = 0; col < n; ++col) {
        int m1 = INT_MIN, m2 = INT_MIN;
        for (int row = 0; row < n; ++row) {
            int v = matrix[row * n + col];
            if (v >= m1) { m2 = m1; m1 = v; }
            else if (v > m2) { m2 = v; }
        }
        out[col] = m1 + m2;
    }
}

// ---------------- helpers ----------------
void printMatrix(const std::vector<int>& m, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            std::cout << std::setw(4) << m[i * n + j];
        std::cout << "\n";
    }
}

void printHeader(const int n) {
    std::cout << "\n========================================\n";
    std::cout << "Matrix size: " << n << " x " << n << "\n";
}

void printVec(const std::vector<int>& v, const char* label) {
    std::cout << label;
    for (auto x : v) std::cout << x << " ";
    std::cout << "\n";
}

// ---------------- main ----------------
int main() {
    std::srand((unsigned)std::time(nullptr));

    // sample list of matric sizes to test; smaller ones can be faster on CPU
    std::vector<int> sizes = { 8, 1024, 5000, 10000 };

    // warmup kernel (avoids first-launch penalty)
    rowTop2SumKernel << <1, 1 >> > (nullptr, nullptr, 0);
    cudaDeviceSynchronize();

    for (int n : sizes) {
        printHeader(n);

        size_t matrixBytes = (size_t)n * n * sizeof(int);
        size_t vectorBytes = (size_t)n * sizeof(int);

        std::vector<int> h_matrix(n * n);
        std::vector<int> h_rowCpu(n), h_colCpu(n);
        std::vector<int> h_rowGpu(n), h_colGpu(n);

        for (auto& x : h_matrix) x = std::rand() % 100;

        int* d_matrix, * d_row, * d_col;
        CUDA_CHECK(cudaMalloc(&d_matrix, matrixBytes));
        CUDA_CHECK(cudaMalloc(&d_row, vectorBytes));
        CUDA_CHECK(cudaMalloc(&d_col, vectorBytes));

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // ---------- CPU ----------
        auto c1 = std::chrono::high_resolution_clock::now();
        cpuRowTop2Sum(h_matrix, h_rowCpu, n);
        auto c2 = std::chrono::high_resolution_clock::now();
        cpuColTop2Sum(h_matrix, h_colCpu, n);
        auto c3 = std::chrono::high_resolution_clock::now();

        double cpuRow = std::chrono::duration<double, std::milli>(c2 - c1).count();
        double cpuCol = std::chrono::duration<double, std::milli>(c3 - c2).count();

        // ---------- GPU ----------
        float h2d, rowMs, colMs, d2h;

        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(d_matrix, h_matrix.data(), matrixBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&h2d, start, stop));

        int threads = 256;
        int blocks = (n + threads - 1) / threads;

        CUDA_CHECK(cudaEventRecord(start));
        rowTop2SumKernel << <blocks, threads >> > (d_matrix, d_row, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&rowMs, start, stop));

        CUDA_CHECK(cudaEventRecord(start));
        colTop2SumKernel << <blocks, threads >> > (d_matrix, d_col, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&colMs, start, stop));

        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(h_rowGpu.data(), d_row, vectorBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_colGpu.data(), d_col, vectorBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&d2h, start, stop));

        // ---------- optional debug ----------
        if (n <= 16) {
            std::cout << "\nMatrix:\n";
            printMatrix(h_matrix, n);

            printVec(h_rowGpu, "Row GPU: ");
            printVec(h_rowCpu, "Row CPU: ");
            printVec(h_colGpu, "Col GPU: ");
            printVec(h_colCpu, "Col CPU: ");
        }

        // ---------- results ----------
        std::cout << "\nTiming (ms)\n";
        std::cout << std::fixed << std::setprecision(3);

        std::cout << std::left
            << std::setw(16) << "Operation"
            << std::setw(16) << "CPU (ms)"
            << std::setw(16) << "GPU (ms)"
            << std::setw(16) << "Speedup"
            << "\n";

        std::cout << std::string(64, '-') << "\n";

        std::cout << std::left
            << std::setw(16) << "Row"
            << std::setw(16) << cpuRow
            << std::setw(16) << rowMs
            << std::setw(16) << (cpuRow / rowMs)
            << "x\n";

        std::cout << std::left
            << std::setw(16) << "Column"
            << std::setw(16) << cpuCol
            << std::setw(16) << colMs
            << std::setw(16) << (cpuCol / colMs)
            << "x\n";

        std::cout << "\nTransfer:\n";
        std::cout << "  H2D: " << h2d << " ms\n";
        std::cout << "  D2H: " << d2h << " ms\n";

        CUDA_CHECK(cudaFree(d_matrix));
        CUDA_CHECK(cudaFree(d_row));
        CUDA_CHECK(cudaFree(d_col));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    return 0;
}