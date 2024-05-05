#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cblas.h>
#include <cuda_runtime.h>
using namespace std;
#define BLOCK_SIZE 32   //BLOCK_SIZE is the TILE_DIM here

int till_total_testcases = 0; 

template<typename T>
__global__ void matrixmul_kernel(T* A, T* B, T* C, unsigned int size) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    __shared__ T As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T Bs[BLOCK_SIZE][BLOCK_SIZE];

    T sum = 0.0;

    for (int t = 0; t < (size + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        if (t * BLOCK_SIZE + tx < size && row < size) {
            As[ty][tx] = A[row * size + t * BLOCK_SIZE + tx];
        } else {
            As[ty][tx] = 0.0;
        }

        if (t * BLOCK_SIZE + ty < size && col < size) {
            Bs[ty][tx] = B[(t * BLOCK_SIZE + ty) * size + col];
        } else {
            Bs[ty][tx] = 0.0;
        }

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    if (row < size && col < size) {
        C[row * size + col] = sum;
    }
}

template<typename T>
void matrixmul_gpu(T* A, T* B, T* C, unsigned int size, T* times) { 
    cudaEvent_t start, stop; 
    cudaEventCreate(&start); 
    cudaEventCreate(&stop); 

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((size + BLOCK_SIZE - 1) / BLOCK_SIZE, (size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEventRecord(start); 
    matrixmul_kernel<T> <<<numBlocks, threadsPerBlock>>>(A, B, C, size); 
    cudaDeviceSynchronize(); 
    cudaEventRecord(stop); 
    cudaEventSynchronize(stop);

    float milliseconds = 0; 
    cudaEventElapsedTime(&milliseconds, start, stop);
    times[till_total_testcases] = milliseconds;
    till_total_testcases++; 

    cudaEventDestroy(start); 
    cudaEventDestroy(stop); 
 
}

template<typename T>
void random_fill(T* array, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        array[i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
    }
}

template<typename T>
double compute_error(T* C_gpu, T* C_ref, size_t size) {

    double norm_diff = 0.0;
    double norm_ref = 0.0;

    // Compute the squared norm of the difference and the reference 
    for (size_t i = 0; i < size; ++i) {
        T diff = C_gpu[i] - C_ref[i];
        norm_diff += static_cast<double>(diff) * static_cast<double>(diff);
        norm_ref += static_cast<double>(C_ref[i]) * static_cast<double>(C_ref[i]);
    }

    // Compute the relative error
    double relative_error = std::sqrt(norm_diff) / std::sqrt(norm_ref);
    return relative_error;
}

// Flops formula (2 * N * N *  N ) - (N * N)

long long calculate_flops(long long N)
{
    long long temp = N * N;

    return (2 * temp * N) - temp;
}

float calculate_th_put(float flops,float avg_time) // time in param is in milli seconds
{
    float seconds = avg_time / 1000.0;
    return (flops /(seconds * 1e9));
}

int main() {
    const unsigned int num_test = 100;
    srand(time(NULL));

    cudaError_t cudaStatus;
    float* A_float, * B_float, * C_gpu_float, * C_ref_float, * times;

    times = new float[num_test];
    unsigned int sizes[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192};

    float* flops = new float[8];
    float* th_put = new float[8];

    for (unsigned int each_size = 0; each_size < 8; each_size++)
    {
        unsigned int n;
        n = sizes[each_size]; 
        cout << "Calculating for Size: " << n <<endl;
        till_total_testcases = 0;
        for (unsigned int i = 0; i < num_test; i++) { 


            A_float = new float[n * n];
            B_float = new float[n * n];

            C_gpu_float = new float[n * n];
            C_ref_float = new float[n * n];



            random_fill<float>(A_float, n * n);
            random_fill<float>(B_float, n * n);


            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, A_float, n, B_float, n, 0.0, C_ref_float, n);

            float* d_A_float;
            float* d_B_float;
            float* d_C_float;

            cudaStatus = cudaMalloc((void**)&d_A_float, n * n * sizeof(float));
            cudaStatus = cudaMalloc((void**)&d_B_float, n * n * sizeof(float));
            cudaStatus = cudaMalloc((void**)&d_C_float, n * n * sizeof(float));

            cudaStatus = cudaMemcpy(d_A_float, A_float, n * n * sizeof(float), cudaMemcpyHostToDevice);
            cudaStatus = cudaMemcpy(d_B_float, B_float, n * n * sizeof(float), cudaMemcpyHostToDevice);


            matrixmul_gpu<float>(d_A_float, d_B_float, d_C_float, n, times);

            cudaStatus = cudaMemcpy(C_gpu_float, d_C_float, n * n * sizeof(float), cudaMemcpyDeviceToHost);


            double error_float = compute_error<float>(C_gpu_float, C_ref_float, n * n);
            std::cout << "Size: " << n * n << ", Error (float): " << error_float << std::endl;

            delete[] A_float;
            delete[] B_float;
            delete[] C_gpu_float;
            delete[] C_ref_float;

            cudaFree(d_A_float);
            cudaFree(d_B_float);
            cudaFree(d_C_float);
        }
        float avg_time = 0;
        for (int p = 0; p < num_test; p++)
        {
            avg_time = avg_time + times[p];
        }
        avg_time = avg_time / num_test;
        cout << "Average time for Size: " << n << " is " << avg_time << " milli seconds" << endl;
        flops[each_size] = calculate_flops(n); 
        cout << "Flops for Size: " << n << " are " << flops[each_size] << endl; 
        th_put[each_size] = calculate_th_put(flops[each_size], avg_time);
        cout << "ThroughPut for Size: " << n << " is " << th_put[each_size] << endl;
        cout << endl;
    }

   
    return 0;
}
