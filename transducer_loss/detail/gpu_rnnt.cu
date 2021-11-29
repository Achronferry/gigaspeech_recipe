#pragma once

#include <tuple>
#include <cmath>
#include <cstring>
#include <limits>
#include <algorithm>
#include <numeric>
#include <chrono>

#if !defined(RNNT_DISABLE_OMP) && !defined(APPLE)
#include <omp.h>
#endif

#include "reduce.h"
#include "gpu_rnnt_kernel.h"

typedef struct CUstream_st* CUstream;

void log_softmax(const float* const acts, float* denom, int alphabet_size, int minibatch, int maxT, int maxU, CUstream stream) {

    // trans_acts + pred_acts -> log_softmax denominator
    reduce_max(acts, denom, alphabet_size, minibatch * maxT * maxU, 0, stream);
    reduce_exp(acts, denom, alphabet_size, minibatch * maxT * maxU, 1, stream);
}

rnntStatus_t compute_cost_and_score(const float* const acts,
                                    float* grads,
                                    float* costs,
                                    const int* const labels,
                                    const int* const label_lengths,
                                    const int* const input_lengths,
                                    int alphabet_size,
                                    int minibatch,
                                    int maxT,
                                    int maxU,
                                    int blank,
                                    float fastemit_lambda,
                                    CUstream stream,
                                    void *gpu_workspace)
{
    
    bool training = (grads != nullptr);
    size_t bytes_used = 0;
    // denom
    float* denom = reinterpret_cast<float*>(static_cast<char*>(gpu_workspace) + bytes_used);
    bytes_used += sizeof(float) * maxT * maxU * minibatch;
    // alphas & betas
    float* alphas = reinterpret_cast<float*>(static_cast<char*>(gpu_workspace) + bytes_used);
    bytes_used += sizeof(float) * maxT * maxU * minibatch;
    float* betas = reinterpret_cast<float*>(static_cast<char*>(gpu_workspace) + bytes_used);
    bytes_used += sizeof(float) * maxT * maxU * minibatch;
    // logllh
    float* llForward = reinterpret_cast<float*>(static_cast<char*>(gpu_workspace) + bytes_used);
    bytes_used += sizeof(float) * minibatch;
    float* llBackward = reinterpret_cast<float*>(static_cast<char*>(gpu_workspace) + bytes_used);
    bytes_used += sizeof(float) * minibatch;

    if (training) {
        // zero grads
        cudaMemsetAsync(grads, 0, sizeof(float) * minibatch * maxT * maxU * alphabet_size, stream);
    }
    // denom
#if defined(DEBUG_TIME)
     auto start = std::chrono::high_resolution_clock::now();
#endif
    log_softmax(acts, denom, alphabet_size, minibatch, maxT, maxU, stream);
#if defined(DEBUG_TIME)
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "DEBUG: log_softmax " << elapsed.count() * 1000 << " ms\n";
    // alphas
    start = std::chrono::high_resolution_clock::now();
#endif
#if defined(USE_NAIVE_KERNEL)
    compute_alphas_kernel_naive<float><<<1, minibatch, 0, stream>>>(acts, denom, alphas, llForward, 
        input_lengths, label_lengths, labels, minibatch, maxT, maxU, alphabet_size, blank);
#else
    compute_alphas_kernel<float><<<minibatch, maxU, 0, stream>>>(acts, denom, alphas, llForward, 
        input_lengths, label_lengths, labels, minibatch, maxT, maxU, alphabet_size, blank);
#endif
#if defined(DEBUG_TIME)
    cudaStreamSynchronize(stream);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "DEBUG: compute_alphas_kernel " << elapsed.count() * 1000 << " ms\n";
#endif
#if defined(DEBUG_KERNEL)
    float* cpu_alphas = new float[minibatch * maxT * maxU];
    int* cpu_xlen = new int[minibatch];
    int* cpu_ylen = new int[minibatch];
    cudaMemcpy(cpu_alphas, alphas, sizeof(float) * minibatch * maxT * maxU, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_xlen, input_lengths, sizeof(int) * minibatch, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_ylen, label_lengths, sizeof(int) * minibatch, cudaMemcpyDeviceToHost);
    printf("gpu alphas\n");
    for (int b = 0; b < minibatch; b++) {
        int T = cpu_xlen[b];
        int U = cpu_ylen[b] + 1;
        printf("B %d, T %d, U %d\n", b, T, U);
        for (int t = 0; t < T; t++) {
            for (int u = 0; u < U; u++) {
                printf("%.2f ", cpu_alphas[(b*maxT+t)*maxU+u]);
            }
            printf("\n");
        }
        printf("\n");
    }
#endif
    if (training) {
        // betas
#if defined(DEBUG_TIME)
        start = std::chrono::high_resolution_clock::now();
#endif
#if defined(USE_NAIVE_KERNEL)
        compute_betas_kernel_naive<float><<<1, minibatch, 0, stream>>>(acts, denom, betas, llBackward,
            input_lengths, label_lengths, labels, minibatch, maxT, maxU, alphabet_size, blank);
#else
        compute_betas_kernel<float><<<minibatch, maxU, 0, stream>>>(acts, denom, betas, llBackward,
            input_lengths, label_lengths, labels, minibatch, maxT, maxU, alphabet_size, blank);
#endif
#if defined(DEBUG_TIME)
        cudaStreamSynchronize(stream);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "DEBUG: compute_betas_kernel " << elapsed.count() * 1000 << " ms\n";
#endif
#if defined(DEBUG_KERNEL)
    float* cpu_betas = new float[minibatch * maxT * maxU];
    cudaMemcpy(cpu_betas, betas, sizeof(float) * minibatch * maxT * maxU, cudaMemcpyDeviceToHost);
    printf("gpu betas\n");
    for (int b = 0; b < minibatch; b++) {
        int T = cpu_xlen[b];
        int U = cpu_ylen[b] + 1;
        printf("B %d, T %d, U %d\n", b, T, U);
        for (int t = 0; t < T; t++) {
            for (int u = 0; u < U; u++) {
                printf("%.2f ", cpu_betas[(b*maxT+t)*maxU+u]);
            }
            printf("\n");
        }
        printf("\n");
    }
#endif

        // gradient
#if defined(DEBUG_TIME)
        start = std::chrono::high_resolution_clock::now();
#endif
        // TODO optimize gradient kernel

        if (fastemit_lambda > 0.0f) {
            compute_fastemit_grad_kernel<128, float><<<minibatch * maxT * maxU, 128, 0, stream>>>(grads, 
                acts, denom, alphas, betas, llForward, input_lengths, label_lengths, labels, 
                minibatch, maxT, maxU, alphabet_size, blank, fastemit_lambda);
        } else {
            compute_grad_kernel<128, float><<<minibatch * maxT * maxU, 128, 0, stream>>>(grads, 
                acts, denom, alphas, betas, llForward, input_lengths, label_lengths, labels, 
                minibatch, maxT, maxU, alphabet_size, blank);
        }
#if defined(DEBUG_TIME)
        cudaStreamSynchronize(stream);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "DEBUG: compute_grad_kernel " << elapsed.count() * 1000 << " ms\n";
#endif
    }
    // cost
    cudaMemcpyAsync(costs, llForward, sizeof(float) * minibatch, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    for (int mb = 0; mb < minibatch; ++mb) {
        costs[mb] = -costs[mb];
    }
    return RNNT_STATUS_SUCCESS;
}

rnntStatus_t cost_and_grad_gpu(const float* const acts,
                       float* grads,
                       float* costs,
                       const int* const pad_labels,
                       const int* const label_lengths,
                       const int* const input_lengths,
                       int alphabet_size,
                       int minibatch,
                       int maxT,
                       int maxU,
                       int blank,
                       float fastemit_lambda,
                       CUstream stream,
                       void *gpu_workspace)

{

    if (acts == nullptr ||
        grads == nullptr || 
        costs == nullptr ||
        pad_labels == nullptr ||
        label_lengths == nullptr ||
        input_lengths == nullptr)
        return RNNT_STATUS_INVALID_VALUE;
    // return RNNT_STATUS_INVALID_VALUE;

    return compute_cost_and_score(acts, grads, costs, pad_labels, 
            label_lengths, input_lengths, alphabet_size, minibatch, 
            maxT, maxU, blank, fastemit_lambda, stream, gpu_workspace);
}

rnntStatus_t score_forward_gpu(const float* const acts,
                       float *costs,
                       const int* const pad_labels,
                       const int* const label_lengths,
                       const int* const input_lengths,
                       int alphabet_size,
                       int minibatch,
                       int maxT,
                       int maxU,
                       int blank,
                       float fastemit_lambda,
                       CUstream stream,
                       void *gpu_workspace)
{
    
    if (acts == nullptr ||
        costs == nullptr ||
        pad_labels == nullptr ||
        label_lengths == nullptr ||
        input_lengths == nullptr)
        return RNNT_STATUS_INVALID_VALUE;
    
    // return RNNT_STATUS_INVALID_VALUE;

    return compute_cost_and_score(acts, nullptr, costs, pad_labels, 
            label_lengths, input_lengths, alphabet_size, minibatch,
            maxT, maxU, blank, fastemit_lambda, stream, gpu_workspace);
}

