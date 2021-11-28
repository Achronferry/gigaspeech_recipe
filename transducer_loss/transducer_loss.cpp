#include <iostream>
#include <numeric>

#include <torch/extension.h>
// #include "rnnt.h"
#include "detail/cpu_rnnt.h"
// #include "detail/gpu_rnnt.h"
typedef struct CUstream_st* CUstream;


#include "THC.h"
extern THCState* state;

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
                       void *gpu_workspace);

rnntStatus_t score_forward_gpu(const float* const acts,
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
                       void *gpu_workspace);


/*
typedef enum {
    RNNT_STATUS_SUCCESS = 0,
    RNNT_STATUS_MEMOPS_FAILED = 1,
    RNNT_STATUS_INVALID_VALUE = 2,
    RNNT_STATUS_EXECUTION_FAILED = 3,
    RNNT_STATUS_UNKNOWN_ERROR = 4
} rnntStatus_t;
*/

/*
typedef enum {
    RNNT_CPU = 0,
    RNNT_GPU = 1
} rnntComputeLocation;
*/

/*
int get_warprnnt_version() {
    return 1;
}
*/

/*
struct rnntOptions {
    /// indicates where the rnnt calculation should take place {RNNT_CPU | RNNT_GPU}
    rnntComputeLocation loc;

    /// The maximum number of threads that can be used
    unsigned int num_threads;

    /// used when loc == RNNT_GPU, which stream the kernels should be launched in
    CUstream stream;

    /// the label value/index that the RNNT calculation should use as the blank label
    int blank_label;

    /// the maximum length of time steps
    int maxT;

    /// the maximum length of label sequence
    int maxU;

    /// memory structure
    bool batch_first;

    float fastemit_lambda;
};
*/



rnntStatus_t compute_rnnt_loss(const float* const activations, //BTUV
                             float* gradients,
                             const int* const flat_labels,
                             const int* const label_lengths,
                             const int* const input_lengths,
                             int alphabet_size,
                             int minibatch,
                             float *costs,
                             void *workspace,
                             rnntOptions options)
{

    if (activations == nullptr ||
        flat_labels == nullptr ||
        label_lengths == nullptr ||
        input_lengths == nullptr ||
        costs == nullptr ||
        workspace == nullptr ||
        alphabet_size <= 0 ||
        minibatch <= 0 ||
        options.maxT <= 0 ||
        options.maxU <= 0 ||
        options.fastemit_lambda < 0)
        return RNNT_STATUS_INVALID_VALUE;

    if (options.loc == RNNT_CPU)
    {
        CpuRNNT<float> rnnt(minibatch, options.maxT, options.maxU, alphabet_size, workspace, 
                            options.blank_label, options.fastemit_lambda, options.num_threads, options.batch_first);

        if (gradients != NULL)
            return rnnt.cost_and_grad(activations, gradients,
                                        costs,
                                        flat_labels, label_lengths,
                                        input_lengths);
        else
            return rnnt.score_forward(activations, costs, flat_labels,
                                        label_lengths, input_lengths);
    }
    else if (options.loc == RNNT_GPU)
    {
        /*
        GpuRNNT<float> rnnt(minibatch, options.maxT, options.maxU, alphabet_size, workspace,
                            options.blank_label, options.fastemit_lambda, options.num_threads, options.stream);

        if (gradients != NULL)
            return rnnt.cost_and_grad(activations, gradients,
                                        costs,
                                        flat_labels, label_lengths,
                                        input_lengths);
        else
            return rnnt.score_forward(activations, costs, flat_labels,
                                        label_lengths, input_lengths);
        */


        if (gradients != NULL)
            return cost_and_grad_gpu(activations, gradients,
                                        costs,
                                        flat_labels, label_lengths,
                                        input_lengths, alphabet_size,
                                        minibatch, options.maxT,
                                        options.maxU, options.blank_label,
                                        options.fastemit_lambda,
                                        options.stream, workspace);
        else
            return score_forward_gpu(activations, costs, flat_labels,
                                        label_lengths, input_lengths,
                                        alphabet_size, minibatch,
                                        options.maxT, options.maxU,
                                        options.blank_label,
                                        options.fastemit_lambda,
                                        options.stream, workspace);

    }
    else
    {
        return RNNT_STATUS_INVALID_VALUE;
    }
}

rnntStatus_t get_workspace_size(int maxT, int maxU,
                               int minibatch,
                               bool gpu,
                               size_t* size_bytes,
                               size_t dtype_size)
{
    if (minibatch <= 0 ||
        maxT <= 0 ||
        maxU <= 0)
        return RNNT_STATUS_INVALID_VALUE;

    *size_bytes = 0;

    // per minibatch memory
    size_t per_minibatch_bytes = 0;

    // alphas & betas
    per_minibatch_bytes += dtype_size * maxT * maxU * 2;

    if (!gpu) {
        // blank & label log probability cache
        per_minibatch_bytes += dtype_size * maxT * maxU * 2;
    } else {
        // softmax denominator
        per_minibatch_bytes += dtype_size * maxT * maxU;
        // forward-backward loglikelihood
        per_minibatch_bytes += dtype_size * 2;
    }

    *size_bytes = per_minibatch_bytes * minibatch;

    return RNNT_STATUS_SUCCESS;
}



int cpu_rnnt(torch::Tensor acts,
            torch::Tensor labels,
            torch::Tensor input_lengths,
            torch::Tensor label_lengths,
            torch::Tensor costs,
            torch::Tensor grads,
            int blank_label,
            float fastemit_lambda,
            int num_threads)
{

    int maxT = acts.size(0);
    int maxU = acts.size(1);
    int minibatch_size = acts.size(2);
    int alphabet_size = acts.size(3);

	if (true)
    {
		minibatch_size = acts.size(0);
		maxT = acts.size(1);
		maxU = acts.size(2);
	}

    rnntOptions options;
    memset(&options, 0, sizeof(options));
    options.maxT = maxT;
    options.maxU = maxU;
    options.fastemit_lambda = fastemit_lambda;
    options.blank_label = blank_label;
    options.batch_first = true;
    options.loc = RNNT_CPU;
    options.num_threads = num_threads;
// #if defined(RNNT_DISABLE_OMP) || defined(APPLE)
    // have to use at least one
    options.num_threads = std::max(options.num_threads, (unsigned int) 1);
// #endif

    size_t cpu_size_bytes = 0;
    get_workspace_size(maxT, maxU, minibatch_size, false, &cpu_size_bytes, sizeof(float));

    float* cpu_workspace = (float*) new unsigned char[cpu_size_bytes];
    compute_rnnt_loss(acts.data<float>(), grads.data<float>(),
            labels.data<int>(), label_lengths.data<int>(),
            input_lengths.data<int>(), alphabet_size,
            minibatch_size, costs.data<float>(),
            cpu_workspace, options);

    delete cpu_workspace;
    return 0;
}



int gpu_rnnt(torch::Tensor acts,
            torch::Tensor labels,
            torch::Tensor input_lengths,
            torch::Tensor label_lengths,
            torch::Tensor costs,
            torch::Tensor grads,
            int blank_label,
            float fastemit_lambda,
            int num_threads)
{

    int minibatch_size = acts.size(0);
    int maxT = acts.size(1);
    int maxU = acts.size(2);
    int alphabet_size = acts.size(3);

    rnntOptions options;
    memset(&options, 0, sizeof(options));
    options.maxT = maxT;
    options.maxU = maxU;
    options.blank_label = blank_label;
    options.loc = RNNT_GPU;
    options.stream = at::cuda::getCurrentCUDAStream();
    options.fastemit_lambda = fastemit_lambda;
    options.num_threads = num_threads;
// #if defined(RNNT_DISABLE_OMP) || defined(APPLE)
    // have to use at least one
    options.num_threads = std::max(options.num_threads, (unsigned int) 1);
// #endif

    size_t gpu_size_bytes;
    get_workspace_size(maxT, maxU, minibatch_size,  true, &gpu_size_bytes, sizeof(float));

    cudaSetDevice(acts.get_device());

    void* gpu_workspace = THCudaMalloc(state, gpu_size_bytes);

    compute_rnnt_loss(acts.data<float>(), grads.data<float>(),
            labels.data<int>(), label_lengths.data<int>(),
            input_lengths.data<int>(), alphabet_size,
            minibatch_size, costs.data<float>(),
            gpu_workspace, options);

    THCudaFree(state, gpu_workspace);
    return 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cpu_rnnt", &cpu_rnnt, "RNNT CPU version");
    m.def("gpu_rnnt", &gpu_rnnt, "RNNT GPU version");
}
