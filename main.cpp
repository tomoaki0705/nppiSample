#include <iostream>
#include "npp.h"
#include <omp.h>
#include <string>

#define nppSafeCall(expr, status)  if(status == 0){ auto code = expr ; if(code < 0) { std::cerr << "thread" + std::to_string(omp_get_thread_num()) + " Error code :" << code << " (" << __LINE__ << ")" << std::endl; status = -1; } }
#define cudaSafeCall nppSafeCall

int main()
{
    #pragma omp parallel num_threads(2)
    {
        int status = 0;
        // Source image : 512x512 1channel 8bit, all pixels set to 5
         //cudaMallocPitch(&ptr, (int*)&512, 512, 512)
        unsigned char *ptrSrc;
        size_t pitch = 512;
        size_t width = pitch;
        size_t height = width;
        cudaSafeCall( cudaMallocPitch((void**)&ptrSrc, &pitch, width, height), status );
        //cv::cuda::GpuMat src(512, 512, CV_8UC1);
        // cudaGetDevice
        // cudaGetDeviceCount
        // cudaGetDevice
        // cudaMalloc(&ptr, 52428800)
        // cudaMemset2D(data, step, val, cols * elemSize(), rows)
        // cudaMemset2D(&ptr, 512, 5, 512 * elemSize(), 512)
        unsigned char *ptrPool = NULL;
        cudaSafeCall( cudaMalloc((void**)&ptrPool, width*height * 200), status );
        cudaSafeCall( cudaMemset2D((void**)&ptrSrc, pitch, 5, width, height), status );

        // Destination buffer where mean and stddev value is stored
        // cudaMalloc(&ptr, 8 * 2 * 1)
        // cudaMalloc(&mat->data, elemSize * cols * rows)
        //cv::cuda::GpuMat dst(1, 2, CV_64FC1);
        Npp64f *ptrDst = NULL;
        cudaSafeCall(cudaMalloc((void**)&ptrDst, sizeof(double) * 1 * 2), status);

        // Create scratch buffer
        int bufSize;
        NppiSize sz;
        sz.width = (int)width;
        sz.height = (int)height;
        nppSafeCall( nppiMeanStdDevGetBufferHostSize_8u_C1R(sz, &bufSize), status );
        // cudaMalloc
        //cv::cuda::GpuMat buf(1, bufSize, CV_8UC1);
        unsigned char *ptrBuffer = NULL;
        cudaSafeCall( cudaMalloc((void**)&ptrBuffer, (size_t)(sizeof(char)*bufSize)), status );
        
        // Create stream
        cudaStream_t stream;
        cudaSafeCall( cudaStreamCreate(&stream), status );

        // Set npp to use this stream
        nppSetStream(stream);

        nppSafeCall( nppiMean_StdDev_8u_C1R(ptrSrc, (int)pitch, sz, ptrBuffer, ptrDst, ptrDst + 1), status );

        // Wait until npp call finish
        cudaSafeCall( cudaStreamSynchronize(stream), status );

        // Destroy stream
        cudaSafeCall( cudaStreamDestroy(stream), status );

        std::cout << "thread" << std::to_string(omp_get_thread_num()) << " status : " << status << std::endl;
        //// Print output (expects mean = 5, stddev = 0)
        //cv::Mat h_dst;
        //dst.download(h_dst);
        //std::string out = "thread" + std::to_string(omp_get_thread_num()) + " : (mean)" + std::to_string(h_dst.at<Npp64f>(0, 0)) + " (stddev)" + std::to_string(h_dst.at<Npp64f>(0, 1)) + "\n";
        //std::cout << out;
    }
    return 0;
}
