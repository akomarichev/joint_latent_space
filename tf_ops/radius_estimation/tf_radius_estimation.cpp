#include <cstdio>
#include <ctime>
#include <cstring> // memset
#include <cstdlib> // rand, RAND_MAX
#include <cmath> // sqrtf
#include <vector>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <float.h>

using namespace tensorflow;
using namespace std;

REGISTER_OP("RadiusEstimationOrg")
  .Input("input_xyz: float32")
  .Input("query_xyz: float32")
  .Input("query_normals: float32")
  .Input("idx: int32")
  .Input("pts_cnt: int32")
  .Output("outr: float32")
  .Output("centroids: float32")
  .Output("dist: float32")
  .Output("nu: float32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
    c->set_output(0, c->input(4));
    c->set_output(1, c->input(1));
    c->set_output(2, c->input(4));
    c->set_output(3, c->input(4));
    return Status::OK();
  });

REGISTER_OP("ClipRadiusOrg")
  .Input("radiuses: float32")
  .Input("clip_values: float32")
  .Output("clipped_radiuses: float32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
    c->set_output(0, c->input(0));
    return Status::OK();
  });

void radiusEstimationLauncher(int b, int m, int n, int m_q, int k, const float *input, const float *queries, const float *queries_norm, const int *idx, const int *pts_cnt, float *outr, float *centroids, float *dist, float *nu_arr);
class RadiusEstimationOrgGpuOp : public OpKernel {
      public:
          explicit RadiusEstimationOrgGpuOp(OpKernelConstruction * context):OpKernel(context){}
          
          void Compute(OpKernelContext* context) override {
            const Tensor& input_xyz_tensor = context->input(0);
            OP_REQUIRES(context, input_xyz_tensor.dims() == 3, errors::InvalidArgument("RadiusEstimation expects (b,m,n) input_xyz shape"));

            int b = input_xyz_tensor.shape().dim_size(0);
            int m = input_xyz_tensor.shape().dim_size(1);
            int n = input_xyz_tensor.shape().dim_size(2);

            const Tensor& query_xyz_tensor = context->input(1);
            OP_REQUIRES(context, query_xyz_tensor.dims() == 3, errors::InvalidArgument("RadiusEstimation expects (b,m_q,n) query_xyz shape"));

            int m_q = query_xyz_tensor.shape().dim_size(1);

            const Tensor& query_normals_tensor = context->input(2);
            OP_REQUIRES(context, query_normals_tensor.dims() == 3, errors::InvalidArgument("RadiusEstimation expects (b,m_q,n) query_normals shape"));

            const Tensor& idx_tensor = context->input(3);
            OP_REQUIRES(context, idx_tensor.dims() == 3, errors::InvalidArgument("RadiusEstimation expects (b,m_q,k) idx shape"));

            int k = idx_tensor.shape().dim_size(2);

            const Tensor& pts_cnt_tensor = context->input(4);
            OP_REQUIRES(context, pts_cnt_tensor.dims() == 2, errors::InvalidArgument("RadiusEstimation expects (b,m_q) pts_cnt shape"));

            Tensor *outr_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,m_q}, &outr_tensor));

            Tensor *centroids_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{b,m_q,n}, &centroids_tensor));

            Tensor *dist_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape{b,m_q}, &dist_tensor));

            Tensor *nu_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(3, TensorShape{b,m_q}, &nu_tensor));

            auto input_flat = input_xyz_tensor.flat<float>();
            const float *input = &(input_flat(0));
            auto queries_flat = query_xyz_tensor.flat<float>();
            const float *queries = &(queries_flat(0));
            auto queries_norm_flat = query_normals_tensor.flat<float>();
            const float *queries_norm = &(queries_norm_flat(0));
            auto idx_flat = idx_tensor.flat<int>();
            const int *idx = &(idx_flat(0));
            auto pts_cnt_flat = pts_cnt_tensor.flat<int>();
            const int *pts_cnt = &(pts_cnt_flat(0));
            auto outr_flat = outr_tensor->flat<float>();
            float *outr = &(outr_flat(0));
            auto centroids_flat = centroids_tensor->flat<float>();
            float *centroids = &(centroids_flat(0));
            auto dist_flat = dist_tensor->flat<float>();
            float *dist = &(dist_flat(0));
            auto nu_flat = nu_tensor->flat<float>();
            float *nu = &(nu_flat(0));
            radiusEstimationLauncher(b, m, n, m_q, k, input, queries, queries_norm, idx, pts_cnt, outr, centroids, dist, nu);
          }
};
REGISTER_KERNEL_BUILDER(Name("RadiusEstimationOrg").Device(DEVICE_GPU), RadiusEstimationOrgGpuOp);


void clipRadiusesLauncher(int b, int m, const float *radiuses, const float *clip_values, float *clip_radiuses);
class ClipRadiusOrgGpuOp : public OpKernel {
      public:
          explicit ClipRadiusOrgGpuOp(OpKernelConstruction * context):OpKernel(context){}
          
          void Compute(OpKernelContext* context) override {
            const Tensor& radiuses_tensor = context->input(0);
            OP_REQUIRES(context, radiuses_tensor.dims() == 2, errors::InvalidArgument("ClipRadiuses expects (b,m) input_xyz shape"));

            int b = radiuses_tensor.shape().dim_size(0);
            int m = radiuses_tensor.shape().dim_size(1);

            const Tensor& clip_values_tensor = context->input(1);
            OP_REQUIRES(context, clip_values_tensor.dims() == 2, errors::InvalidArgument("ClipRadiuses expects (b,m) query_xyz shape"));

            Tensor *clip_radiuses_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,m}, &clip_radiuses_tensor));

            auto radiuses_flat = radiuses_tensor.flat<float>();
            const float *radiuses = &(radiuses_flat(0));
            auto clip_values_flat = clip_values_tensor.flat<float>();
            const float *clip_values = &(clip_values_flat(0));
            
            auto clip_radiuses_flat = clip_radiuses_tensor->flat<float>();
            float *clip_radiuses = &(clip_radiuses_flat(0));
            cudaMemset(clip_radiuses, 0.0, sizeof(float)*b*m);
            clipRadiusesLauncher(b, m, radiuses, clip_values, clip_radiuses);
          }
};
REGISTER_KERNEL_BUILDER(Name("ClipRadiusOrg").Device(DEVICE_GPU), ClipRadiusOrgGpuOp);