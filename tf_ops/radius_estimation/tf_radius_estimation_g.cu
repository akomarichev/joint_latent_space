#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <cstdio>

#define M_PI 3.14159265358979323846

__global__ void estimate_radiuses_gpu(int b, int m, int n, int m_q, int k, const float *input, const float *queries, const float *queries_norm, const int *idx, const int *pts_cnt, float *outr, float *centroids, float *dist, float *nu_arr){
  int batch_index = blockIdx.x;
  input+=m*n*batch_index;
  queries+=m_q*n*batch_index;
  queries_norm+=m_q*n*batch_index;
  idx+=m_q*k*batch_index;
  pts_cnt+=m_q*batch_index;
  outr+=m_q*batch_index;
  centroids+=m_q*n*batch_index;
  dist+=m_q*batch_index;

  int index = threadIdx.x;
  int stride = blockDim.x;

  for(int i=index; i<m_q; i+=stride){
      // 1. Calculate neighbor average c (c_x, c_y, c_z);
      float c_x = 0.0, c_y = 0.0, c_z = 0.0;
      for(int j=0; j<pts_cnt[i]; ++j){
        int pnt_idx = idx[i*k+j];
        c_x += input[pnt_idx*n + 0];
        c_y += input[pnt_idx*n + 1];
        c_z += input[pnt_idx*n + 2];
      }
      c_x /= pts_cnt[i];
      c_y /= pts_cnt[i];
      c_z /= pts_cnt[i];

      centroids[i*n + 0] = c_x;
      centroids[i*n + 1] = c_y;
      centroids[i*n + 2] = c_z;

      // 2. Project calculated 'c' on unit normal vector by using dot product
      float n_x = queries_norm[i*n + 0];
      float n_y = queries_norm[i*n + 1];
      float n_z = queries_norm[i*n + 2];
      float x_q = queries[i*n + 0];
      float y_q = queries[i*n + 1];
      float z_q = queries[i*n + 2];
      float d_x = c_x - x_q;
      float d_y = c_y - y_q;
      float d_z = c_z - z_q;
      float d = fabs(d_x*n_x + d_y*n_y + d_z*n_z);

      dist[i] = d;

      // 3. Calculate the average distance to neighbors
      float nu = 0.0;
      float nu_max = 0.0;
      for(int j=0; j<pts_cnt[i]; ++j){
        int pnt_idx = idx[i*k+j];
        float x_p = input[pnt_idx*n + 0];
        float y_p = input[pnt_idx*n + 1];
        float z_p = input[pnt_idx*n + 2];
        nu += max(sqrtf((x_p-x_q)*(x_p-x_q)+(y_p-y_q)*(y_p-y_q)+(z_p-z_q)*(z_p-z_q)),1e-20f);
        nu_max = max(nu_max, nu);
      }
      nu /= (pts_cnt[i] - 1);

      nu_arr[i] = nu;

      // 4. Calculate radius: r = nu / 2*d
      float epsilon = 0.1;
      float c1 = 0.5;
      float c2 = 0.5;
      float rho = (pts_cnt[i] - 1) / (M_PI * nu_max * nu_max);
      float sigma = 0.01;
      float curvature = 2 * d / (nu * nu);
      float r = powf( ( c1 * sigma / sqrtf(rho * epsilon) + c2 * (sigma * sigma) ) / curvature, 1.0/3.0);
      outr[i] = r;
  }
}

__global__ void clip_radiuses_gpu(int b, int m, const float *radiuses, const float *clip_values, float *clip_radiuses){
  int batch_index = blockIdx.x;
  radiuses+=m*batch_index;
  clip_values+=m*batch_index;
  clip_radiuses+=m*batch_index;

  int index = threadIdx.x;
  int stride = blockDim.x;

  for(int i=index; i<m; i+=stride){
      // Clip radiuses;
      if(radiuses[i] > clip_values[i]) clip_radiuses[i] = clip_values[i];
      else clip_radiuses[i] = radiuses[i];
  }
}

void radiusEstimationLauncher(int b, int m, int n, int m_q, int k, const float *input, const float *queries, const float *queries_norm, const int *idx, const int *pts_cnt, float *outr, float *centroids, float *dist, float *nu_arr){
  estimate_radiuses_gpu<<<b,256>>>(b,m,n,m_q,k,input,queries,queries_norm,idx,pts_cnt,outr,centroids, dist, nu_arr);
  cudaDeviceSynchronize();
}

void clipRadiusesLauncher(int b, int m, const float *radiuses, const float *clip_values, float *clip_radiuses){
  clip_radiuses_gpu<<<b,256>>>(b,m,radiuses,clip_values,clip_radiuses);
  cudaDeviceSynchronize();
}
