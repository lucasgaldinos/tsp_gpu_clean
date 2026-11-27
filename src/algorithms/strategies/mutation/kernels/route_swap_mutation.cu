/**
 * GPU 0-Delimited VRP Route Swap Mutation Kernel (`DP17-7`, `BUG-114`)
 *
 * Unbiased intra/inter-route customer swap on `(batch_size, width)` `int32` 0-delimited
 * routes using distinct customer ordinals. Preserves exact customer bijection
 * `{1..dimension-1}` and depot `0` count with zero D2H.
 */

extern "C" __global__ void route_swap_mutation_kernel(
    const int *__restrict__ routes,           // [batch_size, width]
    int *__restrict__ out_routes,             // [batch_size, width]
    const unsigned char *__restrict__ active, // [batch_size]
    const int *__restrict__ k1_arr,           // [batch_size] or nullptr
    const int *__restrict__ k2_arr,           // [batch_size] or nullptr
    const double *__restrict__ u1_arr,        // [batch_size] fallback when k1_arr == nullptr
    const double *__restrict__ u2_arr,        // [batch_size] fallback when k2_arr == nullptr
    const int batch_size,
    const int width)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size)
    {
        return;
    }

    const int *src = routes + (size_t)idx * (size_t)width;
    int *dst = out_routes + (size_t)idx * (size_t)width;

    int n_cust = 0;
    for (int col = 0; col < width; ++col)
    {
        const int val = src[col];
        dst[col] = val;
        if (val > 0)
        {
            n_cust += 1;
        }
    }

    if (active[idx] == 0 || n_cust < 2)
    {
        return;
    }

    int k1 = 0;
    int k2 = 0;
    if (k1_arr != nullptr && k2_arr != nullptr)
    {
        k1 = k1_arr[idx];
        k2 = k2_arr[idx];
    }
    else
    {
        k1 = (int)(u1_arr[idx] * (double)n_cust);
        if (k1 >= n_cust)
        {
            k1 = n_cust - 1;
        }
        int s = (int)(u2_arr[idx] * (double)(n_cust - 1));
        if (s >= n_cust - 1)
        {
            s = n_cust - 2;
        }
        k2 = s + ((s >= k1) ? 1 : 0);
    }

    if (k1 < 0 || k1 >= n_cust || k2 < 0 || k2 >= n_cust || k1 == k2)
    {
        return;
    }

    int pos1 = -1;
    int pos2 = -1;
    int seen = 0;
    for (int col = 0; col < width; ++col)
    {
        if (dst[col] > 0)
        {
            if (seen == k1)
            {
                pos1 = col;
            }
            if (seen == k2)
            {
                pos2 = col;
            }
            seen += 1;
        }
    }

    if (pos1 >= 0 && pos2 >= 0 && pos1 != pos2)
    {
        const int tmp = dst[pos1];
        dst[pos1] = dst[pos2];
        dst[pos2] = tmp;
    }
}
