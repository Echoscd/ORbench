// gpu_baseline.cu — Dynamic Time Warping (GPU baseline)
//
// Faithfully ported from cuDTW++ (asbschmidt/cuDTW), Schmidt & Hundt
// "cuDTW++: Ultra-Fast Dynamic Time Warping on CUDA-enabled GPUs"
// (Euro-Par 2020). The kernel below is a verbatim copy of
//   src/include/kernels/SHFL_FULLDTW_1023.cuh::shfl_FullDTW_1023
// adapted only by:
//   - replacing the cuDTW template (value_t, index_t) with (float, int)
//   - removing dead commented-out code lines for readability
// All arithmetic, register layout, warp-shuffle propagation, constant-
// memory query access, and output-write conventions are preserved bit-for-
// bit. As in the original kernel, this version is length-specialized to
// num_features = 1023 (lane = 1024 = 32 lanes x 32 cells).
//
// Layout conventions (matching cuDTW main.cu):
//   * Subjects: float[num_entries * num_features], row-major (one row per series)
//   * Query:    placed in __constant__ memory `cQuery` of size up to 16384
//   * Output:   one DTW distance per subject

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define WARP_SIZE 32

// max features fitting into constant memory (matches cuDTW main.cu line 38)
constexpr int max_features = (1 << 16) / sizeof(float);  // 16384
__constant__ float cQuery[max_features];

// ===== verbatim port of cuDTW shfl_FullDTW_1023 =====
__global__
void shfl_FullDTW_1023(const float* Subject,
                       float*       Dist,
                       int          num_entries,
                       int          num_features)
{
    const int blid = blockIdx.x;
    const int thid = threadIdx.x;
    const int lane = num_features + 1;
    const int base = blid * num_features;
    const int l    = thid;

    float penalty_left = INFINITY;
    float penalty_diag = 0;
    float penalty_here0  = INFINITY;
    float penalty_here1  = INFINITY;
    float penalty_here2  = INFINITY;
    float penalty_here3  = INFINITY;
    float penalty_here4  = INFINITY;
    float penalty_here5  = INFINITY;
    float penalty_here6  = INFINITY;
    float penalty_here7  = INFINITY;
    float penalty_here8  = INFINITY;
    float penalty_here9  = INFINITY;
    float penalty_here10 = INFINITY;
    float penalty_here11 = INFINITY;
    float penalty_here12 = INFINITY;
    float penalty_here13 = INFINITY;
    float penalty_here14 = INFINITY;
    float penalty_here15 = INFINITY;
    float penalty_here16 = INFINITY;
    float penalty_here17 = INFINITY;
    float penalty_here18 = INFINITY;
    float penalty_here19 = INFINITY;
    float penalty_here20 = INFINITY;
    float penalty_here21 = INFINITY;
    float penalty_here22 = INFINITY;
    float penalty_here23 = INFINITY;
    float penalty_here24 = INFINITY;
    float penalty_here25 = INFINITY;
    float penalty_here26 = INFINITY;
    float penalty_here27 = INFINITY;
    float penalty_here28 = INFINITY;
    float penalty_here29 = INFINITY;
    float penalty_here30 = INFINITY;
    float penalty_here31 = INFINITY;
    float penalty_temp0;
    float penalty_temp1;

    if (thid == 0) {
        penalty_left = INFINITY;
        penalty_diag = INFINITY;
        penalty_here0  = INFINITY;
        penalty_here1  = INFINITY;
        penalty_here2  = INFINITY;
        penalty_here3  = INFINITY;
        penalty_here4  = INFINITY;
        penalty_here5  = INFINITY;
        penalty_here6  = INFINITY;
        penalty_here7  = INFINITY;
        penalty_here8  = INFINITY;
        penalty_here9  = INFINITY;
        penalty_here10 = INFINITY;
        penalty_here11 = INFINITY;
        penalty_here12 = INFINITY;
        penalty_here13 = INFINITY;
        penalty_here14 = INFINITY;
        penalty_here15 = INFINITY;
        penalty_here16 = INFINITY;
        penalty_here17 = INFINITY;
        penalty_here18 = INFINITY;
        penalty_here19 = INFINITY;
        penalty_here20 = INFINITY;
        penalty_here21 = INFINITY;
        penalty_here22 = INFINITY;
        penalty_here23 = INFINITY;
        penalty_here24 = INFINITY;
        penalty_here25 = INFINITY;
        penalty_here26 = INFINITY;
        penalty_here27 = INFINITY;
        penalty_here28 = INFINITY;
        penalty_here29 = INFINITY;
        penalty_here30 = INFINITY;
        penalty_here31 = INFINITY;
    }

    const float subject_value0  = l == 0 ? 0 : Subject[base + 32*l - 1];
    const float subject_value1  = Subject[base + 32*l +  0];
    const float subject_value2  = Subject[base + 32*l +  1];
    const float subject_value3  = Subject[base + 32*l +  2];
    const float subject_value4  = Subject[base + 32*l +  3];
    const float subject_value5  = Subject[base + 32*l +  4];
    const float subject_value6  = Subject[base + 32*l +  5];
    const float subject_value7  = Subject[base + 32*l +  6];
    const float subject_value8  = Subject[base + 32*l +  7];
    const float subject_value9  = Subject[base + 32*l +  8];
    const float subject_value10 = Subject[base + 32*l +  9];
    const float subject_value11 = Subject[base + 32*l + 10];
    const float subject_value12 = Subject[base + 32*l + 11];
    const float subject_value13 = Subject[base + 32*l + 12];
    const float subject_value14 = Subject[base + 32*l + 13];
    const float subject_value15 = Subject[base + 32*l + 14];
    const float subject_value16 = Subject[base + 32*l + 15];
    const float subject_value17 = Subject[base + 32*l + 16];
    const float subject_value18 = Subject[base + 32*l + 17];
    const float subject_value19 = Subject[base + 32*l + 18];
    const float subject_value20 = Subject[base + 32*l + 19];
    const float subject_value21 = Subject[base + 32*l + 20];
    const float subject_value22 = Subject[base + 32*l + 21];
    const float subject_value23 = Subject[base + 32*l + 22];
    const float subject_value24 = Subject[base + 32*l + 23];
    const float subject_value25 = Subject[base + 32*l + 24];
    const float subject_value26 = Subject[base + 32*l + 25];
    const float subject_value27 = Subject[base + 32*l + 26];
    const float subject_value28 = Subject[base + 32*l + 27];
    const float subject_value29 = Subject[base + 32*l + 28];
    const float subject_value30 = Subject[base + 32*l + 29];
    const float subject_value31 = Subject[base + 32*l + 30];

    int   counter = 1;
    float query_value     = INFINITY;
    float new_query_value = cQuery[thid];
    if (thid == 0) query_value = new_query_value;
    if (thid == 0) penalty_here1 = 0;
    new_query_value = __shfl_down_sync(0xFFFFFFFF, new_query_value, 1, 32);

    penalty_temp0 = penalty_here0;
    penalty_here0 = (query_value-subject_value0) * (query_value-subject_value0) + min(penalty_left, min(penalty_here0, penalty_diag));
    penalty_temp1 = INFINITY;
    penalty_here1 = (query_value-subject_value1) * (query_value-subject_value1) + min(penalty_here0, min(penalty_here1, penalty_temp0));
    penalty_temp0 = penalty_here2;
    penalty_here2 = (query_value-subject_value2) * (query_value-subject_value2) + min(penalty_here1, min(penalty_here2, penalty_temp1));
    penalty_temp1 = penalty_here3;
    penalty_here3 = (query_value-subject_value3) * (query_value-subject_value3) + min(penalty_here2, min(penalty_here3, penalty_temp0));
    penalty_temp0 = penalty_here4;
    penalty_here4 = (query_value-subject_value4) * (query_value-subject_value4) + min(penalty_here3, min(penalty_here4, penalty_temp1));
    penalty_temp1 = penalty_here5;
    penalty_here5 = (query_value-subject_value5) * (query_value-subject_value5) + min(penalty_here4, min(penalty_here5, penalty_temp0));
    penalty_temp0 = penalty_here6;
    penalty_here6 = (query_value-subject_value6) * (query_value-subject_value6) + min(penalty_here5, min(penalty_here6, penalty_temp1));
    penalty_temp1 = penalty_here7;
    penalty_here7 = (query_value-subject_value7) * (query_value-subject_value7) + min(penalty_here6, min(penalty_here7, penalty_temp0));
    penalty_temp0 = penalty_here8;
    penalty_here8 = (query_value-subject_value8) * (query_value-subject_value8) + min(penalty_here7, min(penalty_here8, penalty_temp1));
    penalty_temp1 = penalty_here9;
    penalty_here9 = (query_value-subject_value9) * (query_value-subject_value9) + min(penalty_here8, min(penalty_here9, penalty_temp0));
    penalty_temp0 = penalty_here10;
    penalty_here10 = (query_value-subject_value10) * (query_value-subject_value10) + min(penalty_here9, min(penalty_here10, penalty_temp1));
    penalty_temp1 = penalty_here11;
    penalty_here11 = (query_value-subject_value11) * (query_value-subject_value11) + min(penalty_here10, min(penalty_here11, penalty_temp0));
    penalty_temp0 = penalty_here12;
    penalty_here12 = (query_value-subject_value12) * (query_value-subject_value12) + min(penalty_here11, min(penalty_here12, penalty_temp1));
    penalty_temp1 = penalty_here13;
    penalty_here13 = (query_value-subject_value13) * (query_value-subject_value13) + min(penalty_here12, min(penalty_here13, penalty_temp0));
    penalty_temp0 = penalty_here14;
    penalty_here14 = (query_value-subject_value14) * (query_value-subject_value14) + min(penalty_here13, min(penalty_here14, penalty_temp1));
    penalty_temp1 = penalty_here15;
    penalty_here15 = (query_value-subject_value15) * (query_value-subject_value15) + min(penalty_here14, min(penalty_here15, penalty_temp0));
    penalty_temp0 = penalty_here16;
    penalty_here16 = (query_value-subject_value16) * (query_value-subject_value16) + min(penalty_here15, min(penalty_here16, penalty_temp1));
    penalty_temp1 = penalty_here17;
    penalty_here17 = (query_value-subject_value17) * (query_value-subject_value17) + min(penalty_here16, min(penalty_here17, penalty_temp0));
    penalty_temp0 = penalty_here18;
    penalty_here18 = (query_value-subject_value18) * (query_value-subject_value18) + min(penalty_here17, min(penalty_here18, penalty_temp1));
    penalty_temp1 = penalty_here19;
    penalty_here19 = (query_value-subject_value19) * (query_value-subject_value19) + min(penalty_here18, min(penalty_here19, penalty_temp0));
    penalty_temp0 = penalty_here20;
    penalty_here20 = (query_value-subject_value20) * (query_value-subject_value20) + min(penalty_here19, min(penalty_here20, penalty_temp1));
    penalty_temp1 = penalty_here21;
    penalty_here21 = (query_value-subject_value21) * (query_value-subject_value21) + min(penalty_here20, min(penalty_here21, penalty_temp0));
    penalty_temp0 = penalty_here22;
    penalty_here22 = (query_value-subject_value22) * (query_value-subject_value22) + min(penalty_here21, min(penalty_here22, penalty_temp1));
    penalty_temp1 = penalty_here23;
    penalty_here23 = (query_value-subject_value23) * (query_value-subject_value23) + min(penalty_here22, min(penalty_here23, penalty_temp0));
    penalty_temp0 = penalty_here24;
    penalty_here24 = (query_value-subject_value24) * (query_value-subject_value24) + min(penalty_here23, min(penalty_here24, penalty_temp1));
    penalty_temp1 = penalty_here25;
    penalty_here25 = (query_value-subject_value25) * (query_value-subject_value25) + min(penalty_here24, min(penalty_here25, penalty_temp0));
    penalty_temp0 = penalty_here26;
    penalty_here26 = (query_value-subject_value26) * (query_value-subject_value26) + min(penalty_here25, min(penalty_here26, penalty_temp1));
    penalty_temp1 = penalty_here27;
    penalty_here27 = (query_value-subject_value27) * (query_value-subject_value27) + min(penalty_here26, min(penalty_here27, penalty_temp0));
    penalty_temp0 = penalty_here28;
    penalty_here28 = (query_value-subject_value28) * (query_value-subject_value28) + min(penalty_here27, min(penalty_here28, penalty_temp1));
    penalty_temp1 = penalty_here29;
    penalty_here29 = (query_value-subject_value29) * (query_value-subject_value29) + min(penalty_here28, min(penalty_here29, penalty_temp0));
    penalty_temp0 = penalty_here30;
    penalty_here30 = (query_value-subject_value30) * (query_value-subject_value30) + min(penalty_here29, min(penalty_here30, penalty_temp1));
    penalty_here31 = (query_value-subject_value31) * (query_value-subject_value31) + min(penalty_here30, min(penalty_here31, penalty_temp0));

    query_value = __shfl_up_sync(0xFFFFFFFF, query_value, 1, 32);
    if (thid == 0) query_value = new_query_value;
    new_query_value = __shfl_down_sync(0xFFFFFFFF, new_query_value, 1, 32);
    counter++;

    penalty_diag = penalty_left;
    penalty_left = __shfl_up_sync(0xFFFFFFFF, penalty_here31, 1, 32);
    if (thid == 0) penalty_left = INFINITY;

    for (int k = 3; k < lane + WARP_SIZE - 1; k++) {
        const int i = k - l;

        penalty_temp0 = penalty_here0;
        penalty_here0 = (query_value-subject_value0) * (query_value-subject_value0) + min(penalty_left, min(penalty_here0, penalty_diag));
        penalty_temp1 = penalty_here1;
        penalty_here1 = (query_value-subject_value1) * (query_value-subject_value1) + min(penalty_here0, min(penalty_here1, penalty_temp0));
        penalty_temp0 = penalty_here2;
        penalty_here2 = (query_value-subject_value2) * (query_value-subject_value2) + min(penalty_here1, min(penalty_here2, penalty_temp1));
        penalty_temp1 = penalty_here3;
        penalty_here3 = (query_value-subject_value3) * (query_value-subject_value3) + min(penalty_here2, min(penalty_here3, penalty_temp0));
        penalty_temp0 = penalty_here4;
        penalty_here4 = (query_value-subject_value4) * (query_value-subject_value4) + min(penalty_here3, min(penalty_here4, penalty_temp1));
        penalty_temp1 = penalty_here5;
        penalty_here5 = (query_value-subject_value5) * (query_value-subject_value5) + min(penalty_here4, min(penalty_here5, penalty_temp0));
        penalty_temp0 = penalty_here6;
        penalty_here6 = (query_value-subject_value6) * (query_value-subject_value6) + min(penalty_here5, min(penalty_here6, penalty_temp1));
        penalty_temp1 = penalty_here7;
        penalty_here7 = (query_value-subject_value7) * (query_value-subject_value7) + min(penalty_here6, min(penalty_here7, penalty_temp0));
        penalty_temp0 = penalty_here8;
        penalty_here8 = (query_value-subject_value8) * (query_value-subject_value8) + min(penalty_here7, min(penalty_here8, penalty_temp1));
        penalty_temp1 = penalty_here9;
        penalty_here9 = (query_value-subject_value9) * (query_value-subject_value9) + min(penalty_here8, min(penalty_here9, penalty_temp0));
        penalty_temp0 = penalty_here10;
        penalty_here10 = (query_value-subject_value10) * (query_value-subject_value10) + min(penalty_here9, min(penalty_here10, penalty_temp1));
        penalty_temp1 = penalty_here11;
        penalty_here11 = (query_value-subject_value11) * (query_value-subject_value11) + min(penalty_here10, min(penalty_here11, penalty_temp0));
        penalty_temp0 = penalty_here12;
        penalty_here12 = (query_value-subject_value12) * (query_value-subject_value12) + min(penalty_here11, min(penalty_here12, penalty_temp1));
        penalty_temp1 = penalty_here13;
        penalty_here13 = (query_value-subject_value13) * (query_value-subject_value13) + min(penalty_here12, min(penalty_here13, penalty_temp0));
        penalty_temp0 = penalty_here14;
        penalty_here14 = (query_value-subject_value14) * (query_value-subject_value14) + min(penalty_here13, min(penalty_here14, penalty_temp1));
        penalty_temp1 = penalty_here15;
        penalty_here15 = (query_value-subject_value15) * (query_value-subject_value15) + min(penalty_here14, min(penalty_here15, penalty_temp0));
        penalty_temp0 = penalty_here16;
        penalty_here16 = (query_value-subject_value16) * (query_value-subject_value16) + min(penalty_here15, min(penalty_here16, penalty_temp1));
        penalty_temp1 = penalty_here17;
        penalty_here17 = (query_value-subject_value17) * (query_value-subject_value17) + min(penalty_here16, min(penalty_here17, penalty_temp0));
        penalty_temp0 = penalty_here18;
        penalty_here18 = (query_value-subject_value18) * (query_value-subject_value18) + min(penalty_here17, min(penalty_here18, penalty_temp1));
        penalty_temp1 = penalty_here19;
        penalty_here19 = (query_value-subject_value19) * (query_value-subject_value19) + min(penalty_here18, min(penalty_here19, penalty_temp0));
        penalty_temp0 = penalty_here20;
        penalty_here20 = (query_value-subject_value20) * (query_value-subject_value20) + min(penalty_here19, min(penalty_here20, penalty_temp1));
        penalty_temp1 = penalty_here21;
        penalty_here21 = (query_value-subject_value21) * (query_value-subject_value21) + min(penalty_here20, min(penalty_here21, penalty_temp0));
        penalty_temp0 = penalty_here22;
        penalty_here22 = (query_value-subject_value22) * (query_value-subject_value22) + min(penalty_here21, min(penalty_here22, penalty_temp1));
        penalty_temp1 = penalty_here23;
        penalty_here23 = (query_value-subject_value23) * (query_value-subject_value23) + min(penalty_here22, min(penalty_here23, penalty_temp0));
        penalty_temp0 = penalty_here24;
        penalty_here24 = (query_value-subject_value24) * (query_value-subject_value24) + min(penalty_here23, min(penalty_here24, penalty_temp1));
        penalty_temp1 = penalty_here25;
        penalty_here25 = (query_value-subject_value25) * (query_value-subject_value25) + min(penalty_here24, min(penalty_here25, penalty_temp0));
        penalty_temp0 = penalty_here26;
        penalty_here26 = (query_value-subject_value26) * (query_value-subject_value26) + min(penalty_here25, min(penalty_here26, penalty_temp1));
        penalty_temp1 = penalty_here27;
        penalty_here27 = (query_value-subject_value27) * (query_value-subject_value27) + min(penalty_here26, min(penalty_here27, penalty_temp0));
        penalty_temp0 = penalty_here28;
        penalty_here28 = (query_value-subject_value28) * (query_value-subject_value28) + min(penalty_here27, min(penalty_here28, penalty_temp1));
        penalty_temp1 = penalty_here29;
        penalty_here29 = (query_value-subject_value29) * (query_value-subject_value29) + min(penalty_here28, min(penalty_here29, penalty_temp0));
        penalty_temp0 = penalty_here30;
        penalty_here30 = (query_value-subject_value30) * (query_value-subject_value30) + min(penalty_here29, min(penalty_here30, penalty_temp1));
        penalty_here31 = (query_value-subject_value31) * (query_value-subject_value31) + min(penalty_here30, min(penalty_here31, penalty_temp0));

        if (counter % 32 == 0) new_query_value = cQuery[i + 2*thid - 1];
        query_value = __shfl_up_sync(0xFFFFFFFF, query_value, 1, 32);
        if (thid == 0) query_value = new_query_value;
        new_query_value = __shfl_down_sync(0xFFFFFFFF, new_query_value, 1, 32);
        counter++;

        penalty_diag = penalty_left;
        penalty_left = __shfl_up_sync(0xFFFFFFFF, penalty_here31, 1, 32);
        if (thid == 0) penalty_left = INFINITY;
    }

    // Final cell update (epilogue, matches the cuDTW source)
    penalty_temp0 = penalty_here0;
    penalty_here0 = (query_value-subject_value0) * (query_value-subject_value0) + min(penalty_left, min(penalty_here0, penalty_diag));
    penalty_temp1 = penalty_here1;
    penalty_here1 = (query_value-subject_value1) * (query_value-subject_value1) + min(penalty_here0, min(penalty_here1, penalty_temp0));
    penalty_temp0 = penalty_here2;
    penalty_here2 = (query_value-subject_value2) * (query_value-subject_value2) + min(penalty_here1, min(penalty_here2, penalty_temp1));
    penalty_temp1 = penalty_here3;
    penalty_here3 = (query_value-subject_value3) * (query_value-subject_value3) + min(penalty_here2, min(penalty_here3, penalty_temp0));
    penalty_temp0 = penalty_here4;
    penalty_here4 = (query_value-subject_value4) * (query_value-subject_value4) + min(penalty_here3, min(penalty_here4, penalty_temp1));
    penalty_temp1 = penalty_here5;
    penalty_here5 = (query_value-subject_value5) * (query_value-subject_value5) + min(penalty_here4, min(penalty_here5, penalty_temp0));
    penalty_temp0 = penalty_here6;
    penalty_here6 = (query_value-subject_value6) * (query_value-subject_value6) + min(penalty_here5, min(penalty_here6, penalty_temp1));
    penalty_temp1 = penalty_here7;
    penalty_here7 = (query_value-subject_value7) * (query_value-subject_value7) + min(penalty_here6, min(penalty_here7, penalty_temp0));
    penalty_temp0 = penalty_here8;
    penalty_here8 = (query_value-subject_value8) * (query_value-subject_value8) + min(penalty_here7, min(penalty_here8, penalty_temp1));
    penalty_temp1 = penalty_here9;
    penalty_here9 = (query_value-subject_value9) * (query_value-subject_value9) + min(penalty_here8, min(penalty_here9, penalty_temp0));
    penalty_temp0 = penalty_here10;
    penalty_here10 = (query_value-subject_value10) * (query_value-subject_value10) + min(penalty_here9, min(penalty_here10, penalty_temp1));
    penalty_temp1 = penalty_here11;
    penalty_here11 = (query_value-subject_value11) * (query_value-subject_value11) + min(penalty_here10, min(penalty_here11, penalty_temp0));
    penalty_temp0 = penalty_here12;
    penalty_here12 = (query_value-subject_value12) * (query_value-subject_value12) + min(penalty_here11, min(penalty_here12, penalty_temp1));
    penalty_temp1 = penalty_here13;
    penalty_here13 = (query_value-subject_value13) * (query_value-subject_value13) + min(penalty_here12, min(penalty_here13, penalty_temp0));
    penalty_temp0 = penalty_here14;
    penalty_here14 = (query_value-subject_value14) * (query_value-subject_value14) + min(penalty_here13, min(penalty_here14, penalty_temp1));
    penalty_temp1 = penalty_here15;
    penalty_here15 = (query_value-subject_value15) * (query_value-subject_value15) + min(penalty_here14, min(penalty_here15, penalty_temp0));
    penalty_temp0 = penalty_here16;
    penalty_here16 = (query_value-subject_value16) * (query_value-subject_value16) + min(penalty_here15, min(penalty_here16, penalty_diag));
    penalty_temp1 = penalty_here17;
    penalty_here17 = (query_value-subject_value17) * (query_value-subject_value17) + min(penalty_here16, min(penalty_here17, penalty_temp0));
    penalty_temp0 = penalty_here18;
    penalty_here18 = (query_value-subject_value18) * (query_value-subject_value18) + min(penalty_here17, min(penalty_here18, penalty_temp1));
    penalty_temp1 = penalty_here19;
    penalty_here19 = (query_value-subject_value19) * (query_value-subject_value19) + min(penalty_here18, min(penalty_here19, penalty_temp0));
    penalty_temp0 = penalty_here20;
    penalty_here20 = (query_value-subject_value20) * (query_value-subject_value20) + min(penalty_here19, min(penalty_here20, penalty_temp1));
    penalty_temp1 = penalty_here21;
    penalty_here21 = (query_value-subject_value21) * (query_value-subject_value21) + min(penalty_here20, min(penalty_here21, penalty_temp0));
    penalty_temp0 = penalty_here22;
    penalty_here22 = (query_value-subject_value22) * (query_value-subject_value22) + min(penalty_here21, min(penalty_here22, penalty_temp1));
    penalty_temp1 = penalty_here23;
    penalty_here23 = (query_value-subject_value23) * (query_value-subject_value23) + min(penalty_here22, min(penalty_here23, penalty_temp0));
    penalty_temp0 = penalty_here24;
    penalty_here24 = (query_value-subject_value24) * (query_value-subject_value24) + min(penalty_here23, min(penalty_here24, penalty_temp1));
    penalty_temp1 = penalty_here25;
    penalty_here25 = (query_value-subject_value25) * (query_value-subject_value25) + min(penalty_here24, min(penalty_here25, penalty_temp0));
    penalty_temp0 = penalty_here26;
    penalty_here26 = (query_value-subject_value26) * (query_value-subject_value26) + min(penalty_here25, min(penalty_here26, penalty_temp1));
    penalty_temp1 = penalty_here27;
    penalty_here27 = (query_value-subject_value27) * (query_value-subject_value27) + min(penalty_here26, min(penalty_here27, penalty_temp0));
    penalty_temp0 = penalty_here28;
    penalty_here28 = (query_value-subject_value28) * (query_value-subject_value28) + min(penalty_here27, min(penalty_here28, penalty_temp1));
    penalty_temp1 = penalty_here29;
    penalty_here29 = (query_value-subject_value29) * (query_value-subject_value29) + min(penalty_here28, min(penalty_here29, penalty_temp0));
    penalty_temp0 = penalty_here30;
    penalty_here30 = (query_value-subject_value30) * (query_value-subject_value30) + min(penalty_here29, min(penalty_here30, penalty_temp1));
    penalty_here31 = (query_value-subject_value31) * (query_value-subject_value31) + min(penalty_here30, min(penalty_here31, penalty_temp0));

    if (thid == blockDim.x - 1) Dist[blid] = penalty_here31;
}

// ===== Persistent device state =====
static int    g_num_entries  = 0;
static int    g_num_features = 0;
static float* d_subjects  = nullptr;
static float* d_distances = nullptr;

extern "C" void solution_free(void)
{
    if (d_subjects)  { cudaFree(d_subjects);  d_subjects  = nullptr; }
    if (d_distances) { cudaFree(d_distances); d_distances = nullptr; }
    g_num_entries  = 0;
    g_num_features = 0;
}

extern "C" void solution_compute(int          num_entries,
                                 int          num_features,
                                 const float* subjects,
                                 const float* query,
                                 float*       distances)
{
    if (num_features != 1023) {
        fprintf(stderr,
                "[gpu_baseline] cuDTW SHFL_FULLDTW_1023 is length-specialized "
                "to num_features=1023; got %d\n",
                num_features);
    }

    size_t subj_bytes  = (size_t)num_entries * num_features * sizeof(float);
    size_t dist_bytes  = (size_t)num_entries * sizeof(float);

    if (g_num_entries != num_entries || g_num_features != num_features) {
        solution_free();
        cudaMalloc(&d_subjects,  subj_bytes);
        cudaMalloc(&d_distances, dist_bytes);
        g_num_entries  = num_entries;
        g_num_features = num_features;
    }

    cudaMemcpy(d_subjects, subjects, subj_bytes, cudaMemcpyHostToDevice);
    // Match cuDTW main.cu line 164: query lives in __constant__ memory.
    cudaMemcpyToSymbol(cQuery, query,
                       (size_t)num_features * sizeof(float));

    // Launch matches cuDTW DTW.hpp shfl_FullDTW_1023 dispatch:
    //   grid  = (num_entries, 1, 1)
    //   block = (32, 1, 1)
    const dim3 grid (num_entries, 1, 1);
    const dim3 block(WARP_SIZE,   1, 1);

    shfl_FullDTW_1023<<<grid, block>>>(d_subjects, d_distances,
                                       num_entries, num_features);

    cudaMemcpy(distances, d_distances,
               (size_t)num_entries * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}
