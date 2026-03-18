// cpu_reference.c -- collision_detection CPU baseline
//
// Grid broad phase + SAT narrow phase for 2D convex polygon collision.
// NO file I/O. All I/O handled by task_io_cpu.c.

#include <stdlib.h>
#include <string.h>
#include <math.h>

// ===== Module-level state =====
static int g_N;
static int g_total_verts;
static float g_world_size;
static float g_cell_size;
static int g_grid_dim;

static const int*   g_poly_offsets;
static const float* g_vx;
static const float* g_vy;
static const float* g_aabb;  // N*4: min_x, min_y, max_x, max_y

// Grid: cell_heads[cell] = first polygon index, next[entry] = next polygon
// Using a flat hash map approach
#define MAX_GRID_CELLS 1000003  // prime for hashing
#define MAX_ENTRIES    8000000  // max polygon-to-cell entries

static int* g_cell_heads;    // MAX_GRID_CELLS, -1 = empty
static int* g_entry_poly;    // MAX_ENTRIES: polygon ID for each entry
static int* g_entry_next;    // MAX_ENTRIES: next entry in same cell
static int  g_num_entries;

// Candidate pairs (sorted unique)
#define MAX_PAIRS 20000000
static int* g_pair_i;
static int* g_pair_j;
static int  g_num_pairs;

void solution_init(int N, int total_verts,
                   int world_size_x100, int cell_size_x100,
                   const int* poly_offsets,
                   const float* vertices_x, const float* vertices_y,
                   const float* aabb) {
    g_N = N;
    g_total_verts = total_verts;
    g_world_size = (float)world_size_x100 / 100.0f;
    g_cell_size = (float)cell_size_x100 / 100.0f;
    g_grid_dim = (int)ceilf(g_world_size / g_cell_size) + 1;
    g_poly_offsets = poly_offsets;
    g_vx = vertices_x;
    g_vy = vertices_y;
    g_aabb = aabb;

    g_cell_heads = (int*)malloc(MAX_GRID_CELLS * sizeof(int));
    g_entry_poly = (int*)malloc(MAX_ENTRIES * sizeof(int));
    g_entry_next = (int*)malloc(MAX_ENTRIES * sizeof(int));
    g_pair_i = (int*)malloc(MAX_PAIRS * sizeof(int));
    g_pair_j = (int*)malloc(MAX_PAIRS * sizeof(int));
}

// ===== Grid hash =====
static int grid_hash(int gx, int gy) {
    unsigned int h = (unsigned int)(gx * 73856093) ^ (unsigned int)(gy * 19349663);
    return (int)(h % MAX_GRID_CELLS);
}

// ===== AABB overlap =====
static int aabb_test(int i, int j) {
    return g_aabb[i*4+2] >= g_aabb[j*4+0] &&
           g_aabb[j*4+2] >= g_aabb[i*4+0] &&
           g_aabb[i*4+3] >= g_aabb[j*4+1] &&
           g_aabb[j*4+3] >= g_aabb[i*4+1];
}

// ===== SAT test =====
static int sat_test(int poly_a, int poly_b) {
    int sa = g_poly_offsets[poly_a], ea = g_poly_offsets[poly_a + 1];
    int sb = g_poly_offsets[poly_b], eb = g_poly_offsets[poly_b + 1];
    int na = ea - sa, nb = eb - sb;

    // Test edges of polygon A
    for (int i = 0; i < na; i++) {
        int j = (i + 1) % na;
        float nx = -(g_vy[sa + j] - g_vy[sa + i]);
        float ny =   g_vx[sa + j] - g_vx[sa + i];

        float min_a = 1e30f, max_a = -1e30f;
        for (int k = 0; k < na; k++) {
            float p = nx * g_vx[sa + k] + ny * g_vy[sa + k];
            if (p < min_a) min_a = p;
            if (p > max_a) max_a = p;
        }

        float min_b = 1e30f, max_b = -1e30f;
        for (int k = 0; k < nb; k++) {
            float p = nx * g_vx[sb + k] + ny * g_vy[sb + k];
            if (p < min_b) min_b = p;
            if (p > max_b) max_b = p;
        }

        if (max_a < min_b || max_b < min_a) return 0;
    }

    // Test edges of polygon B
    for (int i = 0; i < nb; i++) {
        int j = (i + 1) % nb;
        float nx = -(g_vy[sb + j] - g_vy[sb + i]);
        float ny =   g_vx[sb + j] - g_vx[sb + i];

        float min_a = 1e30f, max_a = -1e30f;
        for (int k = 0; k < na; k++) {
            float p = nx * g_vx[sa + k] + ny * g_vy[sa + k];
            if (p < min_a) min_a = p;
            if (p > max_a) max_a = p;
        }

        float min_b = 1e30f, max_b = -1e30f;
        for (int k = 0; k < nb; k++) {
            float p = nx * g_vx[sb + k] + ny * g_vy[sb + k];
            if (p < min_b) min_b = p;
            if (p > max_b) max_b = p;
        }

        if (max_a < min_b || max_b < min_a) return 0;
    }

    return 1;
}

// Sort packed pairs
static int cmp_ll(const void* a, const void* b) {
    long long va = *(const long long*)a;
    long long vb = *(const long long*)b;
    return (va > vb) - (va < vb);
}

void solution_compute(int N, int* counts) {
    memset(counts, 0, (size_t)N * sizeof(int));

    // === Phase 1: Build grid ===
    memset(g_cell_heads, -1, MAX_GRID_CELLS * sizeof(int));
    g_num_entries = 0;

    for (int i = 0; i < N; i++) {
        int gx0 = (int)(g_aabb[i*4+0] / g_cell_size);
        int gy0 = (int)(g_aabb[i*4+1] / g_cell_size);
        int gx1 = (int)(g_aabb[i*4+2] / g_cell_size);
        int gy1 = (int)(g_aabb[i*4+3] / g_cell_size);
        if (gx0 < 0) gx0 = 0;
        if (gy0 < 0) gy0 = 0;

        for (int gx = gx0; gx <= gx1; gx++) {
            for (int gy = gy0; gy <= gy1; gy++) {
                int h = grid_hash(gx, gy);
                if (g_num_entries < MAX_ENTRIES) {
                    int e = g_num_entries++;
                    g_entry_poly[e] = i;
                    g_entry_next[e] = g_cell_heads[h];
                    g_cell_heads[h] = e;
                }
            }
        }
    }

    // === Phase 2: Collect candidate pairs from grid cells ===
    g_num_pairs = 0;

    for (int c = 0; c < MAX_GRID_CELLS; c++) {
        if (g_cell_heads[c] < 0) continue;

        // Collect polygons in this cell
        int cell_polys[512];
        int cell_count = 0;
        int e = g_cell_heads[c];
        while (e >= 0 && cell_count < 512) {
            cell_polys[cell_count++] = g_entry_poly[e];
            e = g_entry_next[e];
        }

        // Generate pairs
        for (int a = 0; a < cell_count; a++) {
            for (int b = a + 1; b < cell_count; b++) {
                int pi = cell_polys[a], pj = cell_polys[b];
                if (pi > pj) { int tmp = pi; pi = pj; pj = tmp; }
                if (g_num_pairs < MAX_PAIRS) {
                    g_pair_i[g_num_pairs] = pi;
                    g_pair_j[g_num_pairs] = pj;
                    g_num_pairs++;
                }
            }
        }
    }

    // === Phase 3: Sort and deduplicate pairs ===
    long long* packed = (long long*)malloc((size_t)g_num_pairs * sizeof(long long));
    for (int k = 0; k < g_num_pairs; k++) {
        packed[k] = ((long long)g_pair_i[k] << 32) | (unsigned int)g_pair_j[k];
    }
    qsort(packed, (size_t)g_num_pairs, sizeof(long long), cmp_ll);

    // === Phase 4: Deduplicate + AABB test + SAT ===
    int prev = -1;
    for (int k = 0; k < g_num_pairs; k++) {
        // Skip duplicates
        if (k > 0 && packed[k] == packed[k - 1]) continue;

        int pi = (int)(packed[k] >> 32);
        int pj = (int)(packed[k] & 0xFFFFFFFF);

        // AABB test
        if (!aabb_test(pi, pj)) continue;

        // SAT test
        if (sat_test(pi, pj)) {
            counts[pi]++;
            counts[pj]++;
        }
    }

    free(packed);
}
