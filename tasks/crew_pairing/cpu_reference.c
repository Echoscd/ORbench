// cpu_reference.c -- crew_pairing CPU baseline (SPPRC + Greedy Set Cover)
//
// Two-phase algorithm similar to column generation:
//   Phase 1: For each starting leg, use SPPRC (Shortest Path with Resource
//            Constraints) to generate high-quality candidate pairings.
//   Phase 2: Greedy set cover -- sort candidates by cost_per_leg, greedily
//            select non-overlapping pairings to cover all legs.
//
// NO file I/O here. All I/O is handled by task_io_cpu.c.

#include <stdlib.h>
#include <string.h>
#include <float.h>

// ===== Tuning parameters =====
#define MAX_LABELS_PER_NODE   50
#define MAX_PAIRINGS_PER_START 20
#define MAX_PAIRING_LEGS      25
#define MAX_CONNECT_MIN       (36 * 60)   // 36 hours max connection time
#define POS_FEE               10000.0f
#define MAX_CANDIDATES        200000
#define MAX_PQ_SIZE           500000
#define MAX_CONNECTIONS_TOTAL 2000000

// ===== Module-level state =====
static int g_N;
static int g_num_stations;
static int g_base_station;
static const int* g_dep_min;
static const int* g_arr_min;
static const int* g_dep_stn;
static const int* g_arr_stn;
static float g_duty_rate;
static float g_pairing_rate;
static int g_max_duty_min;
static int g_max_block_min;
static int g_max_legs_duty;
static int g_min_rest_min;

// Connection graph (CSR format)
static int* g_conn_offsets;   // N+1
static int* g_conn_targets;   // total connections
static int  g_conn_total;

// ===== SPPRC data structures =====

typedef struct {
    float cost;
    int   duty_start_min;
    int   duty_block_min;
    int   duty_legs;
} Label;

typedef struct {
    int flight_idx;   // -1 = root
    int label_idx;
} Pred;

typedef struct {
    float cost;
    int   flight_idx;
    int   label_idx;
} PQEntry;

// Candidate pairing from SPPRC
typedef struct {
    float cost;
    float cost_per_leg;
    int   num_legs;
    int   legs[MAX_PAIRING_LEGS];
} Candidate;

// ===== Min-heap for priority queue =====

static PQEntry* g_pq;
static int g_pq_size;

static void pq_init(void) { g_pq_size = 0; }

static void pq_push(float cost, int fi, int li) {
    if (g_pq_size >= MAX_PQ_SIZE) return;
    int i = g_pq_size++;
    g_pq[i].cost = cost;
    g_pq[i].flight_idx = fi;
    g_pq[i].label_idx = li;
    // sift up
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (g_pq[parent].cost <= g_pq[i].cost) break;
        PQEntry tmp = g_pq[i];
        g_pq[i] = g_pq[parent];
        g_pq[parent] = tmp;
        i = parent;
    }
}

static PQEntry pq_pop(void) {
    PQEntry top = g_pq[0];
    g_pq[0] = g_pq[--g_pq_size];
    // sift down
    int i = 0;
    for (;;) {
        int left = 2 * i + 1, right = 2 * i + 2, smallest = i;
        if (left < g_pq_size && g_pq[left].cost < g_pq[smallest].cost)
            smallest = left;
        if (right < g_pq_size && g_pq[right].cost < g_pq[smallest].cost)
            smallest = right;
        if (smallest == i) break;
        PQEntry tmp = g_pq[i];
        g_pq[i] = g_pq[smallest];
        g_pq[smallest] = tmp;
        i = smallest;
    }
    return top;
}

// ===== Per-node label storage =====
// labels_at[flight_idx][j] for j in 0..label_count[flight_idx]-1

static Label** g_labels;       // g_labels[i] -> array of MAX_LABELS_PER_NODE
static Pred**  g_preds;        // g_preds[i]  -> array of MAX_LABELS_PER_NODE
static int*    g_label_count;  // number of active labels at each node

// ===== Candidate pool =====
static Candidate* g_candidates;
static int g_num_candidates;

// ===== Working buffers =====
static int* g_path_buf;  // for path reconstruction

void solution_init(int N, int num_stations, int base_station,
                   const int* dep_minutes, const int* arr_minutes,
                   const int* dep_stations, const int* arr_stations,
                   float duty_cost_per_hour, float pairing_cost_per_hour,
                   int max_duty_min, int max_block_min,
                   int max_legs_duty, int min_rest_min) {
    g_N = N;
    g_num_stations = num_stations;
    g_base_station = base_station;
    g_dep_min = dep_minutes;
    g_arr_min = arr_minutes;
    g_dep_stn = dep_stations;
    g_arr_stn = arr_stations;
    g_duty_rate = duty_cost_per_hour;
    g_pairing_rate = pairing_cost_per_hour;
    g_max_duty_min = max_duty_min;
    g_max_block_min = max_block_min;
    g_max_legs_duty = max_legs_duty;
    g_min_rest_min = min_rest_min;

    // Build connection graph
    // First pass: count connections per leg
    g_conn_offsets = (int*)calloc((size_t)(N + 1), sizeof(int));
    // Temporary: collect all connections
    int* temp_targets = (int*)malloc(MAX_CONNECTIONS_TOTAL * sizeof(int));
    int total = 0;

    for (int i = 0; i < N; i++) {
        g_conn_offsets[i] = total;
        for (int j = i + 1; j < N; j++) {
            int gap = dep_minutes[j] - arr_minutes[i];
            if (gap >= MAX_CONNECT_MIN) break;  // sorted by dep_min
            if (gap < 0) continue;
            // Station continuity: crew must be at the same station
            if (arr_stations[i] != dep_stations[j]) continue;
            if (total < MAX_CONNECTIONS_TOTAL) {
                temp_targets[total++] = j;
            }
        }
    }
    g_conn_offsets[N] = total;
    g_conn_total = total;

    g_conn_targets = (int*)malloc((size_t)total * sizeof(int));
    memcpy(g_conn_targets, temp_targets, (size_t)total * sizeof(int));
    free(temp_targets);

    // Allocate SPPRC working memory
    g_pq = (PQEntry*)malloc(MAX_PQ_SIZE * sizeof(PQEntry));

    g_labels = (Label**)malloc((size_t)N * sizeof(Label*));
    g_preds  = (Pred**)malloc((size_t)N * sizeof(Pred*));
    g_label_count = (int*)malloc((size_t)N * sizeof(int));
    for (int i = 0; i < N; i++) {
        g_labels[i] = (Label*)malloc(MAX_LABELS_PER_NODE * sizeof(Label));
        g_preds[i]  = (Pred*)malloc(MAX_LABELS_PER_NODE * sizeof(Pred));
    }

    g_candidates = (Candidate*)malloc(MAX_CANDIDATES * sizeof(Candidate));
    g_path_buf = (int*)malloc(MAX_PAIRING_LEGS * sizeof(int));
}

// ===== Try to add a label with dominance checking =====
static void try_add_label(Label new_label, int fi, int pred_fi, int pred_li) {
    int new_duty_span = g_arr_min[fi] - new_label.duty_start_min;

    // Check dominance against existing labels
    int keep_count = 0;
    int dominated = 0;

    for (int i = 0; i < g_label_count[fi]; i++) {
        Label* ex = &g_labels[fi][i];
        int ex_duty_span = g_arr_min[fi] - ex->duty_start_min;

        // Existing dominates new?
        if (ex->cost <= new_label.cost &&
            ex_duty_span <= new_duty_span &&
            ex->duty_block_min <= new_label.duty_block_min &&
            ex->duty_legs <= new_label.duty_legs) {
            dominated = 1;
            break;
        }

        // New dominates existing? If so, skip existing
        if (new_label.cost <= ex->cost &&
            new_duty_span <= ex_duty_span &&
            new_label.duty_block_min <= ex->duty_block_min &&
            new_label.duty_legs <= ex->duty_legs) {
            continue;  // drop this existing label
        }

        // Keep existing
        if (keep_count != i) {
            g_labels[fi][keep_count] = g_labels[fi][i];
            g_preds[fi][keep_count] = g_preds[fi][i];
        }
        keep_count++;
    }

    if (dominated) return;

    g_label_count[fi] = keep_count;

    if (keep_count < MAX_LABELS_PER_NODE) {
        int idx = keep_count;
        g_labels[fi][idx] = new_label;
        g_preds[fi][idx].flight_idx = pred_fi;
        g_preds[fi][idx].label_idx = pred_li;
        g_label_count[fi] = keep_count + 1;
        pq_push(new_label.cost, fi, idx);
    } else {
        // Replace worst (highest cost) label
        int worst = -1;
        float worst_cost = new_label.cost;
        for (int i = 0; i < keep_count; i++) {
            if (g_labels[fi][i].cost > worst_cost) {
                worst_cost = g_labels[fi][i].cost;
                worst = i;
            }
        }
        if (worst >= 0) {
            g_labels[fi][worst] = new_label;
            g_preds[fi][worst].flight_idx = pred_fi;
            g_preds[fi][worst].label_idx = pred_li;
            pq_push(new_label.cost, fi, worst);
        }
    }
}

// ===== Reconstruct path from predecessors =====
static int reconstruct_path(int fi, int li, int* path) {
    int len = 0;
    int cur_fi = fi, cur_li = li;
    while (cur_fi >= 0 && len < MAX_PAIRING_LEGS) {
        path[len++] = cur_fi;
        Pred p = g_preds[cur_fi][cur_li];
        if (p.flight_idx < 0) break;
        cur_fi = p.flight_idx;
        cur_li = p.label_idx;
    }
    // Reverse
    for (int i = 0; i < len / 2; i++) {
        int tmp = path[i];
        path[i] = path[len - 1 - i];
        path[len - 1 - i] = tmp;
    }
    return len;
}

// ===== Phase 1: SPPRC from a single starting leg =====
// Returns number of candidates added
static int spprc_from_start(int start, Candidate* out, int max_out) {
    int N = g_N;

    // Clear labels
    for (int i = 0; i < N; i++) g_label_count[i] = 0;

    // Initial label at start node
    int block = g_arr_min[start] - g_dep_min[start];
    int duty_span = block;
    float pos_cost = (g_dep_stn[start] != g_base_station) ? POS_FEE : 0.0f;
    float init_cost = (float)duty_span * g_duty_rate / 60.0f
                    + (float)block * g_pairing_rate / 60.0f
                    + pos_cost;

    g_labels[start][0].cost = init_cost;
    g_labels[start][0].duty_start_min = g_dep_min[start];
    g_labels[start][0].duty_block_min = block;
    g_labels[start][0].duty_legs = 1;
    g_preds[start][0].flight_idx = -1;
    g_preds[start][0].label_idx = -1;
    g_label_count[start] = 1;

    pq_init();
    pq_push(init_cost, start, 0);

    // Best pairings heap (max-heap by cost, keep lowest MAX_PAIRINGS_PER_START)
    // Use simple array + insertion
    int n_best = 0;
    float best_costs[MAX_PAIRINGS_PER_START];
    int   best_fi[MAX_PAIRINGS_PER_START];
    int   best_li[MAX_PAIRINGS_PER_START];

    while (g_pq_size > 0) {
        PQEntry e = pq_pop();
        float cur_cost = e.cost;
        int cur_fi = e.flight_idx;
        int cur_li = e.label_idx;

        // Stale entry?
        if (cur_li >= g_label_count[cur_fi]) continue;
        if (cur_cost > g_labels[cur_fi][cur_li].cost + 0.01f) continue;

        Label* cur = &g_labels[cur_fi][cur_li];

        // Record this as a candidate pairing
        if (n_best < MAX_PAIRINGS_PER_START) {
            best_costs[n_best] = cur_cost;
            best_fi[n_best] = cur_fi;
            best_li[n_best] = cur_li;
            n_best++;
        } else {
            // Find worst (highest cost) in best array
            int worst = 0;
            for (int i = 1; i < n_best; i++) {
                if (best_costs[i] > best_costs[worst]) worst = i;
            }
            if (cur_cost < best_costs[worst]) {
                best_costs[worst] = cur_cost;
                best_fi[worst] = cur_fi;
                best_li[worst] = cur_li;
            }
        }

        // Check path length before extending
        // (approximate: duty_legs is only current duty, but total path could be longer)
        // We use a simple heuristic: if cost is very high, skip

        // Extend to neighbors
        int conn_start = g_conn_offsets[cur_fi];
        int conn_end   = g_conn_offsets[cur_fi + 1];

        for (int ci = conn_start; ci < conn_end; ci++) {
            int next = g_conn_targets[ci];
            int sit_time = g_dep_min[next] - g_arr_min[cur_fi];
            int next_block = g_arr_min[next] - g_dep_min[next];

            if (sit_time < g_min_rest_min) {
                // Same duty: check constraints
                int new_legs = cur->duty_legs + 1;
                int new_block = cur->duty_block_min + next_block;
                int new_span = g_arr_min[next] - cur->duty_start_min;

                if (new_legs <= g_max_legs_duty &&
                    new_block <= g_max_block_min &&
                    new_span <= g_max_duty_min) {
                    // Cost increment: duty time extension + block time
                    float cost_incr =
                        (float)(g_arr_min[next] - g_arr_min[cur_fi]) * g_duty_rate / 60.0f +
                        (float)next_block * g_pairing_rate / 60.0f;

                    Label nl;
                    nl.cost = cur->cost + cost_incr;
                    nl.duty_start_min = cur->duty_start_min;
                    nl.duty_block_min = new_block;
                    nl.duty_legs = new_legs;
                    try_add_label(nl, next, cur_fi, cur_li);
                }
            }

            // New duty after rest (sit_time >= min_rest only)
            if (sit_time >= g_min_rest_min) {
                int new_span = g_arr_min[next] - g_dep_min[next];
                float cost_incr =
                    (float)new_span * g_duty_rate / 60.0f +
                    (float)next_block * g_pairing_rate / 60.0f;

                Label nl;
                nl.cost = cur->cost + cost_incr;
                nl.duty_start_min = g_dep_min[next];
                nl.duty_block_min = next_block;
                nl.duty_legs = 1;
                try_add_label(nl, next, cur_fi, cur_li);
            }
        }
    }

    // Reconstruct paths for best pairings
    int count = 0;
    for (int i = 0; i < n_best && count < max_out; i++) {
        int path_len = reconstruct_path(best_fi[i], best_li[i], g_path_buf);
        if (path_len <= 0 || path_len > MAX_PAIRING_LEGS) continue;

        out[count].cost = best_costs[i];
        out[count].num_legs = path_len;
        out[count].cost_per_leg = best_costs[i] / (float)path_len;
        memcpy(out[count].legs, g_path_buf, (size_t)path_len * sizeof(int));
        count++;
    }
    return count;
}

// Precomputed single-leg costs (for savings metric)
static float* g_single_leg_cost;

// Compare candidates by savings (descending)
static int cmp_candidate_savings(const void* a, const void* b) {
    // cost_per_leg is repurposed to store negative savings (so ascending sort = descending savings)
    float sa = ((const Candidate*)a)->cost_per_leg;
    float sb = ((const Candidate*)b)->cost_per_leg;
    return (sa > sb) - (sa < sb);
}

// ===== Simple greedy solver (fallback / comparison) =====
// Same algorithm as the original baseline: sequential first-fit
#define MAX_GREEDY_PAIRINGS 4096
static int gp_last_leg[MAX_GREEDY_PAIRINGS];
static int gp_duty_start[MAX_GREEDY_PAIRINGS];
static int gp_duty_block[MAX_GREEDY_PAIRINGS];
static int gp_duty_legs[MAX_GREEDY_PAIRINGS];

static void solve_greedy(int N, int* assignments) {
    int num_p = 0;
    for (int i = 0; i < N; i++) {
        int block = g_arr_min[i] - g_dep_min[i];
        int best_p = -1;

        for (int p = 0; p < num_p; p++) {
            int last = gp_last_leg[p];
            if (g_arr_stn[last] != g_dep_stn[i]) continue;
            int gap = g_dep_min[i] - g_arr_min[last];
            if (gap < 0) continue;

            if (gap >= g_min_rest_min) {
                best_p = p;
                break;
            } else {
                int nl = gp_duty_legs[p] + 1;
                int nb = gp_duty_block[p] + block;
                int ns = g_arr_min[i] - gp_duty_start[p];
                if (nl <= g_max_legs_duty && nb <= g_max_block_min && ns <= g_max_duty_min) {
                    best_p = p;
                    break;
                }
            }
        }

        if (best_p >= 0) {
            int last = gp_last_leg[best_p];
            int gap = g_dep_min[i] - g_arr_min[last];
            assignments[i] = best_p;
            gp_last_leg[best_p] = i;
            if (gap >= g_min_rest_min) {
                gp_duty_start[best_p] = g_dep_min[i];
                gp_duty_block[best_p] = block;
                gp_duty_legs[best_p] = 1;
            } else {
                gp_duty_block[best_p] += block;
                gp_duty_legs[best_p]++;
            }
        } else {
            if (num_p < MAX_GREEDY_PAIRINGS) {
                int p = num_p++;
                assignments[i] = p;
                gp_last_leg[p] = i;
                gp_duty_start[p] = g_dep_min[i];
                gp_duty_block[p] = block;
                gp_duty_legs[p] = 1;
            } else {
                assignments[i] = 0;
            }
        }
    }
}

// ===== Compute total cost from assignments (for comparison) =====
static float compute_cost(int N, const int* assignments) {
    // Find max pairing id
    int max_p = -1;
    for (int i = 0; i < N; i++)
        if (assignments[i] > max_p) max_p = assignments[i];
    int num_p = max_p + 1;

    // Group legs by pairing, sorted by departure
    // Simple approach: iterate through legs (already sorted by dep_min)
    float total = 0.0f;

    // Allocate large enough buffer for any pairing size
    int* legs = (int*)malloc((size_t)N * sizeof(int));

    for (int p = 0; p < num_p; p++) {
        // Collect legs for this pairing
        int cnt = 0;
        for (int i = 0; i < N; i++) {
            if (assignments[i] == p) legs[cnt++] = i;
        }
        if (cnt == 0) continue;

        // Block hours
        float block_h = 0.0f;
        for (int j = 0; j < cnt; j++)
            block_h += (float)(g_arr_min[legs[j]] - g_dep_min[legs[j]]) / 60.0f;

        // Duty hours (split by rest)
        float duty_h = 0.0f;
        int duty_start = g_dep_min[legs[0]];
        int prev_arr = g_arr_min[legs[0]];
        for (int j = 1; j < cnt; j++) {
            int rest = g_dep_min[legs[j]] - prev_arr;
            if (rest >= g_min_rest_min) {
                duty_h += (float)(prev_arr - duty_start) / 60.0f;
                duty_start = g_dep_min[legs[j]];
            }
            prev_arr = g_arr_min[legs[j]];
        }
        duty_h += (float)(prev_arr - duty_start) / 60.0f;

        float pos = (g_dep_stn[legs[0]] != g_base_station) ? POS_FEE : 0.0f;
        total += duty_h * g_duty_rate + block_h * g_pairing_rate + pos;
    }
    free(legs);
    return total;
}

// ===== Phase 2: Greedy set cover =====
void solution_compute(int N, int* assignments) {
    // Precompute cost of each leg as a single-leg pairing
    if (!g_single_leg_cost) {
        g_single_leg_cost = (float*)malloc((size_t)N * sizeof(float));
    }
    for (int i = 0; i < N; i++) {
        int block = g_arr_min[i] - g_dep_min[i];
        float pos = (g_dep_stn[i] != g_base_station) ? POS_FEE : 0.0f;
        g_single_leg_cost[i] = (float)block * g_duty_rate / 60.0f
                             + (float)block * g_pairing_rate / 60.0f
                             + pos;
    }

    // === Strategy A: SPPRC + savings-based greedy set cover ===
    int* assign_spprc = (int*)malloc((size_t)N * sizeof(int));
    g_num_candidates = 0;

    for (int s = 0; s < N; s++) {
        int room = MAX_CANDIDATES - g_num_candidates;
        if (room <= 0) break;
        int max_per = room < MAX_PAIRINGS_PER_START ? room : MAX_PAIRINGS_PER_START;
        int added = spprc_from_start(s, &g_candidates[g_num_candidates], max_per);
        g_num_candidates += added;
    }

    // Compute savings
    for (int c = 0; c < g_num_candidates; c++) {
        Candidate* cand = &g_candidates[c];
        float solo_total = 0.0f;
        for (int j = 0; j < cand->num_legs; j++)
            solo_total += g_single_leg_cost[cand->legs[j]];
        cand->cost_per_leg = -(solo_total - cand->cost);
    }

    qsort(g_candidates, (size_t)g_num_candidates, sizeof(Candidate), cmp_candidate_savings);

    char* covered = (char*)calloc((size_t)N, sizeof(char));
    int pairing_id = 0;
    for (int i = 0; i < N; i++) assign_spprc[i] = -1;

    for (int c = 0; c < g_num_candidates; c++) {
        Candidate* cand = &g_candidates[c];
        if (cand->cost_per_leg > 0.0f) break;

        int overlap = 0;
        for (int j = 0; j < cand->num_legs; j++) {
            if (covered[cand->legs[j]]) { overlap = 1; break; }
        }
        if (overlap) continue;

        for (int j = 0; j < cand->num_legs; j++) {
            covered[cand->legs[j]] = 1;
            assign_spprc[cand->legs[j]] = pairing_id;
        }
        pairing_id++;
    }
    for (int i = 0; i < N; i++)
        if (assign_spprc[i] < 0) assign_spprc[i] = pairing_id++;

    float cost_spprc = compute_cost(N, assign_spprc);
    free(covered);

    // === Strategy B: Simple sequential greedy ===
    int* assign_greedy = (int*)malloc((size_t)N * sizeof(int));
    solve_greedy(N, assign_greedy);
    float cost_greedy = compute_cost(N, assign_greedy);

    // === Pick better solution ===
    if (cost_spprc <= cost_greedy) {
        memcpy(assignments, assign_spprc, (size_t)N * sizeof(int));
    } else {
        memcpy(assignments, assign_greedy, (size_t)N * sizeof(int));
    }

    free(assign_spprc);
    free(assign_greedy);
}
