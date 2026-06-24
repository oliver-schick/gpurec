// Fast C++ stochastic-backtrack reconciliation sampler (undated DTL+O), the
// compiled analogue of gpurec/core/sampler.py. Given a family's forward DP
// tables (Pi, Pibar in log2) plus the rates/E/Ebar the forward used, it draws
// n reconciliations by stochastically backtracking along the DP sum -- exactly
// AleRax MultiModel::backtrace / ALE sample_undated -- and accumulates, per
// species branch s:
//   copies[s]   += #{S, SL, leaf} events at s   (ALE branch_counts["copies"])
//   presence[s] += 1 per sample in which s is occupied
//
// Event weights at clade c on species s (children s1,s2), per split c->(L,R):
//   D : pD*Pi[L,s]*Pi[R,s]   T(R moves): Pi[L,s]*Pibar[R,s]   T(L moves): Pi[R,s]*Pibar[L,s]
//   S : pS*Pi[L,s1]*Pi[R,s2] (+ the swapped orientation)
//   DL: 2*pD*E*Pi[c,s] (self-loop)   TL-lost: Pi[c,s]*Ebar (self-loop)
//   TL-move: Pibar[c,s]*E   SL: pS*E[s2]*Pi[c,s1] (+ swapped)   leaf: pS if c maps to s
// Self-loops (DL, TL-lost) are realised as a resample loop (the geometric series).
// OpenMP parallelises across samples; per-thread accumulators are reduced at the end.
#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <cstdint>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

inline double exp2_(double x) { return std::exp2(x); }
constexpr double NEG_INF = -std::numeric_limits<double>::infinity();

// --- xoshiro256+ PRNG (fast, per-thread) -------------------------------------
struct Rng {
  uint64_t s[4];
  static inline uint64_t splitmix(uint64_t &x) {
    uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
  void seed(uint64_t v) { for (int i = 0; i < 4; i++) s[i] = splitmix(v); }
  static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
  inline uint64_t next() {
    uint64_t r = s[0] + s[3];
    uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
    return r;
  }
  inline double uniform() { return (next() >> 11) * (1.0 / 9007199254740992.0); }  // [0,1)
};

enum Kind { K_D, K_TR, K_TL, K_S, K_DL, K_TLLOST, K_TLMOVE, K_SL, K_LEAF };
struct Act { int kind; int L, R, sa, sb; double lw; };

struct Fam {
  const double *Pi, *Pibar, *E, *Ebar, *lpS, *lpD, *lpO, *T;  // T: [S*S] linear donor-major
  const int64_t *sptr, *sL, *sR; const double *slp;
  const int64_t *leafsp, *c1, *c2;
  std::vector<double> pimax;  // [C] max_s Pi[c,s], for recipient sampling
  int C, S, root;
};

// sample a transfer recipient r ~ T[donor, r] * exp2(Pi[clade, r] - pimax[clade])
inline int sample_recipient(const Fam &f, int donor, int clade, Rng &rng) {
  const double *Trow = f.T + (size_t)donor * f.S;
  const double *Pc = f.Pi + (size_t)clade * f.S;
  double mx = f.pimax[clade];
  double total = 0.0;
  for (int r = 0; r < f.S; r++) total += Trow[r] * exp2_(Pc[r] - mx);
  if (!(total > 0.0) || !std::isfinite(total)) {  // degenerate: argmax
    int best = 0; double bw = -1;
    for (int r = 0; r < f.S; r++) { double w = Trow[r] * exp2_(Pc[r] - mx); if (w > bw) { bw = w; best = r; } }
    return best;
  }
  double u = rng.uniform() * total, acc = 0.0;
  for (int r = 0; r < f.S; r++) { acc += Trow[r] * exp2_(Pc[r] - mx); if (u <= acc) return r; }
  return f.S - 1;
}

inline int sample_origination(const Fam &f, Rng &rng) {
  const double *Pr = f.Pi + (size_t)f.root * f.S;
  double mx = NEG_INF;
  for (int e = 0; e < f.S; e++) { double v = f.lpO[e] + Pr[e]; if (v > mx) mx = v; }
  double total = 0.0;
  for (int e = 0; e < f.S; e++) total += exp2_(f.lpO[e] + Pr[e] - mx);
  double u = rng.uniform() * total, acc = 0.0;
  for (int e = 0; e < f.S; e++) { acc += exp2_(f.lpO[e] + Pr[e] - mx); if (u <= acc) return e; }
  return f.S - 1;
}

// recursive backtrace; `scratch` is reused (saved/restored around recursion).
void backtrace(const Fam &f, int cid, int s, Rng &rng, std::vector<Act> &scratch,
               std::vector<int64_t> &stamp, int64_t sample_id,
               double *presence, double *copies) {
  const int S = f.S;
  if (stamp[s] != sample_id) { stamp[s] = sample_id; presence[s] += 1.0; }  // occupied
  const int s1 = (int)f.c1[s], s2 = (int)f.c2[s];
  const bool has_children = (s1 != S);

  while (true) {                                   // resample loop realises DL/TL-lost self-loops
    size_t base = scratch.size();
    double maxlw = NEG_INF;
    auto push = [&](int kind, double lw, int L, int R, int sa, int sb) {
      scratch.push_back({kind, L, R, sa, sb, lw});
      if (lw > maxlw) maxlw = lw;
    };
    const double Pcs = f.Pi[(size_t)cid * S + s];
    const double Pbcs = f.Pibar[(size_t)cid * S + s];
    for (int64_t k = f.sptr[cid]; k < f.sptr[cid + 1]; k++) {
      int L = (int)f.sL[k], R = (int)f.sR[k]; double lsp = f.slp[k];
      double PiLs = f.Pi[(size_t)L * S + s], PiRs = f.Pi[(size_t)R * S + s];
      push(K_D,  lsp + f.lpD[s] + PiLs + PiRs, L, R, 0, 0);
      push(K_TR, lsp + PiLs + f.Pibar[(size_t)R * S + s], L, R, 0, 0);   // R moves, L stays
      push(K_TL, lsp + PiRs + f.Pibar[(size_t)L * S + s], L, R, 0, 0);   // L moves, R stays
      if (has_children) {
        push(K_S, lsp + f.lpS[s] + f.Pi[(size_t)L * S + s1] + f.Pi[(size_t)R * S + s2], L, R, s1, s2);
        push(K_S, lsp + f.lpS[s] + f.Pi[(size_t)L * S + s2] + f.Pi[(size_t)R * S + s1], L, R, s2, s1);
      }
    }
    push(K_DL,     1.0 + f.lpD[s] + f.E[s] + Pcs, 0, 0, 0, 0);
    push(K_TLLOST, Pcs + f.Ebar[s],               0, 0, 0, 0);
    push(K_TLMOVE, Pbcs + f.E[s],                 0, 0, 0, 0);
    if (has_children) {
      push(K_SL, f.lpS[s] + f.E[s2] + f.Pi[(size_t)cid * S + s1], 0, 0, s1, 0);  // c -> s1
      push(K_SL, f.lpS[s] + f.E[s1] + f.Pi[(size_t)cid * S + s2], 0, 0, s2, 0);  // c -> s2
    }
    if (f.leafsp[cid] == s) push(K_LEAF, f.lpS[s], 0, 0, 0, 0);

    double total = 0.0;
    for (size_t i = base; i < scratch.size(); i++) total += exp2_(scratch[i].lw - maxlw);
    double u = rng.uniform() * total, acc = 0.0;
    size_t pick = scratch.size() - 1;
    for (size_t i = base; i < scratch.size(); i++) { acc += exp2_(scratch[i].lw - maxlw); if (u <= acc) { pick = i; break; } }
    Act a = scratch[pick];
    scratch.resize(base);                          // pop before recursing

    if (a.kind == K_DL || a.kind == K_TLLOST) continue;   // self-loop: resample same cell
    switch (a.kind) {
      case K_LEAF:   copies[s] += 1.0; return;
      case K_S:      copies[s] += 1.0;
                     backtrace(f, a.L, a.sa, rng, scratch, stamp, sample_id, presence, copies);
                     backtrace(f, a.R, a.sb, rng, scratch, stamp, sample_id, presence, copies); return;
      case K_SL:     copies[s] += 1.0;
                     backtrace(f, cid, a.sa, rng, scratch, stamp, sample_id, presence, copies); return;
      case K_D:      backtrace(f, a.L, s, rng, scratch, stamp, sample_id, presence, copies);
                     backtrace(f, a.R, s, rng, scratch, stamp, sample_id, presence, copies); return;
      case K_TR: {   backtrace(f, a.L, s, rng, scratch, stamp, sample_id, presence, copies);
                     int rr = sample_recipient(f, s, a.R, rng);
                     backtrace(f, a.R, rr, rng, scratch, stamp, sample_id, presence, copies); return; }
      case K_TL: {   backtrace(f, a.R, s, rng, scratch, stamp, sample_id, presence, copies);
                     int rr = sample_recipient(f, s, a.L, rng);
                     backtrace(f, a.L, rr, rng, scratch, stamp, sample_id, presence, copies); return; }
      case K_TLMOVE:{int rr = sample_recipient(f, s, cid, rng);
                     backtrace(f, cid, rr, rng, scratch, stamp, sample_id, presence, copies); return; }
      default: return;
    }
  }
}

}  // namespace

// Returns (presence[S], copies[S]) accumulated over n_samples (NOT divided by n).
std::vector<torch::Tensor> sample_accumulate(
    torch::Tensor Pi, torch::Tensor Pibar, torch::Tensor E, torch::Tensor Ebar,
    torch::Tensor lpS, torch::Tensor lpD, torch::Tensor lpO, torch::Tensor T,
    torch::Tensor sptr, torch::Tensor sL, torch::Tensor sR, torch::Tensor slp,
    torch::Tensor leafsp, torch::Tensor c1, torch::Tensor c2,
    int64_t root, int64_t C, int64_t S, int64_t n_samples, int64_t seed, int64_t n_threads) {
  auto cd = [](torch::Tensor t) { return t.contiguous().to(torch::kFloat64); };
  auto ci = [](torch::Tensor t) { return t.contiguous().to(torch::kInt64); };
  Pi = cd(Pi); Pibar = cd(Pibar); E = cd(E); Ebar = cd(Ebar); lpS = cd(lpS); lpD = cd(lpD);
  lpO = cd(lpO); T = cd(T); slp = cd(slp);
  sptr = ci(sptr); sL = ci(sL); sR = ci(sR); leafsp = ci(leafsp); c1 = ci(c1); c2 = ci(c2);

  Fam f;
  f.Pi = Pi.data_ptr<double>(); f.Pibar = Pibar.data_ptr<double>();
  f.E = E.data_ptr<double>(); f.Ebar = Ebar.data_ptr<double>();
  f.lpS = lpS.data_ptr<double>(); f.lpD = lpD.data_ptr<double>(); f.lpO = lpO.data_ptr<double>();
  f.T = T.data_ptr<double>();
  f.sptr = sptr.data_ptr<int64_t>(); f.sL = sL.data_ptr<int64_t>(); f.sR = sR.data_ptr<int64_t>();
  f.slp = slp.data_ptr<double>(); f.leafsp = leafsp.data_ptr<int64_t>();
  f.c1 = c1.data_ptr<int64_t>(); f.c2 = c2.data_ptr<int64_t>();
  f.C = (int)C; f.S = (int)S; f.root = (int)root;
  f.pimax.resize(C);
  for (int c = 0; c < (int)C; c++) {
    double mx = NEG_INF; const double *row = f.Pi + (size_t)c * S;
    for (int s = 0; s < (int)S; s++) if (row[s] > mx) mx = row[s];
    f.pimax[c] = mx;
  }

  auto presence = torch::zeros({S}, torch::kFloat64);
  auto copies = torch::zeros({S}, torch::kFloat64);
  double *pres_g = presence.data_ptr<double>(), *cop_g = copies.data_ptr<double>();

#ifdef _OPENMP
  if (n_threads > 0) omp_set_num_threads((int)n_threads);
#endif
#pragma omp parallel
  {
    int tid = 0;
#ifdef _OPENMP
    tid = omp_get_thread_num();
#endif
    std::vector<double> pres_t(S, 0.0), cop_t(S, 0.0);
    std::vector<int64_t> stamp(S, 0);
    std::vector<Act> scratch; scratch.reserve(256);
    Rng rng; rng.seed((uint64_t)seed * 2654435761ULL + (uint64_t)tid + 1);
#pragma omp for schedule(static)
    for (int64_t smp = 0; smp < n_samples; smp++) {
      int64_t sid = smp + 1;
      int e0 = sample_origination(f, rng);
      backtrace(f, f.root, e0, rng, scratch, stamp, sid, pres_t.data(), cop_t.data());
    }
#pragma omp critical
    {
      for (int s = 0; s < (int)S; s++) { pres_g[s] += pres_t[s]; cop_g[s] += cop_t[s]; }
    }
  }
  return {presence, copies};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sample_accumulate", &sample_accumulate,
        "stochastic-backtrack reconciliation sampler: returns (presence[S], copies[S])");
}
