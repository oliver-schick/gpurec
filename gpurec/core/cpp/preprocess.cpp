#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <deque>
#include <map>
#include <omp.h>
#include <queue>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tree_utils.hpp"
#include "clade_utils.hpp"

namespace py = pybind11;

namespace {

// ============================================================================
// Constants and Utilities
// ============================================================================

constexpr size_t BITS_PER_WORD = 64;

inline size_t bitvec_num_words(int num_leaves) {
  return (static_cast<size_t>(num_leaves) + BITS_PER_WORD - 1) / BITS_PER_WORD;
}

// Default way of parsing names: assume species name is prefix before first '_'
// TODO: replicate ALERax behavior instead of this.
std::string extract_species_name(const std::string &leaf_name) {
  auto pos = leaf_name.find('_');
  if (pos != std::string::npos) {
    return leaf_name.substr(0, pos);
  }
  return leaf_name;
}

// ============================================================================
// Tensor Conversion Helpers
// ============================================================================

torch::Tensor to_long_tensor(const std::vector<int64_t>& vec) {
  return torch::from_blob(
      const_cast<int64_t*>(vec.data()),
      {static_cast<long>(vec.size())},
      torch::TensorOptions().dtype(torch::kInt64))
      .clone();
}

torch::Tensor to_double_tensor(const std::vector<double>& vec) {
  return torch::from_blob(
      const_cast<double*>(vec.data()),
      {static_cast<long>(vec.size())},
      torch::TensorOptions().dtype(torch::kFloat64))
      .clone();
}

torch::Tensor to_double_matrix(const std::vector<double>& mat, int rows, int cols) {
  return torch::from_blob(
      const_cast<double*>(mat.data()),
      {rows, cols},
      torch::TensorOptions().dtype(torch::kFloat64))
      .clone();
}

torch::Tensor to_uint8_tensor(const std::vector<uint8_t>& vec) {
  return torch::from_blob(
      const_cast<uint8_t*>(vec.data()),
      {static_cast<long>(vec.size())},
      torch::TensorOptions().dtype(torch::kUInt8))
      .clone();
}

// ============================================================================
// Clade and CladeSplit Classes
// ============================================================================

/**
 * @brief Represents a clade (subset of leaves in a tree)
 *
 * A clade is uniquely identified by its BitVec representation.
 * Provides hashing and canonical ordering for deduplication.
 */
class Clade {
private:
  BitVec bits_;
  int size_;

public:
  explicit Clade(const BitVec& bits)
      : bits_(bits), size_(bit_count(bits)) {}
  
  explicit Clade(BitVec&& bits)
      : bits_(std::move(bits)), size_(bit_count(bits_)) {}

  const BitVec& bits() const { return bits_; }
  int size() const { return size_; }
  /**
   * @brief Equality comparison based on BitVec content
   */
  bool operator==(const Clade& other) const {
    return bits_ == other.bits_;
  }

  /**
   * @brief Lexicographic ordering for canonical split keys
   */
  bool operator<(const Clade& other) const {
    if (size_ != other.size_) {
      return size_ < other.size_;
    }
    return bitvec_lex_less(bits_, other.bits_);
  }
};

/**
 * @brief Manages a collection of clades with ID assignment
 */
class CladeRegistry {
private:
  std::vector<Clade> clades_;
  std::unordered_map<BitVec, int, BitVecHash, BitVecEqual> bitvec_to_id_;

public:
  CladeRegistry() {
    clades_.reserve(1024);
    bitvec_to_id_.reserve(1024);
  }

  /**
   * @brief Get or create a clade, returning its ID
   */
  int get_or_create(const BitVec& bits) {
    auto it = bitvec_to_id_.find(bits);
    if (it != bitvec_to_id_.end()) {
      return it->second;
    }
    int new_id = static_cast<int>(clades_.size());
    clades_.emplace_back(bits);
    bitvec_to_id_.emplace(clades_.back().bits(), new_id);
    return new_id;
  }

  int get_or_create(BitVec&& bits) {
    auto it = bitvec_to_id_.find(bits);
    if (it != bitvec_to_id_.end()) {
      return it->second;
    }
    int new_id = static_cast<int>(clades_.size());
    clades_.emplace_back(std::move(bits));
    bitvec_to_id_.emplace(clades_.back().bits(), new_id);
    return new_id;
  }

  const Clade& get(int id) const {
    return clades_[id];
  }

  size_t size() const {
    return clades_.size();
  }

  const std::vector<Clade>& all() const {
    return clades_;
  }
};

/**
 * @brief Represents a split (parent clade divided into two child clades)
 */
class CladeSplit {
private:
  int parent_id_;
  int left_id_;
  int right_id_;
  double weight_;

public:
  CladeSplit(int parent, int left, int right, double weight = 1.0)
      : parent_id_(parent), left_id_(left), right_id_(right), weight_(weight) {}

  int parent() const { return parent_id_; }
  int left() const { return left_id_; }
  int right() const { return right_id_; }
  double weight() const { return weight_; }

  void add_weight(double w) { weight_ += w; }

  /**
   * @brief Create canonical key for deduplication
   *
   * Ensures left_id <= right_id for consistent hashing
   */
  std::tuple<int, int, int> canonical_key() const {
    int l = left_id_;
    int r = right_id_;
    if (l > r) {
      std::swap(l, r);
    }
    return std::make_tuple(parent_id_, l, r);
  }

  /**
   * @brief Create canonical PairKey for root splits
   *
   * Orders by clade size, then lexicographically
   */
  static PairKey canonical_root_key(
      int a_id, const Clade& a,
      int b_id, const Clade& b) {
    if (a.size() < b.size()) {
      return PairKey{a_id, b_id};
    }
    if (a.size() > b.size()) {
      return PairKey{b_id, a_id};
    }
    if (a < b) {
      return PairKey{a_id, b_id};
    }
    if (b < a) {
      return PairKey{b_id, a_id};
    }
    return PairKey{std::min(a_id, b_id), std::max(a_id, b_id)};
  }
};

/**
 * @brief Result structure for clade and split computation
 * 
 * Contains the set of clades, splits, and root clade ID.
 */
struct CladeData {
  CladeRegistry clades;
  std::vector<CladeSplit> splits;
  int root_clade_id;
};


struct CCPArrays {
  std::vector<int64_t> split_parents_sorted;
  std::vector<int64_t> split_lefts_sorted;
  std::vector<int64_t> split_rights_sorted;
  std::vector<int64_t> split_leftrights_sorted;
  std::vector<double> log_split_probs_sorted;
  std::vector<int64_t> parents_sorted;
  std::vector<int64_t> seg_counts;
  std::vector<int64_t> ptr;
  std::vector<int64_t> ptr_ge2;
  std::vector<int64_t> split_counts;
  std::vector<int64_t> split_order;
  int num_segs_ge2;
  int num_segs_eq1;
  int num_segs_eq0;
  int stop_reduce_ptr_idx;
  int end_rows_ge2;

  // Clade inclusion DAG: edge (inclusion_children[i], inclusion_parents[i])
  // means inclusion_children[i] ⊆ inclusion_parents[i]
  std::vector<int64_t> inclusion_children;
  std::vector<int64_t> inclusion_parents;
  int64_t ubiquitous_clade_id;  // Root of the inclusion DAG (contains all leaves)
};

// ============================================================================
// Core Algorithms
// ============================================================================

/**
 * @brief Compute all clades and splits from a gene tree
 *
 * For a tree with n leaves, generates exactly 4n-5 clades:
 * - n singleton clades (one per leaf)
 * - 3(n-2) clades from internal nodes (3 per node of degree 3)
 * - 1 root clade (all leaves)
 */
CladeData compute_clades_and_splits(
    TreeNode *gene_root,
    const std::vector<std::string> &leaf_names,
    const std::unordered_map<std::string, int> &leaf_to_index) {

  CladeData result;
  const int num_leaves = static_cast<int>(leaf_names.size());
  const size_t num_words = bitvec_num_words(num_leaves);

  std::vector<TreeNode *> postorder_nodes;
  collect_nodes_postorder(gene_root, postorder_nodes);
  const size_t num_nodes = postorder_nodes.size();

  std::unordered_map<TreeNode *, size_t> node_index;
  node_index.reserve(num_nodes);
  for (size_t i = 0; i < num_nodes; ++i) {
    node_index[postorder_nodes[i]] = i;
  }

  std::vector<BitVec> node_clades(num_nodes, BitVec(num_words, 0ULL));
  std::vector<int> node_clade_ids(num_nodes, -1);
  std::vector<int> node_above_ids(num_nodes, -1);

  // First pass: build clades for each node
  // Will create the n leaf clades, n-2 internal clade nodes (corresponding to this fixed root choice), and the ubiquitous clade
  // We will still have to construct clades corresponding to other rootings.
  for (TreeNode *node : postorder_nodes) {
    size_t idx = node_index[node];
    BitVec bits(num_words, 0ULL);
    if (node->children.empty()) {
      int leaf_idx = leaf_to_index.at(node->name);
      set_bit(bits, leaf_idx);
    } else {
      const BitVec &left = node_clades[node_index[node->children[0]]];
      const BitVec &right = node_clades[node_index[node->children[1]]];
      for (size_t w = 0; w < num_words; ++w) {
        bits[w] = left[w] | right[w];
      }
    }
    node_clades[idx] = bits;
    node_clade_ids[idx] = result.clades.get_or_create(std::move(bits));
  }

  const BitVec &root_bits = node_clades[node_index[gene_root]];
  result.root_clade_id = node_clade_ids[node_index[gene_root]];

  result.splits.reserve(num_nodes * 3);
  std::unordered_set<PairKey, PairKeyHash, PairKeyEqual> root_split_keys;
  root_split_keys.reserve(num_nodes * 2);

  // Second pass: compute "above" clades and root splits
  for (TreeNode *node : postorder_nodes) {
    if (node == gene_root) {
      continue;
    }
    size_t idx = node_index[node];
    const BitVec &below_bits = node_clades[idx];
    BitVec above_bits = bit_difference(root_bits, below_bits);
    if (is_empty(above_bits)) {
      continue;
    }

    int below_id = node_clade_ids[idx];
    int above_id = result.clades.get_or_create(std::move(above_bits));
    node_above_ids[idx] = above_id;

    const Clade& below_clade = result.clades.get(below_id);
    const Clade& above_clade = result.clades.get(above_id);

    PairKey key = CladeSplit::canonical_root_key(below_id, below_clade, above_id, above_clade);
    if (root_split_keys.insert(key).second) {
      result.splits.emplace_back(result.root_clade_id, below_id, above_id, 1.0);
    }
  }

  // Third pass: create internal splits
  for (TreeNode *node : postorder_nodes) {
    if (node == gene_root || node->children.size() != 2) {
      continue;
    }
    size_t idx = node_index[node];
    int parent_id = node_clade_ids[idx];
    TreeNode *left_node = node->children[0];
    TreeNode *right_node = node->children[1];
    size_t left_idx = node_index[left_node];
    size_t right_idx = node_index[right_node];
    int left_id = node_clade_ids[left_idx];
    int right_id = node_clade_ids[right_idx];
    result.splits.emplace_back(parent_id, left_id, right_id, 1.0);

    int above_id = node_above_ids[idx];
    if (above_id >= 0) {
      int left_plus_id = node_above_ids[right_idx];
      int right_plus_id = node_above_ids[left_idx];
      result.splits.emplace_back(left_plus_id, left_id, above_id, 1.0);
      result.splits.emplace_back(right_plus_id, right_id, above_id, 1.0);
    }
  }

  return result;
}

/**
 * @brief Convert CladeData to sorted CCP arrays for efficient computation
 */
CCPArrays build_ccp_arrays(
    const CladeData& clade_data,
    bool include_inclusion = true) {
  CCPArrays result;
  const size_t C = clade_data.clades.size();
  const size_t N_splits = clade_data.splits.size();
  const std::vector<CladeSplit>& splits = clade_data.splits;

  std::vector<int64_t> split_parents(N_splits);
  std::vector<int64_t> split_lefts(N_splits);
  std::vector<int64_t> split_rights(N_splits);
  std::vector<double> split_weights(N_splits);
  result.split_counts.resize(C, 0);
  // Creating the vectors of split components
  // As well as the number of ways each clade is split (split_counts)
  for (size_t i = 0; i < N_splits; ++i) {
    split_parents[i] = splits[i].parent();
    split_lefts[i] = splits[i].left();
    split_rights[i] = splits[i].right();
    split_weights[i] = splits[i].weight();
    result.split_counts[splits[i].parent()] += 1;
  }

  std::vector<double> sum_weights(C, 0.0);
  for (size_t i = 0; i < N_splits; ++i) {
    sum_weights[split_parents[i]] += split_weights[i];
  }

  std::vector<double> split_probs(N_splits, 0.0);
  std::vector<double> log_split_probs(N_splits, 0.0);
  for (size_t i = 0; i < N_splits; ++i) {
    double denom = sum_weights[split_parents[i]];
    split_probs[i] = denom > 0.0 ? split_weights[i] / denom : 0.0;
    log_split_probs[i] = split_probs[i] > 0.0
                              ? std::log(split_probs[i])
                              : -std::numeric_limits<double>::infinity();
  }

  result.split_order.resize(N_splits);
  std::iota(result.split_order.begin(), result.split_order.end(), 0);

  result.parents_sorted.resize(C);
  std::iota(result.parents_sorted.begin(), result.parents_sorted.end(), 0);
  std::stable_sort(result.parents_sorted.begin(), result.parents_sorted.end(),
                   [&](int64_t a, int64_t b) {
                     if (result.split_counts[a] != result.split_counts[b]) {
                       return result.split_counts[a] > result.split_counts[b];
                     }
                     return a < b;
                   });

  std::vector<int64_t> parent_rank(C);
  for (size_t i = 0; i < C; ++i) {
    parent_rank[result.parents_sorted[i]] = static_cast<int64_t>(i);
  }

  std::stable_sort(result.split_order.begin(), result.split_order.end(),
                   [&](int64_t lhs, int64_t rhs) {
                     int64_t ra = parent_rank[split_parents[lhs]];
                     int64_t rb = parent_rank[split_parents[rhs]];
                     if (ra != rb) {
                       return ra < rb;
                     }
                     return lhs < rhs;
                   });

  result.split_parents_sorted.resize(N_splits);
  result.split_lefts_sorted.resize(N_splits);
  result.split_rights_sorted.resize(N_splits);
  result.log_split_probs_sorted.resize(N_splits);
  for (size_t i = 0; i < N_splits; ++i) {
    size_t idx = result.split_order[i];
    result.split_parents_sorted[i] = split_parents[idx];
    result.split_lefts_sorted[i] = split_lefts[idx];
    result.split_rights_sorted[i] = split_rights[idx];
    result.log_split_probs_sorted[i] = log_split_probs[idx];
  }

  result.split_leftrights_sorted.resize(2 * N_splits);
  for (size_t i = 0; i < N_splits; ++i) {
    result.split_leftrights_sorted[i] = result.split_lefts_sorted[i];
    result.split_leftrights_sorted[i + N_splits] = result.split_rights_sorted[i];
  }

  result.seg_counts.resize(C);
  for (size_t i = 0; i < C; ++i) {
    result.seg_counts[i] = result.split_counts[result.parents_sorted[i]];
  }

  result.ptr.resize(C + 1);
  result.ptr[0] = 0;
  for (size_t i = 0; i < C; ++i) {
    result.ptr[i + 1] = result.ptr[i] + result.seg_counts[i];
  }

  result.num_segs_ge2 = 0;
  result.num_segs_eq1 = 0;
  result.num_segs_eq0 = 0;
  for (int64_t count : result.seg_counts) {
    if (count >= 2) {
      ++result.num_segs_ge2;
    } else if (count == 1) {
      ++result.num_segs_eq1;
    } else {
      ++result.num_segs_eq0;
    }
  }

  result.stop_reduce_ptr_idx = result.num_segs_ge2;
  result.end_rows_ge2 = static_cast<int>(result.ptr[result.stop_reduce_ptr_idx]);
  result.ptr_ge2.assign(result.ptr.begin(), result.ptr.begin() + result.stop_reduce_ptr_idx + 1);

  result.ubiquitous_clade_id = -1;
  if (include_inclusion) {
    // Compute clade inclusion DAG
    // Find ubiquitous clade (contains all leaves, has maximum size)
    int max_size = 0;
    for (size_t i = 0; i < C; ++i) {
      const Clade& clade = clade_data.clades.get(i);
      if (clade.size() > max_size) {
        max_size = clade.size();
        result.ubiquitous_clade_id = static_cast<int64_t>(i);
      }
    }

    // Compute all inclusion relationships (child ⊆ parent)
    // For clades A and B: A ⊆ B iff (A.bits & B.bits) == A.bits
    for (size_t i = 0; i < C; ++i) {
      const Clade& clade_i = clade_data.clades.get(i);
      const BitVec& bits_i = clade_i.bits();

      for (size_t j = 0; j < C; ++j) {
        if (i == j) continue;  // Skip self-loops

        const Clade& clade_j = clade_data.clades.get(j);
        const BitVec& bits_j = clade_j.bits();

        // Check if clade_i ⊆ clade_j (all bits in i are also in j)
        bool is_subset = true;
        for (size_t w = 0; w < bits_i.size(); ++w) {
          if ((bits_i[w] & bits_j[w]) != bits_i[w]) {
            is_subset = false;
            break;
          }
        }

        if (is_subset) {
          result.inclusion_children.push_back(static_cast<int64_t>(i));
          result.inclusion_parents.push_back(static_cast<int64_t>(j));
        }
      }
    }
  }

  return result;
}

/**
 * @brief Compute ancestors and recipients matrices for a species tree
 */
std::pair<std::vector<double>, std::vector<double>> compute_ancestors_and_recipients(
    TreeNode* species_root,
    int num_species) {

  std::vector<TreeNode*> species_postorder;
  collect_nodes_postorder(species_root, species_postorder);

  std::unordered_map<TreeNode*, int> species_node_to_index;
  species_node_to_index.reserve(species_postorder.size());
  for (size_t i = 0; i < species_postorder.size(); ++i) {
    species_node_to_index[species_postorder[i]] = static_cast<int>(i);
  }

  std::vector<double> ancestors(num_species * num_species, 0.0);
  for (TreeNode* node : species_postorder) {
    int idx = species_node_to_index[node];
    TreeNode* cur = node;
    while (cur) {
      int anc_idx = species_node_to_index[cur];
      ancestors[idx * num_species + anc_idx] = 1.0;
      cur = cur->parent;
    }
  }

  std::vector<double> recipients(num_species * num_species, 0.0);
  for (int i = 0; i < num_species; ++i) {
    double total = 0.0;
    for (int j = 0; j < num_species; ++j) {
      if (ancestors[i * num_species + j] == 0.0) {
        recipients[i * num_species + j] = 1.0;
        total += 1.0;
      }
    }
    if (total > 0.0) {
      for (int j = 0; j < num_species; ++j) {
        recipients[i * num_species + j] /= total;
      }
    }
  }

  return {std::move(ancestors), std::move(recipients)};
}

/**
 * @brief Amalgamate clades and splits from multiple gene trees
 *
 * Takes the union of all clades across gene trees. For each split,
 * the weight is the number of trees containing that split.
 */
CladeData amalgamate_clades_and_splits(
    const std::vector<std::string> &gene_paths,
    std::vector<std::string> &leaf_names,
    std::unordered_map<std::string, int> &leaf_to_index) {

  if (gene_paths.empty()) {
    throw std::runtime_error("No gene tree paths provided");
  }

  // Collect all unique leaf names from all trees
  std::unordered_set<std::string> all_leaves_set;
  std::vector<std::unique_ptr<TreeNode>> gene_trees;
  gene_trees.reserve(gene_paths.size());

  for (const std::string &path : gene_paths) {
    std::unique_ptr<TreeNode> tree = parse_newick_file(path);
    std::vector<std::string> tree_leaves;
    std::unordered_map<std::string, int> tree_leaf_map;
    collect_leaf_names(tree.get(), tree_leaves, tree_leaf_map);
    for (const std::string &name : tree_leaves) {
      all_leaves_set.insert(name);
    }
    gene_trees.push_back(std::move(tree));
  }

  // Create unified leaf ordering
  leaf_names.clear();
  leaf_names.reserve(all_leaves_set.size());
  for (const std::string &name : all_leaves_set) {
    leaf_names.push_back(name);
  }
  std::sort(leaf_names.begin(), leaf_names.end());

  leaf_to_index.clear();
  leaf_to_index.reserve(leaf_names.size());
  for (size_t i = 0; i < leaf_names.size(); ++i) {
    leaf_to_index[leaf_names[i]] = static_cast<int>(i);
  }

  const int num_leaves = static_cast<int>(leaf_names.size());
  if (num_leaves == 0) {
    throw std::runtime_error("No leaves found in gene trees");
  }

  CladeData result;
  const size_t num_words = bitvec_num_words(num_leaves);

  // Map from canonical split key to split index for weight accumulation
  std::map<std::tuple<int,int,int>, size_t> split_index_map;

  // Create root clade (all leaves)
  BitVec root_bits(num_words, 0ULL);
  for (int i = 0; i < num_leaves; ++i) {
    set_bit(root_bits, i);
  }
  result.root_clade_id = result.clades.get_or_create(std::move(root_bits));

  // Process each gene tree
  for (size_t tree_idx = 0; tree_idx < gene_trees.size(); ++tree_idx) {
    TreeNode *gene_root = gene_trees[tree_idx].get();

    std::vector<TreeNode *> postorder_nodes;
    collect_nodes_postorder(gene_root, postorder_nodes);
    const size_t num_nodes = postorder_nodes.size();

    std::unordered_map<TreeNode *, size_t> node_index;
    node_index.reserve(num_nodes);
    for (size_t i = 0; i < num_nodes; ++i) {
      node_index[postorder_nodes[i]] = i;
    }

    std::vector<BitVec> node_clades(num_nodes, BitVec(num_words, 0ULL));
    std::vector<int> node_clade_ids(num_nodes, -1);
    std::vector<int> node_above_ids(num_nodes, -1);

    // Build clades for each node using global leaf indexing
    for (TreeNode *node : postorder_nodes) {
      size_t idx = node_index[node];
      BitVec bits(num_words, 0ULL);
      if (node->children.empty()) {
        int global_idx = leaf_to_index.at(node->name);
        set_bit(bits, global_idx);
      } else {
        const BitVec &left = node_clades[node_index[node->children[0]]];
        const BitVec &right = node_clades[node_index[node->children[1]]];
        for (size_t w = 0; w < num_words; ++w) {
          bits[w] = left[w] | right[w];
        }
      }
      node_clades[idx] = bits;
      node_clade_ids[idx] = result.clades.get_or_create(std::move(bits));
    }

    const BitVec &tree_root_bits = node_clades[node_index[gene_root]];
    int tree_root_id = node_clade_ids[node_index[gene_root]];

    std::unordered_set<PairKey, PairKeyHash, PairKeyEqual> tree_root_split_keys;

    // Compute "above" clades and root splits
    for (TreeNode *node : postorder_nodes) {
      if (node == gene_root) {
        continue;
      }
      size_t idx = node_index[node];
      const BitVec &below_bits = node_clades[idx];
      BitVec above_bits = bit_difference(tree_root_bits, below_bits);
      if (is_empty(above_bits)) {
        continue;
      }

      int below_id = node_clade_ids[idx];
      int above_id = result.clades.get_or_create(std::move(above_bits));
      node_above_ids[idx] = above_id;

      const Clade& below_clade = result.clades.get(below_id);
      const Clade& above_clade = result.clades.get(above_id);
      PairKey pkey = CladeSplit::canonical_root_key(below_id, below_clade, above_id, above_clade);

      if (tree_root_split_keys.insert(pkey).second) {
        CladeSplit split(tree_root_id, below_id, above_id, 1.0);
        auto key = split.canonical_key();
        auto it = split_index_map.find(key);
        if (it != split_index_map.end()) {
          result.splits[it->second].add_weight(1.0);
        } else {
          size_t new_idx = result.splits.size();
          result.splits.push_back(std::move(split));
          split_index_map[key] = new_idx;
        }
      }
    }

    // Create internal splits
    for (TreeNode *node : postorder_nodes) {
      if (node == gene_root || node->children.size() != 2) {
        continue;
      }
      size_t idx = node_index[node];
      int parent_id = node_clade_ids[idx];
      TreeNode *left_node = node->children[0];
      TreeNode *right_node = node->children[1];
      size_t left_idx = node_index[left_node];
      size_t right_idx = node_index[right_node];
      int left_id = node_clade_ids[left_idx];
      int right_id = node_clade_ids[right_idx];

      CladeSplit split(parent_id, left_id, right_id, 1.0);
      auto key = split.canonical_key();
      auto it = split_index_map.find(key);
      if (it != split_index_map.end()) {
        result.splits[it->second].add_weight(1.0);
      } else {
        size_t new_idx = result.splits.size();
        result.splits.push_back(std::move(split));
        split_index_map[key] = new_idx;
      }

      int above_id = node_above_ids[idx];
      if (above_id >= 0) {
        int left_plus_id = node_above_ids[right_idx];
        int right_plus_id = node_above_ids[left_idx];

        CladeSplit split1(left_plus_id, left_id, above_id, 1.0);
        auto key1 = split1.canonical_key();
        auto it1 = split_index_map.find(key1);
        if (it1 != split_index_map.end()) {
          result.splits[it1->second].add_weight(1.0);
        } else {
          size_t new_idx = result.splits.size();
          result.splits.push_back(std::move(split1));
          split_index_map[key1] = new_idx;
        }

        CladeSplit split2(right_plus_id, right_id, above_id, 1.0);
        auto key2 = split2.canonical_key();
        auto it2 = split_index_map.find(key2);
        if (it2 != split_index_map.end()) {
          result.splits[it2->second].add_weight(1.0);
        } else {
          size_t new_idx = result.splits.size();
          result.splits.push_back(std::move(split2));
          split_index_map[key2] = new_idx;
        }
      }
    }
  }

  return result;
}

// Union-Find for outgoing packet identification
class UnionFind {
  std::vector<int> par, rnk;
public:
  explicit UnionFind(int n) : par(n), rnk(n, 0) {
    std::iota(par.begin(), par.end(), 0);
  }
  int find(int x) {
    while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; }
    return x;
  }
  void unite(int x, int y) {
    x = find(x); y = find(y);
    if (x == y) return;
    if (rnk[x] < rnk[y]) std::swap(x, y);
    par[y] = x;
    if (rnk[x] == rnk[y]) rnk[x]++;
  }
};

/**
 * @brief Lightweight CCP arrays for wave scheduling only.
 *
 * Skips the O(C²) inclusion DAG, sorted ordering, probabilities, etc.
 * Only produces split_parents, split_lefts, split_rights, split_counts.
 */
struct CCPLight {
  std::vector<int64_t> split_parents;
  std::vector<int64_t> split_lefts;
  std::vector<int64_t> split_rights;
  std::vector<int64_t> split_counts;
  size_t C;  // number of clades
};

CCPLight build_ccp_light(const CladeData& clade_data) {
  CCPLight result;
  result.C = clade_data.clades.size();
  const size_t N = clade_data.splits.size();
  const auto &splits = clade_data.splits;

  result.split_parents.resize(N);
  result.split_lefts.resize(N);
  result.split_rights.resize(N);
  result.split_counts.assign(result.C, 0);

  for (size_t i = 0; i < N; ++i) {
    result.split_parents[i] = splits[i].parent();
    result.split_lefts[i]   = splits[i].left();
    result.split_rights[i]  = splits[i].right();
    result.split_counts[splits[i].parent()]++;
  }
  return result;
}

// ============================================================================
// Scheduling-ready data: adjacency, λ, packets — built from CCPLight
// ============================================================================

struct SchedData {
  size_t C;
  std::vector<std::vector<int64_t>> children;
  std::vector<std::vector<int64_t>> parents_of;
  std::vector<int32_t> remaining;
  std::vector<int64_t> split_counts;
  std::vector<int32_t> lambda;       // longest path to root
  int64_t root_id;
  std::vector<int> out_packet;
  std::vector<int> in_packet;
};

SchedData build_sched_data(const CladeData &clade_data) {
  CCPLight ccp = build_ccp_light(clade_data);
  SchedData sd;
  sd.C = ccp.C;
  const size_t C = ccp.C;
  const size_t N = ccp.split_parents.size();
  sd.split_counts = std::move(ccp.split_counts);

  // Build adjacency
  sd.children.resize(C);
  sd.parents_of.resize(C);
  sd.remaining.assign(C, 0);
  {
    std::vector<std::unordered_set<int64_t>> child_sets(C);
    for (size_t i = 0; i < N; ++i) {
      int64_t p = ccp.split_parents[i];
      int64_t l = ccp.split_lefts[i];
      int64_t r = ccp.split_rights[i];
      if (child_sets[p].insert(l).second) {
        sd.children[p].push_back(l);
        sd.parents_of[l].push_back(p);
        sd.remaining[p]++;
      }
      if (l != r && child_sets[p].insert(r).second) {
        sd.children[p].push_back(r);
        sd.parents_of[r].push_back(p);
        sd.remaining[p]++;
      }
    }
  }

  // Find root
  sd.root_id = -1;
  for (size_t c = 0; c < C; ++c) {
    if (sd.parents_of[c].empty()) {
      sd.root_id = static_cast<int64_t>(c);
      break;
    }
  }

  // BFS levels (ρ)
  std::vector<int32_t> bfs_level(C, 0);
  std::vector<int32_t> remaining_bfs = sd.remaining;
  std::queue<int64_t> q;
  for (size_t c = 0; c < C; ++c) {
    if (remaining_bfs[c] == 0) q.push(static_cast<int64_t>(c));
  }
  int32_t max_wave = 0;
  while (!q.empty()) {
    int64_t c = q.front(); q.pop();
    for (int64_t p : sd.parents_of[c]) {
      if (bfs_level[p] <= bfs_level[c]) {
        bfs_level[p] = bfs_level[c] + 1;
        if (bfs_level[p] > max_wave) max_wave = bfs_level[p];
      }
      if (--remaining_bfs[p] == 0) q.push(p);
    }
  }
  int32_t n_waves = max_wave + 1;

  // Compute λ: longest path from clade to root (reverse topological order)
  std::vector<std::vector<int64_t>> clades_at_level(n_waves);
  for (size_t c = 0; c < C; ++c)
    clades_at_level[bfs_level[c]].push_back(static_cast<int64_t>(c));

  sd.lambda.assign(C, 0);
  for (int lev = n_waves - 1; lev >= 0; --lev) {
    for (int64_t c : clades_at_level[lev]) {
      for (int64_t ch : sd.children[c]) {
        sd.lambda[ch] = std::max(sd.lambda[ch], sd.lambda[c] + 1);
      }
    }
  }

  // Outgoing packets (union-find over co-children, skip root splits)
  UnionFind uf(static_cast<int>(C));
  for (size_t i = 0; i < N; ++i) {
    int64_t p = ccp.split_parents[i];
    if (p == sd.root_id) continue;
    int64_t l = ccp.split_lefts[i];
    int64_t r = ccp.split_rights[i];
    if (l != r) uf.unite(static_cast<int>(l), static_cast<int>(r));
  }
  sd.out_packet.resize(C);
  for (size_t c = 0; c < C; ++c) sd.out_packet[c] = uf.find(static_cast<int>(c));

  // Incoming packets
  sd.in_packet.assign(C, -1);
  for (size_t c = 0; c < C; ++c) {
    if (!sd.children[c].empty() && static_cast<int64_t>(c) != sd.root_id) {
      sd.in_packet[c] = sd.out_packet[sd.children[c][0]];
    }
  }

  return sd;
}

// ============================================================================
// Phased wave scheduling — returns actual clade assignments
// ============================================================================

/**
 * @brief Three-phase wave scheduling from SchedData.
 *
 * Phase 1: leaf clades (split_count == 0), sorted by outgoing packet.
 * Phase 2: internal non-root clades (split_count >= 1), λ-priority greedy.
 * Phase 3: root Ω.
 *
 * Returns vector of waves, each wave is a vector of clade IDs.
 * Also returns a vector of phase labels (1, 2, or 3) per wave.
 */
std::pair<std::vector<std::vector<int64_t>>, std::vector<int>>
compute_phased_waves_impl(SchedData &sd, int max_wave_size) {
  const size_t C = sd.C;
  auto &remaining    = sd.remaining;
  auto &lambda       = sd.lambda;
  auto &out_packet   = sd.out_packet;
  auto &in_packet    = sd.in_packet;
  auto &parents_of   = sd.parents_of;
  auto &children_adj = sd.children;
  int64_t root_id    = sd.root_id;

  std::vector<std::vector<int64_t>> waves;
  std::vector<int> phases;

  // ==== Phase 1: leaf clades ====
  std::vector<int64_t> leaf_clades;
  for (size_t c = 0; c < C; ++c) {
    if (static_cast<int64_t>(c) == root_id) continue;
    if (sd.split_counts[c] == 0) {
      leaf_clades.push_back(static_cast<int64_t>(c));
    }
  }
  std::sort(leaf_clades.begin(), leaf_clades.end(),
    [&](int64_t a, int64_t b) {
      if (out_packet[a] != out_packet[b]) return out_packet[a] < out_packet[b];
      return a < b;
    });

  for (size_t start = 0; start < leaf_clades.size();
       start += static_cast<size_t>(max_wave_size)) {
    size_t end = std::min(start + static_cast<size_t>(max_wave_size),
                          leaf_clades.size());
    std::vector<int64_t> batch(leaf_clades.begin() + start,
                                leaf_clades.begin() + end);
    waves.push_back(batch);
    phases.push_back(1);

    for (int64_t c : batch) {
      for (int64_t p : parents_of[c]) {
        --remaining[p];
      }
    }
  }

  // ==== Phase 2: internal non-root clades ====
  int max_lambda = 0;
  for (size_t c = 0; c < C; ++c)
    if (static_cast<int>(lambda[c]) > max_lambda) max_lambda = lambda[c];

  std::vector<std::vector<int64_t>> ready_buckets(max_lambda + 1);
  for (size_t c = 0; c < C; ++c) {
    if (static_cast<int64_t>(c) == root_id) continue;
    if (sd.split_counts[c] == 0) continue;
    if (remaining[c] == 0) {
      ready_buckets[lambda[c]].push_back(static_cast<int64_t>(c));
    }
  }

  int top_bucket = max_lambda;
  while (top_bucket >= 0 && ready_buckets[top_bucket].empty()) --top_bucket;

  while (top_bucket >= 0) {
    std::vector<int64_t> batch;
    batch.reserve(max_wave_size);

    std::unordered_map<int, int32_t> o_count;
    std::unordered_map<int, int32_t> i_count;

    while (static_cast<int>(batch.size()) < max_wave_size && top_bucket >= 0) {
      auto &bucket = ready_buckets[top_bucket];
      if (bucket.empty()) {
        --top_bucket;
        while (top_bucket >= 0 && ready_buckets[top_bucket].empty())
          --top_bucket;
        if (top_bucket < 0) break;
        continue;
      }

      // Find best in bucket: maximize (o_count[out_packet], i_count[in_packet])
      int best_idx = 0;
      int32_t best_o = -1, best_i = -1;
      for (size_t j = 0; j < bucket.size(); ++j) {
        int64_t c = bucket[j];
        int32_t oc = 0, ic = 0;
        auto oit = o_count.find(out_packet[c]);
        if (oit != o_count.end()) oc = oit->second;
        if (in_packet[c] >= 0) {
          auto iit = i_count.find(in_packet[c]);
          if (iit != i_count.end()) ic = iit->second;
        }
        if (oc > best_o || (oc == best_o && ic > best_i)) {
          best_o = oc; best_i = ic; best_idx = static_cast<int>(j);
        }
      }

      int64_t selected = bucket[best_idx];
      bucket[best_idx] = bucket.back();
      bucket.pop_back();

      batch.push_back(selected);
      o_count[out_packet[selected]]++;
      if (in_packet[selected] >= 0)
        i_count[in_packet[selected]]++;

      if (bucket.empty()) {
        while (top_bucket >= 0 && ready_buckets[top_bucket].empty())
          --top_bucket;
      }
    }

    waves.push_back(batch);
    phases.push_back(2);

    for (int64_t c : batch) {
      for (int64_t p : parents_of[c]) {
        if (p == root_id) { --remaining[p]; continue; }
        if (--remaining[p] == 0) {
          ready_buckets[lambda[p]].push_back(p);
          if (lambda[p] > top_bucket) top_bucket = lambda[p];
        }
      }
    }
  }

  // ==== Phase 3: root ====
  if (root_id >= 0) {
    waves.push_back({root_id});
    phases.push_back(3);
  }

  return {waves, phases};
}


/**
 * @brief Python-facing function: parse gene trees and return phased wave assignments.
 *
 * @param gene_tree_paths  Vector of gene tree file paths (for one family).
 * @param max_wave_size    Maximum clades per wave.
 * @return Dict with:
 *   "waves"  — list of lists of clade IDs
 *   "phases" — list of phase labels (1, 2, 3)
 */
py::dict compute_phased_waves(
    const std::vector<std::string> &gene_tree_paths,
    int max_wave_size) {

  std::vector<std::string> leaf_names;
  std::unordered_map<std::string, int> leaf_to_index;
  CladeData clade_data = amalgamate_clades_and_splits(
      gene_tree_paths, leaf_names, leaf_to_index);
  SchedData sd = build_sched_data(clade_data);

  auto [waves, phases] = compute_phased_waves_impl(sd, max_wave_size);

  py::dict result;
  // Convert waves to py::list of py::list
  py::list py_waves;
  for (const auto &w : waves) {
    py::list pw;
    for (int64_t c : w) pw.append(c);
    py_waves.append(pw);
  }
  result["waves"] = py_waves;
  result["phases"] = phases;
  result["root_id"] = sd.root_id;
  return result;
}


// ============================================================================
// Clade Wave Scheduling (legacy BFS)
// ============================================================================

/**
 * @brief Assign clades to waves via BFS topological sort (Kahn's algorithm).
 *
 * Returns (wave_assignment, n_waves) where wave_assignment[c] is the wave
 * index for clade c.  Clades with no splits are wave 0; the root is last.
 *
 * Complexity: O(C + N_splits).
 */
std::pair<std::vector<int32_t>, int32_t>
compute_clade_waves(const CCPArrays &ccp, size_t C) {
  const size_t N = ccp.split_parents_sorted.size();

  // Build: children_sets[p] — distinct child IDs of clade p
  //        depended_by[c]   — parents that have c as a child
  std::vector<std::unordered_set<int64_t>> children_sets(C);
  std::vector<std::vector<int64_t>> depended_by(C);

  for (size_t i = 0; i < N; ++i) {
    int64_t p = ccp.split_parents_sorted[i];
    int64_t l = ccp.split_lefts_sorted[i];
    int64_t r = ccp.split_rights_sorted[i];

    if (children_sets[p].insert(l).second) {
      depended_by[l].push_back(p);
    }
    if (l != r && children_sets[p].insert(r).second) {
      depended_by[r].push_back(p);
    }
  }

  // remaining[p] = number of distinct children not yet processed
  std::vector<int32_t> remaining(C);
  for (size_t c = 0; c < C; ++c) {
    remaining[c] = static_cast<int32_t>(children_sets[c].size());
  }
  children_sets.clear();  // free memory

  std::vector<int32_t> level(C, 0);
  std::queue<int64_t> q;
  for (size_t c = 0; c < C; ++c) {
    if (remaining[c] == 0) {
      q.push(static_cast<int64_t>(c));
    }
  }

  int32_t max_wave = 0;
  while (!q.empty()) {
    int64_t c = q.front(); q.pop();
    for (int64_t p : depended_by[c]) {
      if (level[p] <= level[c]) {
        level[p] = level[c] + 1;
        if (level[p] > max_wave) max_wave = level[p];
      }
      if (--remaining[p] == 0) {
        q.push(p);
      }
    }
  }

  return {level, max_wave + 1};
}

/**
 * @brief Process multiple gene families, each with their own gene tree samples
 *
 * @param species_path Path to the species tree
 * @param families Map from family name to vector of gene tree paths for that family
 * @param include_details Whether to compute and return full debug/detail fields.
 *                        Defaults to false because likelihood construction only
 *                        needs the light CCP/scheduler payload.
 * @return Dictionary with shared species data and per-family CCPs
 */
py::dict preprocess_multiple_families(
    const std::string &species_path,
    const std::map<std::string, std::vector<std::string>> &families,
    bool include_details = false) {

  // Parse species tree once (shared across all families)
  std::unique_ptr<TreeNode> species_root = parse_newick_file(species_path);

  SpeciesData species_data;
  std::vector<TreeNode *> species_order;
  enumerate_species(species_root.get(), species_order, species_data);
  auto species_name_to_index = build_species_name_map(species_data);

  auto [ancestors, recipients] = compute_ancestors_and_recipients(species_root.get(), species_data.S);

  std::vector<int64_t> s_P_indexes;
  std::vector<int64_t> s_C1_indexes;
  std::vector<int64_t> s_C2_indexes;
  for (int i = 0; i < species_data.S; ++i) {
    if (species_data.children[i].size() == 2) {
      int left = species_data.children[i][0];
      int right = species_data.children[i][1];
      s_P_indexes.push_back(i);
      s_C1_indexes.push_back(left);
      s_C2_indexes.push_back(right);
    } else if (!species_data.children[i].empty()) {
      throw std::runtime_error("Species tree must be strictly binary");
    }
  }

  std::vector<int64_t> s_P_indexes_ext = s_P_indexes;
  for (int64_t idx : s_P_indexes) {
    s_P_indexes_ext.push_back(idx + species_data.S);
  }

  std::vector<int64_t> s_C12_indexes = s_C1_indexes;
  s_C12_indexes.insert(s_C12_indexes.end(), s_C2_indexes.begin(), s_C2_indexes.end());

  py::dict species_dict;
  species_dict["S"] = species_data.S;
  species_dict["names"] = species_data.names;
  species_dict["s_P_indexes"] = to_long_tensor(s_P_indexes_ext);
  species_dict["s_C12_indexes"] = to_long_tensor(s_C12_indexes);
  species_dict["ancestors_dense"] = to_double_matrix(ancestors, species_data.S, species_data.S);
  species_dict["Recipients_mat"] = to_double_matrix(recipients, species_data.S, species_data.S);
  species_dict["species_name_to_index"] = species_name_to_index;

  // Process each gene family
  py::dict families_dict;

  for (const auto& [family_name, gene_paths] : families) {
    std::vector<std::string> leaf_names;
    std::unordered_map<std::string, int> leaf_to_index;

    CladeData clade_data = amalgamate_clades_and_splits(gene_paths, leaf_names, leaf_to_index);
    CCPArrays ccp = build_ccp_arrays(clade_data, include_details);

    const size_t C = clade_data.clades.size();

    // Build leaf-to-species mapping
    std::vector<int64_t> leaf_row_index;
    std::vector<int64_t> leaf_col_index;
    std::vector<std::vector<int64_t>> clade_leaves_indices;
    std::vector<std::string> clade_leaf_labels;
    std::vector<uint8_t> clade_is_leaf;
    if (include_details) {
      clade_leaves_indices.resize(C);
      clade_leaf_labels.resize(C);
      clade_is_leaf.resize(C, 0);
    }

    for (size_t cid = 0; cid < C; ++cid) {
      const Clade& clade = clade_data.clades.get(cid);
      const BitVec& bits = clade.bits();
      for (size_t word_index = 0; word_index < bits.size(); ++word_index) {
        uint64_t word = bits[word_index];
        while (word) {
          unsigned long bit = __builtin_ctzll(word);
          size_t leaf_idx = word_index * BITS_PER_WORD + bit;
          if (leaf_idx < leaf_names.size()) {
            if (include_details) {
              clade_leaves_indices[cid].push_back(static_cast<int64_t>(leaf_idx));
            }
            if (clade.size() == 1) {
              const std::string &leaf_name = leaf_names[leaf_idx];
              std::string species = extract_species_name(leaf_name);
              auto it = species_name_to_index.find(species);
              if (it == species_name_to_index.end()) {
                throw std::runtime_error("Species " + species +
                                         " not found for gene leaf " +
                                         leaf_name);
              }
              leaf_row_index.push_back(static_cast<int64_t>(cid));
              leaf_col_index.push_back(static_cast<int64_t>(it->second));
              if (include_details) {
                clade_leaf_labels[cid] = leaf_name;
              }
            }
          }
          word &= word - 1ULL;
        }
      }
      if (include_details) {
        std::vector<int64_t> &indices = clade_leaves_indices[cid];
        std::sort(indices.begin(), indices.end());
      }
      if (include_details && clade.size() == 1) {
        clade_is_leaf[cid] = 1;
      }
    }

    py::dict ccp_dict;
    if (include_details) {
      ccp_dict["clade_leaves"] = clade_leaves_indices;
      ccp_dict["clade_leaf_labels"] = clade_leaf_labels;
      ccp_dict["clade_is_leaf"] = to_uint8_tensor(clade_is_leaf);
    }
    ccp_dict["split_counts"] = to_long_tensor(ccp.split_counts);
    ccp_dict["split_order"] = to_long_tensor(ccp.split_order);
    ccp_dict["split_parents_sorted"] = to_long_tensor(ccp.split_parents_sorted);
    ccp_dict["split_leftrights_sorted"] = to_long_tensor(ccp.split_leftrights_sorted);
    ccp_dict["log_split_probs_sorted"] = to_double_tensor(ccp.log_split_probs_sorted);
    ccp_dict["parents_sorted"] = to_long_tensor(ccp.parents_sorted);
    ccp_dict["seg_parent_ids"] = to_long_tensor(ccp.parents_sorted);
    ccp_dict["seg_counts"] = to_long_tensor(ccp.seg_counts);
    ccp_dict["ptr"] = to_long_tensor(ccp.ptr);
    ccp_dict["ptr_ge2"] = to_long_tensor(ccp.ptr_ge2);
    ccp_dict["num_segs_ge2"] = ccp.num_segs_ge2;
    ccp_dict["num_segs_eq1"] = ccp.num_segs_eq1;
    ccp_dict["num_segs_eq0"] = ccp.num_segs_eq0;
    ccp_dict["stop_reduce_ptr_idx"] = ccp.stop_reduce_ptr_idx;
    ccp_dict["end_rows_ge2"] = ccp.end_rows_ge2;
    ccp_dict["C"] = static_cast<int64_t>(C);
    ccp_dict["N_splits"] = static_cast<int64_t>(clade_data.splits.size());
    ccp_dict["root_clade_id"] = clade_data.root_clade_id;
    if (include_details) {
      ccp_dict["inclusion_children"] = to_long_tensor(ccp.inclusion_children);
      ccp_dict["inclusion_parents"] = to_long_tensor(ccp.inclusion_parents);
      ccp_dict["ubiquitous_clade_id"] = ccp.ubiquitous_clade_id;
    }

    auto [wave_level, n_waves] = compute_clade_waves(ccp, C);
    std::vector<int64_t> wave_level_i64(wave_level.begin(), wave_level.end());
    ccp_dict["clade_wave_level"] = to_long_tensor(wave_level_i64);
    ccp_dict["n_waves"] = static_cast<int64_t>(n_waves);

    // Also compute phased waves (default max_wave_size = C, i.e. no limit).
    // Keep this in sync with preprocess() so Python code can use the same
    // scheduler metadata whether families are preprocessed singly or batched.
    {
      SchedData sd = build_sched_data(clade_data);
      auto [phased_waves, phased_phases] = compute_phased_waves_impl(sd, static_cast<int>(C));
      py::list py_waves;
      for (const auto &w : phased_waves) {
        py_waves.append(to_long_tensor(w));
      }
      ccp_dict["phased_waves"] = py_waves;
      ccp_dict["phased_phases"] = phased_phases;
    }

    py::dict family_dict;
    family_dict["ccp"] = ccp_dict;
    family_dict["root_clade_id"] = clade_data.root_clade_id;
    family_dict["leaf_row_index"] = to_long_tensor(leaf_row_index);
    family_dict["leaf_col_index"] = to_long_tensor(leaf_col_index);

    families_dict[family_name.c_str()] = family_dict;
  }

  py::dict result;
  result["species"] = species_dict;
  result["families"] = families_dict;

  return result;
}

py::dict preprocess(const std::string &species_path,
                    const std::vector<std::string> &gene_paths) {
  std::unique_ptr<TreeNode> species_root = parse_newick_file(species_path);

  std::vector<std::string> leaf_names;
  std::unordered_map<std::string, int> leaf_to_index;

  CladeData clade_data = amalgamate_clades_and_splits(gene_paths, leaf_names, leaf_to_index);
  CCPArrays ccp = build_ccp_arrays(clade_data);

  const size_t C = clade_data.clades.size();

  SpeciesData species_data;
  std::vector<TreeNode *> species_order;
  enumerate_species(species_root.get(), species_order, species_data);
  auto species_name_to_index = build_species_name_map(species_data);

  auto [ancestors, recipients] = compute_ancestors_and_recipients(species_root.get(), species_data.S);

  std::vector<int64_t> leaf_row_index;
  std::vector<int64_t> leaf_col_index;
  std::vector<std::vector<int64_t>> clade_leaves_indices(C);
  std::vector<std::string> clade_leaf_labels(C);
  std::vector<uint8_t> clade_is_leaf(C, 0);

  for (size_t cid = 0; cid < C; ++cid) {
    const Clade& clade = clade_data.clades.get(cid);
    const BitVec& bits = clade.bits();
    std::vector<int64_t> &indices = clade_leaves_indices[cid];
    for (size_t word_index = 0; word_index < bits.size(); ++word_index) {
      uint64_t word = bits[word_index];
      while (word) {
        unsigned long bit = __builtin_ctzll(word);
        size_t leaf_idx = word_index * BITS_PER_WORD + bit;
        if (leaf_idx < leaf_names.size()) {
          indices.push_back(static_cast<int64_t>(leaf_idx));
          if (clade.size() == 1) {
            const std::string &leaf_name = leaf_names[leaf_idx];
            std::string species = extract_species_name(leaf_name);
            auto it = species_name_to_index.find(species);
            if (it == species_name_to_index.end()) {
              throw std::runtime_error("Species " + species +
                                       " not found for gene leaf " +
                                       leaf_name);
            }
            leaf_row_index.push_back(static_cast<int64_t>(cid));
            leaf_col_index.push_back(static_cast<int64_t>(it->second));
            clade_leaf_labels[cid] = leaf_name;
          }
        }
        word &= word - 1ULL;
      }
    }
    std::sort(indices.begin(), indices.end());
    if (clade.size() == 1) {
      clade_is_leaf[cid] = 1;
    }
  }

  py::dict species_dict;
  species_dict["S"] = species_data.S;
  species_dict["names"] = species_data.names;

  std::vector<int64_t> s_P_indexes;
  std::vector<int64_t> s_C1_indexes;
  std::vector<int64_t> s_C2_indexes;
  for (int i = 0; i < species_data.S; ++i) {
    if (species_data.children[i].size() == 2) {
      int left = species_data.children[i][0];
      int right = species_data.children[i][1];
      s_P_indexes.push_back(i);
      s_C1_indexes.push_back(left);
      s_C2_indexes.push_back(right);
    } else if (!species_data.children[i].empty()) {
      throw std::runtime_error("Species tree must be strictly binary");
    }
  }
  std::vector<int64_t> s_P_indexes_ext = s_P_indexes;
  for (int64_t idx : s_P_indexes) {
    s_P_indexes_ext.push_back(idx + species_data.S);
  }
  std::vector<int64_t> s_C12_indexes = s_C1_indexes;
  s_C12_indexes.insert(s_C12_indexes.end(), s_C2_indexes.begin(),
                       s_C2_indexes.end());

  species_dict["s_P_indexes"] = to_long_tensor(s_P_indexes_ext);
  species_dict["s_C12_indexes"] = to_long_tensor(s_C12_indexes);
  species_dict["ancestors_dense"] = to_double_matrix(ancestors, species_data.S, species_data.S);
  species_dict["Recipients_mat"] = to_double_matrix(recipients, species_data.S, species_data.S);
  species_dict["species_name_to_index"] = species_name_to_index;

  py::dict ccp_dict;
  ccp_dict["clade_leaves"] = clade_leaves_indices;
  ccp_dict["clade_leaf_labels"] = clade_leaf_labels;
  ccp_dict["clade_is_leaf"] = to_uint8_tensor(clade_is_leaf);
  ccp_dict["split_counts"] = to_long_tensor(ccp.split_counts);
  ccp_dict["split_order"] = to_long_tensor(ccp.split_order);
  ccp_dict["split_parents_sorted"] = to_long_tensor(ccp.split_parents_sorted);
  ccp_dict["split_leftrights_sorted"] = to_long_tensor(ccp.split_leftrights_sorted);
  ccp_dict["log_split_probs_sorted"] = to_double_tensor(ccp.log_split_probs_sorted);
  ccp_dict["parents_sorted"] = to_long_tensor(ccp.parents_sorted);
  ccp_dict["seg_parent_ids"] = to_long_tensor(ccp.parents_sorted);
  ccp_dict["seg_counts"] = to_long_tensor(ccp.seg_counts);
  ccp_dict["ptr"] = to_long_tensor(ccp.ptr);
  ccp_dict["ptr_ge2"] = to_long_tensor(ccp.ptr_ge2);
  ccp_dict["num_segs_ge2"] = ccp.num_segs_ge2;
  ccp_dict["num_segs_eq1"] = ccp.num_segs_eq1;
  ccp_dict["num_segs_eq0"] = ccp.num_segs_eq0;
  ccp_dict["stop_reduce_ptr_idx"] = ccp.stop_reduce_ptr_idx;
  ccp_dict["end_rows_ge2"] = ccp.end_rows_ge2;
  ccp_dict["C"] = static_cast<int64_t>(C);
  ccp_dict["N_splits"] = static_cast<int64_t>(clade_data.splits.size());
  ccp_dict["root_clade_id"] = clade_data.root_clade_id;
  ccp_dict["inclusion_children"] = to_long_tensor(ccp.inclusion_children);
  ccp_dict["inclusion_parents"] = to_long_tensor(ccp.inclusion_parents);
  ccp_dict["ubiquitous_clade_id"] = ccp.ubiquitous_clade_id;

  auto [wave_level, n_waves] = compute_clade_waves(ccp, C);
  std::vector<int64_t> wave_level_i64(wave_level.begin(), wave_level.end());
  ccp_dict["clade_wave_level"] = to_long_tensor(wave_level_i64);
  ccp_dict["n_waves"] = static_cast<int64_t>(n_waves);

  // Also compute phased waves (default max_wave_size = C, i.e. no limit)
  {
    SchedData sd = build_sched_data(clade_data);
    auto [phased_waves, phased_phases] = compute_phased_waves_impl(sd, static_cast<int>(C));
    py::list py_waves;
    for (const auto &w : phased_waves) {
      py_waves.append(to_long_tensor(w));
    }
    ccp_dict["phased_waves"] = py_waves;
    ccp_dict["phased_phases"] = phased_phases;
  }

  py::dict result;
  result["species"] = species_dict;
  result["ccp"] = ccp_dict;
  result["leaf_row_index"] = to_long_tensor(leaf_row_index);
  result["leaf_col_index"] = to_long_tensor(leaf_col_index);

  return result;
}

// ============================================================================
// Lightweight wave stats — gene trees only, no species tree needed
// ============================================================================

/**
 * @brief For each family, compute wave scheduling stats (no species tree).
 *
 * Uses Hu's critical-path priority scheduling (longest path to sink):
 *   1. Compute bottom-level bl(c) = longest path from c to the root (sink).
 *      Leaves deep in the tree get high bl → processed first.
 *   2. Greedy list scheduling: at each time step, pop the W highest-bl ready
 *      clades into a batch.
 *
 * Hu (1961) proved this optimal for in-trees; on general DAGs it is the
 * best known O(n log n) heuristic.  It naturally groups siblings (same bl)
 * and keeps required_clades local.
 *
 * @param families      Map from family name to list of gene tree file paths.
 * @param max_wave_size Maximum clades per batch (GPU kernel width).
 * @return List of dicts per batch: family, wave, n_clades, n_splits,
 *         required_clades.
 */
py::list compute_wave_stats(
    const std::map<std::string, std::vector<std::string>> &families,
    int max_wave_size) {

  struct FamilyEntry { std::string name; std::vector<std::string> paths; };
  std::vector<FamilyEntry> fam_vec;
  fam_vec.reserve(families.size());
  for (const auto &[name, paths] : families) {
    fam_vec.push_back({name, paths});
  }
  const int n_fam = static_cast<int>(fam_vec.size());

  struct Row {
    std::string family;
    int wave;
    int64_t n_clades, n_splits, required_clades;
  };
  std::vector<std::vector<Row>> per_family_rows(n_fam);

  #pragma omp parallel for schedule(dynamic, 1)
  for (int fi = 0; fi < n_fam; ++fi) {
    const std::string &family_name = fam_vec[fi].name;
    const std::vector<std::string> &gene_paths = fam_vec[fi].paths;

    std::vector<std::string> leaf_names;
    std::unordered_map<std::string, int> leaf_to_index;

    CladeData clade_data = amalgamate_clades_and_splits(gene_paths, leaf_names, leaf_to_index);
    CCPArrays ccp = build_ccp_arrays(clade_data);

    const size_t C = clade_data.clades.size();
    const size_t N = ccp.split_parents_sorted.size();

    // ---- Build adjacency ----
    std::vector<std::vector<int64_t>> children(C);
    std::vector<std::vector<int64_t>> parents_of(C);
    std::vector<int32_t> remaining(C, 0);
    {
      std::vector<std::unordered_set<int64_t>> child_sets(C);
      for (size_t i = 0; i < N; ++i) {
        int64_t p = ccp.split_parents_sorted[i];
        int64_t l = ccp.split_lefts_sorted[i];
        int64_t r = ccp.split_rights_sorted[i];
        if (child_sets[p].insert(l).second) {
          children[p].push_back(l);
          parents_of[l].push_back(p);
          remaining[p]++;
        }
        if (l != r && child_sets[p].insert(r).second) {
          children[p].push_back(r);
          parents_of[r].push_back(p);
          remaining[p]++;
        }
      }
    }

    // ---- Hu: compute bottom-level (longest path to root/sink) ----
    // BFS from sink (root) backwards through reversed edges.
    // bl[root] = 0; for all others bl[c] = max over parents p of (bl[p]+1).
    // Equivalently: do a reverse-topological pass from root.
    // We already have compute_clade_waves which gives BFS-from-leaves levels;
    // bottom-level = max_level - bfs_level.
    auto [bfs_level, n_bfs_waves] = compute_clade_waves(ccp, C);
    int32_t max_level = n_bfs_waves - 1;
    std::vector<int32_t> bl(C);
    for (size_t c = 0; c < C; ++c) {
      bl[c] = max_level - bfs_level[c];
    }

    // ---- Greedy list scheduling: eager parents + sibling-grouped leaves ----
    //
    // Priority = (is_internal, bl, sibling_key, clade_id).
    //
    // is_internal = 1 for non-leaf clades, 0 for leaves (split_counts==0).
    // This ensures that when a parent becomes ready, it is processed in the
    // very next batch (before remaining leaves), creating a pipeline:
    //   batch 1: 256 sibling-grouped leaves → ~128 parents ready
    //   batch 2: 128 parents + 128 leaves   → ~64 grandparents + ~64 parents
    //   batch 3: ...interleaved...
    // Each batch stays close to W = 256 because parents fill the slots that
    // would otherwise sit empty while leaves alone trickle through.
    //
    // Among internal clades: higher bl (deeper in tree) first.
    // Among leaves: group by sibling_key (binary parent ID) so that sibling
    // pairs land in the same batch and unlock their parent immediately.

    using PQEntry = std::tuple<int32_t, int32_t, int64_t, int64_t>;
    //                         is_internal, bl, sibling_key, clade_id
    std::priority_queue<PQEntry> ready;

    // Identify leaf clades and compute sibling_key for them
    std::vector<int64_t> sibling_key(C, 0);
    std::vector<int32_t> is_internal(C, 1);
    for (size_t c = 0; c < C; ++c) {
      if (ccp.split_counts[c] == 0) {
        is_internal[c] = 0;  // leaf clade
        if (!parents_of[c].empty()) {
          // Find parent with fewest children (closest binary split)
          int64_t best_parent = parents_of[c][0];
          size_t best_size = children[best_parent].size();
          for (size_t j = 1; j < parents_of[c].size(); ++j) {
            int64_t p = parents_of[c][j];
            if (children[p].size() < best_size) {
              best_size = children[p].size();
              best_parent = p;
            }
          }
          sibling_key[c] = best_parent;
        }
      }
    }

    for (size_t c = 0; c < C; ++c) {
      if (remaining[c] == 0) {
        ready.push({is_internal[c], bl[c], sibling_key[c],
                     static_cast<int64_t>(c)});
      }
    }

    std::vector<Row> &rows = per_family_rows[fi];
    int wave_idx = 0;

    while (!ready.empty()) {
      // Pop W highest-priority ready clades into a batch.
      std::vector<int64_t> batch;
      batch.reserve(max_wave_size);
      while (!ready.empty() && (int)batch.size() < max_wave_size) {
        batch.push_back(std::get<3>(ready.top()));
        ready.pop();
      }

      // Compute stats
      std::unordered_set<int64_t> batch_set(batch.begin(), batch.end());
      int64_t n_splits = 0;
      std::unordered_set<int64_t> needed;
      for (int64_t c : batch) {
        n_splits += ccp.split_counts[c];
        for (int64_t child : children[c]) {
          if (batch_set.find(child) == batch_set.end()) {
            needed.insert(child);
          }
        }
      }
      rows.push_back({family_name, wave_idx,
                      static_cast<int64_t>(batch.size()), n_splits,
                      static_cast<int64_t>(needed.size())});

      // Update parents; newly-ready go into the priority queue.
      // Parents are always internal (is_internal=1) → they get higher
      // priority than remaining leaves, so they're processed next.
      for (int64_t c : batch) {
        for (int64_t p : parents_of[c]) {
          if (--remaining[p] == 0) {
            ready.push({is_internal[p], bl[p], sibling_key[p], p});
          }
        }
      }
      ++wave_idx;
    }
  }

  // Collect results in family order (deterministic output)
  py::list result;
  for (int fi = 0; fi < n_fam; ++fi) {
    for (const auto &r : per_family_rows[fi]) {
      py::dict row;
      row["family"]          = r.family;
      row["wave"]            = r.wave;
      row["n_clades"]        = r.n_clades;
      row["n_splits"]        = r.n_splits;
      row["required_clades"] = r.required_clades;
      result.append(row);
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// Union-Find for outgoing packet identification
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Packet-aware highest-label scheduler (from scheduling_proposal.md)
//
// For each family (single binary gene tree), exploits the oriented-edge
// structure:
//   - Outgoing packets O(u): children that co-appear in a split → completing
//     2+ from O(u) in one wave immediately unlocks a parent.
//   - Incoming packets I(u): parents whose children come from the same O(u) →
//     they share predecessor clades, reducing required_clades.
//
// Priority per selection: (λ, o_tail, i_head)
//   λ     = longest path from clade to root (true bottom-level)
//   o_tail = #items already selected from the same outgoing packet in this wave
//   i_head = #items already selected from the same incoming packet in this wave
// ---------------------------------------------------------------------------
py::list compute_packet_wave_stats(
    const std::map<std::string, std::vector<std::string>> &families,
    int max_wave_size) {

  struct FamilyEntry { std::string name; std::vector<std::string> paths; };
  std::vector<FamilyEntry> fam_vec;
  fam_vec.reserve(families.size());
  for (const auto &[name, paths] : families)
    fam_vec.push_back({name, paths});
  const int n_fam = static_cast<int>(fam_vec.size());

  struct Row {
    std::string family;
    int wave;
    int64_t n_clades, n_splits, required_clades;
  };
  std::vector<std::vector<Row>> per_family_rows(n_fam);

  #pragma omp parallel for schedule(dynamic, 1)
  for (int fi = 0; fi < n_fam; ++fi) {
    const std::string &family_name = fam_vec[fi].name;
    const std::vector<std::string> &gene_paths = fam_vec[fi].paths;

    std::vector<std::string> leaf_names;
    std::unordered_map<std::string, int> leaf_to_index;
    CladeData clade_data = amalgamate_clades_and_splits(
        gene_paths, leaf_names, leaf_to_index);
    CCPArrays ccp = build_ccp_arrays(clade_data);

    const size_t C = clade_data.clades.size();
    const size_t N = ccp.split_parents_sorted.size();

    // ---- Build adjacency ----
    std::vector<std::vector<int64_t>> children(C);
    std::vector<std::vector<int64_t>> parents_of(C);
    std::vector<int32_t> remaining(C, 0);
    {
      std::vector<std::unordered_set<int64_t>> child_sets(C);
      for (size_t i = 0; i < N; ++i) {
        int64_t p = ccp.split_parents_sorted[i];
        int64_t l = ccp.split_lefts_sorted[i];
        int64_t r = ccp.split_rights_sorted[i];
        if (child_sets[p].insert(l).second) {
          children[p].push_back(l);
          parents_of[l].push_back(p);
          remaining[p]++;
        }
        if (l != r && child_sets[p].insert(r).second) {
          children[p].push_back(r);
          parents_of[r].push_back(p);
          remaining[p]++;
        }
      }
    }

    // ---- Find root (Ω): clade with no parents ----
    int64_t root_id = -1;
    for (size_t c = 0; c < C; ++c) {
      if (parents_of[c].empty()) { root_id = static_cast<int64_t>(c); break; }
    }

    // ---- Compute λ: longest path from clade to root ----
    // Process in reverse topological order (root first, leaves last).
    // Use BFS levels to determine processing order.
    auto [bfs_level, n_waves] = compute_clade_waves(ccp, C);

    // Group clades by BFS level
    std::vector<std::vector<int64_t>> clades_at_level(n_waves);
    for (size_t c = 0; c < C; ++c)
      clades_at_level[bfs_level[c]].push_back(static_cast<int64_t>(c));

    // Propagate λ from root (highest BFS level) down to leaves
    std::vector<int32_t> lambda(C, 0);
    for (int lev = n_waves - 1; lev >= 0; --lev) {
      for (int64_t c : clades_at_level[lev]) {
        for (int64_t ch : children[c]) {
          lambda[ch] = std::max(lambda[ch], lambda[c] + 1);
        }
      }
    }

    // ---- Build outgoing packets (union-find over co-children) ----
    // For non-root splits: children l, r share an outgoing packet.
    UnionFind uf(static_cast<int>(C));
    for (size_t i = 0; i < N; ++i) {
      int64_t p = ccp.split_parents_sorted[i];
      if (p == root_id) continue;  // skip Ω's splits
      int64_t l = ccp.split_lefts_sorted[i];
      int64_t r = ccp.split_rights_sorted[i];
      if (l != r) uf.unite(static_cast<int>(l), static_cast<int>(r));
    }
    // out_packet[c] = union-find representative
    std::vector<int> out_packet(C);
    for (size_t c = 0; c < C; ++c) out_packet[c] = uf.find(static_cast<int>(c));

    // ---- Build incoming packets ----
    // For clade p with children l, r: in_packet[p] = out_packet[l].
    // For leaf clades (no children): in_packet = -1 (not applicable).
    std::vector<int> in_packet(C, -1);
    for (size_t c = 0; c < C; ++c) {
      if (!children[c].empty()) {
        in_packet[c] = out_packet[children[c][0]];
      }
    }

    // ---- Forward pass: packet-aware scheduling ----
    // Bucket ready clades by λ for efficient max-λ lookup.
    int max_lambda = 0;
    for (size_t c = 0; c < C; ++c)
      if (static_cast<int>(lambda[c]) > max_lambda) max_lambda = lambda[c];

    std::vector<std::vector<int64_t>> ready_buckets(max_lambda + 1);
    for (size_t c = 0; c < C; ++c) {
      if (remaining[c] == 0 && static_cast<int64_t>(c) != root_id) {
        ready_buckets[lambda[c]].push_back(static_cast<int64_t>(c));
      }
    }

    int top_bucket = max_lambda;
    while (top_bucket >= 0 && ready_buckets[top_bucket].empty()) --top_bucket;

    auto count_ready = [&]() -> size_t {
      size_t n = 0;
      for (int b = 0; b <= top_bucket; ++b) n += ready_buckets[b].size();
      return n;
    };

    std::vector<Row> &rows = per_family_rows[fi];
    int wave_idx = 0;

    while (top_bucket >= 0 || (root_id >= 0 && remaining[root_id] == 0)) {
      // Collect one wave
      std::vector<int64_t> batch;
      batch.reserve(max_wave_size);

      // Check if only root remains
      bool root_ready = (root_id >= 0 && remaining[root_id] == 0);
      if (top_bucket < 0 && root_ready) {
        // Only root left
        batch.push_back(root_id);
        root_id = -1;  // mark as scheduled
      } else {
        // Per-wave counters for outgoing and incoming packets
        std::unordered_map<int, int32_t> o_count;  // out_packet_id → count
        std::unordered_map<int, int32_t> i_count;  // in_packet_id → count

        while (static_cast<int>(batch.size()) < max_wave_size && top_bucket >= 0) {
          // Scan the top bucket for the best clade:
          //   maximize (o_count[out_packet], i_count[in_packet])
          auto &bucket = ready_buckets[top_bucket];
          if (bucket.empty()) {
            --top_bucket;
            while (top_bucket >= 0 && ready_buckets[top_bucket].empty())
              --top_bucket;
            if (top_bucket < 0) break;
            continue;
          }

          int best_idx = 0;
          int32_t best_o = -1, best_i = -1;
          for (size_t j = 0; j < bucket.size(); ++j) {
            int64_t c = bucket[j];
            int32_t oc = 0, ic = 0;
            auto oit = o_count.find(out_packet[c]);
            if (oit != o_count.end()) oc = oit->second;
            auto iit = i_count.find(in_packet[c]);
            if (iit != i_count.end() && in_packet[c] >= 0) ic = iit->second;
            if (oc > best_o || (oc == best_o && ic > best_i)) {
              best_o = oc; best_i = ic; best_idx = static_cast<int>(j);
            }
          }

          // Select the best clade
          int64_t selected = bucket[best_idx];
          bucket[best_idx] = bucket.back();
          bucket.pop_back();

          batch.push_back(selected);
          o_count[out_packet[selected]]++;
          if (in_packet[selected] >= 0)
            i_count[in_packet[selected]]++;

          // If bucket is now empty, move to next
          if (bucket.empty()) {
            while (top_bucket >= 0 && ready_buckets[top_bucket].empty())
              --top_bucket;
          }
        }
      }

      // Compute stats for this wave
      std::unordered_set<int64_t> batch_set(batch.begin(), batch.end());
      int64_t n_splits = 0;
      std::unordered_set<int64_t> needed;
      for (int64_t c : batch) {
        n_splits += ccp.split_counts[c];
        for (int64_t child : children[c]) {
          if (batch_set.find(child) == batch_set.end()) {
            needed.insert(child);
          }
        }
      }
      rows.push_back({family_name, wave_idx,
                       static_cast<int64_t>(batch.size()), n_splits,
                       static_cast<int64_t>(needed.size())});

      // Update parents: newly-ready clades go into buckets
      for (int64_t c : batch) {
        for (int64_t p : parents_of[c]) {
          if (p == root_id) {
            --remaining[p];
            continue;
          }
          if (--remaining[p] == 0) {
            ready_buckets[lambda[p]].push_back(p);
            if (lambda[p] > top_bucket) top_bucket = lambda[p];
          }
        }
      }
      ++wave_idx;
    }
  }

  // Collect results in family order
  py::list result;
  for (int fi = 0; fi < n_fam; ++fi) {
    for (const auto &r : per_family_rows[fi]) {
      py::dict row;
      row["family"]          = r.family;
      row["wave"]            = r.wave;
      row["n_clades"]        = r.n_clades;
      row["n_splits"]        = r.n_splits;
      row["required_clades"] = r.required_clades;
      result.append(row);
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// Phased scheduler: separates clades into three categories
//
//   Phase 1 — leaf clades (split_count = 0): base cases, no matmul.
//             Grouped by outgoing packet so sibling leaves land together,
//             maximising parent readiness for phase 2.
//
//   Phase 2 — internal non-root clades (split_count ≥ 1): the matmul kernel.
//             Packet-aware scheduling with priority (λ, o_tail, i_head).
//             Handles any split_count (single-tree or multi-tree CCP).
//
//   Phase 3 — root Ω (split_count ≫ 1): reduction over all its splits, alone.
//
// Output columns: family, phase, wave, n_clades, n_splits, required_clades
// ---------------------------------------------------------------------------
py::list compute_phased_wave_stats(
    const std::map<std::string, std::vector<std::string>> &families,
    int max_wave_size) {

  struct FamilyEntry { std::string name; std::vector<std::string> paths; };
  std::vector<FamilyEntry> fam_vec;
  fam_vec.reserve(families.size());
  for (const auto &[name, paths] : families)
    fam_vec.push_back({name, paths});
  const int n_fam = static_cast<int>(fam_vec.size());

  struct Row {
    std::string family;
    int phase, wave;
    int64_t n_clades, n_splits, required_clades;
  };
  std::vector<std::vector<Row>> per_family_rows(n_fam);

  #pragma omp parallel for schedule(dynamic, 1)
  for (int fi = 0; fi < n_fam; ++fi) {
    const std::string &family_name = fam_vec[fi].name;
    const std::vector<std::string> &gene_paths = fam_vec[fi].paths;

    std::vector<std::string> leaf_names;
    std::unordered_map<std::string, int> leaf_to_index;
    CladeData clade_data = amalgamate_clades_and_splits(
        gene_paths, leaf_names, leaf_to_index);
    SchedData sd = build_sched_data(clade_data);

    const size_t C = sd.C;
    auto &children_adj = sd.children;
    auto &parents_of   = sd.parents_of;
    auto &remaining    = sd.remaining;
    auto &lambda       = sd.lambda;
    auto &out_packet   = sd.out_packet;
    auto &in_packet    = sd.in_packet;
    int64_t root_id    = sd.root_id;

    // ---- Categorize clades ----
    std::vector<int64_t> leaf_clades;
    for (size_t c = 0; c < C; ++c) {
      if (static_cast<int64_t>(c) == root_id) continue;
      if (sd.split_counts[c] == 0) {
        leaf_clades.push_back(static_cast<int64_t>(c));
      }
    }

    // ==================================================================
    // Phase 1: leaf clades, grouped by outgoing packet
    // ==================================================================
    // Sort leaves by outgoing packet so siblings land in the same wave,
    // maximising the number of parents unlocked for phase 2.
    std::sort(leaf_clades.begin(), leaf_clades.end(),
      [&](int64_t a, int64_t b) {
        if (out_packet[a] != out_packet[b]) return out_packet[a] < out_packet[b];
        return a < b;
      });

    std::vector<Row> &rows = per_family_rows[fi];
    int wave_idx = 0;

    for (size_t start = 0; start < leaf_clades.size();
         start += static_cast<size_t>(max_wave_size)) {
      size_t end = std::min(start + static_cast<size_t>(max_wave_size),
                            leaf_clades.size());
      std::vector<int64_t> batch(leaf_clades.begin() + start,
                                  leaf_clades.begin() + end);

      // Leaf clades: n_splits = 0, required_clades = 0
      rows.push_back({family_name, /*phase=*/1, wave_idx,
                       static_cast<int64_t>(batch.size()), 0, 0});

      // Update parents
      for (int64_t c : batch) {
        for (int64_t p : parents_of[c]) {
          if (p == root_id) { --remaining[p]; continue; }
          --remaining[p];
        }
      }
      ++wave_idx;
    }

    // ==================================================================
    // Phase 2: internal non-root clades (split_count ≥ 1)
    // ==================================================================
    // Collect initially-ready internal clades (all children done after phase 1)
    int max_lambda = 0;
    for (size_t c = 0; c < C; ++c)
      if (static_cast<int>(lambda[c]) > max_lambda) max_lambda = lambda[c];

    std::vector<std::vector<int64_t>> ready_buckets(max_lambda + 1);
    for (size_t c = 0; c < C; ++c) {
      if (static_cast<int64_t>(c) == root_id) continue;
      if (sd.split_counts[c] == 0) continue;  // leaves already done
      if (remaining[c] == 0) {
        ready_buckets[lambda[c]].push_back(static_cast<int64_t>(c));
      }
    }

    int top_bucket = max_lambda;
    while (top_bucket >= 0 && ready_buckets[top_bucket].empty()) --top_bucket;

    // Track which clades have been scheduled (for required_clades computation)
    std::vector<bool> scheduled(C, false);
    for (int64_t c : leaf_clades) scheduled[c] = true;

    while (top_bucket >= 0) {
      std::vector<int64_t> batch;
      batch.reserve(max_wave_size);

      // Per-wave packet counters
      std::unordered_map<int, int32_t> o_count;
      std::unordered_map<int, int32_t> i_count;

      while (static_cast<int>(batch.size()) < max_wave_size && top_bucket >= 0) {
        auto &bucket = ready_buckets[top_bucket];
        if (bucket.empty()) {
          --top_bucket;
          while (top_bucket >= 0 && ready_buckets[top_bucket].empty())
            --top_bucket;
          if (top_bucket < 0) break;
          continue;
        }

        // Find best in bucket: maximize (o_count[out_packet], i_count[in_packet])
        int best_idx = 0;
        int32_t best_o = -1, best_i = -1;
        for (size_t j = 0; j < bucket.size(); ++j) {
          int64_t c = bucket[j];
          int32_t oc = 0, ic = 0;
          auto oit = o_count.find(out_packet[c]);
          if (oit != o_count.end()) oc = oit->second;
          if (in_packet[c] >= 0) {
            auto iit = i_count.find(in_packet[c]);
            if (iit != i_count.end()) ic = iit->second;
          }
          if (oc > best_o || (oc == best_o && ic > best_i)) {
            best_o = oc; best_i = ic; best_idx = static_cast<int>(j);
          }
        }

        int64_t selected = bucket[best_idx];
        bucket[best_idx] = bucket.back();
        bucket.pop_back();

        batch.push_back(selected);
        o_count[out_packet[selected]]++;
        if (in_packet[selected] >= 0)
          i_count[in_packet[selected]]++;

        if (bucket.empty()) {
          while (top_bucket >= 0 && ready_buckets[top_bucket].empty())
            --top_bucket;
        }
      }

      // Compute stats
      std::unordered_set<int64_t> batch_set(batch.begin(), batch.end());
      int64_t n_splits = 0;
      std::unordered_set<int64_t> needed;
      for (int64_t c : batch) {
        n_splits += sd.split_counts[c];
        for (int64_t child : children_adj[c]) {
          if (batch_set.find(child) == batch_set.end()) {
            needed.insert(child);
          }
        }
      }
      rows.push_back({family_name, /*phase=*/2, wave_idx,
                       static_cast<int64_t>(batch.size()), n_splits,
                       static_cast<int64_t>(needed.size())});

      // Mark scheduled and update parents
      for (int64_t c : batch) {
        scheduled[c] = true;
        for (int64_t p : parents_of[c]) {
          if (p == root_id) { --remaining[p]; continue; }
          if (--remaining[p] == 0) {
            ready_buckets[lambda[p]].push_back(p);
            if (lambda[p] > top_bucket) top_bucket = lambda[p];
          }
        }
      }
      ++wave_idx;
    }

    // ==================================================================
    // Phase 3: root Ω
    // ==================================================================
    if (root_id >= 0) {
      int64_t root_splits = sd.split_counts[root_id];
      // required_clades = all distinct children of root
      int64_t root_required = static_cast<int64_t>(children_adj[root_id].size());
      rows.push_back({family_name, /*phase=*/3, wave_idx,
                       1, root_splits, root_required});
      ++wave_idx;
    }
  }

  // Collect results in family order
  py::list result;
  for (int fi = 0; fi < n_fam; ++fi) {
    for (const auto &r : per_family_rows[fi]) {
      py::dict row;
      row["family"]          = r.family;
      row["phase"]           = r.phase;
      row["wave"]            = r.wave;
      row["n_clades"]        = r.n_clades;
      row["n_splits"]        = r.n_splits;
      row["required_clades"] = r.required_clades;
      result.append(row);
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// Phased + cross-family: three phases, each with global batching across
// all families.
//
//   Phase 1 — leaf clades from all families (split_count = 0).
//             Sorted by (family, outgoing_packet) for sibling grouping.
//   Phase 2 — internal non-root clades from all families (split_count ≥ 1).
//             Global priority queue keyed by λ (longest path to root).
//   Phase 3 — root Ω from all families.
//
// Output: batch, phase, n_clades, n_splits, n_families, required_clades
// ---------------------------------------------------------------------------
py::list compute_phased_cross_family_wave_stats(
    const std::map<std::string, std::vector<std::string>> &families,
    int max_wave_size) {

  struct FamilyEntry { std::string name; std::vector<std::string> paths; };
  std::vector<FamilyEntry> fam_vec;
  fam_vec.reserve(families.size());
  for (const auto &[name, paths] : families)
    fam_vec.push_back({name, paths});
  const int n_fam = static_cast<int>(fam_vec.size());

  // SchedData serves as FamData — built with the lightweight path
  std::vector<SchedData> fam_data(n_fam);

  // Parse all families in parallel (uses CCPLight, skips O(C²) inclusion DAG)
  #pragma omp parallel for schedule(dynamic, 1)
  for (int fi = 0; fi < n_fam; ++fi) {
    std::vector<std::string> leaf_names;
    std::unordered_map<std::string, int> leaf_to_index;
    CladeData clade_data = amalgamate_clades_and_splits(
        fam_vec[fi].paths, leaf_names, leaf_to_index);
    fam_data[fi] = build_sched_data(clade_data);
  }

  // ================================================================
  // Phase 1: leaf clades from all families
  // ================================================================
  // Collect (family, clade) for all leaves, sort by (family, out_packet)
  // for sibling grouping.
  struct FC { int f; int64_t c; };
  std::vector<FC> all_leaves;
  for (int fi = 0; fi < n_fam; ++fi) {
    for (size_t c = 0; c < fam_data[fi].C; ++c) {
      if (static_cast<int64_t>(c) == fam_data[fi].root_id) continue;
      if (fam_data[fi].split_counts[c] == 0) {
        all_leaves.push_back({fi, static_cast<int64_t>(c)});
      }
    }
  }
  std::sort(all_leaves.begin(), all_leaves.end(),
    [&](const FC &a, const FC &b) {
      if (a.f != b.f) return a.f < b.f;
      return fam_data[a.f].out_packet[a.c] < fam_data[b.f].out_packet[b.c];
    });

  struct Row {
    int batch, phase;
    int64_t n_clades, n_splits, n_families, required_clades;
  };
  std::vector<Row> rows;
  int batch_idx = 0;

  for (size_t start = 0; start < all_leaves.size();
       start += static_cast<size_t>(max_wave_size)) {
    size_t end = std::min(start + static_cast<size_t>(max_wave_size),
                          all_leaves.size());
    int64_t n = static_cast<int64_t>(end - start);

    // Count distinct families in this batch
    std::unordered_set<int> fams;
    for (size_t i = start; i < end; ++i) fams.insert(all_leaves[i].f);

    rows.push_back({batch_idx, /*phase=*/1, n, 0,
                     static_cast<int64_t>(fams.size()), 0});

    // Update remaining for parents
    for (size_t i = start; i < end; ++i) {
      int fi = all_leaves[i].f;
      int64_t c = all_leaves[i].c;
      for (int64_t p : fam_data[fi].parents_of[c]) {
        --fam_data[fi].remaining[p];
      }
    }
    ++batch_idx;
  }

  // ================================================================
  // Phase 2: internal non-root clades from all families
  // ================================================================
  // Collect initially-ready internal clades after phase 1
  using Entry = std::tuple<int32_t, int, int64_t>;  // (λ, family, clade)
  std::priority_queue<Entry> ready;

  for (int fi = 0; fi < n_fam; ++fi) {
    for (size_t c = 0; c < fam_data[fi].C; ++c) {
      if (static_cast<int64_t>(c) == fam_data[fi].root_id) continue;
      if (fam_data[fi].split_counts[c] == 0) continue;  // leaf, done
      if (fam_data[fi].remaining[c] == 0) {
        ready.push({fam_data[fi].lambda[c], fi, static_cast<int64_t>(c)});
      }
    }
  }

  while (!ready.empty()) {
    std::vector<FC> batch;
    batch.reserve(max_wave_size);
    while (!ready.empty() && static_cast<int>(batch.size()) < max_wave_size) {
      auto top = ready.top();
      ready.pop();
      batch.push_back({std::get<1>(top), std::get<2>(top)});
    }

    // Compute stats
    int64_t n_splits = 0;
    std::unordered_set<int> families_in_batch;
    std::unordered_map<int, std::unordered_set<int64_t>> batch_by_fam;
    for (auto &fc : batch) {
      n_splits += fam_data[fc.f].split_counts[fc.c];
      families_in_batch.insert(fc.f);
      batch_by_fam[fc.f].insert(fc.c);
    }

    // required_clades: distinct (family, child) not in current batch
    std::set<std::pair<int, int64_t>> needed;
    for (auto &fc : batch) {
      for (int64_t child : fam_data[fc.f].children[fc.c]) {
        if (batch_by_fam[fc.f].count(child) == 0) {
          needed.insert({fc.f, child});
        }
      }
    }

    rows.push_back({batch_idx, /*phase=*/2,
                     static_cast<int64_t>(batch.size()), n_splits,
                     static_cast<int64_t>(families_in_batch.size()),
                     static_cast<int64_t>(needed.size())});

    // Update parents
    for (auto &fc : batch) {
      for (int64_t p : fam_data[fc.f].parents_of[fc.c]) {
        if (--fam_data[fc.f].remaining[p] == 0) {
          if (p != fam_data[fc.f].root_id) {
            ready.push({fam_data[fc.f].lambda[p], fc.f, p});
          }
        }
      }
    }
    ++batch_idx;
  }

  // ================================================================
  // Phase 3: root clades from all families
  // ================================================================
  std::vector<FC> all_roots;
  for (int fi = 0; fi < n_fam; ++fi) {
    if (fam_data[fi].root_id >= 0) {
      all_roots.push_back({fi, fam_data[fi].root_id});
    }
  }
  for (size_t start = 0; start < all_roots.size();
       start += static_cast<size_t>(max_wave_size)) {
    size_t end = std::min(start + static_cast<size_t>(max_wave_size),
                          all_roots.size());
    int64_t n = static_cast<int64_t>(end - start);
    int64_t total_splits = 0;
    int64_t total_required = 0;
    for (size_t i = start; i < end; ++i) {
      int fi = all_roots[i].f;
      int64_t c = all_roots[i].c;
      total_splits += fam_data[fi].split_counts[c];
      total_required += static_cast<int64_t>(fam_data[fi].children[c].size());
    }
    rows.push_back({batch_idx, /*phase=*/3, n, total_splits,
                     n, total_required});
    ++batch_idx;
  }

  // Convert to Python
  py::list result;
  for (const auto &r : rows) {
    py::dict row;
    row["batch"]           = r.batch;
    row["phase"]           = r.phase;
    row["n_clades"]        = r.n_clades;
    row["n_splits"]        = r.n_splits;
    row["n_families"]      = r.n_families;
    row["required_clades"] = r.required_clades;
    result.append(row);
  }
  return result;
}

// ---------------------------------------------------------------------------
// Cross-family batching (unphased): schedule clades from ALL families into
// global batches of size ≤ max_wave_size.
// ---------------------------------------------------------------------------
py::list compute_cross_family_wave_stats(
    const std::map<std::string, std::vector<std::string>> &families,
    int max_wave_size) {

  struct FamilyEntry { std::string name; std::vector<std::string> paths; };
  std::vector<FamilyEntry> fam_vec;
  fam_vec.reserve(families.size());
  for (const auto &[name, paths] : families)
    fam_vec.push_back({name, paths});
  const int n_fam = static_cast<int>(fam_vec.size());

  // Per-family data needed for global scheduling
  struct FamData {
    size_t C;
    std::vector<std::vector<int64_t>> children;
    std::vector<std::vector<int64_t>> parents_of;
    std::vector<int32_t> remaining;   // children not yet scheduled
    std::vector<int64_t> split_counts;
    std::vector<int32_t> bl;          // bottom-level (priority key)
  };
  std::vector<FamData> fam_data(n_fam);

  // Parse all families in parallel
  #pragma omp parallel for schedule(dynamic, 1)
  for (int fi = 0; fi < n_fam; ++fi) {
    std::vector<std::string> leaf_names;
    std::unordered_map<std::string, int> leaf_to_index;
    CladeData clade_data = amalgamate_clades_and_splits(
        fam_vec[fi].paths, leaf_names, leaf_to_index);
    CCPArrays ccp = build_ccp_arrays(clade_data);

    auto &fd = fam_data[fi];
    fd.C = clade_data.clades.size();
    const size_t C = fd.C;
    const size_t N = ccp.split_parents_sorted.size();

    fd.children.resize(C);
    fd.parents_of.resize(C);
    fd.remaining.assign(C, 0);
    fd.split_counts.assign(ccp.split_counts.begin(), ccp.split_counts.end());

    // Build adjacency
    std::vector<std::unordered_set<int64_t>> child_sets(C);
    for (size_t i = 0; i < N; ++i) {
      int64_t p = ccp.split_parents_sorted[i];
      int64_t l = ccp.split_lefts_sorted[i];
      int64_t r = ccp.split_rights_sorted[i];
      if (child_sets[p].insert(l).second) {
        fd.children[p].push_back(l);
        fd.parents_of[l].push_back(p);
        fd.remaining[p]++;
      }
      if (l != r && child_sets[p].insert(r).second) {
        fd.children[p].push_back(r);
        fd.parents_of[r].push_back(p);
        fd.remaining[p]++;
      }
    }

    // Compute bottom-level for priority
    auto [bfs_level, n_waves] = compute_clade_waves(ccp, C);
    int32_t max_level = n_waves - 1;
    fd.bl.resize(C);
    for (size_t c = 0; c < C; ++c)
      fd.bl[c] = max_level - bfs_level[c];
  }

  // Global priority queue: (bl, family_index, clade_id)
  using Entry = std::tuple<int32_t, int, int64_t>;
  std::priority_queue<Entry> ready;

  for (int fi = 0; fi < n_fam; ++fi) {
    for (size_t c = 0; c < fam_data[fi].C; ++c) {
      if (fam_data[fi].remaining[c] == 0) {
        ready.push({fam_data[fi].bl[c], fi, static_cast<int64_t>(c)});
      }
    }
  }

  // Greedy global batching
  struct Row {
    int batch;
    int64_t n_clades, n_splits, n_families, required_clades;
  };
  std::vector<Row> rows;
  int batch_idx = 0;

  while (!ready.empty()) {
    // Pop up to max_wave_size ready clades from any family
    std::vector<std::pair<int, int64_t>> batch;  // (family, clade)
    batch.reserve(max_wave_size);
    while (!ready.empty() && static_cast<int>(batch.size()) < max_wave_size) {
      auto top = ready.top();
      ready.pop();
      batch.push_back({std::get<1>(top), std::get<2>(top)});
    }

    // Compute stats
    int64_t n_splits = 0;
    std::unordered_set<int> families_in_batch;

    // Build per-family set of batch clades (for intra-batch child exclusion)
    std::unordered_map<int, std::unordered_set<int64_t>> batch_by_fam;
    for (auto &[f, c] : batch) {
      n_splits += fam_data[f].split_counts[c];
      families_in_batch.insert(f);
      batch_by_fam[f].insert(c);
    }

    // required_clades = distinct (family, child) pairs from earlier batches
    int64_t required = 0;
    // Use a flat counter: for each clade in the batch, count its children
    // not in this batch.  Since children are per-family, use the per-family set.
    // To count distinct, use a set of (f*max_C + child) — but max_C varies.
    // Simpler: use a set<pair<int,int64_t>>.
    std::set<std::pair<int, int64_t>> needed;
    for (auto &[f, c] : batch) {
      for (int64_t child : fam_data[f].children[c]) {
        if (batch_by_fam[f].count(child) == 0) {
          needed.insert({f, child});
        }
      }
    }
    required = static_cast<int64_t>(needed.size());

    rows.push_back({batch_idx, static_cast<int64_t>(batch.size()),
                     n_splits, static_cast<int64_t>(families_in_batch.size()),
                     required});

    // Update parents — newly-ready clades enter the global queue
    for (auto &[f, c] : batch) {
      for (int64_t p : fam_data[f].parents_of[c]) {
        if (--fam_data[f].remaining[p] == 0) {
          ready.push({fam_data[f].bl[p], f, p});
        }
      }
    }
    ++batch_idx;
  }

  // Convert to Python
  py::list result;
  for (const auto &r : rows) {
    py::dict row;
    row["batch"]           = r.batch;
    row["n_clades"]        = r.n_clades;
    row["n_splits"]        = r.n_splits;
    row["n_families"]      = r.n_families;
    row["required_clades"] = r.required_clades;
    result.append(row);
  }
  return result;
}

} // namespace

PYBIND11_MODULE(preprocess_cpp, m) {
  m.def("preprocess", &preprocess, "Preprocess species and gene trees");
  m.def("preprocess_multiple_families", &preprocess_multiple_families,
        py::arg("species_path"),
        py::arg("families"),
        py::arg("include_details") = false,
        "Preprocess multiple gene families with shared species tree. Defaults to light output; pass include_details=True for full debug fields.");
  m.def("compute_phased_waves", &compute_phased_waves,
        "Three-phase scheduler returning actual wave assignments (single family)");
  m.def("compute_wave_stats", &compute_wave_stats,
        "Compute wave scheduling stats for gene families (no species tree needed)");
  m.def("compute_packet_wave_stats", &compute_packet_wave_stats,
        "Packet-aware highest-label scheduler (oriented-edge structure)");
  m.def("compute_phased_wave_stats", &compute_phased_wave_stats,
        "Three-phase scheduler: leaves / internal (split>=1) / root (per family)");
  m.def("compute_phased_cross_family_wave_stats",
        &compute_phased_cross_family_wave_stats,
        "Three-phase scheduler with cross-family global batching");
  m.def("compute_cross_family_wave_stats", &compute_cross_family_wave_stats,
        "Compute wave stats with cross-family batching (global schedule)");
  m.def("bench_parse", [](
      const std::map<std::string, std::vector<std::string>> &families,
      bool include_inclusion) {
    namespace chr = std::chrono;
    struct FamilyEntry { std::string name; std::vector<std::string> paths; };
    std::vector<FamilyEntry> fam_vec;
    for (const auto &[name, paths] : families) fam_vec.push_back({name, paths});
    const int n_fam = static_cast<int>(fam_vec.size());

    double t_parse = 0, t_ccp = 0, t_adj = 0, t_waves = 0;

    for (int fi = 0; fi < n_fam; ++fi) {
      auto t0 = chr::high_resolution_clock::now();
      std::vector<std::string> leaf_names;
      std::unordered_map<std::string, int> leaf_to_index;
      CladeData clade_data = amalgamate_clades_and_splits(
          fam_vec[fi].paths, leaf_names, leaf_to_index);
      auto t1 = chr::high_resolution_clock::now();

      CCPArrays ccp = build_ccp_arrays(clade_data, include_inclusion);
      auto t2 = chr::high_resolution_clock::now();

      const size_t C = clade_data.clades.size();
      const size_t N = ccp.split_parents_sorted.size();
      std::vector<std::vector<int64_t>> children_adj(C);
      std::vector<std::vector<int64_t>> parents_of(C);
      std::vector<int32_t> remaining(C, 0);
      {
        std::vector<std::unordered_set<int64_t>> child_sets(C);
        for (size_t i = 0; i < N; ++i) {
          int64_t p = ccp.split_parents_sorted[i];
          int64_t l = ccp.split_lefts_sorted[i];
          int64_t r = ccp.split_rights_sorted[i];
          if (child_sets[p].insert(l).second) {
            children_adj[p].push_back(l);
            parents_of[l].push_back(p);
            remaining[p]++;
          }
          if (l != r && child_sets[p].insert(r).second) {
            children_adj[p].push_back(r);
            parents_of[r].push_back(p);
            remaining[p]++;
          }
        }
      }
      auto t3 = chr::high_resolution_clock::now();

      auto [bfs_level, n_waves] = compute_clade_waves(ccp, C);
      auto t4 = chr::high_resolution_clock::now();

      t_parse += chr::duration<double>(t1 - t0).count();
      t_ccp   += chr::duration<double>(t2 - t1).count();
      t_adj   += chr::duration<double>(t3 - t2).count();
      t_waves += chr::duration<double>(t4 - t3).count();
    }
    fprintf(stderr, "bench_parse (%d families, single-thread, include_inclusion=%s):\n",
            n_fam, include_inclusion ? "true" : "false");
    fprintf(stderr, "  amalgamate_clades_and_splits: %.3fs (%.1f ms/fam)\n", t_parse, t_parse*1000/n_fam);
    fprintf(stderr, "  build_ccp_arrays:             %.3fs (%.1f ms/fam)\n", t_ccp, t_ccp*1000/n_fam);
    fprintf(stderr, "  build adjacency:              %.3fs (%.1f ms/fam)\n", t_adj, t_adj*1000/n_fam);
    fprintf(stderr, "  compute_clade_waves:          %.3fs (%.1f ms/fam)\n", t_waves, t_waves*1000/n_fam);
    fprintf(stderr, "  TOTAL:                        %.3fs (%.1f ms/fam)\n",
            t_parse+t_ccp+t_adj+t_waves, (t_parse+t_ccp+t_adj+t_waves)*1000/n_fam);
  }, py::arg("families"), py::arg("include_inclusion") = true,
     "Benchmark parsing stages");
}
