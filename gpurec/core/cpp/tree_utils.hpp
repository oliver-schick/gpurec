// Tree definitions and Newick parsing utilities
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct TreeNode {
  std::string name;
  std::vector<TreeNode *> children;
  TreeNode *parent{nullptr};

  TreeNode() = default;

  // Recursively delete all children to prevent memory leaks
  ~TreeNode() {
    for (TreeNode *child : children) {
      delete child;
    }
  }

  // Prevent copying (raw pointer ownership makes copying unsafe)
  TreeNode(const TreeNode&) = delete;
  TreeNode& operator=(const TreeNode&) = delete;
};

class NewickParser {
public:
  explicit NewickParser(const std::string &text);
  TreeNode *parse();
  // Parse ONE tree from the current position, consume a trailing ';' if present,
  // and leave pos_ at the start of the next tree (no trailing-character check).
  TreeNode *parse_one();
  // True if only whitespace remains (no further tree to parse).
  bool at_end();

private:
  TreeNode *parse_subtree();
  void parse_leaf_label(TreeNode *node);
  void parse_node_label(TreeNode *node);
  void skip_branch_length();
  void skip_whitespace();
  static void trim(std::string &s);

  const std::string text_;
  size_t pos_;
};

// Parse a Newick file into a TreeNode structure (owned by unique_ptr)
std::unique_ptr<TreeNode> parse_newick_file(const std::string &path);

// Parse ALL Newick trees from a file (a multi-tree sample, e.g. a UFBoot
// bootstrap distribution with one ';'-terminated tree per line). Returns one
// TreeNode per tree. Honors the GPUREC_MAX_TREES_PER_FAMILY env var (cap on the
// number of trees read; 0/unset = read all). A single-tree file yields one tree,
// so this is a drop-in superset of parse_newick_file.
std::vector<std::unique_ptr<TreeNode>>
parse_newick_file_all(const std::string &path);

// Post-order traversal collection
void collect_nodes_postorder(TreeNode *node, std::vector<TreeNode *> &order);

// Gene leaf helpers
void collect_leaf_names(TreeNode *node, std::vector<std::string> &leaf_names,
                        std::unordered_map<std::string, int> &leaf_to_idx);

// Species helpers
struct SpeciesData {
  int S;
  std::vector<std::string> names;
  std::vector<std::vector<int>> children;
};

void enumerate_species(TreeNode *root, std::vector<TreeNode *> &order,
                       SpeciesData &out);

std::unordered_map<std::string, int>
build_species_name_map(const SpeciesData &species);

