#include "tree_utils.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

NewickParser::NewickParser(const std::string &text) : text_(text), pos_(0) {}

TreeNode *NewickParser::parse() {
  TreeNode *root = parse_subtree();
  skip_whitespace();
  if (pos_ < text_.size() && text_[pos_] == ';') {
    ++pos_;
  }
  skip_whitespace();
  if (pos_ != text_.size()) {
    throw std::runtime_error("Unexpected trailing characters in Newick string");
  }
  return root;
}

TreeNode *NewickParser::parse_subtree() {
  skip_whitespace();
  TreeNode *node = new TreeNode();
  if (pos_ >= text_.size()) {
    throw std::runtime_error("Unexpected end of Newick string");
  }
  if (text_[pos_] == '(') {
    ++pos_;
    while (true) {
      TreeNode *child = parse_subtree();
      child->parent = node;
      node->children.push_back(child);
      skip_whitespace();
      if (pos_ >= text_.size()) {
        throw std::runtime_error("Unexpected end while parsing children");
      }
      char c = text_[pos_];
      if (c == ',') {
        ++pos_;
        continue;
      }
      if (c == ')') {
        ++pos_;
        break;
      }
      throw std::runtime_error("Expected ',' or ')' in Newick string");
    }
    parse_node_label(node);
  } else {
    parse_leaf_label(node);
  }
  skip_branch_length();
  return node;
}

void NewickParser::parse_leaf_label(TreeNode *node) {
  size_t start = pos_;
  while (pos_ < text_.size()) {
    char c = text_[pos_];
    if (c == ':' || c == ',' || c == ')' || c == '(' || c == ';') {
      break;
    }
    ++pos_;
  }
  node->name = text_.substr(start, pos_ - start);
  trim(node->name);
}

void NewickParser::parse_node_label(TreeNode *node) {
  skip_whitespace();
  size_t start = pos_;
  while (pos_ < text_.size()) {
    char c = text_[pos_];
    if (c == ':' || c == ',' || c == ')' || c == '(' || c == ';') {
      break;
    }
    ++pos_;
  }
  node->name = text_.substr(start, pos_ - start);
  trim(node->name);
}

void NewickParser::skip_branch_length() {
  skip_whitespace();
  if (pos_ < text_.size() && text_[pos_] == ':') {
    ++pos_;
    while (pos_ < text_.size()) {
      char c = text_[pos_];
      if (std::isdigit(static_cast<unsigned char>(c)) || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') {
        ++pos_;
      } else {
        break;
      }
    }
  }
}

void NewickParser::skip_whitespace() {
  while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) {
    ++pos_;
  }
}

void NewickParser::trim(std::string &s) {
  auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
  auto begin = std::find_if(s.begin(), s.end(), not_space);
  auto end = std::find_if(s.rbegin(), s.rend(), not_space).base();
  if (begin >= end) {
    s.clear();
  } else {
    s = std::string(begin, end);
  }
}

std::unique_ptr<TreeNode> parse_newick_file(const std::string &path) {
  std::ifstream f(path);
  if (!f) {
    throw std::runtime_error("Unable to open Newick file: " + path);
  }
  std::ostringstream buffer;
  buffer << f.rdbuf();
  std::string text = buffer.str();
  NewickParser parser(text);
  return std::unique_ptr<TreeNode>(parser.parse());
}

void collect_nodes_postorder(TreeNode *node, std::vector<TreeNode *> &order) {
  for (TreeNode *child : node->children) {
    collect_nodes_postorder(child, order);
  }
  order.push_back(node);
}

void collect_leaf_names(TreeNode *node, std::vector<std::string> &leaf_names,
                        std::unordered_map<std::string, int> &leaf_to_idx) {
  if (node->children.empty()) {
    auto it = leaf_to_idx.find(node->name);
    if (it == leaf_to_idx.end()) {
      int idx = static_cast<int>(leaf_names.size());
      leaf_to_idx[node->name] = idx;
      leaf_names.push_back(node->name);
    }
    return;
  }
  for (TreeNode *child : node->children) {
    collect_leaf_names(child, leaf_names, leaf_to_idx);
  }
}

void enumerate_species(TreeNode *root, std::vector<TreeNode *> &order, SpeciesData &out) {
  collect_nodes_postorder(root, order);
  int S = static_cast<int>(order.size());
  std::unordered_map<TreeNode *, int> index;
  out.names.resize(S);
  out.children.resize(S);
  for (int i = 0; i < S; ++i) {
    index[order[i]] = i;
  }
  for (int i = 0; i < S; ++i) {
    TreeNode *node = order[i];
    out.names[i] = node->name;
    for (TreeNode *child : node->children) {
      out.children[i].push_back(index[child]);
    }
  }
  out.S = S;
}

std::unordered_map<std::string, int> build_species_name_map(const SpeciesData &species) {
  std::unordered_map<std::string, int> mapping;
  for (int i = 0; i < species.S; ++i) {
    if (!species.names[i].empty()) {
      mapping[species.names[i]] = i;
    }
  }
  return mapping;
}

