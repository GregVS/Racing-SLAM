#pragma once

#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace slam {

class KDTree2D {
  private:
    struct Node {
        size_t index;
        Eigen::Vector2f point;
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;

        Node(const Eigen::Vector2f& p, size_t idx)
            : point(p), index(idx), left(nullptr), right(nullptr)
        {
        }
    };

    std::unique_ptr<Node> root;
    std::vector<Eigen::Vector2f> points;

    std::unique_ptr<Node> build_tree(std::vector<size_t>& indices, int depth, int start, int end);
    void radius_search_helper(const Node* node,
                              const Eigen::Vector2f& target,
                              float radius,
                              std::vector<size_t>& result,
                              int depth) const;

  public:
    KDTree2D() = default;

    void build(const std::vector<Eigen::Vector2f>& points);
    std::vector<size_t> radius_search(const Eigen::Vector2f& target, float radius) const;
};

}; // namespace slam