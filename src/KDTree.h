#pragma once

#include <Eigen/Dense>
#include <vector>

namespace slam {

class KDTree2D {
  private:
    struct Node {
        size_t index;
        Eigen::Vector2f point;
        Node* left;
        Node* right;

        Node(const Eigen::Vector2f& p, size_t idx)
            : point(p), index(idx), left(nullptr), right(nullptr)
        {
        }
    };

    Node* root;
    std::vector<Eigen::Vector2f> points;

    Node* build_tree(std::vector<size_t>& indices, int depth, int start, int end);
    void radius_search_helper(Node* node,
                              const Eigen::Vector2f& target,
                              float radius,
                              std::vector<size_t>& result,
                              int depth) const;
    void delete_tree(Node* node);

  public:
    KDTree2D();
    KDTree2D(const KDTree2D& other) = delete;
    KDTree2D(KDTree2D&& other) noexcept
    {
        root = other.root;
        points = other.points;
        other.root = nullptr;
        other.points.clear();
    }
    ~KDTree2D();

    void build(const std::vector<Eigen::Vector2f>& points);
    std::vector<size_t> radius_search(const Eigen::Vector2f& target, float radius) const;
};

}; // namespace slam