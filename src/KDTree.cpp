#include "KDTree.h"

#include <algorithm>
#include <iostream>

namespace slam {

void KDTree2D::build(const std::vector<Eigen::Vector2f>& inputPoints)
{
    points = inputPoints;
    std::vector<size_t> indices(points.size());
    for (size_t i = 0; i < indices.size(); i++) {
        indices[i] = i;
    }
    root = build_tree(indices, 0, 0, points.size());
}

std::unique_ptr<KDTree2D::Node>
KDTree2D::build_tree(std::vector<size_t>& indices, int depth, int start, int end)
{
    if (start >= end)
        return nullptr;

    int axis = depth % 2; // 0 for x, 1 for y
    int mid = (start + end) / 2;

    // Sort points based on current axis
    std::nth_element(indices.begin() + start,
                     indices.begin() + mid,
                     indices.begin() + end,
                     [&](size_t a, size_t b) {
                         if (axis == 0) {
                             return points[a].x() < points[b].x();
                         } else {
                             return points[a].y() < points[b].y();
                         }
                     });

    auto node = std::make_unique<Node>(points[indices[mid]], indices[mid]);
    node->left = build_tree(indices, depth + 1, start, mid);
    node->right = build_tree(indices, depth + 1, mid + 1, end);
    return node;
}

std::vector<size_t> KDTree2D::radius_search(const Eigen::Vector2f& target, float radius) const
{
    std::vector<size_t> result;
    radius_search_helper(root.get(), target, radius * radius, result, 0);
    return result;
}

void KDTree2D::radius_search_helper(const Node* node,
                                    const Eigen::Vector2f& target,
                                    float squared_radius,
                                    std::vector<size_t>& result,
                                    int depth) const
{
    if (!node)
        return;

    float dx = node->point.x() - target.x();
    float dy = node->point.y() - target.y();
    float dist_squared = dx * dx + dy * dy;

    if (dist_squared <= squared_radius) {
        result.push_back(node->index);
    }

    int axis = depth % 2;
    float delta = (axis == 0) ? dx : dy;

    // Always traverse the side of the split that contains the target point
    const Node* near_child = (delta > 0) ? node->left.get() : node->right.get();
    const Node* far_child = (delta > 0) ? node->right.get() : node->left.get();

    radius_search_helper(near_child, target, squared_radius, result, depth + 1);

    // Only traverse the other side if it could contain points within the radius
    if (delta * delta <= squared_radius) {
        radius_search_helper(far_child, target, squared_radius, result, depth + 1);
    }
}

}; // namespace slam