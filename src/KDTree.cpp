#include <algorithm>

#include "KDTree.h"
#include <iostream>

KDTree2D::KDTree2D() : root(nullptr) {}

KDTree2D::~KDTree2D() {
    deleteTree(root);
}

void KDTree2D::deleteTree(Node* node) {
    if (node) {
        deleteTree(node->left);
        deleteTree(node->right);
        delete node;
    }
}

void KDTree2D::build(const std::vector<cv::Point2f>& inputPoints) {
    points = inputPoints;
    std::vector<size_t> indices(points.size());
    for (size_t i = 0; i < indices.size(); i++) {
        indices[i] = i;
    }
    root = buildTree(indices, 0, 0, points.size());
}

KDTree2D::Node* KDTree2D::buildTree(std::vector<size_t>& indices, int depth, 
                                   int start, int end) {
    if (start >= end) return nullptr;

    int axis = depth % 2;  // 0 for x, 1 for y
    int mid = (start + end) / 2;

    // Sort points based on current axis
    std::nth_element(indices.begin() + start, indices.begin() + mid,
                    indices.begin() + end,
                    [&](size_t a, size_t b) {
                        if (axis == 0) {
                            return points[a].x < points[b].x;
                        } else {
                            return points[a].y < points[b].y;
                        }
                    });

    Node* node = new Node(points[indices[mid]], indices[mid]);
    node->left = buildTree(indices, depth + 1, start, mid);
    node->right = buildTree(indices, depth + 1, mid + 1, end);
    return node;
}

std::vector<size_t> KDTree2D::radiusSearch(const cv::Point2f& target, float radius) const {
    std::vector<size_t> result;
    radiusSearchHelper(root, target, radius * radius, result, 0);
    return result;
}

void KDTree2D::radiusSearchHelper(Node* node, const cv::Point2f& target, 
                                float squaredRadius, std::vector<size_t>& result, 
                                int depth) const {
    if (!node) return;

    float dx = node->point.x - target.x;
    float dy = node->point.y - target.y;
    float dist_squared = dx * dx + dy * dy;

    if (dist_squared <= squaredRadius) {
        result.push_back(node->index);
    }

    int axis = depth % 2;
    float delta = (axis == 0) ? dx : dy;

    // Always traverse the side of the split that contains the target point
    Node* nearChild = (delta > 0) ? node->left : node->right;
    Node* farChild = (delta > 0) ? node->right : node->left;

    radiusSearchHelper(nearChild, target, squaredRadius, result, depth + 1);

    // Only traverse the other side if it could contain points within the radius
    if (delta * delta <= squaredRadius) {
        radiusSearchHelper(farChild, target, squaredRadius, result, depth + 1);
    }
}
