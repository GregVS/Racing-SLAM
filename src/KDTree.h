#pragma once

#include <vector>
#include <opencv2/core.hpp>

namespace slam
{

class KDTree2D {
private:
    struct Node {
        size_t index;
        cv::Point2f point;
        Node *left;
        Node *right;

        Node(const cv::Point2f &p, size_t idx)
                : point(p)
                , index(idx)
                , left(nullptr)
                , right(nullptr)
        {
        }
    };

    Node *root;
    std::vector<cv::Point2f> points;

    Node *build_tree(std::vector<size_t> &indices, int depth, int start, int end);
    void radius_search_helper(Node *node, const cv::Point2f &target, float radius, std::vector<size_t> &result,
                              int depth) const;
    void delete_tree(Node *node);

public:
    KDTree2D();
    KDTree2D(const KDTree2D &other) = delete;
    KDTree2D(KDTree2D &&other) noexcept
    {
        root = other.root;
        points = other.points;
        other.root = nullptr;
        other.points.clear();
    }
    ~KDTree2D();

    void build(const std::vector<cv::Point2f> &points);
    std::vector<size_t> radius_search(const cv::Point2f &target, float radius) const;
};

};