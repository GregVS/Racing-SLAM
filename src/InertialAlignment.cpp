#include "InertialAlignment.h"

#include <cmath>

namespace slam::imu {

namespace {

/** Two unit vectors spanning the plane orthogonal to direction - VINS-Mono inspired */
Eigen::Matrix<double, 3, 2> tangent_basis(const Eigen::Vector3d& direction)
{
    const Eigen::Vector3d seed = std::abs(direction.x()) > 0.9 ? Eigen::Vector3d::UnitY() : Eigen::Vector3d::UnitX();
    Eigen::Matrix<double, 3, 2> basis;
    basis.col(0) = (seed - direction * direction.dot(seed)).normalized();
    basis.col(1) = direction.cross(basis.col(0));
    return basis;
}

} // namespace

Alignment align(const std::vector<Eigen::Matrix3d>& rotations,
                const std::vector<Eigen::Vector3d>& positions,
                const std::vector<double>& times,
                const std::vector<Preintegrated>& summaries)
{
    Alignment alignment;
    const size_t count = positions.size();
    if (count < 3 || rotations.size() != count || times.size() != count || summaries.size() + 1 != count) {
        return alignment;
    }

    // Only estimate the scale and gravity, velocities are computed later
    Eigen::MatrixXd system(3 * (count - 2), 4);
    Eigen::VectorXd target(3 * (count - 2));
    std::vector<double> gravity_coefficient(count - 2, 0.0);
    size_t row = 0;
    for (size_t k = 0; k + 2 < count; k++) {
        const double first = times[k + 1] - times[k];
        const double second = times[k + 2] - times[k + 1];
        if (first <= 0.0 || second <= 0.0) {
            return alignment;
        }

        system.block<3, 1>(3 * row, 0) =
            second * (positions[k + 1] - positions[k]) - first * (positions[k + 2] - positions[k + 1]);
        gravity_coefficient[row] = 0.5 * first * second * (first + second);
        system.block<3, 3>(3 * row, 1) = gravity_coefficient[row] * Eigen::Matrix3d::Identity();
        target.segment<3>(3 * row) = second * rotations[k] * summaries[k].position -
                                     first * second * rotations[k] * summaries[k].velocity -
                                     first * rotations[k + 1] * summaries[k + 1].position;

        row++;
    }

    Eigen::Vector4d solution;
    Eigen::Matrix3d constrained_normal = Eigen::Matrix3d::Zero();
    {
        const Eigen::JacobiSVD<Eigen::MatrixXd> decomposition(system, Eigen::ComputeThinU | Eigen::ComputeThinV);
        solution = decomposition.solve(target);
        if (!solution.allFinite() || solution.tail<3>().norm() <= 0.0) {
            return alignment;
        }

        const double magnitude = GRAVITY_ENU.norm();
        Eigen::Vector3d direction = solution.tail<3>().normalized();
        Eigen::MatrixXd reduced(3 * row, 3);
        Eigen::VectorXd shifted(3 * row);
        reduced.col(0) = system.col(0);
        for (int pass = 0; pass < 4; pass++) {
            const Eigen::Matrix<double, 3, 2> basis = tangent_basis(direction);
            for (size_t k = 0; k < row; k++) {
                reduced.block<3, 2>(3 * k, 1) = gravity_coefficient[k] * basis;
                shifted.segment<3>(3 * k) = target.segment<3>(3 * k) - gravity_coefficient[k] * magnitude * direction;
            }
            const Eigen::Matrix3d normal = reduced.transpose() * reduced;
            const Eigen::Vector3d step = normal.ldlt().solve(reduced.transpose() * shifted);
            if (!step.allFinite()) {
                return alignment;
            }
            solution[0] = step[0];
            direction = (magnitude * direction + basis * step.tail<2>()).normalized();
            constrained_normal = normal;
        }
        solution.tail<3>() = magnitude * direction;
    }

    if (!solution.allFinite()) {
        return alignment;
    }

    alignment.scale = solution[0];
    alignment.gravity = solution.tail<3>();
    alignment.gravity_magnitude_error = alignment.gravity.norm() - GRAVITY_ENU.norm();
    alignment.triples = row;
    const size_t unknowns = 3;
    const size_t freedom = 3 * row > unknowns ? 3 * row - unknowns : 1;
    const double deviation = (system * solution - target).norm() / std::sqrt(static_cast<double>(freedom));
    const Eigen::VectorXd misfit = system * solution - target;
    double unexplained = 0.0;
    for (size_t k = 0; k < row; k++) {
        unexplained += misfit.segment<3>(3 * k).squaredNorm() / (gravity_coefficient[k] * gravity_coefficient[k]);
    }
    alignment.residual = std::sqrt(unexplained / static_cast<double>(3 * row));

    {
        const Eigen::Matrix3d covariance = constrained_normal.inverse();
        if (covariance.allFinite() && covariance(0, 0) > 0.0) {
            alignment.scale_uncertainty = deviation * std::sqrt(covariance(0, 0)) / std::abs(alignment.scale);
        }
        const double tangential = covariance.bottomRightCorner<2, 2>().trace() / 2.0;
        if (std::isfinite(tangential) && tangential > 0.0) {
            alignment.gravity_uncertainty = deviation * std::sqrt(tangential) / GRAVITY_ENU.norm();
        }
    }

    if (alignment.scale <= 0.0) {
        return alignment;
    }
    alignment.valid = true;

    // Compute velocities from positions, scale, and gravity
    alignment.velocities.resize(count);
    for (size_t k = 0; k + 1 < count; k++) {
        const double interval = times[k + 1] - times[k];
        alignment.velocities[k] =
            (alignment.scale * (positions[k + 1] - positions[k]) - 0.5 * alignment.gravity * interval * interval -
             rotations[k] * summaries[k].position) /
            interval;
    }
    alignment.velocities.back() = alignment.velocities[count - 2] +
                                  alignment.gravity * (times[count - 1] - times[count - 2]) +
                                  rotations[count - 2] * summaries[count - 2].velocity;
    return alignment;
}

} // namespace slam::imu
