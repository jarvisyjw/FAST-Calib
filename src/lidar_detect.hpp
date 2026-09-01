/* 
Developer: Chunran Zheng <zhengcr@connect.hku.hk>

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef LIDAR_DETECT_HPP
#define LIDAR_DETECT_HPP
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PointStamped.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <pcl/filters/voxel_grid.h>
#include <cfloat>
#include <cstdint>
#include <queue>
#include <vector>
#include "common_lib.h"

class LidarDetect
{
private:
    double x_min_, x_max_, y_min_, y_max_, z_min_, z_max_;
    double circle_radius_;

    // 存储中间结果的点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr edge_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr center_z0_cloud_;

public:
    ros::Publisher filtered_pub_;
    ros::Publisher plane_pub_;
    ros::Publisher aligned_pub_;
    ros::Publisher edge_pub_;
    ros::Publisher center_z0_pub_;
    ros::Publisher center_pub_;

    LidarDetect(ros::NodeHandle &nh, Params &params)
        : filtered_cloud_(new pcl::PointCloud<pcl::PointXYZ>),
          plane_cloud_(new pcl::PointCloud<pcl::PointXYZ>),
          aligned_cloud_(new pcl::PointCloud<pcl::PointXYZ>),
          edge_cloud_(new pcl::PointCloud<pcl::PointXYZ>),
          center_z0_cloud_(new pcl::PointCloud<pcl::PointXYZ>)
    {
        x_min_ = params.x_min;
        x_max_ = params.x_max;
        y_min_ = params.y_min;
        y_max_ = params.y_max;
        z_min_ = params.z_min;
        z_max_ = params.z_max;
        circle_radius_ = params.circle_radius;

        filtered_pub_ = nh.advertise<sensor_msgs::PointCloud2>("filtered_cloud", 1);
        plane_pub_ = nh.advertise<sensor_msgs::PointCloud2>("plane_cloud", 1);
        aligned_pub_ = nh.advertise<sensor_msgs::PointCloud2>("aligned_cloud", 1);
        edge_pub_ = nh.advertise<sensor_msgs::PointCloud2>("edge_cloud", 1);
        center_z0_pub_ = nh.advertise<sensor_msgs::PointCloud2>("center_z0_cloud", 10);
        center_pub_ = nh.advertise<sensor_msgs::PointCloud2>("center_cloud", 10);
    }

    void detect_lidar(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr center_cloud)
    {
        // 1. X、Y、Z方向滤波
        filtered_cloud_->reserve(cloud->size());

        pcl::PassThrough<pcl::PointXYZ> pass_x;
        pass_x.setInputCloud(cloud);
        pass_x.setFilterFieldName("x");
        pass_x.setFilterLimits(x_min_, x_max_);  // 设置X轴范围
        pass_x.filter(*filtered_cloud_);
    
        pcl::PassThrough<pcl::PointXYZ> pass_y;
        pass_y.setInputCloud(filtered_cloud_);
        pass_y.setFilterFieldName("y");
        pass_y.setFilterLimits(y_min_, y_max_);  // 设置Y轴范围
        pass_y.filter(*filtered_cloud_);
    
        pcl::PassThrough<pcl::PointXYZ> pass_z;
        pass_z.setInputCloud(filtered_cloud_);
        pass_z.setFilterFieldName("z");
        pass_z.setFilterLimits(z_min_, z_max_);  // 设置Z轴范围
        pass_z.filter(*filtered_cloud_);
    
        ROS_INFO("Filtered cloud size: %ld", filtered_cloud_->size());
        
        pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
        voxel_filter.setInputCloud(filtered_cloud_);
        voxel_filter.setLeafSize(0.005f, 0.005f, 0.005f);
        voxel_filter.filter(*filtered_cloud_);
        ROS_INFO("Filtered cloud size: %ld", filtered_cloud_->size());

        // 2. 平面分割
        plane_cloud_->reserve(filtered_cloud_->size());

        pcl::ModelCoefficients::Ptr plane_coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr plane_inliers(new pcl::PointIndices);
        pcl::SACSegmentation<pcl::PointXYZ> plane_segmentation;
        plane_segmentation.setModelType(pcl::SACMODEL_PLANE);
        plane_segmentation.setMethodType(pcl::SAC_RANSAC);
        plane_segmentation.setDistanceThreshold(0.01);  // 平面分割阈值
        plane_segmentation.setInputCloud(filtered_cloud_);
        plane_segmentation.segment(*plane_inliers, *plane_coefficients);
    
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(filtered_cloud_);
        extract.setIndices(plane_inliers);
        extract.filter(*plane_cloud_);
        ROS_INFO("Plane cloud size: %ld", plane_cloud_->size());
    
        // 3. 平面点云对齐   
        aligned_cloud_->reserve(plane_cloud_->size());

        Eigen::Vector3d normal(plane_coefficients->values[0],
            plane_coefficients->values[1],
            plane_coefficients->values[2]);
        normal.normalize();
        Eigen::Vector3d z_axis(0, 0, 1);

        Eigen::Vector3d axis = normal.cross(z_axis);
        double angle = acos(normal.dot(z_axis));

        Eigen::AngleAxisd rotation(angle, axis);
        Eigen::Matrix3d R = rotation.toRotationMatrix();

        // 应用旋转矩阵，将平面对齐到 Z=0 平面
        float average_z = 0.0;
        int cnt = 0;
        for (const auto& pt : *plane_cloud_) {
            Eigen::Vector3d point(pt.x, pt.y, pt.z);
            Eigen::Vector3d aligned_point = R * point;
            aligned_cloud_->push_back(pcl::PointXYZ(aligned_point.x(), aligned_point.y(), 0.0));
            average_z += aligned_point.z();
            cnt++;
        }
        average_z /= cnt;

        // 4. 提取边缘点
        edge_cloud_->reserve(aligned_cloud_->size());

        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        normal_estimator.setInputCloud(aligned_cloud_);
        normal_estimator.setRadiusSearch(0.03); // 设置法线估计的搜索半径
        normal_estimator.compute(*normals);
    
        pcl::PointCloud<pcl::Boundary> boundaries;
        pcl::BoundaryEstimation<pcl::PointXYZ, pcl::Normal, pcl::Boundary> boundary_estimator;
        boundary_estimator.setInputCloud(aligned_cloud_);
        boundary_estimator.setInputNormals(normals);
        boundary_estimator.setRadiusSearch(0.03); // 设置边界检测的搜索半径
        boundary_estimator.setAngleThreshold(M_PI / 4); // 设置角度阈值
        boundary_estimator.compute(boundaries);
    
        for (size_t i = 0; i < aligned_cloud_->size(); ++i) {
            if (boundaries.points[i].boundary_point > 0) {
                edge_cloud_->push_back(aligned_cloud_->points[i]);
            }
        }
        ROS_INFO("Extracted %ld edge points.", edge_cloud_->size());

        // 5. 对边缘点进行聚类
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(edge_cloud_);
    
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.02); // 设置聚类距离阈值
        ec.setMinClusterSize(50);     // 最小点数
        ec.setMaxClusterSize(1000);   // 最大点数
        ec.setSearchMethod(tree);
        ec.setInputCloud(edge_cloud_);
        ec.extract(cluster_indices);
    
        ROS_INFO("Number of edge clusters: %ld", cluster_indices.size());
    
        // 6. 对每个聚类进行圆拟合
        center_z0_cloud_->reserve(4);
        Eigen::Matrix3d R_inv = R.inverse();
    
        // 对每个聚类进行圆拟合
        for (size_t i = 0; i < cluster_indices.size(); ++i) 
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
            for (const auto& idx : cluster_indices[i].indices) {
                cluster->push_back(edge_cloud_->points[idx]);
            }
    
            // 圆拟合
            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
            pcl::SACSegmentation<pcl::PointXYZ> seg;
            seg.setOptimizeCoefficients(true);
            seg.setModelType(pcl::SACMODEL_CIRCLE2D);
            seg.setMethodType(pcl::SAC_RANSAC);
            seg.setDistanceThreshold(0.01); // 设置距离阈值
            seg.setMaxIterations(1000);     // 设置最大迭代次数
            seg.setInputCloud(cluster);
            seg.segment(*inliers, *coefficients);
    
            if (inliers->indices.size() > 0) 
            {
                // 计算拟合误差
                double error = 0.0;
                for (const auto& idx : inliers->indices) 
                {
                    double dx = cluster->points[idx].x - coefficients->values[0];
                    double dy = cluster->points[idx].y - coefficients->values[1];
                    double distance = sqrt(dx * dx + dy * dy) - circle_radius_; // 距离误差
                    error += abs(distance);
                }
                error /= inliers->indices.size();
    
                // 如果拟合误差较小，则认为是一个圆洞
                if (error < 0.025) 
                {
                    // 将恢复后的圆心坐标添加到点云中
                    pcl::PointXYZ center_point;
                    center_point.x = coefficients->values[0];
                    center_point.y = coefficients->values[1];
                    center_point.z = 0.0;
                    center_z0_cloud_->push_back(center_point);

                    // 将圆心坐标逆变换回原始坐标系
                    Eigen::Vector3d aligned_point(center_point.x, center_point.y, center_point.z + average_z);
                    Eigen::Vector3d original_point = R_inv * aligned_point;

                    pcl::PointXYZ center_point_origin;
                    center_point_origin.x = original_point.x();
                    center_point_origin.y = original_point.y();
                    center_point_origin.z = original_point.z();
                    center_cloud->points.push_back(center_point_origin);
                }
            }
        }

        // Fallback for sparse clouds (e.g. online capture of a few Livox
        // frames): the boundary estimator above needs dense, uniform sampling
        // and fails on sparse rose-pattern scans. Holes are large EMPTY
        // regions in the board plane though, so detect them on a 2D occupancy
        // grid instead.
        if (center_cloud->size() < 4)
        {
            ROS_INFO("Boundary-based detection found %zu circles, falling back to grid hole detection.",
                     center_cloud->size());
            center_cloud->clear();
            center_z0_cloud_->clear();
            detectHolesOnGrid(center_cloud, R_inv, average_z);
        }
    }

private:
    // Detect circular holes as interior empty regions of the rasterized
    // plane-aligned cloud, then fit a circle to the occupied ring of cells
    // around each region. Robust at low point density.
    void detectHolesOnGrid(pcl::PointCloud<pcl::PointXYZ>::Ptr center_cloud,
                           const Eigen::Matrix3d &R_inv, double average_z)
    {
        if (aligned_cloud_->empty()) return;
        const double RES = 0.01;          // 1 cm grid cells
        const int GAP_BRIDGE = 2;         // closing kernel radius (5x5)
        const int RIM_DIST = 2;           // ring width around a hole (cells)
        const double R_TOL = 0.04;        // |fitted r - circle_radius| gate

        float minx = FLT_MAX, miny = FLT_MAX, maxx = -FLT_MAX, maxy = -FLT_MAX;
        for (const auto &p : *aligned_cloud_)
        {
            minx = std::min(minx, p.x); miny = std::min(miny, p.y);
            maxx = std::max(maxx, p.x); maxy = std::max(maxy, p.y);
        }
        minx -= 0.05f; miny -= 0.05f; maxx += 0.05f; maxy += 0.05f;
        const int W = int((maxx - minx) / RES) + 2;
        const int H = int((maxy - miny) / RES) + 2;
        auto at = [W](int x, int y) { return y * W + x; };

        // Rasterize; remember which points land in each cell (for rim fitting)
        std::vector<uint8_t> occ(W * H, 0);
        std::vector<std::vector<int>> cell_pts(W * H);
        for (size_t i = 0; i < aligned_cloud_->size(); ++i)
        {
            const auto &p = aligned_cloud_->points[i];
            int gx = int((p.x - minx) / RES), gy = int((p.y - miny) / RES);
            if (gx < 0 || gx >= W || gy < 0 || gy >= H) continue;
            occ[at(gx, gy)] = 1;
            cell_pts[at(gx, gy)].push_back(int(i));
        }

        // Morphological closing (dilate then erode with a square kernel) to
        // bridge 1-2 cell scan gaps; the holes (~24 cm) are far larger.
        std::vector<uint8_t> closed(W * H, 0), dilated(W * H, 0);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
            {
                if (!occ[at(x, y)]) continue;
                for (int dy = -GAP_BRIDGE; dy <= GAP_BRIDGE; ++dy)
                    for (int dx = -GAP_BRIDGE; dx <= GAP_BRIDGE; ++dx)
                    {
                        int nx = x + dx, ny = y + dy;
                        if (nx >= 0 && nx < W && ny >= 0 && ny < H) dilated[at(nx, ny)] = 1;
                    }
            }
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
            {
                bool full = true;
                for (int dy = -GAP_BRIDGE; dy <= GAP_BRIDGE && full; ++dy)
                    for (int dx = -GAP_BRIDGE; dx <= GAP_BRIDGE; ++dx)
                    {
                        int nx = x + dx, ny = y + dy;
                        if (nx < 0 || nx >= W || ny < 0 || ny >= H || !dilated[at(nx, ny)])
                        { full = false; break; }
                    }
                closed[at(x, y)] = full ? 1 : 0;
            }

        // Exterior = empty cells reachable from the grid border (BFS).
        std::vector<uint8_t> exterior(W * H, 0);
        std::queue<std::pair<int, int>> q;
        for (int x = 0; x < W; ++x) { q.push({x, 0}); q.push({x, H - 1}); }
        for (int y = 0; y < H; ++y) { q.push({0, y}); q.push({W - 1, y}); }
        while (!q.empty())
        {
            int x = q.front().first, y = q.front().second; q.pop();
            if (x < 0 || x >= W || y < 0 || y >= H) continue;
            int id = at(x, y);
            if (closed[id] || exterior[id]) continue;
            exterior[id] = 1;
            q.push({x + 1, y}); q.push({x - 1, y}); q.push({x, y + 1}); q.push({x, y - 1});
        }

        // Interior empty regions = hole candidates.
        std::vector<uint8_t> seen(W * H, 0);
        const double hole_area = M_PI * circle_radius_ * circle_radius_;
        std::vector<uint8_t> rim_mask(W * H, 0);  // for the debug edge cloud
        int holes_found = 0;
        for (int y = 0; y < H && holes_found < 8; ++y)
            for (int x = 0; x < W && holes_found < 8; ++x)
            {
                int id0 = at(x, y);
                if (closed[id0] || exterior[id0] || seen[id0]) continue;
                // BFS this region
                std::vector<std::pair<int, int>> region;
                q.push({x, y});
                seen[id0] = 1;
                while (!q.empty())
                {
                    int cx = q.front().first, cy = q.front().second; q.pop();
                    region.push_back({cx, cy});
                    const int dxs[4] = {1, -1, 0, 0}, dys[4] = {0, 0, 1, -1};
                    for (int k = 0; k < 4; ++k)
                    {
                        int nx = cx + dxs[k], ny = cy + dys[k];
                        if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
                        int nid = at(nx, ny);
                        if (closed[nid] || exterior[nid] || seen[nid]) continue;
                        seen[nid] = 1;
                        q.push({nx, ny});
                    }
                }
                double area = region.size() * RES * RES;
                if (area < 0.3 * hole_area || area > 3.0 * hole_area) continue;

                // Rim: original (unclosed) occupied cells within RIM_DIST.
                std::vector<Eigen::Vector2d> rim;
                for (const auto &cell : region)
                    for (int dy = -RIM_DIST; dy <= RIM_DIST; ++dy)
                        for (int dx = -RIM_DIST; dx <= RIM_DIST; ++dx)
                        {
                            int nx = cell.first + dx, ny = cell.second + dy;
                            if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
                            int nid = at(nx, ny);
                            if (!occ[nid] || rim_mask[nid] == 2) continue;
                            rim_mask[nid] = 2;  // claimed by this hole
                            for (int pi : cell_pts[nid])
                            {
                                const auto &p = aligned_cloud_->points[pi];
                                rim.push_back({p.x, p.y});
                            }
                        }
                if (rim.size() < 10) continue;

                // Kasa algebraic circle fit: x^2+y^2 + a x + b y + c = 0
                Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
                Eigen::Vector3d v = Eigen::Vector3d::Zero();
                for (const auto &p : rim)
                {
                    Eigen::Vector3d row(p.x(), p.y(), 1.0);
                    M += row * row.transpose();
                    v += row * (-(p.x() * p.x() + p.y() * p.y()));
                }
                Eigen::Vector3d sol = M.ldlt().solve(v);
                double cx = -sol[0] / 2.0, cy = -sol[1] / 2.0;
                double r2 = (sol[0] * sol[0] + sol[1] * sol[1]) / 4.0 - sol[2];
                if (r2 <= 0) continue;
                double r = std::sqrt(r2);
                if (std::fabs(r - circle_radius_) > R_TOL) continue;

                ++holes_found;
                center_z0_cloud_->push_back(pcl::PointXYZ(cx, cy, 0.0));
                Eigen::Vector3d original_point = R_inv * Eigen::Vector3d(cx, cy, average_z);
                center_cloud->push_back(pcl::PointXYZ(original_point.x(),
                                                      original_point.y(),
                                                      original_point.z()));
                ROS_INFO("Grid hole: center=(%.3f, %.3f) r=%.3f m, area=%.4f m^2, rim=%zu pts",
                         cx, cy, r, area, rim.size());
            }

        // Show the rim points used by the grid fitter in the Layers panel.
        if (holes_found > 0)
        {
            edge_cloud_->clear();
            for (size_t i = 0; i < aligned_cloud_->size(); ++i)
            {
                const auto &p = aligned_cloud_->points[i];
                int gx = int((p.x - minx) / RES), gy = int((p.y - miny) / RES);
                if (gx < 0 || gx >= W || gy < 0 || gy >= H) continue;
                if (rim_mask[at(gx, gy)] == 2) edge_cloud_->push_back(p);
            }
        }
        ROS_INFO("Grid hole detection found %d holes.", holes_found);
    }

public:
    // 获取中间结果的点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr getFilteredCloud() const { return filtered_cloud_; }
    pcl::PointCloud<pcl::PointXYZ>::Ptr getPlaneCloud() const { return plane_cloud_; }
    pcl::PointCloud<pcl::PointXYZ>::Ptr getAlignedCloud() const { return aligned_cloud_; }
    pcl::PointCloud<pcl::PointXYZ>::Ptr getEdgeCloud() const { return edge_cloud_; }
    pcl::PointCloud<pcl::PointXYZ>::Ptr getCenterZ0Cloud() const { return center_z0_cloud_; }
};

typedef std::shared_ptr<LidarDetect> LidarDetectPtr;

#endif