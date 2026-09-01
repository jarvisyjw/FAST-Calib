/*
Intermediate pipeline-cloud export shared by the offline (fast_calib) and
online (fast_calib_online) nodes. Used by the web visualization platform;
enabled via the "save_intermediate" ROS param.
*/

#ifndef INTERMEDIATE_SAVE_HPP
#define INTERMEDIATE_SAVE_HPP

#include "common_lib.h"
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

inline void saveIntermediateClouds(
    const Params &params,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_input,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_filtered,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_plane,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_aligned,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_edge,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &lidar_centers,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &qr_centers)
{
  std::string dir = params.output_path;
  if (dir.back() != '/') dir += '/';

  auto saveCloud = [&dir](const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                          const std::string &name) {
    if (cloud->empty()) return;
    cloud->width = cloud->size();
    cloud->height = 1;
    cloud->is_dense = true;
    try {
      if (pcl::io::savePCDFileASCII(dir + name, *cloud) == 0)
        std::cout << BOLDYELLOW << "[Record] Saved " << name << " (" << cloud->size()
                  << " pts) to " << BOLDWHITE << dir << name << RESET << std::endl;
      else
        std::cerr << BOLDRED << "[Error] Failed to save " << dir + name << RESET << std::endl;
    } catch (const std::exception &e) {
      std::cerr << BOLDRED << "[Error] Failed to save " << dir + name << ": " << e.what()
                << RESET << std::endl;
    }
  };

  // Drop non-finite points, then downsample the raw input cloud to keep the
  // file size manageable.
  pcl::PointCloud<pcl::PointXYZ>::Ptr input_clean(new pcl::PointCloud<pcl::PointXYZ>);
  input_clean->reserve(cloud_input->size());
  for (const auto &pt : cloud_input->points) {
    if (std::isfinite(pt.x) && std::isfinite(pt.y) && std::isfinite(pt.z))
      input_clean->push_back(pt);
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr input_down(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::VoxelGrid<pcl::PointXYZ> voxel;
  voxel.setInputCloud(input_clean);
  voxel.setLeafSize(0.05f, 0.05f, 0.05f);
  voxel.filter(*input_down);

  saveCloud(input_down, "input_cloud.pcd");
  saveCloud(cloud_filtered, "filtered_cloud.pcd");
  saveCloud(cloud_plane, "plane_cloud.pcd");
  saveCloud(cloud_aligned, "aligned_cloud.pcd");
  saveCloud(cloud_edge, "edge_cloud.pcd");
  saveCloud(lidar_centers, "lidar_centers.pcd");
  saveCloud(qr_centers, "qr_centers.pcd");
}

#endif
