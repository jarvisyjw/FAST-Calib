/*
Online variant of the FAST-Calib calibration node.

Instead of reading a rosbag file, this node subscribes to live LiDAR and
camera topics (real drivers or `rosbag play`), buffers the latest messages,
and runs the standard detection + SVD pipeline each time the "~/capture"
service (std_srvs/Trigger) is called. Successful captures write the same
output files as the offline node and append to circle_center_record.txt, so
the multi-scene joint solve works unchanged.
*/

#include "qr_detect.hpp"
#include "lidar_detect.hpp"
#include "intermediate_save.hpp"
#include "CustomMsg.h"

#include <cv_bridge/cv_bridge.h>
#include <ros/master.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_srvs/Trigger.h>

#include <cmath>
#include <cstdio>
#include <deque>
#include <mutex>

class OnlineCalib
{
public:
  OnlineCalib(ros::NodeHandle &nh, const Params &params) : nh_(nh), params_(params)
  {
    // Private namespace -> service is /fast_calib_online/capture.
    ros::NodeHandle pnh("~");
    capture_server_ = pnh.advertiseService("capture", &OnlineCalib::captureCallback, this);
    // Downsampled live view of the LiDAR stream for the web UI (the vendored
    // Livox CustomMsg has no Python bindings, so we republish as PointCloud2).
    live_pub_ = pnh.advertise<sensor_msgs::PointCloud2>("live_cloud", 1);
    ROS_INFO("[online] Ready. Call the '%s' service to capture a frame pair and run calibration.",
             capture_server_.getService().c_str());
  }

private:
  // ----------------------------------------------------------- subscribers

  static std::string topicType(const std::string &topic)
  {
    ros::master::V_TopicInfo topics;
    ros::master::getTopics(topics);
    for (const auto &t : topics)
      if (t.name == topic) return t.datatype;
    return "";
  }

  // Throttled (~2 Hz) downsampled republish of the incoming LiDAR stream,
  // only while someone (the web UI) is listening.
  bool livePubDue()
  {
    if (live_pub_.getNumSubscribers() == 0) return false;
    ros::WallTime now = ros::WallTime::now();
    if ((now - last_live_pub_).toSec() < 0.5) return false;
    last_live_pub_ = now;
    return true;
  }

  void publishLiveCloud(const pcl::PointCloud<pcl::PointXYZ> &cloud,
                        const std_msgs::Header &header)
  {
    const size_t max_pts = 50000;
    size_t stride = cloud.size() / max_pts + 1;
    pcl::PointCloud<pcl::PointXYZ> sampled;
    sampled.reserve(cloud.size() / stride + 1);
    for (size_t i = 0; i < cloud.size(); i += stride)
      sampled.push_back(cloud.points[i]);
    sampled.width = sampled.size();
    sampled.height = 1;

    sensor_msgs::PointCloud2 out;
    pcl::toROSMsg(sampled, out);
    out.header = header;
    live_pub_.publish(out);
  }

  void livoxCallback(const livox_ros_driver::CustomMsg::ConstPtr &msg)
  {
    {
      std::lock_guard<std::mutex> lk(mtx_);
      livox_msg_ = msg;
      pc2_msg_.reset();
      lidar_arrival_ = ros::WallTime::now();
      ++lidar_count_;
    }
    if (!livePubDue()) return;
    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.reserve(msg->point_num);
    for (uint i = 0; i < msg->point_num; ++i)
    {
      pcl::PointXYZ p;
      p.x = msg->points[i].x;
      p.y = msg->points[i].y;
      p.z = msg->points[i].z;
      cloud.push_back(p);
    }
    publishLiveCloud(cloud, msg->header);
  }

  void pc2Callback(const sensor_msgs::PointCloud2::ConstPtr &msg)
  {
    {
      std::lock_guard<std::mutex> lk(mtx_);
      pc2_msg_ = msg;
      livox_msg_.reset();
      lidar_arrival_ = ros::WallTime::now();
      ++lidar_count_;
    }
    if (!livePubDue()) return;
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromROSMsg(*msg, cloud);
    publishLiveCloud(cloud, msg->header);
  }

  void imageCallback(const sensor_msgs::Image::ConstPtr &msg)
  {
    std::lock_guard<std::mutex> lk(mtx_);
    image_history_.push_back({ros::WallTime::now(), msg, nullptr});
    if (image_history_.size() > 80) image_history_.pop_front();  // ~3 s @ 24 Hz: covers the accumulation window
    ++image_count_;
  }

  void compressedImageCallback(const sensor_msgs::CompressedImage::ConstPtr &msg)
  {
    std::lock_guard<std::mutex> lk(mtx_);
    image_history_.push_back({ros::WallTime::now(), nullptr, msg});
    if (image_history_.size() > 80) image_history_.pop_front();  // ~3 s @ 24 Hz: covers the accumulation window
    ++image_count_;
  }

  // Wait until both topics are published, then (re)subscribe with the
  // detected message types.
  bool ensureSubscriptions(const ros::WallTime &deadline, std::string &err)
  {
    while (ros::ok() && ros::WallTime::now() < deadline)
    {
      std::string lidar_type = topicType(params_.lidar_topic);
      std::string camera_type = topicType(params_.camera_topic);
      if (!lidar_type.empty() && !camera_type.empty())
      {
        if (lidar_type != lidar_sub_type_)
        {
          lidar_sub_.shutdown();
          if (lidar_type == "livox_ros_driver/CustomMsg" ||
              lidar_type == "livox_ros_driver2/CustomMsg")
            lidar_sub_ = nh_.subscribe(params_.lidar_topic, 10, &OnlineCalib::livoxCallback, this);
          else if (lidar_type == "sensor_msgs/PointCloud2")
            lidar_sub_ = nh_.subscribe(params_.lidar_topic, 10, &OnlineCalib::pc2Callback, this);
          else
          {
            err = "unsupported LiDAR message type: " + lidar_type;
            return false;
          }
          lidar_sub_type_ = lidar_type;
          ROS_INFO("[online] Subscribed to %s (%s)", params_.lidar_topic.c_str(), lidar_type.c_str());
        }
        if (camera_type != camera_sub_type_)
        {
          camera_sub_.shutdown();
          if (camera_type == "sensor_msgs/Image")
            camera_sub_ = nh_.subscribe(params_.camera_topic, 10, &OnlineCalib::imageCallback, this);
          else if (camera_type == "sensor_msgs/CompressedImage")
            camera_sub_ = nh_.subscribe(params_.camera_topic, 10, &OnlineCalib::compressedImageCallback, this);
          else
          {
            err = "unsupported camera message type: " + camera_type;
            return false;
          }
          camera_sub_type_ = camera_type;
          ROS_INFO("[online] Subscribed to %s (%s)", params_.camera_topic.c_str(), camera_type.c_str());
        }
        return true;
      }
      ROS_INFO_THROTTLE(2.0, "[online] Waiting for topics %s (%s) and %s (%s) ...",
                        params_.lidar_topic.c_str(), lidar_type.empty() ? "down" : "up",
                        params_.camera_topic.c_str(), camera_type.empty() ? "down" : "up");
      ros::WallDuration(0.2).sleep();
    }
    err = "timed out waiting for topics " + params_.lidar_topic + " and " +
          params_.camera_topic + " (start the drivers or `rosbag play`)";
    return false;
  }

  // -------------------------------------------------------------- capture

  bool captureCallback(std_srvs::Trigger::Request &, std_srvs::Trigger::Response &res)
  {
    // Re-read params so GUI edits (topics, intrinsics, filter box) apply
    // without restarting the node.
    params_ = loadParameters(nh_);

    const double timeout = 10.0;
    const double sync_tol = params_.sync_tolerance;
    ros::WallTime deadline = ros::WallTime::now() + ros::WallDuration(timeout);

    std::string err;
    if (!ensureSubscriptions(deadline, err))
    {
      res.success = false;
      res.message = err;
      return true;
    }

    uint64_t lidar_count_seen, image_count_seen;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      lidar_count_seen = lidar_count_;
      image_count_seen = image_count_;
    }

    // Accumulate `accumulate_frames` consecutive LiDAR frames into ONE cloud.
    // The offline pipeline (data_preprocess.hpp) works on the WHOLE bag
    // accumulated into a single dense cloud; a single 100 ms Livox frame is
    // far too sparse for the circle-detection thresholds (~700 board points
    // vs tens of thousands offline), which is why the same params fail online.
    // Keep the board still during this accumulation window.
    // The camera frame nearest to the MIDDLE of the window is paired with it.
    // Wall-time arrival stamps are used because rosbag play publishes /clock
    // (sim time jumps backward on loop) and header stamps keep recording time.
    const int accum_target =
        params_.accumulate_frames > 1 ? params_.accumulate_frames : 1;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_input(new pcl::PointCloud<pcl::PointXYZ>);
    uint64_t consumed = lidar_count_seen;
    ros::WallTime first_arrival, last_arrival;
    int accum_n = 0;

    while (ros::ok() && ros::WallTime::now() < deadline && accum_n < accum_target)
    {
      bool stalled = false;
      {
        std::lock_guard<std::mutex> lk(mtx_);
        if (lidar_count_ > consumed)
        {
          consumed = lidar_count_;
          if (livox_msg_)
          {
            cloud_input->reserve(cloud_input->size() + livox_msg_->point_num);
            for (uint i = 0; i < livox_msg_->point_num; ++i)
            {
              pcl::PointXYZ p;
              p.x = livox_msg_->points[i].x;
              p.y = livox_msg_->points[i].y;
              p.z = livox_msg_->points[i].z;
              cloud_input->points.push_back(p);
            }
          }
          else if (pc2_msg_)
          {
            pcl::PointCloud<pcl::PointXYZ> tmp;
            pcl::fromROSMsg(*pc2_msg_, tmp);
            *cloud_input += tmp;
          }
          last_arrival = lidar_arrival_;
          if (accum_n == 0) first_arrival = last_arrival;
          ++accum_n;
        }
        else if (accum_n > 0 && (ros::WallTime::now() - last_arrival).toSec() > 2.0)
          stalled = true;  // stream stopped mid-capture: proceed with what we have
      }
      if (stalled) break;
      ros::WallDuration(0.005).sleep();
    }

    uint64_t lidar_got, image_got;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      lidar_got = lidar_count_ - lidar_count_seen;
      image_got = image_count_ - image_count_seen;
    }

    if (accum_n == 0)
    {
      res.success = false;
      char msg[256];
      snprintf(msg, sizeof(msg),
               "no fresh LiDAR frame within %.0f s (received %llu msgs on %s since the "
               "call - is the driver or `rosbag play` still running?)",
               timeout, (unsigned long long)lidar_got, params_.lidar_topic.c_str());
      res.message = msg;
      ROS_WARN("[online] %s", res.message.c_str());
      return true;
    }
    if (accum_n < accum_target)
      ROS_WARN("[online] Only %d/%d LiDAR frames accumulated (stream stalled or timed out) - "
               "sparser cloud, detection may fail", accum_n, accum_target);

    // Pair the camera frame nearest to the middle of the accumulation window.
    ros::WallTime midpoint =
        first_arrival + ros::WallDuration((last_arrival - first_arrival).toSec() * 0.5);
    sensor_msgs::Image::ConstPtr img_msg;
    sensor_msgs::CompressedImage::ConstPtr cimg_msg;
    double best_dt = 1e9;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      for (const auto &entry : image_history_)
      {
        double dt = std::fabs((entry.arrival - midpoint).toSec());
        if (dt < best_dt)
        {
          best_dt = dt;
          img_msg = entry.img;
          cimg_msg = entry.cimg;
        }
      }
    }
    bool have_image = img_msg || cimg_msg;
    if (!have_image || best_dt > sync_tol)
    {
      res.success = false;
      char msg[320];
      if (image_got == 0)
        snprintf(msg, sizeof(msg),
                 "no camera frame received on %s since the call - check the camera "
                 "topic name and that images are being published",
                 params_.camera_topic.c_str());
      else
        snprintf(msg, sizeof(msg),
                 "no camera frame within +/-%.2f s of the capture window "
                 "(%llu camera msgs since the call, closest was %.2f s away - "
                 "streams lagging? try a slower `rosbag play -r 0.5`)",
                 sync_tol, (unsigned long long)image_got, best_dt);
      res.message = msg;
      ROS_WARN("[online] %s", res.message.c_str());
      return true;
    }

    ROS_INFO("[online] Accumulated %d LiDAR frames, paired camera frame |dt| = %.0f ms",
             accum_n, best_dt * 1000.0);

    // Convert the paired camera image (cloud_input was accumulated above).
    cv::Mat img_input;
    try
    {
      if (img_msg)
        img_input = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8)->image;
      else
        img_input = cv_bridge::toCvCopy(cimg_msg, sensor_msgs::image_encodings::BGR8)->image;
    }
    catch (cv_bridge::Exception &e)
    {
      res.success = false;
      res.message = std::string("cv_bridge exception: ") + e.what();
      return true;
    }

    ROS_INFO("[online] Captured %ld points (%d LiDAR frames) + %dx%d image, running calibration...",
             (long)cloud_input->size(), accum_n, img_input.cols, img_input.rows);

    // Same pipeline as the offline node (src/main.cpp).
    QRDetectPtr qrDetectPtr(new QRDetect(nh_, params_));
    LidarDetectPtr lidarDetectPtr(new LidarDetect(nh_, params_));

    PointCloud<PointXYZ>::Ptr qr_center_cloud(new PointCloud<PointXYZ>);
    qr_center_cloud->reserve(4);
    qrDetectPtr->detect_qr(img_input, qr_center_cloud);

    PointCloud<PointXYZ>::Ptr lidar_center_cloud(new PointCloud<PointXYZ>);
    lidar_center_cloud->reserve(4);
    lidarDetectPtr->detect_lidar(cloud_input, lidar_center_cloud);

    PointCloud<PointXYZ>::Ptr qr_centers(new PointCloud<PointXYZ>);
    PointCloud<PointXYZ>::Ptr lidar_centers(new PointCloud<PointXYZ>);
    sortPatternCenters(qr_center_cloud, qr_centers, "camera");
    sortPatternCenters(lidar_center_cloud, lidar_centers, "lidar");

    if (qr_centers->size() != TARGET_NUM_CIRCLES || lidar_centers->size() != TARGET_NUM_CIRCLES)
    {
      // Report how much of the cloud survived the distance filter so the user
      // can tell "crop box wrong" (0 pts) from "plane/circle fitting failed".
      size_t filtered_n = lidarDetectPtr->getFilteredCloud()
                              ? lidarDetectPtr->getFilteredCloud()->size()
                              : 0;
      res.success = false;
      char msg[384];
      snprintf(msg, sizeof(msg),
               "detection failed (qr centers: %zu, lidar centers: %zu, need 4+4; "
               "filtered cloud: %zu pts%s) - frame NOT recorded",
               qr_centers->size(), lidar_centers->size(), filtered_n,
               filtered_n == 0 ? " - distance filter crop box misses the board, widen "
                                 "x/y/z min/max in the Distance filter panel"
                               : "");
      res.message = msg;
      ROS_WARN("[online] %s", res.message.c_str());
      // Dump the pipeline intermediates anyway so the Layers panel can show
      // what the filter/plane/edge stages produced for this failed frame.
      if (params_.save_intermediate)
      {
        saveIntermediateClouds(params_, cloud_input,
                               lidarDetectPtr->getFilteredCloud(),
                               lidarDetectPtr->getPlaneCloud(),
                               lidarDetectPtr->getAlignedCloud(),
                               lidarDetectPtr->getEdgeCloud(),
                               lidar_centers, qr_centers);
      }
      return true;
    }

    // Only successful captures enter the multi-scene record.
    saveTargetHoleCenters(lidar_centers, qr_centers, params_);

    Eigen::Matrix4f transformation;
    pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> svd;
    svd.estimateRigidTransformation(*lidar_centers, *qr_centers, transformation);

    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_lidar_centers(new pcl::PointCloud<pcl::PointXYZ>);
    aligned_lidar_centers->reserve(lidar_centers->size());
    alignPointCloud(lidar_centers, aligned_lidar_centers, transformation);

    double rmse = computeRMSE(qr_centers, aligned_lidar_centers);
    ROS_INFO("[online] Capture RMSE: %.4f m", rmse);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    projectPointCloudToImage(cloud_input, transformation, qrDetectPtr->cameraMatrix_,
                             qrDetectPtr->distCoeffs_, img_input, colored_cloud);

    saveCalibrationResults(params_, transformation, colored_cloud, qrDetectPtr->imageCopy_);

    if (params_.save_intermediate)
    {
      saveIntermediateClouds(params_, cloud_input,
                             lidarDetectPtr->getFilteredCloud(),
                             lidarDetectPtr->getPlaneCloud(),
                             lidarDetectPtr->getAlignedCloud(),
                             lidarDetectPtr->getEdgeCloud(),
                             lidar_centers, qr_centers);
    }

    res.success = true;
    char msg[64];
    snprintf(msg, sizeof(msg), "RMSE: %.4f m", rmse);
    res.message = msg;
    return true;
  }

  ros::NodeHandle &nh_;
  Params params_;
  ros::ServiceServer capture_server_;

  ros::Subscriber lidar_sub_, camera_sub_;
  std::string lidar_sub_type_, camera_sub_type_;
  ros::Publisher live_pub_;
  ros::WallTime last_live_pub_;

  std::mutex mtx_;
  livox_ros_driver::CustomMsg::ConstPtr livox_msg_;
  sensor_msgs::PointCloud2::ConstPtr pc2_msg_;
  struct StampedImage
  {
    ros::WallTime arrival;
    sensor_msgs::Image::ConstPtr img;
    sensor_msgs::CompressedImage::ConstPtr cimg;
  };
  std::deque<StampedImage> image_history_;  // recent camera frames for pairing
  ros::WallTime lidar_arrival_;
  uint64_t lidar_count_ = 0, image_count_ = 0;
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "fast_calib_online");
  ros::NodeHandle nh;

  Params params = loadParameters(nh);
  OnlineCalib node(nh, params);

  // Service callback and subscription callbacks run on separate threads so
  // the capture service can spin-wait for fresh frames.
  ros::AsyncSpinner spinner(2);
  spinner.start();
  ros::waitForShutdown();
  return 0;
}
