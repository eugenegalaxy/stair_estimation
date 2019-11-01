#include <ros/ros.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>


using PointT =  pcl::PointXYZRGB;

pcl::visualization::PCLVisualizer::Ptr visualiseCloud (
    pcl::PointCloud<PointT>::ConstPtr cloud,  std::string file_name)
{
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer (file_name));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud<PointT> (cloud, "Point Cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "Point Cloud");
  viewer->addCoordinateSystem (0.5);
  viewer->initCameraParameters ();
  return (viewer);
}

int openPCD(int argc, char** argv, pcl::PointCloud<PointT>::Ptr &input_cloud){

std::string folder_name = "/home/eugeneswag/ros_workspace/src/stair_estimation/src/pcd/";
  if (pcl::io::loadPCDFile<PointT>(folder_name + argv[1], *input_cloud) == -1){
    return 0;
  }
  std::cout << endl;
  std::cout << "Loaded " << input_cloud->width * input_cloud->height  << " points"
            << "(" << input_cloud->width << " x " << input_cloud->height << ")" << std::endl;
  return 1;
}

int main (int argc, char** argv)
{
  ros::init (argc, argv, "pcl_visualisation", ros::init_options::AnonymousName);
  ros::NodeHandle nh;
  ros::Rate loop_rate(4);

  pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
  openPCD(argc, argv, cloud);
  std::string file_name = argv[1];
  pcl::visualization::PCLVisualizer::Ptr viewer;
  viewer = visualiseCloud(cloud, file_name);
  while (!viewer->wasStopped ()){
    viewer->spinOnce (100);
    loop_rate.sleep ();
  }
}