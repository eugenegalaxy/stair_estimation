#include <iostream>
#include <thread>
#include <string>
#include <tuple>
#include <sys/stat.h>

#include <ros/ros.h>
#include <librealsense2/rs.hpp>

#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/organized_multi_plane_segmentation.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/point_types.h>
#include <pcl/io/io.h>


#define WIDTH        640
#define HEIGHT       480 
#define FRAMERATE    30

using PointT =  pcl::PointXYZRGB;

// Global variables
int imageAmount;
std::string file_format = ".pcd";



pcl::visualization::PCLVisualizer::Ptr visualiseCloud (
    pcl::PointCloud<PointT>::ConstPtr cloud, PointT pl_Centroids[], int planes_number)
{
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud<PointT> (cloud, "Point Cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "Point Cloud");
  // viewer->addCoordinateSystem (0.5);
  viewer->initCameraParameters ();

  float sphere_radius = 0.01;
  int r(255), g(0), b(0);
  for(size_t i = 0; i < planes_number; i++){
    viewer->addSphere(pl_Centroids[i],sphere_radius, r, g, b, std::to_string(i));
  }
  return (viewer);
}


int openPCD(pcl::PointCloud<PointT>::Ptr &input_cloud, char abs_path_string[]){

  if (pcl::io::loadPCDFile<PointT>(abs_path_string, *input_cloud) == -1){
    return 0;
  }
  std::cout << endl;
  std::cout << "Loaded " << input_cloud->width * input_cloud->height  << " points"
            << "(" << input_cloud->width << " x " << input_cloud->height << ")" << std::endl;
  return 1;
}

bool FileExists(const std::string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}

void savePointCloud(std::string saveFileName, pcl::PointCloud<PointT>::Ptr &cloud_to_save){
  int i = 0;
  while(FileExists(saveFileName + std::to_string(i) + file_format) == true){
    i++;
  }
  std::cout << "File: " << (saveFileName + std::to_string(i) + file_format) << endl;
  pcl::io::savePCDFileASCII(saveFileName + std::to_string(i) + file_format, *cloud_to_save); 
}




// Can't use together with OrganizedMultiPlaneSegmentation
void downsample(pcl::PointCloud<PointT>::Ptr &input_cloud)
{
  std::cout << "Downsampling enabled. Hard coded Voxel Grid Leaf size: 0.01" << endl;
  const float vox_grid_leaf_size = 0.1;
  pcl::VoxelGrid<PointT> vox_grid;
  vox_grid.setLeafSize(vox_grid_leaf_size, vox_grid_leaf_size, vox_grid_leaf_size);
  vox_grid.setInputCloud(input_cloud);
  vox_grid.filter(*input_cloud);
  std::cout << "Downsampled to " << input_cloud->width * input_cloud->height  << " points"
            << "(" << input_cloud->width << " x " << input_cloud->height << ")" <<std::endl;
}


void compute_surface_normals(pcl::PointCloud<PointT>::Ptr &input_cloud, // Input Point cloud
    pcl::PointCloud<pcl::Normal>::Ptr &output_normals, // Output pointer to store Normals 
    float maxDepthChange,
    float smoothingSize)
{
  pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> ne;
  ne.setNormalEstimationMethod(ne.COVARIANCE_MATRIX);
  ne.setMaxDepthChangeFactor(maxDepthChange);
  ne.setDepthDependentSmoothing(true);
  ne.setNormalSmoothingSize(smoothingSize);
  ne.setInputCloud(input_cloud);
  ne.compute(*output_normals); 
}

void doMultiPlaneSegment(
  int argc, char** argv,
  pcl::PointCloud<PointT>::Ptr        &input_cloud,
  pcl::PointCloud<pcl::Normal>::Ptr   &cloud_normals,
  std::vector<pcl::PlanarRegion<PointT>, Eigen::aligned_allocator<pcl::PlanarRegion<PointT> > > &regions,
  std::vector<pcl::ModelCoefficients> &model_coefficients,
  std::vector<pcl::PointIndices>      &inlier_indices,
  pcl::PointCloud<pcl::Label>::Ptr    &labels,
  std::vector<pcl::PointIndices>      &label_indices,
  std::vector<pcl::PointIndices>      &boundary_indices)
{
  pcl::OrganizedMultiPlaneSegmentation<PointT, pcl::Normal, pcl::Label> seg;
  seg.setInputCloud(input_cloud);
  seg.setInputNormals(cloud_normals);

  unsigned int min_inliers =  atoi (argv[2]);
  seg.setMinInliers(min_inliers);
  float angular_thr = std::stof (argv[3]);
  seg.setAngularThreshold(0.01745329 * angular_thr); // 0.01745329 = 1degree
  float distance_thr = std::stof (argv[4]);
  seg.setDistanceThreshold(distance_thr);
  double maximum_curv = std::stof (argv[5]);
  seg.setMaximumCurvature(maximum_curv);
  std::cout << endl;
  std::cout << "Segmenting with:\n" 
            << "Minimum Inliers   : " << min_inliers  << "\n"
            << "Angular Threshold : " << angular_thr  << "\n"
            << "Distance Threshold: " << distance_thr << "\n"
            << "Maximum Curvature : " << maximum_curv << "\n"
            << endl;

  seg.segmentAndRefine(regions, model_coefficients, inlier_indices, labels,
                        label_indices, boundary_indices);

}

std::vector<int> vectorDifference(std::vector<int> v1, std::vector<int> v2)
{
  std::vector<int> diff;    
  std::set_difference(v1.begin(), v1.end(), v2.begin(), v2.end(),
  std::inserter(diff, diff.begin()));
  // for (auto i : v1) std::cout << i << ' ';
  // std::cout << "minus ";
  // for (auto i : v2) std::cout << i << ' ';
  // std::cout << "is: ";
  // for (auto i : diff) std::cout << i << ' ';
  // std::cout << '\n';
  return diff;
}

template <typename T>
void swap(T *xp, T *yp)  
{  
    T temp = *xp;  
    *xp = *yp;  
    *yp = temp;  
}  
  
// A function to implement bubble sort  
void planeBubbleSort(std::vector<int>  &numbers, PointT centroids[],
  std::vector<std::vector<float>> &normals, float p_value[],
  std::vector<pcl::PointIndices> &indices, int size) 
{ 
  int i, j; 
  bool swapped; 
  for (i = 0; i < size-1; i++) 
  { 
    swapped = false; 
    for (j = 0; j < size-i-1; j++){ 
      if (centroids[j].z > centroids[j+1].z) { 


        iter_swap(numbers.begin() + j, numbers.begin() + j+1); // iter_swap for std::vectors.
        iter_swap(normals.begin() + j, normals.begin() + j+1);
        iter_swap(indices.begin() + j, indices.begin() + j+1);
        swap(&centroids[j], &centroids[j+1]); // swap for regular c++ arrays
        swap(&p_value[j], &p_value[j+1]);
        swapped = true; 
      } 
    }
    // IF no two elements were swapped by inner loop, then break 
    if (swapped == false) {break;}
  } 
} 

template <typename T>
double dotProduct(T vect_A[], T vect_B[], int size) 
{ 
    double product = 0; 
    for (int i = 0; i < size; i++){
      product = product + (vect_A[i] * vect_B[i]); 
    }
    return product; 
} 

std::tuple<int, int, int> RGB_Texture(
  rs2::video_frame texture, rs2::texture_coordinate Texture_XY)
{
  // Get Width and Height coordinates of texture
  int width  = texture.get_width();  // Frame width in pixels
  int height = texture.get_height(); // Frame height in pixels
  
  // Normals to Texture Coordinates conversion
  int x_value = std::min(std::max(int(Texture_XY.u * width  + .5f), 0), width - 1);
  int y_value = std::min(std::max(int(Texture_XY.v * height + .5f), 0), height - 1);

  int bytes = x_value * texture.get_bytes_per_pixel();   // Get # of bytes per pixel
  int strides = y_value * texture.get_stride_in_bytes(); // Get line width in bytes
  int Text_Index =  (bytes + strides);

  const auto New_Texture = reinterpret_cast<const uint8_t*>(texture.get_data());
  
  // RGB components to save in tuple
  int NT1 = New_Texture[Text_Index];
  int NT2 = New_Texture[Text_Index + 1];
  int NT3 = New_Texture[Text_Index + 2];

  return std::tuple<int, int, int>(NT1, NT2, NT3);
}

pcl::PointCloud<PointT>::Ptr PCL_Conversion(const rs2::points& points, const rs2::video_frame& color)
{

  pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
  std::tuple<uint8_t, uint8_t, uint8_t> RGB_Color;

  // Convert data captured from Realsense camera to Point Cloud
  auto sp = points.get_profile().as<rs2::video_stream_profile>();
  
  cloud->width  = static_cast<uint32_t>( sp.width()  );   
  cloud->height = static_cast<uint32_t>( sp.height() );
  cloud->is_dense = false;
  cloud->points.resize( points.size() );

  auto Texture_Coord = points.get_texture_coordinates();
  auto Vertex = points.get_vertices();

  for (int i = 0; i < points.size(); i++)
  {   
    cloud->points[i].x = Vertex[i].x;
    cloud->points[i].y = Vertex[i].y;
    cloud->points[i].z = Vertex[i].z;

    RGB_Color = RGB_Texture(color, Texture_Coord[i]);

    // Mapping Color (BGR due to Camera Model)
    cloud->points[i].r = std::get<2>(RGB_Color); // Reference tuple<2>
    cloud->points[i].g = std::get<1>(RGB_Color); // Reference tuple<1>
    cloud->points[i].b = std::get<0>(RGB_Color); // Reference tuple<0>
  }
  return cloud;
}

// Capture a single frame and obtain depth + RGB values from it   
std::tuple<rs2::points, rs2::video_frame> getOneFrame(rs2::pipeline &pipeline)
{
  rs2::pointcloud raw_pointcloud;
  auto frames = pipeline.wait_for_frames();
  auto depth = frames.get_depth_frame();
  auto RGB = frames.get_color_frame();
  raw_pointcloud.map_to(RGB); // Map Color texture to each point
  auto points = raw_pointcloud.calculate(depth); // Generate Point Cloud
  return  std::make_tuple(points, RGB);
}

void initializeCamera(rs2::pipeline &pipeline)
{
  rs2::config cfg; // Create a configuration for configuring the pipeline with a non default profile
  cfg.enable_stream(RS2_STREAM_COLOR,    WIDTH, HEIGHT, RS2_FORMAT_BGR8, FRAMERATE);
  cfg.enable_stream(RS2_STREAM_INFRARED, WIDTH, HEIGHT, RS2_FORMAT_Y8,   FRAMERATE);
  cfg.enable_stream(RS2_STREAM_DEPTH,    WIDTH, HEIGHT, RS2_FORMAT_Z16,  FRAMERATE);
  rs2::pipeline_profile selection = pipeline.start(cfg); 
  rs2::device selected_device = selection.get_device();
  auto depth_sensor = selected_device.first<rs2::depth_sensor>();

  if (depth_sensor.supports(RS2_OPTION_EMITTER_ENABLED)){
    depth_sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 1.f); // Enable emitter
    depth_sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 0.f); // Disable emitter
  }
  if (depth_sensor.supports(RS2_OPTION_LASER_POWER)){
    // Query min and max values:
    rs2::option_range range = depth_sensor.get_option_range(RS2_OPTION_LASER_POWER);
    depth_sensor.set_option(RS2_OPTION_LASER_POWER, range.max); // Set max power
    depth_sensor.set_option(RS2_OPTION_LASER_POWER, 0.f); // Disable laser
  }
}

void doStatistics(std::vector<std::vector<int>> divisor_mat)
{
  int divisor_arr[imageAmount + 1] = {0};
  float dead_pt, healed_pt, alive_pt, total_points(WIDTH*HEIGHT);
  float dead_pt_pr, healed_pt_pr, alive_pt_pr; // percentages

  // Count all possible divisor amounts in the divisor matrix
  for(int i = 0; i < divisor_mat.size(); i++ ){
    for(int j = 0; j < divisor_mat[i].size(); j++){
          divisor_arr[divisor_mat[i][j]] +=1;
    }
  }

  // std::cout << endl;
  // std::cout << "Total count of all divisors by value:" << endl;
  // for(int c = 0; c < sizeof(divisor_arr)/sizeof(*divisor_arr); c++){
  //   std::cout << divisor_arr[c] << " for divisor = " << c << endl;
  // }
  // std::cout << endl;

  dead_pt = divisor_arr[0];
  alive_pt = divisor_arr[imageAmount];
  healed_pt = total_points - dead_pt - alive_pt;
  dead_pt_pr = (dead_pt/total_points) * 100;
  alive_pt_pr = (alive_pt/total_points) * 100;
  healed_pt_pr = (healed_pt/total_points) * 100;

  std::cout << "Statistics:" << endl;
  std::cout << "Total points: " << total_points << endl;
  std::cout << "Always dead points: " << dead_pt << "(" << dead_pt_pr << "%)" << endl;
  std::cout << "Always alive points: " << alive_pt << "(" << alive_pt_pr << "%)" << endl;
  std::cout << "Restored points: " << healed_pt << "(" << healed_pt_pr << "%)" << endl;
  std::cout << endl;

}

void AveragePointCloud(
  pcl::PointCloud<PointT>::Ptr output_cloud)
{
   rs2::pipeline pipe; // RealSense pipeline, encapsulating the actual device and sensors
  initializeCamera(pipe);

  // Drop several frames for auto-exposure
  for (int i = 0; i < 30; i++) {
    auto frames = pipe.wait_for_frames(); 
  }

  // Initiate first pointcloud that will store the end-result
  rs2::points points;
  rs2::frameset empty_frame;
  auto RGB_frame = empty_frame.get_color_frame();
  std::tie(points,RGB_frame) = getOneFrame(pipe);
  pcl::PointCloud<PointT>::Ptr final_cloud(new pcl::PointCloud<PointT>);
  final_cloud = PCL_Conversion(points, RGB_frame);
  

  // Defining variables for averaging loop
  pcl::PointCloud<PointT>::Ptr raw_cloud[imageAmount];
  int columns = final_cloud->width;
  int rows = final_cloud->height;
  std::vector<std::vector<int>> divisor_matrix; // 2D matrix to store denominators for each pixel in the cloud.
  divisor_matrix.resize(rows, std::vector<int>(columns, imageAmount));

  for (int k = 0; k < imageAmount; k++){

    rs2::points points;
    rs2::frameset empty_frame;
    auto RGB_frame = empty_frame.get_color_frame();
    std::tie(points,RGB_frame) = getOneFrame(pipe);
    raw_cloud[k] = PCL_Conversion(points, RGB_frame);


    if(k==2){ // TEMPORARY. Save 2nd frame just for comparison with Final
      std::string saveFileName = "/home/eugeneswag/ros_workspace/src/stair_estimation/src/pcd/raw_cloud";
      savePointCloud(saveFileName, raw_cloud[k]);
    }
    // Method: Iterate through each point of a raw_cloud and add it to final_cloud same position
    // Result: final_cloud will contain points, where each point value is a summation of all raw_cloud points in the same place.
    for(int n = 0; n < raw_cloud[k]->width;n++){    // n - columns (width)
      for(int m = 0; m < raw_cloud[k]->height;m++){ // m - rows    (height)
        PointT in, out, last;
        in = (*raw_cloud[k])(n,m); // get frame's point
        out = (*final_cloud)(n,m); // get output's current value point
        last.x = in.x + out.x;
        last.y = in.y + out.y;
        last.z = in.z + out.z;
        last.rgb = out.rgb;
        (*final_cloud)(n,m) = last;

        if ((in.x == 0.0) && (in.y == 0.0) && (in.z == 0.0))
        { // If current frame has null point-cloud, substract -1 from divisor
          // Divisor matrix is filled with integers from 0 to imageAmount,
          // depending how many times a dead point (0,0,0) is met among raw_cloud's
          // WARNING: notice the [m][n] swap! 
            divisor_matrix[m][n] -= 1;
        }
      }
    }
  }
  // Divides the summations in final_cloud by corresponding position value in divisor_matrix
  // Result: final_cloud now contains clouds points that have average value of all "non-zero" points from raw_cloud's.
  for(int n = 0; n < final_cloud->width; n++){
    for(int m = 0; m < final_cloud->height; m++){
      PointT in, out;
      in = (*final_cloud)(n,m); // get frame's point
      if (divisor_matrix[m][n] != 0){
        out.x = -(in.x / divisor_matrix[m][n]); // inverting X-axis to flip the image
        out.y = -(in.y / divisor_matrix[m][n]); // inverting Y-axis to flip the image
        out.z = in.z / divisor_matrix[m][n];
      }
      else{ out.x = 0; out.y = 0; out.z = 0; }

      // out.rgb = in.rgb / imageAmount;
      (*final_cloud)(n,m) = out;
    }
  }

  doStatistics(divisor_matrix);

  pcl::copyPointCloud(*final_cloud, *output_cloud);
  // std::cout << "Averaging has been completed" << endl;

  std::string saveFileName = "/home/eugeneswag/ros_workspace/src/stair_estimation/src/pcd/averaged_cloud";
  savePointCloud(saveFileName, output_cloud);
}

void setImageAmount(int argc, char** argv, int &fps){
  fps = atoi (argv[1]);
}




int main (int argc, char** argv){

  ros::init (argc, argv, "stair_estimation");
  ros::NodeHandle nh;
  ros::Rate loop_rate(4);

  setImageAmount(argc, argv, imageAmount);

  std::cout << "\n";
  std::cout << "=========================== ATTEMPT x\n";
  std::cout << "\n"; 
  // Open PCD file
  std::cout << "Averaging: " << argv[1] << endl;
  pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);

  AveragePointCloud(cloud);
  
  // char abs_path_string[] = "/home/eugeneswag/ros_workspace/src/stair_estimation/src/pcd/averaged_cloud1.pcd";
  // int file_found;
  // file_found = openPCD(cloud, abs_path_string);
  // if(file_found == 0){
  //   std::cout << "No file found. Try changing the path. Terminating.\n";
  //   return 0;
  // }
  

  // Compute normals of points.
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
  float max_depth_change = 0.01;
  float smoothing_size = 50.0;
  compute_surface_normals(cloud, cloud_normals, max_depth_change, smoothing_size);


  // Find multiple planes in pointcloud. 
  std::vector<pcl::PlanarRegion<PointT>, Eigen::aligned_allocator<pcl::PlanarRegion<PointT> > > regions;
  std::vector<pcl::ModelCoefficients> model_coefficients;
  std::vector<pcl::PointIndices>      inlier_indices; // Contains pointcloud with all planes found (PLANES NOT SEPARATED)
  pcl::PointCloud<pcl::Label>::Ptr    labels(new pcl::PointCloud<pcl::Label>);
  std::vector<pcl::PointIndices>      label_indices;
  std::vector<pcl::PointIndices>      boundary_indices;
  doMultiPlaneSegment(argc, argv, cloud, cloud_normals, regions, model_coefficients, inlier_indices,
    labels, label_indices, boundary_indices);

  pcl::PointCloud<PointT>::Ptr inlier_cloud   (new pcl::PointCloud<PointT>);
  pcl::PointCloud<PointT>::Ptr boundary_cloud (new pcl::PointCloud<PointT>);
  pcl::copyPointCloud(*cloud, boundary_indices, *boundary_cloud);
  pcl::copyPointCloud(*cloud, inlier_indices, *inlier_cloud);

  int numFoundPlanes = regions.size();
  if (numFoundPlanes == 0){
    std::cout << "No planes found, please try with different segmentation parameters or take a new pointcloud.\n";
    std::cout << "Terminating the program...\n";
    std::cout << "\n";
    return 0;
  }
  else if (numFoundPlanes == 1){
    std::cout << "Only 1 plane found, please try with different segmentation parameters or take a new pointcloud.\n";
    std::cout << "Terminating the program...\n";
    std::cout << "\n";
    return 0;
  }

  std::cout << "Found planes: " << numFoundPlanes << endl;
  std::cout << endl;

  PointT pl_Centroids[numFoundPlanes];
  std::vector<std::vector<float>> pl_Normals;
  pl_Normals.resize(numFoundPlanes, std::vector<float>(3, 0));
  float pl_PValue[numFoundPlanes];

  std::vector<int> planeNrToDelete(numFoundPlanes);
  std::vector<int> planeNrOriginal(numFoundPlanes);
  for (int e = 0; e < numFoundPlanes; e++){
    planeNrOriginal[e] = e;
  }

  int planeCount = 0;
  for(int i = 0; i < numFoundPlanes; i++){
    Eigen::Vector3f centroid = regions[i].getCentroid();
    pl_Centroids[i].x = centroid[0];
    pl_Centroids[i].y = centroid[1];
    pl_Centroids[i].z = centroid[2];
    pl_Normals[i][0] = model_coefficients[i].values[0];
    pl_Normals[i][1] = model_coefficients[i].values[1];
    pl_Normals[i][2] = model_coefficients[i].values[2];
    pl_PValue[i]     = model_coefficients[i].values[3];

    // Plane filtering
    if (pl_Centroids[i].x > 0.5 || pl_Centroids[i].x < -0.5 
      || pl_Centroids[i].y > 1.5 || pl_Centroids[i].z > 1.5 
      || pl_Normals[i][0] > 0.5 || pl_Normals[i][0] < -0.5
      || pl_Normals[i][1] > 0.8 || pl_Normals[i][1] < -0.8
      // || (pl_Normals[i][2] < 0.90 && pl_Normals[i][2] > -0.90)
      )        
    {
      planeNrToDelete[planeCount] = i;
      std::cout << "(REMOVED)";
      planeCount++;
    } 
    
    std::cout << "Plane " << i << ": "  << (double)regions[i].getCount() <<" Inliers."
              << " Centroid: (" << pl_Centroids[i].x << ", " <<  pl_Centroids[i].y << ", " <<  pl_Centroids[i].z << "); "
              << "Normal: (" << pl_Normals[i][0] << ", " << pl_Normals[i][1] << ", " << pl_Normals[i][2] << ")." << endl;
  }

    std::vector<int> saved_pl_Numbers;
    saved_pl_Numbers = vectorDifference(planeNrOriginal,planeNrToDelete);
    int numSavedPlanes = saved_pl_Numbers.size();
    if (numSavedPlanes == 0){
      std::cout << "No planes saved, please try with different segmentation parameters or take a new pointcloud.\n";
      std::cout << "Terminating the program...\n";
      std::cout << "\n";
      return 0;
    }
    else if (numSavedPlanes == 1){
      std::cout << "Only 1 plane saved. Minimum number of planes to compute distance = 2.\n";
      std::cout <<  "Please try with different segmentation parameters or take a new pointcloud.\n";
      std::cout << "Terminating the program...\n";
      std::cout << "\n";
      return 0;
    }

    std::cout << endl;
    std::cout << "Saved Planes ID: ";
    for (int i = 0; i < numSavedPlanes; i++){

      std::cout << saved_pl_Numbers[i] << " ";
    }
    std::cout << endl;
                                                                     // Saved:
  PointT saved_pl_Centroids[numSavedPlanes];                         // Centroids
  std::vector<std::vector<float>> saved_pl_Normals;                  // Normals
  saved_pl_Normals.resize(numFoundPlanes, std::vector<float>(3, 0));
  float saved_pl_PValue[numFoundPlanes];                             // Plane P values
  std::vector<pcl::PointIndices> saved_pl_Indices(numSavedPlanes);   // Indices

  // Updating Saved data from original data.
  for (int v = 0; v < numSavedPlanes; v++){
    saved_pl_Centroids[v] = pl_Centroids[saved_pl_Numbers[v]];
    saved_pl_Normals[v]   = pl_Normals[saved_pl_Numbers[v]];
    saved_pl_PValue[v]    = pl_PValue[saved_pl_Numbers[v]];
    saved_pl_Indices[v]   = inlier_indices[saved_pl_Numbers[v]];
  }

  // Ordering all Plane data as follows: plane closest to camera (Z-axis) is first, next closest is 2nd and so on...
  planeBubbleSort(saved_pl_Numbers, saved_pl_Centroids, saved_pl_Normals, saved_pl_PValue, saved_pl_Indices, numSavedPlanes);

  double step_height, step_depth_withEq, step_depth_noEq;
  int plane_index = 0;

  while((saved_pl_Centroids[plane_index + 1].z - saved_pl_Centroids[plane_index].z) < 0.05){
    plane_index++;
  }

  double N[3] = {-saved_pl_Normals[plane_index][0], -saved_pl_Normals[plane_index][1], -saved_pl_Normals[plane_index][2]};
  double step1[3] = {saved_pl_Centroids[plane_index].x, saved_pl_Centroids[plane_index].y, saved_pl_Centroids[plane_index].z};
  double step2[3] = {saved_pl_Centroids[plane_index+1].x, saved_pl_Centroids[plane_index+1].y, saved_pl_Centroids[plane_index+1].z};
  double step_diff[3];
  for(int i = 0; i < 3; i++){
    step_diff[i] = (step2[i]*100) - (step1[i]*100);
  }

  step_depth_withEq = dotProduct(step_diff, N, 3);
  step_depth_noEq = step2[2] - step1[2];
  step_height = step2[1] - step1[1];

  std::cout << step_height  * 100 << " " << step_depth_withEq << " " << step_depth_noEq * 100 << "\n";

  std::cout << "Step Depth is " << step_depth_withEq << "cm"
            << " and Height is " << step_height  * 100 << "cm. WITH EQUATIONS.\n";

  std::cout << "Step Depth is " << step_depth_noEq  * 100 << "cm"
            << " and Height is " << step_height  * 100 << "cm. NO EQUATIONS.\n";

  pcl::PointCloud<PointT>::Ptr result_cloud (new pcl::PointCloud<PointT>);
  pcl::copyPointCloud(*cloud, saved_pl_Indices, *result_cloud);

  std::string saveFileName = "/home/eugeneswag/ros_workspace/src/stair_estimation/src/pcd/filtered_cloud";
  savePointCloud(saveFileName, result_cloud);

  pcl::visualization::PCLVisualizer::Ptr viewer;
  viewer = visualiseCloud(result_cloud, pl_Centroids, numFoundPlanes);
  while (!viewer->wasStopped ()){
    viewer->spinOnce (100);
    loop_rate.sleep ();
  }
}

