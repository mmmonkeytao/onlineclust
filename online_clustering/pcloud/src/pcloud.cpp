#include <stdexcept>
#include <cstdio>
#include <string>
#include <sstream>

#include <pcl/range_image/range_image.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/features/don.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/search/kdtree.h>

#include <vtkImageActor.h>
#include <vtkImageData.h>
#include <vtkImageMapper3D.h>
#include <vtkImageMapToColors.h>
#include <vtkImageProperty.h>
#include <vtkInteractorStyleImage.h>
#include <vtkLookupTable.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkSmartPointer.h>
#include <vtkVersion.h>
#include <vtkImageViewer2.h>
#include <vtkPNGWriter.h>
#include <vtkImageShiftScale.h>
#include <vtkImageCast.h>

#include "pcloud.h"

using namespace onlineclust;
using namespace std;

void PCloud::load_pcloud(const char *dir, int nfiles)
{
  this->cloud = new _pclType1[nfiles];
  
  for(int i = 0; i < nfiles; ++i ){
    string infile(dir);
    stringstream ss;
    ss << i;
    infile = infile + "kitti." + ss.str() + ".pcd";
    
    if (pcl::io::loadPCDFile<pcl::PointXYZI> (infile.c_str(), cloud[i]) == -1){
      cout << "\nCan't load file: ";
      throw runtime_error(infile.c_str());
    }
    else
      {
	cout << "Successfully load pcl data file: " << infile << endl;
      }
  }
}

void PCloud::proc_pcloud(_pclType1::Ptr& pcloud, uint ith)
{
  // segmentation using different normals
  // this->diff_normal_segmentation(ocloud, pcloud, ith);
  //pcloud = new _pclType1{this->cloud[ith]};

  // seg plane model
  plane_model_segmentation(this->cloud[ith], pcloud);
  
  // clustering
  std::vector<pcl::PointIndices> cluster_indices;
  euclidean_cluster_extraction(pcloud, cluster_indices);
  //vis_pointcloud2rangeimage(pcloud, cluster_indices);
	    
  // calculate normals
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
  cloud_normal(pcloud, cloud_normals);

}

void PCloud::cloud_normal(_pclType1::Ptr &cloud, pcl::PointCloud<pcl::Normal>::Ptr &cloud_normals)
{

    pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> ne1;
    ne1.setInputCloud (cloud);
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZI> ());
    ne1.setSearchMethod (tree);

    ne1.setRadiusSearch (this->fparam.radius);
    ne1.compute (*cloud_normals);
}


void PCloud::vis_pointcloud2rangeimage(_pclType1::Ptr &pcloud, std::vector<pcl::PointIndices> &cluster_indices)
{
    pcl::ExtractIndices<pcl::PointXYZI> extract;
    std::cout << "no. of clusters: " << cluster_indices.size() << endl;
    // --------------------------------------------
    // -----save sub-clusters in rangeImage[]-----
    // --------------------------------------------
    pcl::RangeImage range_image[cluster_indices.size()];

    uint i = 0;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {
        // extract sub-clusters
        _pclType1::Ptr sub_cluster(new _pclType1);

        extract.setInputCloud (pcloud);
        extract.setIndices( boost::shared_ptr<pcl::PointIndices>( new pcl::PointIndices( *it ) ) );
        extract.setNegative (false);
        extract.filter (*sub_cluster);

        // We now want to create a range image from the above point cloud, with a 1deg angular resolution
        float angularResolution = (float) (  0.25f * (M_PI/180.0f));  //   1.0 degree in radians
        float maxAngleWidth     = (float) (360.0f * (M_PI/180.0f));  // 360.0 degree in radians
        float maxAngleHeight    = (float) (180.0f * (M_PI/180.0f));  // 180.0 degree in radians
        Eigen::Affine3f sensorPose = (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f);
        pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::LASER_FRAME;
        float noiseLevel = 0.00;
        float minRange = 0.0f;
        int borderSize = 1;

        range_image[i].createFromPointCloud(*sub_cluster.get(), angularResolution, maxAngleWidth, maxAngleHeight,sensorPose, coordinate_frame, noiseLevel, minRange, borderSize);

        //range_image[i].createFromPointCloudWithViewpoints(*sub_cluster.get(), angularResolution, maxAngleWidth, maxAngleHeight,
        //                                                  coordinate_frame, noiseLevel, minRange, borderSize);

        i++;
    }

    // --------------------------------------------
    // -----Open 3D viewer and add point cloud-----
    // --------------------------------------------
    pcl::visualization::PCLVisualizer viewer ("3D Viewer");
    viewer.setBackgroundColor (1, 1, 1);
    boost::shared_ptr<pcl::RangeImage> range_image_ptr(new pcl::RangeImage);

    //viewer.addCoordinateSystem (1.0f, "global");
    //pcl::visualization::PointCloudColorHandlerCustom<PointType> point_cloud_color_handler (point_cloud_ptr, 150, 150, 150);
    //viewer.addPointCloud (point_cloud_ptr, point_cloud_color_handler, "original point cloud");
    viewer.initCameraParameters ();

    // --------------------------
    // -----Show range image-----
    // --------------------------
    pcl::visualization::RangeImageVisualizer range_image_widget ("Range image");

    //--------------------
    // -----Main loop-----
    //--------------------
    i = 0;
    while (!viewer.wasStopped ())
    {

      if (i < cluster_indices.size())
	{
          range_image_widget.spinOnce ();
          viewer.spinOnce ();
          pcl_sleep (1);

	  //scene_sensor_pose = viewer.getViewerPose();
	  *range_image_ptr = range_image[i];

	  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointWithRange> range_image_color_handler (range_image_ptr, 0, 0, 0);
	  viewer.addPointCloud (range_image_ptr, range_image_color_handler, "range image");
	  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "range image");

	  setViewerPose(viewer, range_image[i].getTransformationToWorldSystem ());
	  range_image_widget.setSize(500,500);
	  range_image_widget.showRangeImage (range_image[i], -std::numeric_limits<float>::infinity (),
					     std::numeric_limits<float>::infinity (),
					     true );
	  i++;

	}
      else
	{
          range_image_widget.spinOnce (100);
          viewer.spinOnce (100);
	}

    }
  
    std::cout << "Start to save .png files" << endl;
    for(i = 0; i < cluster_indices.size(); ++i)
      {
	// store range_image[0]
	int width = range_image[i].width;
	int height = range_image[i].height;

	// Create a "grayscale" 16x16 image, 1-component pixels of type "double"
	vtkSmartPointer<vtkImageData> image =
	  vtkSmartPointer<vtkImageData>::New();
	int imageExtent[6] = { 0, width-1, 0, height-1, 0, 0 };
	image->SetExtent(imageExtent);
#if VTK_MAJOR_VERSION <= 5
	image->SetNumberOfScalarComponents(1);
	image->SetScalarTypeToDouble();
#else
	image->AllocateScalars(VTK_DOUBLE, 1);
#endif
 
	//double scalarvalue = 0.0;
	float *pixel_value = range_image[i].getRangesArray();
	float max = 0, min = 6000;
	for (int y = imageExtent[2]; y <= imageExtent[3]; y++)
	  {
	    for (int x = imageExtent[0]; x <= imageExtent[1]; x++)
	      {
		double* pixel = static_cast<double*>(image->GetScalarPointer(x, y, 0));
		if( pixel_value[x + y * width] < 0.0)
		  pixel[0] = 0;
		else {
		  pixel[0] = pixel_value[x + y * width];
		  if(pixel[0] > max) max = pixel[0];
		  if(pixel[0] < min) min = pixel[0];
		}
	      }
	  }

	// Map the scalar values in the image to colors with a lookup table:
	vtkSmartPointer<vtkLookupTable> lookupTable =
	  vtkSmartPointer<vtkLookupTable>::New();
	lookupTable->SetNumberOfTableValues(500);
	lookupTable->SetRange(min, max);
	lookupTable->Build();
 
	// Pass the original image and the lookup table to a filter to create
	// a color image:
	vtkSmartPointer<vtkImageMapToColors> scalarValuesToColors =
	  vtkSmartPointer<vtkImageMapToColors>::New();
	scalarValuesToColors->SetLookupTable(lookupTable);
	scalarValuesToColors->PassAlphaToOutputOn();
#if VTK_MAJOR_VERSION <= 5
	scalarValuesToColors->SetInput(image);
#else
	scalarValuesToColors->SetInputData(image);
#endif
	scalarValuesToColors->SetOutputFormatToRGB();

	// Output
	// set file name
	std::string fname("RangeImage.");
	std::stringstream ss;
	ss << i;
	fname = fname + ss.str() + ".png";
	vtkSmartPointer<vtkPNGWriter> writer =
	  vtkSmartPointer<vtkPNGWriter>::New();
	writer->SetFileName(fname.c_str());
	writer->SetInputConnection(scalarValuesToColors->GetOutputPort());
	writer->Write();
	std::cout << "Saved RangeImage." << i << ".png" << endl;
      }
}

void PCloud::setViewerPose (pcl::visualization::PCLVisualizer& viewer, const Eigen::Affine3f& viewer_pose)
{
  Eigen::Vector3f pos_vector = viewer_pose * Eigen::Vector3f(0, 0, 0);
  Eigen::Vector3f look_at_vector = viewer_pose.rotation () * Eigen::Vector3f(0, 0, 1) + pos_vector;
  Eigen::Vector3f up_vector = viewer_pose.rotation () * Eigen::Vector3f(0, -1, 0);
  viewer.setCameraPosition (pos_vector[0], pos_vector[1], pos_vector[2],
                            look_at_vector[0], look_at_vector[1], look_at_vector[2],
                            up_vector[0], up_vector[1], up_vector[2]);
}

void PCloud::euclidean_cluster_extraction(_pclType1::Ptr &pcloud,  std::vector<pcl::PointIndices>&cluster_indices)
{
      // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZI>);
    tree->setInputCloud (pcloud);

    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
    ec.setClusterTolerance (this->fparam.segradius);
    ec.setMinClusterSize (200);
    ec.setMaxClusterSize (10000);
    ec.setSearchMethod (tree);
    ec.setInputCloud (pcloud);
    ec.extract (cluster_indices);

    int j = 0;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {
      _pclType1::Ptr cloud_cluster (new _pclType1);
      for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++)
        {
            //cloud_cluster->points.push_back (cloud_in->points[*pit]); //*
            pcloud->points[*pit].intensity = j/(double)cluster_indices.size() ;
        }
      /*
      cloud_cluster->width = cloud_cluster->points.size ();
      cloud_cluster->height = 1;
      cloud_cluster->is_dense = true;

      std::cout << "PointCloud representing the Cluster: No." << j << " " << cloud_cluster->points.size () << " data points." << std::endl;
      */
      j++;
    }

}


void PCloud::diff_normal_segmentation(_pclType1& cloud, _pclType1::Ptr& pcloud, uint ith)
{
 // filters, remove outliers
    pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor;
    sor.setInputCloud (cloud.makeShared());
    sor.setMeanK (this->fparam.NUM_NEIGHBORS);
    sor.setStddevMulThresh (this->fparam.STD_DEVIATION);
    //sor.filter (*cloud_filtered);
    sor.filter (*(cloud.makeShared()));

    cout << "Start processing Point Clouds: " << ith << endl;

    /* Difference of Normals Based Segmentation */
    // Create a search tree, use KDTreee for non-organized data.
    pcl::search::Search<pcl::PointXYZI>::Ptr tree;
    if (cloud.isOrganized())
    {
        cout << "Points Cloud " << ith << " is organized." << endl;
        tree.reset (new pcl::search::OrganizedNeighbor<pcl::PointXYZI> ());
    }
    else
    {
        cout << "Points Cloud " << ith << " is unorganized." << endl;
        tree.reset (new pcl::search::KdTree<pcl::PointXYZI> (false));
    }

    // Set the input pointcloud for the search tree
    tree->setInputCloud (cloud.makeShared());

    if(this->fparam.scale1 >= this->fparam.scale2)
    {
        cerr << "Error: Large scale must be > small scale!" << endl;
        exit (EXIT_FAILURE);
    }

    // Compute normals using both small and large scales at each point
    pcl::NormalEstimationOMP<pcl::PointXYZI, pcl::PointNormal> ne;
    ne.setInputCloud (cloud.makeShared());
    ne.setSearchMethod (tree);

    /**
    * NOTE: setting viewpoint is very important, so that we can ensure
    * normals are all pointed in the same direction!
    */
    ne.setViewPoint (std::numeric_limits<float>::max (), std::numeric_limits<float>::max (), std::numeric_limits<float>::max ());

    // calculate normals with the small scale
    cout << "Calculating normals for scale..." << this->fparam.scale1 << endl;
    pcl::PointCloud<pcl::PointNormal>::Ptr normals_small_scale (new pcl::PointCloud<pcl::PointNormal>);

    ne.setRadiusSearch (this->fparam.scale1);
    ne.compute (*normals_small_scale);

    // calculate normals with the large scale
    cout << "Calculating normals for scale..." << this->fparam.scale2 << endl;
    pcl::PointCloud<pcl::PointNormal>::Ptr normals_large_scale (new pcl::PointCloud<pcl::PointNormal>);

    ne.setRadiusSearch (this->fparam.scale2);
    ne.compute (*normals_large_scale);

    // Create output cloud for DoN results
    pcl::PointCloud<pcl::PointNormal>::Ptr doncloud (new pcl::PointCloud<pcl::PointNormal>);
    pcl::copyPointCloud<pcl::PointXYZI, pcl::PointNormal>(*(cloud.makeShared()), *doncloud);

    cout << "Calculating DoN... " << endl;
    // Create DoN operator
    pcl::DifferenceOfNormalsEstimation<pcl::PointXYZI, pcl::PointNormal, pcl::PointNormal> don;
    don.setInputCloud (cloud.makeShared());
    don.setNormalScaleLarge (normals_large_scale);
    don.setNormalScaleSmall (normals_small_scale);

    if (!don.initCompute ())
    {
        std::cerr << "Error: Could not intialize DoN feature operator" << std::endl;
        exit (EXIT_FAILURE);
    }

    // Compute DoN
    don.computeFeature (*doncloud);

    // Filter by magnitude
    cout << "Filtering out DoN mag <= " << this->fparam.threshold << "  " << endl;

    // Build the condition for filtering
    pcl::ConditionOr<pcl::PointNormal>::Ptr range_cond (
                new pcl::ConditionOr<pcl::PointNormal> ()
                );
    range_cond->addComparison (pcl::FieldComparison<pcl::PointNormal>::ConstPtr (new pcl::FieldComparison<pcl::PointNormal> ("curvature", pcl::ComparisonOps::GT, this->fparam.threshold)));

    // Build the filter
    pcl::ConditionalRemoval<pcl::PointNormal> condrem;
    condrem.setCondition(range_cond);
    condrem.setInputCloud (doncloud);

    pcl::PointCloud<pcl::PointNormal>::Ptr doncloud_filtered (new pcl::PointCloud<pcl::PointNormal>);

    // Apply filter
    condrem.filter (*doncloud_filtered);

    doncloud = doncloud_filtered;

    // Filter by magnitude
    cout << "Clustering using EuclideanClusterExtraction with tolerance <= " << this->fparam.segradius << "..." << endl;

    pcl::search::KdTree<pcl::PointNormal>::Ptr segtree (new pcl::search::KdTree<pcl::PointNormal>);
    segtree->setInputCloud (doncloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointNormal> ec;

    ec.setClusterTolerance (this->fparam.segradius);
    ec.setMinClusterSize (100);
    ec.setMaxClusterSize (4000);
    ec.setSearchMethod (segtree);
    ec.setInputCloud (doncloud);
    ec.extract(cluster_indices);

    int j = 0;
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_cluster_don (new pcl::PointCloud<pcl::PointNormal>);
    //pcl::PointCloud<pcl::PointXYZI>::Ptr ocloud (new pcl::PointCloud<pcl::PointXYZI>);
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it, j++)
    {

        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++)
        {
            //cloud_cluster_don->points.push_back (doncloud->points[*pit]);
            pcl::PointXYZI point;
            point.x = doncloud->points[*pit].x;
            point.y = doncloud->points[*pit].y;
            point.z = doncloud->points[*pit].z;
            point.intensity = j;

            pcloud->points.push_back(point);
        }

        cloud_cluster_don->width = int (cloud_cluster_don->points.size ());
        cloud_cluster_don->height = 1;
        cloud_cluster_don->is_dense = true;

    }

    //pcl::copyPointCloud<pcl::PointNormal, pcl::PointXYZI>(*doncloud, *cloud_filtered);
    // filter again
    /*sor.setInputCloud (ccloud);
    sor.setMeanK (NUM_NEIGHBORS);
    sor.setStddevMulThresh (STD_DEVIATION);
    sor.filter (*ccloud);*/
  
}

void PCloud::plane_model_segmentation(_pclType1 &cloud, _pclType1::Ptr &pcloud)
{
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices ());

    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);

    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZI> seg;
    // Optional
    seg.setOptimizeCoefficients (true);
    // Mandatory
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold(this->fparam.distance_threshold);

    seg.setInputCloud (cloud.makeShared());
    seg.segment (*inliers, *coefficients);

    if (inliers->indices.size () == 0)
    {
      PCL_ERROR ("Could not estimate a planar model for the given dataset.");
    }

    std::cerr << "(plane_model_seg)Model coefficients: " << coefficients->values[0] << " "
                                        << coefficients->values[1] << " "
                                        << coefficients->values[2] << " "
                                        << coefficients->values[3] << std::endl;

    std::cerr << "Model inliers: " << inliers->indices.size () << std::endl;
    /*for (size_t i = 0; i < inliers->indices.size (); ++i)
      std::cerr << inliers->indices[i] << "    " << cloud->points[inliers->indices[i]].x << " "
                                                 << cloud->points[inliers->indices[i]].y << " "
                                                 << cloud->points[inliers->indices[i]].z << std::endl;*/


    // Create the filtering object
    pcl::ExtractIndices<pcl::PointXYZI> extract;

    // Extract the inliers
    extract.setInputCloud (cloud.makeShared());
    extract.setIndices (inliers);
    extract.setNegative (false);
    extract.filter (*pcloud);
    //std::cerr << "PointCloud representing the planar component: " << cloud_p->width * cloud_p->height << " data points." << std::endl;

    //std::stringstream ss;
    //ss << "table_scene_lms400_plane_" << i << ".pcd";
    //writer.write<pcl::PointXYZI> (ss.str (), *cloud_p, false);

    // Create the filtering object
    extract.setNegative (true);
    extract.filter (*pcloud);
    //cloud_filtered.swap (cloud_f);

}
