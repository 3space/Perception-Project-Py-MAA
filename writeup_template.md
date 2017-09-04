## Project: Perception Pick & Place
### Writeup Template: You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---


# Required Steps for a Passing Submission:
1. Extract features and train an SVM model on new objects (see `pick_list_*.yaml` in `/pr2_robot/config/` for the list of models you'll be trying to identify). 
2. Write a ROS node and subscribe to `/pr2/world/points` topic. This topic contains noisy point cloud data that you must work with.
3. Use filtering and RANSAC plane fitting to isolate the objects of interest from the rest of the scene.
4. Apply Euclidean clustering to create separate clusters for individual items.
5. Perform object recognition on these objects and assign them labels (markers in RViz).
6. Calculate the centroid (average in x, y and z) of the set of points belonging to that each object.
7. Create ROS messages containing the details of each object (name, pick_pose, etc.) and write these messages out to `.yaml` files, one for each of the 3 scenarios (`test1-3.world` in `/pr2_robot/worlds/`).  See the example `output.yaml` for details on what the output should look like.  
8. Submit a link to your GitHub repo for the project or the Python code for your perception pipeline and your output `.yaml` files (3 `.yaml` files, one for each test world).  You must have correctly identified 100% of objects from `pick_list_1.yaml` for `test1.world`, 80% of items from `pick_list_2.yaml` for `test2.world` and 75% of items from `pick_list_3.yaml` in `test3.world`.
9. Congratulations!  Your Done!

# Extra Challenges: Complete the Pick & Place
7. To create a collision map, publish a point cloud to the `/pr2/3d_map/points` topic and make sure you change the `point_cloud_topic` to `/pr2/3d_map/points` in `sensors.yaml` in the `/pr2_robot/config/` directory. This topic is read by Moveit!, which uses this point cloud input to generate a collision map, allowing the robot to plan its trajectory.  Keep in mind that later when you go to pick up an object, you must first remove it from this point cloud so it is removed from the collision map!
8. Rotate the robot to generate collision map of table sides. This can be accomplished by publishing joint angle value(in radians) to `/pr2/world_joint_controller/command`
9. Rotate the robot back to its original state.
10. Create a ROS Client for the “pick_place_routine” rosservice.  In the required steps above, you already created the messages you need to use this service. Checkout the [PickPlace.srv](https://github.com/udacity/RoboND-Perception-Project/tree/master/pr2_robot/srv) file to find out what arguments you must pass to this service.
11. If everything was done correctly, when you pass the appropriate messages to the `pick_place_routine` service, the selected arm will perform pick and place operation and display trajectory in the RViz window
12. Place all the objects from your pick list in their respective dropoff box and you have completed the challenge!
13. Looking for a bigger challenge?  Load up the `challenge.world` scenario and see if you can get your perception pipeline working there!


[//]: # (Image References)

[image1]: ./misc/table.png
[image2]: ./misc/objects.png
[image3]: ./misc/cluster.png
[image4]: ./misc/confusion_matrix.png
[image5]: ./misc/test_world_1.png
[image6]: ./misc/test_world_2.png
[image7]: ./misc/test_world_3.png

## [Rubric](https://review.udacity.com/#!/rubrics/1067/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Exercise 1, 2 and 3 pipeline implemented
#### 1. Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting implemented.
For the point cloud of the scene the first filter implemented was the voxel grid filter for down sampling of the data.  This was implemented as so:
    
    #################  Voxel Grid Filter ###############
    vox = cloud.make_voxel_grid_filter()
    # Choose a voxel (also known as leaf) size
    LEAF_SIZE = 0.005 #5cm
    # Set the voxel (or leaf) size  
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    # Call the filter function to obtain the resultant downsampled point cloud
    cloud_filtered = vox.filter()
    ####################################################

Next an Outlier Removal Filter was applied to external sources of noise from the scene:

    ###################  Outlier Filter ################
    outlier_filter = cloud_filtered.make_statistical_outlier_filter()
    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(50)
    # Set threshold scale factor
    x = 1.1
    # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(x)
    # Finally call the filter function for magic
    cloud_filtered = outlier_filter.filter()
    ###################################################

Then a Passthrough filter was implemented to remove extraneous data from the camera data.  The z-axis was chosen to focus on data centered on the height of the table, and the y-axis filter was used to remove data from the left and right of the table.
    
    ############### Passthrough Filter ###############
    passthrough_z = cloud_filtered.make_passthrough_filter()
    filter_axis = 'z'
    passthrough_z.set_filter_field_name (filter_axis)
    passthrough_z.set_filter_limits(0.6, 0.9)
    cloud_filtered = passthrough_z.filter()

    passthrough_y = cloud_filtered.make_passthrough_filter()
    filter_axis = 'y'
    passthrough_y.set_filter_field_name (filter_axis)
    passthrough_y.set_filter_limits(-0.5, 0.5)
    cloud_filtered = passthrough_y.filter()
    ################################################## 



#### 2. Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.  
For the next step in the pipline the RANSAC algorithm was used to identify points in the data belonging only to the table and objects, respectively.  This is shown here:
    
    ##################### RANSAC #####################
    seg = cloud_filtered.make_segmenter()

    # Set the model you wish to fit 
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    # Max distance for a point to be considered fitting the model
    # Experiment with different values for max_distance 
    # for segmenting the table
    max_distance = 0.01

   seg.set_distance_threshold(max_distance)

   # Call the segment function to obtain set of inlier indices and model coefficients
   inliers, coefficients = seg.segment()

   # TODO: Extract inliers and outliers
   cloud_table = cloud_filtered.extract(inliers, negative=False)
   cloud_objects = cloud_filtered.extract(inliers, negative=True)
   ###################################################

And these are images of the table and object clusters:

![alt text][image1]
![alt text][image2]

Once the table plane was filtered out, the Euclidean Clustering algorithm was used to separate the remaining data points into indivdual objects.  This is the code here: 

    ############# Euclidean Clustering ###############
    white_cloud = XYZRGB_to_XYZ(cloud_objects) # Apply function to convert XYZRGB to XYZ
    tree = white_cloud.make_kdtree()

    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()

    # Set tolerances for distance threshold 
    # as well as minimum and maximum cluster size (in points)
    ec.set_ClusterTolerance(0.01)
    ec.set_MinClusterSize(100)
    ec.set_MaxClusterSize(7000)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()
    #################################################

Then the clusters were given individual colors to more easily visualize.  Show here in the following code and image:
 
    ################ Color Cluster ##################
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                             white_cloud[indice][1],
                                             white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    #Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)
    ################################################

![alt text][image3]

#### 3. Complete Exercise 3 Steps.  Features extracted and SVM trained.  Object recognition implemented.
The features capture and SVM training were done using sensor_stick scipts and simulations.  For this iteration, 64 random orientations were chosen for the training and 128 bins were used for the color histograms , which gave good results. The following image shows confusion matrix for the training:

![alt text][image4]

Object recognition was done in lines 233~267 in 'project_template.py'.  For this, histogram features were extracted and predictions were made based on the model provided by the training.  

### 4.  Conclusion
The perception pipeline was implemented according to the guidelines of exercises 1-3, with the addition of the outlier removal filter to remove sources of ambient noise from the environment.  A smaller leaf size was chosen than previosly used.  Doing this I could obtain more data for the point clouds detected.  The passthrough filter was chosen to focus on the table by removing data in the z-direction.  A y-axis passthrough filter was also used to remove extraneous data to the left and right of the table, such as the drop boxes.  The detected objects were written to file on line 274 of 'project_template.py'.  This method could be improved as each time the pcl callback is invoked, the file will be overwritten. Her are some pictures of the output I obtained. As you can see in test world 2 the book is misidentified, but all others seem accurate.  This could probably be remedied with more training orientations.  And here are the results:

![alt text][image5]
![alt text][image6]
![alt text][image7]


### Pick and Place Setup

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.



Spend some time at the end to discuss your code, what techniques you used, what worked and why, where the implementation might fail and how you might improve it if you were going to pursue this project further.  



