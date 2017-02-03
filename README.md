# Pipeline for Advanced Lane Regression (PipeAdLR)

## Identifying Lane Regions on a Road

***

_"I have a cunning plan."_

&ndash; Baldrick, _Black Adder_

The Pipeline for Advanced Lane Regression (PipeAdLR) implements a solution to the scenario described in Project 4 of Udacity's [Self-Driving Car Engineer](https://www.udacity.com/drive) nanodegree. It is concerned with identifying lane markings on a road, calculating their radius of curvature and the car's apparent displacement from the lane's center. PipeAdLR complies to the sequence of operations outlined in Lesson 14 and the Project rubric. Specifically, the following steps are performed:

1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images;
1. Load a video record of a trip along a road with visible lane markings;
1. Apply distortion correction to raw images;
1. Apply thresholds to gradient (horizontal Sobel) and color space (HLS) transforms of the undistorted image to generate a binary map;
1. Apply a perspective transform to transform binary images from up-front perspective to top-down (or "bird's-eye view").
1. Search the top-down binary map for pixels belonging to left and right lane marks, then convert them to 2D point coordinate arrays;
1. Match polynomial curves to the points collected from either lane line, to be used as guides for lane boundaries;
1. Determine [radius of curvature](http://www.intmath.com/applications-differentiation/8-radius-curvature.php) and viewer position relative to lane boundaries;
1. Draw lane boundaries (warped back to up-front perspective) and write lane curvature and vehicle position estimates onto the original raw image;
1. Write a new video record containing the annotated frames.

The next sections describe in detail the implementation of the steps above, experiments performed with the completed system, and perspectives for further enhancement.

## Implementation

PipeAdLR was implemented as an [IPython](https://ipython.org/) [notebook](P4.ipynb) relying on a variety of libraries for advanced operations. Specifically, [numpy](http://www.numpy.org/) is used for multidimensional types (arrays, matrices, images) and general data manipulation; [OpenCV](http://www.opencv.org/) for more advanced image processing routines; [scipy](http://scipy.org/) and [scikit-learn](http://scikit-learn.org) for statistics and machine learning; [matplotlib](http://matplotlib.org) for visualization; and [moviepy](http://zulko.github.io/moviepy/) for video file input/output.

PipeAdLR makes extensive use of Python classes to model the variety of concepts and data involved in the process of lane detection. A recurrent motif is the use of _callable objects_ to encapsulate a function with parameters that can be calculated once and then reused over the system's life-cycle, or otherwise keep state that is used between calls. This can be seen for example in the `PointExtractor` class, which extracts point coordinates from lane marking images, and keeps track of the position of previously detected points to speedup detection in subsequent inputs. Another common pattern are classes that bind together several related types of data, computed at construction time from an input argument. This is illustrated by the `Corners` class, which takes a chessboard image as input, extracts corner coordinates in image and object reference frames, and stores results to its `imgpoints` and `objpoints` attributes. This makes easier to move around and access related data sets throughout the system.

The algorithms implemented on PipeAdLR often require a couple manually adjusted parameters to be provided. Those add up fast, and can become hard to manage. Therefore, a global `params` object is used to hold together all such settings in a single place. Global data generally isn't a good idea, but here it's forgivable since it's no more than an alternative way to specify what would otherwise be defined as a set of constants. The following subsections give further details on those algorithms and their implementation.

### Camera Calibration and Distortion Correction

OpenCV provides the [cv2.undistort()](http://docs.opencv.org/2.4.13/modules/imgproc/doc/geometric_transformations.html#cv2.undistort) function to correct image distortions predicted from camera parameters, [cv2.calibrateCamera()](http://docs.opencv.org/2.4.13/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#cv2.calibrateCamera) to compute such parameters from a set of image points specified both in image and object reference frames, and [cv2.findChessboardCorners()](http://docs.opencv.org/2.4.13/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#cv2.findChessboardCorners) to compute such points from a set of images of a chessboard.

In our case, calibration images were provided as part of project materials (see folder [camera_cal](camera_cal/)). One difficulty with this particular calibration set is that not all chessboard corners are visible in all images; furthermore, the rather extreme distortions in some images cause the `cv2.findChessboardCorners()` to sometimes not recognize all visible corners (see section 2 in the [notebook](P4.ipynb) for images illustrating the problem). Restricting calibration only to those corners that were visible and detected in all images would severely reduce the number of data points available for calibration; on the other hand, computing object-frame coordinates for points in "incomplete" sets (i.e. from images where not all corners were found) would require knowing their positions relative to the board. Fortunately it was noticed that even when corner detection failed to retrieve all data points, the set would always form a rectangle and include at least one of the extreme top-left, top-right, bottom-left or bottom-right corners. This enabled a simple algorithm to be devised to compute position offsets for incomplete image-frame point sets, which could then be used to compute corresponding object-frame points.

Class `Corners` is responsible for computing sets of image-frame and object-frame points from chessboard images. It is used by function `Undistorter()` to extract image-frame and object frame points from a collection of chessboard images, compute camera parameters with `cv2.calibrateCamera()`, and finally return a closure on `cv2.undistort()` with camera parameters fixed as arguments; the resulting function can be used on raw images to return distortion-corrected versions.

### Binary Map Generation

After camera distortions are corrected, the next step in lane identification is to compute a binary map reporting the locations of pixels likely to belong in a lane marking. Observation indicates lane marking pixels commonly display the following features:

* A high degree of lightness;
* Colors in the range from white to brighter shades of gray, or alternatively yellow;
* For pixels in the border of markings, a high gradient value in the horizontal direction.

This suggests a strategy for implementing a binary map with a high correlation to lane marking pixels:

1. Transform the input RGB image to the HLS color space;
2. Set pixels (in the output binary image) with hue in a yellow-centered range `(Ha, Hb)` and saturation past a threshold `St`;
3. Set pixels with lightness past a threshold `Lt`;
4. Convert the original image to grayscale, compute the horizontal Sobel transform, and set pixels with an absolute gradient past a threshold `Et`.

See section 3 of the [notebook](P4.ipynb) for example binary maps computed in the above manner.

### Perspective Transform

Before lane line pixels can be selected, the binary map must be converted from its current up-front perspective to a top-down, "bird's view" one. This can be done through OpenCV functions [cv2.getPerspectiveTransform()](http://docs.opencv.org/2.4.13/modules/imgproc/doc/geometric_transformations.html#cv2.getPerspectiveTransform) (which computes a perspective transform matrix between two corresponding sets of four points) and [cv2.warpPerspective()](http://docs.opencv.org/2.4.13/modules/imgproc/doc/geometric_transformations.html#cv2.warpPerspective) (which applies a perspective transform to an image). Additionally, [cv2.perspectiveTransform()](http://docs.opencv.org/2.4.13/modules/core/doc/operations_on_arrays.html#cv2.perspectiveTransform) applies a perspective transform to a set of points, which will be useful later when detected lane boundaries need to be transform back into up-front perspective.

PipeAdLR function `perspective_matrices()` computes a pair of perspective transform matrices for images of a given `size = (width, height)`, one for conversion from up-front to top-down perspective, and another in the opposite direction. It uses sets of points computed by function `perspective_corners()`, which by its turn uses system parameters to compute the positions of corners in the given image scale. Functions `transform_image()` and `transform_points()` are convenient shortcuts to the OpenCV transform functions.

See section 4 of the [notebook](P4.ipynb) for perspective transform examples.

### Lane Line Point Extraction

Top-down perspective binary maps must next be processed for identification of lane line pixels, and their extraction as 2D point coordinates. In fact, PipeAdLR takes the opposite approach: all set pixels in a binary map are extracted as points, and then those found to belong to a lane line are selected for further processing.

Point selection is performed in two steps. First, a Probability Density Function (PDF) is computed from the distribution of points along the horizontal axis. Only points extracted from the bottom half of the image are used, which already provide enough information (and since the upper half is significantly noisier, it wouldn't help otherwise). The two highest peaks of the distribution are taken as the center location for the left and right lane lines.

Next, a sliding window of dimensions `(W_width, W_height)` is used to select lane points, starting from the bottom at the position of either peak, and working its way up. The window moves up in units of `W_height`; at each level, it slides from `k - W_width` to `k + W_width`, where `k` is the final position at the previous level (or the peaks of the PDF for the first level). The window stops at the position with the highest point density within its borders; these are the points that are extracted at each level.

The PDF is not computed for every input map: rather, after it is computed once and initial `k` values are found for either lane line, the positions of highest density at the bottom are used as the `k` values for the next frame. Only if a point search turns empty is the PDF calculation repeated.

See section 5 of the [notebook](P4.ipynb) for examples.

### Lane Boundary Estimation and Attribute Computation

Extracted points for either lane line are then submitted to polynomial regression. The RANSAC algorithm is used due to its robustness even in the presence of outliers. Further, estimates after the first one are averaged into a running estimate to smooth out occasional mistakes. The radius of curvature and the viewer's distance to either lane boundary are also computed at this step.

See section 6 of the [notebook](P4.ipynb) for examples.

### Frame Annotation

Finally, original video frames are annotated with the estimated location of lane boundaries. The radius of curvature of each lane and the estimated viewer position relative to them are written to the image, on the top-left region.

See section 7 of the [notebook](P4.ipynb) for examples.

## Experiments

Three videos were used for experiments. File [project_video.mp4](project_video.mp4) was the evaluation target for the project, while [challenge_video.mp4](challenge_video.mp4) and [harder_challenge_video.mp4](harder_challenge_video.mp4) provided optional challenges to test the limits of the implemented system. For each input three output videos were computed, showing computed binary maps (`<name>_binary_map.mp4`), selected lane points and estimated boundaries  (`<name>_lane_regression.mp4`), and original frames with annotations (`<name>_annotated.mp4`). The results can be seen in section 9 of the [notebook](P4.ipynb).

Results were generally very accurate for the target video. Immediate lane regression sometimes produces spurious results (as can be seen in the [project_video_lane_regression.mp4](project_video_lane_regression.mp4) video), but these are averaged out in the running estimate.

In stark contrast to target video results, the annotations in the two challenge videos were wildly off. Reasons are clear once binary map videos are studied: the filters that worked so well in the target video produce an excessive amount of false positives in the challenges, and to add insult to injury, lane pixels are often left out.

## Conclusions

The Pipeline for Advanced Lane Regression (PipeAdLR) is a system for lane boundary identification, using a video recorded from a vehicle's front-mounted camera as input. Experiments show it can work well in specific (though fairly realistic) conditions, but has trouble generalizing to variations in visual conditions. Specifically, images with brightness saturation and poor contrast throw the lower levels of the pipeline off, leading to unavoidable estimate failures later on. This could be addressed by applying some sort of normalization to the input. This, therefore, should be the focus of further development.
