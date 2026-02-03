#### README File for Faces in Focus - Morphing Scripts
#

AUTHOR: Lukas Kunz, Universiteit Leiden, February 2026

#


Entailed files
* facespace_morph.py
    * contains all functions needed for Jupyter notebook.
* Morph_images.ipynb
    * contains the flow of all functions to create morphs.
* shape_predictor_81_face_landmarks.dat
    * data file, containing a trained set of landmarks of faces.

#

Overview
  * This script provides a set of functions to preprocess, align, and morph artificial face stimuli. It includes grayscale conversion, landmark detection, and face morphing functionalities. The morphing process is based on facial landmarks and Delaunay triangulation to ensure smooth transitions between faces.

#

Dependencies
  Ensure the following Python libraries are installed before running the script:
  * opencv-python (cv2)
  * dlib
  * numpy
  * PIL (Pillow)
  * matplotlib
  * shutil
  * pathlib

  You may install missing dependencies using:
  pip install opencv-python dlib numpy pillow matplotlib

#

Functionality
  * Step 1: Load Libraries & Landmark File
    * The script imports necessary libraries for image processing.
    *  The get_landmarks(p) function returns the path to the facial landmark predictor.

  * Step 2: Preprocessing - Grayscale Conversion & Directory Copying
    * grayscale_images_in_directory(input_directory, output_directory):
      Converts all images in a directory to grayscale and saves them.
    * copy_and_grayscale_tree(src, dest): Recursively copies a directory while converting PNG images to grayscale.

  * Step 3: Morphing Process
    * The morphing functions follow these steps:
      * 1 Facial Landmark Detection
        * generate_face_correspondences(theImage1, theImage2, predictor_path): Detects facial landmarks in two images using dlib and generates correspondence points.
      * 2 Image Alignment & Cropping
        * calculate_margin_help(img1, img2): Calculates size differences between two images.
        * crop_image(img1, img2): Crops images to match dimensions.
      * 3 Triangulation & Warping
        * make_delaunay(f_w, f_h, theList, img1, img2): Performs Delaunay triangulation on face landmarks.
        * apply_affine_transform(src, srcTri, dstTri, size): Applies affine transformation to warp facial regions.
        * morph_triangle(img1, img2, img, t1, t2, t, alpha): Warps and blends triangular regions between images.
      * 4 Generating Morph Sequence
        * generate_morph_sequence(duration, frame_rate, img1, img2, points1, points2, tri_list, output_path, ...): Creates a sequence of morphed frames for animation.
      * 5 Wrapper Function for Morphing
        * doMorphing(img1, img2, duration, frame_rate, output_path, ...): Integrates all morphing steps into a single function.

#

Processing Folders
    * process_folders(gray_path, output_path, p): Iterates through a directory structure, detects image pairs, and applies the morphing process.

#

Notes
* Ensure the facial landmark model (shape_predictor_68_face_landmarks.dat) is available.
* The script works best with aligned and high-quality face images.

#

Contact: Lukas Kunz, Universiteit Leiden
