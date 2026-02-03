#####       HELPFUNCTIONS FOR MORPHING OF ARTIFICIAL FACE STIMULI 




###     (Step 1)
# 
#    
#       LOAD LIBRARIES

import cv2 # type: ignore
import dlib # type: ignore
import glob
import os
import numpy as np # type: ignore
from PIL import Image # type: ignore
import matplotlib.pyplot as plt # type: ignore
import shutil
from pathlib import Path



# Search for landamrk file

def get_landmarks(p):
    predictor_path = p
    return(predictor_path)

###     (Step 2)
# 
#    
#       GRAYSCALE AND COPY TREE STRUCTURE

def grayscale_images_in_directory(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Loop through all files in the input directory
    for filename in os.listdir(input_directory):
        input_path = os.path.join(input_directory, filename)
        
        # Check if the file is an image
        try:
            with Image.open(input_path) as img:
                # Convert the image to grayscale
                gray_img = img.convert('L')
                
                # Save the grayscale image in the output directory
                output_path = os.path.join(output_directory, filename)
                gray_img.save(output_path)
                
                # print(f"Converted {filename} to grayscale and saved to {output_path}")
        except IOError:
            print(f"Skipping {filename}, not a valid image file")
    
    print("Grayscaling and Copying complete.")

def copy_and_grayscale_tree(src, dest):
    # Iterate over the directory tree
    for dirpath, dirnames, filenames in os.walk(src):
        # Create the new directory structure
        structure = os.path.join(dest, os.path.relpath(dirpath, src))
        if not os.path.isdir(structure):
            os.makedirs(structure)
        
        # Process the images in the current directory
        for file in filenames:
            if file.lower().endswith('.png'):  # Process only PNG files
                input_file_path = os.path.join(dirpath, file)
                output_file_path = os.path.join(structure, file)
                
                # Convert to grayscale and save in the new directory
                try:
                    with Image.open(input_file_path) as img:
                        gray_img = img.convert('L')
                        gray_img.save(output_file_path)
                        # print(f"Converted {input_file_path} to grayscale and saved to {output_file_path}")
                except IOError:
                    print(f"Skipping {input_file_path}, not a valid image file")
            else:
                # Copy non-image files without modification
                shutil.copy2(os.path.join(dirpath, file), structure)
    print("Grayscaling and Copying complete.")


###     (Step 3)
#
#
#       MORPHING


# Custom exception for no face found
class NoFaceFound(Exception):
    pass

# Calculate image dimensions
def calculate_margin_help(img1, img2):
    size1 = img1.shape
    size2 = img2.shape
    diff0 = abs(size1[0] - size2[0]) // 2
    diff1 = abs(size1[1] - size2[1]) // 2
    avg0 = (size1[0] + size2[0]) // 2
    avg1 = (size1[1] + size2[1]) // 2
    return [size1, size2, diff0, diff1, avg0, avg1]

# Crop image
def crop_image(img1, img2):
    [size1, size2, diff0, diff1, avg0, avg1] = calculate_margin_help(img1, img2)
    if size1[0] == size2[0] and size1[1] == size2[1]:
        return [img1, img2]
    elif size1[0] <= size2[0] and size1[1] <= size2[1]:
        scale0 = size1[0] / size2[0]
        scale1 = size1[1] / size2[1]
        res = cv2.resize(img2, None, fx=scale0, fy=scale0, interpolation=cv2.INTER_AREA) if scale0 > scale1 else cv2.resize(img2, None, fx=scale1, fy=scale1, interpolation=cv2.INTER_AREA)
        return crop_image_help(img1, res)
    elif size1[0] >= size2[0] and size1[1] >= size2[1]:
        scale0 = size2[0] / size1[0]
        scale1 = size2[1] / size1[1]
        res = cv2.resize(img1, None, fx=scale0, fy=scale0, interpolation=cv2.INTER_AREA) if scale0 > scale1 else cv2.resize(img1, None, fx=scale1, fy=scale1, interpolation=cv2.INTER_AREA)
        return crop_image_help(res, img2)
    elif size1[0] >= size2[0] and size1[1] <= size2[1]:
        return [img1[diff0:avg0, :], img2[:, -diff1:avg1]]
    else:
        return [img1[:, diff1:avg1], img2[-diff0:avg0, :]]

# Crop image helper function
def crop_image_help(img1, img2):
    [size1, size2, diff0, diff1, avg0, avg1] = calculate_margin_help(img1, img2)
    if size1[0] == size2[0] and size1[1] == size2[1]:
        return [img1, img2]
    elif size1[0] <= size2[0] and size1[1] <= size2[1]:
        return [img1, img2[-diff0:avg0, -diff1:avg1]]
    elif size1[0] >= size2[0] and size1[1] >= size2[1]:
        return [img1[diff0:avg0, diff1:avg1], img2]
    elif size1[0] >= size2[0] and size1[1] <= size2[1]:
        return [img1[diff0:avg0, :], img2[:, -diff1:avg1]]
    else:
        return [img1[:, diff1:avg1], img2[diff0:avg0, :]]

# Locate facial landmarks
def generate_face_correspondences(theImage1, theImage2, predictor_path):
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    corresp = np.zeros((68, 2))
    
    imgList = crop_image(theImage1, theImage2)
    list1 = []
    list2 = []
    j = 1

    for img in imgList:
        size = (img.shape[0], img.shape[1])
        currList = list1 if j == 1 else list2
        dets = detector(img, 1)

        if len(dets) == 0:
            raise NoFaceFound("Sorry, but I couldn't find a face in the image.")
        j += 1

        for k, rect in enumerate(dets):
            shape = predictor(img, rect)
            for i in range(0, 68):
                x = shape.part(i).x
                y = shape.part(i).y
                currList.append((x, y))
                corresp[i][0] += x
                corresp[i][1] += y

            currList.extend([
                (1, 1), (size[1] - 1, 1), ((size[1] - 1) // 2, 1), 
                (1, size[0] - 1), (1, (size[0] - 1) // 2), 
                ((size[1] - 1) // 2, size[0] - 1), 
                (size[1] - 1, size[0] - 1), 
                ((size[1] - 1), (size[0] - 1) // 2)
            ])

    narray = corresp / 2
    narray = np.append(narray, [[1, 1]], axis=0)
    narray = np.append(narray, [[size[1] - 1, 1]], axis=0)
    narray = np.append(narray, [[(size[1] - 1) // 2, 1]], axis=0)
    narray = np.append(narray, [[1, size[0] - 1]], axis=0)
    narray = np.append(narray, [[1, (size[0] - 1) // 2]], axis=0)
    narray = np.append(narray, [[(size[1] - 1) // 2, size[0] - 1]], axis=0)
    narray = np.append(narray, [[size[1] - 1, size[0] - 1]], axis=0)
    narray = np.append(narray, [[(size[1] - 1), (size[0] - 1) // 2]], axis=0)
    
    return [size, imgList[0], imgList[1], list1, list2, narray]

# Check if a point is inside a rectangle
def rect_contains(rect, point):
    return rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3]

# Draw the delaunay triangles
def draw_delaunay(f_w, f_h, subdiv, dictionary1):
    list4 = []
    triangleList = subdiv.getTriangleList()
    r = (0, 0, f_w, f_h)

    for t in triangleList:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            list4.append((dictionary1[pt1], dictionary1[pt2], dictionary1[pt3]))

    return list4

# Triangulate
def make_delaunay(f_w, f_h, theList, img1, img2):
    rect = (0, 0, f_w, f_h)
    subdiv = cv2.Subdiv2D(rect)
    theList = theList.tolist()
    points = [(int(x[0]), int(x[1])) for x in theList]
    dictionary = {x[0]: x[1] for x in list(zip(points, range(76)))}
    for p in points:
        subdiv.insert(p)
    list4 = draw_delaunay(f_w, f_h, subdiv, dictionary)
    return list4

# Apply affine transform calculated using srcTri and dstTri to src and output an image of size.
def apply_affine_transform(src, srcTri, dstTri, size):
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst

# Warps and alpha blends triangular regions from img1 and img2 to img
def morph_triangle(img1, img2, img, t1, t2, t, alpha):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    t1Rect = [(t1[i][0] - r1[0], t1[i][1] - r1[1]) for i in range(3)]
    t2Rect = [(t2[i][0] - r2[0], t2[i][1] - r2[1]) for i in range(3)]
    tRect = [(t[i][0] - r[0], t[i][1] - r[1]) for i in range(3)]

    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = apply_affine_transform(img1Rect, t1Rect, tRect, size)
    warpImage2 = apply_affine_transform(img2Rect, t2Rect, tRect, size)

    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * (1 - mask) + imgRect * mask

# Generate morph sequence
def generate_morph_sequence(duration, frame_rate, img1, img2, points1, points2, tri_list, output_path, fs1g, fs2g, fs1e, fs2e, avatar1_name, avatar2_name, p):
    num_images = int(duration * frame_rate)
    gif = []

    for j in range(num_images):
        img1 = np.float32(img1)
        img2 = np.float32(img2)
        points = []
        alpha = j / (num_images - 1)

        for i in range(len(points1)):
            x = (1 - alpha) * points1[i][0] + alpha * points2[i][0]
            y = (1 - alpha) * points1[i][1] + alpha * points2[i][1]
            points.append((x, y))

        morphed_frame = np.zeros(img1.shape, dtype=img1.dtype)

        for i in range(len(tri_list)):
            x = int(tri_list[i][0])
            y = int(tri_list[i][1])
            z = int(tri_list[i][2])

            t1 = [points1[x], points1[y], points1[z]]
            t2 = [points2[x], points2[y], points2[z]]
            t = [points[x], points[y], points[z]]

            morph_triangle(img1, img2, morphed_frame, t1, t2, t, alpha)

        res = Image.fromarray(cv2.cvtColor(np.uint8(morphed_frame), cv2.COLOR_BGR2RGB))

        # Naming logic based on conditions
        if fs1e == fs2e:
            if int(fs1g[1:]) >= 50:              # morph from neutral to male
                j = j + 49
                file_name = f"{avatar1_name}_{fs1e}_G{j+1:02d}.png"

            elif int(fs1g[1:]) < 50:         # from female to neutral
                j = j - 1
                file_name = f"{avatar1_name}_{fs1e}_G{j+1:02d}.png"


        elif fs1g == fs2g:
            if int(fs1e[1:]) >= 50:         # from neutral to fear
                j = j + 49
                file_name = f"{avatar1_name}_E{j+1:02d}_{fs1g}.png"

            elif int(fs1e[1:]) < 50:       # from happy to neutral
                j = j - 1
                file_name = f"{avatar1_name}_E{j+1:02d}_{fs1g}.png"


        res.save(os.path.join(output_path, file_name))

# Define Wrapper Function
def doMorphing(img1, img2, duration, frame_rate, output_path, facespace1gen, facespace2gen, facespace1emo, facespace2emo, avatar1_name, avatar2_name, p):
    [size, img1, img2, points1, points2, list3] = generate_face_correspondences(img1, img2, p)
    tri = make_delaunay(size[1], size[0], list3, img1, img2)
    generate_morph_sequence(duration, frame_rate, img1, img2, points1, points2, tri, output_path, facespace1gen, facespace2gen, facespace1emo, facespace2emo, avatar1_name, avatar2_name, p)



# Define processing folders function
def process_folders(gray_path, output_path, p):
    # Loop through gray_path to find all subfolders and PNG files
    for subdir, _, files in os.walk(gray_path):
        png_files = [f for f in files if f.endswith('.png')]

        # Ensure we have exactly two PNG files
        if len(png_files) != 2:
            print(f"WARNING: Skipping {subdir} as it does not contain exactly two PNG files.")
            continue

        # Identify img1 and img2 based on filename pattern
        img1_path = None
        img2_path = None

        if png_files[0][4:6] == png_files[1][4:6]:        # same emotion, morph on gender
            if int(png_files[0][8:10]) < int(png_files[1][8:10]):
                img1_path = os.path.join(subdir, png_files[0])
                img2_path = os.path.join(subdir, png_files[1])
            else:
                img1_path = os.path.join(subdir, png_files[1])
                img2_path = os.path.join(subdir, png_files[0])
        elif png_files[0][8:10] == png_files[1][8:10]:     # same gender, morph emotion
            if int(png_files[0][4:6]) < int(png_files[1][4:6]):
                img1_path = os.path.join(subdir, png_files[0])
                img2_path = os.path.join(subdir, png_files[1])
            else:
                img1_path = os.path.join(subdir, png_files[1])
                img2_path = os.path.join(subdir, png_files[0])

        # Additional check to ensure both paths are assigned correctly
        if not img1_path or not img2_path:
            print(f"WARNING: In folder {subdir}, the required PNG files were not detected or properly assigned.")
            continue

        if img1_path and img2_path:
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)

            img1 = cv2.resize(img1, (500, 500), interpolation=cv2.INTER_AREA)
            img2 = cv2.resize(img2, (500, 500), interpolation=cv2.INTER_AREA)

            relative_path = os.path.relpath(subdir, gray_path)
            gif_output_dir = os.path.join(output_path, relative_path)
            os.makedirs(gif_output_dir, exist_ok=True)

            img1_path_parts = os.path.normpath(img1_path).split(os.sep)
            img1_name = img1_path_parts[-1]
            img2_path_parts = os.path.normpath(img2_path).split(os.sep)
            img2_name = img2_path_parts[-1]



            img1_name_without_extension = os.path.splitext(img1_name)[0]
            avatar1_name, gen1_name, emo1_name = img1_name_without_extension.split('_')

            img2_name_without_extension = os.path.splitext(img2_name)[0]
            avatar2_name, gen2_name, emo2_name = img2_name_without_extension.split('_')

            print("=====================")
            print(f"PATH: {gif_output_dir}")
            print("=====================")
            print(f"Morph from: {img1_name_without_extension}")
            print(f"gender: {gen1_name}")
            print(f"emotion: {emo1_name}")
            print("=== *** === *** ===")
            print(f"Morph to: {img2_name_without_extension}")
            print(f"gender: {gen2_name}")
            print(f"emotion: {emo2_name}")
            print("=====================")

            doMorphing(img1, img2, 5, 10, gif_output_dir, gen1_name, gen2_name, emo1_name, emo2_name, avatar1_name, avatar2_name, p)
                        
            print(f"MORPHING COMPLETE: {avatar1_name}_{emo1_name}_{gen1_name} to {avatar1_name}_{emo2_name}_{gen2_name}")
        
        else:
            print(f"WARNING: In folder {subdir}, the required PNG files were not detected.")



###     (Step 4)
# 
#    
#       Fill up facespace


# Define processing folders function
def fill_facespace(facespace_prep_path, output_path, p):
    # Loop through gray_path to find all subfolders and PNG files
    for subdir, _, files in os.walk(facespace_prep_path):
        
        # Identify img1 and img2 based on filename pattern
        img1_path = None
        img2_path = None

        for i in range(1,17):           # loop through 16 avatars
            avatar = f"{int(i):02d}"
            for j in range(100):        # loop through all emotions, morph gender from 00 to 50 and 50 to 99
                emo = f"{int(j):02d}"
                if emo == f"{int('00'):02d}" or emo == f"{int('50'):02d}" or emo == f"{int('99'):02d}": # dont morph emo 00, 50 or 99
                    continue
                for g in range(2):
                    if g == 0:
                        img1_path = os.path.join(subdir, f"{avatar}_E{emo}_G00.png")    # from female to neutral
                        img2_path = os.path.join(subdir, f"{avatar}_E{emo}_G50.png")
                    else:
                        img1_path = os.path.join(subdir, f"{avatar}_E{emo}_G50.png")    # from neutral to male 
                        img2_path = os.path.join(subdir, f"{avatar}_E{emo}_G99.png")


                    # Additional check to ensure both paths are assigned correctly
                    if not img1_path or not img2_path:
                        print(f"WARNING: In folder {subdir}, the required PNG files were not detected or properly assigned.")
                        continue

                    elif img1_path and img2_path:
                        img1 = cv2.imread(img1_path)
                        img2 = cv2.imread(img2_path)

                        img1 = cv2.resize(img1, (500, 500), interpolation=cv2.INTER_AREA)
                        img2 = cv2.resize(img2, (500, 500), interpolation=cv2.INTER_AREA)

                        relative_path = os.path.relpath(subdir, facespace_prep_path)
                        gif_output_dir = os.path.join(output_path, relative_path)
                        os.makedirs(gif_output_dir, exist_ok=True)

                        path = Path(gif_output_dir)

                        # Extract parts of the path
                        parts = list(path.parts)
                        folder1 = parts[-1]  # The last part is the filename
                        folder2 = parts[-2]

                        # Set up naming of facespace
                        img1_path_parts = os.path.normpath(img1_path).split(os.sep)
                        img1_name = img1_path_parts[-1]
                        img2_path_parts = os.path.normpath(img2_path).split(os.sep)
                        img2_name = img2_path_parts[-1]


                        img1_name_without_extension = os.path.splitext(img1_name)[0]
                        avatar1_name, emo1_name, gen1_name = img1_name_without_extension.split('_')

                        img2_name_without_extension = os.path.splitext(img2_name)[0]
                        avatar2_name, emo2_name, gen2_name = img2_name_without_extension.split('_')

                        facespace1emo  = emo1_name
                        facespace2emo = emo2_name
                        facespace1gen = gen1_name
                        facespace2gen = gen2_name

                        print("=====================")
                        print(f"PATH: {gif_output_dir}")
                        print("=====================")
                        print(f"image1_name: {img1_name_without_extension}")
                        print(f"gen1_name: {gen1_name}, {facespace1gen}")
                        print(f"emo1_name: {emo1_name}, {facespace1emo}")
                        print("=====================")
                        print(f"image2_name: {img2_name_without_extension}")
                        print(f"gen2_name: {gen2_name}, {facespace2gen}")
                        print(f"emo2_name: {emo2_name}, {facespace2emo}")
                        print("=====================")

                        doMorphing(img1, img2, 5, 10, gif_output_dir, facespace1gen, facespace2gen, facespace1emo, facespace2emo, avatar, avatar, p)
                                    
                        print(f"MORPHING COMPLETE: {avatar}_{facespace1emo}_{facespace1gen} to {avatar}_{facespace2emo}_{facespace2gen}")
                    
                    else:
                        print(f"WARNING: In folder {subdir}, the required PNG files were not detected.")

###     (Step 5)
# 
#    
#       Copy Files to facespace target

def copy_jpg_files(source_dir, target_dir):
    # Ensure the target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    count = 1

    # Walk through the source directory
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.png'):  # Check for .jpg extension (case-insensitive)
                source_file = os.path.join(root, file)
                target_file = os.path.join(target_dir, file)
                
                # Copy file to the target directory
                shutil.copy2(source_file, target_file)
                count += 1
    print(f"Copied {count} files")
