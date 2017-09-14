# face_detection_and_alignment_with_dlib_cv
This repository contains code which can be used to detect faces ( using dlib )and align them (using cv).

To use this repository please see the instructions below :-

1. Create a directory named data in the project folder and put all your images in it. 
2. Run the preprocessor.py file in the  format python preprocess.py --input-dir data --output-dir output --crop-dim 180.
3. A folder name output will be created with all the faces detected.

You need to install opencv and dlib to us this.

This code is heavily inspired from https://hackernoon.com/building-a-facial-recognition-pipeline-with-deep-learning-in-tensorflow-66e7645015b8. 

