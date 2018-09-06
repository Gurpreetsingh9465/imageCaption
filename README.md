# Image Caption
This is th inference of the show and tell model using inception V3 trained on mscoco dataset learn more 
https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf

## Requirement
1. Tensorflow
2. PIL
3. numpy

## Examples
!["first2"](https://raw.githubusercontent.com/Gurpreetsingh9465/imageCaption/blob/master/examples/Screenshot%20(10).png "first2")
(https://raw.githubusercontent.com/Gurpreetsingh9465/imageCaption/blob/master/examples/Screenshot%20(11).png )
(https://raw.githubusercontent.com/Gurpreetsingh9465/imageCaption/blob/master/examples/Screenshot%20(15).png)
(https://raw.githubusercontent.com/Gurpreetsingh9465/imageCaption/blob/master/examples/Screenshot%20(16).png)
(https://raw.githubusercontent.com/Gurpreetsingh9465/imageCaption/blob/master/examples/Screenshot%20(17).png)

## Steps

1. clone the git repository and extract it.
2. download the pre trained .pb file https://drive.google.com/file/d/1_AH6KCuk8ZiwPln9KnIaW6_KruDORn4e/view?usp=drivesdk .
3. move the optimized.pb to the same folder.

## structure
by now folder structure should look like
```shell
--dictionary.txt
--model.py
--show_and_tell.py
--optimized.pb
```

## Running the inference
Make sure the path of the image file should not contain white spaces
```shell
python show_and_tell.py path_of_image_file
```


