# Heatmap using OpenCV and Python

cretaing heatmap for moving objects from a static camera

![](images/heatmap1.png)

## Steps
1. creating video capture object
2. creating a mask(heatmap) to apply heatmap on the video
3. taking difference of the consecutive frames using a while loop
4. removing noises
5. applying the difference on heatmap and noramilizing it 
6. applying colormap
7. adding colormap to the output
