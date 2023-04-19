# Medical_Imaging_Project
OBJECTIVE:
The goal is to develop an algorithm for estimating cortical thickness map from a 
raw T1-weighted image and segmenting gray and white matter. 
INTRODUCTION:
• Cortical thickness map is the thickness of the gray matter of the human 
cortex brain at every point. It is defined as the distance between the white 
matter surface and the pial surface. It is as 3D distance measured from the 
white--gray matter boundary in the tissue classified brain volume to the cortical 
surface (gray--CSF boundary). The thickness of the cortex is used for identifying 
affected brain regions by disease for assessing treatment and studying brain 
development and ageing.
• Cortical thickness measures the width of gray matter of the human cortex and 
can be calculated from T1-weighted magnetic resonance images (MRI).
• To ensure the accuracy of cortical thickness measures, we compute them in the 
3D image volume at a voxel level using gray matter segmentation of the image 
rather than using the vertices in the surface mesh.
Brain Extraction
Brain extraction is a critical preprocessing step in the analysis of neuroimaging 
studies conducted with magnetic resonance imaging (MRI) and influences the 
accuracy of downstream analyses.
Skull Stripping of MR Brain Images
Skull stripping is designed to eliminate non-brain tissues from MR brain images for 
many clinical applications and analyses, its accuracy and speed are considered as 
the key factors in the brain image segmentation and analysis. The accurate and 
automated skull stripping methods help to improve the speed and accuracy of 
prognostic and diagnostic procedures in medical applications.
• Mathematical morphology-based method for Skull Stripping
these methods use the morphological erosion and dilation operations to 
separate the skull from the brain region. It requires a combination of 
thresholding and edge detection methods to find the initial ROI (region of 
interest). It consists of histogram-based thresholding and morphological 
operations. Based on the brain anatomical knowledge, it discriminates 
between the desired and undesired structures. This method is implemented 
using a sequence of conventional and novel morphological operations, using 
2D and 3D operations


