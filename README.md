# The Classification of Bryozoan on Kelp Tissue Using Aerial Imagery

<b>Frankie Pothier</b>
<b>December 19th, 2024</b>

<b>Abstract:</b> This study develops an algorithm for classifying bryozoans on kelp tissues using
aerial imagery. The primary objective is to detect and quantify bryozoan coverage on kelp and
label the aerial data for use in training a neural network. Understanding the extent of bryozoan
colonization on kelp is critical, as bryozoan can have significant negative impacts on kelp forest
ecosystems. The methodology employed in this study includes the creation of binary masks
through thresholding to differentiate between water, kelp, and bryozoan. After removing water
from the images, pixel values were adjusted to enhance the contrast between kelp and bryozoan
pixels. A morphological opening transformation was applied to eliminate noise and misclassified
kelp tissue. Model performance was assessed through visual inspection and the computation of a
confusion matrix to quantify classification accuracy. The model demonstrated strong
performance in avoiding misclassification of kelp as bryozoan, although it was less effective at
detecting all bryozoan coverage. This limitation was mainly due to some bryozoan being deep in
the water column, resulting in much lower reflectance values than the majority of bryozoan in
the images. A key limitation of this study was the variability of marine environments, which
makes it challenging to capture such diversity in a single algorithm. Despite these limitations, the
algorithm performed adequately, offering valuable insights for further refinement and
application.

### 1.0 Introduction

Kelp Forests are a vital ecosystem covering around 25% of the world’s coastlines (Wernberg
et al., 2019), which provide important habitats and food for a wide range of marine species
(Wernberg et al., 2019; Mora-Soto et al., 2024; Starko et al., 2024). These ecosystems thrive in
cold, nutrient-rich waters found in shallow subtidal zones within temperate and subpolar regions
(Teagle et al., 2017; Wernberg et al., 2019). Unfortunately, kelp forests have been classified as
the second most vulnerable costal marine ecosystem by the Intergovernmental panel on climate
change (IPCC) to effects caused by climate change (Pörtner et al. 2019; Starko et al., 2024).
In recent decades, there has been a significant decline in the size of kelp forests, with some
areas disappearing entirely (Timmer et al., 2024; Starko et al., 2024). These losses in kelp
ecosystems highlight the critical need to examine and understand the impacts of climate change
on them at both local and global scales (Starko et al., 2024). Due to the dynamic nature of the
marine environment, these effects are not evenly distributed across all kelp ecosystems, with
some thriving while others see significant losses. Rising ocean temperatures pose one of the
greatest threats to kelp forests, driving increases in marine heatwaves and higher surface
temperatures (Pörtner et al., 2019; Mora-Soto et al., 2024; Starko et al., 2024). Other threats to
kelp forests include, but are not limited to, overfishing, sewage runoff, invasive species, and
other anthropogenic stressors. However, the extent to which these factors impact kelp forests
remains uncertain (Mora-Soto et al., 2024; Starko et al., 2024).

The coastlines of British Columbia, Oregon, and Washington host at least 28 different
species of kelp (Mora-Soto et al., 2024). However, over the past 1.5 to 3 decades, these regions
have experienced significant kelp deforestation due to climate-related threats (Starko et al.,
2024), with some areas around Vancouver Island seeing up to a 74% loss of kelp forests. The
invasive European bryozoan Membranipora membranacea poses a significant threat to these
species, particularly because it thrives in warmer waters. The combination of rising sea
temperatures and the spread of bryozoan colonies could have a profoundly negative impact on
these ecosystems (Saier & Chapman, 2004; Pratt et al., 2022).

Bryozoans are small aquatic organisms that live in colonies often found on kelp blades
(Saier & Chapman, 2004). These colonies can cause significant damage to individual kelp plants
and entire kelp forests (Saier & Chapman, 2004; Pratt et al., 2022). By forming a stiff crust on
kelp tissue, bryozoans act as a physical barrier, blocking the vital nutrients and light needed for
photosynthesis (Saier & Chapman, 2004; Pratt et al., 2022). This crust also stiffens the kelp
blades, making them more brittle and increasing the likelihood of breakage during storms or
wave surges (Saier & Chapman, 2004; Pratt et al., 2022). Additionally, if bryozoan colonies
cover the reproductive parts of the kelp, they can obstruct the release of reproductive spores,
further hindering the growth and regeneration of kelp forests (Saier & Chapman, 2004). Thus, it
is important to examine the extent to which bryozoans cover kelp tissue to gain a deeper
understanding of the effects caused by this invasive species.

Remote sensing has proven to be an effective tool for mapping and analyzing the temporal
and spatial distribution of kelp forests (Schroeder et al., 2019; Mora-Soto et al., 2024; Timmer et
al., 2024). This approach often relies on the red-edge (approximately 670–750 nm) or near-
infrared (NIR, approximately 800–1000 nm) portions of the electromagnetic spectrum, as kelp
forests exhibit high reflectance in these bands, while surrounding water demonstrates much
lower reflectance due to its absorption of these longer wavelengths (Schroeder et al., 2019;
Timmer et al., 2024). However, when kelp tissue is fully submerged, the water absorbs red-edge
and NIR wavelengths, making it challenging to detect these submerged areas in satellite and
aerial imagery (Schroeder et al., 2019; Timmer et al., 2024).

Although studies have shown that the red-edge band can penetrate deeper into the water
column and detect more submerged kelp tissue, its availability is limited on older satellites, thus
the NIR band is still commonly used (Timmer et al., 2024). Both bands, however, struggle to
effectively detect fully submerged kelp, often leading to inaccuracies in mapping the full extent
of kelp forests (Timmer et al., 2024). While many studies have applied remote sensing
techniques to map and classify kelp forests, there is a notable gap in research exploring the use of
remote sensing to quantify the extent of kelp tissue covered by bryozoans.

This study aims to develop an algorithm for classifying bryozoan coverage on kelp tissues
using aerial imagery. The primary objective is to accurately detect and quantify the extent of
bryozoan coverage on kelp and to annotate the aerial data accordingly. These labeled datasets
will facilitate the training of a neural network, advancing the automation of bryozoan detection
in future analyses.

### 2.0 Study Area and Data

The study area, Cumshewa Inlet, is located off the coast of Haida Gwaii in Northern British
Columbia (Figure 1). This inlet is characterized by cold, nutrient-rich waters, which provide an
optimal environment for kelp forests. The focus of this study is to detect bryozoan coverage on
two main kelp species: Nereocystis luetkeana (Bull kelp) and Macrocystis pyrifera (Giant kelp)
using drone imagery captured from approximately 4–5 meters above the water.

Bull kelp features a single gas bladder connected to a stem-like structure called a stipe,
which can grow several meters in length and is anchored to the seafloor by a root-like system
(Timmer et al., 2024). The gas bladder supports up to 60 long, blade-like leaves, each capable of
reaching lengths of up to 4 meters. These blades trail downward into the water while the gas
bladder keeps the kelp buoyant at the surface (Timmer et al., 2024).

In contrast, giant kelp consists of numerous smaller structures called fronds (Timmer et al.,
2024). Each frond grows from a holdfast, a root-like structure that anchors the kelp to the ocean
floor. The fronds contain thin stalks with small air bladders, known as pneumatocysts, that help
keep the kelp afloat. Attached to each pneumatocyst is a flat, leaf-like blade that hangs
downward in the water, its weight exceeding the buoyant force of the air bladder (Timmer et al.,
2024). From an aerial perspective, the overlapping structures of giant kelp often appear as a
dense, indistinct mass, making them difficult to distinguish from each other (Timmer et al.,
2024).

The imagery used in this study was collected by Aidan Wright and his team through the
Spectral Lab at the University of Victoria. The images were captured using a DJI Mavic 3M
drone, flown at an altitude of 4 to 5 meters above the water. Multispectral bands in the green,
red, red-edge, and NIR portions of the spectrum were collected, along with RGB images (Table
1). These images were taken at different times of day to ensure the tide was at an optimal level of
below 2 meters, allowing for maximum visibility of the kelp.

Figure 1. The map above shows the study area, Cumshewa Inlet, located off the coast of Haida
Gwaii, BC. All imagery used in this study was collected by Aidan Wright and his team from the
Spectral Lab at the University of Victoria.

FIX::::: Table 1: The table below shows the wavelengths at which each band was collected, along with
the spatial resolution of the imagery.

Bands Wavelengths Spatial resolution
Green 560nm  16nm 2592 by 1944 pixels
Red 650nm  16nm 2592 by 1944 pixels
Red-edge 730nm  16nm 2592 by 1944 pixels
NIR 860nm  26nm 2592 by 1944 pixels
RGB image -- 4400 by 3305 pixels

### 3.0 Methods

This study utilized remote sensing techniques to classify and quantify bryozoan coverage on
kelp tissue. The process began by grouping multispectral and RGB images. The multispectral
images were upscaled to match the resolution of the RGB images, which were subsequently split
into their three individual channels. Using the red-edge band, a threshold was applied to
differentiate kelp tissue from the surrounding water. Once the water was removed from each
RGB channel, the pixel values classified as kelp were normalized by reclassifying the lowest
value to zero and the highest value to one. This normalization enhanced the contrast between
kelp and bryozoan pixels by stretching the pixel values across a standardized range.

To isolate bryozoan pixels, a threshold range was applied to each RGB image, generating
binary masks that were then refined using a morphological opening transformation. These
transformations helped remove noise and enhance the clarity of the bryozoan regions. Finally,
the binary masks from the three channels were combined into a single bryozoan mask. The
resulting masks were visually inspected and evaluated using a confusion matrix to assess the
model’s accuracy in detecting bryozoans on kelp tissue.

#### 3.1 Differentiating Spectrum Signatures

Thresholding is a fundamental technique in remote sensing, used to classify images or
separate pixels based on their spectral signatures (Schroeder et al., 2019). By applying a specific
threshold value, pixels are assigned to one of two binary classes: those with values above the
threshold are categorized as one class (e.g., 1), while those below the threshold are assigned to
another (e.g., 0).

In this study, thresholding was used to mask out water and differentiate kelp from bryozoan
coverage on kelp tissue. A binary mask was created using the red-edge band, with pixels below
the threshold classified as 0 (water) and those above classified as 1 (kelp). The red-edge band
was chosen over the NIR band due to its ability to penetrate deeper into the water column,
allowing for the detection of submerged kelp tissue at greater depths. To further distinguish kelp
tissue from bryozoan crust, a separate threshold range was applied to each channel of the RGB
images. These thresholds were initially determined by analyzing the electromagnetic histograms
for each band and were refined through iterative testing to optimize accuracy.

#### 3.2 Amplification and Normalization of Pixel Values

Normalization is a widely used technique in remote sensing to reclassify the range of pixel
values in an image, often by stretching or amplifying these values to fit a new scale. This process
maps the lowest pixel value in the original image to the new minimum and the highest pixel
value to the new maximum, redistributing all intermediate values proportionally across the new
range (1).

FIX:::: new_value = ( value−minorig maxorig−minorig) × (maxnew− minnew) + minnew (1)

As shown in Equation (1), the normalization process begins by linearly mapping the original
pixel values, which lie within a specific range, to a new range [𝑚𝑖𝑛new, 𝑚𝑎𝑥new]. This involves
subtracting the minimum value of the original range and dividing by the range's span (the
difference between the maximum and minimum). The normalized value is then scaled to fit the
new range and shifted by the new minimum value, ensuring that all pixel values are
proportionally adjusted to the desired scale.

#### 3.3 Refining Bryozoan Masks

Morphological transformations, grounded in set theory, are used to incorporate contextual
information into the analysis and classification of images (Tuia et al., 2009; Rishikeshan &
Ramesh, 2018). In this study, a morphological opening transformation was applied to remove
noise and refine the bryozoan binary masks generated through thresholding techniques. This
transformation consists of two sequential steps: erosion and dilation, both performed using a
defined kernel or moving window that scans the image in small segments. The opening
transformation is mathematically represented as:

(𝐴 ∘ 𝐵) = (𝐴 ⊖ 𝐵) ⊕ 𝐵 (2)

The Here, A represents the image, and B represents the structuring element (kernel). The
process begins with erosion, where the image A is eroded by the kernel B. This step shrinks
objects in the image by removing boundary pixels based on the kernel's shape, effectively
eliminating small structures and noise. Following erosion, dilation is applied to expand the
eroded objects by adding pixels back to their boundaries, restoring their original shape while
preserving the noise reduction (Destival, 1986; Tuia et al., 2009; Rishikeshan & Ramesh, 2018).
The size and shape of the kernel play a critical role in the transformation's performance.

Larger kernels create a more global effect, modifying larger areas of the image, whereas smaller
kernels produce a localized effect, preserving finer details and subtle features (Destival, 1986;
Tuia et al., 2009). In this study, a 3x3 kernel with a circular structure was selected. This
configuration was sufficient to remove noise without significantly altering the bryozoan masks,
while the circular shape complemented the natural circular structure of the bryozoans.

By combining erosion and dilation, the morphological opening transformation effectively
refines binary masks. Erosion removes unwanted pixels that do not match the kernel's structure,
eroding object boundaries. Dilation then restores the eroded objects by incorporating adjacent
pixels, ensuring that the final output preserves the original shapes of the objects while
maintaining noise reduction (Destival, 1986).

#### 3.4 Verifying the Accuracy of Bryozoan Masks

To evaluate the model's performance in classifying bryozoans on kelp tissue, a confusion
matrix was employed to derive key metrics: accuracy, precision, recall, and the F1 score. The
model-generated mask, created by classifying image pixels, was compared against a reference
mask manually annotated to identify areas with and without bryozoans. This reference mask
served as the ground truth for performance evaluation.

The confusion matrix offered a detailed breakdown of classification outcomes, allowing for
the quantification of the model's accuracy in distinguishing bryozoan-covered pixels from non-
bryozoan pixels. Accuracy, a general performance metric, represented the proportion of correctly
classified pixels. However, given the imbalance in the dataset, where bryozoan pixels were
significantly fewer than non-bryozoan pixels, accuracy alone was insufficient for a complete
evaluation.

Precision assessed the reliability of the model’s "bryozoan" predictions by measuring the
proportion of predicted bryozoan pixels that were correctly classified. In contrast, recall
evaluated the model's sensitivity, quantifying its ability to identify all true bryozoan pixels within
the image. The F1 score provided a harmonic mean of precision and recall, delivering a balanced
performance metric that was particularly valuable for addressing the challenges posed by
imbalanced datasets.

These metrics collectively offered a comprehensive assessment of the model’s reliability and
accuracy in detecting bryozoans on kelp tissue. By accounting for both false positives and false
negatives, the analysis ensured a robust evaluation of the model's capability to classify pixels
effectively.

### 4.0 Results

Five images were selected to verify the algorithm's accuracy in detecting bryozoans on kelp
tissue. These images were chosen due to their high bryozoan abundance and the presence of
challenging conditions, including overlapping non-kelp seaweed, water glare, and bryozoans
located at greater depths within the water column.

To evaluate the accuracy of the bryozoan masks generated by the algorithm, manual rasters
were created by isolating the portions of the images containing bryozoans. These manually
extracted regions were then compared with the algorithm-generated masks. Regions with
bryozoans located at depths beyond the effective range of the red-edge band were excluded from
the test data, as the band could not reliably detect kelp tissue in these areas. The results of the
five test rasters and their corresponding bryozoan masks are displayed in Figures 2–6.

Figure 2. The images above depict the first test site, which demonstrated the highest overall
performance (see table 2). The binary mask on the left was generated by the algorithm, while the
image on the right represents the manually created reference mask used to evaluate the accuracy
of the algorithm-generated bryozoan mask.

Figure 3. The images above show the second test site. The binary mask on the left was generated
by the algorithm, while the image on the right serves as a reference mask to assess the accuracy
of the bryozoan mask (See table 1). Although the mask captured the general trend of bryozoan
distribution on the site, there were instances of misclassifications, including errors in
distinguishing between kelp and bryozoan.

Figure 4. The binary mask from the third test site, which showed the lowest precision, is
displayed on the left. The image on the right serves as the reference for evaluating the accuracy
of the bryozoan mask generated by the algorithm.

Figure 5. The masks above are from the fourth test site. The binary mask on the left was
generated by the algorithm, while the image on the right serves as the reference for assessing the
accuracy of the bryozoan mask.

Figure 6. The mask above is from the fifth test site. The binary mask on the left was generated by
the algorithm, while the image on the right serves as the reference for assessing the accuracy of
the bryozoan mask. While the model was able to identify the general areas of bryozoan coverage,
it was unable to detect the full extent of it at this location.

Table 2: The table below presents the results calculated from each confusion matrix.
Test Site No. Accuracy Precision Recall F1 Score
1 0.9735 0.8597 0.2741 0.4157
2 0.9939 0.5442 0.2262 0.3196
3 0.9956 0.2097 0.1663 0.1855
4 0.9860 0.6191 0.1658 0.2616
5 0.9860 0.6191 0.1658 0.2616

### 5.0 Discussion

The algorithm developed in this study performed well in areas with a high abundance of
bryozoan on kelp tissue, as well as in regions with minimal surface-floating seaweed. However,
it struggled to detect bryozoan coverage in areas where the bryozoan was located at varying
depths in the water or when the kelp was obscured by other types of seaweed.

#### 5.1 Accuracy

The accuracy percentages for all five test sites were consistently above 97%, with results
varying by no more than two percentage points across sites (Table 2). While these values appear
promising, they can be misleading, as accuracy tends to favor the larger class (non-bryozoan, in
this case), inflating the overall score. This issue arises because accuracy does not account for
class imbalance, and the larger proportion of non-bryozoan pixels can disproportionately
increase the accuracy metric.

#### 5.2 Precision

Precision varied significantly across the test sites compared to accuracy, reflecting the
influence of sample size and the presence of bryozoan in each image. Precision is calculated as
the ratio of true positives to the sum of true positives and false positives. Sites with larger
amounts of bryozoan had a greater sample size, which improved the precision. This trend was
especially evident at test site one, where the highest proportion of bryozoan was present. The
algorithm achieved higher precision at this site, as the contrast between bryozoan pixels and kelp
pixels was more distinct when the image was enhanced, resulting in fewer false positives.

In contrast, test site three had the lowest precision rating. This site had a smaller amount
of bryozoan and contained other types of seaweed, such as eelgrass, floating on top of the kelp
canopy. These seaweeds were often misclassified as bryozoan, leading to an increase in false
positives. Furthermore, the image enhancement amplified noise, which further contributed to a
rise in false positives, as shown in Figure 7. These factors resulted in a significant decrease in
precision for test site three.

For test sites two, four, and five, the precision values ranged from 55% to 60%,
indicating that the algorithm performed moderately well at identifying bryozoan at these
locations. These sites exhibited characteristics similar to both site one and site three, with some
areas containing abundant bryozoan and others featuring seaweed. As a result, a precision range
of 55% to 60% was expected, as the algorithm had to contend with mixed image features and
varying amounts of bryozoan coverage across the sites.

Figure 7. The image above shows test site three. A large portion of the image is covered in a mix
of seaweed affecting the model’s ability to detect bryozoan.

#### 5.3 Recall

The recall rating for each test site represents the ratio of true positives to the sum of true
positives and false negatives. A lower recall indicates that the algorithm misclassified bryozoan
pixels as non-bryozoan, leading to false negatives. This is a key factor in evaluating the model's
performance, as false negatives signify missed detections of bryozoan, which can significantly
impact the results.

Test sites one and two had the highest recall values, though still relatively low. These
values were sufficient to identify the general locations of bryozoan in the images, though not
always with full accuracy. The lower recall values, particularly for deeper bryozoan, can be
attributed to the fact that bryozoan located at greater depths in the water column is much less
likely to be detected. As these deeper layers are obscured by water absorption, the algorithm’s
ability to capture all instances of bryozoan is diminished.
False negatives played a significant role in determining the recall scores, as they reflect
the model's inability to detect bryozoan in certain areas. This parameter highlights the
importance of reducing false negatives, as any missed bryozoan pixels lower the overall recall,
making it a critical factor in assessing the model’s detection capabilities.

#### 5.4 F1 Score

The F1 score provides a more balanced evaluation of model performance by combining
both precision and recall into a single metric. It is particularly valuable in situations with
imbalanced datasets, as it accounts for both false positives and false negatives. A higher F1 score
indicates a better model, with values closer to 1 denoting improved performance, while scores
closer to 0 suggest the model struggles with misclassification.

For test sites one and two, the F1 scores were approximately 0.4 and 0.3, respectively.
These scores suggest that the model performed moderately well at these sites, which contained a
relatively high abundance of bryozoan, making it easier for the algorithm to detect. However,
these values still indicate some room for improvement in terms of accuracy and detection.
The F1 scores for the remaining test sites ranged from 0.18 to 0.27, reflecting a
noticeable decrease in model performance. These lower scores suggest that the algorithm
struggled with more complex imagery, where bryozoan was less abundant relative to the surface
seaweed. The presence of other seaweeds, such as eelgrass, often led to misclassification, and the
reduced amount of bryozoan in the images made detection more challenging, further contributing
to the lower F1 scores at these sites.

### 7.0 Conclusion

This study developed an algorithm for classifying bryozoan coverage on kelp tissues
using aerial imagery, providing a valuable tool for monitoring the impact of the invasive species
bryozoan on kelp forests. The algorithm performed moderately well in detecting bryozoan
coverage, particularly in areas with high bryozoan abundance and minimal interference from
other seaweeds. However, the model faced challenges in detecting bryozoan at greater depths in
the water column and struggled with misclassifying other seaweed species, which share similar
spectral characteristics with bryozoans.

The evaluation of model performance through key metrics such as accuracy, precision,
recall, and F1 score indicated that while the algorithm performed adequately in simpler
environments, its effectiveness decreased in more complex settings with higher seaweed
diversity or bryozoans located at greater depths. The results also highlighted the importance of
enhancing image quality and incorporating additional spectral bands to mitigate issues with noise
and improve the differentiation between kelp and bryozoan pixels.

Despite these limitations, the findings from this study offer significant insights into the
application of remote sensing for monitoring bryozoan colonization in kelp forests. Future
research could focus on integrating additional spectral bands, such as the blue multispectral
band, to improve detection capabilities and exploring specialized indices that may enhance the
contrast between bryozoans and kelp. Expanding the number of test images and incorporating
more diverse environmental conditions into model training would also improve the robustness
and generalizability of the algorithm.

By addressing these limitations, future studies can refine the algorithm and contribute to
more accurate and efficient monitoring of bryozoan impacts on kelp ecosystems, ultimately
supporting better management strategies for preserving these critical marine habitats.

### 8.0 References

Destival, I. (1986). Mathematical morphology applied to remote sensing. Acta Astronautica,
13(6–7), 371–385. https://doi.org/10.1016/0094-5765(86)90092-5

Mora-Soto, A., Schroeder, S., Gendall, L., Wachmann, A., Narayan, G. R., Read, S., Pearsall, I.,
Rubidge, E., Lessard, J., Martell, K., Wills, P., & Costa, M. (2024). Kelp dynamics and
environmental drivers in the southern Salish Sea, British Columbia, Canada. Frontiers in
Marine Science, 11. https://doi.org/10.3389/fmars.2024.1323448

Pörtner HO, Roberts DC, Masson-Delmotte V, Zhai P and others. (2019). IPCC special report
on the ocean and cryo- sphere in a changing climate. Geneva.

Pratt, C. J., Denley, D., & Metaxas, A. (2022). Selection of predictor variables for species
distribution models: a case study with an invasive marine bryozoan. Oecologia, 198(2),
319–336. https://doi.org/10.1007/s00442-022-05110-1

Rishikeshan, C. A., & Ramesh, H. (2018). An automated mathematical morphology driven
algorithm for water body extraction from remotely sensed images. ISPRS Journal of
Photogrammetry and Remote Sensing: Official Publication of the International Society
for Photogrammetry and Remote Sensing (ISPRS), 146, 11–21.
https://doi.org/10.1016/j.isprsjprs.2018.08.014

Saier, B., & Chapman, A. S. (2004). Crusts of the alien bryozoan Membranipora membranacea
can negatively impact spore output from native kelps (Laminaria longicruris). Botanica
Marina, 47(4). https://doi.org/10.1515/bot.2004.031

Schroeder, S. B., Dupont, C., Boyer, L., Juanes, F., & Costa, M. (2019). Passive remote sensing
technology for mapping bull kelp (Nereocystis luetkeana): A review of techniques and
regional case study. Global Ecology and Conservation, 19(e00683), e00683.
https://doi.org/10.1016/j.gecco.2019.e00683

Starko, S., Timmer, B., Reshitnyk, L., Csordas, M., McHenry, J., Schroeder, S., Hessing-Lewis,
M., Costa, M., Zielinksi, A., Zielinksi, R., Cook, S., Underhill, R., Boyer, L., Fretwell,
C., Yakimishyn, J., Heath, W. A., Gruman, C., Hingmire, D., Baum, J. K., & Neufeld, C.
J. (2024). Local and regional variation in kelp loss and stability across coastal British
Columbia. Marine Ecology Progress Series, 733, 1–26.
https://doi.org/10.3354/meps14548

Teagle, H., Hawkins, S. J., Moore, P. J., & Smale, D. A. (2017). The role of kelp species as
biogenic habitat formers in coastal marine ecosystems. Journal of Experimental Marine
Biology and Ecology, 492, 81–98. https://doi.org/10.1016/j.jembe.2017.01.017

Timmer, B., Reshitnyk, L. Y., Hessing-Lewis, M., Juanes, F., Gendall, L., & Costa, M. (2024).
Capturing accurate kelp canopy extent: integrating tides, currents, and species-level
morphology in kelp remote sensing. Frontiers in Environmental Science, 12.
https://doi.org/10.3389/fenvs.2024.1338483

Tuia, D., Pacifici, F., Kanevski, M., & Emery, W. J. (2009). Classification of very high spatial
resolution imagery using mathematical morphology and support vector machines. IEEE
Transactions on Geoscience and Remote Sensing: A Publication of the IEEE Geoscience
and Remote Sensing Society, 47(11), 3866–3879.
https://doi.org/10.1109/tgrs.2009.2027895

Wernberg, T., Krumhansl, K., Filbee-Dexter, K., & Pedersen, M. F. (2019). Status and trends for
the world’s kelp forests. In World Seas: An Environmental Evaluation (pp. 57–78).
Elsevier.
