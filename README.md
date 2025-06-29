# The Classification of Bryozoan on Kelp Tissue Using Aerial Imagery

<b>Frankie Pothier
December 19th, 2024</b>

<b>Abstract: <b/> This study develops an algorithm for classifying bryozoans on kelp tissues using
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

<b> 1.0 Introduction</b>

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
