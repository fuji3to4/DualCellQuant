#@ ImagePlus(label="Target Image") imp_tar
#@ String(label="Target Threshold Method", required=true, choices={'Moments','IsoData','Otsu', 'Huang','0'}) threshold_tar
#@ ImagePlus(label="Reference Image") imp_ref
#@ String(label="Reference Threshold Method", required=true, choices={'Moments','IsoData','Otsu', 'Huang','0'}) threshold_ref
#@ boolean(label="Show Each Mask") show_each_mask

from ij import IJ
from ij import ImagePlus
from ij.plugin import ImageCalculator
from ij.process import Blitter
from ij.measure import ResultsTable

# List to store masks
all_masks_tar = []

# Process each slice separately in case of multi-page TIFF
for i in range(1, imp_tar.getStack().getSize() + 1):

    # Duplicate the target image for each slice
    imp_slice = ImagePlus("Slice " + str(i), imp_tar.getStack().getProcessor(i).duplicate())
    
    if threshold_tar !='0':
        method = threshold_tar + " dark"
        IJ.setAutoThreshold(imp_slice, method)
        ImProc = imp_slice.getProcessor()
        min_thr = ImProc.getMinThreshold()
    else:
        min_thr=0

    #max_thr = imp_slice.getStatistics().max - 1
    max_thr = 2 ** 12 -1
    print(min_thr, max_thr)

    IJ.setThreshold(imp_slice, min_thr, max_thr)
    mask_slice = imp_slice.createThresholdMask()
    IJ.resetThreshold(imp_slice)
    
    # Check if the mask is not None before appending it to the list
    if mask_slice:
        # Create a new ImagePlus for each slice and add it to the list
        mask_tar = ImagePlus("Mask_Tar_Slice" + str(i), mask_slice)
        all_masks_tar.append(mask_tar)

# Create a new ImagePlus with a stack of all masks
if all_masks_tar:
    stack_masks = all_masks_tar[0].getStack()  # Start with the first mask
    for i in range(1, len(all_masks_tar)):
        stack_masks.addSlice(all_masks_tar[i].getProcessor())

    # Create a new ImagePlus with the stack of masks
    imp_all_masks_tar = ImagePlus("Masks Tar", stack_masks)
    if show_each_mask:
        imp_all_masks_tar.show()

# List to store masks
all_masks_ref = []

# Process each slice separately in case of multi-page TIFF
for i in range(1, imp_ref.getStack().getSize() + 1):
    # Duplicate the target image for each slice
    imp_slice = ImagePlus("Slice " + str(i), imp_ref.getStack().getProcessor(i).duplicate())
    
    if threshold_ref !='0':
        method = threshold_ref + " dark"
        IJ.setAutoThreshold(imp_slice, method)
        ImProc = imp_slice.getProcessor()
        min_thr = ImProc.getMinThreshold()
    else:
        min_thr=0

    #max_thr = imp_slice.getStatistics().max - 1
    max_thr = 2 ** 12 -1
    print(min_thr, max_thr)

    IJ.setThreshold(imp_slice, min_thr, max_thr)
    mask_slice = imp_slice.createThresholdMask()
    IJ.resetThreshold(imp_slice)
    
    # Check if the mask is not None before appending it to the list
    if mask_slice:
        # Create a new ImagePlus for each slice and add it to the list
        mask_ref = ImagePlus("Mask_Ref_Slice" + str(i), mask_slice)
        all_masks_ref.append(mask_ref)

# Create a new ImagePlus with a stack of all masks
if all_masks_ref:
    stack_masks = all_masks_ref[0].getStack()  # Start with the first mask
    for i in range(1, len(all_masks_ref)):
        stack_masks.addSlice(all_masks_ref[i].getProcessor())

    # Create a new ImagePlus with the stack of masks
    imp_all_masks_ref = ImagePlus("Masks Ref", stack_masks)
    if show_each_mask:
        imp_all_masks_ref.show()


mask_both = ImageCalculator.run(imp_all_masks_tar, imp_all_masks_ref, "AND create stack")
IJ.run(mask_both, "Divide...", "value=255 stack")
mask_both.setTitle("Masks Both")
mask_both.setDisplayRange(0, 1)
if show_each_mask:
	mask_both.show()
	
mask_both_na = mask_both.duplicate()
mask_both_na.setTitle("Masks Both NA")
IJ.run(mask_both_na, "32-bit", "");
for i in range(1, mask_both_na.getStackSize() + 1):
	mask_both_na.setSlice(i)
	
	# Get ImageProcessor
	ip = mask_both_na.getProcessor()

    # Set pixels with 0 to NaN
	width = ip.getWidth()
	height = ip.getHeight()
	for x in range(width):
		for y in range(height):
			if ip.getPixelValue(x, y) == 0:
				ip.putPixelValue(x, y, float('nan'))

mask_both_na.updateAndDraw()
if show_each_mask:
	mask_both_na.show()

result_tar = ImageCalculator.run(imp_tar, mask_both_na, "Multiply create 32-bit stack")
result_ref = ImageCalculator.run(imp_ref, mask_both_na, "Multiply create 32-bit stack")
result_tar.show()
result_ref.show()


rt = ResultsTable()

for i in range(1, result_tar.getStackSize() + 1):
	rt.incrementCounter();	
	result_tar.setSlice(i)
	if result_tar.getStackSize() ==1:
		rt.addValue("Image", result_tar.getTitle());
	else:
		rt.addValue("Image", result_tar.getTitle()+"_Slice"+str(i));
	
	mean_tar = result_tar.getStatistics().mean
	area_tar = result_tar.getStatistics().area
	min_tar = result_tar.getStatistics().min
	max_tar = result_tar.getStatistics().max

	rt.addValue("Mean",mean_tar);
	rt.addValue("Area",area_tar);
	rt.addValue("Min",min_tar);
	rt.addValue("Max",max_tar);


	rt.incrementCounter();
	result_ref.setSlice(i)
	if result_tar.getStackSize() ==1:
		rt.addValue("Image", result_ref.getTitle());
	else:
		rt.addValue("Image", result_ref.getTitle()+"_Slice"+str(i));
		
	mean_ref = result_ref.getStatistics().mean
	area_ref = result_ref.getStatistics().area
	min_ref = result_ref.getStatistics().min
	max_ref = result_ref.getStatistics().max

	rt.addValue("Mean",mean_ref);
	rt.addValue("Area",area_ref);
	rt.addValue("Min",min_ref);
	rt.addValue("Max",max_ref);


rt.show("Results")