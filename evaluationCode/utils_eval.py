import numpy as np

# This function takes the prediction and label of a single image, returns intersection and union areas for each class
# To compute over many images do:
# for i in range(Nimages):
# 	(area_intersection[:,i], area_union[:,i]) = intersectionAndUnion(imPred[i], imLab[i])
# IoU = 1.0 * np.sum(area_intersection, axis=1) / np.sum(np.spacing(1)+area_union, axis=1)
def intersectionAndUnion(imPred, imLab, numClass):
	imPred = np.asarray(imPred)
	imLab = np.asarray(imLab)

	# Remove classes from unlabeled pixels in gt image. 
	# We should not penalize detections in unlabeled portions of the image.
	imPred = imPred * (imLab>0)

	# Compute area intersection:
	intersection = imPred * (imPred==imLab)
	(area_intersection,_) = np.histogram(intersection, bins=numClass, range=(1, numClass))

	# Compute area union:
	(area_pred,_) = np.histogram(imPred, bins=numClass, range=(1, numClass))
	(area_lab,_) = np.histogram(imLab, bins=numClass, range=(1, numClass))
	area_union = area_pred + area_lab - area_intersection
	
	return (area_intersection, area_union)

# This function takes the prediction and label of a single image, returns pixel-wise accuracy
# To compute over many images do:
# for i = range(Nimages):
#	(pixel_accuracy[i], pixel_correct[i], pixel_labeled[i]) = pixelAccuracy(imPred[i], imLab[i])
# mean_pixel_accuracy = 1.0 * np.sum(pixel_correct) / (np.spacing(1) + np.sum(pixel_labeled))
def pixelAccuracy(imPred, imLab):
	imPred = np.asarray(imPred)
	imLab = np.asarray(imLab)

	# Remove classes from unlabeled pixels in gt image. 
	# We should not penalize detections in unlabeled portions of the image.
	pixel_labeled = np.sum(imLab>0)
	pixel_correct = np.sum((imPred==imLab)*(imLab>0))
	pixel_accuracy = 1.0 * pixel_correct / pixel_labeled

	return (pixel_accuracy, pixel_correct, pixel_labeled)
