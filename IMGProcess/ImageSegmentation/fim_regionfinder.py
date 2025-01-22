import numpy as np
from PIL import Image
from skimage.measure import label, regionprops
from sklearn.linear_model import Lasso, Ridge, LassoCV

import matplotlib.pyplot as plt


def divideRegion(region, maxArea):
    if region.area <= maxArea:
        offset = np.array([0, 0])  # coordinate offset in pixels
        return [region], [offset]
    else:
        threshold = region.mean_intensity + 0.01 * (region.max_intensity - region.mean_intensity)
        mask = region.intensity_image > threshold
        labelImage = label(mask)
        subregions = regionprops(labelImage, intensity_image = region.intensity_image)
        regionList = []
        offsetList = []
        for subregion in subregions:
            newSubregions, newOffsets = divideRegion(subregion, maxArea)
            for newOffset in newOffsets:
                newOffset += np.array(region.bbox[:2])
            regionList += newSubregions
            offsetList += newOffsets
        return regionList, offsetList


def findHighestSubregion(region, maxArea):
    if region.area <= maxArea:
        offset = np.array([0, 0])  # coordinate offset in pixels
        return region, offset
    else:
        threshold = (region.max_intensity + region.mean_intensity) / 2.0
        mask = region.intensity_image > threshold
        labelImage = label(mask)
        subregions = regionprops(labelImage, intensity_image = region.intensity_image)
        maxI = region.min_intensity
        for subregion in subregions:
            if subregion.max_intensity > maxI:
                maxI = subregion.max_intensity
                highestSubregion = subregion
        newRegion, offset = findHighestSubregion(highestSubregion, maxArea)
        return newRegion, offset + np.array(region.bbox[:2])


def FindPotentialLocations(im, maskThreshold, detThreshold, minArea, maxArea):
    mask = im > maskThreshold
    labelImg = label(mask)
    regions = regionprops(labelImg, intensity_image=im)
    
    internalRegions = []
    maxIndices = []
    for region in regions:
        offset = np.array([0, 0])
        region.offset = offset
        
        if region.area < minArea:
            continue
        if region.max_intensity < detThreshold:
            continue

        if region.area > maxArea:
            newRegions, newOffsets = divideRegion(region, maxArea)
            for nr, no in zip(newRegions, newOffsets):
                nr.offset = no
                internalRegions.append(nr)
        else:
            newRegions, newOffsets = [region], [offset]
            maxIndices.append(np.array(region.weighted_centroid) + offset)
            
        for newRegion, newOffset in zip(newRegions, newOffsets):
            maxIndices.append(np.array(newRegion.weighted_centroid) + newOffset)
    
    if len(maxIndices) == 0:
        maxIndices = np.zeros((0,)), np.zeros((0, ))
    else:
        maxIndices = np.array(maxIndices).T
    
    
    return maxIndices, [r for r in regions if r.area > minArea], internalRegions


def _gauss_render(imsz, centers, PSFsigmax, PSFsigmay):
    norm = 2.0 * np.pi * PSFsigmax * PSFsigmay
    imszy, imszx = imsz
    YY, XX = np.meshgrid(np.arange(imszy)+0.5, np.arange(imszx)+0.5, indexing='ij')
    ZZ = np.zeros((imszy, imszx))
    for center in centers:
        ZZ += 1.0 / norm * np.exp(-0.5*(XX-center[0])**2/PSFsigmax**2 - 0.5*(YY-center[1])**2/PSFsigmay**2)
    return ZZ


def _gauss_projection(imsz, nbins, PSFsigmax, PSFsigmay):
    PSFsigmax = PSFsigmax * nbins
    PSFsigmay = PSFsigmay * nbins
    norm = 2.0 * np.pi * PSFsigmax * PSFsigmay
    
    imszy_i, imszx_i = imsz
    YY, XX = np.meshgrid(np.arange(imszy_i)+0.5, np.arange(imszx_i)+0.5, indexing='ij')
    imvec_yy = YY.flatten() * nbins
    imvec_xx = XX.flatten() * nbins

    YY, XX = np.meshgrid(np.arange(imszy_i*nbins)+0.5, np.arange(imszx_i*nbins)+0.5, indexing='ij')
    csvec_yy = YY.flatten()
    csvec_xx = XX.flatten()

    projection = np.zeros((imszy_i*imszx_i, imszy_i*imszx_i*nbins*nbins))
    for i in range(imszy_i * imszx_i):
        for j in range(imszy_i * nbins * imszx_i * nbins):
            projection[i, j] = 1.0 / norm * np.exp(-0.5 * ((imvec_xx[i]-csvec_xx[j])**2 / PSFsigmax**2 + (imvec_yy[i]-csvec_yy[j])**2 / PSFsigmay**2))

    return projection



if __name__ == '__main__':

    nbins = 3
    PSFsigmax = 1.45
    PSFsigmay = 1.45
    
    fname = 'D:/Data/simulations/20240721_loctest/ndim_3_nch_1_density_2.00_EM_01/GTImStack.tif'
    with Image.open(fname, 'r') as imobj:
        imszy = imobj.height
        imszx = imobj.width
        im = np.array(imobj, dtype=np.float64)
    projection = _gauss_projection((imszy, imszx), nbins, PSFsigmax, PSFsigmay)


    test_lasso = LassoCV().fit(projection, im.flatten())
    rec_l1 = test_lasso.coef_.reshape((imszy*nbins, imszx*nbins))
    print(test_lasso.alpha_)
    
    

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(im)
    axs[0].set_aspect('equal')
    axs[1].imshow(rec_l1)
    axs[1].set_aspect('equal')
    plt.show()


