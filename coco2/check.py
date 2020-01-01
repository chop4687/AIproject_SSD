import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import skimage.io as io
import numpy as np
import cv2
import pickle
coco = COCO('/home/coin/datasets/MS-COCO/annotations/instances_val2017.json')
# display COCO categories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

catIds1 = coco.getCatIds(catNms=['apple'])
imgIds1 = coco.getImgIds(catIds=catIds1)
imgIds1 = coco.getImgIds(catIds=catIds1)

catIds2 = coco.getCatIds(catNms=['banana'])
imgIds2 = coco.getImgIds(catIds=catIds2)
imgIds2 = coco.getImgIds(catIds=catIds2)

catIds3 = coco.getCatIds(catNms=['broccoli'])
imgIds3 = coco.getImgIds(catIds=catIds3)
imgIds3 = coco.getImgIds(catIds=catIds3)

imgIds = imgIds1 + imgIds2 + imgIds3
imgIds = list(set(imgIds))
imgIds.sort()
result = []
for i in range(200):
    img = coco.loadImgs(imgIds[i])[0]
    temp = io.imread(img['coco_url'])

    io.imsave('/home/junkyu/SSD2/coco2/cocodata_val/'+str(imgIds[i])+'.jpg',temp)
    bbboxes = []
    if imgIds[i] in imgIds1:
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds1, iscrowd=None)
        anns = coco.loadAnns(annIds)

        for bbox in anns:
            bboxes = []
            bboxes.append(bbox['bbox'][0])
            bboxes.append(bbox['bbox'][1])
            bboxes.append(bbox['bbox'][0] + bbox['bbox'][2])
            bboxes.append(bbox['bbox'][1] + bbox['bbox'][3])
            bboxes.append(0)
            bbboxes.append(tuple(bboxes))

    if imgIds[i] in imgIds2:
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds2, iscrowd=None)
        anns = coco.loadAnns(annIds)

        for bbox in anns:
            bboxes = []
            bboxes.append(bbox['bbox'][0])
            bboxes.append(bbox['bbox'][1])
            bboxes.append(bbox['bbox'][0] + bbox['bbox'][2])
            bboxes.append(bbox['bbox'][1] + bbox['bbox'][3])
            bboxes.append(1)
            bbboxes.append(tuple(bboxes))

    if imgIds[i] in imgIds3:
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds3, iscrowd=None)
        anns = coco.loadAnns(annIds)

        for bbox in anns:
            bboxes = []
            bboxes.append(bbox['bbox'][0])
            bboxes.append(bbox['bbox'][1])
            bboxes.append(bbox['bbox'][0] + bbox['bbox'][2])
            bboxes.append(bbox['bbox'][1] + bbox['bbox'][3])
            bboxes.append(2)
            bbboxes.append(tuple(bboxes))

    result.append(bbboxes)

output = open('annotation_val.pkl', 'wb')
pickle.dump(result, output)
output.close()
