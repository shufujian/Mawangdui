# Mawangdui
## Structure Of This Reop

```shell
Mawangdui:
-character data
    -阿
        -阿_1.jpg
        -阿_2.jpg
    -哀
        -哀_1.jpg
        -哀_2.jpg
    ...
        ...
-code
    -BAGAN_GP
    -Classification
```

## The process for creating this dataset
The wood slips, bamboo slips, and silk manuscripts found in Changsha's Mawangdui Han Tomb had serious problems, including severe damage, the disappearance of text handwriting, corrosion, and darkening of old books.
![帛书图片](README.assets/帛书原图.jpg)

So "Mawangdui bamboo and silk word the whole series (full 3)" was chosen as the basis for making the dataset.The content pictures are shown below:
![全字编图片](README.assets/全字编.jpg)

Get all single Chinese character images by deep learning method. Finally the final dataset is obtained by manual sorting.
![单字图片](README.assets/单字图片.jpg)

## character data
This folder contains all thef the original dataset.
>All the Handwritten single-word images are from "*Mawangdui bamboo and silk word the whole series (full 3)*" (chinese:《马王堆汉墓简帛文字全编》).
>
>The dataset contains **3339** categories, with a total of **93,841** single-word images.
>
>The category of the dataset is a **Chinese character**, but some Chinese characters do not **exist** now.

![单字图片](README.assets/单字图片.png)
![单字图片](README.assets/单字图片.jpg)



## code

Contain two folders, BAGAN_GP and Classification.

### BAGAN_GP

Used to expand the dataset.

The output images is shown below.

<img src="README.assets/生成图片_2.png" alt="生成图片_2" style="zoom:50%;" />

> You can visit the [Original Repository](https://github.com/GH920/improved-bagan-gp) for more detailed information.

### Clasiffication

Use sample ResNet and DenseNet to train a Classification model。
