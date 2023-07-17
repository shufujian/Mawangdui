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
## character data
This folder contains all the images of the original dataset.
>All the Handwritten single-word images are from 《马王堆汉墓简帛文字全编》
>
>The dataset contains **3339** categories, with a total of **93,841** single-word images.
>
>The category of the dataset is a **Chinese character**, but some Chinese characters do not **exist** now.

![单字图片](README.assets/单字图片.png)



## code

Contain two folders, BAGAN_GP and Classification.

### BAGAN_GP

Used to expand the dataset.

The output images is shown below.

<img src="README.assets/生成图片_2.png" alt="生成图片_2" style="zoom:50%;" />

> You can visit the [Original Repository](https://github.com/GH920/improved-bagan-gp) for more detailed information.

### Clasiffication

Use sample ResNet and DenseNet to train a Classification model。
