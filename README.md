# Neural Image Styler

A configurable tf.keras based neural network image styler.
Based on [this tensorflow tutorial](https://github.com/tensorflow/models/tree/master/research/nst_blogpost)

## Command Line
```
usage: neural_image_styler [-h] --input INPUT --style_reference
                           STYLE_REFERENCE --output OUTPUT
                           [--iterations ITERATIONS]
                           [--content_weight CONTENT_WEIGHT]
                           [--style_weight STYLE_WEIGHT]
                           [--show_plots SHOW_PLOTS]

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         image file to be processed
  --style_reference STYLE_REFERENCE
                        image file to extract style from
  --output OUTPUT       output image file save location
  --iterations ITERATIONS
                        number of fit iterations
  --content_weight CONTENT_WEIGHT
  --style_weight STYLE_WEIGHT
  --show_plots SHOW_PLOTS
                        show debug plots

```

## Colab
This package can be experimented with in a colab environment

### Cloud Colab
See [this Colab](https://colab.research.google.com/drive/1urljZn4L132A5WANcnhlskX6pLr6G-tc#scrollTo=plYn0tZ5HJ-H) for a runnable colab, be sure to use a GPU runtime

### Local Colab
See [this guide](https://research.google.com/colaboratory/local-runtimes.html) for running a local Colab runtime to utilize local gpu resources.

## Example
The `output image` is the `content image` stylized by the `style image`.
![result](https://github.com/jake-g/neural_image_styler/blob/master/assets/result.png)

As more iterations are completed, the output begins to take shape.
![iterations](https://github.com/jake-g/neural_image_styler/blob/master/assets/iterations.png)

The final result has the lowest loss.
![final](https://github.com/jake-g/neural_image_styler/blob/master/assets/final.png)

