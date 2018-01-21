# Traffic Sign Detection
Traffic sign detection based on Faster R-CNN, implemented in Keras. The annotated dataset comes from German Traffic Sign Detection Benchmark.

NOTES:
- config.py contains all settings for the train or test run. The anchor box sizes are selected from [8, 16, 32, 64] and anchor ratios from [1:1, 1:2, 2:1].
- The base network shared by RPN and classifier are implemented in 2-layer FCnet, ZFnet and VGG16.

Example output:

![ex1](https://imgur.com/vkH58TY.png)
![ex2](https://imgur.com/y1cYou4.png)
![ex3](http://i.imgur.com/EbvGBaG.png)
![ex4](http://i.imgur.com/i5UAgLb.png)

