# -*- coding:utf-8 -*-
from preprocessing import load_image, mask
import matplotlib.pyplot as plt
from googlenet_model import googlenet

def main():
    training_set = load_image()
    training_set = mask(training_set)
    # plt.imshow(training_set[0])
    # plt.show()
    results = googlenet(training_set,
            dropout_keep_prob=0.4, 
            num_classes=2, 
            is_training=True, 
            restore_logits = None, 
            scope='')
    print results[0]



if __name__ == '__main__':
    main()