import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import csv
import time
import cv2
from PIL import Image
model_path='./model/model.ckpt'
import scipy
import numpy as np
from scipy import *
import matplotlib.pyplot as plt
import scipy.misc
import numpy as np
# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.' \
                                                            '  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # Define the name of the tensors
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # Get the needed layers' outputs for building FCN-VGG16
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    image_input = tf.get_default_graph().get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = tf.get_default_graph().get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out = tf.get_default_graph().get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4_out = tf.get_default_graph().get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7_out = tf.get_default_graph().get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out



def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # making sure the resulting shape are the same
    vgg_layer7_logits = tf.layers.conv2d(
        vgg_layer7_out, num_classes, kernel_size=1,
        kernel_initializer= tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-4), name='vgg_layer7_logits')
    vgg_layer4_logits = tf.layers.conv2d(
        vgg_layer4_out, num_classes, kernel_size=1,
        kernel_initializer= tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-4), name='vgg_layer4_logits')
    vgg_layer3_logits = tf.layers.conv2d(
        vgg_layer3_out, num_classes, kernel_size=1,
        kernel_initializer= tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-4), name='vgg_layer3_logits')

    # # Apply the transposed convolutions to get upsampled version, and then merge the upsampled layers
    fcn_decoder_layer1 = tf.layers.conv2d_transpose(
        vgg_layer7_logits, num_classes, kernel_size=4, strides=(2, 2),
        padding='same',
        kernel_initializer= tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-4), name='fcn_decoder_layer1')

    # add the first skip connection from the vgg_layer4_out
    fcn_decoder_layer2 = tf.add(
        fcn_decoder_layer1, vgg_layer4_logits, name='fcn_decoder_layer2')

    # then follow this with another transposed convolution layer and make shape the same as layer3
    fcn_decoder_layer3 = tf.layers.conv2d_transpose(
        fcn_decoder_layer2, num_classes, kernel_size=4, strides=(2, 2),
        padding='same',
        kernel_initializer= tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-4), name='fcn_decoder_layer3')

    # apply the same steps for the third layer output.
    fcn_decoder_layer4 = tf.add(
        fcn_decoder_layer3, vgg_layer3_logits, name='fcn_decoder_layer4')
    fcn_decoder_output = tf.layers.conv2d_transpose(
        fcn_decoder_layer4, num_classes, kernel_size=16, strides=(8, 8),
        padding='same',
        kernel_initializer= tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-4), name='fcn_decoder_layer4')

    return fcn_decoder_output



def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    # make logits a 2D tensor where each row represents a pixel and each column a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1,num_classes))
    # define loss function
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels= correct_label))
    # define training operation
    optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss



def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # Create log file
    log_filename = "./training_progress.csv"
    log_fields = ['learning_rate', 'exec_time (s)', 'training_loss']
    log_file = open(log_filename, 'w')
    log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    log_writer.writeheader()


    sess.run(tf.global_variables_initializer())

    lr = 0.0001

    print("Training...")
    print()
    for i in range(epochs):
        print("EPOCH {} ...".format(i+1))
        training_loss = 0
        training_samples = 0
        starttime = time.clock()
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: image, correct_label: label,
                                          keep_prob: 0.8, learning_rate: lr})
            print("batch loss: = {:.3f}".format(loss))
            training_samples += 1
            training_loss += loss

        training_loss /= training_samples
        endtime = time.clock()
        training_time = endtime-starttime

        print("Average loss for the current epoch: = {:.3f}\n".format(training_loss))
        log_writer.writerow({'learning_rate': lr, 'exec_time (s)': round(training_time, 2) , 'training_loss': round(training_loss,4)})
        log_file.flush()





def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)


    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # TODO: Build NN using load_vgg, layers, and optimize function
        epochs = 30
        batch_size = 8

        # TF placeholders
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)

        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate)

        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        print("Model is saved to file: %s" % save_path)

        # TODO: predict the testing data and save the augmented images
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)


def predict_images(frame, print_speed=False):
    num_classes = 2
    image_shape = (160, 576)
    runs_dir = './runs'

    # Path to vgg model
    vgg_path = os.path.join('./data', 'vgg')

    with tf.Session() as sess:
        # Predict the logits
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        logits = tf.reshape(nn_last_layer, (-1, num_classes))

        # Restore the saved model
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        print("Restored the saved Model in file: %s" % model_path)

        # Predict the samples
        return helper.pred_samples(frame,runs_dir, sess, image_shape, logits, keep_prob, input_image, print_speed)


def gen_output(frame,sess, logits, keep_prob, image_pl, image_shape):
 
    #for image_file in glob(os.path.join(data_folder, '*.png')):
    cv2.imshow("preview",frame)
    image = scipy.misc.imresize(scipy.misc.imread(frame), image_shape)
    
        #print(image.shape)
       # image.reshape(1,160,576,4)
    startTime = time.clock()
    im_softmax = sess.run(
        [tf.nn.softmax(logits)],
        {keep_prob: 1.0, input_image: [image]})
    im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    cv2.imshow("preview",image)
    street_im = scipy.misc.toimage(image)
    street_im.paste(mask, box=None, mask=mask)

    endTime = time.clock()
    speed_ = 1.0 / (endTime - startTime)

    #yield os.path.basename(image_file), np.array(street_im), speed_
    yield mask

def pred_samples(frame,output_dir, sess, image_shape, logits, keep_prob, input_image, print_speed=False):
    #cv2.imshow("preview",frame)
    # Make folder for current run
   # output_dir = os.path.join(runs_dir, str(time.time()))
   # if os.path.exists(output_dir):
   #     shutil.rmtree(output_dir)
   # os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    #print('Predicting images...')
    # start epoch training timer
    print(image_shape)
    #image_outputs = gen_output(
    #    frame,sess, logits, keep_prob, input_image, image_shape)
   # image = scipy.misc.imresize(scipy.misc.imread(frame), image_shape)
    #img = cv2.imread(frame)
    image = cv2.resize(frame, dsize=(576, 160), interpolation=cv2.INTER_CUBIC) 
        #print(image.shape)
       # image.reshape(1,160,576,4)
    startTime = time.clock()
    im_softmax = sess.run(
        [tf.nn.softmax(logits)],
        {keep_prob: 1.0, input_image: [image]})
    im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    
    street_im = scipy.misc.toimage(image)
    street_im.paste(mask, box=None, mask=mask)
    #cv2.imshow("preview",mask) 
    #plt.imshow(street_im)
    cv2.imshow("preview",np.asarray(street_im))
    endTime = time.clock()
    speed_ = 1.0 / (endTime - startTime)
    #img = Image.fromarray(image_outputs)
    #scipy.misc.imshow(img)
    #print(image_outputs)
    #scipy.misc.imsave(os.path.join(output_dir, "ad"),image_outputs)
    counter = 0
    #for name, image, speed_ in image_outputs:
     #   scipy.misc.imsave(os.path.join(output_dir, name), image)
      #  if print_speed is True:
       #     counter+=1
        #    print("Processing file: {0:05d},\tSpeed: {1:.2f} fps".format(counter, speed_))

        # sum_time += laptime

    # pngCounter = len(glob1(data_dir,'*.png'))

    #print('All augmented images are saved to: {}.'.format(output_dir))
    return street_im


if __name__ == '__main__':

    training_flag = False   # True: train the NN; False: predict with trained NN

    if training_flag:
        # run unittest before training
        tests.test_load_vgg(load_vgg, tf)
        tests.test_layers(layers)
        tests.test_optimize(optimize)
        tests.test_train_nn(train_nn)

        # train the NN and save the model
        run()
    else:
        cv2.namedWindow("preview")
        vc = cv2.VideoCapture(2)
        if vc.isOpened(): # try to get the first frame
            rval, frame = vc.read()
        else:
            rval = False
        num_classes = 2
        image_shape = (160, 576)
        runs_dir = './runs'
        output_dir = os.path.join(runs_dir, str(time.time()))
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        # Path to vgg model
        vgg_path = os.path.join('./data', 'vgg')

        with tf.Session() as sess:
        # Predict the logits
            input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
            nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
            logits = tf.reshape(nn_last_layer, (-1, num_classes))

            # Restore the saved model
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            print("Restored the saved Model in file: %s" % model_path)




            while rval:
                #cv2.imshow("preview", frame)
                rval, frame = vc.read()
               # print('Predicting images...')
                # start epoch training timer
                pred_samples(frame,runs_dir, sess, image_shape, logits, keep_prob, input_image, print_speed=False)
               # image_outputs = gen_output(
                #frame,sess, logits, keep_prob, input_image, image_shape)
                #img = Image.fromarray(image_outputs)
                # use the pre-trained model to predict more images
                test_data_path = './data/data_road/testing/image_2'
                #cv2.imshow("preview",pred_samples(frame,runs_dir, sess, image_shape, logits, keep_prob, input_image, print_speed=False)) 
                #image_outputs.show()
               # print(image_outputs)
                key = cv2.waitKey(20)
                if key == 27: # exit on ESC
                    break
cv2.destroyWindow("preview")
