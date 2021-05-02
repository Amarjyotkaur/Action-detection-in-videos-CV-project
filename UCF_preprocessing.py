import numpy as np
import scipy.misc
import os, cv2, random
import shutil
import scipy.misc
import time
from PIL import Image
from pre_processing.optical_flow_prep import optical_flow_prep


#creates list of train and test directories
def combine_list_txt(list_dir):
    testlisttxt = 'testlist.txt'
    trainlisttxt = 'trainlist.txt'

    testlist = []
    txt_path = os.path.join(list_dir, testlisttxt)
    with open(txt_path) as fo:
        for line in fo:
            testlist.append(line[:line.rfind(' ')])

    trainlist = []
    txt_path = os.path.join(list_dir, trainlisttxt)
    with open(txt_path) as fo:
        for line in fo:
            trainlist.append(line[:line.rfind(' ')])

    return trainlist, testlist


def process_frame(frame, img_size, x, y, mean=None, normalization=True):

    
    framearr = np.asarray(frame)
    frame = np.array(Image.fromarray(framearr).resize(img_size))
    #x = random.randrange(frame.shape[0] - img_size[0])
    #y = random.randrange(frame.shape[1] - img_size[1])
    #frame = frame[x:x + img_size[0], y:y + img_size[1], :]
    #frame= scipy.misc.imresize(frame, img_size) DEPRECATED
    frame = frame.astype(dtype='float16')
    if mean is not None:
        frame -= mean
    if normalization:
        frame /= 255

    return frame


#  creates frames from videos and the calls process_frame function and down sample image resolution to 216*216, and make sequence length 10
def process_clip(src_dir, dst_dir, seq_len, img_size, mean=None, normalization=True,
                 consistent=True, continuous_seq=False):

    all_frames = []                                   
    cap = cv2.VideoCapture(src_dir)
    while cap.isOpened():
        succ, frame = cap.read()
        if not succ:
            break
        # append frame that is not all zeros
        if frame.any():
            all_frames.append(frame)
    # save all frames
    if seq_len is None:
        all_frames = np.stack(all_frames, axis=0)
        dst_dir = os.path.splitext(dst_dir)[0] + '.npy'
        np.save(dst_dir, all_frames)
    else:
        clip_length = len(all_frames)
        if clip_length <= 20:
            print(src_dir, ' has no enough frames')
        step_size = int(clip_length / (seq_len + 1))
        frame_sequence = []
        # select random first frame index for continuous sequence
        if continuous_seq:
            print(clip_length)
            print(seq_len)
            print(clip_length-seq_len)
            print(src_dir)
            start_index = random.randrange(clip_length-seq_len)
        x, y = None, None
        xy_set = False
        for i in range(seq_len):
            if continuous_seq:
                index = start_index + i
            else:
                index = i*step_size + random.randrange(step_size)
            frame = all_frames[index]
            print(type(frame))
            frame = process_frame(frame, img_size, x, y, mean=mean, normalization=normalization)
            frame_sequence.append(frame)
        frame_sequence = np.stack(frame_sequence, axis=0)
        dst_dir = os.path.splitext(dst_dir)[0]+'.npy'
        np.save(dst_dir, frame_sequence)

    cap.release()


def preprocessing(list_dir, smtsmt_dir, dest_dir, seq_len, img_size, overwrite=False, normalization=True,
                  mean_subtraction=True, horizontal_flip=True, random_crop=True, consistent=True, continuous_seq=False):
    '''
    Extract video data to sequence of fixed length, and save it in npy file
    :param list_dir:
    :param smtsmt_dir:
    :param dest_dir:
    :param seq_len:
    :param img_size:
    :param overwrite: whether overwirte dest_dir
    :param normalization: normalize to (0, 1)
    :param mean_subtraction: subtract mean of RGB channels
    :param horizontal_flip: add random noise to sequence data
    :param random_crop: cropping using random location
    :param consistent: whether horizontal flip, random crop is consistent in the sequence
    :param continuous_seq: whether frames extracted are continuous
    :return:
    '''
    if os.path.exists(dest_dir):
        if overwrite:
            shutil.rmtree(dest_dir)
        else:
            raise IOError('Destination directory already exists')
    os.mkdir(dest_dir)
    trainlist, testlist = combine_list_txt(list_dir)
    train_dir = os.path.join(dest_dir, 'train')
    test_dir = os.path.join(dest_dir, 'test')
    os.mkdir(train_dir)
    os.mkdir(test_dir)
    if mean_subtraction:
        mean = calc_mean(smtsmt_dir, img_size).astype(dtype='float16')
        np.save(os.path.join(dest_dir, 'mean.npy'), mean)
    else:
        mean = None
    
    print('Preprocessing something something data ...')
    for clip_list, sub_dir in [(trainlist, train_dir), (testlist, test_dir)]:
        for clip in clip_list:
            clip_name = os.path.basename(clip)
            clip_category = os.path.dirname(clip)
            category_dir = os.path.join(sub_dir, clip_category)
            src_dir = os.path.join(smtsmt_dir, clip)
            dst_dir = os.path.join(category_dir, clip_name)
            # print(dst_dir)
            if not os.path.exists(category_dir):
                os.mkdir(category_dir)
            process_clip(src_dir, dst_dir, seq_len, img_size, mean=mean, normalization=normalization, consistent=consistent, continuous_seq=continuous_seq)
    print('Preprocessing done ...')


def calc_mean(smtsmt_dir, img_size):
    frames = []
    print('Calculating RGB mean ...')
    for dirpath, dirnames, filenames in os.walk(smtsmt_dir):
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            if os.path.exists(path):
                cap = cv2.VideoCapture(path)
                if cap.isOpened():
                    ret, frame = cap.read()
                    # successful read and frame should not be all zeros
                    if ret and frame.any():
                        if frame.shape != (240, 320, 3):
                            frame = scipy.misc.imresize(frame, (240, 320, 3))
                        frames.append(frame)
                cap.release()
    frames = np.stack(frames)
    mean = frames.mean(axis=0, dtype='int64')
    mean = scipy.misc.imresize(mean, img_size)
    print('RGB mean is calculated over', len(frames), 'video frames')
    return mean


def preprocess_listtxt(list_dir, index_dir, txt_dir, dest_dir):
    class_dict = dict()
    with open(index_dir) as fo:
        for line in fo:
            class_index, class_name = line.split()
            class_dict[class_name] = class_index

    with open(txt_dir, 'r') as fo:
        lines = [line for line in fo]

    with open(dest_dir, 'w') as fo:
        for line in lines:
            class_name = os.path.dirname(line)
            class_index = class_dict[class_name]
            fo.write(line.rstrip('\n') + ' {}\n'.format(class_index))


def preprocess_flow_image(flow_dir):
    videos = os.listdir(flow_dir)
    for video in videos:
        video_dir = os.path.join(flow_dir, video)
        flow_images = os.listdir(video_dir)
        for flow_image in flow_images:
            flow_image_dir = os.path.join(video_dir, flow_image)
            img = scipy.misc.imread(flow_image_dir)
            if np.max(img) < 140 and np.min(img) > 120:
                print('remove', flow_image_dir)
                os.remove(flow_image_dir)


def regenerate_data(data_dir, list_dir, smtsmt_dir):
    '''
    calls data and times the preprocessing function above
    '''
    start_time = time.time()
    sequence_length = 10
    image_size = (216, 216)

    dest_dir = os.path.join(data_dir, 'smtsmt-Preprocessed-OF')
    # generate sequence for optical flow
    preprocessing(list_dir, smtsmt_dir, dest_dir, sequence_length, image_size, overwrite=True, normalization=False,
                  mean_subtraction=False, horizontal_flip=False, random_crop=True, consistent=True, continuous_seq=True)

    # compute optical flow data
    src_dir = 'D:\SUTD\Term-7\Deep_Learning\BigProject\Action_Detection_In_Videos\data\smtsmt-Preprocessed-OF'
    dest_dir = 'D:\SUTD\Term-7\Deep_Learning\BigProject\Action_Detection_In_Videos\data\OF_data'
    optical_flow_prep(src_dir, dest_dir, mean_sub=True, overwrite=True)

    elapsed_time = time.time() - start_time
    print('Regenerating data takes:', int(elapsed_time / 60), 'minutes')

#Need to change the directories below to our directories
if __name__ == '__main__':
    '''
        extract frames from videos as npy files
    '''
    sequence_length = 10
    image_size = (216, 216, 3)

    data_dir = 'D:\SUTD\Term-7\Deep_Learning\BigProject\Action_Detection_In_Videos\data'
    list_dir = os.path.join(data_dir, 'TrainTestlist')
    smtsmt_dir = os.path.join(data_dir, 'smtsmt')
    #frames_dir = os.path.join(data_dir, 'frames/mean.npy')

    regenerate_data(data_dir, list_dir, smtsmt_dir)