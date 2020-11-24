# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 09:18:53 2019

@author: Oyelade
"""

def crop_img(img, bbox):
    '''Crop an image using bounding box
    '''
    x,y,w,h = bbox
    return img[y:y+h, x:x+w]


def add_img_margins(img, margin_size):
    '''Add all zero margins to an image
    '''
    enlarged_img = np.zeros((img.shape[0]+margin_size*2, 
                             img.shape[1]+margin_size*2))
    enlarged_img[margin_size:margin_size+img.shape[0], 
                 margin_size:margin_size+img.shape[1]] = img
    return enlarged_img

def read_resize_img(fname, target_size=None, target_height=None, 
                    target_scale=None, gs_255=False, rescale_factor=None):
    '''Read an image (.png, .jpg, .dcm) and resize it to target size.
    '''
    if target_size is None and target_height is None:
        raise Exception('One of [target_size, target_height] must not be None')
    if path.splitext(fname)[1] == '.dcm':
        img = dicom.read_file(fname).pixel_array
    else:
        if gs_255:
            img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    if target_height is not None:
        target_width = int(float(target_height)/img.shape[0]*img.shape[1])
    else:
        target_height, target_width = target_size
    if (target_height, target_width) != img.shape:
        img = cv2.resize(
            img, dsize=(target_width, target_height), 
            interpolation=cv2.INTER_CUBIC)
    img = img.astype('float32')
    if target_scale is not None:
        img_max = img.max() if img.max() != 0 else target_scale
        img *= target_scale/img_max
    if rescale_factor is not None:
        img *= rescale_factor
    return img

def get_roi_patches(img, key_pts, roi_size=(256, 256)):
    '''Extract image patches according to a key points list
    '''
    def clip(v, minv, maxv):
        '''Clip a coordinate value to be within an image's bounds
        '''
        v = minv if v < minv else v
        v = maxv if v > maxv else v
        return v

    patches = np.zeros((len(key_pts),) + roi_size, dtype='float32')
    for i, kp in enumerate(key_pts):
        if isinstance(kp, np.ndarray):
            xc, yc = kp
        else:
            xc, yc = kp.pt
        x = int(xc - roi_size[1]/2)
        x = clip(x, 0, img.shape[1])
        y = int(yc - roi_size[0]/2)
        y = clip(y, 0, img.shape[0])
        roi = img[y:y+roi_size[0], x:x+roi_size[1]]
        patch = np.zeros(roi_size)
        patch[0:roi.shape[0], 0:roi.shape[1]] = roi
        patches[i] = patch

    return patches


def sweep_img_patches(img, patch_size, stride, target_scale=None, 
                      equalize_hist=False):
    nb_row = round(float(img.shape[0] - patch_size)/stride + .49)
    nb_col = round(float(img.shape[1] - patch_size)/stride + .49)
    nb_row = int(nb_row)
    nb_col = int(nb_col)
    sweep_hei = patch_size + (nb_row - 1)*stride
    sweep_wid = patch_size + (nb_col - 1)*stride
    y_gap = int((img.shape[0] - sweep_hei)/2)
    x_gap = int((img.shape[1] - sweep_wid)/2)
    patch_list = []
    for y in xrange(y_gap, y_gap + nb_row*stride, stride):
        for x in xrange(x_gap, x_gap + nb_col*stride, stride):
            patch = img[y:y+patch_size, x:x+patch_size].copy()
            if target_scale is not None:
                patch_max = patch.max() if patch.max() != 0 else target_scale
                patch *= target_scale/patch_max
            if equalize_hist:
                patch = cv2.equalizeHist(patch.astype('uint8'))
            patch_list.append(patch.astype('float32'))
    return np.stack(patch_list), nb_row, nb_col


def read_breast_imgs(breast_dat, **kwargs):
            '''Read the images for both views for a breast
            '''
            #!!! if a view is missing, use a zero-filled 2D array.
            #!!! this may need to be changed depending on the deep learning design.
            if breast_dat['CC'] is None:
                img_cc = np.zeros(self.target_size, dtype='float32')
            else:
                img_cc = draw_img(breast_dat['CC'], **kwargs)
            if breast_dat['MLO'] is None:
                img_mlo = np.zeros(self.target_size, dtype='float32')
            else:
                img_mlo = draw_img(breast_dat['MLO'], **kwargs)
            # Convert all to lists of image arrays.
            if not isinstance(img_cc, list):
                img_cc = [img_cc]
            if not isinstance(img_mlo, list):
                img_mlo = [img_mlo]
            # Reshape each image array in the image lists.
            for i, img_cc_ in enumerate(img_cc):
                # Always have one channel.
                if self.data_format == 'channels_first':
                    img_cc[i] = img_cc_.reshape((1, img_cc_.shape[0], img_cc_.shape[1]))
                else:
                    img_cc[i] = img_cc_.reshape((img_cc_.shape[0], img_cc_.shape[1], 1))
            for i, img_mlo_ in enumerate(img_mlo):
                if self.data_format == 'channels_first':
                    img_mlo[i] = img_mlo_.reshape((1, img_mlo_.shape[0], img_mlo_.shape[1]))
                else:
                    img_mlo[i] = img_mlo_.reshape((img_mlo_.shape[0], img_mlo_.shape[1], 1))
            # Only predictin mode needs lists of image arrays.
            if not self.prediction_mode:
                img_cc = img_cc[0]
                img_mlo = img_mlo[0]

            return (img_cc, img_mlo)
            
        # build batch of image data
        adv = 0
        # last_eidx = None
        for eidx in index_array:
            # last_eidx = eidx  # no copying because sampling a diff img is expected.
            subj_id = self.exam_list[eidx][0]
            exam_idx = self.exam_list[eidx][1]
            exam_dat = self.exam_list[eidx][2]

            img_cc, img_mlo = read_breast_imgs(exam_dat['L'], exam=self.exam_list[eidx])
            if not self.prediction_mode:
                batch_x_cc[adv] = img_cc
                batch_x_mlo[adv] = img_mlo
                adv += 1
            else:
                # left_cc = img_cc
                # left_mlo = img_mlo
                batch_x_cc.append(img_cc)
                batch_x_mlo.append(img_mlo)

            img_cc, img_mlo = read_breast_imgs(exam_dat['R'], exam=self.exam_list[eidx])
            if not self.prediction_mode:
                batch_x_cc[adv] = img_cc
                batch_x_mlo[adv] = img_mlo
                adv += 1
            else:
                # right_cc = img_cc
                # right_mlo = img_mlo
                batch_x_cc.append(img_cc)
                batch_x_mlo.append(img_mlo)
                batch_subj.append(subj_id)
                batch_exam.append(exam_idx)

        # transform and standardize.
        for i in xrange(current_batch_size):
            if self.prediction_mode:
                for ii, img_cc_ in enumerate(batch_x_cc[i]):
                    if not np.all(img_cc_ == 0):
                        batch_x_cc[i][ii] = \
                            self.image_data_generator.standardize(img_cc_)
                for ii, img_mlo_ in enumerate(batch_x_mlo[i]):
                    if not np.all(img_mlo_ == 0):
                        batch_x_mlo[i][ii] = \
                            self.image_data_generator.standardize(img_mlo_)
            else:
                if not np.all(batch_x_cc[i] == 0):
                    if not self.validation_mode:
                        batch_x_cc[i] = self.image_data_generator.\
                            random_transform(batch_x_cc[i])
                    batch_x_cc[i] = self.image_data_generator.\
                        standardize(batch_x_cc[i])
                if not np.all(batch_x_mlo[i] == 0):
                    if not self.validation_mode:
                        batch_x_mlo[i] = self.image_data_generator.\
                            random_transform(batch_x_mlo[i])
                    batch_x_mlo[i] = self.image_data_generator.\
                        standardize(batch_x_mlo[i])

        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            def save_aug_img(img, bi, view, ii=None):
                '''Save an augmented image
                Args:
                    img (array): image array.
                    bi (int): breast index.
                    view (str): view name.
                    ii ([int]): (within breast) image index.
                '''
                if not self.prediction_mode:
                    fname_base = '{prefix}_{index}_{view}_{hash}'.\
                        format(prefix=self.save_prefix, index=bi, view=view, 
                               hash=rng.randint(1e4))
                else:
                    fname_base = '{prefix}_{bi}_{view}_{ii}_{hash}'.\
                        format(prefix=self.save_prefix, bi=bi, view=view, ii=ii,
                               hash=rng.randint(1e4))
                fname = fname_base + '.' + self.save_format
                if self.data_format == 'channels_first':
                    img = img.reshape((img.shape[1], img.shape[2]))
                else:
                    img = img.reshape((img.shape[0], img.shape[1]))
                # it seems only 8-bit images are supported.
                cv2.imwrite(path.join(self.save_to_dir, fname), img)


            for i in xrange(current_batch_size):
                if not self.prediction_mode:
                    img_cc = batch_x_cc[i]
                    img_mlo = batch_x_mlo[i]
                    save_aug_img(img_cc, current_index*2 + i, 'cc')
                    save_aug_img(img_mlo, current_index*2 + i, 'mlo')
                else:
                    for ii, img_cc_ in enumerate(batch_x_cc[i]):
                        save_aug_img(img_cc_, current_index*2 + i, 'cc', ii)
                    for ii, img_mlo_ in enumerate(batch_x_mlo[i]):
                        save_aug_img(img_mlo_, current_index*2 + i, 'mlo', ii)
        

        # build batch of labels
        flat_classes = self.classes[index_array, :].ravel()  # [L, R, L, R, ...]
        # flat_classes = flat_classes[np.logical_not(np.isnan(flat_classes))]
        flat_classes[np.isnan(flat_classes)] = 0  # fill in non-cancerous labels.
        if self.class_mode == 'sparse':
            batch_y = flat_classes
        elif self.class_mode == 'binary':
            batch_y = flat_classes.astype('float32')
        elif self.class_mode == 'categorical':
            batch_y = to_categorical(flat_classes, self.nb_class)
        else:  # class_mode is None.
            if self.prediction_mode:
                return [batch_subj, batch_exam, batch_x_cc, batch_x_mlo]
            else:
                return [batch_x_cc, batch_x_mlo]
        if self.prediction_mode:
            return [batch_subj, batch_exam, batch_x_cc, batch_x_mlo], batch_y
        else:
            return [batch_x_cc, batch_x_mlo], batch_y
        #### An illustration of what is returned in prediction mode: ####
        # let exam_blob = next(pred_datgen_exam)
        #
        # then           exam_blob[0][1][0][0]
        #                          /  |   \  \
        #                         /   |    \  \
        #                        /    |     \  \--- 1st img
        #                       img   cc    1st
        #                      tuple view  breast
        #
        # if class_mode is None, then the first index is not needed.