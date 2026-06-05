from import_py import *

def thru_plane_position(dcm):
    """Gets spatial coordinate of image origin whose axis
    is perpendicular to image plane.
    """
    orientation = tuple((float(o) for o in dcm.ImageOrientationPatient))
    position = tuple((float(p) for p in dcm.ImagePositionPatient))
    rowvec, colvec = orientation[:3], orientation[3:]
    normal_vector = np.cross(rowvec, colvec)
    slice_pos = np.dot(position, normal_vector)
    return slice_pos

def load_dcm(dcm_path, resize=False):
    dcm_files = []
    for dirName, subdirList, fileList in os.walk(dcm_path):
        for filename in fileList:
            if ".dcm" in filename.lower():
                dcm_files.append(os.path.join(dirName, filename))

    if len(dcm_files) == 0:
        raise RuntimeError("No dicom files")

    ref = read_file(dcm_files[0])

    dcms = []

    dim = len(dcm_files), ref.Rows, ref.Columns 

    ret = np.zeros(dim, dtype=ref.pixel_array.dtype)
    for i, dcm in enumerate(dcm_files):
        ds = read_file(dcm)
        dcms.append(ds)
        
    dcm_slices = [(dcm, thru_plane_position(dcm)) for dcm in dcms]
    dcm_slices = natsorted(dcm_slices, key=itemgetter(1), reverse=True) 
    
    initial_patient_position = dcm_slices[-1][0].ImagePositionPatient # (w, h, d) position

    spacings = np.diff([dcm_slice[1] for dcm_slice in dcm_slices])
    slice_spacing = np.mean(spacings)

    # All slices will have the same in-plane shape
    shape = (int(dcm_slices[0][0].Columns), int(dcm_slices[0][0].Rows))
    nslices = len(dcm_slices)

    # Final 3D array will be N_Slices x Columns x Rows
    shape = (nslices, *shape)

    # Calculate size of a voxel in mm
    pixel_spacing = tuple(float(spac) for spac in dcm_slices[0][0].PixelSpacing)
    voxel_spacing = (*pixel_spacing, slice_spacing) # (x, y, z)

    metainfo = (initial_patient_position, voxel_spacing)

    dcms = []
    for i, (dcm, _) in enumerate(dcm_slices):
        ret[i, :, :] = dcm.pixel_array
        dcms.append(dcm)

    zoom_ratio = None
    if resize:
        if dcms[0].PixelSpacing[0] != 0.2 or dcms[0].PixelSpacing[1] != 0.2:
            zoom_x = dcms[0].PixelSpacing[0] / 0.2
            zoom_y = dcms[0].PixelSpacing[1] / 0.2
            ret = zoom(ret, (zoom_x, zoom_y, zoom_x)) 
            dim = ret.shape
            zoom_ratio = (zoom_x, zoom_y, zoom_x) 

    sample = zoom(ret, 1/4)
    gm = GaussianMixture(n_components=2)
    gm.fit(np.reshape(sample, (-1, 1)))
    if gm.means_[0] < gm.means_[1]:
        mean1, mean2 = gm.means_
        cov1, cov2 = gm.covariances_
        cnt1, cnt2 = np.bincount(gm.predict(np.reshape(sample, (-1, 1))))
    else:
        mean2, mean1 = gm.means_
        cov2, cov1 = gm.covariances_
        cnt2, cnt1 = np.bincount(gm.predict(np.reshape(sample, (-1, 1))))
    min_thresh = mean1 - np.sqrt(cov1) * scipy.stats.t.ppf((1 + 0.999) / 2., cnt1 - 1)
    min_thresh = min_thresh[0][0]
    ret[ret < min_thresh] = min_thresh

    ret[ret>5000] = 5000

    if resize:
        return ret, dim, zoom_ratio
    return ret, dim, None, metainfo

def load_anno(path):
    with open(path, 'r') as f:
        return json.load(f)

def stretch(dim, lower_mat, ratio=12/10):
    trans_mat = np.array([
        [1., 0., 0., -dim[2]/2.],
        [0., 1., 0., -dim[1]/2.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]
    ])
    stretch_mat = np.array([
        [ratio, 0., 0., 0.],
        [0., ratio, 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]
    ])
    ret_mat = np.dot(np.linalg.inv(trans_mat), np.dot(stretch_mat, trans_mat))
    return ret_mat

def base_transform_mat(info, way, dim):
    theta = np.radians(info['angle'])
    # theta = np.radians(0.0)
    c, s = np.cos(theta), np.sin(theta)
    z_pad = MARGIN if way == LOWER else DEPTH - MARGIN
    translate = [[1, 0, 0, 0],
                 [0, 1, 0, -(info['p'][0])],
                 [0, 0, 1, -(info['p'][1])],
                 [0, 0, 0, 1]]
    r_x = [[1, 0, 0, 0],
           [0, c, -s, 0],
           [0, s, c, 0],
           [0, 0, 0, 1]]
    mat_pad = [[1, 0, 0, 0],
               [0, 1, 0, 25],
               [0, 0, 1, z_pad],
               [0, 0, 0, 1]]
    w_from = int((dim[2] - WIDTH) / 2)
    t_x = [[1, 0, 0, -w_from],
           [0, 1, 0, 0],
           [0, 0, 1, 0],
           [0, 0, 0, 1]]

    translate = np.array(translate)
    mat_pad = np.array(mat_pad)
    r_x = np.array(r_x)
    t_x = np.array(t_x)

    mat = t_x.dot(mat_pad.dot(r_x.dot(translate)))
    return mat

def affine(img, mat, out_size, interpolate=True):
    assert (img.ndim == 3 and mat.shape == (4, 4))
    
    print('--------')
    mat = np.linalg.inv(mat)
    img = np.swapaxes(img, 0, 2) # d h w -> w h d
    # print(img.shape)


    # print(f'mat : {mat}')

    # img = img.astype(np.float32)
    img = img.astype(np.int32)
    mat = mat.astype(np.float32)

    if interpolate:
        out = affine_transform(
            input=img,
            matrix=mat,
            output_shape=out_size,
            order=3,  # interpolation order
            mode='constant',
            cval=0.0,
            prefilter=True
        )
    else:
        out = affine_transform(
            input=img,
            matrix=mat,
            output_shape=out_size,
            order=0,  # no interpolation
            mode='constant',
            cval=0.0,
            prefilter=False
        )

    out = np.swapaxes(out, 0, 2)
    return out

def crop_and_zoom(img, mat, interpolate=True, crop_shape=(WIDTH, HEIGHT, DEPTH)):
    ret_size = np.array(crop_shape)
    ret = affine(img, mat, ret_size, interpolate=interpolate)
    ret = np.squeeze(ret)
    return ret

def load_pretrain(model, resume):
    if os.path.isfile(resume):
        print(f"=> loading checkpoint {resume}")
        state = torch.load(resume, map_location=device)
        if isinstance(state, dict):
            for key in ('state_dict', 'model_state_dict', 'net', 'model'):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
        model.load_state_dict(state)
    else:
        raise ValueError(f"=> no checkpoint found at '{resume}'")
    
def load_dcm_to_mip(directory):
    mip_images = []
    dims = []
    resizes = []

    # dicom_files = [f for f in os.listdir(folder_path) if f.endswith('.dcm')]
    org_dcm, dim, resize, dcm_metainfo = load_dcm(directory, False)

    # normalization
    mip = copy.deepcopy(org_dcm)
    max, min = mip.max(), mip.min()
    mip = (mip - min) / (max - min)
    
    mip_x = np.amax(mip, axis=-1)
    mip_images.append(mip_x)
    dims.append(dim)
    resizes.append(resize)
    
    return mip_images, dims, resizes,org_dcm, dcm_metainfo


def detect_CT(inputCT, poseInfo,initial_position,voxel,dims,output_dir=None, name=None, upper_lower='upper'):

    dcm = inputCT
    dim = inputCT.shape

    print('Original Dimension: {}'.format(dim))

    anno = poseInfo
    
    RATIOS = [1.]

    for ratio in RATIOS:
        if upper_lower == 'lower':
            print('anno : ', anno['Lower'])
            lower_dcm = dcm
            lower_mat = base_transform_mat(anno['Lower'], LOWER, dim) 
            stretch_mat = stretch(dim, lower_mat, ratio=ratio) 
            lower_total_matrix = np.dot(stretch_mat, lower_mat)
            cropped_lower = crop_and_zoom(lower_dcm.squeeze(), lower_total_matrix)
    
            cropped_img = cropped_lower
            affine_mat = lower_total_matrix
            dims =dims
            initial_position=initial_position
            voxel =voxel
            # import nibabel as nib

            # nii_data = nib.Nifti1Image(cropped_img, affine=np.eye(4))
            # nii_data = nib.Nifti1Image(cropped_img, affine=None)
            # nib.save(nii_data, "outputs/lower.nii.gz")
        else:
            print('anno : ', anno['Upper'])
            upper_dcm = dcm
            
            print(anno['Upper'], UPPER, dim)
            
            upper_mat = base_transform_mat(anno['Upper'], UPPER, dim)
            stretch_mat = stretch(dim, upper_mat, ratio=ratio)
            upper_total_matrix = np.dot(stretch_mat, upper_mat)
            
            cropped_upper = crop_and_zoom(upper_dcm.squeeze(), upper_total_matrix)
 
            cropped_img = cropped_upper
            affine_mat = upper_total_matrix
            dims = dims
            initial_position=initial_position
            voxel =voxel

            # import nibabel as nib
            
            # nii_data = nib.Nifti1Image(cropped_img, affine=np.eye(4))
            # nii_data = nib.Nifti1Image(cropped_img, affine=None)
            # nib.save(nii_data, "outputs/upper.nii.gz")

    return cropped_img, affine_mat ,initial_position,voxel,dims

def crop_image(image_data, bbox):
    """ 바운딩 박스에 따라 이미지를 자르는 함수 """
    x_min, y_min, z_min, x_max, y_max, z_max = bbox
    # image_data = resize_img(image_data, (64, 128, 128))
    x_min, y_min, z_min = max(0, x_min), max(0, y_min), max(0, z_min)
    cropped_img_data = image_data[int(x_min):int(x_max), int(y_min):int(y_max), int(z_min):int(z_max)]
    return resize_img(cropped_img_data, (128, 64, 64))

def perform_segmentation(model, cropped_image):
    """ 세분화 추론 함수 """
    cropped_image_tensor = torch.tensor(cropped_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs, _ = model(cropped_image_tensor)
    predicted_labels = outputs.cpu().numpy().squeeze().squeeze()
    return np.where(predicted_labels > 0.5, 1, 0)

# RealignedCT 클래스 정의
class RealignedCT(Dataset):
    def __init__(self, detection_data, transform):
        super(RealignedCT, self).__init__()
        
        self.detection_data = detection_data
        self.transform = transform
        self.tooth_per_image = 16


    def __len__(self):
        return len(self.detection_data)

    def __getitem__(self, idx):
        crop_img, mat, flag ,i,vox,dims= self.detection_data[idx]

        resize_image = resize_img(crop_img, size=(RESIZE_DEPTH, RESIZE_HEIGHT, RESIZE_WIDTH))
        real_image =  resize_image.copy()
        cls = np.ones(self.tooth_per_image) 



        proccessed_out = {
            'flag': flag,
            'real_image' : real_image,
            'resize_image': resize_image,
            'cls': cls,
            'mat' : mat,
            'org' : crop_img,
            'dim' : dims,
            'vox' : vox,
            'i' : i
        }
        
        proccessed_out = self.transform(proccessed_out)
        
        return proccessed_out

# get_detection_dataloader 함수 정의
def get_detection_dataloader(detection_data, test_transform):
    test_dataset = RealignedCT(detection_data=detection_data, transform=test_transform)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    return test_dataloader



def main_1(model_box,model_seg,dataloader,exclude_list,save_path):

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            UPPER_TOOTH_NUM = [
                '21', '22', '23', '24', '25', '26', '27', '28',
                '11', '12', '13', '14', '15', '16', '17', '18', 
            ]

            LOWER_TOOTH_NUM = [
                '31', '32', '33', '34', '35', '36', '37', '38',
                '41', '42', '43', '44', '45', '46', '47', '48',
            ]
            flag = data['flag']
            mat = data["mat"].cpu().numpy().squeeze()
            org_dim = data['dim']
            org_dim = [tensor.item() if isinstance(tensor, torch.Tensor) else tensor for tensor in org_dim]
            vox = data["vox"]
            i = data["i"]
            tensor1_neg = vox * -1
            tensor2_neg = i * -1

            values1 = tensor1_neg.squeeze().numpy()
            values2 = tensor2_neg.squeeze().numpy()

            result_matrix = np.array([
                [values1[0], 0, 0, values2[0]],
                [0, values1[1], 0, values2[1]],
                [0, 0, values1[2],-values2[2]],
                [0, 0, 0, 1]
            ], dtype=np.float64)
            resize_image = data['resize_image']
            resize_image_np = data['real_image'].cpu().numpy().squeeze()  
            _, output, offset, score_output = model_box(resize_image)
            score_output = score_output.squeeze(0).detach().cpu().numpy()
            pred_center = hadamard_product(output.squeeze(0)).detach().cpu().numpy()
            pred_offset = offset.squeeze(0).detach().cpu().numpy()

            pred_coord1 = pred_center - pred_offset / 2
            pred_coord2 = pred_center + pred_offset / 2
            pred_bbox = np.concatenate((pred_coord1, pred_coord2), axis=1)
            adjustments = np.array([-4, -2, -2, 4, 2, 2])
            pred_bbox = pred_bbox + adjustments
            
            if flag[0] == 'upper':
                TOOTH_NUM = UPPER_TOOTH_NUM
            elif flag[0] == 'lower':
                TOOTH_NUM = LOWER_TOOTH_NUM

            no_metal_tooth_list = np.array(TOOTH_NUM)
            box_dict = {no_metal_tooth_list[idx]: pred_bbox[idx].tolist() for idx in range(len(no_metal_tooth_list))}
            print(f'start_{flag[0]}')
            print(org_dim)

            mat = np.linalg.inv(mat)
            org_dims = copy.deepcopy(org_dim)
            org_dims[0], org_dims[1], org_dims[2] = org_dim[2], org_dim[1], org_dim[0]
            
            # re_gt_lower_img_sum = np.zeros((112,280,280))
            for tooth_num, bbox in box_dict.items():
                # if int(tooth_num) in exclude_list: # 교정시 발치한 치아 사전 정보
                #     continue
                img_data = np.zeros((64,128,128))

                cropped_image = crop_image(resize_image_np, bbox)

                x_min, y_min, z_min, x_max, y_max, z_max = bbox
                
                x_min = max(0, min(x_min, 64))
    
                y_min = max(0, min(y_min, 128))
     
                z_min = max(0, min(z_min, 128))

         
                x_max = min(64, max(x_max, 0))
   
                y_max = min(128, max(y_max, 0))
     
                z_max = min(128, max(z_max, 0))
                
                segmentation_result = perform_segmentation(model_seg, cropped_image)
                cropped_img_data = img_data[int(x_min):int(x_max), int(y_min):int(y_max), int(z_min):int(z_max)]
                shape = cropped_img_data.shape
                gt_org = resize_img(segmentation_result,shape,mode="bilinear")

                img_data[int(x_min):int(x_max), int(y_min):int(y_max), int(z_min):int(z_max)] = gt_org
                img_data = resize_img(img_data,(112,280,280),mode='bilinear')
                img_data = np.where(img_data >= 0.5, 1, 0)
                img_data[img_data == 1] = int(tooth_num)
                
                org_lower = affine(img_data, mat, out_size=org_dims, interpolate=False)
                org_lower = rearrange(org_lower, 'd h w -> w h d')
                org_lower = np.flip(org_lower, axis=-1)
                
                output_nifti = nib.Nifti1Image(org_lower, affine=result_matrix)
                output_filepath = os.path.join(save_path, f'{tooth_num}.nii.gz')
                nib.save(output_nifti, output_filepath)
                print(f'Saved tooth {tooth_num} segmentation to {output_filepath}')

            print(f'end_{flag[0]}')
            
            
            
def main_2(model_box,model_seg,dataloader,exclude_list,save_path):

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            UPPER_TOOTH_NUM = [
                '21', '22', '23', '24', '25', '26', '27', '28',
                '11', '12', '13', '14', '15', '16', '17', '18', 
            ]

            LOWER_TOOTH_NUM = [
                '31', '32', '33', '34', '35', '36', '37', '38',
                '41', '42', '43', '44', '45', '46', '47', '48',
            ]
            flag = data['flag']
            mat = data["mat"].cpu().numpy().squeeze()
            org_dim = data['dim']
            org_dim = [tensor.item() if isinstance(tensor, torch.Tensor) else tensor for tensor in org_dim]
            vox = data["vox"]
            i = data["i"]
            tensor1_neg = vox * -1
            tensor2_neg = i * -1

            values1 = tensor1_neg.squeeze().numpy()
            values2 = tensor2_neg.squeeze().numpy()

            result_matrix = np.array([
                [values1[0], 0, 0, values2[0]],
                [0, values1[1], 0, values2[1]],
                [0, 0, values1[2],-values2[2]],
                [0, 0, 0, 1]
            ], dtype=np.float64)
            resize_image = data['resize_image']
            resize_image_np = data['real_image'].cpu().numpy().squeeze()  
            _, output, offset, score_output = model_box(resize_image)
            score_output = score_output.squeeze(0).detach().cpu().numpy()
            pred_center = hadamard_product(output.squeeze(0)).detach().cpu().numpy()
            pred_offset = offset.squeeze(0).detach().cpu().numpy()

            pred_coord1 = pred_center - pred_offset / 2
            pred_coord2 = pred_center + pred_offset / 2
            pred_bbox = np.concatenate((pred_coord1, pred_coord2), axis=1)
            adjustments = np.array([-4, -2, -2, 4, 2, 2])
            pred_bbox = pred_bbox + adjustments
            
            if flag[0] == 'upper':
                TOOTH_NUM = UPPER_TOOTH_NUM
            elif flag[0] == 'lower':
                TOOTH_NUM = LOWER_TOOTH_NUM

            no_metal_tooth_list = np.array(TOOTH_NUM)
            box_dict = {no_metal_tooth_list[idx]: pred_bbox[idx].tolist() for idx in range(len(no_metal_tooth_list))}
            print(f'start_{flag[0]}')
            
            re_gt_lower_img_sum = np.zeros((112,280,280))
            for tooth_num, bbox in box_dict.items():
                if int(tooth_num) in exclude_list: # 교정시 발치한 치아 사전 정보
                    continue
                img_data = np.zeros((64,128,128))

                cropped_image = crop_image(resize_image_np, bbox)

                x_min, y_min, z_min, x_max, y_max, z_max = bbox
                
                x_min = max(0, min(x_min, 64))
    
                y_min = max(0, min(y_min, 128))
     
                z_min = max(0, min(z_min, 128))

         
                x_max = min(64, max(x_max, 0))
   
                y_max = min(128, max(y_max, 0))
     
                z_max = min(128, max(z_max, 0))
                
                segmentation_result = perform_segmentation(model_seg, cropped_image)
                cropped_img_data = img_data[int(x_min):int(x_max), int(y_min):int(y_max), int(z_min):int(z_max)]
                shape = cropped_img_data.shape
                gt_org = resize_img(segmentation_result,shape,mode="bilinear")

                img_data[int(x_min):int(x_max), int(y_min):int(y_max), int(z_min):int(z_max)] = gt_org
                img_data = resize_img(img_data,(112,280,280),mode='bilinear')
                img_data = np.where(img_data >= 0.5, 1, 0)
                img_data[img_data == 1] = int(tooth_num)

                re_gt_lower_img_sum += img_data
            
            mat = np.linalg.inv(mat)
            org_dims = copy.deepcopy(org_dim)
            org_dims[0], org_dims[1], org_dims[2] = org_dim[2], org_dim[1], org_dim[0]

            
            org_lower = affine(re_gt_lower_img_sum, mat, out_size=org_dims, interpolate=False)
            org_lower = rearrange(org_lower, 'd h w -> w h d')
            org_lower = np.flip(org_lower, axis=-1)

            output_nifti = nib.Nifti1Image(org_lower, affine=result_matrix)
            output_filepath = os.path.join(save_path, f'{flag[0]}_segmentation.nii.gz')
            nib.save(output_nifti, output_filepath)
            
            print(f'end_{flag[0]}')
            
            
