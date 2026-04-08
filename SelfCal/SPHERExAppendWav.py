import numpy as np
import glob
import os
import h5py
from astropy.io import fits
from multiprocessing import Pool, Manager
from multiprocessing.shared_memory import SharedMemory
from tqdm import tqdm
from SelfCal.MapHelper import make_linear_interp_matrix, compute_crop, check_invalid
from SelfCal.MakeMap import load_reproj_file
from SelfCal.SPHERExUtility import load_calibration

worker_context = {}

def create_shared_array(shape, dtype, data=None):
    """
    Allocates shared memory with a UNIQUE system-generated name.
    Returns: (shm_object, unique_name, numpy_array)
    """
    d_size = np.dtype(dtype).itemsize
    n_bytes = int(np.prod(shape) * d_size)
    
    # name=None asks the OS for a unique name, preventing collisions
    shm = SharedMemory(name=None, create=True, size=n_bytes)
    
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    if data is not None:
        arr[:] = data[:]
        
    return shm, shm.name, arr

def init_worker(shm_info, lock, reproj_list, cache_list, ref_shape, sigma):
    """
    Worker initializer: Attaches to existing shared memory blocks using unique names
    but maps them to standard logical names in worker_context.
    """
    worker_context['reproj_list'] = reproj_list
    worker_context['cache_list'] = cache_list
    worker_context['ref_shape'] = ref_shape
    worker_context['sigma'] = sigma
    worker_context['lock'] = lock
    worker_context['shm_handles'] = [] # Keep references open

    # Reattach to all shared memory blocks
    # unique_name: the messy system name (e.g. "psm_8273...")
    # logical_name: the variable name your code expects (e.g. "det_BC")
    for unique_name, logical_name, shape, dtype in shm_info:
        try:
            shm = SharedMemory(name=unique_name)
            worker_context['shm_handles'].append(shm)
            
            # Create numpy view
            arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            
            # Store it under the logical name so the calculation code doesn't need changing
            worker_context[logical_name] = arr
            
        except FileNotFoundError:
            print(f"Worker failed to attach to SharedMemory: {unique_name} ({logical_name})")

def _wavcoadd_batch_worker(batch_indices):
    # --- Logic remains exactly the same as before ---
    reproj_list = worker_context['reproj_list']
    cache_list  = worker_context['cache_list']
    ref_shape   = worker_context['ref_shape']
    sigma       = worker_context['sigma']
    
    # Inputs (Read-Only) - Accessed via logical names
    det_BC      = worker_context['det_BC']
    det_BW      = worker_context['det_BW']
    mean_map    = worker_context['mean_map']
    std_map     = worker_context['std_map']
    
    # Outputs (Read-Write)
    total_BCBW_sum     = worker_context['total_BCBW_sum']
    total_BW_sum       = worker_context['total_BW_sum']
    total_meanvar_sum  = worker_context['total_meanvar_sum']
    
    # Local Accumulation Arrays
    batch_BCBW_sum = np.zeros(ref_shape, dtype=np.float32)
    batch_BW_sum = np.zeros(ref_shape, dtype=np.float32)
    batch_meanvar_sum = np.zeros(ref_shape, dtype=np.float32)
    
    for i in batch_indices:
        sub_mapping = load_reproj_file(reproj_list[i], fields=['sub_mapping'])['sub_mapping']
        sub_shape = sub_mapping.shape[1:]
        sub_mapping_flat = sub_mapping.reshape(2, np.prod(sub_mapping.shape[1:]))
        
        interp_matrix = make_linear_interp_matrix(sub_mapping_flat[::-1], input_shape=np.shape(det_BC))
        det_stack_flat = np.array([det_BC.ravel(), det_BW.ravel()])
        sub_stack_flat = (interp_matrix * det_stack_flat.T).T
        sub_BC, sub_BW = sub_stack_flat.reshape(2, sub_shape[0], sub_shape[1])

        with h5py.File(cache_list[i], 'r') as hf:
            ref_coords = hf['ref_coords'][:]
            sub_data = hf['sub_data'][:]
            sub_weight = hf['sub_weight'][:]
            # Newer cache files store a tight bbox of nonzero weight in the
            # original (full) sub-frame coordinates. Crop sub_BC/sub_BW to that
            # bbox so they line up with the cropped sub_data.
            if 'sub_bbox' in hf:
                rmin, rmax, cmin, cmax = hf['sub_bbox'][:]
                sub_BC = sub_BC[rmin:rmax, cmin:cmax]
                sub_BW = sub_BW[rmin:rmax, cmin:cmax]

        sub_crop, ref_crop = compute_crop(ref_shape, ref_coords)
        data_crop = sub_data[sub_crop]
        weight_crop = sub_weight[sub_crop]
        valid = weight_crop > 0
        mean_crop = mean_map[ref_crop]
        std_crop = std_map[ref_crop]
        
        clip_mask = np.abs(data_crop - mean_crop) <= sigma * std_crop
        valid_clipped = valid & clip_mask

        batch_BCBW_sum[ref_crop] += np.where(valid_clipped, (sub_BC*sub_BW)[sub_crop] * weight_crop, 0.0)
        batch_BW_sum[ref_crop] += np.where(valid_clipped, sub_BW[sub_crop] * weight_crop, 0.0)
        batch_meanvar_sum[ref_crop] += np.where(valid_clipped, (sub_BW * ((sub_BW**2) / 12 + sub_BC**2))[sub_crop] * weight_crop, 0.0)

    # Critical Section: Write to Global Shared Memory
    lock = worker_context['lock']
    with lock:
        total_BCBW_sum     += batch_BCBW_sum
        total_BW_sum       += batch_BW_sum
        total_meanvar_sum  += batch_meanvar_sum

    return True

def wav_coadd(det_BC, det_BW, mean_map, std_map, reproj_list, cache_list, ref_shape, sigma, batch_size=40, max_workers=40):
    """
    Perform wavelength coaddition using multiprocessing and shared memory.
    """
    # --- Initialize Shared Memory ---
    shm_objects = []
    # shm_info stores: (unique_system_name, logical_user_name, shape, dtype)
    shm_info = [] 
    
    try:
        # 1. Setup Input Arrays
        input_arrays = [
            ('det_BC', det_BC),
            ('det_BW', det_BW),
            ('mean_map', mean_map),
            ('std_map', std_map)
        ]
        
        for logical_name, arr in input_arrays:
            # We don't pass a name; we let the OS generate one and return it
            shm, unique_name, _ = create_shared_array(arr.shape, arr.dtype, data=arr)
            shm_objects.append(shm)
            shm_info.append((unique_name, logical_name, arr.shape, arr.dtype))

        # 2. Setup Output Arrays
        output_shapes = [
            ('total_BCBW_sum', ref_shape),
            ('total_BW_sum', ref_shape),
            ('total_meanvar_sum', ref_shape),
        ]
        
        for logical_name, shape in output_shapes:
            shm, unique_name, arr = create_shared_array(shape, np.float32)
            arr.fill(0) 
            shm_objects.append(shm)
            shm_info.append((unique_name, logical_name, shape, np.float32))

        # --- Multiprocessing ---
        all_indices = np.arange(len(reproj_list))
        tasks = [all_indices[i:i + batch_size] for i in range(0, len(all_indices), batch_size)]
        print(f"Processing {len(reproj_list)} files in {len(tasks)} batches with {max_workers} workers...")

        with Manager() as manager:
            global_lock = manager.Lock()
            
            with Pool(processes=max_workers, initializer=init_worker, 
                        initargs=(shm_info, global_lock, reproj_list, cache_list, ref_shape, sigma)) as pool:
                list(tqdm(pool.imap_unordered(_wavcoadd_batch_worker, tasks), total=len(tasks)))

        # --- Aggregate ---    
        # We retrieve data using the saved buffer references from the main process
        # Indices 4, 5, 6 correspond to the output arrays created above
        BCBW_sum = np.ndarray(ref_shape, dtype=np.float32, buffer=shm_objects[4].buf).copy()
        BW_sum = np.ndarray(ref_shape, dtype=np.float32, buffer=shm_objects[5].buf).copy()
        meanvar_sum = np.ndarray(ref_shape, dtype=np.float32, buffer=shm_objects[6].buf).copy()

        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            wav_mean_map = BCBW_sum / BW_sum
            wav_std_map = np.sqrt(meanvar_sum/BW_sum - wav_mean_map**2)
            
        wav_mean_map[~np.isfinite(wav_mean_map)] = 0
        wav_std_map[~np.isfinite(wav_std_map)] = 0

    finally:
        # --- Cleanup ---
        # Crucial: Unlink ensures the OS frees the memory, but only after we are done
        for shm in shm_objects:
            try:
                shm.close()
                shm.unlink()
            except:
                pass
                
    return wav_mean_map, wav_std_map

if __name__ == "__main__":
    detector = 4
    batch_size = 40 
    max_workers = 40
    sigma = 3.0

    det_BC, det_BW = load_calibration(band=detector, calibration_dir='/home/thomasli/spherex/SPHEREx_Spectral_Calibration')

    reproj_dir = '/mnt/md124/thomasli/selfcal/outputs/nep_det4_3p1arcsec/reprojected'
    reproj_list = sorted(glob.glob(reproj_dir + '/*.h5'))

    cache_dir = '/home/thomasli/spherex/selfcal/cache'
    cache_list = sorted(glob.glob(os.path.join(cache_dir, '*.h5')))

    mos_hdul = fits.open('/mnt/md124/thomasli/selfcal/outputs/nep_det4_3p1arcsec/mosaic/mosaic_34ch_det4_ch22.fits')
    mean_map = mos_hdul[1].data.astype(np.float32) 
    std_map = mos_hdul[3].data.astype(np.float32)
    ref_shape = mean_map.shape

    wav_mean, wav_std = wav_coadd(det_BC, det_BW, mean_map, std_map, reproj_list, cache_list, ref_shape, sigma, 
                            batch_size=batch_size, max_workers=max_workers)