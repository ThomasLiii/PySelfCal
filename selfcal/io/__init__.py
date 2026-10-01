"""Reading raw exposures, reprojecting them into frame files, and calibration-file I/O.

The pipeline stages hand data to each other in files: the reprojection writes one frame file
(HDF5) per reprojected frame, the calibration reads the frame files and writes a ``cal_*.h5``
file, and the mosaic reads both.

- :mod:`~selfcal.io.frames`: the two data contracts, the exposure reader
  (:class:`~selfcal.io.frames.ExposureData`) and the frame file
  (:func:`~selfcal.io.frames.write_frame`).
- :mod:`~selfcal.io.reprojection`: :func:`~selfcal.io.reprojection.batch_reproject` reprojects
  every detector of every exposure onto a box of the reference grid, one frame file each.
- :mod:`~selfcal.io.reproj`: reads frame files and owns their ``exp_<exposure>_det_<detector>.h5``
  names.
- :mod:`~selfcal.io.frame_select`: picks the frames of a tile from each frame file's
  ``ref_coords`` box.
- :mod:`~selfcal.io.exposure_filter`: selects raw exposures by FITS header values, with cached
  header reads.
- :mod:`~selfcal.io.calfile`: :class:`~selfcal.io.calfile.CalFile`, the one reader of every
  calibration product, whatever its layout version.
- :mod:`~selfcal.io.cal_writer`: writes the sky blocks of a ``cal_*.h5`` file in the v3 layout.
- :mod:`~selfcal.io.parallel_h5`: gzip HDF5 datasets compressed in threads, with the same stored
  bytes as h5py's serial writer.
"""
