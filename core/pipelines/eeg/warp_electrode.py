import numpy as np
import sys
import nibabel as nb
import nipy.algorithms.registration as nar

import dipy.align.vector_fields as vfu
from dipy.align.metrics import CCMetric
from dipy.align.imwarp import DiffeomorphicMap
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration

import tractconverter.formats.tck


def sym_diffeo_reg(static,static_mask,moving):
	# affine registration
	
        reg = 'affine'
        for sm in [10,5]:
                registration = nar.HistogramRegistration(
                        from_img=moving,
                        to_img=static,
                        to_mask=static_mask.get_data()>0,
                        similarity='cr',
                        interp='tri',
                        smooth=sm)
                reg = registration.optimize(reg, optimizer='powell', xtol=1e-3, ftol=1e-3)

	# diffeomorphic registration
	metric = CCMetric(3)
	level_iters = [4, 8, 8, 2]
	sdr = SymmetricDiffeomorphicRegistration(metric, level_iters, step_length=.1, opt_tol=1e-3)
	mapping=sdr.optimize(
		static.get_data(), moving.get_data(),
		static.affine, moving.affine,
		reg.inv().as_affine())
	return mapping


def warp_electrodes(mapping, static, locations):

    loc_vox = nb.affines.apply_affine(np.linalg.inv(static.affine), locations)
    loc_warp = locations + vfu.interpolate_vector_3d(mapping.backward, loc_vox)[0]
    loc_warp_aff = nb.affines.apply_affine(np.linalg.inv(mapping.prealign), loc_warp)
    return loc_warp_aff

"""
import numpy as np
import nibabel as nb
import joblib

ec64mr_channels=np.loadtxt('./channels_braincapmr64_reref62_markers.csv',dtype=np.str)
ec64mr_chan_locs=ec64mr_channels[:,1:].astype(np.float)
def warp_el(s,static):
    moving=nb.load('t1_%s.nii'%s)
    mapping=core.pipelines.eeg.warp_electrode.sym_diffeo_reg(static,moving)
    warp_elec=core.pipelines.eeg.warp_electrode.warp_electrodes(mapping,static,ec64mr_chan_locs)
    np.savetxt('./ec64mr_warp_%s.csv'%s,warp_elec,fmt='%f')

joblib.Parallel(n_jobs=7)([joblib.delayed(warp_el)(s,static) for s in subjects])

"""


# for s in 07MG 08MP 09SH 155GE 15MG 16VA 34IC 53DB 57SH 60AL 76SP 80BK 81MP ; do freeview t1_$s.nii:isosurface=100,2000 -c ec64mr_warp_$s.csv:radius=4 ; done
