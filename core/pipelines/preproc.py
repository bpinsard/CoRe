import os,sys,glob,datetime
import numpy as np

from nipype.interfaces import spm, fsl, afni, nitime, utility, lif, dcmstack as np_dcmstack, freesurfer, nipy, ants

import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio
import nibabel as nb

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
afni.base.AFNICommand.set_default_output_type('NIFTI_GZ')
np_dcmstack.DCMStackBase.set_default_output_type('NIFTI_GZ')

sys.path.insert(0,'/home/bpinsard/data/src/misc')
import generic_pipelines
from generic_pipelines.utils import wrap, fname_presuffix_basename, wildcard

from nipype import config
cfg = dict(execution={'stop_on_first_crash': False})
config.update_config(cfg)

data_dir = '/home/bpinsard/data/raw/UNF/CoRe'
mri_data_dir = os.path.join(data_dir,'MRI')
proc_dir = '/home/bpinsard/data/analysis/'
project_dir = '/home/bpinsard/data/projects/CoRe'


SEQ_INFO = [('CoReTSeq', np.asarray([1,4,2,3,1])),
            ('CoReIntSeq', np.asarray([1,3,2,4,1])),
            ('mvpa_CoReOtherSeq', np.asarray([1,3,4,2,1])),
            ('mvpa_CoreEasySeq', np.asarray([4,3,2,1,4]))]


subject_ids = [1,11,23,22,63,50,67,79,54,107,128,162,102,82]
#subject_ids = subject_ids[:-1]
#subject_ids = [184]
subject_ids = [82]

tr = 2.16
file_pattern = '_%(PatientName)s_%(SeriesDescription)s_%(SeriesDate)s_%(SeriesTime)s'
meta_tag_force=['PatientID','PatientName','SeriesDate','SeriesTime']

def dicom_dirs():

    subjects_info = pe.Node(
        utility.IdentityInterface(fields=['subject_id']),
        name='subjects_info')
    subjects_info.iterables = [('subject_id', subject_ids)]

    anat_dirs = pe.Node(
        nio.DataGrabber(infields=['subject_id'],
                        outfields=['aa_scout','localizer','t1_mprage','t1_mprage_all_echos','t1_mprage_12ch_2mm_iso'],
                        sort_filelist = True,
                        raise_on_empty = False,
                        base_directory = mri_data_dir, template=''),
        name='anat_dirs')
    anat_dirs.inputs.template = 'CoRe*_%03d*/??-%s'
    ## all this will change with 32 channel head coil
    anat_dirs.inputs.template_args = dict(
        t1_mprage=[['subject_id','MEMPRAGE_4e_p2_1mm_iso RMS/*']],
        t1_mprage_all_echos=[['subject_id','MEMPRAGE_4e_p2_1mm_iso/*']],
        aa_scout=[['subject_id','AAScout']],
        localizer=[['subject_id','localizer_32Channel']],
        t1_mprage_12ch_2mm_iso=[['subject_id','MPRAGE_12ch_ipatx2_2x2x2']],
        )

    func_dirs = pe.Node(
        nio.DataGrabber(infields=['subject_id', 'day'],
                        outfields=['aa_scout','localizer',
                                   'fmri_resting_state','fmri_task1','fmri_task2',
                                   'fmri_fieldmap', 'fmri_pa'],
                        sort_filelist = True,
                        raise_on_empty = False,
                        base_directory = mri_data_dir, template=''),
        name='func_dirs')
    func_dirs.inputs.template = 'CoRe_%03d_D%d/??-*%s'
    func_dirs.inputs.template_args = dict(
        aa_scout=[['subject_id','day','AAScout']],
        localizer=[['subject_id','day','localizer_12Channel']],
        fmri_resting_state=[['subject_id','day','BOLD_Resting_State']],
        fmri_all = [['subject_id','day','BOLD_*']],
        fmri_pa=[['subject_id','day','BOLD_PA']],
        fmri_fieldmap=[['subject_id','day','gre_field_map_BOLD/*']],)

    func_dirs.iterables = [('day',[1,2,3])] # [('day',[1,2,3])]

    n_all_func_dirs = pe.JoinNode(
        utility.IdentityInterface(fields=['fmri_all','fmri_fieldmap_all','fmri_pa_all']),
        joinsource = 'func_dirs',
        name='all_func_dirs')
    w = pe.Workflow(name='core_sleep')
    
    for n in [anat_dirs, func_dirs]:
        w.connect([(subjects_info,n,[('subject_id',)*2])])
    w.connect([
            (func_dirs, n_all_func_dirs,[('fmri_all',)*2,
                                         ('fmri_fieldmap','fmri_fieldmap_all',),
                                         ('fmri_pa','fmri_pa_all')]),
            ])        
    return w

def preproc_anat():

    w = dicom_dirs()
    
    n_t1_convert = pe.Node(
        np_dcmstack.DCMStackAnatomical(
            meta_force_add=meta_tag_force,
            out_file_format = 't1_mprage'+'_%(PatientName)s_%(SeriesDate)s_%(SeriesTime)s',
            voxel_order='LAS'),
        name='convert_t1_dicom')
    
    
    n_t1_12ch_2mm_iso_convert = pe.Node(
        np_dcmstack.DCMStackAnatomical(
            meta_force_add=meta_tag_force,
            out_file_format = 't1_mprage_12ch_2mm_iso'+'_%(PatientName)s_%(SeriesDate)s_%(SeriesTime)s',
            voxel_order='LAS'),
        name='convert_t1_12ch_2mm_iso_dicom')
    
    t1_pipeline = generic_pipelines.t1_new.t1_freesurfer_pipeline()
    t1_pipeline.inputs.freesurfer.args='-use-gpu'
    t1_pipeline.inputs.freesurfer.openmp = 8
    wm_surface = generic_pipelines.t1_new.extract_wm_surface()
    
    n_crop_t1 = pe.Node(
        nipy.Crop(out_file='%s_crop.nii.gz',
                  outputtype='NIFTI_GZ'),
        name='crop_t1')
    t1_pipeline.connect([
            (t1_pipeline.get_node('autobox_mask_fs'),n_crop_t1,
             [('%s_%s'%(c,p),)*2 for c in 'xyz' for p in ('min','max')]),
            (t1_pipeline.get_node('freesurfer'),n_crop_t1,
             [('norm','in_file')])])
    
    n_fs32k_surf = generic_pipelines.fmri_surface.surface_32k()
    
    n_reg_crop = pe.Node(
        freesurfer.Tkregister(reg_file='reg_crop.dat',
                              freeview='freeview.mat',
                              fsl_reg_out='fsl_reg_out.mat',
                              lta_out='lta.mat',
                              xfm_out='xfm.mat',
                              reg_header=True),
        'reg_crop')
    n_compute_pvmaps = pe.Node(
        freesurfer.ComputeVolumeFractions(args='--gm pve.gm.mgz'),
        name='compute_pvmaps')
    
    
    ants_for_sbctx = generic_pipelines.fmri_surface.ants_for_subcortical()
    ants_for_sbctx.inputs.inputspec.template = '/home/bpinsard/data/src/Pipelines/global/templates/MNI152_T1_1mm_brain.nii.gz'
    ants_for_sbctx.inputs.inputspec.coords = os.path.join(generic_pipelines.__path__[0],'../data','Atlas_ROIs.csv')
    
    t1_pipeline.connect([
            (t1_pipeline.get_node('freesurfer'),n_reg_crop,[
                    ('norm','target'),('subject_id',)*2,('subjects_dir',)*2]),
            (t1_pipeline.get_node('autobox_mask_fs'),n_reg_crop,[
                    ('out_file','mov')]),
            (n_reg_crop, n_compute_pvmaps, [
                    ('reg_file',)*2]),
            (t1_pipeline.get_node('autobox_mask_fs'), n_compute_pvmaps, [
                    ('out_file','in_file')]),
            (t1_pipeline.get_node('freesurfer'),n_compute_pvmaps,[
                    ('subjects_dir',)*2]),            
            ])
    
    w.base_dir = proc_dir
    w.connect([
            (w.get_node('anat_dirs'),n_t1_convert,[('t1_mprage','dicom_files')]),
            (w.get_node('anat_dirs'),n_t1_12ch_2mm_iso_convert,[('t1_mprage_12ch_2mm_iso','dicom_files')]),
            (w.get_node('subjects_info'),t1_pipeline,[(('subject_id',wrap(str),[]),'inputspec.subject_id')]),
            (n_t1_convert,t1_pipeline,[('nifti_file','inputspec.t1_file')]),
            (t1_pipeline, wm_surface, [(('freesurfer.aparc_aseg',utility.select,0),'inputspec.aseg')]),
            (t1_pipeline,n_fs32k_surf,[('freesurfer.subjects_dir','fs_source.base_directory'),]),
            (w.get_node('subjects_info'),n_fs32k_surf,[('subject_id','fs_source.subject')]),
            (t1_pipeline, ants_for_sbctx,[('crop_t1.out_file','inputspec.t1')]),
       ])

    return w


def preproc_fmri():
    
    w = dicom_dirs()
    templates = dict(
        subjects_dir=[['t1_preproc/_subject_id','subject_id','freesurfer']],
        norm = [['t1_preproc/_subject_id','subject_id','freesurfer/*[!e]/mri/norm.mgz']],
        white_matter_surface = [['extract_wm_surface/_subject_id','subject_id','surf_decimate/rlh.aparc+aseg_wm.nii_smoothed_mask.all']],
        cropped_mask = [['t1_preproc/_subject_id','subject_id','autobox_mask_fs/*.nii.gz']],
        cropped_t1 = [['t1_preproc/_subject_id','subject_id','crop_t1/*.nii.gz']],
        pve_maps = [
            ['t1_preproc/_subject_id','subject_id','compute_pvmaps/*.gm.mgz'],
            ['t1_preproc/_subject_id','subject_id','compute_pvmaps/*.wm.mgz'],
            ['t1_preproc/_subject_id','subject_id','compute_pvmaps/*.csf.mgz']],
        lowres_surf_lh = [
            ['surface_32k/_subject_id','subject_id',
             'white_resample_surf/mapflow/_white_resample_surf0/lh.white_converted.32k.gii'],
            ['surface_32k/_subject_id','subject_id',
             'pial_resample_surf/mapflow/_pial_resample_surf0/lh.pial_converted.32k.gii']],
        lowres_surf_rh = [
            ['surface_32k/_subject_id','subject_id',
             'white_resample_surf/mapflow/_white_resample_surf1/rh.white_converted.32k.gii'],
            ['surface_32k/_subject_id','subject_id',
             'pial_resample_surf/mapflow/_pial_resample_surf1/rh.pial_converted.32k.gii']],
        lowres_rois_coords = [['ants_for_subcortical/_subject_id','subject_id','coords_itk2nii/atlas_coords_nii.csv']],
        )
    n_anat_grabber = pe.Node(
        nio.DataGrabber(
            infields=['subject_id'],
            outfields=templates.keys(),
            sort_filelist=True,
            raise_on_empty=False,
            base_directory = proc_dir),
        name='anat_grabber')
    n_anat_grabber.inputs.template = 'core_sleep/%s_%s/%s'
    n_anat_grabber.inputs.template_args = templates


    n_fmri_convert = pe.MapNode(
        np_dcmstack.DCMStackfMRI(
            meta_force_add=meta_tag_force,
            out_file_format = 'fmri'+file_pattern),
        iterfield=['dicom_files'],
        name='fmri_convert')

    ## FIELDMAP PREPROC #####################################################################

    n_fieldmap_fmri_convert = pe.MapNode(
        np_dcmstack.DCMStackFieldmap(
            meta_force_add=meta_tag_force,
            out_file_format = 'fieldmap_bold'+file_pattern,
            voxel_order='LAS'),
        iterfield=['dicom_files'],
        name='convert_fieldmap_dicom')

    n_flirt_fmap2t1 = pe.MapNode(
        fsl.FLIRT(cost='mutualinfo',
                  dof=6,
                  out_matrix_file = 'fmap2t1.mat',
                  no_search=True,
                  uses_qform=True),
        iterfield=['in_file'],
        name='flirt_fmap2t1')

    n_flirt2xfm = pe.MapNode(
        freesurfer.preprocess.Tkregister(reg_file='reg.mat',
                                         xfm_out='fmap2t1.xfm'),
        iterfield=['mov','fsl_reg'],
        name='flirt2xfm')

    n_xfm2mat = pe.MapNode(
        utility.Function(
            input_names = ['xfm'],
            output_names = ['mat'],
            function=xfm2mat),
        iterfield=['xfm'],
        name='xfm2mat')

    def group_by_3(lst):
        return reduce(lambda l,x: (l+[x[i:i+3] for i in range(0,len(x),3)] if x is not None else l),lst,[])
    def first_fieldmap(lst):
        return reduce(lambda l,x: (l+[x[:3]] if x is not None else l),lst,[])

    w.connect([
            (w.get_node('all_func_dirs'),n_fieldmap_fmri_convert,[(('fmri_fieldmap_all',first_fieldmap), 'dicom_files')]),

            (n_fieldmap_fmri_convert, n_flirt_fmap2t1,[('magnitude_file','in_file')]),
            (n_anat_grabber, n_flirt_fmap2t1,[('cropped_t1','reference'),]),
            
            (n_flirt_fmap2t1, n_flirt2xfm,[('out_matrix_file','fsl_reg')]),
            (n_fieldmap_fmri_convert, n_flirt2xfm,[('magnitude_file','mov')]),
            (n_anat_grabber, n_flirt2xfm,[('cropped_t1','target'),]),

            (n_flirt2xfm, n_xfm2mat,[('xfm_out','xfm')]),
            ])

    ## Phase Inversion TOPUP APPA fieldmap ###############################################################

    n_merge_appa = generic_pipelines.fmri.merge_appa()
    
    n_topup = pe.MapNode(
        fsl.TOPUP(),
        iterfield=['in_file'],
        name='topup')
    n_topup.inputs.readout_times = [0.016695]*8
    n_topup.inputs.encoding_direction = ['x-']*4+['x']*4

    w.connect([
            (n_fmri_convert, n_merge_appa,[('nifti_file','in_files')]),
            (n_merge_appa, n_topup,[('out_files','in_file')]),
            ])

    ###############################################    ###############################################
    def name_struct(f,n,*args):
        if isinstance(f,list):
            f = tuple(f)
        return [tuple([n,f]+list(args))]

    def repeater(l,n):
        return [l]*n

    n_group_hemi_surfs = pe.Node(
        utility.Merge(2),
        name='group_hemi_surfs')

    n_st_realign = pe.MapNode(
        nipy.SpaceTimeRealigner(
            tr=tr,
            slice_info=2,
            slice_times='ascending'),
        iterfield=['in_file'],
        name='st_realign')

    n_bbreg_epi = pe.MapNode(
        freesurfer.BBRegister(contrast_type='t2',reg_frame=0,registered_file=True,init='fsl'),
        iterfield=['source_file'],
        name='bbreg_epi')

    n_mask2epi= pe.Node(
        freesurfer.ApplyVolTransform(interp='nearest',inverse=True),
        name='mask2epi')

    n_epi2t1_bbreg2xfm = pe.MapNode(
        freesurfer.preprocess.Tkregister(xfm_out='fmap2t1.xfm',
                                         freeview='freeview.mat',
                                         fsl_reg_out='fsl_reg_out.mat',
                                         lta_out='lta.mat',),
        iterfield=['reg_file','mov'],
        name='epi2t1_bbreg2xfm')

    n_epi2t1_xfm2mat = pe.MapNode(
        utility.Function(
            input_names = ['xfm'],
            output_names = ['mat'],
            function=xfm2mat),
        iterfield=['xfm'],
        name='epi2t1_xfm2mat')

    def repeat_fieldmaps(fmri_scans, fieldmaps, fieldmap_regs):
        fmaps_out = []
        fmap_regs_out = []
        i = 0
        for sess in fmri_scans:
            if sess != None:
                fmaps_out += [fieldmaps[i]]*len(sess)
                fmap_regs_out += [fieldmap_regs[i]]*len(sess)
                i += 1
        return fmaps_out, fmap_regs_out
    n_repeat_fieldmaps = pe.Node(
        utility.Function(
            input_names = ['fmri_scans','fieldmaps','fieldmap_regs'],
            output_names = ['fieldmaps','fieldmap_regs'],
            function = repeat_fieldmaps),
        name='repeat_fieldmaps')

    n_convert_motion_par = generic_pipelines.fmri_surface.n_convert_motion_par

    n_noise_corr = pe.MapNode(
        nipy.preprocess.OnlineFilter(
            echo_time=.03,
            echo_spacing=.000265,
            phase_encoding_dir=1,
            resampled_first_frame='frame1.nii',
            out_file_format='ts.h5'),
        iterfield = ['dicom_files','motion','fieldmap','fieldmap_reg'],
        overwrite=False,
        name = 'noise_correction')

    n_smooth_bp = pe.MapNode(
        generic_pipelines.fmri_surface.GrayOrdinatesBandPassSmooth(
            data_field = 'FMRI/DATA',
            smoothing_factor=3,
            smoothing_steps=3,
            filter_range=(.008,.1),
            TR=2.16),
        iterfield = ['in_file'],
        name = 'smooth_bp')

    n_surf_resample = pe.MapNode(
        nipy.preprocess.SurfaceResampling(
            echo_time=.03,
            echo_spacing=.000265,
            phase_encoding_dir=1,
            resampled_first_frame='frame1.nii',
            out_file_format='ts.h5'),
        iterfield = ['dicom_files','motion','fieldmap','fieldmap_reg'],
        overwrite=False,
        name = 'surf_resample')

    """    
    n_smooth_bp_nofilt = pe.MapNode(
        generic_pipelines.fmri_surface.GrayOrdinatesBandPassSmooth(
            data_field = 'FMRI/DATA',
            smoothing_factor=3,
            smoothing_steps=3,
            filter_range=(.008,.1),
            TR=2.16),
        iterfield = ['in_file'],
        name = 'smooth_bp_nofilt')
        """


    bold_seqs = ['fmri_resting_state','fmri_seqA','fmri_seqB','fmri_mvpa','fmri_pa']

    def flatten_remove_none(l):
        from nipype.interfaces.utility import flatten
        return flatten([e for e in l if e!=None])
    
    w.base_dir = proc_dir
    si = w.get_node('subjects_info')
    w.connect([
            (si, n_anat_grabber, [('subject_id',)*2]),

             (n_anat_grabber, n_group_hemi_surfs,[
                    (('lowres_surf_lh',name_struct,'CORTEX_LEFT'),'in1'),
                    (('lowres_surf_rh',name_struct,'CORTEX_RIGHT'),'in2')]),

            (w.get_node('all_func_dirs'),n_fmri_convert,[(('fmri_all',flatten_remove_none),'dicom_files')]),

            (n_fmri_convert,n_st_realign,[('nifti_file','in_file')]),
            (n_anat_grabber,n_bbreg_epi,[('subjects_dir',)*2]),
            (si,n_bbreg_epi,[(('subject_id',wrap(str),[]),'subject_id')]),
            (n_st_realign,n_bbreg_epi,[('out_file','source_file')]),


#            (n_st_realign,n_bbreg2xfm,[(('out_file',generic_pipelines.utils.getitem_rec,0),'mov')]),
#            (n_bbreg_epi, n_bbreg2xfm,[('out_reg_file','reg_file')]),
#            (n_anat_grabber,n_bbreg2xfm,[('subjects_dir',)*2]),            
#            (n_bbreg_epi,n_convert_motion_par,[('out_reg_file','epi2t1')]),
            (n_st_realign,n_epi2t1_bbreg2xfm,[('out_file','mov')]),
            (n_bbreg_epi, n_epi2t1_bbreg2xfm,[('out_reg_file','reg_file')]),
            (n_anat_grabber,n_epi2t1_bbreg2xfm,[('norm','target')]),
            (n_epi2t1_bbreg2xfm, n_epi2t1_xfm2mat,[('xfm_out','xfm')]),
            

            (n_epi2t1_xfm2mat,n_convert_motion_par,[('mat','epi2t1')]),
            (n_st_realign, n_convert_motion_par,[('par_file','motion')]),
            (n_fmri_convert, n_convert_motion_par,[('nifti_file','matrix_file')]),


            (n_convert_motion_par, n_noise_corr,[('motion','motion')]),
            (w.get_node('all_func_dirs'),n_noise_corr,[(('fmri_all', flatten_remove_none),'dicom_files')]),

            (n_anat_grabber, n_noise_corr,[
                    ('norm','surfaces_volume_reference'),
                    ('cropped_mask','mask'),
                    (('lowres_rois_coords',name_struct,'SUBCORTICAL_CEREBELLUM',
                      '/home/bpinsard/data/projects/motion_correction/code/aparc.a2009s+aseg_subcortical_subset.txt'),
                     'resample_rois'),
                    ('pve_maps','partial_volume_maps')
                    ]),
            (w.get_node('all_func_dirs'),n_repeat_fieldmaps,[('fmri_all','fmri_scans')]),
            (n_fieldmap_fmri_convert, n_repeat_fieldmaps, [('fieldmap_file','fieldmaps')]),
            (n_xfm2mat, n_repeat_fieldmaps, [('mat','fieldmap_regs')]),
            (n_repeat_fieldmaps, n_noise_corr,[
                    ('fieldmaps','fieldmap'),
                    ('fieldmap_regs','fieldmap_reg')]),

            (n_group_hemi_surfs, n_noise_corr,[('out','resample_surfaces')]),
            (n_noise_corr, n_smooth_bp,[('out_file','in_file')]),
            
            ])
    if False:
        w.connect([
            (n_anat_grabber, n_surf_resample,[
                    ('norm','surfaces_volume_reference'),
                    ('cropped_mask','mask'),
                    (('lowres_rois_coords',name_struct,'SUBCORTICAL_CEREBELLUM',
                      '/home/bpinsard/data/projects/motion_correction/code/aparc.a2009s+aseg_subcortical_subset.txt'),
                     'resample_rois'),
                    ]),
            (n_repeat_fieldmaps, n_surf_resample,[
                    ('fieldmaps','fieldmap'),
                    ('fieldmap_regs','fieldmap_reg')]),


            (n_convert_motion_par, n_surf_resample,[('motion','motion')]),
            (w.get_node('all_func_dirs'), n_surf_resample,[(('fmri_all', flatten_remove_none),'dicom_files')]),
            (n_group_hemi_surfs, n_surf_resample,[('out','resample_surfaces')]),

#            (n_surf_resample, n_smooth_bp_nofilt,[('out_file','in_file')]),            
            ])

    n_dataset_noisecorr = pe.Node(
        CreateDataset(tr=tr,
                      behavioral_data_path=os.path.join(data_dir,'Behavior'),
                      design=os.path.join(project_dir,'data/design.csv')),
        name='dataset_noisecorr')
    n_dataset_smoothed = n_dataset_noisecorr.clone('dataset_smoothed')
    
    w.connect([
            (si,n_dataset_noisecorr,[('subject_id',)*2]),
            (n_noise_corr,n_dataset_noisecorr,[('out_file','ts_files')]),
            (w.get_node('all_func_dirs'),n_dataset_noisecorr,[(('fmri_all',flatten_remove_none),'dicom_dirs')]),

#            (si,n_dataset_smoothed,[('subject_id',)*2]),
#            (n_smooth_bp,n_dataset_smoothed,[('out_file','ts_files')]),
#            (w.get_node('all_func_dirs'),n_dataset_smoothed,[(('fmri_all',flatten_remove_none),'dicom_dirs')]),

            ])
    return w 


def xfm2mat(xfm):
    import os
    import numpy as np
    mat=np.loadtxt(xfm, skiprows=5,usecols=range(4))
    np.savetxt("out.mat",np.vstack([mat,[0,0,0,1]]))
    return os.path.abspath("out.mat")        

    
from nipype.interfaces.base import (TraitedSpec, BaseInterface, traits,
                                    BaseInterfaceInputSpec, isdefined, File, Directory,
                                    InputMultiPath, OutputMultiPath)

from ..mvpa import dataset as mvpa_dataset
import mvpa2.datasets

class CreateDatasetInputSpec(BaseInterfaceInputSpec):
    subject_id = traits.Int(mandatory=True)
    ts_files = traits.List(File(exists=True))
    dicom_dirs = traits.List(Directory(exists=True))
    data_path = traits.Str('FMRI/DATA',usedefault=True,)
    design = File(exists=True,mandatory=True)
    behavioral_data_path=Directory()
    
    tr = traits.Float(mandatory=True)

class CreateDatasetOutputSpec(TraitedSpec):
    dataset = File(exists=True, mandatory=True)
    glm_dataset = File(exists=True, mandatory=True)
    glm_stim_dataset = File(exists=True, mandatory=True)

class CreateDataset(BaseInterface):

    input_spec = CreateDatasetInputSpec
    output_spec = CreateDatasetOutputSpec

    def _run_interface(self, runtime):
        dss = []
        dss_glm = []
        dss_glm_stim = []
        scan_id = 0

        subject_id = self.inputs.subject_id
        used_ts_files = []

        empty_to_none =lambda x: x if len(x) else None
        design = np.atleast_2d(
            np.loadtxt(
                self.inputs.design,
                dtype=np.object,
                delimiter=',',
                converters={0:int,1:str,2:str,3:empty_to_none,4:empty_to_none,5:empty_to_none,6:bool}))
        scan_id=-1
        for day,ses_name,mri_name,seq_name,beh,scan_idx,optional in design:
            seq_idx = None
            seq_info = None
            if not seq_name is None and not beh is None:
                seq_info = SEQ_INFO
                seq_idx = [[s[0] for s in seq_info].index(seq_name)] * 14
            behavior_file = None
            if not beh is None:
                # take the last behavioral file which matches in case of failed task
                behavior_file = sorted(glob.glob(
                    os.path.join(
                        self.inputs.behavioral_data_path,
                        'CoRe_%03d_D%d/CoRe_%03d_%s_?.mat'%(subject_id,day,subject_id,beh))))
                if not len(behavior_file): 
                    if optional:
                        continue
                    else:
                        raise RuntimeError('missing data')
                behavior_file = behavior_file[-1]
            # deal with multiple scans
            ts_files = [f for f,dd in zip(self.inputs.ts_files,self.inputs.dicom_dirs)\
                            if ('_D%d/'%day in dd and mri_name in dd)]

            if scan_idx is not None:
                scan_idx = int(scan_idx)
                if scan_idx >= len(ts_files):
                    if optional:
                        continue
                    else:
                        raise RuntimeError('missing data')
                ts_files = [ts_files[int(scan_idx)]]
            ts_files = [f for f in ts_files if f not in used_ts_files]

            for ts_file in ts_files:
                scan_id += 1
#                print day, ses_name, mri_name, ts_file[-12:], str(behavior_file).split('/')[-1]
#                continue
                ds = mvpa_dataset.ds_from_ts(ts_file, behavior_file,
                                             seq_info=seq_info, seq_idx=seq_idx,
                                             tr=self.inputs.tr)
                ds.sa['scan_name'] = [ses_name]*ds.nsamples
                ds.sa['scan_id'] = [scan_id]*ds.nsamples
                dss.append(ds)
                if beh is not None:
                    reg_groups = np.unique([n.split('_')[0] for n in ds.sa.regressors_exec.dtype.names])
                    ds_glm = mvpa_dataset.ds_tr2glm(ds, 'regressors_exec', reg_groups)
                    dss_glm.append(ds_glm)
                    reg_groups = np.unique([n.split('_')[0] for n in ds.sa.regressors_stim.dtype.names])
                    ds_glm = mvpa_dataset.ds_tr2glm(ds, 'regressors_stim', reg_groups)
                    dss_glm_stim.append(ds_glm)

                    del ds.sa['regressors_exec'], ds.sa['regressors_stim']
                # used ts files to avoid repeating
                used_ts_files.append(ts_file)
        # stack all
        ds = mvpa2.datasets.vstack(dss)
        ds.a.update(dss[0].a)
#        ds.a = dss[0].a
        ds.sa['chunks'] = np.cumsum(np.ediff1d(ds.chunks, to_begin=[0])!=0)
        ds_glm = mvpa2.datasets.vstack(dss_glm)
        ds_glm.a.update(dss[0].a)
        ds_glm.sa['chunks'] = np.cumsum(np.ediff1d(ds_glm.chunks, to_begin=[0])!=0)
        ds_glm_stim = mvpa2.datasets.vstack(dss_glm_stim)
        ds_glm_stim.a.update(dss[0].a)
        ds_glm_stim.sa['chunks'] = np.cumsum(np.ediff1d(ds_glm_stim.chunks, to_begin=[0])!=0)
        mvpa_dataset.add_aparc_ba_fa(
            ds, self.inputs.subject_id,
            pproc_tpl='/home/bpinsard/data/analysis/core_sleep/surface_32k/_subject_id_%s')
        ds_glm.fa = ds.fa
        ds_glm_stim.fa = ds.fa
        
        outputs = self._list_outputs()
        ds.save(outputs['dataset'])
        ds_glm.save(outputs['glm_dataset'])
        ds_glm_stim.save(outputs['glm_stim_dataset'])

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['dataset'] = os.path.abspath('./ds_%s.h5'%self.inputs.subject_id)
        outputs['glm_dataset'] = os.path.abspath('./glm_ds_%s.h5'%self.inputs.subject_id)
        outputs['glm_stim_dataset'] = os.path.abspath('./glm_stim_ds_%s.h5'%self.inputs.subject_id)
        return outputs


#def preproc_eeg():
    
